// Verified headers: [image_processor.h, util_logging.h, application.h, timing.h, chrono...]
// Verification timestamp: 2026-01-06 17:08:04
#include "image_processor.h"
#include "util_logging.h"
#include "application.h"
#include "timing.h"
#include <chrono>
#include <future>
#include <thread>
#include <libcamera/formats.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include <unistd.h>

extern std::atomic<bool> g_running;

int libcamera_pixel_format_to_opencv_type(const libcamera::PixelFormat& format) {
    if (format.fourcc() == libcamera::formats::BGRA8888.fourcc()) return CV_8UC4;
    if (format.fourcc() == libcamera::formats::BGR888.fourcc()) return CV_8UC3;
    if (format.fourcc() == libcamera::formats::RGBA8888.fourcc()) return CV_8UC4;
    if (format.fourcc() == libcamera::formats::RGB888.fourcc()) return CV_8UC3;
    if (format.fourcc() == libcamera::formats::YUYV.fourcc()) return CV_8UC2;
    if (format.fourcc() == libcamera::formats::YUV420.fourcc()) return CV_8UC1; // Planar
    return -1;
}

ImageProcessor::ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                               TripleBuffer<DetectionResults>* detection_buffer,
                               TripleBuffer<OverlayBallisticPoint>* ballistic_overlay_buffer,
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int output_width, int output_height,
                               const std::string& module_name)
    : input_queue_(input_queue), 
      output_queue_(output_queue), 
      detection_buffer_ptr_(detection_buffer),
            ballistic_overlay_buffer_(ballistic_overlay_buffer),
            buffer_pool_(buffer_pool),
            image_data_pool_(image_data_pool),
            input_pixel_format_(input_pixel_format),
            output_width_(output_width),
            output_height_(output_height),
            module_name_(module_name),
            skip_factor_(1),
            frame_counter_(0),
            is_tpu_stream_(false),
            running_(false),
            is_running_(false),
            avg_queue_pop_time_us_(0),
            avg_preprocess_time_us_(0),
            app_ref_(nullptr) {
      }
ImageProcessor::ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int output_width, int output_height,
                               const std::string& module_name)
    : input_queue_(input_queue), 
      output_queue_(output_queue), 
      detection_buffer_ptr_(nullptr),
      buffer_pool_(buffer_pool),
      image_data_pool_(image_data_pool),
      input_pixel_format_(input_pixel_format),
      output_width_(output_width), 
      output_height_(output_height),
      module_name_(module_name),
      skip_factor_(1),
      frame_counter_(0),
      is_tpu_stream_(false),
      running_(false),
      avg_queue_pop_time_us_(0),
      avg_preprocess_time_us_(0),
      app_ref_(nullptr) {
}

ImageProcessor::~ImageProcessor() {
    stop();
}

bool ImageProcessor::start() {
    if (!running_.exchange(true)) {
        worker_thread_ = std::thread(&ImageProcessor::worker_thread_func, this);
        return true;
    }
    return false;
}

void ImageProcessor::stop() {
    if (running_.exchange(false)) {
        // 1. Force OpenCV to relinquish multi-threading resources (TBB deadlock fix)
        cv::setNumThreads(0);

        // 2. Poison Pill: Wake up worker thread blocked on wait_pop
        ImageData* poison_pill = image_data_pool_->acquire();
        if (poison_pill) {
            poison_pill->buffer = nullptr;
            input_queue_.push(poison_pill);
        }

        // 3. Use timed join to prevent indefinite blocking
        if (worker_thread_.joinable()) {
            auto shared_promise = std::make_shared<std::promise<void>>();
            std::future<void> future = shared_promise->get_future();
            
            std::thread joiner_thread([this, shared_promise]() {
                try {
                    if (worker_thread_.joinable()) {
                        worker_thread_.join();
                    }
                    shared_promise->set_value();
                } catch (...) {}
            });
            
            if (future.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
                APP_LOG_WARNING("[SHUTDOWN] ImageProcessor worker thread did not join within 3s, detaching.");
                if (worker_thread_.joinable()) {
                    worker_thread_.detach();
                }
                joiner_thread.detach();
            } else {
                if (joiner_thread.joinable()) {
                    joiner_thread.join();
                }
            }
        }

        // 4. RESOURCE CLEANUP (MUST happen AFTER thread join)
        std::lock_guard<std::mutex> lock(fd_map_mutex_);
        for (auto& pair : fd_map_) {
            if (pair.second.start != MAP_FAILED) {
                munmap(pair.second.start, pair.second.length);
            }
            if (pair.second.internal_fd >= 0) {
                close(pair.second.internal_fd);
            }
        }
        fd_map_.clear();
    }
}

void ImageProcessor::start(const std::string& phone_ip) {
    if (is_running_.exchange(true)) return;
    APP_LOG_INFO("ImageProcessor [" + std::string(is_tpu_stream_ ? "TPU" : "Viz") + "] received START signal for " + phone_ip);
}

bool ImageProcessor::is_running() const {
    return running_.load();
}

// Helper for RAII DMA synchronization
struct ScopedDmaSync {
    int fd;
    bool active;
    
    ScopedDmaSync(int f) : fd(f), active(false) {
        if (fd >= 0) {
            struct dma_buf_sync sync_start = {0};
            sync_start.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
            if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync_start) == 0) {
                active = true;
            } else {
                APP_LOG_ERROR("DMA_BUF_SYNC_START failed for FD " + std::to_string(fd));
            }
        }
    }
    
    ~ScopedDmaSync() {
        if (active && fd >= 0) {
            struct dma_buf_sync sync_end = {0};
            sync_end.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
            if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync_end) < 0) {
                APP_LOG_ERROR("DMA_BUF_SYNC_END failed for FD " + std::to_string(fd));
            }
        }
    }
};

void ImageProcessor::worker_thread_func() {
    set_thread_name("ImageProcessor");

    int opencv_input_type = libcamera_pixel_format_to_opencv_type(input_pixel_format_);
    if (opencv_input_type == -1) {
        running_ = false;
        return;
    }

    std::unique_ptr<GpuOverlay> gpu_overlay;
    if (!is_tpu_stream_) {
        gpu_overlay = std::make_unique<GpuOverlay>(output_width_, output_height_);
        if (!gpu_overlay->initialize()) {
            APP_LOG_ERROR("ImageProcessor: Failed to initialize GpuOverlay.");
        }
    }

    while (running_ && g_running.load(std::memory_order_acquire)) {
        ImageData* input_image_ptr = nullptr;
        if (input_queue_.wait_pop(input_image_ptr, std::chrono::milliseconds(10))) {
            if (!input_image_ptr) {
                if (!running_) return;
                continue;
            }

            // Accounting: visualization stream only processes if START signal was received
            if (!is_tpu_stream_ && !is_running_.load()) {
                if (app_ref_) {
                    app_ref_->inc_cam_to_viz_dropped();
                }
                image_data_pool_->release(input_image_ptr);
                continue;
            }

            struct ProcessingGuard {
                ImageProcessor* proc;
                ImageData* input_image_ptr;
                bool output_produced = false;
                ProcessingGuard(ImageProcessor* p, ImageData* ptr) : proc(p), input_image_ptr(ptr) {}
                ~ProcessingGuard() {
                    if (proc->app_ref_) {
                        if (proc->is_tpu_stream_) {
                            // Stage 1: Camera -> TPU Processor
                            proc->app_ref_->inc_cam_to_tpu_proc_consumed();
                            
                            // Stage 2: TPU Processor -> Inference Engine
                            // (Every frame taken from Cam queue MUST result in either a produced or dropped frame for Inf Engine)
                            proc->app_ref_->inc_proc_to_inf_produced();
                            if (!output_produced) {
                                proc->app_ref_->inc_proc_to_inf_dropped();
                            }
                        } else {
                            // Stage 1: Camera -> Viz Processor
                            proc->app_ref_->inc_cam_to_viz_consumed();
                            // (Viz Processor is a sink, it doesn't produce for a next stage in top-level accounting)
                        }
                    }
                    if (input_image_ptr) {
                        proc->image_data_pool_->release(input_image_ptr);
                    }
                }
            } guard(this, input_image_ptr);

            ImageData& input_image = *input_image_ptr;

            uint64_t process_start_time = get_time_raw_ms();
            
            // Determine if we should process this frame based on skipping and validity
            bool should_process = true;
            
            if (frame_counter_++ % skip_factor_ != 0) {
                should_process = false;
            }

            bool is_fd_valid = (input_image.fd >= 0 && input_image.length > 0);
            bool is_buffer_valid = (input_image.buffer && !input_image.buffer->data.empty());

            if (!is_fd_valid && !is_buffer_valid) {
                should_process = false;
            }
            
            // Check dims
            if (input_image.width == 0 || input_image.height == 0) {
                should_process = false;
            }

            if (should_process) {
                uint8_t* frame_data_ptr = nullptr;
                int current_fd = -1;
                
                // Enclose DMA sync in a block to ensure SYNC_END before timing updates/loop end
                {
                    std::unique_ptr<ScopedDmaSync> dma_sync;

                    if (is_fd_valid && !is_buffer_valid) {
                        // Zero-Copy Path
                        struct stat sb;
                        // Optimization: Try to find by FD first to avoid fstat if possible?
                        // But FD reuse makes this risky without verification.
                        // We keep fstat but ensure it's the only overhead.
                        if (fstat(input_image.fd, &sb) == 0) {
                            BufferKey key = {sb.st_dev, sb.st_ino};
                            
                            // Check cache first (Hot Path Lock)
                            {
                                std::lock_guard<std::mutex> lock(fd_map_mutex_);
                                auto it = fd_map_.find(key);
                                if (it != fd_map_.end()) {
                                    frame_data_ptr = static_cast<uint8_t*>(it->second.start);
                                    current_fd = it->second.internal_fd;
                                }
                            }
                            
                            // If not in cache, map it (Outside Lock)
                            if (frame_data_ptr == nullptr) {
                                int dup_fd = dup(input_image.fd);
                                if (dup_fd >= 0) {
                                    void* addr = mmap(nullptr, input_image.length, PROT_READ, MAP_SHARED, dup_fd, 0);
                                    if (addr != MAP_FAILED) {
                                        // Insert into cache (Hot Path Lock)
                                        {
                                            std::lock_guard<std::mutex> lock(fd_map_mutex_);
                                            // Re-check in case another thread inserted it
                                            auto it = fd_map_.find(key);
                                            if (it != fd_map_.end()) {
                                                munmap(addr, input_image.length);
                                                close(dup_fd);
                                                frame_data_ptr = static_cast<uint8_t*>(it->second.start);
                                                current_fd = it->second.internal_fd;
                                            } else {
                                                MappedBuffer mb = {addr, input_image.length, dup_fd};
                                                fd_map_[key] = mb;
                                                frame_data_ptr = static_cast<uint8_t*>(addr);
                                                current_fd = dup_fd;
                                            }
                                        }
                                    } else {
                                        APP_LOG_ERROR("ImageProcessor: mmap failed for FD " + std::to_string(input_image.fd));
                                        close(dup_fd);
                                    }
                                } else {
                                    APP_LOG_ERROR("ImageProcessor: dup failed for FD " + std::to_string(input_image.fd));
                                }
                            }
                        } else {
                            APP_LOG_ERROR("ImageProcessor: fstat failed for FD " + std::to_string(input_image.fd));
                        }
                        
                        // Initialize RAII Sync if we have a valid internal FD
                        if (current_fd >= 0) {
                            dma_sync = std::make_unique<ScopedDmaSync>(current_fd);
                            if (!dma_sync->active) {
                                // Sync failed, abort processing this frame
                                frame_data_ptr = nullptr; 
                            }
                        }
                    } else {
                        // Traditional Copy Path
                        frame_data_ptr = input_image.buffer->data.data();
                    }

                    if (frame_data_ptr) {
                        cv::Mat input_frame_mat;
                        bool conversion_success = true;
                        
                        if (input_image.format == libcamera::formats::RGB888) {
                            // AUTHORITY: The stream is labeled RGB888 but contains BGR888 data.
                            input_frame_mat = cv::Mat(input_image.height, input_image.width, CV_8UC3, frame_data_ptr);
                        } else if (input_image.format == libcamera::formats::YUV420) {
                            cv::Mat yuv_mat = cv::Mat(input_image.height * 3 / 2, input_image.width, CV_8UC1, frame_data_ptr);
                            cv::cvtColor(yuv_mat, input_frame_mat, cv::COLOR_YUV2BGR_I420);
                        } else {
                            APP_LOG_ERROR("ImageProcessor: Unsupported format: " + input_image.format.toString());
                            conversion_success = false;
                        }

                        if (conversion_success) {
                            // Ensure input_frame_mat is in the correct format for its destination
                            if (is_tpu_stream_) {
                                                                                        cv::cvtColor(input_frame_mat, input_frame_mat, cv::COLOR_BGR2RGB);                            } else {
                                // Visualization/Encoder expects BGR. Input is BGR. No swap needed.
                            }

                            cv::Mat processed_mat;
                            if (input_image.width == (unsigned int)output_width_ && input_image.height == (unsigned int)output_height_) {
                                processed_mat = input_frame_mat;
                            } else {
                                cv::resize(input_frame_mat, processed_mat, cv::Size(output_width_, output_height_), 0, 0, cv::INTER_NEAREST);
                            }

                            if (gpu_overlay && !is_tpu_stream_) {
                                const DetectionResults* detections_ptr = nullptr;
                                if (detection_buffer_ptr_ != nullptr) {
                                    if (detection_buffer_ptr_->update_consumer()) {
                                        if (app_ref_) {
                                            app_ref_->increment_inference_results_consumed_by_overlay();
                                        }
                                    }
                                    detections_ptr = &detection_buffer_ptr_->get_read_buffer();
                                }
                                
                                const OverlayBallisticPoint* bp_ptr = nullptr;
                                if (ballistic_overlay_buffer_) {
                                    ballistic_overlay_buffer_->update_consumer();
                                    bp_ptr = &ballistic_overlay_buffer_->get_read_buffer();
                                }

                                static DetectionResults empty_detections;
                                gpu_overlay->render(processed_mat.data, 
                                                     detections_ptr ? *detections_ptr : empty_detections,
                                                     bp_ptr, frame_counter_);

                                // --- AUDIT LOGGING ---
                                if (frame_counter_ % 30 == 0) {
                                    char audit_buf[256];
                                    snprintf(audit_buf, sizeof(audit_buf), 
                                             "GPU_OVERLAY_AUDIT: frame_id=%lu, detections=%zu, ballistic=%d",
                                             frame_counter_, detections_ptr ? detections_ptr->size() : 0, bp_ptr ? 1 : 0);
                                    APP_LOG_INFO(audit_buf);
                                    
                                    if (cv::imwrite("/tmp/overlaid_output.jpg", processed_mat)) {
                                        APP_LOG_DEBUG("[AUDIT] Snapshot saved to /tmp/overlaid_output.jpg");
                                    }
                                }
                            }

                            std::shared_ptr<PooledBuffer<uint8_t>> processed_buffer_data = buffer_pool_->acquire();
                            if (processed_buffer_data) {
                                size_t required_size = processed_mat.total() * processed_mat.elemSize();
                                if (processed_buffer_data->data.size() < required_size) {
                                    processed_buffer_data->data.resize(required_size);
                                }
                                
                                if (!processed_mat.empty() && processed_mat.data != nullptr) {
                                    std::memcpy(processed_buffer_data->data.data(), processed_mat.data, required_size);
                                    processed_buffer_data->size = required_size;

                                    ImageData* output_image_data = image_data_pool_->acquire();
                                    if (output_image_data) {
                                        *output_image_data = ImageData(input_image.capture_time, input_image.frame_id);
                                        output_image_data->width = output_width_;
                                        output_image_data->height = output_height_;
                                        // Use correct format for the next stage (Inference Engine needs RGB, Encoder needs BGR)
                                        output_image_data->format = (is_tpu_stream_) ? libcamera::formats::RGB888 : libcamera::formats::BGR888;
                                        output_image_data->buffer = processed_buffer_data;
                                        output_image_data->fd = -1;
                                        output_image_data->t_capture_raw_ms = input_image.t_capture_raw_ms;
                                        
                                        uint64_t current_time_ms = get_time_raw_ms();
                                        float image_proc_ms = static_cast<float>(current_time_ms - process_start_time);
                                        output_image_data->image_proc_ms = image_proc_ms;

                                        // Propagate camera telemetry
                                        output_image_data->cam_exposure_ms = input_image.cam_exposure_ms;
                                        output_image_data->cam_isp_latency_ms = input_image.cam_isp_latency_ms;
                                        output_image_data->cam_buffer_usage_percent = input_image.cam_buffer_usage_percent;
                                        
                                        if (output_queue_.push(output_image_data)) {
                                            guard.output_produced = true;
                                            // std::cerr << "DEBUG: ImageProcessor[" << module_name_ << "] pushed frame " << input_image.frame_id << std::endl;
                                            
                                            // Log to unified CSV
                                            CsvLogEntry ip_entry;
                                            copy_to_array(ip_entry.module, module_name_.c_str());
                                            copy_to_array(ip_entry.event, "frame_processed");
                                            ip_entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                                            ip_entry.call_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                                            ip_entry.cam_frame_id = input_image.frame_id;
                                            ip_entry.image_proc_ms = image_proc_ms;
                                            
                                            // Propagate camera telemetry
                                            ip_entry.cam_exposure_ms = input_image.cam_exposure_ms;
                                            ip_entry.cam_isp_latency_ms = input_image.cam_isp_latency_ms;
                                            ip_entry.cam_buffer_usage_percent = input_image.cam_buffer_usage_percent;

                                            Logger::getInstance().log_csv(ip_entry);
                                        } else {
                                            output_image_data->buffer.reset();
                                            image_data_pool_->release(output_image_data);
                                            APP_LOG_ERROR("ImageProcessor: Output queue push failed (Full).");
                                        }
                                    } else {
                                        APP_LOG_ERROR("ImageProcessor: Failed to acquire ImageData from pool.");
                                    }
                                }
                            } else {
                                APP_LOG_ERROR("ImageProcessor: Failed to acquire buffer from pool (Starvation).");
                            }
                        }
                    }
                } // dma_sync destructor fires here
            }

            uint64_t process_end_time = get_time_raw_ms();
            long long preprocess_duration_us = (long long)(process_end_time - process_start_time) * 1000;
            avg_preprocess_time_us_.store(static_cast<long long>(avg_preprocess_time_us_.load() * 0.9 + preprocess_duration_us * 0.1));
        }
        }
    }
    