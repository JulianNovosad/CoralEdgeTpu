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
                               int output_width, int output_height)
    : input_queue_(input_queue), 
      output_queue_(output_queue), 
      detection_buffer_ptr_(detection_buffer),
      ballistic_overlay_buffer_(ballistic_overlay_buffer),
      buffer_pool_(buffer_pool),
      image_data_pool_(image_data_pool),
      input_pixel_format_(input_pixel_format),
      output_width_(output_width), 
      output_height_(output_height),
      skip_factor_(1),
      frame_counter_(0),
      is_tpu_stream_(false),
      running_(false),
      avg_queue_pop_time_us_(0),
      avg_preprocess_time_us_(0),
      app_ref_(nullptr) {
}

ImageProcessor::ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int output_width, int output_height)
    : input_queue_(input_queue), 
      output_queue_(output_queue), 
      detection_buffer_ptr_(nullptr),
      buffer_pool_(buffer_pool),
      image_data_pool_(image_data_pool),
      input_pixel_format_(input_pixel_format),
      output_width_(output_width), 
      output_height_(output_height),
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
            std::promise<bool> promise;
            std::future<bool> future = promise.get_future();
            
            std::thread timer_thread([this, &promise]() {
                std::this_thread::sleep_for(std::chrono::seconds(3));
                if (worker_thread_.joinable()) {
                    // Note: Cannot use APP_LOG here if it's during final shutdown
                    // but stop() is usually safe.
                    promise.set_value(false);
                } else {
                    promise.set_value(true);
                }
            });
            
            if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout) {
                timer_thread.join();
            } else {
                future.get();
                if (worker_thread_.joinable()) {
                    worker_thread_.join();
                }
            }
            if (timer_thread.joinable()) {
                timer_thread.join();
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

    while (running_ && g_running.load(std::memory_order_acquire)) {
        ImageData* input_image_ptr = nullptr;
        if (input_queue_.wait_pop(input_image_ptr, std::chrono::milliseconds(10))) {
            if (!input_image_ptr) {
                if (!running_) return;
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
                            input_frame_mat = cv::Mat(input_image.height, input_image.width, CV_8UC3, frame_data_ptr);
                        } else if (input_image.format == libcamera::formats::YUV420) {
                            cv::Mat yuv_mat = cv::Mat(input_image.height * 3 / 2, input_image.width, CV_8UC1, frame_data_ptr);
                            cv::cvtColor(yuv_mat, input_frame_mat, cv::COLOR_YUV2BGR_I420);
                        } else if (input_image.format == libcamera::formats::YUYV) {
                            cv::Mat yuyv_mat = cv::Mat(input_image.height, input_image.width, CV_8UC2, frame_data_ptr);
                            cv::cvtColor(yuyv_mat, input_frame_mat, cv::COLOR_YUV2BGR_YUYV, 3);
                        } else {
                            APP_LOG_ERROR("ImageProcessor: Unsupported format: " + input_image.format.toString());
                            conversion_success = false;
                        }

                        if (conversion_success) {
                            cv::Mat processed_mat;
                            if (input_image.width == (unsigned int)output_width_ && input_image.height == (unsigned int)output_height_) {
                                processed_mat = input_frame_mat;
                            } else {
                                cv::resize(input_frame_mat, processed_mat, cv::Size(output_width_, output_height_), 0, 0, cv::INTER_NEAREST);
                            }

                            if (detection_buffer_ptr_ != nullptr) {
                                if (detection_buffer_ptr_->update_consumer()) {
                                    if (app_ref_) {
                                        app_ref_->increment_inference_results_consumed_by_overlay();
                                    }
                                }
                                const auto& detections = detection_buffer_ptr_->get_read_buffer();
                                // Call apply_detections_to_frame even if detections are empty to draw crosshairs/timestamp
                                apply_detections_to_frame(processed_mat, detections);
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
                                        output_image_data->format = libcamera::formats::RGB888;
                                        output_image_data->buffer = processed_buffer_data;
                                        output_image_data->fd = -1;
                                        output_image_data->t_capture_raw_ms = input_image.t_capture_raw_ms;
                                        
                                        if (output_queue_.push(output_image_data)) {
                                            guard.output_produced = true;
                                        } else {
                                            output_image_data->buffer.reset();
                                            image_data_pool_->release(output_image_data);
                                        }
                                    }
                                }
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

void ImageProcessor::apply_detections_to_frame(cv::Mat& frame, const DetectionResults& detections) {
    int frame_width = frame.cols;
    int frame_height = frame.rows;

    // --- Step 2: Frame authority logging (Log once per frame) ---
    // Log width, height, pixel format, and source stage.
    // Assuming source stage is "post-resize" as this is `processed_mat`
    char log_buffer[256];
    snprintf(log_buffer, sizeof(log_buffer), 
             "FRAME_AUTHORITY_AUDIT: width=%d, height=%d, pixel_format=%s, source_stage=post-resize",
             frame_width, frame_height, input_pixel_format_.toString().c_str());
    APP_LOG_INFO(log_buffer);

    // --- Original detection drawing ---
    for (const auto& detection : detections) {
        int x_min = std::max(0, static_cast<int>(detection.xmin * frame_width));
        int y_min = std::max(0, static_cast<int>(detection.ymin * frame_height));
        int x_max = std::min(frame_width - 1, static_cast<int>(detection.xmax * frame_width));
        int y_max = std::min(frame_height - 1, static_cast<int>(detection.ymax * frame_height));

        cv::Scalar box_color(0, 0, 255); // Red color for bounding boxes
        cv::rectangle(frame, cv::Point(x_min, y_min), cv::Point(x_max, y_max), box_color, 2);
        
        // Draw class ID and confidence score
        std::string label = "ID:" + std::to_string(detection.class_id) + " " + 
                           std::to_string(static_cast<int>(detection.score * 100)) + "%";
        cv::putText(frame, label, cv::Point(x_min, y_min - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    // --- Step 1: Force overlay execution (even if AI is dead) ---
    // Draw center crosshair (white)
    int crosshair_size = 20;
    cv::line(frame, cv::Point(frame_width / 2 - crosshair_size, frame_height / 2), 
             cv::Point(frame_width / 2 + crosshair_size, frame_height / 2), cv::Scalar(255, 255, 255), 2);
    cv::line(frame, cv::Point(frame_width / 2, frame_height / 2 - crosshair_size), 
             cv::Point(frame_width / 2, frame_height / 2 + crosshair_size), cv::Scalar(255, 255, 255), 2);

    // Overlay text: "OVERLAY PATH EXECUTED"
    cv::putText(frame, "OVERLAY PATH EXECUTED", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Overlay timestamp + frame counter
    uint64_t current_time_ms = get_time_raw_ms(); // Use raw timing for consistency
    std::string timestamp_str = "Time: " + std::to_string(current_time_ms) + "ms";
    std::string frame_counter_str = "Frame: " + std::to_string(frame_counter_); // Using class member frame_counter_
    cv::putText(frame, timestamp_str, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(frame, frame_counter_str, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    if (ballistic_overlay_buffer_) {
        if (ballistic_overlay_buffer_->update_consumer()) {
            const OverlayBallisticPoint& ballistic_point = ballistic_overlay_buffer_->get_read_buffer();

            // --- CAUSALITY VERIFICATION LOG ---
            int matched = (ballistic_point.frame_id == (int)frame_counter_) ? 1 : 0;
            char verify_log[128];
            snprintf(verify_log, sizeof(verify_log),
                "OVERLAY ballistic_draw frame_id=%d matched=%d x=%d y=%d",
                ballistic_point.frame_id, matched, ballistic_point.x, ballistic_point.y);
            APP_LOG_INFO(verify_log);
            // -----------------------------------

            // Check frame binding
            if (ballistic_point.frame_id == frame_counter_) { // frame_counter_ from ImageProcessor should match
                if (ballistic_point.is_valid) {
                    // Log raw value, coordinate space (pixel), validity flag.
                    char ballistic_log_buffer[256];
                    snprintf(ballistic_log_buffer, sizeof(ballistic_log_buffer), 
                             "BALLISTIC_POINT_AUDIT: raw_value=(%d, %d), coordinate_space=pixel, validity=true",
                             ballistic_point.x, ballistic_point.y);
                    APP_LOG_INFO(ballistic_log_buffer);

                    // Draw the ballistic impact point as a red circle
                    cv::circle(frame, cv::Point(ballistic_point.x, ballistic_point.y), 5, cv::Scalar(0, 0, 255), -1); // Red circle, filled
                } else {
                    // Log raw value, coordinate space (pixel), validity flag.
                    char ballistic_log_buffer[256];
                    snprintf(ballistic_log_buffer, sizeof(ballistic_log_buffer), 
                             "BALLISTIC_POINT_AUDIT: raw_value=(%d, %d), coordinate_space=pixel, validity=false",
                             ballistic_point.x, ballistic_point.y); // Log current x,y even if invalid
                    APP_LOG_INFO(ballistic_log_buffer);

                    // Draw a red “X” at the image center
                    cv::line(frame, cv::Point(frame_width / 2 - crosshair_size, frame_height / 2 - crosshair_size), 
                             cv::Point(frame_width / 2 + crosshair_size, frame_height / 2 + crosshair_size), cv::Scalar(0, 0, 255), 2);
                    cv::line(frame, cv::Point(frame_width / 2 - crosshair_size, frame_height / 2 + crosshair_size), 
                             cv::Point(frame_width / 2 + crosshair_size, frame_height / 2 - crosshair_size), cv::Scalar(0, 0, 255), 2);
                    // Overlay text: BALLISTIC POINT INVALID.
                    cv::putText(frame, "BALLISTIC POINT INVALID", cv::Point(frame_width / 2 - 100, frame_height / 2 + 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
                }
            } else {
                // Ballistic point not for this frame, or buffer not updated. Log this situation.
                char ballistic_log_buffer[256];
                snprintf(ballistic_log_buffer, sizeof(ballistic_log_buffer), 
                         "BALLISTIC_POINT_AUDIT: No matching ballistic point for frame_id=%lu (Ballistic Buffer frame_id=%d)",
                         frame_counter_, ballistic_point.frame_id);
                APP_LOG_DEBUG(ballistic_log_buffer);
            }
        } else {
            // Ballistic buffer not updated this frame, draw invalid
            char ballistic_log_buffer[256];
            snprintf(ballistic_log_buffer, sizeof(ballistic_log_buffer), 
                     "BALLISTIC_POINT_AUDIT: Ballistic buffer not updated for frame_id=%lu", frame_counter_);
            APP_LOG_DEBUG(ballistic_log_buffer);
            
            // Draw a red “X” at the image center
            cv::line(frame, cv::Point(frame_width / 2 - crosshair_size, frame_height / 2 - crosshair_size), 
                     cv::Point(frame_width / 2 + crosshair_size, frame_height / 2 + crosshair_size), cv::Scalar(0, 0, 255), 2);
            cv::line(frame, cv::Point(frame_width / 2 - crosshair_size, frame_height / 2 + crosshair_size), 
                     cv::Point(frame_width / 2 + crosshair_size, frame_height / 2 - crosshair_size), cv::Scalar(0, 0, 255), 2);
            // Overlay text: BALLISTIC POINT INVALID.
            cv::putText(frame, "BALLISTIC POINT INVALID", cv::Point(frame_width / 2 - 100, frame_height / 2 + 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }
    } else {
        // Ballistic buffer not provided, draw invalid
        char ballistic_log_buffer[256];
        snprintf(ballistic_log_buffer, sizeof(ballistic_log_buffer), 
                 "BALLISTIC_POINT_AUDIT: Ballistic buffer not initialized for frame_id=%lu", frame_counter_);
        APP_LOG_DEBUG(ballistic_log_buffer);
        
        // Draw a red “X” at the image center
        cv::line(frame, cv::Point(frame_width / 2 - crosshair_size, frame_height / 2 - crosshair_size), 
                 cv::Point(frame_width / 2 + crosshair_size, frame_height / 2 + crosshair_size), cv::Scalar(0, 0, 255), 2);
        cv::line(frame, cv::Point(frame_width / 2 - crosshair_size, frame_height / 2 + crosshair_size), 
                 cv::Point(frame_width / 2 + crosshair_size, frame_height / 2 - crosshair_size), cv::Scalar(0, 0, 255), 2);
        // Overlay text: BALLISTIC POINT INVALID.
        cv::putText(frame, "BALLISTIC POINT INVALID", cv::Point(frame_width / 2 - 100, frame_height / 2 + 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }


    // --- Step 1: Unconditionally generate /tmp/overlaid_output.jpg ---
    if (cv::imwrite("/tmp/overlaid_output.jpg", frame)) {
        APP_LOG_INFO("[AUDIT] SUCCESS: Snapshot saved to /tmp/overlaid_output.jpg");
    } else {
        APP_LOG_ERROR("[AUDIT] ERROR: cv::imwrite failed to write to /tmp!");
    }
}