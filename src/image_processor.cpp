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

extern std::atomic<bool> shutdown_requested;

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
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int output_width, int output_height)
    : input_queue_(input_queue), 
      output_queue_(output_queue), 
      detection_buffer_ptr_(detection_buffer),
      buffer_pool_(buffer_pool),
      input_pixel_format_(input_pixel_format),
      output_width_(output_width), 
      output_height_(output_height),
      skip_factor_(1),
      frame_counter_(0),
      running_(false),
      avg_queue_pop_time_us_(0),
      avg_preprocess_time_us_(0),
      app_ref_(nullptr) {
}

ImageProcessor::ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int output_width, int output_height)
    : input_queue_(input_queue), 
      output_queue_(output_queue), 
      detection_buffer_ptr_(nullptr),
      buffer_pool_(buffer_pool),
      input_pixel_format_(input_pixel_format),
      output_width_(output_width), 
      output_height_(output_height),
      skip_factor_(1),
      frame_counter_(0),
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
        // Poison Pill: Wake up worker thread blocked on wait_pop
        input_queue_.push(ImageData{});
        // Use timed join to prevent indefinite blocking
        if (worker_thread_.joinable()) {
            std::promise<bool> promise;
            std::future<bool> future = promise.get_future();
            
            std::thread timer_thread([this, &promise]() {
                std::this_thread::sleep_for(std::chrono::seconds(3));
                if (worker_thread_.joinable()) {
                    APP_LOG_WARNING("ImageProcessor worker thread did not join within timeout");
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

    while (running_.load() && !shutdown_requested.load(std::memory_order_acquire)) {
        ImageData input_image;
        // Use a timeout to periodically check running_
        if (input_queue_.wait_pop(input_image, std::chrono::milliseconds(10))) {
            if (!input_image.isValid()) break;
            
            // P=C+D Invariant Enforcement:
            // Every frame popped must be either PROCESSED (pushed to output) or DROPPED.
            // We assume 'Dropped' unless explicit success.
            bool frame_processed_successfully = false;
            
            // RAII Accounting Guard
            struct ProcessingGuard {
                ImageProcessor* proc;
                bool& success;
                ProcessingGuard(ImageProcessor* p, bool& s) : proc(p), success(s) {}
                ~ProcessingGuard() {
                    if (!success) {
                        if (proc->app_ref_) {
                            proc->app_ref_->increment_camera_frames_dropped();
                        }
                    }
                    // If success, we assume the downstream consumer will handle accounting 
                    // or it's considered "consumed" by the system.
                    // Note: If ImageProcessor is a parallel path, we don't increment Consumed 
                    // to avoid double counting with InferenceEngine. We only track Drops.
                }
            } guard(this, frame_processed_successfully);

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
                                if (!detections.empty()) {
                                    apply_detections_to_frame(processed_mat, detections);
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

                                    ImageData output_image_data(input_image.capture_time, input_image.frame_id);
                                    output_image_data.width = output_width_;
                                    output_image_data.height = output_height_;
                                    output_image_data.format = libcamera::formats::RGB888;
                                    output_image_data.buffer = processed_buffer_data;
                                    output_image_data.fd = -1;
                                    output_image_data.t_capture_raw_ms = input_image.t_capture_raw_ms;
                                    
                                    output_queue_.push(std::move(output_image_data));
                                    frame_processed_successfully = true;
                                }
                            } else {
                                APP_LOG_WARNING("ImageProcessor: Buffer pool exhaustion. Dropping frame.");
                                // frame_processed_successfully remains false, trigger increment_dropped via guard
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

    for (const auto& detection : detections) {
        int x_min = std::max(0, static_cast<int>(detection.xmin * frame_width));
        int y_min = std::max(0, static_cast<int>(detection.ymin * frame_height));
        int x_max = std::min(frame_width - 1, static_cast<int>(detection.xmax * frame_width));
        int y_max = std::min(frame_height - 1, static_cast<int>(detection.ymax * frame_height));

        cv::Scalar box_color(0, 0, 255);
        cv::rectangle(frame, cv::Point(x_min, y_min), cv::Point(x_max, y_max), box_color, 2);
    }
}