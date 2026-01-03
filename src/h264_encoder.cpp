#include "h264_encoder.h"
#include "application.h"
#include <x264.h>
#include <future>
#include <thread>

H264Encoder::H264Encoder(ImageQueue& input_queue, 
                         H264Queue& output_queue, 
                         std::shared_ptr<BufferPool<uint8_t>> h264_buffer_pool,
                         std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                         int width, int height, double fps)
    : input_queue_(input_queue),
      output_queue_(output_queue),
      h264_buffer_pool_(h264_buffer_pool),
      image_data_pool_(image_data_pool),
      width_(width),
      height_(height),
      fps_(fps),
      running_(false),
      encoder_(nullptr),
      frame_count_(0) {
    // Initialize x264 structures - MOVED TO WORKER THREAD
}

void H264Encoder::set_application_ref(Application* app) {
    app_ = app;
}

H264Encoder::~H264Encoder() {
    stop();
}

bool H264Encoder::start() {
    if (running_.load()) {
        APP_LOG_WARNING("H264Encoder already running.");
        return true; // Already running, consider it a success
    }

    running_.store(true);
    // Emplace the thread object into the optional.
    // The x264 encoder will be initialized inside worker_thread_func
    worker_thread_.emplace(&H264Encoder::worker_thread_func, this);
    APP_LOG_INFO("H264Encoder started.");
    return true;
}

void H264Encoder::stop() {
    if (!running_.load()) {
        APP_LOG_WARNING("H264Encoder not running.");
        return;
    }

    running_.store(false);
    // Poison Pill: Wake up worker thread blocked on wait_pop
    ImageData* poison_pill = image_data_pool_->acquire();
    if (poison_pill) {
        poison_pill->buffer = nullptr;
        input_queue_.push(poison_pill);
    }
    // Use timed join to prevent indefinite blocking
    if (worker_thread_ && worker_thread_->joinable()) {
        auto shared_promise = std::make_shared<std::promise<void>>();
        std::future<void> future = shared_promise->get_future();
        
        std::thread joiner_thread([this, shared_promise]() {
            try {
                if (worker_thread_->joinable()) {
                    worker_thread_->join();
                }
                shared_promise->set_value();
            } catch (...) {}
        });
        
        if (future.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
            std::cerr << "[SHUTDOWN] H264Encoder worker thread did not join within 3s, detaching." << std::endl;
            if (worker_thread_->joinable()) {
                worker_thread_->detach();
            }
            joiner_thread.detach();
        } else {
            if (joiner_thread.joinable()) {
                joiner_thread.join();
            }
        }
    }
    
    // FORENSIC FIX: Only close encoder AFTER worker thread has joined
    if (encoder_) { // Check if encoder was successfully opened
        x264_encoder_close(encoder_);
        encoder_ = nullptr;
        // x264_picture_clean(&picture_in_); // MOVED TO WORKER THREAD
    }
    APP_LOG_INFO("H264Encoder stopped.");
}

extern std::atomic<bool> g_running;

void H264Encoder::worker_thread_func() {
    APP_LOG_INFO("H264Encoder worker thread started.");
    APP_LOG_INFO("H264Encoder: Initializing x264 with width=" + std::to_string(width_) + ", height=" + std::to_string(height_) + ", fps=" + std::to_string(fps_));

    // Local x264 structures to avoid member variable issues / stack issues
    x264_param_t param;
    x264_picture_t picture_in;
    x264_picture_t picture_out;

    // Initialize x264 parameters
    x264_param_default(&param); // Start with sane defaults
    x264_param_default_preset(&param, "ultrafast", "zerolatency");

    // Configure input resolution and pixel format
    param.i_width = width_;
    param.i_height = height_;
    param.i_fps_num = (int)fps_; 
    param.i_fps_den = 1;
    param.i_timebase_num = 1;
    param.i_timebase_den = (int)fps_; 
    param.i_keyint_max = 30; 
    param.b_intra_refresh = 0; 
    
    // Input pixel format (OpenCV uses BGR, but we convert to YUV420p for x264)
    param.i_csp = X264_CSP_I420; 

    // Stream parameters
    param.i_threads = 2; 
    param.b_vfr_input = 0;
    param.b_repeat_headers = 1;
    param.b_annexb = 1;
    
    // Keyframe and IDR frame settings
    param.i_keyint_min = 1; // Minimum keyframe interval
    param.i_bframe = 0; // No B-frames for simpler decoding and lower latency

    // Apply profile and tune settings
    x264_param_apply_profile(&param, "baseline"); // Baseline profile for broad compatibility
    
    // Additional settings
    param.rc.i_rc_method = X264_RC_CRF;
    param.rc.f_rf_constant = 25; // Standard quality
    
    // Additional ultra-low latency settings
    param.i_sync_lookahead = 0;
    param.b_aud = 0;
    param.b_repeat_headers = 1;

    APP_LOG_INFO("H264Encoder: Attempting to open x264 encoder...");
    encoder_ = x264_encoder_open(&param); 
    if (!encoder_) {
        APP_LOG_ERROR("H264Encoder: Failed to open x264 encoder.");
        running_.store(false); 
        return;
    }
    APP_LOG_INFO("H264Encoder: x264 encoder opened successfully.");

    APP_LOG_INFO("H264Encoder: Attempting to allocate x264 picture_in_...");
    if (x264_picture_alloc(&picture_in, param.i_csp, param.i_width, param.i_height) < 0) {
        std::cerr << "DEBUG: H264Encoder alloc picture failed!" << std::endl;
        APP_LOG_ERROR("H264Encoder: Failed to allocate x264 picture_in_.");
        running_.store(false); 
        return;
    }
    APP_LOG_INFO("H264Encoder: x264 picture_in allocated.");

    APP_LOG_INFO("H264Encoder: Initializing picture_out...");
    x264_picture_init(&picture_out); // Initialize picture_out
    APP_LOG_INFO("H264Encoder: picture_out initialized.");

    APP_LOG_INFO("H264Encoder: Starting main loop...");
    APP_LOG_INFO("H264Encoder: Input Queue Address: " + std::to_string((uintptr_t)&input_queue_));

    while (running_.load() && g_running.load(std::memory_order_acquire)) {
        ImageData* input_image_ptr = nullptr;
        if (input_queue_.wait_pop(input_image_ptr, std::chrono::milliseconds(100))) {
            if (!input_image_ptr) {
                if (!running_) return;
                continue;
            }
            
            struct EncoderGuard {
                H264Encoder* enc;
                ImageData* ptr;
                EncoderGuard(H264Encoder* e, ImageData* p) : enc(e), ptr(p) {}
                ~EncoderGuard() {
                    if (ptr) enc->image_data_pool_->release(ptr);
                }
            } guard(this, input_image_ptr);

            if (!input_image_ptr->isValid() || !input_image_ptr->buffer) {
                continue;
            }
            
            ImageData& image_data = *input_image_ptr;
            
            if (frame_count_ % 30 == 0) {
                APP_LOG_INFO("H264Encoder: Received frame " + std::to_string(image_data.frame_id) + " buffer size=" + std::to_string(image_data.buffer->size));
            }

            image_data.encode_start_time = std::chrono::steady_clock::now();
            
            try {
                // 2. Color conversion
                cv::Mat frame_yuv;
                if (image_data.format == libcamera::formats::BGR888) {
                    cv::Mat frame_bgr_raw(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
                    cv::cvtColor(frame_bgr_raw, frame_yuv, cv::COLOR_BGR2YUV_I420);
                } else {
                    cv::Mat frame_rgb_raw(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
                    cv::cvtColor(frame_rgb_raw, frame_yuv, cv::COLOR_RGB2YUV_I420);
                }

                // 3. Encode I420 frame
                frame_count_++;
                picture_in.i_pts = frame_count_;

                // Copy Y plane
                uint8_t* dst_y = picture_in.img.plane[0];
                uint8_t* src_y = frame_yuv.data;
                for (int i = 0; i < height_; ++i) {
                    std::memcpy(dst_y, src_y, width_);
                    dst_y += picture_in.img.i_stride[0];
                    src_y += width_;
                }

                // Copy U and V planes
                int uv_width = width_ / 2;
                int uv_height = height_ / 2;
                uint8_t* src_u = frame_yuv.data + (width_ * height_);
                uint8_t* src_v = src_u + (uv_width * uv_height);

                uint8_t* dst_u = picture_in.img.plane[1];
                uint8_t* dst_v = picture_in.img.plane[2];

                for (int i = 0; i < uv_height; ++i) {
                    std::memcpy(dst_u, src_u, uv_width);
                    std::memcpy(dst_v, src_v, uv_width);
                    dst_u += picture_in.img.i_stride[1];
                    dst_v += picture_in.img.i_stride[2];
                    src_u += uv_width;
                    src_v += uv_width;
                }

                x264_nal_t* nals;
                int i_nals;
                int encoded_frame_size = x264_encoder_encode(encoder_, &nals, &i_nals, &picture_in, &picture_out);

                if (encoded_frame_size > 0) {
                    auto h264_buffer_shared = h264_buffer_pool_->acquire();
                    if (h264_buffer_shared) {
                        H264Buffer* h264_buffer = h264_buffer_shared.get();
                        if (static_cast<size_t>(encoded_frame_size) <= h264_buffer->data.capacity()) {
                            std::memcpy(h264_buffer->data.data(), nals[0].p_payload, encoded_frame_size);
                            h264_buffer->size = encoded_frame_size;
                            h264_buffer->frame_id = image_data.frame_id;
                            h264_buffer->timestamp_epoch_ms = image_data.t_capture_raw_ms;

                            // Debug: Log first 16 bytes
                            if (frame_count_ % 30 == 0) {
                                std::string hex_dump = "";
                                for (int i = 0; i < std::min(encoded_frame_size, 16); i++) {
                                    char buf[4];
                                    sprintf(buf, "%02X ", h264_buffer->data[i]);
                                    hex_dump += buf;
                                }
                                APP_LOG_INFO("H264Encoder: Produced frame " + std::to_string(h264_buffer->frame_id) + " size=" + std::to_string(encoded_frame_size) + " data=" + hex_dump);
                            }

                            if (!output_queue_.push(h264_buffer)) {
                                // Drop if queue full - prevents backpressure SIGABRT
                                APP_LOG_WARNING("H264Encoder: Output queue full, dropping frame " + std::to_string(image_data.frame_id));
                            } else if (app_) {
                                app_->increment_h264_output_queue_in();
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                APP_LOG_ERROR("H264Encoder: Exception during encoding: " + std::string(e.what()));
            } catch (...) {
                APP_LOG_ERROR("H264Encoder: Unknown exception during encoding");
            }

            last_frame_processed_time_ = std::chrono::steady_clock::now();
            last_frame_id_ = image_data.frame_id;
        }
    }
    x264_picture_clean(&picture_in); // Clean up allocated picture
    APP_LOG_INFO("H264Encoder worker thread stopped.");
} 

bool H264Encoder::is_display_starving() const {
    if (!running_.load()) {
        return true; 
    }
    
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last_frame = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - last_frame_processed_time_
    ).count();
    
    double expected_frame_interval_ms = 1000.0 / fps_;
    double starvation_threshold_ms = expected_frame_interval_ms * 2.0;  
    
    if (time_since_last_frame > starvation_threshold_ms) {
        return true;
    }
    
    return false;
}