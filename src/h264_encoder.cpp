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
    // x264_param_default(&param_);
    // x264_picture_init(&picture_in_);
    // x264_picture_init(&picture_out_);
    
    // Initialize encoding queue and worker thread
    encoding_worker_running_.store(false);
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
    if (worker_thread_ && worker_thread_->joinable()) { // Check if optional holds a thread and if it's joinable
        std::promise<bool> promise;
        std::future<bool> future = promise.get_future();
        
        std::thread timer_thread([this, &promise]() {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            if (worker_thread_ && worker_thread_->joinable()) {
                APP_LOG_WARNING("H264Encoder worker thread did not join within timeout");
                promise.set_value(false);
            } else {
                promise.set_value(true);
            }
        });
        
        if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout) {
            timer_thread.join();
        } else {
            future.get();
            if (worker_thread_ && worker_thread_->joinable()) {
                worker_thread_->join();
            }
        }
        if (timer_thread.joinable()) {
            timer_thread.join();
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
    param.i_threads = 1; // REDUCED TO 1 TO PREVENT CRASH / LOWER MEMORY
    param.b_vfr_input = 0; // Constant frame rate
    param.b_repeat_headers = 1; // Repeat SPS/PPS before IDR frames
    param.b_annexb = 1; // Annex B format (start codes) for better compatibility
    
    // Keyframe and IDR frame settings
    param.i_keyint_min = 1; // Minimum keyframe interval
    param.i_bframe = 0; // No B-frames for simpler decoding and lower latency

    // Apply profile and tune settings
    x264_param_apply_profile(&param, "baseline"); // Baseline profile for broad compatibility
    
    // Additional settings
    param.i_frame_reference = 1;
    param.i_dpb_size = 1;
    param.analyse.b_transform_8x8 = 0;
    param.rc.i_qp_constant = 28;
    param.rc.i_rc_method = X264_RC_CQP;
    param.i_cqm_preset = X264_CQM_FLAT;
    param.i_bframe_adaptive = X264_B_ADAPT_NONE;
    param.i_bframe_bias = 0;
    param.i_frame_packing = -1;
    
    // Additional ultra-low latency settings
    param.rc.i_vbv_buffer_size = 0;
    param.rc.i_vbv_max_bitrate = 0;
    param.i_sync_lookahead = 0;
    param.i_bframe_pyramid = X264_B_PYRAMID_NONE;
    param.b_aud = 0;
    param.b_repeat_headers = 1;
    param.i_slice_max_size = 1500;
    param.i_slice_max_mbs = 0;
    // Performance optimizations
    param.analyse.inter = 0; 
    param.analyse.i_me_method = X264_ME_DIA;
    param.analyse.i_subpel_refine = 0;
    param.rc.i_qp_min = 28;
    param.rc.i_qp_max = 35;

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
        // 1. Pop from input queue ...
        if (input_queue_.wait_pop(input_image_ptr, std::chrono::milliseconds(100))) {
            if (!input_image_ptr || !input_image_ptr->isValid()) {
                if (input_image_ptr) image_data_pool_->release(input_image_ptr);
                break;
            }
            
            ImageData& image_data = *input_image_ptr;
            // Record queue pop time
            image_data.encode_start_time = std::chrono::steady_clock::now();
            
            if (!image_data.buffer) {
                APP_LOG_WARNING("H264Encoder: Received image with null buffer. Skipping.");
                image_data_pool_->release(input_image_ptr);
                continue;
            }

            // 2. Color conversion (Ensure proper format for x264 encoding)
            [[maybe_unused]] auto cvtcolor_start = std::chrono::steady_clock::now();
            cv::Mat frame_bgr, frame_yuv;
            
            size_t expected_bgr_size = image_data.width * image_data.height * 3;
            size_t actual_size = image_data.buffer->data.size();
            
            if (actual_size >= expected_bgr_size) {
                // ImageProcessor produces RGB888, so we convert RGB to YUV420p
                cv::Mat frame_rgb_raw(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
                cv::cvtColor(frame_rgb_raw, frame_yuv, cv::COLOR_RGB2YUV_I420);
            } else {
                // Unknown format fallback
                if (actual_size >= (image_data.width * image_data.height)) {
                    cv::Mat frame_gray(image_data.height, image_data.width, CV_8UC1, image_data.buffer->data.data());
                    cv::cvtColor(frame_gray, frame_bgr, cv::COLOR_GRAY2BGR);
                    cv::cvtColor(frame_bgr, frame_yuv, cv::COLOR_BGR2YUV_I420);
                } else {
                    image_data_pool_->release(input_image_ptr);
                    continue;
                }
            }
            [[maybe_unused]] auto cvtcolor_end = std::chrono::steady_clock::now();

            image_data.buffer.reset(); // Release buffer after conversion

            // 3. Encode I420 frame using x264
            frame_count_++;
            picture_in.i_pts = frame_count_;

            int y_size = width_ * height_;
            int uv_size = y_size / 4;
            std::memcpy(picture_in.img.plane[0], frame_yuv.data, y_size);
            std::memcpy(picture_in.img.plane[1], frame_yuv.data + y_size, uv_size);
            std::memcpy(picture_in.img.plane[2], frame_yuv.data + y_size + uv_size, uv_size);

            x264_nal_t* nals;
            int i_nals;
            int encoded_frame_size = x264_encoder_encode(encoder_, &nals, &i_nals, &picture_in, &picture_out);

            if (encoded_frame_size > 0) {
                auto h264_buffer_shared = h264_buffer_pool_->acquire();
                if (h264_buffer_shared) {
                    H264Buffer* h264_buffer = h264_buffer_shared.get();
                    if (static_cast<size_t>(encoded_frame_size) <= h264_buffer->data.capacity()) {
                        // Copy NAL units to buffer
                        std::memcpy(h264_buffer->data.data(), nals[0].p_payload, encoded_frame_size);
                        h264_buffer->size = encoded_frame_size;
                        h264_buffer->frame_id = image_data.frame_id;
                        h264_buffer->encoder_frame_count = frame_count_;
                        h264_buffer->timestamp_epoch_ms = image_data.t_capture_raw_ms;

                        if (output_queue_.push(h264_buffer)) {
                            if (app_) app_->increment_h264_output_queue_in();
                        }
                    }
                }
            } else if (encoded_frame_size < 0) {
                APP_LOG_ERROR("H264Encoder: x264_encoder_encode failed.");
            }

            last_frame_processed_time_ = std::chrono::steady_clock::now();
            last_frame_id_ = image_data.frame_id;
            image_data_pool_->release(input_image_ptr);
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