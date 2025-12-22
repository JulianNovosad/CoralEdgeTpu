#include "h264_encoder.h"
#include <x264.h>
#include <iostream> // Added for std::cerr/cout

H264Encoder::H264Encoder(ImageQueue& input_queue, 
                         H264Queue& output_queue, 
                         std::shared_ptr<BufferPool<uint8_t>> h264_buffer_pool,
                         int width, int height, double fps)
    : input_queue_(input_queue),
      output_queue_(output_queue),
      h264_buffer_pool_(h264_buffer_pool),
      width_(width),
      height_(height),
      fps_(fps),
      running_(false),
      encoder_(nullptr) {
    // Initialize x264 structures
    x264_param_default(&param_);
    x264_picture_init(&picture_in_);
    x264_picture_init(&picture_out_);
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
    if (worker_thread_ && worker_thread_->joinable()) { // Check if optional holds a thread and if it's joinable
        worker_thread_->join();
    }
    
    if (encoder_) { // Check if encoder was successfully opened
        x264_nal_t* nal;
        int i_nal;
        // Flush the encoder by draining it, but don't process the output.
        // This is important because there might be no consumer for the output_queue_,
        // which would cause the h264_buffer_pool_->acquire() to block/timeout,
        // stalling the shutdown process.
        while (x264_encoder_encode(encoder_, &nal, &i_nal, NULL, &picture_out_) > 0) {
            // Do nothing, just drain the encoder.
        }
        x264_encoder_close(encoder_);
        encoder_ = nullptr;
        x264_picture_clean(&picture_in_);
    }
    APP_LOG_INFO("H264Encoder stopped.");
}

void H264Encoder::worker_thread_func() {

    APP_LOG_INFO("H264Encoder worker thread started.");
    APP_LOG_INFO("H264Encoder: Initializing x264 with width=" + std::to_string(width_) + ", height=" + std::to_string(height_) + ", fps=" + std::to_string(fps_));

    

    // Initialize x264 parameters using member variable

    x264_param_default(&param_); // Start with sane defaults

    x264_param_default_preset(&param_, "ultrafast", "zerolatency");

    

    // Configure input resolution and pixel format

    param_.i_width = width_;

    param_.i_height = height_;

    param_.i_fps_num = static_cast<int>(fps_);

    param_.i_fps_den = 1;

    param_.i_keyint_max = static_cast<int>(fps_); // Keyframe every second

    param_.b_intra_refresh = 1; // IDR frames instead of full I-frames to help with seeking

    

    // Input pixel format (OpenCV uses BGR, but we convert to YUV420p for x264)

    param_.i_csp = X264_CSP_I420; 

    

    // Stream parameters

    param_.i_threads = 1; // Simplify to 1 thread for debugging

    param_.b_vfr_input = 0; // Constant frame rate

    param_.b_repeat_headers = 1; // Repeat SPS/PPS before IDR frames

    param_.b_annexb = 1; // Annex B byte stream format

    

    // Apply profile and tune settings

    x264_param_apply_profile(&param_, "baseline"); // Baseline profile for broad compatibility

    

    // Simplify keyframe interval for debugging

    param_.i_keyint_max = static_cast<int>(fps_);

    

    APP_LOG_INFO("H264Encoder: x264 parameters - width=" + std::to_string(param_.i_width) +

                 ", height=" + std::to_string(param_.i_height) +

                 ", csp=" + std::to_string(param_.i_csp) +

                 ", fps_num=" + std::to_string(param_.i_fps_num) +

                 ", fps_den=" + std::to_string(param_.i_fps_den) +

                 ", keyint_max=" + std::to_string(param_.i_keyint_max) +

                 ", threads=" + std::to_string(param_.i_threads));

    

        APP_LOG_INFO("H264Encoder: Attempting to open x264 encoder...");

        // Open the encoder

        encoder_ = x264_encoder_open(&param_); 

        if (!encoder_) {

            APP_LOG_ERROR("H264Encoder: Failed to open x264 encoder.");

            running_.store(false); // Set running_ to false to stop the thread

            return;

        }

        APP_LOG_INFO("H264Encoder: x264 encoder opened successfully.");

    

        APP_LOG_INFO("H264Encoder: Attempting to allocate x264 picture_in_...");

                // Allocate pictures

                if (x264_picture_alloc(&picture_in_, param_.i_csp, param_.i_width, param_.i_height) < 0) {

                    APP_LOG_ERROR("H264Encoder: Failed to allocate x264 picture_in_.");

                    running_.store(false); // Set running_ to false to stop the thread

                    return;

                }

                APP_LOG_INFO("H264Encoder: x264 picture_in_ allocated.");

    picture_in_.i_pts = 0; // Initialize presentation timestamp
    x264_picture_init(&picture_out_); // Initialize picture_out_

    while (running_.load()) {
        [[maybe_unused]] auto total_loop_start = std::chrono::high_resolution_clock::now();
        ImageData image_data;
        // 1. Pop from input queue
        [[maybe_unused]] auto pop_start = std::chrono::high_resolution_clock::now();
        if (!input_queue_.pop(image_data)) {
            if (!running_.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Prevent busy-waiting
            continue; 
        }
        [[maybe_unused]] auto pop_end = std::chrono::high_resolution_clock::now();
        APP_LOG_DEBUG("H264Encoder: Time to pop from queue: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(pop_end - pop_start).count()) + " us");
        
        // Record queue pop time
        image_data.encode_start_time = std::chrono::high_resolution_clock::now();
        
        long long call_ts = image_data.timestamp_epoch_ms;

        if (!image_data.buffer) {
            APP_LOG_WARNING("H264Encoder: Received image with null buffer. Skipping.");
            continue;
        }

        // 2. Color conversion (BGR to YUV420p)
        [[maybe_unused]] auto cvtcolor_start = std::chrono::high_resolution_clock::now();
        cv::Mat frame_bgr(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
        cv::Mat frame_yuv;
        cv::cvtColor(frame_bgr, frame_yuv, cv::COLOR_BGR2YUV_I420);
        [[maybe_unused]] auto cvtcolor_end = std::chrono::high_resolution_clock::now();
        APP_LOG_DEBUG("H264Encoder: Time for cvtColor: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(cvtcolor_end - cvtcolor_start).count()) + " us");

        image_data.buffer.reset(); // Release buffer after conversion

        // 3. Copying data to x264 picture_in_
        [[maybe_unused]] auto memcpy_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < height_; ++i) {
            memcpy(picture_in_.img.plane[0] + i * picture_in_.img.i_stride[0],
                   frame_yuv.data + i * width_,
                   width_);
        }
        for (int i = 0; i < height_ / 2; ++i) {
            memcpy(picture_in_.img.plane[1] + i * picture_in_.img.i_stride[1],
                   frame_yuv.data + (width_ * height_) + i * (width_ / 2),
                   width_ / 2);
            memcpy(picture_in_.img.plane[2] + i * picture_in_.img.i_stride[2],
                   frame_yuv.data + (width_ * height_) + (width_ * height_) / 4 + i * (width_ / 2),
                   width_ / 2);
        }
        [[maybe_unused]] auto memcpy_end = std::chrono::high_resolution_clock::now();
        APP_LOG_DEBUG("H264Encoder: Time for memcpy to picture_in_: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - memcpy_start).count()) + " us");

        picture_in_.i_pts++;

        x264_nal_t *nal;
        int i_nal;
        // 4. Encode frame
        auto encoding_start_time = std::chrono::high_resolution_clock::now();
        int frame_size = x264_encoder_encode(encoder_, &nal, &i_nal, &picture_in_, &picture_out_);

        if (frame_size < 0) {
            APP_LOG_ERROR("H264Encoder: x264_encoder_encode failed.");
            continue;
        }

        if (frame_size > 0) {
            auto encoding_end_time = std::chrono::high_resolution_clock::now();
            long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(encoding_end_time - encoding_start_time).count();
            APP_LOG_DEBUG("H264Encoder: Time for x264_encoder_encode: " + std::to_string(duration_ms) + " ms");
            
            CsvLogEntry entry;
            entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            copy_to_array(entry.module, "H264Encoder");
            entry.thread_id = static_cast<long long>(std::hash<std::thread::id>()(std::this_thread::get_id()));
            copy_to_array(entry.event, "encode_done");
            entry.call_ts_epoch_ms = call_ts;
            entry.encoder_encode_ms = static_cast<float>(duration_ms);
            // total_encoded_frames and average_fps are summary metrics, not per-frame
            entry.encoder_total_encoded_frames = -1;
            entry.encoder_average_fps = -1.0f;
            Logger::getInstance().log_csv(entry);

            // 5. NAL unit handling and queue push
            [[maybe_unused]] auto nal_handling_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < i_nal; ++i) {
                auto h264_buffer = h264_buffer_pool_->acquire();
                if (h264_buffer) {
                    if (static_cast<size_t>(nal[i].i_payload) > h264_buffer->data.capacity()) {
                        APP_LOG_WARNING("H264Encoder: NAL unit payload is larger than the buffer capacity. Dropping.");
                        continue;
                    }
                     memcpy(h264_buffer->data.data(), nal[i].p_payload, nal[i].i_payload);
                     h264_buffer->size = nal[i].i_payload;
                     if (!push_latest_only(output_queue_, std::move(h264_buffer))) {
                         APP_LOG_WARNING("H264Encoder: Failed to push H264 buffer with latest-only semantics.");
                     }
                } else {
                    APP_LOG_WARNING("H264Encoder: Failed to acquire buffer for NAL unit. Dropping.");
                }
            }
            [[maybe_unused]] auto nal_handling_end = std::chrono::high_resolution_clock::now();
            APP_LOG_DEBUG("H264Encoder: Time for NAL handling and queue push: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(nal_handling_end - nal_handling_start).count()) + " us");
            
            // Record encoding end time
            image_data.encode_end_time = std::chrono::high_resolution_clock::now();
        }
        [[maybe_unused]] auto total_loop_end = std::chrono::high_resolution_clock::now();
        APP_LOG_DEBUG("H264Encoder: Total worker thread loop iteration time: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(total_loop_end - total_loop_start).count()) + " us");
    }
    APP_LOG_INFO("H264Encoder worker thread stopped.");
}