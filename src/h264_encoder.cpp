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

    

    // Initialize x264 parameters

    x264_param_t param;

    x264_param_default(&param); // Start with sane defaults

    x264_param_default_preset(&param, "ultrafast", "zerolatency");

    

    // Configure input resolution and pixel format

    param.i_width = width_;

    param.i_height = height_;

    param.i_fps_num = static_cast<int>(fps_);

    param.i_fps_den = 1;

    param.i_keyint_max = static_cast<int>(fps_); // Keyframe every second

    param.b_intra_refresh = 1; // IDR frames instead of full I-frames to help with seeking

    

    // Input pixel format (OpenCV uses BGR, but we convert to YUV420p for x264)

        param.i_csp = X264_CSP_I420; 

    

        // Stream parameters

        param.i_threads = 1; // Simplify to 1 thread for debugging

        param.b_vfr_input = 0; // Constant frame rate

        param.b_repeat_headers = 1; // Repeat SPS/PPS before IDR frames

        param.b_annexb = 1; // Annex B byte stream format

    

        // Apply profile and tune settings

        x264_param_apply_profile(&param, "baseline"); // Baseline profile for broad compatibility

    

        // Simplify keyframe interval for debugging

        param.i_keyint_max = static_cast<int>(fps_);

    

        APP_LOG_INFO("H264Encoder: x264 parameters - width=" + std::to_string(param.i_width) +

                 ", height=" + std::to_string(param.i_height) +

                 ", csp=" + std::to_string(param.i_csp) +

                 ", fps_num=" + std::to_string(param.i_fps_num) +

                 ", fps_den=" + std::to_string(param.i_fps_den) +

                 ", keyint_max=" + std::to_string(param.i_keyint_max) +

                 ", threads=" + std::to_string(param.i_threads));

    

        APP_LOG_INFO("H264Encoder: Attempting to open x264 encoder...");

        // Open the encoder

        encoder_ = x264_encoder_open(&param); 

        if (!encoder_) {

            APP_LOG_ERROR("H264Encoder: Failed to open x264 encoder.");

            running_.store(false); // Set running_ to false to stop the thread

            return;

        }

        APP_LOG_INFO("H264Encoder: x264 encoder opened successfully.");

    

        APP_LOG_INFO("H264Encoder: Attempting to allocate x264 picture_in_...");

                // Allocate pictures

                if (x264_picture_alloc(&picture_in_, param.i_csp, param.i_width, param.i_height) < 0) {

                    APP_LOG_ERROR("H264Encoder: Failed to allocate x264 picture_in_.");

                    running_.store(false); // Set running_ to false to stop the thread

                    return;

                }

                APP_LOG_INFO("H264Encoder: x264 picture_in_ allocated.");

    picture_in_.i_pts = 0; // Initialize presentation timestamp
    x264_picture_init(&picture_out_); // Initialize picture_out_

    while (running_.load()) {
        ImageData image_data;
        if (!input_queue_.pop(image_data)) {
            if (!running_.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Prevent busy-waiting
            continue; 
        }
        
        long long call_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                              image_data.timestamp.time_since_epoch()).count();

        if (!image_data.buffer) {
            APP_LOG_WARNING("H264Encoder: Received image with null buffer. Skipping.");
            continue;
        }

        cv::Mat frame_bgr(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
        cv::Mat frame_yuv;
        cv::cvtColor(frame_bgr, frame_yuv, cv::COLOR_BGR2YUV_I420);

        image_data.buffer.reset();

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

        picture_in_.i_pts++;

        x264_nal_t *nal;
        int i_nal;
        auto encoding_start_time = std::chrono::high_resolution_clock::now();
        int frame_size = x264_encoder_encode(encoder_, &nal, &i_nal, &picture_in_, &picture_out_);

        if (frame_size < 0) {
            APP_LOG_ERROR("H264Encoder: x264_encoder_encode failed.");
            continue;
        }

        if (frame_size > 0) {
            auto encoding_end_time = std::chrono::high_resolution_clock::now();
            long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(encoding_end_time - encoding_start_time).count();
            
            std::stringstream custom_metrics;
            custom_metrics << "{\"encode_ms\":" << duration_ms << "}";
            APP_LOG_CSV("H264Encoder", "encode_done", call_ts, custom_metrics.str());

            {
                std::lock_guard<std::mutex> lock(encoding_times_mutex_);
                encoding_times_ms_.push_back(duration_ms);
                total_encoded_frames_++;
            }
            for (int i = 0; i < i_nal; ++i) {
                auto h264_buffer = h264_buffer_pool_->acquire();
                if (h264_buffer) {
                    if (static_cast<size_t>(nal[i].i_payload) > h264_buffer->data.capacity()) {
                        APP_LOG_WARNING("H264Encoder: NAL unit payload is larger than the buffer capacity. Dropping.");
                        continue;
                    }
                     memcpy(h264_buffer->data.data(), nal[i].p_payload, nal[i].i_payload);
                     h264_buffer->size = nal[i].i_payload;
                     output_queue_.push(std::move(h264_buffer));
                } else {
                    APP_LOG_WARNING("H264Encoder: Failed to acquire buffer for NAL unit. Dropping.");
                }
            }
        }
    }
    APP_LOG_INFO("H264Encoder worker thread stopped.");
}

void H264Encoder::get_performance_metrics() {
    std::lock_guard<std::mutex> lock(encoding_times_mutex_);

    if (total_encoded_frames_ == 0) {
        APP_LOG_INFO("H264Encoder: No frames encoded for performance metrics.");
        return;
    }

    double average_latency_ms = 0;
    for (long long latency : encoding_times_ms_) {
        average_latency_ms += static_cast<double>(latency);
    }
    average_latency_ms /= total_encoded_frames_;
    double average_fps = 1000.0 / average_latency_ms;

    std::sort(encoding_times_ms_.begin(), encoding_times_ms_.end());
    size_t percentile_99_index = static_cast<size_t>(std::round(total_encoded_frames_ * 0.99));
    size_t percentile_95_index = static_cast<size_t>(std::round(total_encoded_frames_ * 0.95));
    size_t percentile_50_index = static_cast<size_t>(std::round(total_encoded_frames_ * 0.50));

    long long p99_latency_ms = encoding_times_ms_[std::min(percentile_99_index, static_cast<size_t>(total_encoded_frames_ - 1))];
    long long p95_latency_ms = encoding_times_ms_[std::min(percentile_95_index, static_cast<size_t>(total_encoded_frames_ - 1))];
    long long p50_latency_ms = encoding_times_ms_[std::min(percentile_50_index, static_cast<size_t>(total_encoded_frames_ - 1))];

    std::ostringstream json_metrics;
    json_metrics << "{\"p50_latency_ms\":" << std::fixed << std::setprecision(3) << static_cast<double>(p50_latency_ms)
                 << ",\"p95_latency_ms\":" << static_cast<double>(p95_latency_ms)
                 << ",\"p99_latency_ms\":" << static_cast<double>(p99_latency_ms)
                 << ",\"average_fps\":" << average_fps
                 << ",\"total_encoded_frames\":" << total_encoded_frames_
                 << ",\"average_latency_ms\":" << average_latency_ms
                 << "}";

    APP_LOG_CSV("H264Encoder", "PerformanceMetrics", 0LL, json_metrics.str());
    APP_LOG_INFO("--- H264Encoder Performance Metrics ---");
    APP_LOG_INFO("  Total Encoded Frames: " + std::to_string(total_encoded_frames_));
    APP_LOG_INFO("  Average FPS: " + std::to_string(average_fps));
    APP_LOG_INFO("  Average Latency: " + std::to_string(average_latency_ms) + " ms");
    APP_LOG_INFO("  50th Percentile Latency: " + std::to_string(p50_latency_ms) + " ms");
    APP_LOG_INFO("  95th Percentile Latency: " + std::to_string(p95_latency_ms) + " ms");
    APP_LOG_INFO("  99th Percentile Latency: " + std::to_string(p99_latency_ms) + " ms");
    APP_LOG_INFO("---------------------------------------");

    encoding_times_ms_.clear();
    total_encoded_frames_ = 0;
}

