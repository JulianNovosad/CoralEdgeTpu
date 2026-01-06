// Verified headers: [pipeline_structs.h, util_logging.h, buffer_pool.h, optional, thread...]
// Verification timestamp: 2026-01-06 17:08:04
#ifndef H264_ENCODER_H
#define H264_ENCODER_H

#include "pipeline_structs.h"
#include "util_logging.h"
#include "buffer_pool.h"

#include <optional> // For std::optional
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <x264.h>

class Application;

class H264Encoder {
public:
    H264Encoder(ImageQueue& input_queue, 
                H264Queue& output_queue, 
                std::shared_ptr<BufferPool<uint8_t>> h264_buffer_pool,
                std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                int width, int height, double fps);
    ~H264Encoder();

    bool start();
    void stop();
    bool is_running() const { return running_; }
    void get_state() const;
    void set_application_ref(Application* app);
    
    // Timing methods for monitoring
    long long get_encode_timing_us() const { return avg_encode_time_us_.load(); }
    long long get_nal_timing_us() const { return avg_nal_time_us_.load(); }
    float get_latest_bitrate_mbps() const { return latest_bitrate_mbps_.load(); }
    int get_queue_depth() const { 
        return 0; 
    }

private:
    void worker_thread_func();

    ImageQueue& input_queue_;
    H264Queue& output_queue_;
    std::shared_ptr<BufferPool<uint8_t>> h264_buffer_pool_;
    std::shared_ptr<ObjectPool<ImageData>> image_data_pool_;
    Application* app_ = nullptr;
    int width_;
    int height_;
    double fps_;

    std::optional<std::thread> worker_thread_;
    std::atomic<bool> running_;
    
    x264_t *encoder_ = nullptr;
    x264_picture_t picture_in_;
    x264_picture_t picture_out_;
    x264_param_t param_;
    
    // Timing statistics
    mutable std::atomic<long long> avg_encode_time_us_{0};
    mutable std::atomic<long long> avg_nal_time_us_{0};
    mutable std::atomic<float> latest_bitrate_mbps_{0.0f};
    
    // Display starvation detection
    mutable std::atomic<bool> first_frame_sent_{false};
    mutable std::chrono::steady_clock::time_point last_frame_processed_time_;
    mutable int last_frame_id_{-1};
    
    // Throttling logic
    uint64_t frame_counter_{0};
    
    // Frame counter for PTS
    int64_t frame_count_{0};
    
    // Presentation timestamp tracking for monotonicity
    std::chrono::steady_clock::time_point stream_start_time_;
    bool stream_start_time_initialized_{false};
    
    // Method for checking display starvation
    bool is_display_starving() const;
};

#endif // H264_ENCODER_H