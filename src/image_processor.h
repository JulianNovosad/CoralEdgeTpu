#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <opencv2/opencv.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include "pipeline_structs.h"
#include <libcamera/pixel_format.h>

class ImageProcessor {
public:
    // Constructor for processors that apply detection overlays
    ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                   DetectionResultsQueue& detection_queue,
                   std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                   libcamera::PixelFormat input_pixel_format,
                   int output_width, int output_height);
    
    // Constructor for processors that only do basic processing (like for TPU inference)
    ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                   std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                   libcamera::PixelFormat input_pixel_format,
                   int output_width, int output_height);
    ~ImageProcessor();

    bool start();
        void stop();
        bool is_running() const;
    
        // Set skip factor (process only every Nth frame)
        void set_skip_factor(int skip_factor) { skip_factor_ = skip_factor; }
    
        // Timing methods for monitoring
        long long get_queue_pop_timing_us() const { return avg_queue_pop_time_us_; }
        long long get_preprocess_timing_us() const { return avg_preprocess_time_us_; }
        long long get_conversion_timing_us() const { return avg_conversion_time_us_; }
        long long get_visualization_timing_us() const { return avg_visualization_time_us_; }
    
    // Method to set application reference for updating counters
    void set_application_ref(class Application* app) { app_ref_ = app; }

private:
    // Application reference for updating counters
    class Application* app_ref_ = nullptr;
    void worker_thread_func();
    void apply_detections_to_frame(cv::Mat& frame, const std::shared_ptr<DetectionResultBuffer>& detections);

    ImageQueue& input_queue_;
    ImageQueue& output_queue_;
    DetectionResultsQueue* detection_queue_ptr_;  // Pointer to detection queue (null for non-overlay processors)
    std::shared_ptr<BufferPool<uint8_t>> buffer_pool_;
    libcamera::PixelFormat input_pixel_format_;
    int output_width_;
    int output_height_;
    int skip_factor_ = 1;
    uint64_t frame_counter_ = 0;

    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    
    // Timing statistics
    mutable std::atomic<long long> avg_queue_pop_time_us_{0};
    mutable std::atomic<long long> avg_preprocess_time_us_{0};
    mutable std::atomic<long long> avg_conversion_time_us_{0};      // Additional timing variable
    mutable std::atomic<long long> avg_visualization_time_us_{0};   // Additional timing variable
};