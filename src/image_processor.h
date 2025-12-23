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
    ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                   std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                   libcamera::PixelFormat input_pixel_format,
                   int tpu_input_width, int tpu_input_height);
    ~ImageProcessor();

    bool start();
    void stop();
    bool is_running() const;
    
    // Timing methods for monitoring
    long long get_queue_pop_timing_us() const { return avg_queue_pop_time_us_; }
    long long get_preprocess_timing_us() const { return avg_preprocess_time_us_; }

private:
    void worker_thread_func();

    ImageQueue& input_queue_;
    ImageQueue& output_queue_;
    std::shared_ptr<BufferPool<uint8_t>> buffer_pool_;
    libcamera::PixelFormat input_pixel_format_;
    int tpu_input_width_;
    int tpu_input_height_;

    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    
    // Timing statistics
    mutable std::atomic<long long> avg_queue_pop_time_us_{0};
    mutable std::atomic<long long> avg_preprocess_time_us_{0};
};