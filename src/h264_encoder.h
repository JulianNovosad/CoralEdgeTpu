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

class H264Encoder {
public:
    H264Encoder(ImageQueue& input_queue, 
                H264Queue& output_queue, 
                std::shared_ptr<BufferPool<uint8_t>> h264_buffer_pool,
                int width, int height, double fps);
    ~H264Encoder();

    bool start();
    void stop();
    bool is_running() const { return running_; }

private:
    void worker_thread_func();

    ImageQueue& input_queue_;
    H264Queue& output_queue_;
    std::shared_ptr<BufferPool<uint8_t>> h264_buffer_pool_;
    int width_;
    int height_;
    double fps_;

    std::optional<std::thread> worker_thread_;
    std::atomic<bool> running_;
    
    x264_t *encoder_ = nullptr;
    x264_picture_t picture_in_;
    x264_picture_t picture_out_;
    x264_param_t param_;
};

#endif // H264_ENCODER_H
