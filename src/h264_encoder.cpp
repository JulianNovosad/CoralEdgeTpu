#include "h264_encoder.h"
#include <x264.h>

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
        LOG_WARNING("H264Encoder already running.");
        return true; // Already running, consider it a success
    }

    running_.store(true);
    // Emplace the thread object into the optional.
    // The x264 encoder will be initialized inside worker_thread_func
    worker_thread_.emplace(&H264Encoder::worker_thread_func, this);
    LOG_INFO("H264Encoder started.");
    return true;
}

void H264Encoder::stop() {
    if (!running_.load()) {
        LOG_WARNING("H264Encoder not running.");
        return;
    }

    running_.store(false);
    if (worker_thread_ && worker_thread_->joinable()) { // Check if optional holds a thread and if it's joinable
        worker_thread_->join();
    }
    
    if (encoder_) { // Check if encoder was successfully opened
        // Flush the encoder
        x264_nal_t *nal;
        int i_nal;
        int frame_size; // Declare frame_size here
        while ((frame_size = x264_encoder_encode(encoder_, &nal, &i_nal, NULL, &picture_out_)) > 0) {
            for (int i = 0; i < i_nal; ++i) {
                auto h264_buffer = h264_buffer_pool_->acquire();
                if (h264_buffer) {
                    if (nal[i].i_payload > h264_buffer->data.capacity()) {
                        LOG_WARNING("H264Encoder::stop(): NAL unit payload is larger than the buffer capacity. Dropping.");
                        continue;
                    }
                    memcpy(h264_buffer->data.data(), nal[i].p_payload, nal[i].i_payload);
                    h264_buffer->size = nal[i].i_payload;
                    output_queue_.push(std::move(h264_buffer));
                } else {
                    LOG_WARNING("H264Encoder::stop(): Failed to acquire buffer for NAL unit during flush. Dropping.");
                }
            }
        }
        x264_encoder_close(encoder_);
        encoder_ = nullptr;
        x264_picture_clean(&picture_in_);
    }
    LOG_INFO("H264Encoder stopped.");
}

void H264Encoder::worker_thread_func() {

    LOG_INFO("H264Encoder worker thread started.");
    LOG_INFO("H264Encoder: Initializing x264 with width=" + std::to_string(width_) + ", height=" + std::to_string(height_) + ", fps=" + std::to_string(fps_));

    

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

        param.i_keyint_max = 1;

    

        LOG_INFO("H264Encoder: x264 parameters - width=" + std::to_string(param.i_width) +

                 ", height=" + std::to_string(param.i_height) +

                 ", csp=" + std::to_string(param.i_csp) +

                 ", fps_num=" + std::to_string(param.i_fps_num) +

                 ", fps_den=" + std::to_string(param.i_fps_den) +

                 ", keyint_max=" + std::to_string(param.i_keyint_max) +

                 ", threads=" + std::to_string(param.i_threads));

    

        LOG_INFO("H264Encoder: Attempting to open x264 encoder...");

        // Open the encoder

        encoder_ = x264_encoder_open(&param); 

        if (!encoder_) {

            LOG_ERROR("H264Encoder: Failed to open x264 encoder.");

            running_.store(false); // Set running_ to false to stop the thread

            return;

        }

        LOG_INFO("H264Encoder: x264 encoder opened successfully.");

    

        LOG_INFO("H264Encoder: Attempting to allocate x264 picture_in_...");

                // Allocate pictures

                if (x264_picture_alloc(&picture_in_, param.i_csp, param.i_width, param.i_height) < 0) {

                    LOG_ERROR("H264Encoder: Failed to allocate x264 picture_in_.");

                    running_.store(false); // Set running_ to false to stop the thread

                    return;

                }

                LOG_INFO("H264Encoder: x264 picture_in_ allocated.");

    picture_in_.i_pts = 0; // Initialize presentation timestamp
    x264_picture_init(&picture_out_); // Initialize picture_out_

    while (running_.load()) {
        ImageData image_data;
        if (!input_queue_.pop(image_data)) {
            if (!running_.load()) break;
            continue; 
        }

        if (!image_data.buffer) {
            LOG_WARNING("H264Encoder: Received image with null buffer. Skipping.");
            continue;
        }


        cv::Mat frame_bgr(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
        cv::Mat frame_yuv;
        cv::cvtColor(frame_bgr, frame_yuv, cv::COLOR_BGR2YUV_I420); // Convert BGR to YUV420p

        LOG_DEBUG("H264Encoder: frame_yuv details - Size: " + std::to_string(frame_yuv.total() * frame_yuv.elemSize()) +
                  ", Dims: " + std::to_string(frame_yuv.cols) + "x" + std::to_string(frame_yuv.rows) +
                  ", Channels: " + std::to_string(frame_yuv.channels()) +
                  ", isContinuous: " + (frame_yuv.isContinuous() ? "true" : "false"));
        
        LOG_DEBUG("H264Encoder: picture_in_ plane info - Y Stride: " + std::to_string(picture_in_.img.i_stride[0]) +
                  ", U Stride: " + std::to_string(picture_in_.img.i_stride[1]) +
                  ", V Stride: " + std::to_string(picture_in_.img.i_stride[2]) +
                  ", Y Plane Ptr: " + std::to_string(reinterpret_cast<uintptr_t>(picture_in_.img.plane[0])));

        // Copy YUV data to x264 picture_in_
        LOG_DEBUG("H264Encoder: Before memcpy to plane[0] - YUV data: " + std::to_string(reinterpret_cast<uintptr_t>(frame_yuv.data)) + ", Plane[0]: " + std::to_string(reinterpret_cast<uintptr_t>(picture_in_.img.plane[0])) + ", Size: " + std::to_string(width_ * height_));
        memcpy(picture_in_.img.plane[0], frame_yuv.data, width_ * height_);
        LOG_DEBUG("H264Encoder: Before memcpy to plane[1] - YUV data offset: " + std::to_string(reinterpret_cast<uintptr_t>(frame_yuv.data + (width_ * height_))) + ", Plane[1]: " + std::to_string(reinterpret_cast<uintptr_t>(picture_in_.img.plane[1])) + ", Size: " + std::to_string((width_ * height_) / 4));
        memcpy(picture_in_.img.plane[1], frame_yuv.data + (width_ * height_), (width_ * height_) / 4);
        LOG_DEBUG("H264Encoder: Before memcpy to plane[2] - YUV data offset: " + std::to_string(reinterpret_cast<uintptr_t>(frame_yuv.data + (width_ * height_) + ((width_ * height_) / 4))) + ", Plane[2]: " + std::to_string(reinterpret_cast<uintptr_t>(picture_in_.img.plane[2])) + ", Size: " + std::to_string((width_ * height_) / 4));
        memcpy(picture_in_.img.plane[2], frame_yuv.data + (width_ * height_) + ((width_ * height_) / 4), (width_ * height_) / 4);

        picture_in_.i_pts++;

        x264_nal_t *nal;
        int i_nal;
        int frame_size = x264_encoder_encode(encoder_, &nal, &i_nal, &picture_in_, &picture_out_);

        if (frame_size < 0) {
            LOG_ERROR("H264Encoder: x264_encoder_encode failed.");
            continue;
        }

        if (frame_size > 0) {
            for (int i = 0; i < i_nal; ++i) {
                auto h264_buffer = h264_buffer_pool_->acquire();
                if (h264_buffer) {
                    if (nal[i].i_payload > h264_buffer->data.capacity()) {
                        LOG_WARNING("H264Encoder: NAL unit payload is larger than the buffer capacity. Dropping.");
                        continue;
                    }
                     memcpy(h264_buffer->data.data(), nal[i].p_payload, nal[i].i_payload);
                     h264_buffer->size = nal[i].i_payload;
                     output_queue_.push(std::move(h264_buffer));
                } else {
                    LOG_WARNING("H264Encoder: Failed to acquire buffer for NAL unit. Dropping.");
                }
            }
        }
    }
    LOG_INFO("H264Encoder worker thread stopped.");
}

