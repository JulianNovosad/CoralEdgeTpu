#include "h264_encoder.h"
#include <x264.h>

H264Encoder::H264Encoder(ImageQueue& input_queue, 
                         H264Queue& output_queue, 
                         BufferPool<uint8_t>& h264_buffer_pool,
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
        param.i_threads = X264_SYNC_LOOKAHEAD_AUTO; // Use auto threads
        param.b_vfr_input = 0; // Constant frame rate
        param.b_repeat_headers = 1; // Repeat SPS/PPS before IDR frames
        param.b_annexb = 1; // Annex B byte stream format
        
        // Apply profile and tune settings
        x264_param_apply_profile(&param, "baseline"); // Baseline profile for broad compatibility
    // Open the encoder
    encoder_ = x264_encoder_open(&param);
    if (!encoder_) {
        LOG_ERROR("H264Encoder: Failed to open x264 encoder.");
        return false;
    }

    // Allocate pictures
    x264_picture_alloc(&picture_in_, param.i_csp, param.i_width, param.i_height);
    picture_in_.i_pts = 0; // Initialize presentation timestamp

    running_.store(true);
    worker_thread_ = std::thread(&H264Encoder::worker_thread_func, this);
    LOG_INFO("H264Encoder started.");
    return true;
}

void H264Encoder::stop() {
    if (!running_.load()) {
        LOG_WARNING("H264Encoder not running.");
        return;
    }

    running_.store(false);
    input_queue_.set_running(false);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    if (encoder_) {
        // Flush the encoder
        x264_nal_t *nal;
        int i_nal;
        int frame_size;
        while (running_ && x264_encoder_encode(encoder_, &nal, &i_nal, NULL, &picture_out_) > 0) {
             // Acquire a buffer, copy NAL data, and push to the queue
            auto h264_buffer = h264_buffer_pool_.acquire();
            if (h264_buffer && frame_size > 0) {
                memcpy(h264_buffer->data.data(), nal[0].p_payload, frame_size);
                h264_buffer->size = frame_size;
                output_queue_.push(std::move(h264_buffer));
            }
        }
        x264_encoder_close(encoder_);
        encoder_ = nullptr;
    }
    x264_picture_clean(&picture_in_);
    LOG_INFO("H264Encoder stopped.");
}

void H264Encoder::worker_thread_func() {
    LOG_INFO("H264Encoder worker thread started.");
    
    // Get SPS and PPS headers (needed for stream initialization in client)
    x264_nal_t *nal_sps_pps;
    int i_nal_sps_pps;
    if (x264_encoder_headers(encoder_, &nal_sps_pps, &i_nal_sps_pps) < 0) {
        LOG_ERROR("H264Encoder: Failed to get SPS/PPS headers.");
        running_.store(false);
        return;
    }

    for (int i = 0; i < i_nal_sps_pps; ++i) {
        auto h264_buffer = h264_buffer_pool_.acquire();
        if (h264_buffer) {
            memcpy(h264_buffer->data.data(), nal_sps_pps[i].p_payload, nal_sps_pps[i].i_payload);
            h264_buffer->size = nal_sps_pps[i].i_payload;
            output_queue_.push(std::move(h264_buffer));
        }
    }

    while (running_.load()) {
        ImageData image_data;
        if (!input_queue_.pop(image_data)) {
            if (!running_.load()) break;
            continue; 
        }

        if (!image_data.buffer) continue;

        cv::Mat frame_bgr(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
        cv::Mat frame_yuv;
        cv::cvtColor(frame_bgr, frame_yuv, cv::COLOR_BGR2YUV_I420); // Convert BGR to YUV420p

        // Copy YUV data to x264 picture_in_
        memcpy(picture_in_.img.plane[0], frame_yuv.data, width_ * height_);
        memcpy(picture_in_.img.plane[1], frame_yuv.data + (width_ * height_), (width_ * height_) / 4);
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
                auto h264_buffer = h264_buffer_pool_.acquire();
                if (h264_buffer) {
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

