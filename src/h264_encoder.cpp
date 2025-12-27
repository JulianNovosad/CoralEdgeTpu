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

    param_.i_fps_num = 40; // Throttled from 120 FPS

    param_.i_fps_den = 1;

    param_.i_keyint_max = 20; // Keyframe every 0.5 seconds at 40 FPS

    param_.b_intra_refresh = 0; // Traditional Full IDR frames for VLC compatibility

    

    // Input pixel format (OpenCV uses BGR, but we convert to YUV420p for x264)

    param_.i_csp = X264_CSP_I420; 

    

    // Stream parameters

    param_.i_threads = 1; // Simplify to 1 thread for debugging

    param_.b_vfr_input = 0; // Constant frame rate

    param_.b_repeat_headers = 1; // Repeat SPS/PPS before IDR frames

    param_.b_annexb = 1; // Annex B format (start codes) for better compatibility
    
    // Keyframe and IDR frame settings for proper decoding (already set earlier)
    param_.i_keyint_min = 1; // Minimum keyframe interval
    param_.i_bframe = 0; // No B-frames for simpler decoding and lower latency

    

    // Apply profile and tune settings

    x264_param_apply_profile(&param_, "baseline"); // Baseline profile for broad compatibility
    
    // Additional settings for better header handling and streaming
    param_.i_frame_reference = 1; // Limit reference frames for lower latency
    param_.i_dpb_size = 1; // Decoded picture buffer size (minimum for low latency)
    param_.analyse.b_transform_8x8 = 0; // Disable 8x8 transforms for simpler decoding
    param_.rc.i_qp_constant = 28; // Use constant quality
    param_.rc.i_rc_method = X264_RC_CQP; // Constant quantizer
    param_.i_cqm_preset = X264_CQM_FLAT; // Use flat quantization matrices
    // Additional low-latency settings
    param_.i_bframe_adaptive = X264_B_ADAPT_NONE; // Disable adaptive B-frames
    param_.i_bframe_bias = 0; // B-frame bias
    param_.i_frame_packing = -1; // No frame packing
    
    // Additional ultra-low latency settings
    param_.rc.i_vbv_buffer_size = 0; // Disable VBV buffering for lower latency
    param_.rc.i_vbv_max_bitrate = 0; // Disable VBV bitrate limiting
    param_.i_sync_lookahead = 0; // Disable sync lookahead for lower latency
    param_.i_bframe_pyramid = X264_B_PYRAMID_NONE; // Disable B-frame pyramid
    param_.b_aud = 0; // Don't emit access unit delimiters
    param_.b_repeat_headers = 1; // Ensure headers are repeated for robustness
    param_.i_slice_max_size = 1500; // Limit slice size to improve network streaming
    param_.i_slice_max_mbs = 0; // No slice macroblock limit
    // Performance optimizations
    param_.analyse.inter = 0; // Disable inter analysis for speed
    param_.analyse.i_me_method = X264_ME_DIA; // Fastest motion estimation
    param_.analyse.i_subpel_refine = 0; // Minimal subpixel refinement
    param_.rc.i_qp_min = 28; // Minimum quantizer
    param_.rc.i_qp_max = 35; // Maximum quantizer for performance

    



    

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
            // Reduced sleep to 25 microseconds to improve responsiveness
            std::this_thread::sleep_for(std::chrono::microseconds(25));
            continue; 
        }
        
        // Throttling: Only process every 3rd frame (120 FPS -> 40 FPS)
        if (frame_counter_++ % 3 != 0) {
            image_data.buffer.reset(); // Ensure buffer is released
            continue;
        }
        
        // Log when a frame is received for encoding
        APP_LOG_DEBUG("H264Encoder: Received frame for encoding. Frame ID: " + std::to_string(image_data.frame_id) + 
                    ", Timestamp: " + std::to_string(image_data.timestamp_epoch_ms));
        [[maybe_unused]] auto pop_end = std::chrono::high_resolution_clock::now();
        APP_LOG_DEBUG("H264Encoder: Time to pop from queue: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(pop_end - pop_start).count()) + " us");
        
        // Record queue pop time
        image_data.encode_start_time = std::chrono::high_resolution_clock::now();
        
        // Calculate and log input latency if capture time is available
        if (image_data.capture_time.time_since_epoch().count() > 0) {
            auto input_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                image_data.encode_start_time - image_data.capture_time).count();
            (void)input_latency_us; // Suppress unused variable warning in release builds
            APP_LOG_DEBUG("H264Encoder: Input latency (capture to encode start): " + std::to_string(input_latency_us) + " us");
        }

        if (!image_data.buffer) {
            APP_LOG_WARNING("H264Encoder: Received image with null buffer. Skipping.");
            continue;
        }

        // Log input frame information before processing
        static int input_frame_counter = 0;
        if (input_frame_counter < 5) {  // Log first 5 input frames
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            APP_LOG_INFO("INPUT FRAME #" + std::to_string(input_frame_counter) + 
                        ": Width=" + std::to_string(image_data.width) + 
                        ", Height=" + std::to_string(image_data.height) +
                        ", Buffer Size=" + std::to_string(image_data.buffer ? image_data.buffer->data.size() : 0) +
                        ", Format=" + std::to_string(image_data.format.fourcc()) +  // This will show the libcamera format
                        ", Timestamp=" + std::to_string(timestamp) + "ms" +
                        ", Frame ID=" + std::to_string(image_data.frame_id));
            input_frame_counter++;
        }

        // 2. Color conversion (Ensure proper format for x264 encoding)
        [[maybe_unused]] auto cvtcolor_start = std::chrono::high_resolution_clock::now();
        
        // Check if buffer is valid before creating cv::Mat
        if (!image_data.buffer || image_data.buffer->data.data() == nullptr) {
            APP_LOG_WARNING("H264Encoder: Invalid buffer data. Skipping frame.");
            continue;
        }
        
        cv::Mat frame_bgr, frame_yuv;
        
        // Based on the pipeline, the overlaid video should be in the format from VisualizationProcessor
        // Let's be explicit about the expected format and handle it properly
        size_t expected_bgr_size = image_data.width * image_data.height * 3;  // BGR: 3 bytes per pixel
        size_t actual_size = image_data.buffer->data.size();
        
        APP_LOG_DEBUG("H264Encoder: Processing frame - width: " + std::to_string(image_data.width) + 
                     ", height: " + std::to_string(image_data.height) + 
                     ", buffer size: " + std::to_string(actual_size) +
                     ", expected BGR size: " + std::to_string(expected_bgr_size));
        
        if (actual_size >= expected_bgr_size) {
            // Create OpenCV matrix with the expected BGR format
            cv::Mat frame_bgr_raw(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
            if (frame_bgr_raw.empty()) {
                APP_LOG_ERROR("H264Encoder: Failed to create OpenCV matrix from buffer data. Skipping frame.");
                continue;
            }
            // Convert BGR to YUV420p directly without intermediate BGR copy to improve performance
            cv::cvtColor(frame_bgr_raw, frame_yuv, cv::COLOR_BGR2YUV_I420);
        } else {
            APP_LOG_WARNING("H264Encoder: Buffer size (" + std::to_string(actual_size) + 
                           ") is smaller than expected BGR size (" + std::to_string(expected_bgr_size) + 
                           "). Attempting to use available data.");
            
            // Try to work with what we have - could be different format from VisualizationProcessor
            size_t expected_yuyv_size = image_data.width * image_data.height * 2;  // YUYV: 2 bytes per pixel
            size_t expected_gray_size = image_data.width * image_data.height;      // Grayscale: 1 byte per pixel
            
            if (actual_size == expected_yuyv_size) {
                // Input is YUYV format from camera
                cv::Mat frame_yuyv(image_data.height, image_data.width, CV_8UC2, image_data.buffer->data.data());
                if (frame_yuyv.empty()) {
                    APP_LOG_ERROR("H264Encoder: Failed to create YUYV matrix from buffer data. Skipping frame.");
                    continue;
                }
                cv::cvtColor(frame_yuyv, frame_bgr, cv::COLOR_YUV2BGR_YUYV);
                cv::cvtColor(frame_bgr, frame_yuv, cv::COLOR_BGR2YUV_I420);
            } else if (actual_size == expected_gray_size) {
                // Input is grayscale
                cv::Mat frame_gray(image_data.height, image_data.width, CV_8UC1, image_data.buffer->data.data());
                if (frame_gray.empty()) {
                    APP_LOG_ERROR("H264Encoder: Failed to create grayscale matrix from buffer data. Skipping frame.");
                    continue;
                }
                cv::cvtColor(frame_gray, frame_bgr, cv::COLOR_GRAY2BGR);
                cv::cvtColor(frame_bgr, frame_yuv, cv::COLOR_BGR2YUV_I420);
            } else {
                // Unknown format, try to interpret as grayscale and convert to BGR
                APP_LOG_WARNING("H264Encoder: Unknown format, attempting to interpret as grayscale and convert to BGR.");
                if (actual_size >= (image_data.width * image_data.height)) {
                    cv::Mat frame_gray(image_data.height, image_data.width, CV_8UC1, image_data.buffer->data.data());
                    if (frame_gray.empty()) {
                        APP_LOG_ERROR("H264Encoder: Failed to create grayscale matrix from buffer data. Skipping frame.");
                        continue;
                    }
                    cv::cvtColor(frame_gray, frame_bgr, cv::COLOR_GRAY2BGR);
                    cv::cvtColor(frame_bgr, frame_yuv, cv::COLOR_BGR2YUV_I420);
                } else {
                    APP_LOG_ERROR("H264Encoder: Buffer size too small. Skipping frame.");
                    continue;
                }
            }
        }
        [[maybe_unused]] auto cvtcolor_end = std::chrono::high_resolution_clock::now();
        APP_LOG_DEBUG("H264Encoder: Time for cvtColor: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(cvtcolor_end - cvtcolor_start).count()) + " us");

        image_data.buffer.reset(); // Release buffer after conversion

        // 3. Copying data to x264 picture_in_ with proper stride handling
        [[maybe_unused]] auto memcpy_start = std::chrono::high_resolution_clock::now();
        
        // More efficient copying with direct memory access
        // Copy Y plane (luminance)
        const int y_plane_size = width_ * height_;
        memcpy(picture_in_.img.plane[0], frame_yuv.data, y_plane_size);
        
        // Copy U plane (chrominance blue) 
        const int uv_plane_size = width_ * height_ / 4;  // For I420 format
        const int u_offset = y_plane_size;
        memcpy(picture_in_.img.plane[1], frame_yuv.data + u_offset, uv_plane_size);
        
        // Copy V plane (chrominance red)
        const int v_offset = u_offset + uv_plane_size;
        memcpy(picture_in_.img.plane[2], frame_yuv.data + v_offset, uv_plane_size);
        [[maybe_unused]] auto memcpy_end = std::chrono::high_resolution_clock::now();
        APP_LOG_DEBUG("H264Encoder: Time for memcpy to picture_in_: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - memcpy_start).count()) + " us");

        picture_in_.i_pts++;

        x264_nal_t *nal;
        int i_nal;
        // 4. Encode frame
        auto encoding_start_time = std::chrono::high_resolution_clock::now();
        
        // Force the first frame to be a keyframe to ensure SPS/PPS headers are sent
        if (!first_frame_sent_.load()) {
            picture_in_.i_type = X264_TYPE_IDR;
            first_frame_sent_.store(true);
        } else {
            picture_in_.i_type = X264_TYPE_P; // Use P-frames for better performance after first frame
        }
        
        int frame_size = x264_encoder_encode(encoder_, &nal, &i_nal, &picture_in_, &picture_out_);

        if (frame_size < 0) {
            APP_LOG_ERROR("H264Encoder: x264_encoder_encode failed with frame_size: " + std::to_string(frame_size));
            continue;
        }
        
        // Check if encoder is still valid after encoding
        if (!encoder_) {
            APP_LOG_ERROR("H264Encoder: Encoder became invalid during encoding. Stopping thread.");
            running_.store(false);
            break;
        }

        APP_LOG_DEBUG("H264Encoder: x264_encoder_encode returned frame_size: " + std::to_string(frame_size) + ", i_nal: " + std::to_string(i_nal));

        if (frame_size > 0 && nal != nullptr && i_nal > 0) {
            auto encoding_end_time = std::chrono::high_resolution_clock::now();
            long long duration_us = std::chrono::duration_cast<std::chrono::microseconds>(encoding_end_time - encoding_start_time).count();
            APP_LOG_DEBUG("H264Encoder: Time for x264_encoder_encode: " + std::to_string(duration_us / 1000) + " ms");
            
            // Update average encoding time for monitoring
            long long current_avg = avg_encode_time_us_.load();
            avg_encode_time_us_.store((current_avg + duration_us) / 2);
            
            // 5. Bundle all NAL units into a single buffer for the output queue
            // This ensures all NALs of a single frame have the same timestamp in the RTSP server
            auto h264_buffer = h264_buffer_pool_->acquire();
            if (h264_buffer) {
                size_t total_size = 0;
                for (int i = 0; i < i_nal; ++i) {
                    total_size += nal[i].i_payload;
                }

                if (total_size > h264_buffer->data.capacity()) {
                    APP_LOG_ERROR("H264Encoder: Total NAL size (" + std::to_string(total_size) + 
                                 ") exceeds buffer capacity (" + std::to_string(h264_buffer->data.capacity()) + ").");
                } else {
                    size_t offset = 0;
                    for (int i = 0; i < i_nal; ++i) {
                        memcpy(h264_buffer->data.data() + offset, nal[i].p_payload, nal[i].i_payload);
                        
                        // Correctly identify NAL type by skipping start code (Annex-B)
                        uint8_t* p = nal[i].p_payload;
                        int p_size = nal[i].i_payload;
                        int start_offset = 0;
                        if (p_size >= 4 && p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 1) start_offset = 4;
                        else if (p_size >= 3 && p[0] == 0 && p[1] == 0 && p[2] == 1) start_offset = 3;
                        
                        uint8_t nal_type = (p_size > start_offset) ? (p[start_offset] & 0x1F) : 0;
                        
                        offset += nal[i].i_payload;

                        // Log keyframes and SPS/PPS for debugging
                        if (nal_type == 5 || nal_type == 7 || nal_type == 8) {
                            const char* nal_type_str = (nal_type == 5) ? "IDR-Slice" : (nal_type == 7 ? "SPS" : "PPS");
                            APP_LOG_INFO("H264Encoder: Bundled " + std::string(nal_type_str) + 
                                        " NAL. Size: " + std::to_string(nal[i].i_payload));
                        }
                    }
                    h264_buffer->size = total_size;
                    
                    if (output_queue_.push(std::move(h264_buffer))) {
                        APP_LOG_DEBUG("H264Encoder: Successfully pushed bundled H264 buffer to output queue. Total size: " + std::to_string(total_size));
                    } else {
                        APP_LOG_WARNING("H264Encoder: Failed to push H264 buffer to output queue.");
                    }
                }
            } else {
                APP_LOG_WARNING("H264Encoder: Failed to acquire buffer for NAL units. Dropping frame.");
            }

            // Record encoding end time
            image_data.encode_end_time = std::chrono::high_resolution_clock::now();
            
            // Update last frame processing time for starvation detection
            last_frame_processed_time_ = std::chrono::high_resolution_clock::now();
            last_frame_id_ = image_data.frame_id;
        } else {
            APP_LOG_DEBUG("H264Encoder: No NAL units to process (frame_size: " + std::to_string(frame_size) + ", i_nal: " + std::to_string(i_nal) + ")");
        }
        [[maybe_unused]] auto total_loop_end = std::chrono::high_resolution_clock::now();
        APP_LOG_DEBUG("H264Encoder: Total worker thread loop iteration time: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(total_loop_end - total_loop_start).count()) + " us");
    }
    APP_LOG_INFO("H264Encoder worker thread stopped.");
}

bool H264Encoder::is_display_starving() const {
    if (!running_.load()) {
        return true; // If encoder isn't running, display is definitely starving
    }
    
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_since_last_frame = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - last_frame_processed_time_
    ).count();
    
    // Consider starving if no frame has been processed in more than 2x the expected frame interval
    // For example, if FPS is 30, expect frames every ~33ms, so 66ms+ would be starving
    double expected_frame_interval_ms = 1000.0 / fps_;
    double starvation_threshold_ms = expected_frame_interval_ms * 2.0;  // 2x the expected interval
    
    if (time_since_last_frame > starvation_threshold_ms) {
        APP_LOG_WARNING("H264Encoder: Display starvation detected! Last frame was " + 
                       std::to_string(time_since_last_frame) + "ms ago (threshold: " + 
                       std::to_string(starvation_threshold_ms) + "ms)");
        return true;
    }
    
    return false;
}