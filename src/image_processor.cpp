#include "image_processor.h"
#include "util_logging.h" // For logging macros
#include "application.h"  // For Application counter updates
#include <chrono> // For std::chrono::high_resolution_clock
#include <libcamera/formats.h> // For libcamera::formats and to_string()

// Helper function to convert libcamera::PixelFormat to OpenCV Mat type
// This needs to be robust. For now, assuming common formats.
int libcamera_pixel_format_to_opencv_type(const libcamera::PixelFormat& format) {
    if (format.fourcc() == libcamera::formats::BGRA8888.fourcc()) return CV_8UC4;
    if (format.fourcc() == libcamera::formats::BGR888.fourcc()) return CV_8UC3;
    if (format.fourcc() == libcamera::formats::RGBA8888.fourcc()) return CV_8UC4;
    if (format.fourcc() == libcamera::formats::RGB888.fourcc()) return CV_8UC3;
    if (format.fourcc() == libcamera::formats::YUYV.fourcc()) return CV_8UC2; // YUYV is 2 bytes per pixel
    // Add other formats as needed, or throw an error for unsupported ones
    {
        std::stringstream ss;
        ss << "Unsupported libcamera::PixelFormat encountered (FOURCC: " << std::hex << format.fourcc() << "). Worker thread will exit.";
        std::string log_message = ss.str(); // Make explicit string
        APP_LOG_ERROR(log_message);
    }
    return -1; // Indicate an unsupported format
}

// Constructor for processors that apply detection overlays
ImageProcessor::ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                               DetectionResultsQueue& detection_queue,  // New parameter
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int output_width, int output_height)   // Changed parameter name
    : input_queue_(input_queue), output_queue_(output_queue), detection_queue_ptr_(&detection_queue), // Store as pointer
      buffer_pool_(buffer_pool),
      input_pixel_format_(input_pixel_format),
      output_width_(output_width), output_height_(output_height) { // Changed member names
    {
        std::stringstream ss;
        ss << "ImageProcessor initialized with detection overlay support, output size: " << output_width_ << "x" << output_height_ << ", input format: " << input_pixel_format_.toString().c_str();
        std::string log_message = ss.str(); // Make explicit string
        APP_LOG_INFO(log_message);
    }
}

// Constructor for processors that only do basic processing (like for TPU inference)
ImageProcessor::ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int output_width, int output_height)
    : input_queue_(input_queue), output_queue_(output_queue), detection_queue_ptr_(nullptr), // No detection results
      buffer_pool_(buffer_pool),
      input_pixel_format_(input_pixel_format),
      output_width_(output_width), output_height_(output_height) {
    {
        std::stringstream ss;
        ss << "ImageProcessor initialized without detection overlay support, output size: " << output_width_ << "x" << output_height_ << ", input format: " << input_pixel_format_.toString().c_str();
        std::string log_message = ss.str(); // Make explicit string
        APP_LOG_INFO(log_message);
    }
}

// Destructor
ImageProcessor::~ImageProcessor() {
    stop();
    APP_LOG_INFO("ImageProcessor destroyed.");
}

// Start the worker thread
bool ImageProcessor::start() {
    if (!running_.exchange(true)) {
        worker_thread_ = std::thread(&ImageProcessor::worker_thread_func, this);
        APP_LOG_INFO("ImageProcessor worker thread started.");
        return true;
    }
    return false; // Already running
}

// Stop the worker thread
void ImageProcessor::stop() {
    if (running_.exchange(false)) {
        if (worker_thread_.joinable()) {
            worker_thread_.join();
            APP_LOG_INFO("ImageProcessor worker thread stopped.");
        }
    }
}

// Check if the worker thread is running
bool ImageProcessor::is_running() const {
    return running_.load();
}

// Worker thread function where image processing happens
void ImageProcessor::worker_thread_func() {
    APP_LOG_INFO("ImageProcessor worker thread running with overlay support.");
    set_thread_name("ImageProcessor");

    int opencv_input_type = libcamera_pixel_format_to_opencv_type(input_pixel_format_);
    if (opencv_input_type == -1) {
        std::stringstream ss;
        ss << "ImageProcessor: Failed to determine OpenCV input type for format " << input_pixel_format_.toString().c_str() << ". Worker thread exiting.";
        std::string log_message = ss.str(); // Make explicit string
        APP_LOG_ERROR(log_message);
        running_ = false; // Signal that the thread is stopping
        return; // Exit the worker thread
    }

    ImageData input_image;
    while (running_.load()) {
        // Use non-blocking pop to allow checking running flag
        auto pop_start_time = std::chrono::high_resolution_clock::now();
        if (input_queue_.pop(input_image)) {
            auto pop_end_time = std::chrono::high_resolution_clock::now();
            auto pop_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(pop_end_time - pop_start_time).count();
            
            // Update average queue pop timing for monitoring
            long long current_avg = avg_queue_pop_time_us_.load();
            if (current_avg == 0) {
                avg_queue_pop_time_us_.store(pop_duration_us);
            } else {
                // Use exponential moving average for smoother timing display
                avg_queue_pop_time_us_.store((current_avg * 0.9) + (pop_duration_us * 0.1));
            }
            
            auto process_start_time = std::chrono::high_resolution_clock::now();
            
            // Record queue pop time
            input_image.queue_pop_time = process_start_time;
            
            // Frame skipping logic: Only process every skip_factor_ frame
            if (frame_counter_++ % skip_factor_ != 0) {
                input_image.buffer.reset(); // Return buffer to pool
                continue;
            }
            
            // Log when a frame is dequeued for debugging
            APP_LOG_INFO("ImageProcessor: Dequeued frame from input queue. Frame ID: " + std::to_string(input_image.frame_id) + 
                        ", Timestamp: " + std::to_string(input_image.timestamp_epoch_ms) +
                        ", Size: " + std::to_string(input_image.width) + "x" + std::to_string(input_image.height) +
                        ", Format: " + input_image.format.toString().c_str());
            
            // Debug: Check if frame buffer contains valid data and detect raw Bayer format
            if (input_image.buffer && !input_image.buffer->data.empty()) {
                // Check first few bytes to see if they're all zeros (which would indicate black image)
                size_t check_bytes = std::min(static_cast<size_t>(10), input_image.buffer->data.size());
                bool all_zeros = true;
                for (size_t i = 0; i < check_bytes; ++i) {
                    if (input_image.buffer->data[i] != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                
                // Check if format is raw Bayer which isn't displayable
                std::string format_str = input_image.format.toString().c_str();
                if (format_str.find("BGGR") != std::string::npos || 
                    format_str.find("RAW") != std::string::npos) {
                    APP_LOG_WARNING("ImageProcessor: Received raw Bayer format (" + format_str + ") which is not directly displayable. Frame may appear black.");
                }
                
                if (all_zeros) {
                    APP_LOG_WARNING("ImageProcessor: Frame contains all zeros in first " + std::to_string(check_bytes) + " bytes - may be black image");
                } else {
                    APP_LOG_INFO("ImageProcessor: Frame contains non-zero data in first " + std::to_string(check_bytes) + " bytes");
                }
            } else {
                APP_LOG_WARNING("ImageProcessor: Frame buffer is null or empty");
            }

            // Ensure the input image buffer is valid
            if (!input_image.buffer || input_image.buffer->data.empty() || input_image.width == 0 || input_image.height == 0) {
                APP_LOG_ERROR("ImageProcessor received invalid ImageData (buffer is null, empty or dimensions are zero).");
                continue;
            }

            // 2. Process image (color conversion and/or resizing)
            cv::Mat input_frame_mat;
            auto conversion_start = std::chrono::high_resolution_clock::now();
            if (input_image.format == libcamera::formats::RGB888) {
                input_frame_mat = cv::Mat(input_image.height, input_image.width, CV_8UC3, input_image.buffer->data.data());
                APP_LOG_DEBUG("ImageProcessor: Created Mat with size " + std::to_string(input_image.width) + "x" + std::to_string(input_image.height) + " for RGB888 input");
            } else if (input_image.format == libcamera::formats::YUYV) {
                // Convert YUYV to RGB888 - use faster conversion method
                cv::Mat yuyv_mat = cv::Mat(input_image.height, input_image.width, CV_8UC2, input_image.buffer->data.data());
                // Use a faster conversion method
                cv::cvtColor(yuyv_mat, input_frame_mat, cv::COLOR_YUV2BGR_YUYV, 3); // Use 3 channels for faster processing
                APP_LOG_DEBUG("ImageProcessor: Converted YUYV to BGR, created Mat with size " + std::to_string(input_image.width) + "x" + std::to_string(input_image.height));
            } else {
                APP_LOG_ERROR("ImageProcessor: Unexpected input format (FOURCC: " + std::to_string(input_image.format.fourcc()) + "). Expected RGB888 or YUYV. Skipping frame.");
                input_image.buffer.reset(); // Return buffer to pool
                continue;
            }
            auto conversion_end = std::chrono::high_resolution_clock::now();
            
            // Record conversion timing
            input_image.conversion_start_time = conversion_start;
            input_image.conversion_end_time = conversion_end;
            
            // Calculate and store average conversion time
            auto conversion_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(conversion_end - conversion_start).count();
            // Use atomic operation for thread safety
            long long current_conversion_avg = avg_conversion_time_us_.load();
            avg_conversion_time_us_.store((current_conversion_avg + conversion_duration_us) / 2);

            cv::Mat processed_mat;
            APP_LOG_DEBUG("ImageProcessor: Comparing input size " + std::to_string(input_image.width) + "x" + std::to_string(input_image.height) + 
                          " with target size " + std::to_string(output_width_) + "x" + std::to_string(output_height_));
            if (input_image.width == (unsigned int)output_width_ && input_image.height == (unsigned int)output_height_) {
                // No resizing needed, use the input frame directly
                processed_mat = input_frame_mat; // Avoid clone if we're not going to modify the image
                APP_LOG_DEBUG("ImageProcessor: No resizing needed, using frame directly");
            } else {
                // Resize if dimensions differ - use fastest interpolation for real-time processing
                APP_LOG_WARNING("ImageProcessor: Resizing frame from " + std::to_string(input_image.width) + "x" + std::to_string(input_image.height) +
                                " to " + std::to_string(output_width_) + "x" + std::to_string(output_height_) + ".");
                cv::resize(input_frame_mat, processed_mat, cv::Size(output_width_, output_height_), 0, 0, cv::INTER_NEAREST);
            }

            // 3. Try to get matching detection results for this frame (if detection queue is available)
            auto visualization_start = std::chrono::high_resolution_clock::now();
            if (detection_queue_ptr_ != nullptr) {
                std::shared_ptr<DetectionResultBuffer> detection_buffer;
                // Attempt to pop detection results, but don't block if none are available
                if (detection_queue_ptr_->pop(detection_buffer)) {
                    // If we have an application reference, update the consumed counter
                    if (app_ref_) {
                        app_ref_->increment_inference_results_consumed_by_overlay();
                    }
                    // Apply detection overlays to the processed frame
                    apply_detections_to_frame(processed_mat, detection_buffer);
                    APP_LOG_DEBUG("ImageProcessor: Applied " + std::to_string(detection_buffer->size) + " detections to frame " + std::to_string(input_image.frame_id));
                } else {
                    // No detection results available for this frame, continue without overlays
                    APP_LOG_DEBUG("ImageProcessor: No detection results available for frame " + std::to_string(input_image.frame_id));
                }
            } else {
                // No detection queue configured, continue without overlays
                APP_LOG_DEBUG("ImageProcessor: Detection processing disabled for this instance, frame " + std::to_string(input_image.frame_id));
            }
            auto visualization_end = std::chrono::high_resolution_clock::now();
            
            // Record visualization timing
            input_image.visualization_start_time = visualization_start;
            input_image.visualization_end_time = visualization_end;
            
            // Calculate and store average visualization time
            auto visualization_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(visualization_end - visualization_start).count();
            avg_visualization_time_us_ = (avg_visualization_time_us_.load() + visualization_duration_us) / 2; // Running average

            // 4. Acquire a buffer from the pool for the processed image with overlays.
            std::shared_ptr<PooledBuffer<uint8_t>> processed_buffer_data = buffer_pool_->acquire();
            if (!processed_buffer_data) {
                APP_LOG_WARNING("ImageProcessor: Failed to acquire buffer for processed image. Dropping frame.");
                input_image.buffer.reset(); // Return input buffer to pool
                continue;
            }

            // Ensure processed_buffer_data->data has enough capacity
            size_t required_size = processed_mat.total() * processed_mat.elemSize();
            if (processed_buffer_data->data.size() < required_size) {
                APP_LOG_WARNING("ImageProcessor: Resizing pooled buffer from " + std::to_string(processed_buffer_data->data.size()) +
                              " to " + std::to_string(required_size) + ".");
                processed_buffer_data->data.resize(required_size);
            }
            
            // Safety check to ensure processed_mat is valid before copying
            if (processed_mat.empty() || processed_mat.data == nullptr) {
                APP_LOG_ERROR("ImageProcessor: Processed matrix is empty or has null data. Dropping frame.");
                input_image.buffer.reset(); // Return input buffer to pool
                processed_buffer_data.reset(); // Return acquired buffer to pool
                continue;
            }
            
            // Copy processed image data (with overlays) to the pooled buffer
            std::memcpy(processed_buffer_data->data.data(), processed_mat.data, required_size);
            processed_buffer_data->size = required_size;

            // 5. Create new ImageData object and push to output queue
            ImageData output_image_data(input_image.timestamp_epoch_ms, input_image.frame_id);
            output_image_data.width = output_width_;
            output_image_data.height = output_height_;
            output_image_data.format = libcamera::formats::RGB888; // Output is RGB888 with overlays
            output_image_data.buffer = processed_buffer_data;
            // Pass through zero-copy information if available
            output_image_data.fd = input_image.fd;
            output_image_data.offset = input_image.offset;
            output_image_data.length = input_image.length;
            
            // Record timing measurements
            output_image_data.preprocess_start_time = input_image.preprocess_start_time;
            output_image_data.preprocess_end_time = std::chrono::high_resolution_clock::now();

            // Log when a frame is pushed to the output queue for debugging
            APP_LOG_INFO("ImageProcessor: Pushing processed frame to output queue. Frame ID: " + std::to_string(output_image_data.frame_id) + 
                        ", Timestamp: " + std::to_string(output_image_data.timestamp_epoch_ms) +
                        ", Size: " + std::to_string(output_image_data.width) + "x" + std::to_string(output_image_data.height));
            
            // Debug: Check if processed frame buffer contains valid data
            if (output_image_data.buffer && !output_image_data.buffer->data.empty()) {
                // Check first few bytes to see if they're all zeros (which would indicate black image)
                size_t check_bytes = std::min(static_cast<size_t>(10), output_image_data.buffer->data.size());
                bool all_zeros = true;
                for (size_t i = 0; i < check_bytes; ++i) {
                    if (output_image_data.buffer->data[i] != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (all_zeros) {
                    APP_LOG_WARNING("ImageProcessor: Processed frame contains all zeros in first " + std::to_string(check_bytes) + " bytes - may be black image");
                } else {
                    APP_LOG_INFO("ImageProcessor: Processed frame contains non-zero data in first " + std::to_string(check_bytes) + " bytes");
                }
            } else {
                APP_LOG_WARNING("ImageProcessor: Processed frame buffer is null or empty");
            }
            
            // Use blocking push to ensure every processed frame is pushed to the output queue
            if (!output_queue_.push(std::move(output_image_data))) {
                APP_LOG_WARNING("ImageProcessor failed to push processed frame to output queue.");
                // output_image_data.buffer will be destructed here, returning its buffer to the pool.
            } else {
                APP_LOG_DEBUG("ImageProcessor: Successfully pushed processed frame to output queue. Frame ID: " + std::to_string(output_image_data.frame_id));
            }

            auto process_end_time = std::chrono::high_resolution_clock::now();
            [[maybe_unused]] long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(process_end_time - process_start_time).count();
            
            // Calculate detailed timing breakdown
            auto preprocess_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(process_end_time - input_image.queue_pop_time).count();
            // Update average preprocessing time with proper exponential moving average
            long long current_preprocess_avg = avg_preprocess_time_us_.load();
            avg_preprocess_time_us_ = static_cast<long long>(current_preprocess_avg * 0.9 + preprocess_duration_us * 0.1); // 0.1 alpha value for EMA
            
            APP_LOG_DEBUG("ImageProcessor processed frame with overlays in " + std::to_string(duration_ms) + " ms. Input size " +
                          std::to_string(input_image.width) + "x" + std::to_string(input_image.height) + ", Output size " +
                          std::to_string(output_width_) + "x" + std::to_string(output_height_) + ", Format RGB888");
            
            // Log detailed timing breakdown if available
            if (input_image.conversion_start_time.time_since_epoch().count() > 0 && 
                input_image.conversion_end_time.time_since_epoch().count() > 0) {
                auto conversion_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    input_image.conversion_end_time - input_image.conversion_start_time).count();
                APP_LOG_DEBUG("ImageProcessor timing breakdown - Conversion: " + std::to_string(conversion_time_us) + " us");
            }
            
            if (input_image.visualization_start_time.time_since_epoch().count() > 0 && 
                input_image.visualization_end_time.time_since_epoch().count() > 0) {
                auto visualization_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    input_image.visualization_end_time - input_image.visualization_start_time).count();
                APP_LOG_DEBUG("ImageProcessor timing breakdown - Visualization: " + std::to_string(visualization_time_us) + " us");
            }
        } else {
            // Update queue pop timing even when pop fails to prevent 0 timing
            auto pop_end_time = std::chrono::high_resolution_clock::now();
            auto pop_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(pop_end_time - pop_start_time).count();
            
            // Update average queue pop timing for monitoring - use the time it took to fail to pop
            long long current_avg = avg_queue_pop_time_us_.load();
            if (current_avg == 0) {
                avg_queue_pop_time_us_.store(pop_duration_us);
            } else {
                // Use exponential moving average for smoother timing display
                avg_queue_pop_time_us_.store((current_avg * 0.9) + (pop_duration_us * 0.1));
            }
            
            // Minimal sleep to prevent busy-waiting when no input is available
            std::this_thread::sleep_for(std::chrono::microseconds(100)); // Reduced from 1ms to 100us to reduce latency
        }
    }
    
    APP_LOG_INFO("ImageProcessor worker thread finished.");
}

// Function to apply detection results as overlays to an image
void ImageProcessor::apply_detections_to_frame(cv::Mat& frame, const std::shared_ptr<DetectionResultBuffer>& detections) {
    if (!detections) {
        return; // Nothing to do if no detection buffer
    }

    int frame_width = frame.cols;
    int frame_height = frame.rows;

    // Process each detection in the buffer
    for (size_t i = 0; i < detections->size; ++i) {
        const DetectionResult& detection = detections->data[i];

        // Convert normalized coordinates to pixel coordinates
        int x_min = static_cast<int>(detection.xmin * frame_width);
        int y_min = static_cast<int>(detection.ymin * frame_height);
        int x_max = static_cast<int>(detection.xmax * frame_width);
        int y_max = static_cast<int>(detection.ymax * frame_height);

        // Ensure coordinates are within frame bounds
        x_min = std::max(0, x_min);
        y_min = std::max(0, y_min);
        x_max = std::min(frame_width - 1, x_max);
        y_max = std::min(frame_height - 1, y_max);

        // Draw bounding box in bright red - use faster line drawing
        cv::Scalar box_color(0, 0, 255); // Red (BGR format)
        cv::rectangle(frame, cv::Point(x_min, y_min), cv::Point(x_max, y_max), box_color, 2, cv::LINE_8, 0);

        // Draw class ID and score - only if box is large enough to show text
        if ((x_max - x_min) > 20 && (y_max - y_min) > 20) {
            std::string label = "ID:" + std::to_string(detection.class_id) + " S:" + std::to_string(static_cast<int>(detection.score * 100)) + "%";

            // Calculate text size to position it above the bounding box
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline); // Smaller font for better performance

            // Draw label background
            cv::Point label_origin(x_min, std::max(y_min - textSize.height, 0));
            cv::Point label_bottom_right(std::min(label_origin.x + textSize.width, frame_width), 
                                       std::min(label_origin.y + textSize.height + baseline, frame_height));

            // Draw filled rectangle for label background
            cv::rectangle(frame, label_origin, label_bottom_right, box_color, cv::FILLED);

            // Draw label text
            cv::putText(frame, label, cv::Point(label_origin.x, label_origin.y + textSize.height), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_8, false);
        }
    }
}