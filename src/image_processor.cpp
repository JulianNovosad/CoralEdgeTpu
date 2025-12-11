#include "image_processor.h"
#include "util_logging.h" // For logging macros
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

// Constructor
ImageProcessor::ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int tpu_input_width, int tpu_input_height)
    : input_queue_(input_queue), output_queue_(output_queue), buffer_pool_(buffer_pool),
      input_pixel_format_(input_pixel_format),
      tpu_input_width_(tpu_input_width), tpu_input_height_(tpu_input_height) {
    {
        std::stringstream ss;
        ss << "ImageProcessor initialized with TPU input size: " << tpu_input_width_ << "x" << tpu_input_height_ << ", input format: " << input_pixel_format_.toString().c_str();
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
    APP_LOG_INFO("ImageProcessor worker thread running.");
    set_thread_name("ImageProcessor");

    int opencv_input_type = libcamera_pixel_format_to_opencv_type(input_pixel_format_);
    if (opencv_input_type == -1) {
        std::stringstream ss;
        ss << "ImageProcessor: Failed to determine OpenCV input type for format " << input_pixel_format_.toString().c_str() << ". Worker thread exiting.";
        APP_LOG_ERROR(ss.str());
        running_ = false; // Signal that the thread is stopping
        return; // Exit the worker thread
    }

    while (running_.load()) {
        ImageData input_image;
        // Wait for an image to pop from the queue, with a small timeout to allow stopping
        if (input_queue_.pop(input_image)) {
            auto process_start_time = std::chrono::high_resolution_clock::now();

            // Ensure the input image buffer is valid
            if (!input_image.buffer || input_image.buffer->data.empty() || input_image.width == 0 || input_image.height == 0) {
                APP_LOG_ERROR("ImageProcessor received invalid ImageData (buffer is null, empty or dimensions are zero).");
                continue;
            }

            cv::Mat raw_image(input_image.height, input_image.width, opencv_input_type, input_image.buffer->data.data());

            // Acquire buffer from pool for processed image
            size_t required_size = tpu_input_width_ * tpu_input_height_ * 3; // Always BGR 3-channel output
            BufferPool<uint8_t>::PooledPtr processed_buffer_data = buffer_pool_->acquire(); // Acquire without size, relies on pool's max size

            if (!processed_buffer_data) {
                APP_LOG_ERROR("ImageProcessor failed to acquire buffer from pool, dropping frame.");
                continue;
            }
            
            // Check if the acquired buffer has enough capacity
            if (processed_buffer_data->data.capacity() < required_size) {
                 {
                     std::stringstream ss;
                     ss << "Acquired buffer from pool is too small for processed image. Capacity: " << processed_buffer_data->data.capacity()
                        << ", Required: " << required_size << ". Dropping frame.";
                     APP_LOG_ERROR(ss.str());
                 }
                 // The buffer will be released when processed_buffer_data goes out of scope
                 continue;
            }
            
            // Create a Mat pointing to the acquired buffer
            cv::Mat bgr_image_out(tpu_input_height_, tpu_input_width_, CV_8UC3, processed_buffer_data->data.data());

            cv::Mat temp_bgr_image; // Temporary Mat for color conversion if needed
            
            // Perform color conversion first if necessary
            if (input_pixel_format_ == libcamera::formats::BGRA8888) {
                cv::cvtColor(raw_image, temp_bgr_image, cv::COLOR_BGRA2BGR);
            } else if (input_pixel_format_ == libcamera::formats::BGR888) {
                temp_bgr_image = raw_image; // No conversion needed, just reference
            } else if (input_pixel_format_ == libcamera::formats::RGBA8888) {
                cv::cvtColor(raw_image, temp_bgr_image, cv::COLOR_RGBA2BGR);
            } else if (input_pixel_format_ == libcamera::formats::RGB888) {
                cv::cvtColor(raw_image, temp_bgr_image, cv::COLOR_RGB2BGR);
            } else if (input_pixel_format_ == libcamera::formats::YUYV) {
                cv::cvtColor(raw_image, temp_bgr_image, cv::COLOR_YUV2BGR_YUYV);
            }
            else {
                {
                std::stringstream ss;
                ss << "ImageProcessor: Unsupported input pixel format " << input_pixel_format_.toString().c_str() << " for color conversion. Dropping frame.";
                APP_LOG_ERROR(ss.str());
            }
                continue;
            }
            
            // Now resize the (potentially color-converted) image into the final output Mat
            cv::resize(temp_bgr_image, bgr_image_out, cv::Size(tpu_input_width_, tpu_input_height_), 0, 0, cv::INTER_LINEAR);
            
            // CRITICAL: Explicitly set the size of the underlying std::vector to match the actual data length
            processed_buffer_data->data.resize(required_size);

            // Create new ImageData for the processed image
            ImageData processed_image;
            processed_image.buffer = processed_buffer_data; // Use the acquired buffer
            processed_image.width = tpu_input_width_;
            processed_image.height = tpu_input_height_;
            processed_image.format = libcamera::formats::BGR888; // Output is always BGR
            // CRITICAL: Set the size of the processed image buffer to match the BGR output size
            processed_image.buffer->size = tpu_input_width_ * tpu_input_height_ * 3; 
            // Preserve original timestamps and IDs
            processed_image.timestamp_epoch_ms = input_image.timestamp_epoch_ms;

            // Push to output queue for InferenceEngine
            if (!output_queue_.push(processed_image)) {
                APP_LOG_WARNING("ImageProcessor output queue is full, dropping frame.");
            }
            auto process_end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(process_end_time - process_start_time).count();
            {
                std::stringstream ss;
                ss << "ImageProcessor processed frame in " << duration_ms << " ms. Input size " << input_image.width << "x" << input_image.height
                   << ", Output size " << tpu_input_width_ << "x" << tpu_input_height_ << ", Format " << input_pixel_format_.toString().c_str();
                APP_LOG_DEBUG(ss.str());
            }

        } else {
            // Small sleep to prevent busy-waiting if queue is empty
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    APP_LOG_INFO("ImageProcessor worker thread finished.");
}