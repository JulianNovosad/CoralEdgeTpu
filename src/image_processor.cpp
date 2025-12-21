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
        std::string log_message = ss.str(); // Make explicit string
        APP_LOG_ERROR(log_message);
        running_ = false; // Signal that the thread is stopping
        return; // Exit the worker thread
    }

    ImageData input_image;
    while (running_.load()) {
        // Wait for an image to pop from the queue, with a small timeout to allow stopping
        if (input_queue_.pop(input_image)) {
            auto process_start_time = std::chrono::high_resolution_clock::now();

            // Ensure the input image buffer is valid
            if (!input_image.buffer || input_image.buffer->data.empty() || input_image.width == 0 || input_image.height == 0) {
                APP_LOG_ERROR("ImageProcessor received invalid ImageData (buffer is null, empty or dimensions are zero).");
                continue;
            }

        // 2. Process image (color conversion and/or resizing)
        // Ensure that the input_pixel_format_ (which comes from config.json) is used here.
        // We expect RGB888 from CameraCapture and we are configured for RGB888.
        cv::Mat input_frame_mat;
        if (input_image.format == libcamera::formats::RGB888) {
            input_frame_mat = cv::Mat(input_image.height, input_image.width, CV_8UC3, input_image.buffer->data.data());
        } else {
            APP_LOG_ERROR("ImageProcessor: Unexpected input format (FOURCC: " + std::to_string(input_image.format.fourcc()) + "). Expected RGB888. Skipping frame.");
            input_image.buffer.reset(); // Return buffer to pool
            continue;
        }

        cv::Mat processed_mat;
        if (input_image.width == (unsigned int)tpu_input_width_ && input_image.height == (unsigned int)tpu_input_height_) {
            // No resizing or color conversion needed
            // For zero-copy optimization, we can pass the buffer directly in some cases
            // Since dimensions match, we can avoid the copy and just pass through the buffer
            // But we still need to create a new ImageData object with the same buffer
            processed_mat = input_frame_mat; // Just reference the same data
        } else {
            // Resize if dimensions differ
            APP_LOG_WARNING("ImageProcessor: Resizing RGB888 frame from " + std::to_string(input_image.width) + "x" + std::to_string(input_image.height) +
                            " to " + std::to_string(tpu_input_width_) + "x" + std::to_string(tpu_input_height_) + ".");
            cv::resize(input_frame_mat, processed_mat, cv::Size(tpu_input_width_, tpu_input_height_), 0, 0, cv::INTER_LINEAR);
        }

        // 3. Acquire a buffer from the pool for the processed image.
        std::shared_ptr<PooledBuffer<uint8_t>> processed_buffer_data;
        if (input_image.width == (unsigned int)tpu_input_width_ && input_image.height == (unsigned int)tpu_input_height_) {
            // Dimensions match, we can reuse the input buffer for zero-copy operation
            processed_buffer_data = input_image.buffer;
        } else {
            // Dimensions don't match, we need a new buffer
            processed_buffer_data = buffer_pool_->acquire();
            if (!processed_buffer_data) {
                APP_LOG_WARNING("ImageProcessor: Failed to acquire buffer for processed image. Dropping frame.");
                input_image.buffer.reset(); // Return input buffer to pool
                continue;
            }

            // Ensure processed_buffer_data->data has enough capacity
            size_t required_size = processed_mat.total() * processed_mat.elemSize();
            if (required_size > processed_buffer_data->data.capacity()) {
                APP_LOG_ERROR("ImageProcessor: Processed image size (" + std::to_string(required_size) +
                              ") exceeds buffer pool capacity (" + std::to_string(processed_buffer_data->data.capacity()) + "). Dropping frame.");
                input_image.buffer.reset(); // Return input buffer to pool
                processed_buffer_data.reset(); // Return acquired buffer to pool
                continue;
            }
            
            // Copy processed image data to the pooled buffer
            std::memcpy(processed_buffer_data->data.data(), processed_mat.data, required_size);
            processed_buffer_data->size = required_size;
            // Explicitly resize the underlying std::vector to reflect the actual data size
            processed_buffer_data->data.resize(required_size);
        }


        // 4. Create new ImageData object and push to output queue
        ImageData output_image_data(input_image.timestamp_epoch_ms, input_image.frame_id);
        output_image_data.width = tpu_input_width_;
        output_image_data.height = tpu_input_height_;
        output_image_data.format = libcamera::formats::RGB888; // Output is always RGB888 for TPU
        output_image_data.buffer = processed_buffer_data;
        // Pass through zero-copy information if available
        output_image_data.fd = input_image.fd;
        output_image_data.offset = input_image.offset;
        output_image_data.length = input_image.length;

        if (!output_queue_.push(std::move(output_image_data))) {
            APP_LOG_WARNING("ImageProcessor output queue is full. Dropping processed frame.");
            // output_image_data.buffer will be destructed here, returning its buffer to the pool.
        }

        auto process_end_time = std::chrono::high_resolution_clock::now();
        [[maybe_unused]] long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(process_end_time - process_start_time).count();
        APP_LOG_DEBUG("ImageProcessor processed frame in " + std::to_string(duration_ms) + " ms. Input size " +
                      std::to_string(input_image.width) + "x" + std::to_string(input_image.height) + ", Output size " +
                      std::to_string(tpu_input_width_) + "x" + std::to_string(tpu_input_height_) + ", Format RGB888");

        } else {
            // Small sleep to prevent busy-waiting if queue is empty
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    APP_LOG_INFO("ImageProcessor worker thread finished.");
}