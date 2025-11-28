#include "video_overlay_processor.h"
#include "util_logging.h"
#include <iostream>
#include <algorithm> // For std::max, std::min
#include <cmath>     // For std::round

// Define colors for bounding boxes
namespace {
    struct Color {
        uint8_t r, g, b;
    };
    const Color COLORS[] = {
        {0, 255, 0},   // Green
        {0, 0, 255},   // Blue
        {255, 0, 0},   // Red
        {255, 255, 0}, // Yellow
        {0, 255, 255}, // Cyan
        {255, 0, 255}, // Magenta
        {255, 165, 0}  // Orange
    };
    const int NUM_COLORS = sizeof(COLORS) / sizeof(COLORS[0]);

    // Function to get a color based on class_id
    Color get_color_for_class(int class_id) {
        return COLORS[class_id % NUM_COLORS];
    }
}

/**
 * @brief Constructor for VideoOverlayProcessor.
 *
 * Initializes the processor with references to input queues for raw image data
 * and detection results, and an output queue for overlaid MJPEG frames.
 *
 * @param inference_image_queue Reference to the ImageQueue providing raw image data.
 * @param detection_results_queue Reference to the UdpQueue providing detection results.
 * @param overlaid_mjpeg_output_queue Reference to the MjpegQueue for overlaid MJPEG frames.
 * @param labels Reference to a vector of strings for class labels.
 */
VideoOverlayProcessor::VideoOverlayProcessor(
    ImageQueue& inference_image_queue,
    UdpQueue& detection_results_queue,
    MjpegQueue& overlaid_mjpeg_output_queue,
    const std::vector<std::string>& labels)
    : inference_image_queue_(inference_image_queue),
      detection_results_queue_(detection_results_queue),
      overlaid_mjpeg_output_queue_(overlaid_mjpeg_output_queue),
      labels_(labels) {
    // JpegCompressGuard is default constructed, ready for use.
}

/**
 * @brief Destructor for VideoOverlayProcessor.
 *
 * Ensures the processing thread is gracefully stopped.
 */
VideoOverlayProcessor::~VideoOverlayProcessor() {
    stop();
}

/**
 * @brief Starts the video overlay processing thread.
 *
 * Launches the dedicated `processing_thread_func` to continuously
 * retrieve image data and detection results, perform overlay, and push
 * the resulting MJPEG frame to the output queue.
 *
 * @return True if the thread started successfully, false otherwise.
 */
bool VideoOverlayProcessor::start() {
    if (running_) {
        LOG_ERROR("VideoOverlayProcessor is already running.");
        return false;
    }
    running_ = true;
    overlaid_mjpeg_output_queue_.set_running(true); // Ensure output queue is marked as running
    processing_thread_ = std::thread(&VideoOverlayProcessor::processing_thread_func, this);
    LOG_INFO("VideoOverlayProcessor started.");
    return true;
}

/**
 * @brief Stops the video overlay processing thread.
 *
 * Sets the `running_` flag to false, signaling the processing thread
 * to terminate, and then waits for the thread to join.
 */
void VideoOverlayProcessor::stop() {
    if (!running_.exchange(false)) { // Atomically set to false and check previous value.
        return; // Already stopped.
    }
    LOG_INFO("Stopping VideoOverlayProcessor...");
    // Signal all associated queues to stop.
    inference_image_queue_.set_running(false);
    detection_results_queue_.set_running(false);
    overlaid_mjpeg_output_queue_.set_running(false);
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    LOG_INFO("VideoOverlayProcessor stopped.");
}

/**
 * @brief The main loop for the video overlay processing thread.
 *
 * This function continuously retrieves image data from `inference_image_queue_`
 * and detection results from `detection_results_queue_`. It then calls
 * `draw_overlays` to apply bounding boxes and labels, compresses the
 * modified image to MJPEG, and pushes it to `overlaid_mjpeg_output_queue_`.
 */
void VideoOverlayProcessor::processing_thread_func() {
    ImageData current_image;
    std::vector<DetectionResult> current_detections; 

        while (running_) {
            LOG_INFO("VideoOverlayProcessor: processing_thread_func loop started. running_ = " + std::to_string(running_));
            LOG_INFO("VideoOverlayProcessor: Attempting to pop image from queue.");
            if (!inference_image_queue_.pop(current_image)) {
                LOG_INFO("VideoOverlayProcessor: Pop image returned false. running_ = " + std::to_string(running_));
                break; // Exit loop if queue is empty and stop requested.
            }
            LOG_INFO("VideoOverlayProcessor: Successfully popped image. running_ = " + std::to_string(running_));
    
            // Try to get the latest detections.
            if (!detection_results_queue_.pop(current_detections)) {
                LOG_INFO("VideoOverlayProcessor: Pop detections returned false. running_ = " + std::to_string(running_));
                if (!running_) { // Check after pop if shutdown was requested during wait
                    break; // Exit if pop failed due to shutdown
                }
                current_detections.clear(); // Ensure it's empty if pop failed but still running
            }
            LOG_INFO("VideoOverlayProcessor: Popped detections (if any). running_ = " + std::to_string(running_));        ImageData image_to_overlay = current_image;

        // Draw bounding boxes and labels onto the image data if detections are present.
        if (!current_detections.empty()) {
            LOG_INFO("VideoOverlayProcessor: Drawing overlays.");
            draw_overlays(image_to_overlay, current_detections);
            LOG_INFO("VideoOverlayProcessor: Finished drawing overlays.");
        }

        // Compress the overlaid image data into MJPEG format.
        try {
            LOG_INFO("VideoOverlayProcessor: Starting JPEG compression.");
            // The input color space for JpegCompressGuard::compress_image
            // is now assumed to be RGB888 directly. JCS_RGB usually handles this correctly.
            std::vector<uint8_t> mjpeg_data = jpeg_compressor_.compress_image(
                image_to_overlay.data.data(),
                image_to_overlay.width,
                image_to_overlay.height,
                80, // JPEG quality (0-100), 80 is a good balance.
                JCS_RGB // Assuming RGB input for libjpeg.
            );
            LOG_INFO("VideoOverlayProcessor: Finished JPEG compression.");

            if (mjpeg_data.empty()) {
                LOG_ERROR("VideoOverlayProcessor: JPEG compression resulted in empty data. Skipping frame.");
                continue; // Skip pushing empty frame
            }
            
            // Push the compressed MJPEG frame to the output queue.
            ImageFrame overlaid_frame;
            overlaid_frame.width = image_to_overlay.width;
            overlaid_frame.height = image_to_overlay.height;
            overlaid_frame.jpeg_data = std::move(mjpeg_data);
            overlaid_mjpeg_output_queue_.push(std::move(overlaid_frame));
            LOG_INFO("VideoOverlayProcessor: Pushed overlaid frame to HTTP queue."); // Added log
        } catch (const std::runtime_error& e) {
            LOG_ERROR("VideoOverlayProcessor: JPEG compression failed: " + std::string(e.what()));
        }
    }
    LOG_INFO("VideoOverlayProcessor thread exited.");
}

/**
 * @brief Overlays bounding boxes and labels onto the raw image data.
 *
 * This method directly manipulates the pixel data in `image_data.data`
 * to draw rectangles and simple colored squares as placeholders for labels.
 * It assumes the `image_data.data` is in BGR format (Blue, Green, Red byte order).
 *
 * @param image_data The raw image data (BGR format) to draw on. Modified in place.
 * @param detections The detection results to overlay, containing bounding box coordinates and class IDs.
 */
void VideoOverlayProcessor::draw_overlays(ImageData& image_data, const std::vector<DetectionResult>& detections) {
    LOG_INFO("VideoOverlayProcessor: Entering draw_overlays.");
    const int channels = 3; // Assuming BGR format (3 channels).
    const int width = image_data.width;
    const int height = image_data.height;
    uint8_t* pixels = image_data.data.data(); // Get raw pointer to pixel data.

    for (const auto& det : detections) {
        // Get a distinct color for the bounding box based on class ID.
        Color color = get_color_for_class(det.class_id);

        // Scale bounding box coordinates from normalized [0,1] to pixel values.
        // Bounding box order: [ymin, xmin, ymax, xmax] (from TFLite model output usually).
        int xmin = static_cast<int>(det.xmin);
        int ymin = static_cast<int>(det.ymin);
        int xmax = static_cast<int>(det.xmax);
        int ymax = static_cast<int>(det.ymax);

        // Clamp coordinates to image boundaries to prevent drawing out of bounds.
        xmin = std::max(0, xmin);
        ymin = std::max(0, ymin);
        xmax = std::min(width - 1, xmax);
        ymax = std::min(height - 1, ymax);

        const int thickness = 2; // Thickness of the bounding box lines.

        // Draw rectangle borders (iterating over thickness for thicker lines).
        for (int t = 0; t < thickness; ++t) {
            // Top border
            for (int x = xmin; x <= xmax; ++x) {
                if (ymin + t < height) {
                    size_t pixel_idx = ((ymin + t) * width + x) * channels;
                    pixels[pixel_idx + 0] = color.b; // Blue
                    pixels[pixel_idx + 1] = color.g; // Green
                    pixels[pixel_idx + 2] = color.r; // Red
                }
            }
            // Bottom border
            for (int x = xmin; x <= xmax; ++x) {
                if (ymax - t >= 0) {
                    size_t pixel_idx = ((ymax - t) * width + x) * channels;
                    pixels[pixel_idx + 0] = color.b; // Blue
                    pixels[pixel_idx + 1] = color.g; // Green
                    pixels[pixel_idx + 2] = color.r; // Red
                }
            }
            // Left border
            for (int y = ymin; y <= ymax; ++y) {
                if (xmin + t < width) {
                    size_t pixel_idx = (y * width + (xmin + t)) * channels;
                    pixels[pixel_idx + 0] = color.b; // Blue
                    pixels[pixel_idx + 1] = color.g; // Green
                    pixels[pixel_idx + 2] = color.r; // Red
                }
            }
            // Right border
            for (int y = ymin; y <= ymax; ++y) {
                if (xmax - t >= 0) {
                    size_t pixel_idx = (y * width + (xmax - t)) * channels;
                    pixels[pixel_idx + 0] = color.b; // Blue
                    pixels[pixel_idx + 1] = color.g; // Green
                    pixels[pixel_idx + 2] = color.r; // Red
                }
            }
        }

        // Add label text (simple colored square as text rendering is complex without a library).
        // A small colored box is drawn near the top-left corner of the bounding box
        // to indicate the class ID's color.
        if (!labels_.empty() && det.class_id >= 0 && det.class_id < labels_.size()) {
            // std::string label_text = labels_[det.class_id]; // Label text not drawn.
            int label_box_size = 10; // Size of the colored square for the label.
            for (int y = ymin; y < ymin + label_box_size && y < height; ++y) {
                for (int x = xmin; x < xmin + label_box_size && x < width; ++x) {
                    size_t pixel_idx = (y * width + x) * channels;
                    pixels[pixel_idx + 0] = color.b; // Blue
                    pixels[pixel_idx + 1] = color.g; // Green
                    pixels[pixel_idx + 2] = color.r; // Red
                }
            }
        }
    }
}

// BGR to YCbCr conversion - not currently used directly, as libjpeg-turbo JCS_RGB often works with BGR input directly.
// This function is kept for reference or if explicit color space conversion becomes necessary.
/**
 * @brief Converts RGB pixel components to YCbCr (YUV) color space.
 *
 * This function implements a standard approximation for converting Red, Green,
 * Blue color values to Luma (Y) and Chrominance (Cb, Cr) components,
 * clamping the output values to the 0-255 range.
 *
 * @param r Red component (0-255).
 * @param g Green component (0-255).
 * @param b Blue component (0-255).
 * @param y Output Luma component.
 * @param cb Output Blue-difference chrominance component.
 * @param cr Output Red-difference chrominance component.
 */
void VideoOverlayProcessor::rgb_to_ycbcr(uint8_t r, uint8_t g, uint8_t b, uint8_t& y, uint8_t& cb, uint8_t& cr) {
    // Standard BT.601 conversion (approximated for integer math)
    y = static_cast<uint8_t>(std::round( 0.299 * r + 0.587 * g + 0.114 * b));
    cb = static_cast<uint8_t>(std::round(-0.169 * r - 0.331 * g + 0.500 * b + 128));
    cr = static_cast<uint8_t>(std::round( 0.500 * r - 0.419 * g - 0.081 * b + 128));

    // Clamp values to 0-255 (std::min/max with uint8_t can be tricky, cast to int first)
    y = std::min((int)255, std::max((int)0, (int)y));
    cb = std::min((int)255, std::max((int)0, (int)cb));
    cr = std::min((int)255, std::max((int)0, (int)cr));
}

void VideoOverlayProcessor::get_state() const {
    LOG_INFO("--- VideoOverlayProcessor State ---");
    LOG_INFO("  Running: " + std::to_string(running_));
    LOG_INFO("  Number of Labels: " + std::to_string(labels_.size()));
    LOG_INFO("---------------------------------");
}