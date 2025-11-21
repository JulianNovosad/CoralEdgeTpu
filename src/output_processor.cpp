#include "output_processor.h"
#include "camera_capture.h" // For ImageData
#include <iostream>
#include <vector>
#include <algorithm> // For std::min, std::max

// Function to draw a rectangle on a YUV420 image. This is a very basic implementation
// that only draws on the Y (luminance) plane.
void draw_rectangle_yuv420(std::vector<uint8_t>& yuv_data, size_t width, size_t height,
                           int x1, int y1, int x2, int y2, uint8_t color_val_y) {
    // Clamp coordinates to image boundaries
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min((int)width - 1, x2);
    y2 = std::min((int)height - 1, y2);

    // Draw horizontal lines (top and bottom)
    for (int x = x1; x <= x2; ++x) {
        if (y1 >= 0 && y1 < height) yuv_data[y1 * width + x] = color_val_y;
        if (y2 >= 0 && y2 < height) yuv_data[y2 * width + x] = color_val_y;
    }

    // Draw vertical lines (left and right)
    for (int y = y1; y <= y2; ++y) {
        if (x1 >= 0 && x1 < width) yuv_data[y * width + x1] = color_val_y;
        if (x2 >= 0 && x2 < width) yuv_data[y * width + x2] = color_val_y;
    }
}

OutputProcessor::OutputProcessor(DetectionQueue& input_detection_queue, MjpegQueue& output_mjpeg_queue, const std::vector<std::string>& labels)
    : input_detection_queue_(input_detection_queue), output_mjpeg_queue_(output_mjpeg_queue), labels_(labels) {}

OutputProcessor::~OutputProcessor() {
    stop();
}

bool OutputProcessor::start() {
    if (running_) {
        std::cerr << "OutputProcessor is already running." << std::endl;
        return false;
    }
    running_ = true;
    input_detection_queue_.set_running(true); // Ensure detection queue is active
    process_thread_ = std::thread(&OutputProcessor::process_thread_func, this);
    std::cout << "OutputProcessor started." << std::endl;
    return true;
}

void OutputProcessor::stop() {
    if (!running_) {
        return;
    }
    running_ = false;
    input_detection_queue_.set_running(false); // Signal detection queue to stop
    if (process_thread_.joinable()) {
        process_thread_.join();
    }
    std::cout << "OutputProcessor stopped." << std::endl;
}

void OutputProcessor::process_thread_func() {
    std::vector<DetectionResult> detections;
    ImageData original_image; // This should ideally come from a queue from camera, but for now we assume we have it in main.cpp

    while (running_) {
        if (input_detection_queue_.pop(detections)) { // Blocking call
            // For now, we'll create a dummy image based on expected dimensions
            // In a real app, image data would be queued.
            ImageData dummy_image;
            dummy_image.width = 640; // Default width
            dummy_image.height = 480; // Default height
            dummy_image.data.resize(dummy_image.width * dummy_image.height * 3 / 2); // YUV420 size

            std::vector<uint8_t> jpeg_data = draw_boxes_and_encode_jpeg(dummy_image, detections);
            if (!jpeg_data.empty()) {
                ImageFrame frame;
                frame.jpeg_data = std::move(jpeg_data);
                frame.width = dummy_image.width;
                frame.height = dummy_image.height;
                output_mjpeg_queue_.push(std::move(frame));
            }
        }
    }
}

std::vector<uint8_t> OutputProcessor::draw_boxes_and_encode_jpeg(const ImageData& original_image, const std::vector<DetectionResult>& detections) {
    std::vector<uint8_t> image_data_yuv = original_image.data; // Make a mutable copy

    for (const auto& det : detections) {
        // Convert normalized coordinates to pixel coordinates
        int x1 = static_cast<int>(det.xmin);
        int y1 = static_cast<int>(det.ymin);
        int x2 = static_cast<int>(det.xmax);
        int y2 = static_cast<int>(det.ymax);

        // Draw a green box (arbitrary Y value for green)
        draw_rectangle_yuv420(image_data_yuv, original_image.width, original_image.height,
                              x1, y1, x2, y2, 100); // Y=100 for a medium gray/green

        // TODO: Draw label text (e.g., labels_[det.class_id]) on the image
        // For now, we just print the label to console.
        if (det.class_id < labels_.size()) {
            std::cout << "Detected: " << labels_[det.class_id] << std::endl;
        }
    }

    // Compress the modified YUV image to JPEG
    return jpeg_compressor_.compress_image(
        image_data_yuv.data(),
        original_image.width,
        original_image.height,
        75, // JPEG quality
        JCS_YCbCr // Assuming YUV420 input maps to JCS_YCbCr
    );
}