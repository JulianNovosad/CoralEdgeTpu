#include "camera_capture.h"
#include "util_logging.h"
#include "process_supervisor.hpp" // Include the new supervisor

#include <iostream>
#include <vector>
#include <string>
#include <cstring> // For strerror
#include <algorithm> // For std::remove_if
#include <sstream> // For std::ostringstream

CameraCapture::CameraCapture(unsigned int width, unsigned int height, ImageQueue& output_queue, std::chrono::seconds watchdog_timeout)
    : width_(width), height_(height), output_queue_(output_queue) {

    supervisor_ = std::make_unique<ProcessSupervisor<ImageQueue, ImageData>>(
        "Inference rpicam-vid",
        [this]() { return get_command_args(); }, // Command arguments builder
        [this](std::vector<uint8_t>& buffer, size_t bytes_read, ImageQueue& queue) { // Frame parser
            return parse_frame_data(buffer, bytes_read, queue);
        },
        output_queue_,
        watchdog_timeout
    );
}

CameraCapture::~CameraCapture() {
    stop();
}

bool CameraCapture::start() {
    return supervisor_->start();
}

void CameraCapture::stop() {
    supervisor_->stop();
}

bool CameraCapture::is_running() const {
    return supervisor_->is_running();
}

std::vector<std::string> CameraCapture::get_command_args() {
    // Current log output: Model Input Dimensions: 300x300x3
    // Explicitly construct the full JSON string to avoid multi-line issues with the tool
    std::string json_config = "{\"width\":" + std::to_string(width_) + "\"height\":" + std::to_string(height_) + "\"codec\":\"bgr888\"}";
    LOG_JSON("rpicam-vid_inference_config", json_config);

    return {
        "/usr/bin/rpicam-vid",
        "-t", "0",
        "--width", std::to_string(width_),
        "--height", std::to_string(height_),
        "--codec", "bgr888", // Explicitly request BGR888 for inference
        "--nopreview",
        "--output", "-"
    };
}

bool CameraCapture::parse_frame_data(std::vector<uint8_t>& buffer, size_t new_bytes_read, ImageQueue& queue) {
    const size_t expected_frame_size = width_ * height_ * 3;
    bool frame_parsed = false;

    // We don't use new_bytes_read directly here, as `buffer` accumulates data.
    // Instead, we check if `buffer` contains at least one full frame.
    while (buffer.size() >= expected_frame_size) {
        ImageData image_data;
        image_data.width = width_;
        image_data.height = height_;
        image_data.timestamp = std::chrono::high_resolution_clock::now();
        
        // Copy the frame data
        image_data.data.assign(buffer.begin(), buffer.begin() + expected_frame_size);
        
        // Push to queue
        queue.push(std::move(image_data));
        
        // Remove the consumed frame data from the buffer
        buffer.erase(buffer.begin(), buffer.begin() + expected_frame_size);
        frame_parsed = true;
    }
    return frame_parsed;
}
