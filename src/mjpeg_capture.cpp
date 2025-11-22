#include "mjpeg_capture.h"
#include "util_logging.h"
#include "process_supervisor.hpp" // Include the new supervisor

#include <iostream>
#include <vector>
#include <string>
#include <cstring> // For strerror
#include <algorithm> // For std::remove_if
#include <sstream> // For std::ostringstream

MjpegCapture::MjpegCapture(unsigned int width, unsigned int height, unsigned int fps, MjpegQueue& output_queue, std::chrono::seconds watchdog_timeout)
    : width_(width), height_(height), fps_(fps), output_queue_(output_queue) {

    supervisor_ = std::make_unique<ProcessSupervisor<MjpegQueue, ImageFrame>>(
        "MJPEG rpicam-vid",
        [this]() { return get_command_args(); }, // Command arguments builder
        [this](std::vector<uint8_t>& buffer, size_t new_bytes_read, MjpegQueue& queue) { // Frame parser
            return parse_frame_data(buffer, new_bytes_read, queue);
        },
        output_queue_,
        watchdog_timeout
    );
}

MjpegCapture::~MjpegCapture() {
    stop();
}

bool MjpegCapture::start() {
    return supervisor_->start();
}

void MjpegCapture::stop() {
    supervisor_->stop();
}

bool MjpegCapture::is_running() const {
    return supervisor_->is_running();
}

std::vector<std::string> MjpegCapture::get_command_args() {
    // Explicitly construct the full JSON string to avoid multi-line issues with the tool
    std::string json_config = "{\"width\":" + std::to_string(width_) + ",\"height\":" + std::to_string(height_) + ",\"framerate\":" + std::to_string(fps_) + ",\"codec\":\"mjpeg\"}";
    LOG_JSON("rpicam-vid_mjpeg_config", json_config);
    return {
        "/usr/bin/rpicam-vid",
        "--codec", "mjpeg",
        "--width", std::to_string(width_),
        "--height", std::to_string(height_),
        "--framerate", std::to_string(fps_),
        "--timeout", "0", // No timeout, run indefinitely
        "--flush", // Ensure data is flushed
        "--inline", // Embed image data directly into the stream
        "--output", "-" // Output to stdout
    };
}

// MJPEG frame parser implementation
bool MjpegCapture::parse_frame_data(std::vector<uint8_t>& buffer, size_t new_bytes_read, MjpegQueue& queue) {
    bool frame_parsed = false;
    // Look for JPEG Start of Image (SOI) - FF D8
    // Look for JPEG End of Image (EOI) - FF D9

    while (true) {
        const std::vector<uint8_t> soi = {0xFF, 0xD8};
        auto soi_it = std::search(buffer.begin(), buffer.end(), soi.begin(), soi.end());
        if (soi_it == buffer.end()) {
            // No SOI found, or partial SOI at the end of the buffer
            // Keep the buffer content, wait for more data
            break;
        }

        const std::vector<uint8_t> eoi = {0xFF, 0xD9};
        auto eoi_it = std::search(soi_it + 2, buffer.end(), eoi.begin(), eoi.end());
        if (eoi_it == buffer.end()) {
            // SOI found, but no EOI yet. Keep the buffer content, wait for more data.
            // Move SOI to the beginning of the buffer to avoid re-searching already processed data
            buffer.erase(buffer.begin(), soi_it);
            break;
        }

        // Found a complete JPEG frame
        ImageFrame frame;
        frame.width = width_; // These are fixed for the stream
        frame.height = height_;
        // Copy the JPEG data including SOI and EOI
        frame.jpeg_data.assign(soi_it, eoi_it + 2); // +2 to include FF D9

        // Push to queue
        queue.push_mjpeg(std::move(frame)); // Use push_mjpeg to keep only the latest

        // Remove the consumed JPEG frame data from the buffer
        buffer.erase(buffer.begin(), eoi_it + 2);
        frame_parsed = true;
    }
    return frame_parsed;
}
