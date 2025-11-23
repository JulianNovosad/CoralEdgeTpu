/**
 * @file camera_capture.cpp
 * @brief Implements the CameraCapture class for managing a dedicated raw image stream
 *        using rpicam-vid subprocess and robust supervision.
 *
 * This module provides the concrete implementation for launching and supervising
 * an rpicam-vid subprocess, reading raw image data (e.g., YUV420) from its
 * stdout pipe, converting it into ImageData objects, and pushing these objects
 * into multiple thread-safe ImageQueue instances for consumption by various
 * downstream pipeline stages (e.g., inference engine, video overlay processor).
 */

#include "camera_capture.h"
#include "util_logging.h"
#include "process_supervisor.hpp" // Include the new supervisor

#include <iostream>
#include <vector>
#include <string>
#include <cstring> // For strerror
#include <algorithm> // For std::remove_if
#include <sstream> // For std::ostringstream

/**
 * @brief Constructor for CameraCapture.
 *
 * Initializes the camera capture module with specified dimensions and a list
 * of references to output queues where parsed image data will be pushed.
 * It also sets up the ProcessSupervisor for robust subprocess management.
 *
 * @param width The desired width of the raw image stream.
 * @param height The desired height of the raw image stream.
 * @param output_queues A list of references to thread-safe ImageQueue for output.
 * @param watchdog_timeout The duration after which a lack of activity in the
 *                         stream will trigger a subprocess restart (e.g., 5 seconds).
 */
CameraCapture::CameraCapture(unsigned int width, unsigned int height, std::list<std::reference_wrapper<ImageQueue>>& output_queues, std::chrono::seconds watchdog_timeout)
    : width_(width), height_(height), output_queues_(output_queues) {

    // Initialize the ProcessSupervisor for managing the rpicam-vid subprocess.
    // The FrameParserFn lambda now pushes to all registered output queues.
    supervisor_ = std::make_unique<ProcessSupervisor<ImageQueue, ImageData>>(
        "CameraCapture (rpicam-vid)",
        [this]() { return get_command_args(); }, // Command arguments builder
        [this](std::vector<uint8_t>& buffer, size_t bytes_read, ImageQueue& dummy_queue) { // Frame parser
            // The dummy_queue is ignored here; `output_queues_` is used.
            return parse_frame_data(buffer, bytes_read, dummy_queue);
        },
        // Pass the first queue for basic functionality, though it's not directly used by PipeReader's push.
        // ProcessSupervisor needs a single OutputQueueType to match its template.
        // We ensure that output_queues_ is not empty in main.cpp.
        output_queues_.front().get(),
        watchdog_timeout
    );
}

/**
 * @brief Destructor for CameraCapture.
 *
 * Ensures that the capture process and associated threads are gracefully stopped.
 */
CameraCapture::~CameraCapture() {
    stop();
}

/**
 * @brief Starts the camera capture process and its supervisor.
 *
 * Launches the rpicam-vid subprocess and initiates the monitoring and
 * pipe reading threads.
 *
 * @return True if the module started successfully, false otherwise.
 */
bool CameraCapture::start() {
    return supervisor_->start();
}

/**
 * @brief Stops the camera capture process and its supervisor.
 *
 * Sends termination signals to the subprocess and joins all associated threads
 * for a clean shutdown.
 */
void CameraCapture::stop() {
    supervisor_->stop();
}

/**
 * @brief Checks if the camera capture module is currently running.
 *
 * @return True if the module is running, false otherwise.
 */
bool CameraCapture::is_running() const {
    return supervisor_->is_running();
}

/**
 * @brief Generates the command-line arguments for the rpicam-vid subprocess.
 *
 * Constructs a vector of strings representing the arguments to be passed to
 * `/usr/bin/rpicam-vid` for raw image capture (YUV420). Includes
 * parameters for codec, dimensions, and output to stdout. Logs the configuration as JSON.
 *
 * @return A vector of strings containing the rpicam-vid command arguments.
 */
std::vector<std::string> CameraCapture::get_command_args() {
    // Explicitly construct the full JSON string to avoid multi-line issues with the tool
    std::ostringstream json_stream;
    json_stream << "{\"width\":" << width_
                << ",\"height\":" << height_
                << ",\"codec\":\"yuv420\"}";
    LOG_JSON("rpicam-vid_capture_config", json_stream.str());

    return {
        "/usr/bin/rpicam-vid",
        "-t", "0",
        "--width", std::to_string(width_),
        "--height", std::to_string(height_),
        "--codec", "yuv420", // Request YUV420 format
        "--nopreview",
        "--output", "-"
    };
}

/**
 * @brief Parses raw byte data from the pipe into complete ImageData objects.
 *
 * This function is passed as a callback to the ProcessSupervisor. It accumulates
 * bytes in the buffer and, once a full frame's worth of data is received (for
 * YUV420), it constructs an ImageData object and pushes it to all registered
 * output queues. Consumed bytes are removed from the buffer.
 *
 * @param buffer A reference to the buffer accumulating raw byte data.
 * @param bytes_read The number of new bytes read in the current read operation (not directly used for parsing,
 *                   the entire buffer content is processed).
 * @param dummy_queue A dummy ImageQueue argument to match the ProcessSupervisor's FrameParserFn signature.
 *                    The actual output queues are accessed via `output_queues_`.
 * @return True if at least one complete frame was parsed and pushed, false otherwise.
 */
bool CameraCapture::parse_frame_data(std::vector<uint8_t>& buffer, size_t new_bytes_read, ImageQueue& dummy_queue) {
    // For YUV420, the frame size is (width * height * 3) / 2
    // Y plane: width * height
    // U plane: (width/2) * (height/2)
    // V plane: (width/2) * (height/2)
    const size_t expected_frame_size = (width_ * height_ * 3) / 2;
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
        
        // Push to all registered queues
        for (auto& queue_ref : output_queues_) {
            queue_ref.get().push(image_data); // Push a copy to each queue
        }
        
        // Remove the consumed frame data from the buffer
        buffer.erase(buffer.begin(), buffer.begin() + expected_frame_size);
        frame_parsed = true;
    }
    return frame_parsed;
}
