/**
 * @file main.cpp
 * @brief Main entry point for the CoralEdgeTpu Detector application.
 *
 * This application initializes and manages various modules for a real-time object
 * detection pipeline on the Raspberry Pi. It handles camera capture, runs
 * TensorFlow Lite inference with Edge TPU acceleration, sends detection results
 * and raw video via UDP to a mobile app, and provides an HTTP stream of overlaid
 * video for PC viewing. It includes robust process supervision and graceful
 * shutdown handling.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <csignal>
#include <atomic>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <chrono> // Required for std::chrono::seconds
#include <list>   // Required for std::list of ImageQueue references

#include "pipeline_structs.h"
#include "camera_capture.h"
#include "mjpeg_capture.h"
#include "inference.h"
#include "mjpeg_server.h"
#include "udp_sender.h"
#include "util_logging.h"
#include "udp_video_sender.h"         // New: For raw MJPEG video over UDP
#include "video_overlay_processor.h"  // New: For overlaying bounding boxes

/// Global atomic flag to signal application shutdown.
std::atomic<bool> shutdown_requested(false);

/**
 * @brief Signal handler for graceful application shutdown.
 *
 * Catches SIGINT (Ctrl+C) and SIGTERM signals, sets the shutdown_requested flag,
 * and logs the initiation of the cleanup process.
 *
 * @param signal The signal number received.
 */
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        if (!shutdown_requested.exchange(true)) {
             LOG_INFO("Shutdown requested, initiating cleanup...");
        }
    }
}

/**
 * @brief Loads labels from a specified text file.
 *
 * Reads each line from the file into a vector of strings, where each string
 * represents a class label. Logs an error if the file cannot be opened.
 *
 * @param path The filesystem path to the labels file.
 * @return A vector of strings containing the loaded labels. Returns an empty
 *         vector if the file is not found or is empty.
 */
std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open labels file: " + path);
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

/**
 * @brief Main function of the CoralEdgeTpu Detector.
 *
 * Initializes and manages a multi-stream object detection pipeline:
 * - Inference stream for Edge TPU.
 * - Raw MJPEG video stream via UDP to a mobile app.
 * - Bounding box JSON data stream via UDP to a mobile app.
 * - Overlaid video stream (with bounding boxes) via HTTP for PC viewing.
 * Handles robust subprocess supervision and graceful shutdown.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @return 0 on successful execution, 1 on initialization or startup failure.
 */
int main(int argc, char** argv) {
    // Register signal handlers for graceful shutdown on SIGINT or SIGTERM
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler); // Ensure SIGTERM is handled

    // Initialize the logger singleton
    Logger& logger = Logger::getInstance();
    LOG_INFO("CoralEdgeTpu Detector Starting...");

    // --- Application Configuration ---
    const std::string model_path = "/home/pi/CoralEdgeTpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
    const std::string labels_path = "/home/pi/CoralEdgeTpu/coco_labels.txt";
    
    // Camera Stream Configuration (shared for inference and overlay)
    const unsigned int mjpeg_stream_width = 640;
    const unsigned int mjpeg_stream_height = 480;
    const unsigned int mjpeg_stream_fps = 15;

    // UDP and HTTP Ports for various streams (Standardized)
    const int UDP_RAW_VIDEO_PORT = 50000;
    const int UDP_BOUNDING_BOX_PORT = 50010;
    const int HTTP_OVERLAID_VIDEO_PORT = 8081;

    // Mobile App IP address (PLACEHOLDER - **MUST BE UPDATED BY USER**)
    // This IP address should be the actual IP of your mobile device on the network.
    // If set to 127.0.1.1 or 127.0.0.1, the UDP streams will only be sent to the Pi itself.
    const std::string MOBILE_APP_IP = "192.168.178.XXX"; // Example: "192.168.1.10"

    // Watchdog timeout for camera streams (5 seconds of inactivity triggers a restart)
    const std::chrono::seconds camera_watchdog_timeout = std::chrono::seconds(5);

    // --- Thread-Safe Queues for Inter-Module Communication ---
    // Queue for raw image data from camera capture (single instance)
    ImageQueue camera_output_queue; 
    // Queue for raw MJPEG frames from a dedicated MJPEG camera capture to the UDP video sender
    MjpegQueue mjpeg_for_udp_queue;
    // Queue for detection results from inference engine (feeds UDP sender and overlay processor)
    UdpQueue inference_results_queue;
    // Queue for overlaid MJPEG frames from the video overlay processor to the HTTP server
    MjpegQueue overlaid_mjpeg_to_http_queue;

    // --- Module Initialization ---
    // Verify existence of model and labels files
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("Model file not found: " + model_path);
        return 1;
    }
    std::vector<std::string> labels = load_labels(labels_path);
    if (labels.empty()) {
        LOG_ERROR("Labels file not found or is empty: " + labels_path);
        return 1;
    }

    // Initialize the Inference Engine
    std::unique_ptr<InferenceEngine> inference_engine;
    try {
        inference_engine = std::make_unique<InferenceEngine>(model_path, camera_output_queue, inference_results_queue, 2);
    } catch (const std::runtime_error& e) {
        LOG_ERROR("Failed to initialize Inference Engine: " + std::string(e.what()));
        return 1;
    }
    
    // Get inference input dimensions from the loaded model (used for the single camera capture)
    const unsigned int inference_width = inference_engine->get_input_width();
    const unsigned int inference_height = inference_engine->get_input_height();

    // Log the current application configuration
    LOG_INFO("--- Configuration ---");
    LOG_INFO("  Inference Input: " + std::to_string(inference_width) + "x" + std.to_string(inference_height));
    LOG_INFO("  Raw MJPEG for UDP: " + std::to_string(mjpeg_stream_width) + "x" + std::to_string(mjpeg_stream_height) + "@" + std::to_string(mjpeg_stream_fps) + "fps on UDP Port " + std::to_string(UDP_RAW_VIDEO_PORT));
    LOG_INFO("  Bounding Box UDP: on UDP Port " + std::to_string(UDP_BOUNDING_BOX_PORT));
    LOG_INFO("  Overlaid Video HTTP: on HTTP Port " + std::to_string(HTTP_OVERLAID_VIDEO_PORT));
    LOG_INFO("  Mobile App Target IP: " + MOBILE_APP_IP);
    LOG_INFO("---------------------");

    // --- Initialize Camera Capture Modules ---
    // One CameraCapture instance feeds raw frames for both inference and overlay processing.
    // The frames are also pushed to the raw MJPEG UDP sender.
    std::list<std::reference_wrapper<ImageQueue>> primary_camera_output_queues;
    primary_camera_output_queues.push_back(std::ref(camera_output_queue)); // For InferenceEngine
    // primary_camera_output_queues.push_back(std::ref(camera_for_overlay_queue)); // For VideoOverlayProcessor, will add below

    // Single CameraCapture instance for raw image data
    CameraCapture primary_camera(inference_width, inference_height, primary_camera_output_queues, camera_watchdog_timeout);

    // Dedicated MJPEG stream for UDP raw video sender (outputs to mjpeg_for_udp_queue)
    MjpegCapture raw_mjpeg_for_udp_capture(mjpeg_stream_width, mjpeg_stream_height, mjpeg_stream_fps, mjpeg_for_udp_queue, camera_watchdog_timeout);


    // --- Initialize Processing & Sending Modules ---
    // 1. UDP Raw Video Sender (Stream 1)
    UdpVideoSender udp_raw_video_sender(MOBILE_APP_IP, UDP_RAW_VIDEO_PORT, mjpeg_for_udp_queue);
    // 2. Video Overlay Processor (consumes raw images & detections, produces overlaid MJPEG)
    VideoOverlayProcessor video_overlay_processor(camera_output_queue, inference_results_queue, overlaid_mjpeg_to_http_queue, labels);
    // 3. UDP Bounding Box Sender (Stream 2)
    UdpSender udp_bounding_box_sender(MOBILE_APP_IP, UDP_BOUNDING_BOX_PORT, inference_results_queue);
    // 4. HTTP Server for Overlaid Video (Stream 3)
    MjpegServer overlaid_mjpeg_server(HTTP_OVERLAID_VIDEO_PORT, overlaid_mjpeg_to_http_queue);


    // --- Start all modules ---
    if (!primary_camera.start() ||
        !raw_mjpeg_for_udp_capture.start() ||
        !inference_engine->start() ||
        !udp_raw_video_sender.start() ||
        !video_overlay_processor.start() ||
        !udp_bounding_box_sender.start() ||
        !overlaid_mjpeg_server.start()) {
        
        LOG_ERROR("Failed to start one or more modules. Shutting down.");
        // Stop all modules in reverse order to ensure proper cleanup, even if some failed to start.
        overlaid_mjpeg_server.stop();
        udp_bounding_box_sender.stop();
        video_overlay_processor.stop();
        udp_raw_video_sender.stop();
        inference_engine->stop();
        raw_mjpeg_for_udp_capture.stop();
        primary_camera.stop();
        logger.stop_writer_thread(); // Stop logging thread last
        return 1;
    }

    LOG_INFO("Application started successfully. Waiting for shutdown signal (Ctrl+C).");
    // Main application loop: waits for the shutdown signal
    while (!shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // --- Shutdown Modules ---
    LOG_INFO("Shutting down application modules...");
    // Stop all modules gracefully in the defined order (reverse of startup or logical shutdown order).
    overlaid_mjpeg_server.stop();
    udp_bounding_box_sender.stop();
    video_overlay_processor.stop();
    udp_raw_video_sender.stop();
    inference_engine->stop();
    raw_mjpeg_for_udp_capture.stop();
    primary_camera.stop();
    logger.stop_writer_thread(); // Stop logging thread last
    
    LOG_INFO("CoralEdgeTpu Detector Exiting.");
    
    return 0;
}
