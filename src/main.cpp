/**
 * @file main.cpp
 * @brief Main entry point for the CoralEdgeTpu Detector application.
 *
 * This application initializes and manages various modules for a real-time object
 * detection pipeline on a Raspberry Pi with a Coral Edge TPU. It handles camera
 * capture for inference and MJPEG streaming, runs TensorFlow Lite inference,
 * sends detection results via UDP, and provides an MJPEG web server for preview.
 * It also includes robust process supervision for the rpicam-vid camera streams
 * and graceful shutdown handling.
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

#include "pipeline_structs.h"
#include "camera_capture.h"
#include "mjpeg_capture.h"
#include "inference.h"
#include "mjpeg_server.h"
#include "udp_sender.h"
#include "util_logging.h"

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
 * @brief Main function of the CoralEdgeTpu Detector application.
 *
 * Sets up signal handlers, initializes the logger, defines application
 * configurations (model path, camera streams, network settings),
 * initializes various pipeline modules (camera capture, inference engine,
 * UDP sender, MJPEG server), starts them, enters a main loop that waits
 * for a shutdown signal, and then gracefully shuts down all modules.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return 0 on successful execution, 1 on initialization or startup failure.
 */
int main(int argc, char** argv) {
    // Register signal handlers for graceful shutdown on SIGINT or SIGTERM
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize the logger singleton
    Logger& logger = Logger::getInstance();
    LOG_INFO("CoralEdgeTpu Detector Starting...");

    // --- Application Configuration ---
    const std::string model_path = "/home/pi/CoralEdgeTpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
    const std::string labels_path = "/home/pi/CoralEdgeTpu/coco_labels.txt";
    
    // MJPEG Stream Configuration
    const unsigned int mjpeg_width = 640;
    const unsigned int mjpeg_height = 480;
    const unsigned int mjpeg_fps = 15;
    const int http_port = 8080;

    // UDP Sender Configuration
    const std::string udp_target_ip = "127.0.0.1";
    const int udp_target_port = 9000;

    // Watchdog timeout for camera streams (5 seconds of inactivity triggers a restart)
    const std::chrono::seconds camera_watchdog_timeout = std::chrono::seconds(5);

    // --- Thread-Safe Queues for Inter-Module Communication ---
    // Queue for raw image data from camera capture to inference engine
    ImageQueue camera_to_inference_queue;
    // Queue for detection results from inference engine to UDP sender
    UdpQueue inference_to_udp_queue;
    // Queue for MJPEG frames from capture to web server
    MjpegQueue mjpeg_capture_to_server_queue;

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
        inference_engine = std::make_unique<InferenceEngine>(model_path, camera_to_inference_queue, inference_to_udp_queue, 2);
    } catch (const std::runtime_error& e) {
        LOG_ERROR("Failed to initialize Inference Engine: " + std::string(e.what()));
        return 1;
    }
    
    // Get inference input dimensions from the loaded model
    const unsigned int inference_width = inference_engine->get_input_width();
    const unsigned int inference_height = inference_engine->get_input_height();

    // Log the current application configuration
    LOG_INFO("--- Configuration ---");
    LOG_INFO("  Inference Input: " + std::to_string(inference_width) + "x" + std::to_string(inference_height));
    LOG_INFO("  MJPEG Stream: " + std::to_string(mjpeg_width) + "x" + std::to_string(mjpeg_height) + "@" + std::to_string(mjpeg_fps) + "fps");
    LOG_INFO("  HTTP Port: " + std::to_string(http_port));
    LOG_INFO("  UDP Target: " + udp_target_ip + ":" + std::to_string(udp_target_port));
    LOG_INFO("---------------------");

    // Initialize CameraCapture modules
    // Dedicated BGR stream for TensorFlow Lite inference
    CameraCapture inference_camera(inference_width, inference_height, camera_to_inference_queue, camera_watchdog_timeout);
    // Dedicated MJPEG stream for web preview
    MjpegCapture mjpeg_camera(mjpeg_width, mjpeg_height, mjpeg_fps, mjpeg_capture_to_server_queue, camera_watchdog_timeout);

    // Initialize UDP Sender and MJPEG Server modules
    UdpSender udp_sender(udp_target_ip, udp_target_port, inference_to_udp_queue);
    MjpegServer mjpeg_server(http_port, mjpeg_capture_to_server_queue);

    // --- Start all modules ---
    if (!inference_camera.start() || !mjpeg_camera.start() || !inference_engine->start() || !udp_sender.start() || !mjpeg_server.start()) {
        LOG_ERROR("Failed to start one or more modules. Shutting down.");
        // Stop all modules in reverse order to ensure proper cleanup, even if some failed to start.
        mjpeg_server.stop();
        udp_sender.stop();
        inference_engine->stop();
        mjpeg_camera.stop();
        inference_camera.stop();
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
    // Stop all modules gracefully in a defined order
    mjpeg_server.stop();
    udp_sender.stop();
    inference_engine->stop();
    mjpeg_camera.stop();
    inference_camera.stop();
    
    LOG_INFO("CoralEdgeTpu Detector Exiting.");
    logger.stop_writer_thread(); // Stop logging thread last

    return 0;
}
