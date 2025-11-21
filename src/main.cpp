#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <csignal>
#include <atomic>
#include <fstream>
#include <filesystem>

#include "camera_capture.h"
#include "inference.h"
#include "mjpeg_server.h"
#include "udp_sender.h"
#include "jpeg_wrapper.h"
#include "util_logging.h"
#include "output_processor.h"

// Global atomic flag for shutdown
std::atomic<bool> shutdown_requested(false);

// Signal handler
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "Shutdown requested." << std::endl;
        shutdown_requested = true;
    }
}

// Function to load labels from a file
std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open labels file: " << path << std::endl;
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}


int main(int argc, char** argv) {
    // Set up signal handler for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize Logger
    Logger& logger = Logger::getInstance();
    LOG_INFO("CoralEdgeTpu Detector Starting...");

    // Hardcoded configuration
    const std::string model_path = "/home/pi/CoralEdgeTpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
    const std::string labels_path = "/home/pi/CoralEdgeTpu/coco_labels.txt";
    const unsigned int camera_width = 640;
    const unsigned int camera_height = 480;
    const int http_port = 8080;
    const std::string udp_target_ip = "127.0.0.1";
    const int udp_target_port = 9000;

    LOG_INFO("Configuration:");
    LOG_INFO("  Model Path: " + model_path);
    LOG_INFO("  Labels Path: " + labels_path);
    LOG_INFO("  Camera Mode: " + std::to_string(camera_width) + "x" + std::to_string(camera_height));
    LOG_INFO("  HTTP Port: " + std::to_string(http_port));
    LOG_INFO("  UDP Target: " + udp_target_ip + ":" + std::to_string(udp_target_port));

    // Load labels
    std::vector<std::string> labels = load_labels(labels_path);
    if (labels.empty()) {
        LOG_ERROR("Failed to load labels from: " + labels_path);
        return 1;
    }

    // Check if model exists
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("Model file not found: " + model_path);
        std::cerr << "ERROR: Model file not found. Please place your .tflite model at '"
                  << model_path << "'" << std::endl;
        return 1;
    }

    // --- Create shared queues ---
    ThreadSafeQueue camera_to_inference_queue;
    DetectionQueue inference_to_output_processor_queue; // Detections for output processing
    UdpQueue inference_to_udp_queue; // For sending raw detections
    MjpegQueue mjpeg_output_queue; // For MJPEG server

    // --- Initialize Modules ---
    CameraCapture camera_capture(camera_width, camera_height, camera_to_inference_queue);
    InferenceEngine inference_engine(model_path, camera_to_inference_queue, inference_to_output_processor_queue, 2); // 2 inference threads
    OutputProcessor output_processor(inference_to_output_processor_queue, mjpeg_output_queue, labels);
    MjpegServer mjpeg_server(http_port, mjpeg_output_queue);
    UdpSender udp_sender(udp_target_ip, udp_target_port, inference_to_udp_queue);

    // --- Start Modules ---
    bool success = true;
    if (success) {
        success = camera_capture.start();
        if (!success) LOG_ERROR("Failed to start camera capture.");
    }
    if (success) {
        success = inference_engine.start();
        if (!success) LOG_ERROR("Failed to start inference engine.");
    }
    if (success) {
        success = output_processor.start();
        if (!success) LOG_ERROR("Failed to start output processor.");
    }
    if (success) {
        success = mjpeg_server.start();
        if (!success) LOG_ERROR("Failed to start MJPEG server.");
    }
    if (success) {
        success = udp_sender.start();
        if (!success) LOG_ERROR("Failed to start UDP sender.");
    }

    // Main application loop
    if (success) {
        while (!shutdown_requested) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep to reduce CPU usage
        }
    }

    // --- Shutdown Modules (in reverse order of start-up) ---
    LOG_INFO("Shutting down application...");
    udp_sender.stop();
    mjpeg_server.stop();
    output_processor.stop();
    inference_engine.stop();
    camera_capture.stop();
    logger.stop_writer_thread(); // Stop logging thread last

    LOG_INFO("CoralEdgeTpu Detector Exiting.");

    return 0;
}
