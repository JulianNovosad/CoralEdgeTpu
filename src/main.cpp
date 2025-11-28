#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <csignal>
#include <atomic>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <list>
#include <thread>

#include "pipeline_structs.h"
#include "camera_capture.h"
#include "inference.h"
#include "mjpeg_server.h"
#include "udp_sender.h"
#include "util_logging.h"
#include "udp_video_sender.h"
#include "video_overlay_processor.h"
// #include "image_resizer.h" // Obsolete: Image resizing is now handled by the ISP.
// #include "frame_displayer.h" // Obsolete: Display is handled by MJPEG stream.

std::atomic<bool> shutdown_requested(false);


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

int main(int argc, char** argv) {
    Logger& logger = Logger::getInstance();
    LOG_INFO("CoralEdgeTpu Detector Starting...");

    signal(SIGPIPE, SIG_IGN);

    // --- Application Configuration ---
    const std::string model_path = "/home/pi/CoralEdgeTpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
    const std::string labels_path = "/home/pi/CoralEdgeTpu/coco_labels.txt";
    
    const unsigned int high_res_width = 1536;
    const unsigned int high_res_height = 864;

    const int UDP_RAW_VIDEO_PORT = 50000;
    const int UDP_BOUNDING_BOX_PORT = 50010;
    const int HTTP_OVERLAID_VIDEO_PORT = 8081;
    const std::string MOBILE_APP_IP = "10.49.12.162";
    const std::chrono::seconds camera_watchdog_timeout = std::chrono::seconds(5);

    // --- Thread-Safe Queues ---
    ImageQueue full_res_for_resize_queue;
    ImageQueue full_res_for_overlay_queue;
    ImageQueue full_res_for_udp_queue;
    ImageQueue tpu_inference_queue; // New queue for TPU stream
    UdpQueue inference_results_queue;
    MjpegQueue overlaid_mjpeg_to_http_queue;

    // --- Module Initialization ---
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("Model file not found: " + model_path);
        return 1;
    }
    std::vector<std::string> labels = load_labels(labels_path);
    if (labels.empty()) {
        LOG_ERROR("Labels file not found or is empty: " + labels_path);
        return 1;
    }

    std::unique_ptr<InferenceEngine> inference_engine;
    try {
        // The inference engine now reads directly from the camera's TPU queue.
        inference_engine = std::make_unique<InferenceEngine>(model_path, tpu_inference_queue, inference_results_queue, 2);
    } catch (const std::runtime_error& e) {
        LOG_ERROR("Failed to initialize Inference Engine: " + std::string(e.what()));
        return 1;
    }
    
    const unsigned int inference_width = inference_engine->get_input_width();
    const unsigned int inference_height = inference_engine->get_input_height();

    // The TPU stream will now be requested at the exact inference resolution,
    // making the ImageResizer redundant.
    const unsigned int tpu_stream_width = inference_width;
    const unsigned int tpu_stream_height = inference_height;

    LOG_INFO("--- Configuration ---");
    LOG_INFO(std::string("  High-res Camera: ") + std::to_string(high_res_width) + "x" + std::to_string(high_res_height));
    LOG_INFO(std::string("  Inference Input: ") + std::to_string(inference_width) + "x" + std::to_string(inference_height));
    LOG_INFO(std::string("  Raw Video UDP Port: ") + std::to_string(UDP_RAW_VIDEO_PORT));
    LOG_INFO(std::string("  Bounding Box UDP Port: ") + std::to_string(UDP_BOUNDING_BOX_PORT));
    LOG_INFO(std::string("  Overlaid Video HTTP Port: ") + std::to_string(HTTP_OVERLAID_VIDEO_PORT));
    LOG_INFO(std::string("  Mobile App Target IP: ") + MOBILE_APP_IP);
    LOG_INFO("---------------------");

    std::list<std::reference_wrapper<ImageQueue>> primary_camera_output_queues;
    primary_camera_output_queues.push_back(std::ref(full_res_for_overlay_queue));
    primary_camera_output_queues.push_back(std::ref(full_res_for_udp_queue));

    // The tpu_inference_queue now goes directly to the inference engine.
    CameraCapture primary_camera(high_res_width, high_res_height, tpu_stream_width, tpu_stream_height, primary_camera_output_queues, tpu_inference_queue, camera_watchdog_timeout);

    // The ImageResizer is no longer needed as the ISP provides the correct size.
    // ImageResizer image_resizer(tpu_inference_queue, inference_queue, inference_width, inference_height);

    UdpVideoSender udp_raw_video_sender(MOBILE_APP_IP, UDP_RAW_VIDEO_PORT, full_res_for_udp_queue);
    VideoOverlayProcessor video_overlay_processor(full_res_for_overlay_queue, inference_results_queue, overlaid_mjpeg_to_http_queue, labels);
    UdpSender udp_bounding_box_sender(MOBILE_APP_IP, UDP_BOUNDING_BOX_PORT, inference_results_queue);
    MjpegServer overlaid_mjpeg_server(HTTP_OVERLAID_VIDEO_PORT, overlaid_mjpeg_to_http_queue);
    // FrameDisplayer frame_displayer(overlaid_mjpeg_to_http_queue, "Live Feed");

    // --- Start all modules ---
    if (!primary_camera.start() ||
        !inference_engine->start() ||
        !udp_raw_video_sender.start() ||
        !video_overlay_processor.start() ||
        !udp_bounding_box_sender.start() ||
        !overlaid_mjpeg_server.start()) {
        
        LOG_ERROR("Failed to start one or more modules. Shutting down.");
        overlaid_mjpeg_server.stop();
        udp_bounding_box_sender.stop();
        video_overlay_processor.stop();
        udp_raw_video_sender.stop();
        inference_engine->stop();
        primary_camera.stop();
        logger.stop_writer_thread();
        return 1;
    }

    std::thread input_thread([]() {
        char c;
        while (std::cin.get(c)) {
            if (c == 'o') {
                if (!shutdown_requested.exchange(true)) {
                    LOG_INFO("Shutdown requested, initiating cleanup...");
                }
                break;
            }
        }
    });

    LOG_INFO("Application started successfully. Waiting for shutdown signal ('o' followed by Enter).");
    while (!shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    input_thread.join();

    LOG_INFO("Shutting down application modules...");
    overlaid_mjpeg_server.stop();
}
