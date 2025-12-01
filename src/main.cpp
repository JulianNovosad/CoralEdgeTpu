#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>

#include "pipeline_structs.h"
#include "camera_capture.h"
#include "inference.h"
#include "util_logging.h"
#include "config_loader.h"
#include "video_overlay_processor.h" // New include
#include "http_server.h" // New include
#include "h264_encoder.h" // New include for H264Encoder
#include "buffer_pool.h" // For BufferPool

std::atomic<bool> shutdown_requested(false);

std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open labels file: " + path);
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) labels.push_back(line);
    return labels;
}

int main(int argc, char** argv) {
    Logger& logger = Logger::getInstance();
    LOG_INFO("CoralEdgeTpu Detector Starting...");

    ConfigLoader config_loader;
    std::filesystem::path exe_path = argv[0];
    std::filesystem::path config_path = exe_path.parent_path() / ".." / "config.json";
    if (!config_loader.load(config_path.string())) {
        LOG_ERROR("Failed to load configuration. Exiting.");
        return 1;
    }

    signal(SIGPIPE, SIG_IGN);

    const std::string model_path = (config_path.parent_path() / config_loader.get_model_path()).string();
    const std::string labels_path = (config_path.parent_path() / config_loader.get_labels_path()).string();
    const unsigned int cam_w = config_loader.get_high_res_width();
    const unsigned int cam_h = config_loader.get_high_res_height();
    const std::chrono::seconds camera_watchdog_timeout = config_loader.get_camera_watchdog_timeout();
    const double fps = config_loader.get_camera_fps(); // Assuming camera_fps is available in ConfigLoader

    // --- Buffer Pools ---
    // Create a pool for image buffers. Size should be enough for a full frame.
    size_t image_buffer_size = cam_w * cam_h * 3; // e.g., 1920*1080*3
    BufferPool<uint8_t> image_pool(20, image_buffer_size, "ImagePool");
    // Create a pool for detection results. Size for max possible detections.
    BufferPool<DetectionResult> detection_pool(20, 100, "DetectionPool");
    // Create a pool for H.264 encoded packets. Size for a compressed frame.
    BufferPool<uint8_t> h264_pool(50, 256 * 1024, "H264Pool");


    // --- Queues for inference ---
    ImageQueue tpu_inference_queue;
    ImageQueue main_camera_output_queue; // Separate queue for main camera output
    DetectionResultsQueue detection_results_for_overlay_queue; // Queue for detection results to overlay
    ImageQueue overlaid_video_queue; // Queue for video frames with overlays for H264Encoder
    H264Queue h264_output_queue; // Queue for H.264 packets for HTTP server

    // --- Labels inladen ---
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("Model file not found: " + model_path);
        return 1;
    }
    auto labels = load_labels(labels_path);
    if (labels.empty()) {
        LOG_ERROR("Labels file empty: " + labels_path);
        return 1;
    }

    // --- Inference engine initialiseren ---
    std::unique_ptr<InferenceEngine> inference_engine;
    try {
        inference_engine = std::make_unique<InferenceEngine>(model_path, tpu_inference_queue, detection_results_for_overlay_queue, detection_pool, config_loader.get_inference_worker_threads());
    } catch (const std::runtime_error& e) {
        LOG_ERROR("Failed to initialize Inference Engine: " + std::string(e.what()));
        return 1;
    }
    unsigned int inf_w = inference_engine->get_input_width();
    unsigned int inf_h = inference_engine->get_input_height();

    LOG_INFO("Main: Initializing CameraCapture...");
    // --- Camera initialiseren ---
    std::list<std::reference_wrapper<ImageQueue>> camera_queues;
    camera_queues.push_back(std::ref(main_camera_output_queue)); // Main camera output to its own queue
    CameraCapture primary_camera(cam_w, cam_h, inf_w, inf_h, inf_w, inf_h, image_pool, camera_queues, tpu_inference_queue, camera_watchdog_timeout);
    LOG_INFO("Main: CameraCapture initialized.");

    LOG_INFO("Main: Initializing VideoOverlayProcessor...");
    VideoOverlayProcessor overlay_processor(main_camera_output_queue, detection_results_for_overlay_queue, overlaid_video_queue, labels);
    LOG_INFO("Main: VideoOverlayProcessor initialized.");

    LOG_INFO("Main: Initializing H264Encoder...");
    H264Encoder h264_encoder(overlaid_video_queue, h264_output_queue, h264_pool, cam_w, cam_h, fps);
    LOG_INFO("Main: H264Encoder initialized.");

    LOG_INFO("Main: Initializing HttpServer...");
    HttpServer http_server("0.0.0.0:" + std::to_string(config_loader.get_http_overlaid_video_port()), h264_output_queue);
    LOG_INFO("Main: HttpServer initialized.");

    LOG_INFO("Main: Starting CameraCapture, InferenceEngine, VideoOverlayProcessor, H264Encoder, and HttpServer...");
    if (!primary_camera.start() || !inference_engine->start() || !overlay_processor.start() || !h264_encoder.start() || !http_server.start()) {
        LOG_ERROR("Failed to start one or more modules.");
        primary_camera.stop();
        inference_engine->stop();
        overlay_processor.stop();
        h264_encoder.stop();
        http_server.stop();
        return 1;
    }
    LOG_INFO("Main: All modules started successfully.");

    std::thread input_thread([]() {
        char c;
        while (std::cin.get(c)) {
            if (c == 'o') {
                shutdown_requested = true;
                LOG_INFO("Shutdown requested...");
                break;
            }
        }
    });

    LOG_INFO("Running inference. Press 'o' + Enter to quit.");

    while (!shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        inference_engine->get_performance_metrics(); // Optioneel: periodieke metrics
    }

    input_thread.join();
    primary_camera.stop();
    inference_engine->stop();
    overlay_processor.stop();
    h264_encoder.stop(); // Stop the H264Encoder
    http_server.stop();
    logger.stop_writer_thread();

    LOG_INFO("Shutdown complete.");
}