#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <csignal>
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
#include "logic.h" // Include for the new LogicModule
#include "imu_sensor.h" // Include for IMUSensor
#include "application_supervisor.h" // Include for ApplicationSupervisor
#include "system_monitor.h" // Include for SystemMonitor

// std::atomic<bool> shutdown_requested(false); // Now managed by ApplicationSupervisor

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
    Logger& logger = Logger::getInstance("run", "/home/pi/CoralEdgeTpu/logs");
    LOG_INFO("CoralEdgeTpu Detector Starting...");

    ConfigLoader config_loader;
    std::filesystem::path exe_path = argv[0];
    std::filesystem::path config_path = exe_path.parent_path() / ".." / "config.json";
    if (!config_loader.load(config_path.string())) {
        LOG_ERROR("Failed to load configuration. Exiting.");
        return 1;
    }

    signal(SIGPIPE, SIG_IGN);

    // --- Application Supervisor ---
    ApplicationSupervisor supervisor;
    supervisor.setup_signal_handlers();

    const std::string model_path = (config_path.parent_path() / config_loader.get_model_path()).string();
    const std::string labels_path = (config_path.parent_path() / config_loader.get_labels_path()).string();
    const unsigned int cam_w = config_loader.get_high_res_width();
    const unsigned int cam_h = config_loader.get_high_res_height();
    const std::chrono::seconds camera_watchdog_timeout = config_loader.get_camera_watchdog_timeout();
    const double fps = config_loader.get_camera_fps(); // Assuming camera_fps is available in ConfigLoader

    // --- Buffer Pools ---
    // Create a pool for image buffers. Size should be enough for a full frame.
    size_t image_buffer_size = cam_w * cam_h * 3; // e.g., 1920*1080*3
    auto image_pool = std::make_shared<BufferPool<uint8_t>>(20, image_buffer_size, "ImagePool");
    // Create a pool for detection results. Size for max possible detections.
    auto detection_pool = std::make_shared<BufferPool<DetectionResult>>(20, 100, "DetectionPool");
    // Create a pool for H.264 encoded packets. Size for a compressed frame.
    auto h264_pool = std::make_shared<BufferPool<uint8_t>>(50, 1024 * 1024, "H264Pool");


    // --- Queues for inference ---
    ImageQueue tpu_inference_queue;
    ImageQueue main_camera_output_queue; // Separate queue for main camera output
    DetectionResultsQueue detection_results_for_overlay_queue; // Queue for detection results to VideoOverlayProcessor
    DetectionResultsQueue detection_results_for_logic_queue; // Queue for detection results to LogicModule
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
        inference_engine = std::make_unique<InferenceEngine>(model_path, tpu_inference_queue, detection_results_for_overlay_queue, detection_results_for_logic_queue, detection_pool, config_loader.get_inference_worker_threads());
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
    
    // --- Initializing IMUSensor ---
    LOG_INFO("Main: Initializing IMUSensor...");
    auto imu_sensor = std::make_shared<IMUSensor>();
    LOG_INFO("Main: IMUSensor initialized.");

    // --- Initializing LogicModule ---
    LOG_INFO("Main: Initializing LogicModule (centralized ballistics, object tracking, safety).");
    LogicModule logic_module(detection_results_for_logic_queue, imu_sensor);
    LOG_INFO("Main: LogicModule initialized.");

    // --- Initializing SystemMonitor ---
    LOG_INFO("Main: Initializing SystemMonitor...");
    SystemMonitor system_monitor;
    LOG_INFO("Main: SystemMonitor initialized.");

    LOG_INFO("Main: Starting CameraCapture, InferenceEngine, VideoOverlayProcessor, H264Encoder, HttpServer, IMUSensor, LogicModule and SystemMonitor...");
    bool start_ok = true;
    start_ok &= inference_engine->start();
    start_ok &= primary_camera.start();
    start_ok &= overlay_processor.start();
    start_ok &= h264_encoder.start();
    start_ok &= http_server.start();
    start_ok &= imu_sensor->start();
    start_ok &= logic_module.start();
    start_ok &= system_monitor.start();

    if (!start_ok) {
        LOG_ERROR("Failed to start one or more modules. Initiating shutdown.");
        // Register modules that successfully started for proper shutdown
        if (system_monitor.is_running()) supervisor.register_module_stop("SystemMonitor", [&]() { system_monitor.stop(); });
        if (logic_module.is_running()) supervisor.register_module_stop("LogicModule", [&]() { logic_module.stop(); });
        if (imu_sensor->is_running()) supervisor.register_module_stop("IMUSensor", [&]() { imu_sensor->stop(); });
        if (http_server.is_running()) supervisor.register_module_stop("HttpServer", [&]() { http_server.stop(); });
        if (h264_encoder.is_running()) supervisor.register_module_stop("H264Encoder", [&]() { h264_encoder.stop(); });
        if (overlay_processor.is_running()) supervisor.register_module_stop("VideoOverlayProcessor", [&]() { overlay_processor.stop(); });
        if (primary_camera.is_running()) supervisor.register_module_stop("CameraCapture", [&]() { primary_camera.stop(); });
        if (inference_engine->is_running()) supervisor.register_module_stop("InferenceEngine", [&]() { inference_engine->stop(); });
        supervisor.initiate_shutdown();
        return 1;
    }
    LOG_INFO("Main: All modules started successfully.");

    // Register all modules for graceful shutdown
    supervisor.register_module_stop("SystemMonitor", [&]() { system_monitor.stop(); });
    supervisor.register_module_stop("LogicModule", [&]() { logic_module.stop(); });
    supervisor.register_module_stop("IMUSensor", [&]() { imu_sensor->stop(); });
    supervisor.register_module_stop("HttpServer", [&]() { http_server.stop(); });
    supervisor.register_module_stop("H264Encoder", [&]() { h264_encoder.stop(); });
    supervisor.register_module_stop("VideoOverlayProcessor", [&]() { overlay_processor.stop(); });
    supervisor.register_module_stop("CameraCapture", [&]() { primary_camera.stop(); });
    supervisor.register_module_stop("InferenceEngine", [&]() { inference_engine->stop(); });
    
    LOG_INFO("Running application. Press Ctrl+C to quit.");

    while (!shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        inference_engine->get_performance_metrics(); // Optioneel: periodieke metrics
        logic_module.get_performance_metrics(); // Periodieke metrics for LogicModule
        primary_camera.get_performance_metrics(); // Periodieke metrics for CameraCapture
    }

    supervisor.initiate_shutdown();

    logger.stop_writer_thread();

    LOG_INFO("Shutdown complete.");
    return 0;
}