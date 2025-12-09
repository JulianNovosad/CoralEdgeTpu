#include "application.h"
#include "util_logging.h"
#include <filesystem>
#include <iostream>
#include <csignal>

// Forward declaration from main.cpp
std::vector<std::string> load_labels(const std::string& path);
extern std::atomic<bool> shutdown_requested; // Now managed by ApplicationSupervisor

Application::Application(int argc, char** argv) : argc_(argc), argv_(argv) {}

Application::~Application() {
    supervisor_.initiate_shutdown();
    Logger::getInstance().stop_writer_thread();
    LOG_INFO("Shutdown complete.");
}

void Application::setup_pools_and_queues() {
    // const unsigned int cam_w = config_loader_.get_high_res_width();
    // const unsigned int cam_h = config_loader_.get_high_res_height();
    
    size_t image_buffer_size = 1920 * 1080 * 3; // BGR888 - Using default if vars are commented out
    image_pool_ = std::make_shared<BufferPool<uint8_t>>(20, image_buffer_size, "ImagePool");
    detection_pool_ = std::make_shared<BufferPool<DetectionResult>>(20, 100, "DetectionPool");
    h264_pool_ = std::make_shared<BufferPool<uint8_t>>(50, 1024 * 1024, "H264Pool");
}

bool Application::initialize_modules(const std::string& model_path, const std::string& labels_path) {
    // const unsigned int cam_w = config_loader_.get_high_res_width();
    // const unsigned int cam_h = config_loader_.get_high_res_height();
    // const double fps = config_loader_.get_camera_fps();
    const std::chrono::seconds camera_watchdog_timeout = config_loader_.get_camera_watchdog_timeout();
    
    // --- Labels ---
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("Model file not found: " + model_path);
        return false;
    }
    labels_ = load_labels(labels_path);
    if (labels_.empty()) {
        LOG_ERROR("Labels file empty: " + labels_path);
        return false;
    }

    // --- Module Creation ---
    try {
        inference_engine_ = std::make_unique<InferenceEngine>(
            model_path, tpu_inference_queue_, detection_results_for_overlay_queue_, 
            detection_results_for_logic_queue_, detection_pool_, 
            config_loader_.get_detection_score_threshold(), 
            config_loader_.get_inference_worker_threads());
        
        // unsigned int inf_w = inference_engine_->get_input_width();
        // unsigned int inf_h = inference_engine_->get_input_height();

        std::list<std::reference_wrapper<ImageQueue>> camera_queues;
        // primary_camera_ = std::make_unique<CameraCapture>(cam_w, cam_h, inf_w, inf_h, inf_w, inf_h, image_pool_, camera_queues, tpu_inference_queue_, camera_watchdog_timeout);
        orientation_sensor_ = std::make_shared<OrientationSensor>(config_loader_.get_phone_orientation_yaw_port(), config_loader_.get_phone_orientation_pitch_port(), config_loader_.get_phone_orientation_roll_port());
        logic_module_ = std::make_unique<LogicModule>(detection_results_for_logic_queue_, orientation_sensor_, config_loader_);
        // system_monitor_ = std::make_unique<SystemMonitor>();

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize modules: " + std::string(e.what()));
        return false;
    }

    return true;
}

bool Application::start_modules() {
    LOG_INFO("Starting all modules...");
    bool start_ok = true;
    start_ok &= inference_engine_->start();
    // start_ok &= primary_camera_->start();
    start_ok &= orientation_sensor_->start();
    start_ok &= logic_module_->start();
    // start_ok &= system_monitor_->start();

    if (!start_ok) {
        LOG_ERROR("One or more modules failed to start. Shutting down.");
        // if (system_monitor_->is_running()) system_monitor_->stop();
        if (logic_module_->is_running()) logic_module_->stop();
        if (orientation_sensor_->is_running()) orientation_sensor_->stop();
        // if (primary_camera_->is_running()) primary_camera_->stop();
        if (inference_engine_->is_running()) inference_engine_->stop();
        return false;
    }
    LOG_INFO("All modules started successfully.");
    return true;
}

void Application::register_shutdown_handlers() {
    // supervisor_.register_module_stop("SystemMonitor", [&]() { system_monitor_->stop(); });
    supervisor_.register_module_stop("LogicModule", [&]() { logic_module_->stop(); });
    supervisor_.register_module_stop("OrientationSensor", [&]() { orientation_sensor_->stop(); });
    // supervisor_.register_module_stop("CameraCapture", [&]() { primary_camera_->stop(); });
    supervisor_.register_module_stop("InferenceEngine", [&]() { inference_engine_->stop(); });
}

void Application::main_loop() {
    LOG_INFO("Running application. Press Ctrl+C to quit.");
    while (!shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        static auto last_metric_log_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_metric_log_time).count() >= 5) {
            inference_engine_->get_performance_metrics();
            logic_module_->get_performance_metrics();
            // primary_camera_->get_performance_metrics();
            last_metric_log_time = now;
        }
    }
}

int Application::run() {
    std::filesystem::path exe_path = argv_[0];
    std::filesystem::path config_path = exe_path.parent_path() / ".." / "config.json";
    if (!config_loader_.load(config_path.string())) {
        // Fallback logger if config fails
        Logger& logger = Logger::getInstance("run", "/tmp/corallog");
        logger.start_writer_thread();
        LOG_ERROR("Failed to load configuration file at " + config_path.string() + ". Exiting.");
        return 1;
    }

    // Initialize logger with the path from the config
    Logger& logger = Logger::getInstance("run", config_loader_.get_log_path());
    logger.start_writer_thread();
    LOG_INFO("CoralEdgeTpu Detector Starting...");

    signal(SIGPIPE, SIG_IGN);
    supervisor_.setup_signal_handlers();

    const std::string model_path = (config_path.parent_path() / config_loader_.get_model_path()).string();
    const std::string labels_path = (config_path.parent_path() / config_loader_.get_labels_path()).string();

    setup_pools_and_queues();

    if (!initialize_modules(model_path, labels_path)) {
        return 1;
    }
    
    if (!start_modules()) {
        return 1;
    }

    register_shutdown_handlers();
    main_loop();

    return 0;
}