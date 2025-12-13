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
    APP_LOG_INFO("Shutdown complete.");
}

void Application::setup_pools_and_queues() {
    const unsigned int cam_w = config_loader_.get_high_res_width();
    const unsigned int cam_h = config_loader_.get_high_res_height();
    
    size_t image_buffer_size = cam_w * cam_h * 3; // BGR888
    APP_LOG_INFO("Image buffer size per frame (BGR888): " + std::to_string(image_buffer_size) + " bytes.");
    
    // Determine pool size for images from config, default to a reasonable value if not set or invalid
    // For now, let's keep it at 80 as it's what's been tested, but log the memory usage.
    size_t image_pool_count = 10; // Reduced to a more reasonable number

    image_pool_ = std::make_shared<BufferPool<uint8_t>>(image_pool_count, image_buffer_size, "ImagePool");
    APP_LOG_INFO("ImagePool created with " + std::to_string(image_pool_count) + " buffers, total memory: " + std::to_string(image_pool_count * image_buffer_size / (1024 * 1024)) + " MB.");
    detection_pool_ = std::make_shared<BufferPool<DetectionResult>>(20, 100, "DetectionPool");
    h264_pool_ = std::make_shared<BufferPool<uint8_t>>(200, 1024 * 1024, "H264Pool");
}

bool Application::initialize_modules(const std::string& model_path, const std::string& labels_path) {
    const unsigned int cam_w = config_loader_.get_high_res_width();
    const unsigned int cam_h = config_loader_.get_high_res_height();
    const std::chrono::seconds camera_watchdog_timeout = config_loader_.get_camera_watchdog_timeout();
    
    // --- Labels ---
    if (!std::filesystem::exists(model_path)) {
        APP_LOG_ERROR("Model file not found: " + model_path);
        return false;
    }
    labels_ = load_labels(labels_path);
    if (labels_.empty()) {
        APP_LOG_ERROR("Labels file empty: " + labels_path);
        return false;
    }

    // --- Module Creation ---
    try {
        unsigned int inf_w = config_loader_.get_tpu_target_width();
        unsigned int inf_h = config_loader_.get_tpu_target_height();

        main_image_output_queues_.push_back(overlaid_video_queue_); 
        
        APP_LOG_INFO("Creating CameraCapture...");
        primary_camera_ = std::make_unique<CameraCapture>(
            cam_w, cam_h, // Main stream width/height
            config_loader_.get_tpu_stream_width(), config_loader_.get_tpu_stream_height(), // TPU stream width/height (from config)
            config_loader_.get_tpu_stream_fps(), // TPU stream FPS (from config)
            config_loader_.get_tpu_target_width(), config_loader_.get_tpu_target_height(), // Target TPU width/height (from config)
            image_pool_, main_image_output_queues_, raw_image_for_processor_queue_, camera_watchdog_timeout);
        APP_LOG_INFO("CameraCapture created.");

        APP_LOG_INFO("Creating ImageProcessor...");
        image_processor_ = std::make_unique<ImageProcessor>(
            raw_image_for_processor_queue_, tpu_inference_queue_, image_pool_,
            config_loader_.get_tpu_stream_pixel_format(), // Pass configured format
            inf_w, inf_h); // Pass target width/height
        APP_LOG_INFO("ImageProcessor created.");

        APP_LOG_INFO("Creating InferenceEngine...");
        inference_engine_ = std::make_unique<InferenceEngine>(
            model_path, tpu_inference_queue_, detection_results_for_overlay_queue_, 
            detection_results_for_logic_queue_, detection_pool_, 
            config_loader_.get_detection_score_threshold(), 
            config_loader_.get_inference_worker_threads());
        APP_LOG_INFO("InferenceEngine created.");
        
        // Assert that InferenceEngine's actual input dimensions match the configured target.
        // This is a sanity check to ensure the model matches configuration.
        if (static_cast<unsigned int>(inference_engine_->get_input_width()) != inf_w || static_cast<unsigned int>(inference_engine_->get_input_height()) != inf_h) {
             {
                 std::stringstream ss;
                 ss << "InferenceEngine input dimensions (" << inference_engine_->get_input_width() << "x" << inference_engine_->get_input_height()
                    << ") from model do not match configured TPU target dimensions (" << inf_w << "x" << inf_h << "). This will cause errors.";
                 APP_LOG_ERROR(ss.str());
             }
             return false;
        }

        APP_LOG_INFO("Creating OrientationSensor...");
        orientation_sensor_ = std::make_shared<OrientationSensor>(config_loader_.get_phone_orientation_yaw_port(), config_loader_.get_phone_orientation_pitch_port(), config_loader_.get_phone_orientation_roll_port());
        APP_LOG_INFO("OrientationSensor created.");

        APP_LOG_INFO("Creating LogicModule...");
        logic_module_ = std::make_unique<LogicModule>(detection_results_for_logic_queue_, orientation_sensor_, config_loader_);
        APP_LOG_INFO("LogicModule created.");

        APP_LOG_INFO("Creating SystemMonitor...");
        system_monitor_ = std::make_unique<SystemMonitor>();
        APP_LOG_INFO("SystemMonitor created.");

        APP_LOG_INFO("Creating H264Encoder...");
        h264_encoder_ = std::make_unique<H264Encoder>(overlaid_video_queue_, h264_output_queue_, h264_pool_, cam_w, cam_h, config_loader_.get_camera_fps());
        APP_LOG_INFO("H264Encoder created.");

        APP_LOG_INFO("Creating KeyboardMonitor...");
        keyboard_monitor_ = std::make_unique<KeyboardMonitor>();
        APP_LOG_INFO("KeyboardMonitor created.");

    } catch (const std::exception& e) {
        APP_LOG_ERROR("Failed to initialize modules: " + std::string(e.what()));
        return false;
    }

    return true;
}

bool Application::start_modules() {
    APP_LOG_INFO("Starting all modules...");
    bool start_ok = true;
    start_ok &= image_processor_->start(); // Start ImageProcessor first
    start_ok &= inference_engine_->start();
    start_ok &= primary_camera_->start();
    start_ok &= orientation_sensor_->start();
    start_ok &= logic_module_->start();
    start_ok &= system_monitor_->start();
    start_ok &= h264_encoder_->start(); // Start the H264 encoder
    start_ok &= keyboard_monitor_->start();

    // Start the overlay consumer thread
    overlay_consumer_running_ = true;
    overlay_consumer_thread_ = std::thread(&Application::overlay_queue_consumer_thread_func, this);

    if (!start_ok) {
        APP_LOG_ERROR("One or more modules failed to start. Shutting down.");
        if (keyboard_monitor_->is_running()) keyboard_monitor_->stop();
        if (system_monitor_->is_running()) system_monitor_->stop();
        if (logic_module_->is_running()) logic_module_->stop();
        if (orientation_sensor_->is_running()) orientation_sensor_->stop();
        if (primary_camera_->is_running()) primary_camera_->stop();
        if (h264_encoder_->is_running()) h264_encoder_->stop(); // Stop the H264 encoder
        if (image_processor_->is_running()) image_processor_->stop(); // Stop ImageProcessor
        if (inference_engine_->is_running()) inference_engine_->stop();
        return false;
    }
    APP_LOG_INFO("All modules started successfully.");
    return true;
}

void Application::register_shutdown_handlers() {
    supervisor_.register_module_stop("SystemMonitor", [&]() { system_monitor_->stop(); });
    supervisor_.register_module_stop("LogicModule", [&]() { logic_module_->stop(); });
    supervisor_.register_module_stop("OrientationSensor", [&]() { orientation_sensor_->stop(); });
    supervisor_.register_module_stop("CameraCapture", [&]() { primary_camera_->stop(); });
    supervisor_.register_module_stop("H264Encoder", [&]() { h264_encoder_->stop(); }); // Register H264 encoder for shutdown
    supervisor_.register_module_stop("ImageProcessor", [&]() { image_processor_->stop(); }); // Register ImageProcessor for shutdown
    supervisor_.register_module_stop("InferenceEngine", [&]() { inference_engine_->stop(); });
    supervisor_.register_module_stop("KeyboardMonitor", [&]() { keyboard_monitor_->stop(); });
    supervisor_.register_module_stop("OverlayConsumer", [&]() {
        overlay_consumer_running_ = false;
        if (overlay_consumer_thread_.joinable()) {
            overlay_consumer_thread_.join();
        }
    });
}

void Application::overlay_queue_consumer_thread_func() {
    APP_LOG_INFO("Overlay consumer thread started.");
    std::shared_ptr<DetectionResultBuffer> detections_buffer;
    while (overlay_consumer_running_) {
        // Attempt to pop a buffer. If successful, it will be released when
        // detections_buffer goes out of scope or is reassigned.
        if (!detection_results_for_overlay_queue_.pop(detections_buffer)) {
            // If pop fails (queue empty or shut down), wait a bit to avoid busy-waiting.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    APP_LOG_INFO("Overlay consumer thread stopped.");
}

void Application::main_loop() {
    APP_LOG_INFO("Running application. Press 'o' to quit.");
    while (!shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int Application::run() {
    std::filesystem::path exe_path = argv_[0];
    std::filesystem::path config_path = exe_path.parent_path() / ".." / "config.json";
    if (!config_loader_.load(config_path.string())) {
        std::cerr << "ERROR: Failed to load configuration file at " << config_path.string() << ". Exiting." << std::endl;
        return 1;
    }

    // Extract CSV logging configurations - MUST be done BEFORE Logger::init
    std::vector<SubsystemLogConfig> csv_log_configs;
    if (config_loader_.get_json_config().contains("logging") && config_loader_.get_json_config()["logging"].contains("subsystems")) {
        for (const auto& sub_config : config_loader_.get_json_config()["logging"]["subsystems"]) {
            csv_log_configs.push_back({
                sub_config["name"].get<std::string>(),
                sub_config["log_dir_suffix"].get<std::string>(),
                sub_config["max_log_files"].get<int>()
            });
        }
    }

    // Initialize logger immediately after successful config load, and before any LOG_ calls
    Logger::init("run", config_loader_.get_log_path(), csv_log_configs);
    Logger::getInstance().start_writer_thread();
    APP_LOG_INFO("CoralEdgeTpu Detector Starting..."); 

    signal(SIGPIPE, SIG_IGN);
    supervisor_.setup_signal_handlers(); 
    APP_LOG_INFO("Signal handlers for SIGINT and SIGTERM set up."); 


    const std::string model_path = (config_path.parent_path() / config_loader_.get_model_path()).string();
    const std::string labels_path = (config_path.parent_path() / config_loader_.get_labels_path()).string();

    setup_pools_and_queues();

    APP_LOG_INFO("Initializing modules...");
    if (!initialize_modules(model_path, labels_path)) {
        return 1;
    }
    APP_LOG_INFO("Modules initialized.");
    
    APP_LOG_INFO("Starting modules...");
    if (!start_modules()) {
        return 1;
    }
    APP_LOG_INFO("Modules started.");

    register_shutdown_handlers();
    main_loop();

    return 0;
}