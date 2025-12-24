#include "application.h"
#include "util_logging.h"
#include "queue_monitor.h"
#include <filesystem>
#include <iostream>
#include <csignal>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <termios.h>  // for terminal settings

// External declaration for terminal settings
extern struct termios original_termios;

// Forward declaration from main.cpp
std::vector<std::string> load_labels(const std::string& path);
extern std::atomic<bool> shutdown_requested; // Now managed by ApplicationSupervisor

Application::Application(int argc, char** argv) : argc_(argc), argv_(argv), recovery_running_(true), recovery_enabled_(false) {
    // Save original terminal settings
    tcgetattr(STDIN_FILENO, &original_termios);
    
    // Check for reduced resolution flag in command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--reduced-resolution" || std::string(argv[i]) == "-rr") {
            use_reduced_resolution_ = true;
            APP_LOG_INFO("Application: Reduced resolution mode enabled via command line argument");
        }
    }
    
    // Recovery thread will be started after modules are initialized
}

Application::~Application() {
    // Stop the recovery thread
    recovery_running_ = false;
    if (recovery_thread_.joinable()) {
        recovery_thread_.join();
    }
    
    supervisor_.initiate_shutdown();
    
    // Perform post-shutdown cleanup
    post_shutdown_cleanup();
    
    // Perform final cleanup of any remaining processes
    supervisor_.final_cleanup();
    
    Logger::getInstance().stop_writer_thread();
    APP_LOG_INFO("Shutdown complete.");
}

void Application::post_shutdown_cleanup() {
    APP_LOG_INFO("Performing post-shutdown cleanup...");
    
    // Ensure all threads are properly stopped and joined
    // This is already handled by the supervisor_.initiate_shutdown() call
    
    // Release camera and Edge TPU delegates
    release_edge_tpu_resources();
    release_camera_resources();
    
    // Close telemetry sockets and remove temporary files
    clear_telemetry_sockets();
    
    APP_LOG_INFO("Post-shutdown cleanup completed.");
}

void Application::setup_pools_and_queues() {
    const unsigned int cam_w = use_reduced_resolution_ ? 
        config_loader_.get_reduced_res_width() : 
        config_loader_.get_high_res_width();
    const unsigned int cam_h = use_reduced_resolution_ ? 
        config_loader_.get_reduced_res_height() : 
        config_loader_.get_high_res_height();
    
    size_t image_buffer_size = cam_w * cam_h * 3; // BGR888
    APP_LOG_INFO("Image buffer size per frame (BGR888): " + std::to_string(image_buffer_size) + " bytes.");
    
    // Determine pool size for images from config, default to a reasonable value if not set or invalid
    // For now, let's keep it at 80 as it's what's been tested, but log the memory usage.
    size_t image_pool_count = 20; // Increased to handle both TPU inference and visualization processing

    image_pool_ = std::make_shared<BufferPool<uint8_t>>(image_pool_count, image_buffer_size, "ImagePool");
    APP_LOG_INFO("ImagePool created with " + std::to_string(image_pool_count) + " buffers, total memory: " + std::to_string(image_pool_count * image_buffer_size / (1024 * 1024)) + " MB.");
    detection_pool_ = std::make_shared<BufferPool<DetectionResult>>(200, 200, "DetectionPool"); // Increased for better detection handling
    h264_pool_ = std::make_shared<BufferPool<uint8_t>>(300, 1024 * 1024, "H264Pool"); // Increased for better H264 encoding
}

bool Application::initialize_modules(const std::string& model_path, const std::string& labels_path) {
    const unsigned int cam_w = use_reduced_resolution_ ? 
        config_loader_.get_reduced_res_width() : 
        config_loader_.get_high_res_width();
    const unsigned int cam_h = use_reduced_resolution_ ? 
        config_loader_.get_reduced_res_height() : 
        config_loader_.get_high_res_height();
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

        // Send main video frames to main_video_queue_ instead of directly to overlaid_video_queue_
        main_image_output_queues_.push_back(main_video_queue_); 
        
        APP_LOG_INFO("Creating CameraCapture...");
        primary_camera_ = std::make_unique<CameraCapture>(
            cam_w, cam_h, // Main stream width/height
            config_loader_.get_tpu_stream_width(), config_loader_.get_tpu_stream_height(), // TPU stream width/height (from config)
            config_loader_.get_tpu_stream_fps(), // TPU stream FPS (from config)
            config_loader_.get_tpu_target_width(), config_loader_.get_tpu_target_height(), // Target TPU width/height (from config)
            image_pool_, main_image_output_queues_, raw_image_for_processor_queue_, camera_watchdog_timeout);
        APP_LOG_INFO("CameraCapture created.");

        APP_LOG_INFO("Creating ImageProcessor for TPU inference...");
        // Create ImageProcessor for TPU inference (no detection overlays needed)
        image_processor_ = std::make_unique<ImageProcessor>(
            raw_image_for_processor_queue_, tpu_inference_queue_, image_pool_,
            config_loader_.get_tpu_stream_pixel_format(), // Pass configured format
            inf_w, inf_h); // Pass target width/height for TPU processing
        APP_LOG_INFO("ImageProcessor created.");

        APP_LOG_INFO("Creating ImageProcessor for visualization with overlays...");
        // Create a new ImageProcessor instance for the main video stream with detection overlays
        visualization_processor_ = std::make_unique<ImageProcessor>(
            main_video_queue_, overlaid_video_queue_, detection_results_for_overlay_queue_, 
            image_pool_, 
            config_loader_.get_tpu_stream_pixel_format(), // Use same format as TPU stream for consistency
            cam_w, cam_h); // Use main camera dimensions
        APP_LOG_INFO("Visualization ImageProcessor created.");

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
        
        // DEBUGGING: Added std::cout to trace HttpStreamer creation
        std::cout << "DEBUG: Creating HttpStreamer..." << std::endl;
        std::vector<std::string> options = {"listening_ports", "8080"};
        http_streamer_ = std::make_unique<HttpStreamer>(options);
        std::cout << "DEBUG: HttpStreamer created." << std::endl;
        // END DEBUGGING

        APP_LOG_INFO("Creating Monitor...");
        monitor_ = std::make_unique<Monitor>(*this);
        APP_LOG_INFO("Monitor created.");

    } catch (const std::exception& e) {
        APP_LOG_ERROR("Failed to initialize modules: " + std::string(e.what()));
        return false;
    }

    return true;
}

bool Application::start_modules() {
    APP_LOG_INFO("Starting all modules...");
    bool start_ok = true;
    
    // Start modules in dependency order
    // 1. Start low-level services first
    start_ok &= system_monitor_->start();
    start_ok &= orientation_sensor_->start();
    
    // 2. Start processing modules
    start_ok &= image_processor_->start();
    start_ok &= inference_engine_->start();
    start_ok &= visualization_processor_->start();  // Start the visualization processor after inference is ready

    if (!start_ok) {
        APP_LOG_ERROR("One or more modules failed to start. Shutting down.");
        // Stop all modules that were successfully started
        if (keyboard_monitor_ && keyboard_monitor_->is_running()) keyboard_monitor_->stop();
        http_streamer_->stop(); // HttpStreamer doesn't have is_running() method
        if (h264_encoder_ && h264_encoder_->is_running()) h264_encoder_->stop();
        if (primary_camera_ && primary_camera_->is_running()) primary_camera_->stop();
        if (logic_module_ && logic_module_->is_running()) logic_module_->stop();
        if (inference_engine_ && inference_engine_->is_running()) inference_engine_->stop();
        if (image_processor_ && image_processor_->is_running()) image_processor_->stop();
        if (visualization_processor_ && visualization_processor_->is_running()) visualization_processor_->stop();  // Add visualization processor to shutdown
        if (orientation_sensor_ && orientation_sensor_->is_running()) orientation_sensor_->stop();
        if (system_monitor_ && system_monitor_->is_running()) system_monitor_->stop();
        if (monitor_) monitor_->stop();
        
        // Stop overlay consumer thread
        overlay_consumer_running_ = false;
        if (overlay_consumer_thread_.joinable()) {
            overlay_consumer_thread_.join();
        }
        
        // Stop H264 consumer thread
        h264_consumer_running_ = false;
        if (h264_consumer_thread_.joinable()) {
            h264_consumer_thread_.join();
        }
        
        return false;
    }
    
    // 3. Start logic module (depends on inference)
    start_ok &= logic_module_->start();
    
    // 4. Start camera (this may fail if camera is not available)
    if (!primary_camera_->start()) {
        APP_LOG_WARNING("Camera module failed to start. This may be due to camera hardware not being connected or IPA module issues.");
        start_ok = false;
    }
    
    // 5. Start encoder and streaming modules
    start_ok &= h264_encoder_->start();

    // Start the H264 consumer thread before HTTP streamer to avoid latency
    h264_consumer_running_ = true;
    h264_consumer_thread_ = std::thread(&Application::h264_queue_consumer_thread_func, this);

    // Start the overlay consumer thread
    overlay_consumer_running_ = true;
    overlay_consumer_thread_ = std::thread(&Application::overlay_queue_consumer_thread_func, this);

    // Start HTTP streamer after consumer threads are running
    http_streamer_->start(); // HttpStreamer start() returns void, so no &= assignment
    
    // 6. Start input monitoring
    start_ok &= keyboard_monitor_->start();
    
    // 7. Start monitor
    monitor_->start();

    


    if (!start_ok) {
        APP_LOG_ERROR("One or more modules failed to start. Shutting down.");
        // Stop all modules that were successfully started
        if (keyboard_monitor_ && keyboard_monitor_->is_running()) keyboard_monitor_->stop();
        http_streamer_->stop(); // HttpStreamer doesn't have is_running() method
        if (h264_encoder_ && h264_encoder_->is_running()) h264_encoder_->stop();
        if (primary_camera_ && primary_camera_->is_running()) primary_camera_->stop();
        if (logic_module_ && logic_module_->is_running()) logic_module_->stop();
        if (inference_engine_ && inference_engine_->is_running()) inference_engine_->stop();
        if (image_processor_ && image_processor_->is_running()) image_processor_->stop();
        if (visualization_processor_ && visualization_processor_->is_running()) visualization_processor_->stop();
        if (orientation_sensor_ && orientation_sensor_->is_running()) orientation_sensor_->stop();
        if (system_monitor_ && system_monitor_->is_running()) system_monitor_->stop();
        if (monitor_) monitor_->stop();
        
        // Stop overlay consumer thread
        overlay_consumer_running_ = false;
        if (overlay_consumer_thread_.joinable()) {
            overlay_consumer_thread_.join();
        }
        
        // Stop H264 consumer thread
        h264_consumer_running_ = false;
        if (h264_consumer_thread_.joinable()) {
            h264_consumer_thread_.join();
        }
        
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
    supervisor_.register_module_stop("ImageProcessor", [&]() { image_processor_->stop(); }); // Register ImageProcessor for TPU inference
    supervisor_.register_module_stop("VisualizationProcessor", [&]() { visualization_processor_->stop(); }); // Register VisualizationProcessor for shutdown
    supervisor_.register_module_stop("InferenceEngine", [&]() { inference_engine_->stop(); });
    supervisor_.register_module_stop("KeyboardMonitor", [&]() { keyboard_monitor_->stop(); });
    supervisor_.register_module_stop("HttpStreamer", [&]() { http_streamer_->stop(); }); // Register HTTP streamer for shutdown
    supervisor_.register_module_stop("OverlayConsumer", [&]() {
        overlay_consumer_running_ = false;
        if (overlay_consumer_thread_.joinable()) {
            overlay_consumer_thread_.join();
        }
    });
    supervisor_.register_module_stop("H264Consumer", [&]() {
        h264_consumer_running_ = false;
        if (h264_consumer_thread_.joinable()) {
            h264_consumer_thread_.join();
        }
    });
    supervisor_.register_module_stop("Monitor", [&]() { monitor_->stop(); });
}

void Application::recovery_thread_func() {
    APP_LOG_INFO("Recovery thread started.");
    
    while (recovery_running_ && !shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Check every 200ms
        
        // Only attempt recovery if it's enabled (modules have been started)
        if (!recovery_enabled_) {
            continue;
        }
        
        // Check each subsystem and attempt recovery if needed
        {
            std::lock_guard<std::mutex> lock(recovery_mutex_);
            
            // Check CameraCapture
            if (primary_camera_ && !primary_camera_->is_running()) {
                std::string subsystem = "CameraCapture";
                if (recovery_attempts_[subsystem] < max_recovery_attempts_) {
                    recovery_attempts_[subsystem]++;
                    APP_LOG_WARNING("Attempting to recover " + subsystem + " (attempt " + std::to_string(recovery_attempts_[subsystem]) + ")");
                    if (restart_camera_subsystem()) {
                        APP_LOG_INFO(subsystem + " recovered successfully.");
                        recovery_attempts_[subsystem] = 0; // Reset counter on success
                    } else {
                        APP_LOG_ERROR("Failed to recover " + subsystem + ".");
                    }
                }
            }
            
            // Check InferenceEngine
            if (inference_engine_ && !inference_engine_->is_running()) {
                std::string subsystem = "InferenceEngine";
                if (recovery_attempts_[subsystem] < max_recovery_attempts_) {
                    recovery_attempts_[subsystem]++;
                    APP_LOG_WARNING("Attempting to recover " + subsystem + " (attempt " + std::to_string(recovery_attempts_[subsystem]) + ")");
                    if (restart_inference_subsystem()) {
                        APP_LOG_INFO(subsystem + " recovered successfully.");
                        recovery_attempts_[subsystem] = 0; // Reset counter on success
                    } else {
                        APP_LOG_ERROR("Failed to recover " + subsystem + ".");
                    }
                }
            }
            
            // Check LogicModule
            if (logic_module_ && !logic_module_->is_running()) {
                std::string subsystem = "LogicModule";
                if (recovery_attempts_[subsystem] < max_recovery_attempts_) {
                    recovery_attempts_[subsystem]++;
                    APP_LOG_WARNING("Attempting to recover " + subsystem + " (attempt " + std::to_string(recovery_attempts_[subsystem]) + ")");
                    if (restart_logic_subsystem()) {
                        APP_LOG_INFO(subsystem + " recovered successfully.");
                        recovery_attempts_[subsystem] = 0; // Reset counter on success
                    } else {
                        APP_LOG_ERROR("Failed to recover " + subsystem + ".");
                    }
                }
            }
            
            // Check ImageProcessor
            if (image_processor_ && !image_processor_->is_running()) {
                std::string subsystem = "ImageProcessor";
                if (recovery_attempts_[subsystem] < max_recovery_attempts_) {
                    recovery_attempts_[subsystem]++;
                    APP_LOG_WARNING("Attempting to recover " + subsystem + " (attempt " + std::to_string(recovery_attempts_[subsystem]) + ")");
                    if (restart_image_processor_subsystem()) {
                        APP_LOG_INFO(subsystem + " recovered successfully.");
                        recovery_attempts_[subsystem] = 0; // Reset counter on success
                    } else {
                        APP_LOG_ERROR("Failed to recover " + subsystem + ".");
                    }
                }
            }
            
            // Check VisualizationProcessor
            if (visualization_processor_ && !visualization_processor_->is_running()) {
                std::string subsystem = "VisualizationProcessor";
                if (recovery_attempts_[subsystem] < max_recovery_attempts_) {
                    recovery_attempts_[subsystem]++;
                    APP_LOG_WARNING("Attempting to recover " + subsystem + " (attempt " + std::to_string(recovery_attempts_[subsystem]) + ")");
                    if (restart_visualization_subsystem()) {
                        APP_LOG_INFO(subsystem + " recovered successfully.");
                        recovery_attempts_[subsystem] = 0; // Reset counter on success
                    } else {
                        APP_LOG_ERROR("Failed to recover " + subsystem + ".");
                    }
                }
            }
            
            // Check H264Encoder
            if (h264_encoder_ && !h264_encoder_->is_running()) {
                std::string subsystem = "H264Encoder";
                if (recovery_attempts_[subsystem] < max_recovery_attempts_) {
                    recovery_attempts_[subsystem]++;
                    APP_LOG_WARNING("Attempting to recover " + subsystem + " (attempt " + std::to_string(recovery_attempts_[subsystem]) + ")");
                    if (restart_encoder_subsystem()) {
                        APP_LOG_INFO(subsystem + " recovered successfully.");
                        recovery_attempts_[subsystem] = 0; // Reset counter on success
                    } else {
                        APP_LOG_ERROR("Failed to recover " + subsystem + ".");
                    }
                }
            }
            
            // Check OrientationSensor
            if (orientation_sensor_ && !orientation_sensor_->is_running()) {
                std::string subsystem = "OrientationSensor";
                if (recovery_attempts_[subsystem] < max_recovery_attempts_) {
                    recovery_attempts_[subsystem]++;
                    APP_LOG_WARNING("Attempting to recover " + subsystem + " (attempt " + std::to_string(recovery_attempts_[subsystem]) + ")");
                    if (restart_orientation_subsystem()) {
                        APP_LOG_INFO(subsystem + " recovered successfully.");
                        recovery_attempts_[subsystem] = 0; // Reset counter on success
                    } else {
                        APP_LOG_ERROR("Failed to recover " + subsystem + ".");
                    }
                }
            }
        }
        
        // Reset recovery attempt counters every second
        static auto last_reset = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_reset).count() >= 1) {
            std::lock_guard<std::mutex> lock(recovery_mutex_);
            for (auto& pair : recovery_attempts_) {
                pair.second = 0;
            }
            last_reset = now;
        }
    }
    
    APP_LOG_INFO("Recovery thread stopped.");
}

bool Application::restart_camera_subsystem() {
    APP_LOG_INFO("Restarting CameraCapture subsystem...");
    
    try {
        // Stop the camera if it's running
        if (primary_camera_ && primary_camera_->is_running()) {
            primary_camera_->stop();
        }
        
        // Reset the camera capture object completely
        primary_camera_.reset();
        
        // Add a small delay to ensure resources are fully released
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Get camera configuration
        const unsigned int cam_w = use_reduced_resolution_ ? 
            config_loader_.get_reduced_res_width() : 
            config_loader_.get_high_res_width();
        const unsigned int cam_h = use_reduced_resolution_ ? 
            config_loader_.get_reduced_res_height() : 
            config_loader_.get_high_res_height();
        const std::chrono::seconds camera_watchdog_timeout = config_loader_.get_camera_watchdog_timeout();
        
        // Recreate the camera capture object
        primary_camera_ = std::make_unique<CameraCapture>(
            cam_w, cam_h, // Main stream width/height
            config_loader_.get_tpu_stream_width(), config_loader_.get_tpu_stream_height(), // TPU stream width/height (from config)
            config_loader_.get_tpu_stream_fps(), // TPU stream FPS (from config)
            config_loader_.get_tpu_target_width(), config_loader_.get_tpu_target_height(), // Target TPU width/height (from config)
            image_pool_, main_image_output_queues_, raw_image_for_processor_queue_, camera_watchdog_timeout);
        
        // Start the camera
        if (!primary_camera_->start()) {
            APP_LOG_ERROR("Failed to start CameraCapture after restart.");
            return false;
        }
        
        APP_LOG_INFO("CameraCapture restarted successfully.");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception during CameraCapture restart: " + std::string(e.what()));
        return false;
    } catch (...) {
        APP_LOG_ERROR("Unknown exception during CameraCapture restart.");
        return false;
    }
}

bool Application::restart_inference_subsystem() {
    APP_LOG_INFO("Restarting InferenceEngine subsystem...");
    
    try {
        // Stop the inference engine if it's running
        if (inference_engine_ && inference_engine_->is_running()) {
            inference_engine_->stop();
        }
        
        // Get model path
        std::filesystem::path exe_path = argv_[0];
        std::filesystem::path config_path = exe_path.parent_path() / ".." / "config.json";
        const std::string model_path = (config_path.parent_path() / config_loader_.get_model_path()).string();
        
        // Recreate the inference engine
        inference_engine_ = std::make_unique<InferenceEngine>(
            model_path, tpu_inference_queue_, detection_results_for_overlay_queue_, 
            detection_results_for_logic_queue_, detection_pool_, 
            config_loader_.get_detection_score_threshold(), 
            config_loader_.get_inference_worker_threads());
        
        // Start the inference engine
        if (!inference_engine_->start()) {
            APP_LOG_ERROR("Failed to start InferenceEngine after restart.");
            return false;
        }
        
        APP_LOG_INFO("InferenceEngine restarted successfully.");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception during InferenceEngine restart: " + std::string(e.what()));
        return false;
    } catch (...) {
        APP_LOG_ERROR("Unknown exception during InferenceEngine restart.");
        return false;
    }
}

bool Application::restart_logic_subsystem() {
    APP_LOG_INFO("Restarting LogicModule subsystem...");
    
    try {
        // Stop the logic module if it's running
        if (logic_module_ && logic_module_->is_running()) {
            logic_module_->stop();
        }
        
        // Recreate the logic module
        logic_module_ = std::make_unique<LogicModule>(detection_results_for_logic_queue_, orientation_sensor_, config_loader_);
        
        // Start the logic module
        if (!logic_module_->start()) {
            APP_LOG_ERROR("Failed to start LogicModule after restart.");
            return false;
        }
        
        APP_LOG_INFO("LogicModule restarted successfully.");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception during LogicModule restart: " + std::string(e.what()));
        return false;
    } catch (...) {
        APP_LOG_ERROR("Unknown exception during LogicModule restart.");
        return false;
    }
}

bool Application::restart_image_processor_subsystem() {
    APP_LOG_INFO("Restarting ImageProcessor subsystem...");
    
    try {
        // Stop the image processor if it's running
        if (image_processor_ && image_processor_->is_running()) {
            image_processor_->stop();
        }
        
        // Recreate the image processor
        unsigned int inf_w = config_loader_.get_tpu_target_width();
        unsigned int inf_h = config_loader_.get_tpu_target_height();
        
        image_processor_ = std::make_unique<ImageProcessor>(
            raw_image_for_processor_queue_, tpu_inference_queue_, image_pool_,
            config_loader_.get_tpu_stream_pixel_format(), // Pass configured format
            inf_w, inf_h); // Pass target width/height
        
        // Start the image processor
        if (!image_processor_->start()) {
            APP_LOG_ERROR("Failed to start ImageProcessor after restart.");
            return false;
        }
        
        APP_LOG_INFO("ImageProcessor restarted successfully.");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception during ImageProcessor restart: " + std::string(e.what()));
        return false;
    } catch (...) {
        APP_LOG_ERROR("Unknown exception during ImageProcessor restart.");
        return false;
    }
}

bool Application::restart_visualization_subsystem() {
    APP_LOG_INFO("Restarting VisualizationProcessor subsystem...");
    try {
        // Stop the visualization processor if it's running
        if (visualization_processor_ && visualization_processor_->is_running()) {
            visualization_processor_->stop();
        }
        // Get main camera dimensions
        const unsigned int cam_w = use_reduced_resolution_ ? 
            config_loader_.get_reduced_res_width() : 
            config_loader_.get_high_res_width();
        const unsigned int cam_h = use_reduced_resolution_ ? 
            config_loader_.get_reduced_res_height() : 
            config_loader_.get_high_res_height();
        
        // Recreate the visualization processor
        visualization_processor_ = std::make_unique<ImageProcessor>(
            main_video_queue_, overlaid_video_queue_, detection_results_for_overlay_queue_, 
            image_pool_, 
            config_loader_.get_tpu_stream_pixel_format(), // Use same format as TPU stream for consistency
            cam_w, cam_h); // Use main camera dimensions
        // Start the visualization processor
        if (!visualization_processor_->start()) {
            APP_LOG_ERROR("Failed to start VisualizationProcessor after restart.");
            return false;
        }
        APP_LOG_INFO("VisualizationProcessor restarted successfully.");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception during VisualizationProcessor restart: " + std::string(e.what()));
        return false;
    } catch (...) {
        APP_LOG_ERROR("Unknown exception during VisualizationProcessor restart.");
        return false;
    }
}

bool Application::restart_encoder_subsystem() {
    APP_LOG_INFO("Restarting H264Encoder subsystem...");
    
    try {
        // Stop the encoder if it's running
        if (h264_encoder_ && h264_encoder_->is_running()) {
            h264_encoder_->stop();
        }
        
        // Recreate the encoder
        const unsigned int cam_w = use_reduced_resolution_ ? 
            config_loader_.get_reduced_res_width() : 
            config_loader_.get_high_res_width();
        const unsigned int cam_h = use_reduced_resolution_ ? 
            config_loader_.get_reduced_res_height() : 
            config_loader_.get_high_res_height();
        
        h264_encoder_ = std::make_unique<H264Encoder>(overlaid_video_queue_, h264_output_queue_, h264_pool_, cam_w, cam_h, config_loader_.get_camera_fps());
        
        // Start the encoder
        if (!h264_encoder_->start()) {
            APP_LOG_ERROR("Failed to start H264Encoder after restart.");
            return false;
        }
        
        APP_LOG_INFO("H264Encoder restarted successfully.");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception during H264Encoder restart: " + std::string(e.what()));
        return false;
    } catch (...) {
        APP_LOG_ERROR("Unknown exception during H264Encoder restart.");
        return false;
    }
}

bool Application::restart_orientation_subsystem() {
    APP_LOG_INFO("Restarting OrientationSensor subsystem...");
    
    try {
        // Stop the orientation sensor if it's running
        if (orientation_sensor_ && orientation_sensor_->is_running()) {
            orientation_sensor_->stop();
        }
        
        // Recreate the orientation sensor
        orientation_sensor_ = std::make_shared<OrientationSensor>(config_loader_.get_phone_orientation_yaw_port(), config_loader_.get_phone_orientation_pitch_port(), config_loader_.get_phone_orientation_roll_port());
        
        // Start the orientation sensor
        if (!orientation_sensor_->start()) {
            APP_LOG_ERROR("Failed to start OrientationSensor after restart.");
            return false;
        }
        
        APP_LOG_INFO("OrientationSensor restarted successfully.");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception during OrientationSensor restart: " + std::string(e.what()));
        return false;
    } catch (...) {
        APP_LOG_ERROR("Unknown exception during OrientationSensor restart.");
        return false;
    }
}

void Application::overlay_queue_consumer_thread_func() {
    APP_LOG_INFO("Overlay consumer thread started (disabled - visualization processor handles overlays).");
    // The overlay consumer thread is now disabled since the visualization_processor
    // handles the overlay application directly to the main video stream.
    // We still need to consume from the queue to prevent it from filling up and blocking the inference engine.
    std::shared_ptr<DetectionResultBuffer> detections_buffer;
    while (overlay_consumer_running_) {
        if (!detection_results_for_overlay_queue_.pop(detections_buffer)) {
            // Reduced sleep to reduce latency and prevent queue buildup
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        // Just consume the detections without processing them since the visualization_processor handles them
    }
    APP_LOG_INFO("Overlay consumer thread stopped.");
}

void Application::h264_queue_consumer_thread_func() {
    APP_LOG_INFO("H264 queue consumer thread started.");
    
    std::shared_ptr<H264Buffer> h264_buffer;
    while (h264_consumer_running_) {
        // Attempt to pop a buffer from the H264 output queue
        if (h264_output_queue_.pop(h264_buffer)) {
            // Check if the buffer is valid and has data
            if (h264_buffer && h264_buffer->data.data() && h264_buffer->size > 0) {
                // Convert the H264 buffer data to a vector and push to HTTP streamer
                std::vector<uint8_t> data_vector(h264_buffer->data.data(), 
                                               h264_buffer->data.data() + h264_buffer->size);
                
                // Push the H264 data to the HTTP streamer
                if (http_streamer_) {
                    http_streamer_->pushH264Data(data_vector);
                }
            }
        } else {
            // If no data was available, sleep briefly to avoid busy-waiting
            // Reduced sleep to 25 microseconds to reduce latency and improve responsiveness
            std::this_thread::sleep_for(std::chrono::microseconds(25));
        }
    }
    APP_LOG_INFO("H264 queue consumer thread stopped.");
}

int Application::run() {
    APP_LOG_INFO("Application starting...");
    
    // Set up signal handlers to restore terminal settings on exit
    supervisor_.setup_signal_handlers();
    
    // Pre-launch cleanup to ensure a clean state
    pre_launch_cleanup();
    
    // Load configuration
    if (!config_loader_.load("config.json")) {
        APP_LOG_ERROR("Failed to load configuration.");
        return 1;
    }
    
    // Setup buffer pools and queues
    setup_pools_and_queues();
    
    // Initialize modules
    std::filesystem::path exe_path = argv_[0];
    std::filesystem::path config_path = exe_path.parent_path() / ".." / "config.json";
    const std::string model_path = (config_path.parent_path() / config_loader_.get_model_path()).string();
    const std::string labels_path = (config_path.parent_path() / config_loader_.get_labels_path()).string();
    
    if (!initialize_modules(model_path, labels_path)) {
        APP_LOG_ERROR("Failed to initialize modules.");
        return 1;
    }
    
    // Start all modules
    if (!start_modules()) {
        APP_LOG_ERROR("Failed to start modules.");
        return 1;
    }
    
    // Register shutdown handlers for graceful shutdown
    register_shutdown_handlers();
    
    // Start the recovery thread
    recovery_enabled_ = true;
    recovery_thread_ = std::thread(&Application::recovery_thread_func, this);
    
    APP_LOG_INFO("Application running. Press Ctrl+C to stop.");
    
    // Main loop - wait for shutdown signal
    auto last_monitoring_check = std::chrono::high_resolution_clock::now();
    const auto monitoring_interval = std::chrono::milliseconds(500); // Check every 500ms
    
    while (!shutdown_requested) {
        auto current_time = std::chrono::high_resolution_clock::now();
        
        // Perform monitoring checks at specified intervals
        if (current_time - last_monitoring_check >= monitoring_interval) {
            // Check for display starvation
            check_display_starvation();
            
            // Monitor queue depths
            monitor_queue_depths();
            
            // Enforce max latency (this will be implemented to check frame latencies)
            enforce_max_latency();
            
            last_monitoring_check = current_time;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Reduced sleep for more responsive monitoring
    }
    
    APP_LOG_INFO("Shutdown signal received. Stopping application...");
    
    // Stop the recovery thread
    recovery_running_ = false;
    if (recovery_thread_.joinable()) {
        recovery_thread_.join();
    }
    
    supervisor_.initiate_shutdown();
    
    // Join the consumer threads
    overlay_consumer_running_ = false;
    if (overlay_consumer_thread_.joinable()) {
        overlay_consumer_thread_.join();
    }
    
    h264_consumer_running_ = false;
    if (h264_consumer_thread_.joinable()) {
        h264_consumer_thread_.join();
    }
    
    // Perform post-shutdown cleanup
    post_shutdown_cleanup();
    
    APP_LOG_INFO("Application stopped.");
    return 0;
}

// Additional cleanup methods
void Application::release_edge_tpu_resources() {
    // Release Edge TPU resources if any
    APP_LOG_INFO("Releasing Edge TPU resources...");
}

void Application::release_camera_resources() {
    // Release camera resources if any
    APP_LOG_INFO("Releasing camera resources...");
}

void Application::clear_telemetry_sockets() {
    // Close any open telemetry sockets
    APP_LOG_INFO("Clearing telemetry sockets...");
}

void Application::pre_launch_cleanup() {
    APP_LOG_INFO("Performing pre-launch cleanup...");
    // Any cleanup needed before starting the application
    // This could include removing stale files, resetting states, etc.
}

void Application::aggressive_resource_cleanup() {
    APP_LOG_INFO("Performing aggressive resource cleanup...");
    // More thorough cleanup if needed
}

void Application::memory_leak_detection() {
    APP_LOG_INFO("Performing memory leak detection...");
    // Placeholder for memory leak detection if needed
}

void Application::temporary_file_cleanup() {
    APP_LOG_INFO("Performing temporary file cleanup...");
    // Clean up any temporary files
}

void Application::cleanup_ipc_resources() {
    APP_LOG_INFO("Cleaning up IPC resources...");
    // Clean up any inter-process communication resources
}

void Application::cleanup_shared_memory() {
    APP_LOG_INFO("Cleaning up shared memory...");
    // Clean up any shared memory segments
}

void Application::cleanup_zombie_processes() {
    APP_LOG_INFO("Cleaning up zombie processes...");
    // Clean up any zombie processes
}

void Application::generate_cleanup_report() {
    APP_LOG_INFO("Generating cleanup report...");
    // Generate a report of cleanup activities
}

bool Application::check_tpu_availability() {
    // Check if TPU is available
    return true; // Placeholder implementation
}

bool Application::wait_for_tpu_release(int max_wait_seconds) {
    // Wait for TPU to be released
    for (int i = 0; i < max_wait_seconds * 10; ++i) {
        if (check_tpu_availability()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

void Application::force_release_tpu_resources() {
    // Force release of TPU resources
    APP_LOG_INFO("Force releasing TPU resources...");
}

bool Application::verify_tpu_status() {
    // Verify TPU status
    return true; // Placeholder implementation
}

bool Application::terminate_existing_instances() {
    // Terminate any existing instances if needed
    return true; // Placeholder implementation
}

void Application::main_loop() {
    // Main application loop if needed
    // This is now handled in the run() method
}

void Application::debug_queue_monitoring() {
    APP_LOG_INFO("Starting queue size monitoring for 5 seconds...");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = start_time;
    
    while (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() < 5000) {
        
        // Log raw_image_for_processor_queue_ size every 100ms
        size_t raw_image_queue_depth = raw_image_for_processor_queue_.read_available();
        size_t raw_image_queue_capacity = raw_image_for_processor_queue_.write_available() + raw_image_queue_depth;
        APP_LOG_INFO("Raw Image Queue Size: " + std::to_string(raw_image_queue_depth) + "/" + std::to_string(raw_image_queue_capacity));
        
        // Also log other important queue sizes
        size_t main_video_queue_depth = main_video_queue_.read_available();
        size_t main_video_queue_capacity = main_video_queue_.write_available() + main_video_queue_depth;
        APP_LOG_INFO("Main Video Queue Size: " + std::to_string(main_video_queue_depth) + "/" + std::to_string(main_video_queue_capacity));
        
        size_t tpu_inference_queue_depth = tpu_inference_queue_.read_available();
        size_t tpu_inference_queue_capacity = tpu_inference_queue_.write_available() + tpu_inference_queue_depth;
        APP_LOG_INFO("TPU Inference Queue Size: " + std::to_string(tpu_inference_queue_depth) + "/" + std::to_string(tpu_inference_queue_capacity));
        
        // Sleep for 100ms before next check
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        current_time = std::chrono::high_resolution_clock::now();
    }
    
    APP_LOG_INFO("Completed queue size monitoring for 5 seconds.");
}

void Application::monitor_queue_depths() {
    // Log all queue depths using the QueueMonitor utility
    QueueMonitor::log_queue_depths(
        main_video_queue_,
        raw_image_for_processor_queue_,
        overlaid_video_queue_,
        detection_results_for_overlay_queue_,
        detection_results_for_logic_queue_,
        h264_output_queue_
    );
}

void Application::check_display_starvation() {
    // Check if any critical queues are starving (empty)
    bool main_video_starving = QueueMonitor::is_queue_starving(main_video_queue_, 1);
    bool overlaid_video_starving = QueueMonitor::is_queue_starving(overlaid_video_queue_, 1);
    bool h264_output_starving = QueueMonitor::is_queue_starving(h264_output_queue_, 1);
    
    if (main_video_starving) {
        APP_LOG_WARNING("MONITOR: Main video queue is starving (empty) - potential display starvation!");
    }
    
    if (overlaid_video_starving) {
        APP_LOG_WARNING("MONITOR: Overlaid video queue is starving (empty) - potential display starvation!");
    }
    
    if (h264_output_starving) {
        APP_LOG_WARNING("MONITOR: H264 output queue is starving (empty) - potential display starvation!");
    }
    
    // Log when queues are healthy
    if (!main_video_starving && !overlaid_video_starving && !h264_output_starving) {
        APP_LOG_DEBUG("MONITOR: All display queues have sufficient elements");
    }
}

void Application::enforce_max_latency() {
    // For max latency enforcement, we can check the H264 encoder for display starvation
    // and check if the last processed frame is too old
    if (h264_encoder_ && h264_encoder_->is_running()) {
        // Check if the H264 encoder is experiencing display starvation
        if (h264_encoder_->is_display_starving()) {
            APP_LOG_WARNING("MAX LATENCY: H264 encoder is experiencing display starvation!");
        }
    }
    
    // Log the monitoring activity
    APP_LOG_DEBUG("MONITOR: Max latency enforcement check executed");
}

void Application::debug_buffer_pool_monitoring() {
    APP_LOG_INFO("Starting buffer pool monitoring...");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = start_time;
    
    while (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() < 5000) {  // Monitor for 5 seconds
        
        // Log ImagePool usage
        if (image_pool_) {
            size_t available = image_pool_->get_available_buffers();
            size_t total = image_pool_->get_total_buffers();
            size_t in_use = image_pool_->get_current_in_use();
            size_t peak = image_pool_->get_peak_in_use();
            APP_LOG_INFO("ImagePool: Available: " + std::to_string(available) + 
                        ", Total: " + std::to_string(total) + 
                        ", In Use: " + std::to_string(in_use) + 
                        ", Peak: " + std::to_string(peak));
        }
        
        // Log DetectionPool usage
        if (detection_pool_) {
            size_t available = detection_pool_->get_available_buffers();
            size_t total = detection_pool_->get_total_buffers();
            size_t in_use = detection_pool_->get_current_in_use();
            size_t peak = detection_pool_->get_peak_in_use();
            APP_LOG_INFO("DetectionPool: Available: " + std::to_string(available) + 
                        ", Total: " + std::to_string(total) + 
                        ", In Use: " + std::to_string(in_use) + 
                        ", Peak: " + std::to_string(peak));
        }
        
        // Log H264Pool usage
        if (h264_pool_) {
            size_t available = h264_pool_->get_available_buffers();
            size_t total = h264_pool_->get_total_buffers();
            size_t in_use = h264_pool_->get_current_in_use();
            size_t peak = h264_pool_->get_peak_in_use();
            APP_LOG_INFO("H264Pool: Available: " + std::to_string(available) + 
                        ", Total: " + std::to_string(total) + 
                        ", In Use: " + std::to_string(in_use) + 
                        ", Peak: " + std::to_string(peak));
        }
        
        // Sleep for 1 second before next check
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        current_time = std::chrono::high_resolution_clock::now();
    }
    
    APP_LOG_INFO("Completed buffer pool monitoring.");
}

void Application::run_debugging_pipeline() {
    APP_LOG_INFO("Starting debugging pipeline test for 30 seconds...");
    
    // Start all the debugging monitoring functions in separate threads
    std::thread queue_monitor_thread([this]() {
        this->debug_queue_monitoring();
    });
    
    std::thread pool_monitor_thread([this]() {
        this->debug_buffer_pool_monitoring();
    });
    
    // Let the monitoring run for 30 seconds
    std::this_thread::sleep_for(std::chrono::seconds(30));
    
    APP_LOG_INFO("Completed debugging pipeline test for 30 seconds.");
    
    // Join the monitoring threads
    if (queue_monitor_thread.joinable()) {
        queue_monitor_thread.join();
    }
    
    if (pool_monitor_thread.joinable()) {
        pool_monitor_thread.join();
    }
}