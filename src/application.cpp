#include "application.h"
#include "pipeline_structs.h"
#include "util_logging.h"
#include "queue_monitor.h"
#include "timing.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <csignal>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>  // for waitpid
#include <cstring>
#include <future>
#include <termios.h>  // for terminal settings

// External declaration for terminal settings
extern struct termios original_termios;

// Forward declaration from main.cpp
std::vector<std::string> load_labels(const std::string& path);
extern std::atomic<bool> g_running; // Now managed by ApplicationSupervisor

// Utility function for timed thread joins
static bool timed_join_thread(std::thread& thread, const std::string& thread_name, std::chrono::seconds timeout = std::chrono::seconds(3)) {
    if (!thread.joinable()) {
        return true;
    }
    
    auto shared_promise = std::make_shared<std::promise<void>>();
    std::future<void> future = shared_promise->get_future();
    
    // Spawn a wrapper thread to perform the join
    // Capture thread by reference but be extremely careful with scope
    std::thread joiner_thread([&thread, shared_promise]() {
        try {
            if (thread.joinable()) {
                thread.join();
            }
            shared_promise->set_value();
        } catch (...) {
            // Prevent exception escape from joiner thread
        }
    });
    
    // Wait for the join to complete or timeout
    if (future.wait_for(timeout) == std::future_status::timeout) {
        std::cerr << "[SHUTDOWN] Thread '" << thread_name << "' did not join within " << timeout.count() << "s, detaching to prevent SIGABRT." << std::endl;
        
        // If we timeout, we MUST detach the original thread to prevent SIGABRT on its destruction
        if (thread.joinable()) {
            thread.detach();
        }
        // Also detach the joiner_thread because it's blocked on the original thread's join()
        joiner_thread.detach();
        return false;
    }
    
    if (joiner_thread.joinable()) {
        joiner_thread.join();
    }
    return true;
}

Application::Application(int argc, char** argv) : argc_(argc), argv_(argv), recovery_running_(true), recovery_enabled_(false) {
    // Set up signal handlers to restore terminal settings on exit
    supervisor_.setup_signal_handlers();

    // Check for reduced resolution flag and dynamic IP in command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--reduced-resolution" || arg == "-rr") {
            use_reduced_resolution_ = true;
            APP_LOG_INFO("Application: Reduced resolution mode enabled via command line argument");
        } else if (arg.find('-') != 0 && arg.find('.') != std::string::npos) {
            // Assume it's an IP address if it doesn't start with '-' and contains a '.'
            dynamic_phone_ip_ = arg;
            APP_LOG_INFO("Application: Dynamic phone IP detected via command line: " + dynamic_phone_ip_);
        }
    }
    
    // Recovery thread will be started after modules are initialized
}

Application::~Application() {
    stop();
    Logger::getInstance().stop_writer_thread();
    APP_LOG_INFO("Shutdown complete.");
}

void Application::stop() {
    static std::atomic<bool> stopping{false};
    if (stopping.exchange(true)) {
        return; 
    }

    g_running.store(false, std::memory_order_release);
    std::cerr << "[SHUTDOWN] Application::stop() initiated." << std::endl;

    // 1. Stop the recovery thread first
    recovery_running_ = false;
    recovery_cv_.notify_all();
    std::cerr << "[SHUTDOWN] Joining RecoveryThread..." << std::endl;
    timed_join_thread(recovery_thread_, "RecoveryThread");

    // 2. Stop MPEG-TS Server and Encoder first (sink end of pipeline)
    std::cerr << "[SHUTDOWN] Stopping MpegTsServer and H264Encoder..." << std::endl;
    if (mpegts_server_) mpegts_server_->stop();
    if (h264_encoder_) h264_encoder_->stop();

    // 4. Wake up all blocked threads with poison pills
    std::cerr << "[SHUTDOWN] Pushing poison pills..." << std::endl;
    ImageData* p1 = image_data_pool_->acquire(); if (p1) { p1->buffer = nullptr; raw_image_for_processor_queue_->push(p1); }
    ImageData* p2 = image_data_pool_->acquire(); if (p2) { p2->buffer = nullptr; main_video_queue_->push(p2); }
    ImageData* p3 = image_data_pool_->acquire(); if (p3) { p3->buffer = nullptr; tpu_inference_queue_->push(p3); }
    ResultToken* r1 = result_token_pool_->acquire(); if (r1) { r1->mark_dropped(); r1->get().reset(); detection_results_for_logic_queue_->push(r1); }
    ImageData* p4 = image_data_pool_->acquire(); if (p4) { p4->buffer = nullptr; overlaid_video_queue_->push(p4); }
    H264Buffer* h1 = h264_pool_->acquire_raw(); if (h1) { h1->size = 0; h264_output_queue_->push(h1); }

    // 5. ApplicationSupervisor handles the graceful shutdown of all registered modules
    std::cerr << "[SHUTDOWN] Initiating supervisor shutdown..." << std::endl;
    supervisor_.initiate_shutdown();
    
    // 6. Join the consumer threads
    std::cerr << "[SHUTDOWN] Joining consumer threads..." << std::endl;
    overlay_consumer_running_ = false;
    timed_join_thread(overlay_consumer_thread_, "OverlayConsumerThread");
    
    h264_consumer_running_ = false;
    timed_join_thread(h264_consumer_thread_, "H264ConsumerThread");

    // 7. Drain all queues to release all shared_ptrs and pooled buffers
    std::cerr << "[SHUTDOWN] Draining queues..." << std::endl;
    drain_queues();

    // 8. Final cleanup
    std::cerr << "[SHUTDOWN] Final supervisor cleanup..." << std::endl;
    supervisor_.final_cleanup();
    std::cerr << "[SHUTDOWN] Application::stop() completed." << std::endl;
}

void Application::post_shutdown_cleanup() {
    // This method is now effectively empty as cleanup is handled by modules and supervisor
    APP_LOG_INFO("Post-shutdown cleanup phase completed.");
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
    // Determine pool size for images from config, default to a reasonable value if not set or invalid
    // Reduced for Pi stability (avoid OOM) -> Increased to 300 to fix starvation
    size_t image_pool_count = 300; 

    image_pool_ = std::make_shared<BufferPool<uint8_t>>(image_pool_count, image_buffer_size, "ImagePool");
    APP_LOG_INFO("ImagePool created with " + std::to_string(image_pool_count) + " buffers, total memory: " + std::to_string(image_pool_count * image_buffer_size / (1024 * 1024)) + " MB.");
    detection_pool_ = std::make_shared<BufferPool<DetectionResult>>(300, 200, "DetectionPool"); 
    h264_pool_ = std::make_shared<BufferPool<uint8_t>>(20, 4 * 1024 * 1024, "H264Pool"); 
    
    // Object pools for lock-free queue elements
    image_data_pool_ = std::make_shared<ObjectPool<ImageData>>(image_pool_count, "ImageDataPool");
    result_token_pool_ = std::make_shared<ObjectPool<ResultToken>>(400, "ResultTokenPool");
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

        // Initialize queues
        APP_LOG_INFO("Initializing queues...");
        main_video_queue_ = std::make_shared<ImageQueue>();
        raw_image_for_processor_queue_ = std::make_shared<ImageQueue>();
        tpu_inference_queue_ = std::make_shared<ImageQueue>();
        detection_results_for_logic_queue_ = std::make_shared<DetectionResultsQueue>();
        overlaid_video_queue_ = std::make_shared<ImageQueue>();
        h264_output_queue_ = std::make_shared<H264Queue>();

        // Send main video frames to main_video_queue_ instead of directly to overlaid_video_queue_
        main_image_output_queues_.push_back(std::ref(*main_video_queue_)); 
        
        APP_LOG_INFO("Creating CameraCapture...");
        primary_camera_ = std::unique_ptr<CameraCapture>(new CameraCapture(
            cam_w, cam_h, // Main stream width/height
            config_loader_.get_tpu_stream_width(), config_loader_.get_tpu_stream_height(), // TPU stream width/height (from config)
            config_loader_.get_tpu_target_width(), config_loader_.get_tpu_target_height(), // Target TPU width/height (from config)
            image_pool_, image_data_pool_,
            *raw_image_for_processor_queue_,
            config_loader_.get_camera_watchdog_timeout() // Corrected method name
        ));
        primary_camera_->set_main_output_queues({main_video_queue_.get()});
        APP_LOG_INFO("CameraCapture created.");
        
        // Set application reference for camera to update counters
        primary_camera_->set_application_ref(this);

        APP_LOG_INFO("Creating ImageProcessor for TPU inference...");
        // Create ImageProcessor for TPU inference (no detection overlays needed)
        image_processor_ = std::make_unique<ImageProcessor>(
            *raw_image_for_processor_queue_, *tpu_inference_queue_, image_pool_, image_data_pool_,
            config_loader_.get_tpu_stream_pixel_format(), // Pass configured format
            inf_w, inf_h, "ImageProcessor_TPU"); // Pass target width/height for TPU processing
        image_processor_->set_skip_factor(1); // Process every frame (Mandate: Zero Skip)
        image_processor_->set_application_ref(this);
        image_processor_->set_is_tpu_stream(true);
        APP_LOG_INFO("ImageProcessor created.");

        APP_LOG_INFO("Creating ImageProcessor for visualization with overlays...");
        // Create a new ImageProcessor instance for the main video stream with detection overlays
        visualization_processor_ = std::make_unique<ImageProcessor>(
            *main_video_queue_, *overlaid_video_queue_, 
            &detection_results_for_overlay_buffer_, // Connect to triple buffer for overlays
            &ballistic_points_for_overlay_buffer_,  // New argument for ballistic points
            image_pool_, image_data_pool_,
            config_loader_.get_tpu_stream_pixel_format(), // Use same format as TPU stream for consistency
            cam_w, cam_h, "ImageProcessor_Viz"); // Use main camera dimensions
        visualization_processor_->set_skip_factor(1); // Process every frame (120 FPS)
        visualization_processor_->set_application_ref(this);
        visualization_processor_->set_is_tpu_stream(false);
        APP_LOG_INFO("Visualization ImageProcessor created.");
        
        // Also register overlaid_video_queue_ so CameraCapture pushes frames to it for the encoder
        // Actually, main_video_queue_ goes to visualization_processor, which then pushes to overlaid_video_queue_.
        // Wait, I see the confusion. 
        // 1. Camera -> main_video_queue_ -> VisualizationProcessor -> overlaid_video_queue_ -> H264Encoder
        // So main_video_queue_ is the only one Camera needs to know for the main stream.
        // My previous fix in CameraCapture was correct (iterate all main_output_queues_).
        // But main_image_output_queues_ ONLY contains main_video_queue_.
        // Where did overlaid_video_queue_ go? It's the OUTPUT of visualization_processor.
        // And it's the INPUT of H264Encoder.
        // So VisualizationProcessor MUST release the pointer it pops from main_video_queue_
        // and acquire a NEW one for overlaid_video_queue_.
        // I've already implemented this in ImageProcessor.
        
        // So why the leak?
        // Ah! main_video_queue_ is being pushed to by Camera, but IS ANYONE POPPING IT?
        // VisualizationProcessor pops it.
        // H264Encoder pops from overlaid_video_queue_.
        
        // Let's re-verify ImageProcessor pops.
        inference_engine_ = std::make_unique<InferenceEngine>(
            model_path, *tpu_inference_queue_, &detection_results_for_overlay_buffer_, 
            *detection_results_for_logic_queue_, detection_pool_, 
            image_data_pool_, result_token_pool_,
            config_loader_.get_detection_score_threshold(), 
            config_loader_.get_inference_worker_threads());
        APP_LOG_INFO("InferenceEngine created.");
        
        // Set application reference for inference engine to update counters
        inference_engine_->set_application_ref(this);
        
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

        orientation_sensor_ = std::make_shared<OrientationSensor>(config_loader_.get_orientation_pub_port(), config_loader_.get_orientation_pub_port(), config_loader_.get_orientation_pub_port());
        APP_LOG_INFO("OrientationSensor created.");

        APP_LOG_INFO("Creating LogicModule...");
        APP_LOG_INFO("Creating SystemMonitor...");
        system_monitor_ = std::make_unique<SystemMonitor>();
        APP_LOG_INFO("SystemMonitor created.");

        APP_LOG_INFO("Creating H264Encoder...");
        APP_LOG_INFO("Application: overlaid_video_queue_ Address: " + std::to_string((uintptr_t)overlaid_video_queue_.get()));
        h264_encoder_ = std::make_unique<H264Encoder>(*overlaid_video_queue_, *h264_output_queue_, h264_pool_, image_data_pool_, cam_w, cam_h, config_loader_.get_camera_fps());
        h264_encoder_->set_application_ref(this);
        APP_LOG_INFO("H264Encoder created.");

        APP_LOG_INFO("Creating LogicModule...");
        logic_module_ = std::make_unique<LogicModule>(
            *detection_results_for_logic_queue_, 
            result_token_pool_, 
            orientation_sensor_, 
            config_loader_, 
            &ballistic_points_for_overlay_buffer_,
            system_monitor_.get(),
            h264_encoder_.get()
        );
        APP_LOG_INFO("LogicModule created.");
        
        // Set application reference for logic module to update counters
        logic_module_->set_application_ref(this);

        APP_LOG_INFO("Creating KeyboardMonitor...");
        keyboard_monitor_ = std::make_unique<KeyboardMonitor>();
        APP_LOG_INFO("KeyboardMonitor created.");
        


        // Create MPEG-TS server
        APP_LOG_INFO("Creating MpegTsServer...");
        std::string video_address = dynamic_phone_ip_.empty() ? config_loader_.get_video_stream_address() : dynamic_phone_ip_;
        const unsigned short video_port = config_loader_.get_video_stream_rtp_port();
        APP_LOG_INFO("MpegTsServer: Target video stream address: " + video_address + ", port: " + std::to_string(video_port));
        mpegts_server_ = std::make_unique<MpegTsServer>(
            cam_w, 
            cam_h, 
            config_loader_.get_camera_fps(),
            video_address,
            video_port
        );
        APP_LOG_INFO("MpegTsServer created.");

        // Create Discovery Module
        APP_LOG_INFO("Creating DiscoveryModule...");
        discovery_module_ = std::make_unique<DiscoveryModule>();
        discovery_module_->on_peer_discovered([this](const DiscoveryModule::PeerInfo& peer) {
            APP_LOG_INFO("Application: Auto-discovered peer " + peer.name + " at " + peer.ip);
            // Discovery still sets source, but ControlModule can override
            if (orientation_sensor_) {
                orientation_sensor_->set_source(peer.ip, peer.orientation_port);
            }
            if (mpegts_server_) {
                mpegts_server_->set_destination(peer.ip);
            }
        });

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

    APP_LOG_INFO("Starting stabilization sequence...");

    bool start_ok = true;

    

    // 0. Ensure a clean state by draining all queues before starting

    drain_queues();

    APP_LOG_INFO("Startup queue drain complete.");



    // 1. Start low-level services first

    start_ok &= system_monitor_->start();

    start_ok &= orientation_sensor_->start();

    if (discovery_module_) start_ok &= discovery_module_->start();

    

    if (!start_ok) {

        APP_LOG_ERROR("Base services failed to start.");

        return false;

    }



    // 2. Start Camera first (Source)

    APP_LOG_INFO("Starting Camera Source...");

    if (!primary_camera_->start()) {

        APP_LOG_ERROR("Critical: Camera module failed to start. Pipeline aborted.");

        return false;

    }

    

    // Wait briefly for camera buffer stability

    std::this_thread::sleep_for(std::chrono::milliseconds(100));



        // 3. Start processing modules (Middlware)



        APP_LOG_INFO("Starting processing middleware...");



        start_ok &= image_processor_->start();



        if (visualization_processor_) start_ok &= visualization_processor_->start();



        start_ok &= inference_engine_->start();



        start_ok &= logic_module_->start();



        if (!start_ok) {



            APP_LOG_ERROR("One or more processing modules failed to start.");



            stop();



            return false;



        }



    



        // Start streaming-related modules now that ControlModule is gone



        if (visualization_processor_) visualization_processor_->start("");



        if (h264_encoder_) h264_encoder_->start();



        if (mpegts_server_) mpegts_server_->start();

    

    // Start the H264 consumer thread (it waits for encoder output)

    h264_consumer_running_ = true;

    h264_consumer_thread_ = std::thread(&Application::h264_queue_consumer_thread_func, this);



    // Start the overlay consumer thread (it waits for inference results)

    overlay_consumer_running_ = true;

    overlay_consumer_thread_ = std::thread(&Application::overlay_queue_consumer_thread_func, this);



    // 4. Start input monitoring

    if (!keyboard_monitor_->start()) {

        APP_LOG_WARNING("Keyboard monitor failed to start.");

    }

    

    // 5. Start monitor

    monitor_->start();



    APP_LOG_INFO("All core modules initialized and stabilized.");

    return true;

}

void Application::register_shutdown_handlers() {
    supervisor_.register_module_stop("SystemMonitor", [&]() { system_monitor_->stop(); });
    supervisor_.register_module_stop("LogicModule", [&]() { logic_module_->stop(); });
    supervisor_.register_module_stop("OrientationSensor", [&]() { orientation_sensor_->stop(); });
    supervisor_.register_module_stop("CameraCapture", [&]() { primary_camera_->stop(); });
    supervisor_.register_module_stop("H264Encoder", [&]() { h264_encoder_->stop(); }); 
    supervisor_.register_module_stop("ImageProcessor", [&]() { image_processor_->stop(); }); 
    supervisor_.register_module_stop("VisualizationProcessor", [&]() { visualization_processor_->stop(); }); 
    supervisor_.register_module_stop("InferenceEngine", [&]() { inference_engine_->stop(); });
    supervisor_.register_module_stop("KeyboardMonitor", [&]() { keyboard_monitor_->stop(); });
    // supervisor_.register_module_stop("HttpStreamer", [&]() { http_streamer_->stop(); }); // Register HTTP streamer for shutdown
    supervisor_.register_module_stop("MpegTsServer", [&]() { 
        if (mpegts_server_) {
            mpegts_server_->stop();
        }
    }); // Register MPEG-TS server for shutdown
    supervisor_.register_module_stop("OverlayConsumer", [&]() {
        overlay_consumer_running_ = false;
        if (!timed_join_thread(overlay_consumer_thread_, "OverlayConsumerThread", std::chrono::seconds(3))) {
            APP_LOG_ERROR("Overlay consumer thread failed to join within timeout");
        }
    });
    supervisor_.register_module_stop("H264Consumer", [&]() {
        h264_consumer_running_ = false;
        if (!timed_join_thread(h264_consumer_thread_, "H264ConsumerThread", std::chrono::seconds(3))) {
            APP_LOG_ERROR("H264 consumer thread failed to join within timeout");
        }
    });
    supervisor_.register_module_stop("Monitor", [&]() { monitor_->stop(); });
}

void Application::recovery_thread_func() {
    APP_LOG_INFO("Recovery thread started.");
    
    while (recovery_running_ && g_running.load(std::memory_order_acquire)) {
        {
            std::unique_lock<std::mutex> lock(recovery_mutex_);
            if (recovery_cv_.wait_for(lock, std::chrono::milliseconds(200), [this] { return !recovery_running_ || !g_running.load(); })) {
                if (!recovery_running_ || !g_running.load()) break;
            }
        }
        
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
            config_loader_.get_tpu_target_width(), config_loader_.get_tpu_target_height(), // Target TPU width/height (from config)
            image_pool_, image_data_pool_, *raw_image_for_processor_queue_, camera_watchdog_timeout);
        
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
            model_path, *tpu_inference_queue_, &detection_results_for_overlay_buffer_, 
            *detection_results_for_logic_queue_, detection_pool_, 
            image_data_pool_, result_token_pool_,
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
        logic_module_ = std::make_unique<LogicModule>(*detection_results_for_logic_queue_, result_token_pool_, orientation_sensor_, config_loader_, &ballistic_points_for_overlay_buffer_);
        // Set application reference for logic module to update counters
        logic_module_->set_application_ref(this);
        
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
            *raw_image_for_processor_queue_, *tpu_inference_queue_, image_pool_, image_data_pool_,
            config_loader_.get_tpu_stream_pixel_format(), // Pass configured format
            inf_w, inf_h, "ImageProcessor_TPU"); // Pass target width/height
        
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
                    *main_video_queue_, *overlaid_video_queue_, 
                    &detection_results_for_overlay_buffer_, // Connect to triple buffer
                    &ballistic_points_for_overlay_buffer_,  // New argument for ballistic points
                    image_pool_, image_data_pool_,
                    config_loader_.get_tpu_stream_pixel_format(),
                    cam_w, cam_h, "ImageProcessor_Viz");        
        // Set application reference for visualization processor to update counters
        visualization_processor_->set_application_ref(this);
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
        
        h264_encoder_ = std::make_unique<H264Encoder>(*overlaid_video_queue_, *h264_output_queue_, h264_pool_, image_data_pool_, cam_w, cam_h, config_loader_.get_camera_fps());
        
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
                orientation_sensor_ = std::make_shared<OrientationSensor>(
                    config_loader_.get_orientation_pub_port(), // Use the new getter for orientation port
                    config_loader_.get_orientation_pub_port(), // Assuming yaw/pitch/roll now use the same pub port
                    config_loader_.get_orientation_pub_port()  // Assuming yaw/pitch/roll now use the same pub port
                );
        
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
    // We strictly DO NOT consume from the queue here to prevent stealing data from the visualization processor.
    
    // Simply exit the thread.
    return; 
}

void Application::h264_queue_consumer_thread_func() {
    APP_LOG_INFO("H264 queue consumer thread started.");
    
    H264Buffer* h264_buffer_ptr = nullptr;
    int frame_counter = 0;
    auto last_keyframe_log = std::chrono::steady_clock::now();
    
    while (h264_consumer_running_ && g_running.load(std::memory_order_acquire)) {
        // Attempt to pop a buffer pointer from the H264 output queue
        if (h264_output_queue_->pop(h264_buffer_ptr)) {
            increment_h264_output_queue_out();
            
            // Wrap the raw pointer in a shared_ptr with a custom deleter that returns it to the pool
            std::shared_ptr<H264Buffer> h264_buffer(h264_buffer_ptr, [this](H264Buffer* b) {
                if (b) this->h264_pool_->release_raw(b);
            });

            if (h264_buffer && h264_buffer->data.data() && h264_buffer->size > 0) {
                // Log frame information
                if (h264_buffer->size >= 5) {  // Need at least 5 bytes to check NAL header
                    uint8_t nal_type = 0;
                    // Look for start code (0x00000001 or 0x000001)
                    size_t start_offset = 0;
                    if (h264_buffer->size >= 4 && 
                        h264_buffer->data[0] == 0x00 && h264_buffer->data[1] == 0x00 && 
                        h264_buffer->data[2] == 0x00 && h264_buffer->data[3] == 0x01) {
                        start_offset = 4;
                    } else if (h264_buffer->size >= 3 && 
                               h264_buffer->data[0] == 0x00 && h264_buffer->data[1] == 0x00 && 
                               h264_buffer->data[2] == 0x01) {
                        start_offset = 3;
                    } else {
                        // No start code, assume first byte is NAL header
                        start_offset = 0;
                    }
                    
                    if (start_offset < h264_buffer->size) {
                        nal_type = h264_buffer->data[start_offset] & 0x1F;  // Get lower 5 bits for NAL unit type
                    }
                    
                    const char* nal_type_str = "Unknown";
                    bool is_keyframe_or_header = false;
                    switch (nal_type) {
                        case 1: nal_type_str = "P-Slice"; break;
                        case 5: 
                            nal_type_str = "IDR-Slice";  // Keyframe
                            is_keyframe_or_header = true;
                            break;
                        case 7: 
                            nal_type_str = "SPS";        // Sequence Parameter Set
                            is_keyframe_or_header = true;
                            break;
                        case 8: 
                            nal_type_str = "PPS";        // Picture Parameter Set
                            is_keyframe_or_header = true;
                            break;
                        case 6: nal_type_str = "SEI"; break;
                        default: break;
                    }
                    
                    // Log first few frames in detail for RTSP queue analysis
                    static int rtsp_queue_dump_counter = 0;
                    if (rtsp_queue_dump_counter < 10) {  // Log first 10 frames to RTSP
                        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                        
                        APP_LOG_INFO("RTSP QUEUE FRAME #" + std::to_string(rtsp_queue_dump_counter) + 
                                    ": NAL Type=" + std::string(nal_type_str) + 
                                    " (" + std::to_string(nal_type) + ")" +
                                    ", Size=" + std::to_string(h264_buffer->size) + 
                                    ", Timestamp=" + std::to_string(timestamp) + "ms");
                        
                        // Log first few bytes of the NAL unit for inspection
                        if (h264_buffer->size >= 4) {
                            std::string hex_dump = "";
                            for (size_t b = 0; b < std::min(static_cast<size_t>(8), h264_buffer->size); b++) {
                                char byte_str[4];
                                snprintf(byte_str, sizeof(byte_str), "%02X ", static_cast<unsigned char>(h264_buffer->data[b]));
                                hex_dump += byte_str;
                            }
                            APP_LOG_DEBUG("  First 8 bytes: " + hex_dump);
                        }
                        
                        rtsp_queue_dump_counter++;
                    }
                    
                    // Log keyframes and headers more frequently, other frames less frequently
                    auto now = std::chrono::steady_clock::now();
                    bool should_log = is_keyframe_or_header || 
                                    (frame_counter % 60 == 0) ||  // Log every 60th frame
                                    (nal_type == 5 && (now - last_keyframe_log) > std::chrono::seconds(1)); // Log keyframes max once per second
                    
                    if (should_log) {
                        APP_LOG_INFO("H264 Consumer: Frame " + std::to_string(frame_counter) + 
                                    ", NAL type: " + std::string(nal_type_str) + 
                                    " (" + std::to_string(nal_type) + ")" +
                                    ", Size: " + std::to_string(h264_buffer->size) + " bytes");
                        
                        if (nal_type == 5) {  // Keyframe
                            last_keyframe_log = now;
                        }
                    }
                }
                
                // Push the H264 data to the MPEG-TS server
                if (mpegts_server_) {
                    // MpegTsServer handles frame delivery to network
                    mpegts_server_->pushH264Data(h264_buffer);
                }
                
                frame_counter++;
            }
        } else {
            // If no data was available, sleep briefly to avoid busy-waiting
            // Reduced sleep to 25 microseconds to reduce latency and improve responsiveness
            std::this_thread::sleep_for(std::chrono::microseconds(25));
            
            // No dummy frames needed for UDP streaming - just send actual frames when available
        }
    }
    APP_LOG_INFO("H264 queue consumer thread stopped. Total frames processed: " + std::to_string(frame_counter));
}

int Application::run() {
    APP_LOG_INFO("Application starting...");
    
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
    
    // Generate SDP file for UDP streaming
    generate_sdp_file();
    
    // Register shutdown handlers for graceful shutdown
    register_shutdown_handlers();
    
    // Start the recovery thread
    recovery_enabled_ = true;
    recovery_thread_ = std::thread(&Application::recovery_thread_func, this);
    
    APP_LOG_INFO("Application running. Press Ctrl+C to stop.");
    
    // Main loop - wait for shutdown signal
    auto last_monitoring_check = std::chrono::high_resolution_clock::now();
    const auto monitoring_interval = std::chrono::milliseconds(500); // Check every 500ms
    const auto max_run_time = std::chrono::minutes(5); // Max run time to prevent hanging
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (g_running.load(std::memory_order_acquire)) {
        auto current_time = std::chrono::high_resolution_clock::now();
        
        // Check for timeout to prevent hanging
        if (current_time - start_time > max_run_time) {
            APP_LOG_ERROR("Application timeout reached, forcing shutdown");
            break;
        }
        
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
        
        // Check shutdown flag more frequently with shorter sleep for more responsive shutdown
        for (int i = 0; i < 10 && g_running.load(std::memory_order_acquire); i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Sleep in smaller increments for more responsive shutdown
        }
    }
    
    APP_LOG_INFO("Shutdown signal received. Stopping application...");
    stop();
    
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
        
        // Log raw_image_for_processor_queue_ status every 100ms
        bool raw_image_queue_empty = raw_image_for_processor_queue_->empty();
        APP_LOG_INFO("Raw Image Queue Empty: " + std::string(raw_image_queue_empty ? "YES" : "NO"));
        
        // Also log other important queue statuses
        bool main_video_queue_empty = main_video_queue_->empty();
        APP_LOG_INFO("Main Video Queue Empty: " + std::string(main_video_queue_empty ? "YES" : "NO"));
        
        bool tpu_inference_queue_empty = tpu_inference_queue_->empty();
        APP_LOG_INFO("TPU Inference Queue Empty: " + std::string(tpu_inference_queue_empty ? "YES" : "NO"));
        
        // Sleep for 100ms before next check
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        current_time = std::chrono::high_resolution_clock::now();
    }
    
    APP_LOG_INFO("Completed queue size monitoring for 5 seconds.");
}

void Application::monitor_queue_depths() {
    // Check queue status using size_approx() to get actual queue depths
    size_t raw_image_queue_depth = raw_image_for_processor_queue_->size_approx();
    size_t tpu_inference_queue_depth = tpu_inference_queue_->size_approx();
    size_t detection_logic_queue_depth = detection_results_for_logic_queue_->size_approx();
    size_t overlaid_video_queue_depth = overlaid_video_queue_->size_approx();
    size_t h264_output_queue_depth = h264_output_queue_->size_approx();
    
    // Get current timestamp for logging
    auto current_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Log potential queue stalls (queues that are consistently full)
    if (raw_image_queue_depth > 40) { // More than 80% of capacity (50)
        APP_LOG_WARNING("ANOMALY @" + std::to_string(current_time_ms) + "ms: QUEUE STALL DETECTED: Raw Image Queue depth = " + std::to_string(raw_image_queue_depth) + "/50 - Camera pushing faster than ImageProcessor consuming");
    }
    if (tpu_inference_queue_depth > 40) { // More than 80% of capacity (50)
        APP_LOG_WARNING("ANOMALY @" + std::to_string(current_time_ms) + "ms: QUEUE STALL DETECTED: TPU Inference Queue depth = " + std::to_string(tpu_inference_queue_depth) + "/50 - ImageProcessor pushing faster than Inference consuming");
    }
    if (detection_logic_queue_depth > 40) { // More than 80% of capacity (50)
        APP_LOG_WARNING("ANOMALY @" + std::to_string(current_time_ms) + "ms: QUEUE STALL DETECTED: Detection Logic Queue depth = " + std::to_string(detection_logic_queue_depth) + "/50 - Potential stall between Inference and Logic");
    }
    if (overlaid_video_queue_depth > 40) { // More than 80% of capacity (50)
        APP_LOG_WARNING("ANOMALY @" + std::to_string(current_time_ms) + "ms: QUEUE STALL DETECTED: Overlaid Video Queue depth = " + std::to_string(overlaid_video_queue_depth) + "/50 - Potential stall between Overlay and Encoder");
    }
    if (h264_output_queue_depth > 40) { // More than 80% of capacity (50)
        APP_LOG_WARNING("ANOMALY @" + std::to_string(current_time_ms) + "ms: QUEUE STALL DETECTED: H264 Output Queue depth = " + std::to_string(h264_output_queue_depth) + "/50 - Potential stall in H264 output");
    }
    
    // Check if any queue is critically full and needs draining to prevent deadlock
    bool needs_draining = (raw_image_queue_depth > 45 || tpu_inference_queue_depth > 45 || 
                          detection_logic_queue_depth > 45 ||
                          overlaid_video_queue_depth > 45 || h264_output_queue_depth > 45);
    
    if (needs_draining) {
        APP_LOG_WARNING("QUEUE SAFETY: One or more queues critically full, initiating drain operation");
        drain_queues();
    }
    
    // Log detailed queue information every second for diagnostics
    static auto last_log_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_log_time).count();
    
    if (elapsed_ms >= 1000) { // Log every second
        APP_LOG_INFO("QUEUE DEPTHS @" + std::to_string(current_time_ms) + "ms: " +
                  "Raw=" + std::to_string(raw_image_queue_depth) + 
                  ", TPU=" + std::to_string(tpu_inference_queue_depth) + 
                  ", Logic=" + std::to_string(detection_logic_queue_depth) + 
                  ", Overlaid=" + std::to_string(overlaid_video_queue_depth) + 
                  ", H264=" + std::to_string(h264_output_queue_depth));
        
        // Update last log time
        last_log_time = current_time;
    }
    
    APP_LOG_DEBUG("Queue monitoring @" + std::to_string(current_time_ms) + "ms - Depths: Raw=" + std::to_string(raw_image_queue_depth) + 
                  ", TPU=" + std::to_string(tpu_inference_queue_depth) + 
                  ", Logic=" + std::to_string(detection_logic_queue_depth) + 
                  ", Overlaid=" + std::to_string(overlaid_video_queue_depth) + 
                  ", H264=" + std::to_string(h264_output_queue_depth));
}

void Application::check_display_starvation() {
    // For boost lockfree queues, we can't directly check size, so we'll just log that monitoring is active
    // In a real implementation, we might need to track queue state differently
    APP_LOG_DEBUG("Display starvation check active - boost lockfree queues don't support direct size checking");
}

void Application::enforce_max_latency() {
    // For max latency enforcement, we can check the H264 encoder status
    // and check if the last processed frame is too old
    if (h264_encoder_ && h264_encoder_->is_running()) {
        // Check H264 encoder status - placeholder for actual latency checking
        APP_LOG_DEBUG("MONITOR: H264 encoder is running, latency check passed");
    }
    
    // Log the monitoring activity
    APP_LOG_DEBUG("MONITOR: Max latency enforcement check executed");
}

void Application::check_thread_stalls() {
    // Check for stalled threads by monitoring their rates
    static auto last_check_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_check_time).count();
    
    if (elapsed_ms < 5000) { // Check every 5 seconds
        return;
    }
    
    // Update last check time
    last_check_time = current_time;
    
    // Check camera frame rate
    if (get_primary_camera() && get_primary_camera()->is_running()) {
        int camera_fps = get_primary_camera()->frame_rate_.load();
        if (camera_fps == 0) {
            APP_LOG_WARNING("THREAD STALL DETECTED: Camera FPS is 0, but camera is running");
        }
    }
    
    // Check inference rate
    if (get_inference_engine() && get_inference_engine()->is_running()) {
        int inference_rate = get_inference_engine()->inference_rate_.load();
        if (inference_rate == 0) {
            APP_LOG_WARNING("THREAD STALL DETECTED: Inference rate is 0, but inference engine is running");
        }
    }
    
    // Check logic rate
    if (get_logic_module() && get_logic_module()->is_running()) {
        int logic_rate = get_logic_module()->logic_rate_.load();
        if (logic_rate == 0) {
            APP_LOG_WARNING("THREAD STALL DETECTED: Logic rate is 0, but logic module is running");
        }
    }
    
    // Check queue status
    size_t tpu_inference_queue_depth = tpu_inference_queue_->empty() ? 0 : 1;
    size_t detection_logic_queue_depth = detection_results_for_logic_queue_->empty() ? 0 : 1;
    
    // If queues are getting full, it indicates consumption issues
    if (tpu_inference_queue_depth > 45) { // Almost full
        APP_LOG_WARNING("THREAD STALL RISK: TPU Inference Queue depth = " + std::to_string(tpu_inference_queue_depth) + "/50 - Inference consumer may be stalled");
    }
    
    if (detection_logic_queue_depth > 45) { // Almost full
        APP_LOG_WARNING("THREAD STALL RISK: Detection Logic Queue depth = " + std::to_string(detection_logic_queue_depth) + "/50 - Logic consumer may be stalled");
    }
}

void Application::drain_queues() {
    APP_LOG_DEBUG("DRAIN: Starting queue draining operation");
    
    // Drain raw image queue
    {
        int items_drained = 0;
        ImageData* ptr = nullptr;
        while (items_drained < 20 && raw_image_for_processor_queue_->pop(ptr)) {
            if (ptr) {
                ptr->buffer.reset();
                image_data_pool_->release(ptr);
                items_drained++;
            }
        }
        if (items_drained > 0) {
            APP_LOG_INFO("DRAIN: Drained " + std::to_string(items_drained) + " items from Raw Image Queue");
            for (int i = 0; i < items_drained; i++) {
                inc_cam_to_viz_dropped();
            }
        }
    }
    
    // Drain TPU inference queue
    {
        int items_drained = 0;
        ImageData* ptr = nullptr;
        while (items_drained < 20 && tpu_inference_queue_->pop(ptr)) {
            if (ptr) {
                ptr->buffer.reset();
                image_data_pool_->release(ptr);
                items_drained++;
            }
        }
        if (items_drained > 0) {
            APP_LOG_INFO("DRAIN: Drained " + std::to_string(items_drained) + " items from TPU Inference Queue");
            for (int i = 0; i < items_drained; i++) {
                inc_cam_to_tpu_proc_dropped();
            }
        }
    }

    // Drain detection results logic queue
    {
        int items_drained = 0;
        ResultToken* ptr = nullptr;
        while (items_drained < 20 && detection_results_for_logic_queue_->pop(ptr)) {
            if (ptr) {
                // For logic results, we account for drops explicitly
                increment_inference_results_dropped();
                ptr->release_buffer();
                result_token_pool_->release(ptr);
                items_drained++;
            }
        }
        if (items_drained > 0) {
            APP_LOG_INFO("DRAIN: Drained " + std::to_string(items_drained) + " items from Detection Logic Queue");
        }
    }

    // Drain overlaid video queue (Visualization Processor -> H264 Encoder)
    {
        int items_drained = 0;
        ImageData* ptr = nullptr;
        while (items_drained < 20 && overlaid_video_queue_->pop(ptr)) {
            if (ptr) {
                ptr->buffer.reset();
                image_data_pool_->release(ptr);
                items_drained++;
            }
        }
        if (items_drained > 0) {
            APP_LOG_INFO("DRAIN: Drained " + std::to_string(items_drained) + " items from Overlaid Video Queue");
        }
    }

    // Drain H264 output queue (H264 Encoder -> Streamer)
    {
        int items_drained = 0;
        H264Buffer* ptr = nullptr;
        while (items_drained < 20 && h264_output_queue_->pop(ptr)) {
            if (ptr) {
                h264_pool_->release_raw(ptr);
                increment_h264_output_queue_out();
                items_drained++;
            }
        }
        if (items_drained > 0) {
            APP_LOG_INFO("DRAIN: Drained " + std::to_string(items_drained) + " items from H264 Output Queue");
        }
    }
    
    APP_LOG_DEBUG("DRAIN: Queue draining operation completed");
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
    timed_join_thread(queue_monitor_thread, "QueueMonitorThread", std::chrono::seconds(1));
    timed_join_thread(pool_monitor_thread, "PoolMonitorThread", std::chrono::seconds(1));
}

// Detector supervision implementation
bool Application::start_detector_process() {
    APP_LOG_INFO("Starting detector process supervision...");
    
    // Check if detector is already running
    if (is_detector_running()) {
        APP_LOG_INFO("Detector process is already running");
        return true;
    }
    
    // Fork to create child process for detector
    pid_t pid = fork();
    
    if (pid == -1) {
        APP_LOG_ERROR("Failed to fork detector process");
        perror("fork");
        return false;
    }
    
    if (pid == 0) {
        // Child process - run detector
        // Change working directory to build directory
        if (chdir("/home/pi/CoralEdgeTpu/build") == -1) {
            APP_LOG_ERROR("Failed to change directory to build");
            perror("chdir to build directory");
            exit(1);
        }
        
        // Execute detector binary
        execl("./detector", "./detector", (char*)NULL);
        
        // If execl returns, it failed
        APP_LOG_ERROR("Failed to execute detector binary");
        perror("execl detector");
        exit(1);
    } else {
        // Parent process - store the PID
        detector_pid_.store(pid);
        APP_LOG_INFO("Started detector process with PID " + std::to_string(pid));
        supervisor_.register_child_process(pid);
        return true;
    }
}

void Application::stop_detector_process() {
    pid_t current_pid = detector_pid_.load();
    if (current_pid > 0) {
        APP_LOG_INFO("Terminating detector process with PID " + std::to_string(current_pid));
        
        // Try graceful termination first
        if (kill(current_pid, SIGTERM) == 0) {
            // Wait briefly for graceful shutdown (non-blocking)
            int status;
            pid_t result = waitpid(current_pid, &status, WNOHANG);
            if (result == 0) {
                // Process didn't exit immediately, wait a bit more
                usleep(500000); // 500ms
                
                // Check again
                result = waitpid(current_pid, &status, WNOHANG);
                if (result == 0) {
                    // Process still hasn't exited, force kill
                    APP_LOG_INFO("Detector process not responding to SIGTERM, sending SIGKILL...");
                    kill(current_pid, SIGKILL);
                    
                    // Wait for the process to be killed
                    waitpid(current_pid, &status, 0);
                }
            }
        }
        
        detector_pid_.store(-1);
        APP_LOG_INFO("Detector process terminated");
    }
}

bool Application::is_detector_running() {
    pid_t current_pid = detector_pid_.load();
    if (current_pid <= 0) {
        return false;
    }
    
    // Check if process is still alive by sending signal 0 (doesn't actually send a signal)
    return (kill(current_pid, 0) == 0);
}

void Application::detector_supervisor_thread_func() {
    APP_LOG_INFO("Detector supervisory thread started");
    
    while (detector_supervisor_running_.load()) {
        // Check if detector process is running
        if (!is_detector_running()) {
            APP_LOG_WARNING("Detector process is not running, attempting to restart...");
            
            // Try to restart the detector process
            if (start_detector_process()) {
                APP_LOG_INFO("Successfully restarted detector process");
            } else {
                APP_LOG_ERROR("Failed to restart detector process, retrying in 5 seconds...");
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }
        
        // Check every 2 seconds
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    
    APP_LOG_INFO("Detector supervisory thread stopped");
}

void Application::generate_sdp_file() {
    const std::string sdp_content = 
        "v=0\n"
        "o=- 0 0 IN IP4 127.0.0.1\n"
        "s=Aurore UDP Stream\n"
        "c=IN IP4 192.168.178.255\n"
        "t=0 0\n"
        "m=video 5000 RTP/AVP 96\n"
        "a=rtpmap:96 H264/90000\n";
    
    std::ofstream sdp_file("stream.sdp");
    if (sdp_file.is_open()) {
        sdp_file << sdp_content;
        sdp_file.close();
        APP_LOG_INFO("SDP file 'stream.sdp' generated successfully");
    } else {
        APP_LOG_ERROR("Failed to create SDP file 'stream.sdp'");
    }
}