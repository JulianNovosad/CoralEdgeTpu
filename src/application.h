#ifndef APPLICATION_H
#define APPLICATION_H

#include <termios.h>

#include "application_supervisor.h"
#include "camera_capture.h"
#include "config_loader.h"
#include "h264_encoder.h"
#include "inference.h"
#include "logic.h"
#include "orientation_sensor.h"
#include "pipeline_structs.h"
#include "system_monitor.h"
#include "image_processor.h" // New include
#include "keyboard_monitor.h"
#include "discovery_module.h" // Added for auto-discovery

#include "mpegts_server.h"
#include "monitor.h"

#include "buffer_pool.h"
#include <opencv2/opencv.hpp>

#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <future>

// Include EdgeTPU headers for TPU occupancy management
#include "edgetpu_c.h"

// Include signal handling headers
#include <signal.h>
#include <pthread.h>

/**
 * @brief Een applicatieklasse die de volledige CoralEdgeTpu-pijplijn beheert.
 *
 * Deze klasse is verantwoordelijk voor het initialiseren, configureren, starten
 * en stoppen van alle modules in de beeldverwerkings- en inference-pijplijn.
 * Het centraliseert de setup-logica om de `main`-functie schoon te houden.
 */
class Application {
public:
    Application(int argc, char** argv);
    ~Application();

    /**
     * @brief Start de volledige applicatiepijplijn.
     *
     * Initialiseert en start alle modules in de juiste volgorde. Registreert
     * de modules bij de supervisor voor een graceful shutdown.
     *
     * @return 0 bij succes, 1 bij een fout.
     */
    int run();
    void stop();

public:
    // Expose member variables for Monitor class
    std::unique_ptr<ImageProcessor> image_processor_; // New module
    std::unique_ptr<InferenceEngine> inference_engine_;
    std::unique_ptr<CameraCapture> primary_camera_;
    // std::unique_ptr<VideoOverlayProcessor> overlay_processor_;
    std::unique_ptr<H264Encoder> h264_encoder_;

    std::unique_ptr<MpegTsServer> mpegts_server_;
    std::shared_ptr<OrientationSensor> orientation_sensor_;
    std::unique_ptr<LogicModule> logic_module_;
    std::unique_ptr<SystemMonitor> system_monitor_;
    std::unique_ptr<KeyboardMonitor> keyboard_monitor_;
    std::unique_ptr<Monitor> monitor_;
    std::unique_ptr<DiscoveryModule> discovery_module_; // Added for auto-discovery
    std::unique_ptr<ImageProcessor> visualization_processor_; // Added for visualization
    std::shared_ptr<ImageQueue> main_video_queue_; // Added queue
    bool use_reduced_resolution_ = false; // Added flag
    
    // Queues (moved to public for monitor access)
    std::shared_ptr<ImageQueue> raw_image_for_processor_queue_; // New queue
    std::shared_ptr<ImageQueue> tpu_inference_queue_;

    TripleBuffer<DetectionResults> detection_results_for_overlay_buffer_;
    TripleBuffer<OverlayBallisticPoint> ballistic_points_for_overlay_buffer_; 
    std::shared_ptr<DetectionResultsQueue> detection_results_for_logic_queue_;
    std::shared_ptr<ImageQueue> overlaid_video_queue_;
    std::shared_ptr<H264Queue> h264_output_queue_;
    std::string dynamic_phone_ip_; // Added to store phone IP from command line

private:
    void release_edge_tpu_resources();
    void release_camera_resources();
    void clear_telemetry_sockets();
    
    // Enhanced cleanup functions
    void pre_launch_cleanup();
    void post_shutdown_cleanup();
    void aggressive_resource_cleanup();
    void memory_leak_detection();
    void temporary_file_cleanup();
    void cleanup_ipc_resources();
    void cleanup_shared_memory();
    void cleanup_zombie_processes();
    void generate_cleanup_report();
    
    // TPU occupancy management functions
    bool check_tpu_availability();
    bool wait_for_tpu_release(int max_wait_seconds = 10);
    void force_release_tpu_resources();
    bool verify_tpu_status();
    
    bool terminate_existing_instances();
    
    void setup_pools_and_queues();
    bool initialize_modules(const std::string& model_path, const std::string& labels_path);
    bool start_modules();
    void register_shutdown_handlers();
    void main_loop();
    
    // Additional monitoring functions
    void check_display_starvation();
    void monitor_queue_depths();
    void enforce_max_latency();
    void check_thread_stalls();
    void drain_queues();
    void debug_queue_monitoring();
    void debug_buffer_pool_monitoring();
    void run_debugging_pipeline();
    
    // Recovery mechanisms
    void recovery_thread_func();
    bool restart_camera_subsystem();
    bool restart_inference_subsystem();
    bool restart_logic_subsystem();
    bool restart_image_processor_subsystem();
    bool restart_encoder_subsystem();
    bool restart_orientation_subsystem();
    bool restart_visualization_subsystem(); // Added method
    
    int argc_;
    char** argv_;

    ConfigLoader config_loader_;
    ApplicationSupervisor supervisor_;
    
    // Buffer Pools
    std::shared_ptr<BufferPool<uint8_t>> image_pool_;
    std::shared_ptr<BufferPool<DetectionResult>> detection_pool_;
    std::shared_ptr<BufferPool<uint8_t>> h264_pool_;
    
    // Object Pools for queue elements
    std::shared_ptr<ObjectPool<ImageData>> image_data_pool_;
    std::shared_ptr<ObjectPool<ResultToken>> result_token_pool_;

    // Thread for consuming overlay detection results
    std::thread overlay_consumer_thread_;
    std::atomic<bool> overlay_consumer_running_{false};
    void overlay_queue_consumer_thread_func();
    std::list<std::reference_wrapper<ImageQueue>> main_image_output_queues_;
    
    // Thread for consuming H.264 video stream
    std::thread h264_consumer_thread_;
    std::atomic<bool> h264_consumer_running_{false};
    void h264_queue_consumer_thread_func();

    // Recovery mechanisms
    std::atomic<bool> recovery_running_{false};
    std::atomic<bool> recovery_enabled_{false}; // New flag to control when recovery is active
    std::thread recovery_thread_;
    std::mutex recovery_mutex_;
    std::condition_variable recovery_cv_;
    
    // Detector supervision
    std::thread detector_supervisor_thread_;
    std::atomic<bool> detector_supervisor_running_{false};
    std::atomic<pid_t> detector_pid_{-1};
    void detector_supervisor_thread_func();
    bool start_detector_process();
    void stop_detector_process();
    bool is_detector_running();
    
    void generate_sdp_file();
    
    // Recovery counters for each subsystem
    std::map<std::string, int> recovery_attempts_;
    const int max_recovery_attempts_ = 5; // Maximum attempts per second

    std::vector<std::string> labels_;
    
public:
    // --- STAGE 1: Camera -> Processors ---
    // (One frame from camera produces ONE frame for EACH registered consumer queue)
    std::atomic<int64_t> cam_to_viz_produced_{0};
    std::atomic<int64_t> cam_to_viz_dropped_{0};
    std::atomic<int64_t> cam_to_viz_consumed_{0}; // by Viz Processor

    std::atomic<int64_t> cam_to_tpu_proc_produced_{0};
    std::atomic<int64_t> cam_to_tpu_proc_dropped_{0};
    std::atomic<int64_t> cam_to_tpu_proc_consumed_{0}; // by TPU Processor

    // --- STAGE 2: TPU Processor -> Inference Engine ---
    // (Each frame processed by TPU Processor produces ONE frame for Inference Engine)
    std::atomic<int64_t> proc_to_inf_produced_{0};
    std::atomic<int64_t> proc_to_inf_dropped_{0};
    std::atomic<int64_t> proc_to_inf_consumed_{0}; // by Inference Engine

    // --- STAGE 3: Inference Engine -> Logic/Overlay ---
    std::atomic<int64_t> inference_results_produced_{0};
    std::atomic<int64_t> inference_results_dropped_{0};
    std::atomic<int64_t> inference_results_consumed_by_logic_{0};
    std::atomic<int64_t> inference_results_consumed_by_overlay_{0};

    std::atomic<int64_t> h264_output_queue_in_{0};
    std::atomic<int64_t> h264_output_queue_out_{0};

    // Getter methods for Monitor class
    const std::unique_ptr<ImageProcessor>& get_image_processor() const { return image_processor_; }
    const std::unique_ptr<ImageProcessor>& get_visualization_processor() const { return visualization_processor_; }
    const std::unique_ptr<InferenceEngine>& get_inference_engine() const { return inference_engine_; }
    const std::unique_ptr<CameraCapture>& get_primary_camera() const { return primary_camera_; }
    const std::unique_ptr<H264Encoder>& get_h264_encoder() const { return h264_encoder_; }
    const std::shared_ptr<OrientationSensor>& get_orientation_sensor() const { return orientation_sensor_; }
    const std::unique_ptr<LogicModule>& get_logic_module() const { return logic_module_; }
    const std::unique_ptr<SystemMonitor>& get_system_monitor() const { return system_monitor_; }
    const std::unique_ptr<KeyboardMonitor>& get_keyboard_monitor() const { return keyboard_monitor_; }
    
    // Stage 1 Getters
    int64_t get_cam_to_viz_produced() const { return cam_to_viz_produced_.load(); }
    int64_t get_cam_to_viz_dropped() const { return cam_to_viz_dropped_.load(); }
    int64_t get_cam_to_viz_consumed() const { return cam_to_viz_consumed_.load(); }
    int64_t get_cam_to_tpu_proc_produced() const { return cam_to_tpu_proc_produced_.load(); }
    int64_t get_cam_to_tpu_proc_dropped() const { return cam_to_tpu_proc_dropped_.load(); }
    int64_t get_cam_to_tpu_proc_consumed() const { return cam_to_tpu_proc_consumed_.load(); }
    
    // Stage 2 Getters
    int64_t get_proc_to_inf_produced() const { return proc_to_inf_produced_.load(); }
    int64_t get_proc_to_inf_dropped() const { return proc_to_inf_dropped_.load(); }
    int64_t get_proc_to_inf_consumed() const { return proc_to_inf_consumed_.load(); }

    int64_t get_inference_results_produced() const { return inference_results_produced_.load(); }
    int64_t get_inference_results_dropped() const { return inference_results_dropped_.load(); }
    int64_t get_inference_results_consumed_by_logic() const { return inference_results_consumed_by_logic_.load(); }
    int64_t get_inference_results_consumed_by_overlay() const { return inference_results_consumed_by_overlay_.load(); }
    
    int64_t get_h264_output_queue_in() const { return h264_output_queue_in_.load(); }
    int64_t get_h264_output_queue_out() const { return h264_output_queue_out_.load(); }

    // Methods to update counters from modules
    void inc_cam_to_viz_produced() { cam_to_viz_produced_.fetch_add(1); }
    void inc_cam_to_viz_dropped() { cam_to_viz_dropped_.fetch_add(1); }
    void inc_cam_to_viz_consumed() { cam_to_viz_consumed_.fetch_add(1); }

    void inc_cam_to_tpu_proc_produced() { cam_to_tpu_proc_produced_.fetch_add(1); }
    void inc_cam_to_tpu_proc_dropped() { cam_to_tpu_proc_dropped_.fetch_add(1); }
    void inc_cam_to_tpu_proc_consumed() { cam_to_tpu_proc_consumed_.fetch_add(1); }

    void inc_proc_to_inf_produced() { proc_to_inf_produced_.fetch_add(1); }
    void inc_proc_to_inf_dropped() { proc_to_inf_dropped_.fetch_add(1); }
    void inc_proc_to_inf_consumed() { proc_to_inf_consumed_.fetch_add(1); }

    void increment_inference_results_produced(int count) { inference_results_produced_.fetch_add(count); }
    void increment_inference_results_dropped() { inference_results_dropped_.fetch_add(1); }
    void increment_inference_results_consumed_by_logic() { inference_results_consumed_by_logic_.fetch_add(1); }
    void increment_inference_results_consumed_by_overlay() { inference_results_consumed_by_overlay_.fetch_add(1); }

    void increment_h264_output_queue_in() { h264_output_queue_in_.fetch_add(1); }
    void increment_h264_output_queue_out() { h264_output_queue_out_.fetch_add(1); }

private:
};

#endif // APPLICATION_H