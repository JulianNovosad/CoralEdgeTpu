#ifndef APPLICATION_H
#define APPLICATION_H

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

#include "buffer_pool.h"

#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>

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

private:
    // Cleanup functions
    void pre_launch_cleanup();
    void post_shutdown_cleanup();
    bool terminate_existing_instances();
    void release_edge_tpu_resources();
    void release_camera_resources();
    void clear_telemetry_sockets();
    
    void setup_pools_and_queues();
    bool initialize_modules(const std::string& model_path, const std::string& labels_path);
    bool start_modules();
    void register_shutdown_handlers();
    void main_loop();
    
    // Recovery mechanisms
    void recovery_thread_func();
    bool restart_camera_subsystem();
    bool restart_inference_subsystem();
    bool restart_logic_subsystem();
    bool restart_image_processor_subsystem();
    bool restart_encoder_subsystem();
    bool restart_orientation_subsystem();
    
    int argc_;
    char** argv_;

    ConfigLoader config_loader_;
    ApplicationSupervisor supervisor_;
    
    // Buffer Pools
    std::shared_ptr<BufferPool<uint8_t>> image_pool_;
    std::shared_ptr<BufferPool<DetectionResult>> detection_pool_;
    std::shared_ptr<BufferPool<uint8_t>> h264_pool_;

    // Queues
    ImageQueue raw_image_for_processor_queue_; // New queue
    ImageQueue tpu_inference_queue_;

    DetectionResultsQueue detection_results_for_overlay_queue_;
    DetectionResultsQueue detection_results_for_logic_queue_;
    ImageQueue overlaid_video_queue_;
    H264Queue h264_output_queue_;

    // Thread for consuming overlay detection results
    std::thread overlay_consumer_thread_;
    std::atomic<bool> overlay_consumer_running_{false};
    void overlay_queue_consumer_thread_func();
    std::list<std::reference_wrapper<ImageQueue>> main_image_output_queues_;

    // Modules
    std::unique_ptr<ImageProcessor> image_processor_; // New module
    std::unique_ptr<InferenceEngine> inference_engine_;
    std::unique_ptr<CameraCapture> primary_camera_;
    // std::unique_ptr<VideoOverlayProcessor> overlay_processor_;
    std::unique_ptr<H264Encoder> h264_encoder_;
    std::shared_ptr<OrientationSensor> orientation_sensor_;
    std::unique_ptr<LogicModule> logic_module_;
    std::unique_ptr<SystemMonitor> system_monitor_;
    std::unique_ptr<KeyboardMonitor> keyboard_monitor_;

    // Recovery mechanisms
    std::atomic<bool> recovery_running_{false};
    std::atomic<bool> recovery_enabled_{false}; // New flag to control when recovery is active
    std::thread recovery_thread_;
    std::mutex recovery_mutex_;
    
    // Recovery counters for each subsystem
    std::map<std::string, int> recovery_attempts_;
    const int max_recovery_attempts_ = 5; // Maximum attempts per second

    std::vector<std::string> labels_;
};

#endif // APPLICATION_H