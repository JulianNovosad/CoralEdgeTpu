#ifndef APPLICATION_H
#define APPLICATION_H

#include "application_supervisor.h"
#include "camera_capture.h"
#include "config_loader.h"
#include "h264_encoder.h"
#include "http_server.h"
#include "inference.h"
#include "logic.h"
#include "orientation_sensor.h"
#include "pipeline_structs.h"
#include "system_monitor.h"
#include "video_overlay_processor.h"
#include "buffer_pool.h"

#include <memory>
#include <vector>
#include <string>

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
    void setup_pools_and_queues();
    bool initialize_modules(const std::string& model_path, const std::string& labels_path);
    bool start_modules();
    void register_shutdown_handlers();
    void main_loop();
    
    int argc_;
    char** argv_;

    ConfigLoader config_loader_;
    ApplicationSupervisor supervisor_;
    
    // Buffer Pools
    std::shared_ptr<BufferPool<uint8_t>> image_pool_;
    std::shared_ptr<BufferPool<DetectionResult>> detection_pool_;
    std::shared_ptr<BufferPool<uint8_t>> h264_pool_;

    // Queues
    ImageQueue tpu_inference_queue_;
    ImageQueue main_camera_output_queue_;
    DetectionResultsQueue detection_results_for_overlay_queue_;
    DetectionResultsQueue detection_results_for_logic_queue_;
    ImageQueue overlaid_video_queue_;
    H264Queue h264_output_queue_;

    // Modules
    std::unique_ptr<InferenceEngine> inference_engine_;
    std::unique_ptr<CameraCapture> primary_camera_;
    std::unique_ptr<VideoOverlayProcessor> overlay_processor_;
    std::unique_ptr<H264Encoder> h264_encoder_;
    std::unique_ptr<HttpServer> http_server_;
    std::shared_ptr<OrientationSensor> orientation_sensor_;
    std::unique_ptr<LogicModule> logic_module_;
    std::unique_ptr<SystemMonitor> system_monitor_;

    std::vector<std::string> labels_;
};

#endif // APPLICATION_H
