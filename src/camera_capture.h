#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <libcamera/libcamera.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/stream.h>
#include <libcamera/request.h>
#include <libcamera/geometry.h>
#include <libcamera/pixel_format.h>

#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <chrono>
#include <list>
#include <memory>
#include <functional>

#include <opencv2/opencv.hpp>

#include "pipeline_structs.h"
#include "buffer_pool.h"

/**
 * @file camera_capture.h
 * @brief Beheert de video-opnamepijplijn met libcamera.
 *
 * Deze klasse is verantwoordelijk voor het initialiseren van de camera,
 * het configureren van videostreams (zowel voor weergave als voor TPU-inferentie),
 * en het vastleggen van frames in een multi-threaded omgeving.
 */
class CameraCapture {
public:
    /**
     * @brief Constructor voor de CameraCapture klasse.
     * @param main_width Breedte van de hoge-resolutie videostream.
     * @param main_height Hoogte van de hoge-resolutie videostream.
     * @param tpu_width Breedte van de lage-resolutie stream voor de TPU.
     * @param tpu_height Hoogte van de lage-resolutie stream voor de TPU.
     * @param target_tpu_width Doelbreedte voor de TPU-inferentie.
     * @param target_tpu_height Doelhoogte voor de TPU-inferentie.
     * @param image_buffer_pool Een gedeelde pool voor het beheren van image buffers.
     * @param main_output_queues Een lijst van wachtrijen voor de BGR-frames van de hoofdstream.
     * @param tpu_output_queue Een wachtrij voor de BGR-frames bestemd voor de TPU.
     * @param watchdog_timeout Timeout voor de camera-watchdog.
     */
    CameraCapture(unsigned int main_width, unsigned int main_height,
                  unsigned int tpu_width, unsigned int tpu_height,
                  unsigned int target_tpu_width, unsigned int target_tpu_height,
                  std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool,
                  std::list<std::reference_wrapper<ImageQueue>>& main_output_queues,
                  ImageQueue& tpu_output_queue,
                  std::chrono::seconds watchdog_timeout);
    ~CameraCapture();

    /**
     * @brief Start de video-opname thread.
     * @return True als het starten succesvol was, anders false.
     */
    bool start();

    /**
     * @brief Stopt de video-opname thread.
     */
    void stop();

    /**
     * @brief Controleert of de opname-engine draait.
     * @return True als de engine draait, anders false.
     */
    bool is_running() const { return running_; }

    /**
     * @brief Logt de huidige status van de camera-instellingen.
     */
    void get_state() const;

    /**
     * @brief Berekent en logt prestatie-indicatoren zoals FPS en latentie.
     */
    void get_performance_metrics();

    /**
     * @brief Configureert de camera met de opgegeven stream-instellingen.
     * @return True als de configuratie succesvol was, anders false.
     */
    bool setup_camera();

    /**
     * @brief Callback-functie die wordt aangeroepen wanneer een frame-request is voltooid.
     * @param request De voltooide request van libcamera.
     */
    void request_complete_callback(libcamera::Request* request);

    /**
     * @brief Accepteert en configureert de eerste beschikbare camera.
     * @return True als een camera succesvol is geconfigureerd, anders false.
     */
    bool acquire_camera();

    /**
     * @brief Stelt een callback in voor het toevoegen van overlays op de frames.
     * @param callback De functie die wordt aangeroepen met een cv::Mat frame als argument.
     */
    bool init_video_encoder();
    void set_overlay_callback(std::function<void(cv::Mat& frame)> callback) {
        overlay_callback_ = callback;
    }

    unsigned int width_; ///< Breedte van de hoofdstream.
    unsigned int height_; ///< Hoogte van de hoofdstream.
    unsigned int tpu_width_; ///< Breedte van de TPU-stream.
    unsigned int tpu_height_; ///< Hoogte van de TPU-stream.

    unsigned int target_tpu_width_; ///< Doelbreedte voor TPU-inferentie na resizing.
    unsigned int target_tpu_height_; ///< Doelhoogte voor TPU-inferentie na resizing.

    std::list<std::reference_wrapper<ImageQueue>>& main_output_queues_;  ///< Wachtrijen voor BGR-frames bestemd voor de live stream.
    ImageQueue& tpu_output_queue_;  ///< Wachtrij voor BGR-frames bestemd voor de TPU.
    std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool_; ///< Pool voor het beheren van image buffers.
    std::chrono::seconds watchdog_timeout_; ///< Timeout voor de camera-watchdog.

    std::unique_ptr<libcamera::CameraManager> camera_manager_; ///< Beheert de beschikbare camera's.
    std::shared_ptr<libcamera::Camera> camera_; ///< De geselecteerde camera.
    libcamera::Stream* video_stream_ = nullptr; ///< De hoge-resolutie videostream.
    libcamera::Stream* tpu_stream_ = nullptr; ///< De lage-resolutie TPU-stream.
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_; ///< Alloceert frame-buffers.
    std::vector<std::unique_ptr<libcamera::Request>> requests_; ///< Vector van camera requests.
    
    libcamera::PixelFormat actual_pixel_format_; ///< Het daadwerkelijke pixelformaat geconfigureerd door libcamera.
    libcamera::Size actual_size_; ///< De daadwerkelijke resolutie geconfigureerd door libcamera.
    unsigned int actual_stride_ = 0; ///< De daadwerkelijke stride van de frame buffer.

    std::atomic<bool> running_ = false; ///< Vlag om de state van de worker thread te beheren.

    std::chrono::time_point<std::chrono::high_resolution_clock> last_frame_time_; ///< Tijdstip van het laatst verwerkte frame.
    int frame_count_ = 0; ///< Teller voor het aantal verwerkte frames.

    std::unique_ptr<cv::VideoWriter> video_writer_;  ///< H.264 encoder (momenteel niet gebruikt).
    std::function<void(cv::Mat& frame)> overlay_callback_;  ///< Callback voor overlays.

    // Leden voor prestatiemetingen
    std::vector<long long> frame_latencies_ms_; ///< Vector om latenties van frames op te slaan.
    std::mutex frame_latencies_mutex_; ///< Mutex voor thread-veilige toegang tot de latencies vector.
    long long total_frames_processed_ = 0; ///< Totaal aantal verwerkte frames voor prestatieberekening.

private:
    void process_main_video_stream(libcamera::Request* request, std::chrono::high_resolution_clock::time_point capture_timestamp);
    void process_tpu_inference_stream(libcamera::Request* request, std::chrono::high_resolution_clock::time_point capture_timestamp);
    void update_frame_metrics(std::chrono::high_resolution_clock::time_point processing_start_time, std::chrono::high_resolution_clock::time_point processing_end_time);
    void requeue_camera_request(libcamera::Request* request);
    bool process_frame_buffer_helper(const libcamera::FrameBuffer* fb,
                                     const libcamera::StreamConfiguration& cfg,
                                     ImageQueue& queue,
                                     const char* stream_name,
                                     unsigned int target_width,
                                     unsigned int target_height,
                                     std::chrono::high_resolution_clock::time_point capture_timestamp);

};

#endif // CAMERA_CAPTURE_H
