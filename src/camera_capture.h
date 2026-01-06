// Verified headers: [thread, atomic, vector, string, chrono...]
// Verification timestamp: 2026-01-06 17:08:04
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
     * @param tpu_fps Frame rate voor de TPU-stream.
     * @param target_tpu_width Doelbreedte voor de TPU-inferentie.
     * @param target_tpu_height Doelhoogte voor de TPU-inferentie.
     * @param image_buffer_pool Een gedeelde pool voor het beheren van image buffers.
     * @param main_output_queues Een lijst van wachtrijen voor de BGR-frames van de hoofdstream.
     * @param tpu_output_queue Een wachtrij voor de BGR-frames bestemd voor de TPU.
     * @param watchdog_timeout Timeout voor de camera-watchdog.
     */
    CameraCapture(unsigned int main_width, unsigned int main_height,
                                                                            unsigned int tpu_width, unsigned int tpu_height,
                                                                            unsigned int tpu_fps,
                                                                            unsigned int target_tpu_width, unsigned int target_tpu_height,
                                                                            std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool,
                                                                            std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                                                                            ImageQueue& image_processor_input_queue,
                                                                            std::chrono::seconds watchdog_timeout);
                                               
                                                   void set_main_output_queues(const std::vector<ImageQueue*>& queues) { main_output_queues_ = queues; }
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
    
    // Timing methods for monitoring
    long long get_capture_timing_us() const { return avg_capture_time_us_; }
    long long get_total_loop_timing_us() const { return avg_total_loop_time_us_; }
    
private:
    // Timing statistics
    mutable std::atomic<long long> avg_capture_time_us_{0};
    mutable std::atomic<long long> avg_total_loop_time_us_{0};

    unsigned int width_; ///< Breedte van de hoofdstream.
    unsigned int height_; ///< Hoogte van de hoofdstream.
    unsigned int tpu_width_; ///< Breedte van de TPU-stream.
    unsigned int tpu_height_; ///< Hoogte van de TPU-stream.
    unsigned int tpu_fps_; ///< Frame rate voor de TPU-stream.

    unsigned int target_tpu_width_; ///< Doelbreedte voor TPU-inferentie na resizing.
    unsigned int target_tpu_height_; ///< Doelhoogte voor TPU-inferentie na resizing.

    std::vector<ImageQueue*> main_output_queues_;  ///< Wachtrijen voor BGR-frames bestemd voor de live stream.
    ImageQueue& image_processor_input_queue_;  ///< Wachtrij voor ruwe frames bestemd voor de ImageProcessor.
    std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool_; ///< Pool voor het beheren van image buffers.
    std::shared_ptr<ObjectPool<ImageData>> image_data_pool_; ///< Pool for ImageData objects.
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
    
    // Freshness indicators
public:
    std::atomic<long long> last_frame_timestamp_{0}; ///< Timestamp of the last processed frame
    std::atomic<int> frame_rate_{0}; ///< Current frame rate
    
    // Drop counters for proper queue accounting
    std::atomic<int64_t> main_stream_drop_count_{0}; ///< Count of frames dropped from main stream queue
    std::atomic<int64_t> tpu_stream_drop_count_{0};  ///< Count of frames dropped from TPU stream queue
    
    // Public getters for drop counters to be used by Monitor
    int64_t get_main_stream_drop_count() const { return main_stream_drop_count_.load(); }
    int64_t get_tpu_stream_drop_count() const { return tpu_stream_drop_count_.load(); }
    
    // Method to allow application to increment drop counters when draining queues
    void increment_main_stream_drop_count() { main_stream_drop_count_.fetch_add(1); }
    void increment_tpu_stream_drop_count() { tpu_stream_drop_count_.fetch_add(1); }
    
    // Public getters for frame accounting counters
    int64_t get_frames_produced() const { return frames_produced_.load(); }
    int64_t get_frames_consumed_by_inference() const { return frames_consumed_by_inference_.load(); }
    
    // Method to set application reference for updating counters
    void set_application_ref(class Application* app) { app_ref_ = app; }
    
    std::thread& get_request_processor_thread() { return request_processor_thread_; }
private:
    
    // Frame accounting counters
    std::atomic<int64_t> frames_produced_{0};
    std::atomic<int64_t> frames_consumed_by_inference_{0};
    
    // Application reference for updating counters
    class Application* app_ref_ = nullptr;

    std::chrono::steady_clock::time_point last_frame_time_; ///< Tijdstip van het laatst verwerkte frame.
    int frame_count_ = 0; ///< Teller voor het aantal verwerkte frames.
    
    // FPS measurement variables
    std::chrono::steady_clock::time_point first_frame_time_; ///< Tijdstip van het eerste frame voor FPS calculation
    int fps_measurement_frames_ = 0; ///< Counter for FPS measurement frames

    std::unique_ptr<cv::VideoWriter> video_writer_;  ///< H.264 encoder (momenteel niet gebruikt).
    std::function<void(cv::Mat& frame)> overlay_callback_;  ///< Callback voor overlays.

    // Members for dedicated request processing thread
    std::thread request_processor_thread_;
    std::queue<libcamera::Request*> request_queue_;
    std::mutex request_queue_mutex_;
    std::condition_variable request_queue_cond_var_;
    std::atomic<bool> processing_running_ = false;

    int skip_initial_measurements_ = 20; // Number of initial frames to skip for performance metrics

private:
    struct MappedBuffer {
        void* addr;
        size_t length;
    };
    std::map<const libcamera::FrameBuffer*, MappedBuffer> mapped_buffers_;

    void request_processor_thread_func(); // New thread function
    // New helper to process processed TPU frames
        bool process_tpu_processed_frame_buffer(const libcamera::FrameBuffer* fb,
                                           const libcamera::StreamConfiguration& cfg,
                                           std::chrono::steady_clock::time_point capture_time,
                                           uint64_t t_capture_raw_ms,
                                           long long frame_id,
                                           long long exposure_ms,
                                           uint64_t sensor_ts_ns);};

#endif // CAMERA_CAPTURE_H