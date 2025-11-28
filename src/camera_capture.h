#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <libcamera/libcamera.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/stream.h>
#include <libcamera/request.h>
#include <libcamera/geometry.h>
#include <libcamera/pixel_format.h> // For libcamera::PixelFormat and libcamera::formats

#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <chrono>
#include <list>
#include <memory> // For std::unique_ptr

#include "pipeline_structs.h" // For ImageQueue

/**
 * @brief Manages camera capture using the low-level libcamera C++ API.
 *
 * This class directly interacts with libcamera to open, configure, and stream
 * frames from the camera. It converts YUV420 sensor data to RGB888 before
 * pushing frames into shared ImageQueue instances.
 */
class CameraCapture {
public:
    CameraCapture(unsigned int main_width, unsigned int main_height, unsigned int tpu_width, unsigned int tpu_height, std::list<std::reference_wrapper<ImageQueue>>& main_output_queues, ImageQueue& tpu_output_queue, std::chrono::seconds watchdog_timeout);
    ~CameraCapture();
    bool start();
    void stop();
    bool is_running() const { return running_; }
    void get_state() const;
    bool setup_camera();
    void request_complete_callback(libcamera::Request* request);



    unsigned int width_;  ///< Desired output width (RGB888).
    unsigned int height_; ///< Desired output height (RGB888).
    unsigned int tpu_width_;
    unsigned int tpu_height_;
    std::list<std::reference_wrapper<ImageQueue>>& main_output_queues_; ///< Queues for RGB888 ImageData.
    ImageQueue& tpu_output_queue_; // New queue for TPU stream output
    std::chrono::seconds watchdog_timeout_; ///< Not directly used by libcamera API, kept for compatibility.

    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    std::shared_ptr<libcamera::Camera> camera_;
    libcamera::Stream* video_stream_ = nullptr; ///< The YUV420 video stream.
    libcamera::Stream* tpu_stream_ = nullptr;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
    std::vector<std::unique_ptr<libcamera::Request>> requests_; ///< Store requests created.
    std::vector<libcamera::Request*> returned_requests_; // Stores raw pointers to requests returned during shutdown for explicit destruction

    libcamera::PixelFormat actual_pixel_format_; ///< Actual pixel format provided by the camera.
    libcamera::Size actual_size_; ///< Actual size provided by the camera.
    unsigned int actual_stride_ = 0; ///< Actual stride of the YUV420 frame data.

    std::atomic<bool> running_ = false;

    // For FPS calculation
    std::chrono::time_point<std::chrono::high_resolution_clock> last_frame_time_;
    int frame_count_ = 0;
    static const int kFpsReportInterval = 100; // Report FPS every 100 frames

    // For YUV conversion context.
};

#endif // CAMERA_CAPTURE_H
