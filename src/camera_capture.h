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

class CameraCapture {
public:
    CameraCapture(unsigned int main_width, unsigned int main_height,
                  unsigned int tpu_width, unsigned int tpu_height,
                  unsigned int target_tpu_width, unsigned int target_tpu_height,
                  std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool,
                  std::list<std::reference_wrapper<ImageQueue>>& main_output_queues,
                  ImageQueue& tpu_output_queue,
                  std::chrono::seconds watchdog_timeout);
    ~CameraCapture();

    bool start();
    void stop();
    bool is_running() const { return running_; }
    void get_state() const;
    void get_performance_metrics();
    bool setup_camera();
    void request_complete_callback(libcamera::Request* request);

    bool acquire_camera();

    bool init_video_encoder(const std::string& output_uri, int fps);

    void set_overlay_callback(std::function<void(cv::Mat& frame)> callback) {
        overlay_callback_ = callback;
    }

    unsigned int width_;
    unsigned int height_;
    unsigned int tpu_width_;
    unsigned int tpu_height_;

    unsigned int target_tpu_width_;
    unsigned int target_tpu_height_;

    std::list<std::reference_wrapper<ImageQueue>>& main_output_queues_;  // RGB888 frames for live stream
    ImageQueue& tpu_output_queue_;  // RGB888 frames for TPU
    std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool_; // Pool for image buffers
    std::chrono::seconds watchdog_timeout_;

    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    std::shared_ptr<libcamera::Camera> camera_;
    libcamera::Stream* video_stream_ = nullptr;
    libcamera::Stream* tpu_stream_ = nullptr;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
    std::vector<std::unique_ptr<libcamera::Request>> requests_;
    // Removed: std::vector<libcamera::Request*> returned_requests_; as it was unused and non-critical.

    libcamera::PixelFormat actual_pixel_format_;
    libcamera::Size actual_size_;
    unsigned int actual_stride_ = 0;

    std::atomic<bool> running_ = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> last_frame_time_;
    int frame_count_ = 0;
    static const int kFpsReportInterval = 100;

    std::unique_ptr<cv::VideoWriter> video_writer_;  // H.264 encoder
    std::function<void(cv::Mat& frame)> overlay_callback_;  // Overlay callback

    // Performance measurement members
    std::vector<long long> frame_latencies_ms_;
    std::mutex frame_latencies_mutex_;
    long long total_frames_processed_ = 0;

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
