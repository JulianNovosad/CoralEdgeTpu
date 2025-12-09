#include <string>    // For std::string, std::to_string
#include <vector>    // For std::vector
#include <memory>    // For std::unique_ptr, std::make_unique, std::shared_ptr
#include <utility>   // For std::move
#include <chrono>    // For std::chrono
#include <sstream>   // For std::stringstream
#include <cstring>   // For memcpy
#include <sys/mman.h> // For mmap, munmap
#include <unistd.h>  // For close
#include <errno.h>   // For errno, strerror

#include "camera_capture.h"
#include "util_logging.h"

#include <opencv2/opencv.hpp>

#include <libcamera/property_ids.h>
#include <libcamera/control_ids.h>

#include <iostream>
#include <map>
#include <iomanip>

// Helper to convert libcamera PixelFormat to a string for logging
static std::string pixelFormatToString(const libcamera::PixelFormat& format) {
    // Use fourcc directly as it's a stable identifier.
    std::stringstream ss;
    ss << "'" << std::hex << std::setfill('0') << std::setw(8) << format.fourcc() << "'";
    return ss.str();
}

static bool process_frame_buffer(const libcamera::FrameBuffer* fb,
                                 const libcamera::StreamConfiguration& cfg,
                                 std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                                 ImageQueue& queue,
                                 const char* stream_name,
                                 unsigned int target_width,
                                 unsigned int target_height,
                                 std::chrono::high_resolution_clock::time_point capture_timestamp) {
    if (fb->planes().empty()) {
        LOG_ERROR(std::string(stream_name) + " FrameBuffer has no planes.");
        return false;
    }

    const libcamera::FrameBuffer::Plane& plane = fb->planes()[0];
    int fd = plane.fd.get();
    size_t length = plane.length;

    // 1. Acquire a buffer from the pool.
    auto bgr_pooled_buffer = buffer_pool->acquire();
    if (!bgr_pooled_buffer) {
        LOG_WARNING(std::string(stream_name) + " failed to acquire a buffer for BGR image. Dropping frame.");
        return false;
    }

    if (length > bgr_pooled_buffer->data.capacity()) {
        LOG_ERROR(std::string(stream_name) + " BGR frame size (" + std::to_string(length) +
                  ") exceeds buffer pool capacity (" + std::to_string(bgr_pooled_buffer->data.capacity()) + "). Dropping frame.");
        return false;
    }
    
    // 2. Mmap and copy the single BGR plane.
    void* mmap_ptr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) {
        LOG_ERROR(std::string(stream_name) + " Failed to mmap plane buffer: " + std::string(strerror(errno)));
        return false;
    }
    std::memcpy(bgr_pooled_buffer->data.data(), mmap_ptr, length);
    munmap(mmap_ptr, length);
    bgr_pooled_buffer->size = length;

    ImageData image_data(capture_timestamp);
    image_data.width = cfg.size.width;
    image_data.height = cfg.size.height;
    image_data.buffer = std::move(bgr_pooled_buffer);

    // 3. Resize if necessary.
    if (image_data.width != target_width || image_data.height != target_height) {
        // This part remains the same as it already handles BGR images.
        cv::Mat original_image_bgr(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
        cv::Mat resized_image_bgr;
        cv::resize(original_image_bgr, resized_image_bgr, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

        auto resized_bgr_pooled_buffer = buffer_pool->acquire();
        if (!resized_bgr_pooled_buffer) {
            LOG_WARNING(std::string(stream_name) + " failed to acquire a buffer for resized image. Dropping frame.");
            return false;
        }

        size_t resized_bgr_size = resized_image_bgr.total() * resized_image_bgr.elemSize();
        if (resized_bgr_size > resized_bgr_pooled_buffer->data.capacity()) {
            LOG_WARNING(std::string(stream_name) + " resized BGR image size (" + std::to_string(resized_bgr_size) +
                        ") exceeds buffer capacity (" + std::to_string(resized_bgr_pooled_buffer->data.capacity()) + "). Dropping frame.");
            return false;
        }
        std::memcpy(resized_bgr_pooled_buffer->data.data(), resized_image_bgr.data, resized_bgr_size);
        resized_bgr_pooled_buffer->size = resized_bgr_size;

        image_data.width = target_width;
        image_data.height = target_height;
        image_data.buffer = std::move(resized_bgr_pooled_buffer);
    }

    // 4. Push data to the queue.
    LOG_DEBUG("Pushing " + std::string(stream_name) + " to queue. Final dimensions: " +
              std::to_string(image_data.width) + "x" + std::to_string(image_data.height));
    queue.push(std::move(image_data));

    return true;
}


CameraCapture::CameraCapture(unsigned int main_width, unsigned int main_height,
                             unsigned int tpu_width, unsigned int tpu_height,
                             unsigned int target_tpu_width, unsigned int target_tpu_height,
                             std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool,
                             std::list<std::reference_wrapper<ImageQueue>>& main_output_queues,
                             ImageQueue& tpu_output_queue,
                             std::chrono::seconds watchdog_timeout)
    : width_(main_width),
      height_(main_height),
      tpu_width_(tpu_width),
      tpu_height_(tpu_height),
      target_tpu_width_(target_tpu_width),
      target_tpu_height_(target_tpu_height),
      main_output_queues_(main_output_queues),
      tpu_output_queue_(tpu_output_queue),
      image_buffer_pool_(image_buffer_pool),
      watchdog_timeout_(watchdog_timeout),
      camera_manager_(std::make_unique<libcamera::CameraManager>()), // Initialize here
      camera_(nullptr),
      allocator_(nullptr),
      running_(false),
      last_frame_time_(std::chrono::high_resolution_clock::now()),
      total_frames_processed_(0) {
    
    int ret = camera_manager_->start();
    if (ret) {
        throw std::runtime_error("Failed to start CameraManager: " + std::to_string(ret));
    }
    LOG_INFO("Libcamera CameraManager started.");
}

CameraCapture::~CameraCapture() {
    stop();
    if (camera_manager_) {
        camera_manager_->stop();
        LOG_INFO("Libcamera CameraManager stopped.");
    }
}

/**
 * @brief Selects and acquires the camera device.
 * @return True on success, False on failure.
 */
bool CameraCapture::acquire_camera() {
    auto cameras = camera_manager_->cameras();
    if (cameras.empty()) {
        LOG_ERROR("Acquire: FAILURE - No cameras found by libcamera. Check camera connection and driver.");
        return false;
    }

    // Assign the selected camera to the member variable camera_
    camera_ = cameras.front();
    if (!camera_) {
        LOG_ERROR("Acquire: FAILURE - CameraManager returned a null camera pointer for first camera.");
        return false;
    }

    LOG_INFO("Selected Camera ID: " + camera_->id());
    
    int ret = camera_->acquire();
    if (ret) {
        LOG_ERROR("Acquire: FAILURE - Failed to acquire camera (Error: " + std::to_string(ret) + "). Is the camera in use by another process?");
        camera_.reset();
        return false;
    }
    LOG_INFO("Acquire: Camera acquired successfully.");
    return true;
}

bool CameraCapture::start() {
    if (running_) {
        LOG_ERROR("CameraCapture is already running.");
        return false;
    }
    
    // 1. Acquire the camera 
    if (!acquire_camera()) {
        // This log should be printed if acquire_camera() returns false. 
        // If this log is missing, it suggests the program is skipping checks or execution flow is flawed.
        LOG_ERROR("CameraCapture failed to acquire camera. Cannot proceed.");
        return false;
    }

    // CRITICAL SECONDARY CHECK: Should not be necessary, but defensively prevents a nullptr crash.
    if (!camera_) {
        LOG_ERROR("CRITICAL: Camera pointer is null immediately after successful acquire_camera(). Aborting.");
        return false;
    }
    
    // Log to confirm successful state before moving to setup
    LOG_INFO("Camera acquired and confirmed to be non-null. Proceeding to setup.");

    // 2. Setup (Configure streams, allocate buffers, create requests)
    if (!setup_camera()) {
        LOG_ERROR("Failed to setup camera. Releasing camera.");
        
        // Safety check before release(): Only release if the pointer is still valid.
        if (camera_) {
            camera_->release();
        }
        camera_.reset();
        return false;
    }
    
    // 3. Connect request completed signal
    camera_->requestCompleted.connect(this, &CameraCapture::request_complete_callback);

    // 4. Start Capture
    int ret = camera_->start();
    if (ret) {
        LOG_ERROR("Failed to start camera: " + std::to_string(ret));
        camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);
        camera_->release();
        camera_.reset();
        return false;
    }
    LOG_INFO("Libcamera camera started.");

    running_ = true;
    
    // 5. Queue initial requests
    frame_count_ = 0;
    last_frame_time_ = std::chrono::high_resolution_clock::now();
    
    for (auto& req_ptr : requests_) {
        if (camera_->queueRequest(req_ptr.get())) { 
            LOG_ERROR("Failed to queue initial request.");
            // Stop to clean up resources if queueing fails.
            running_ = false; 
            stop(); 
            return false;
        }
    }
    LOG_INFO("CameraCapture: Initial requests queued.");
    return true;
}

void CameraCapture::stop() {
    // 1. Signal shutdown to threads
    if (!running_.exchange(false)) {
        return;
    }
    LOG_INFO("Stopping CameraCapture...");

    if (camera_) {
        // 3. CRITICAL: Disconnect the callback first.
        camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);

        // 4. Stop the camera. This blocks until all pending requests return (flushing).
        camera_->stop();
        LOG_INFO("Libcamera camera stopped.");
        
        camera_->release();
        LOG_INFO("Libcamera camera released.");
    }
    
    // 5. Free buffers
    if (allocator_) {
        if (video_stream_) allocator_->free(video_stream_);
        if (tpu_stream_) allocator_->free(tpu_stream_);
        allocator_.reset();
        LOG_INFO("Libcamera buffers freed.");
    }
    
    // 6. NOW it is safe to destroy the requests and camera shared_ptr.
    requests_.clear(); // Explicit clear for Request objects
    camera_.reset();
    LOG_INFO("Requests and camera pointer cleared.");

    LOG_INFO("CameraCapture stopped.");
}

bool CameraCapture::setup_camera() {
    // --- SAFETY CHECK ---
    if (!camera_) {
        // This is the log line the user reported. It means the pointer was null here.
        LOG_ERROR("CRITICAL: setup_camera called but camera_ is nullptr! Aborting setup.");
        return false;
    }
    
    LOG_INFO("setup_camera: Configuring camera: " + camera_->id());
    int ret = 0; 

    // Configure dual streams: main high-res and TPU viewfinder
    std::vector<libcamera::StreamRole> roles = {
        libcamera::StreamRole::VideoRecording, // main high-res
        libcamera::StreamRole::Viewfinder      // TPU/resized
    };
    
    std::unique_ptr<libcamera::CameraConfiguration> config = camera_->generateConfiguration(roles);

    if (!config) {
        LOG_ERROR("Failed to generate dual stream configuration.");
        return false;
    }

    // Check if the generated configuration contains exactly two streams
    if (config->size() < 2) {
        LOG_ERROR("Generated camera configuration has less than two streams for dual-stream setup (found " + std::to_string(config->size()) + ").");
        return false;
    }

    // Configure main stream (index 0)
    libcamera::StreamConfiguration& mainCfg = config->at(0);
    mainCfg.pixelFormat = libcamera::formats::BGR888;
    mainCfg.size.width = width_;
    mainCfg.size.height = height_;

    // Configure tpu stream (index 1)
    libcamera::StreamConfiguration& tpuCfg = config->at(1);
    tpuCfg.pixelFormat = libcamera::formats::BGR888;
    tpuCfg.size.width = tpu_width_;
    tpuCfg.size.height = tpu_height_;
    
    // Validate and complete the configuration
    libcamera::CameraConfiguration::Status config_status = config->validate();
    LOG_INFO(std::string("CameraConfiguration validate() -> Status: ") + std::to_string(config_status));
    
    if (config_status == libcamera::CameraConfiguration::Invalid) {
        LOG_ERROR("Invalid dual stream camera configuration. Check requested resolutions/formats.");
        return false;
    } else if (config_status == libcamera::CameraConfiguration::Adjusted) {
        LOG_WARNING("Camera configuration adjusted by libcamera for dual streams.");
    }
    
    // Log final chosen config for each stream
    for (unsigned i = 0; i < config->size(); ++i) {
        std::string log_msg = "Final stream[" + std::to_string(i) + "] size=" +
                              std::to_string(config->at(i).size.width) + "x" +
                              std::to_string(config->at(i).size.height) + " fmt=" +
                              pixelFormatToString(config->at(i).pixelFormat);
        LOG_INFO(log_msg);
    }

    // Store the actual configured stream properties for the main video stream
    actual_pixel_format_ = mainCfg.pixelFormat;
    actual_size_ = mainCfg.size;
    actual_stride_ = mainCfg.stride; // Get the stride, important for raw data.

    LOG_INFO("CameraCapture: Configured main stream format: " + pixelFormatToString(actual_pixel_format_) + 
             ", size: " + std::to_string(actual_size_.width) + "x" + std::to_string(actual_size_.height) +
             ", stride: " + std::to_string(actual_stride_) + " (FOURCC: " + std::to_string(actual_pixel_format_.fourcc()) + ")");
    
    ret = camera_->configure(config.get());
    if (ret) {
        LOG_ERROR("Failed to configure camera streams (Error: " + std::to_string(ret) + ").");
        return false;
    }
    LOG_INFO("Libcamera dual streams configured.");
    
    video_stream_ = mainCfg.stream();
    tpu_stream_ = tpuCfg.stream();

    if (video_stream_) {
        LOG_INFO("Actual Video Stream Config: " + std::to_string(video_stream_->configuration().size.width) + "x" + std::to_string(video_stream_->configuration().size.height) + " format: " + pixelFormatToString(video_stream_->configuration().pixelFormat));
    } else {
        LOG_ERROR("Video stream is null after configuration.");
    }

    if (tpu_stream_) {
        LOG_INFO("Actual TPU Stream Config: " + std::to_string(tpu_stream_->configuration().size.width) + "x" + std::to_string(tpu_stream_->configuration().size.height) + " format: " + pixelFormatToString(tpu_stream_->configuration().pixelFormat));
    } else {
        LOG_ERROR("TPU stream is null after configuration.");
    }

    // Allocate buffers for main video stream
    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    ret = allocator_->allocate(video_stream_);
    if (ret < 0) {
        LOG_ERROR("Failed to allocate buffers for main video stream (Error: " + std::to_string(ret) + ").");
        allocator_.reset(); // Ensure allocator is cleaned up if it failed early
        return false;
    }
    LOG_INFO("Libcamera buffers allocated for main video stream. Number of buffers: " + std::to_string(allocator_->buffers(video_stream_).size()));

    // Allocate buffers for TPU stream
    ret = allocator_->allocate(tpu_stream_);
    if (ret < 0) {
        LOG_ERROR("Failed to allocate buffers for TPU stream (Error: " + std::to_string(ret) + ").");
        allocator_.reset();
        return false;
    }
    LOG_INFO("Libcamera buffers allocated for TPU stream. Number of buffers: " + std::to_string(allocator_->buffers(tpu_stream_).size()));

    // Create requests
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& video_buffers = allocator_->buffers(video_stream_);
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& tpu_buffers = allocator_->buffers(tpu_stream_);

    if (video_buffers.size() != tpu_buffers.size()) {
        LOG_ERROR("Mismatched buffer counts between main video stream and TPU stream. This is unexpected.");
        return false;
    }
    
    requests_.clear(); // Ensure clean slate before populating
    
    for (unsigned int i = 0; i < video_buffers.size(); ++i) {
        std::unique_ptr<libcamera::Request> request = camera_->createRequest();
        if (!request) {
            LOG_ERROR("Failed to create request.");
            return false;
        }

        // Set initial controls
        request->controls().set(libcamera::controls::AeEnable, true);
        
        ret = request->addBuffer(video_stream_, video_buffers[i].get()); 
        if (ret) {
            LOG_ERROR("Failed to add main buffer to request (Error: " + std::to_string(ret) + ").");
            return false;
        }

        ret = request->addBuffer(tpu_stream_, tpu_buffers[i].get());
        if (ret) {
            LOG_ERROR("Failed to add TPU buffer to request (Error: " + std::to_string(ret) + ").");
            return false;
        }

        requests_.push_back(std::move(request));
    }
    LOG_INFO("Libcamera requests created and buffers added for both streams.");
    
    return true;
}

void CameraCapture::request_complete_callback(libcamera::Request* request) {
    LOG_DEBUG("CameraCapture: request_complete_callback invoked.");
    if (!running_) {
        return;
    }

    if (request->status() == libcamera::Request::RequestCancelled) {
        LOG_DEBUG("CameraCapture: Request cancelled (likely flushing).");
        return;
    }
    
    if (request->status() != libcamera::Request::RequestComplete) {
        LOG_ERROR("CameraCapture: Request failed with status: " + std::to_string(request->status()));
        // Allow requeueing even on failure to recover
    }

    // Get the hardware capture timestamp from the request metadata
    std::chrono::high_resolution_clock::time_point hardware_capture_timestamp;
    auto md_timestamp = request->metadata().get(libcamera::controls::SensorTimestamp);
    if (md_timestamp) {
        hardware_capture_timestamp = std::chrono::high_resolution_clock::time_point(std::chrono::nanoseconds(*md_timestamp));
    } else {
        LOG_WARNING("CameraCapture: SensorTimestamp not found in request metadata. Using current system time.");
        hardware_capture_timestamp = std::chrono::high_resolution_clock::now();
    }

    // --- PROCESSING START ---
    auto processing_start_time = std::chrono::high_resolution_clock::now();
    
    // Capture buffer map BEFORE calling reuse()
    libcamera::Request::BufferMap captured_buffers = request->buffers();

    // Process Main Video Stream (only if there's a consumer queue)
    if (!main_output_queues_.empty() && captured_buffers.count(video_stream_)) {
        const libcamera::FrameBuffer* video_fb = captured_buffers.at(video_stream_);
        const libcamera::StreamConfiguration& video_cfg = video_stream_->configuration();
        
        process_frame_buffer(video_fb, video_cfg, image_buffer_pool_, main_output_queues_.front().get(), "Main Video Stream", width_, height_, hardware_capture_timestamp);
    } else if (captured_buffers.count(video_stream_)) {
        LOG_DEBUG("CameraCapture: Main Video Stream frame received but no output queues configured. Dropping frame.");
    } else {
        LOG_WARNING("CameraCapture: Video stream buffer missing from completed request.");
    }

    // Process TPU Stream
    if (captured_buffers.count(tpu_stream_)) {
        const libcamera::FrameBuffer* tpu_fb = captured_buffers.at(tpu_stream_);
        const libcamera::StreamConfiguration& tpu_cfg = tpu_stream_->configuration();
        
        process_frame_buffer(tpu_fb, tpu_cfg, image_buffer_pool_, tpu_output_queue_, "TPU Stream", target_tpu_width_, target_tpu_height_, hardware_capture_timestamp);
    } else {
        LOG_WARNING("CameraCapture: TPU stream buffer missing from completed request.");
    }
    
    // --- PERFORMANCE METRICS ---
    auto processing_end_time = std::chrono::high_resolution_clock::now();
    long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(processing_end_time - processing_start_time).count();
    {
        std::lock_guard<std::mutex> lock(frame_latencies_mutex_);
        frame_latencies_ms_.push_back(duration_ms);
        total_frames_processed_++;
    }
    
    // --- REQUEUE ---
    // CRITICAL FIX: Use ReuseBuffers flag for efficient buffer recycling.
    // This preserves the buffers attached to the request, avoiding the need to
    // manually call request->addBuffer() again, which would crash.
    request->reuse(libcamera::Request::ReuseBuffers); 

    // Requeue only if still running
    if (running_) {
        if (camera_->queueRequest(request)) {
            LOG_ERROR("Failed to re-queue request. Stopping capture.");
            stop();
        }
    }
}

void CameraCapture::get_performance_metrics() {
    std::lock_guard<std::mutex> lock(frame_latencies_mutex_);

    if (total_frames_processed_ == 0) {
        LOG_INFO("CameraCapture: No frames processed for performance metrics.");
        return;
    }

    double average_latency_ms = 0;
    for (long long latency : frame_latencies_ms_) {
        average_latency_ms += latency;
    }
    average_latency_ms /= total_frames_processed_;
    double average_fps = 1000.0 / average_latency_ms; // Inverse of average latency to get average FPS

    double sum_sq_diff = 0;
    for (long long latency : frame_latencies_ms_) {
        sum_sq_diff += (latency - average_latency_ms) * (latency - average_latency_ms);
    }
    double std_dev_ms = std::sqrt(sum_sq_diff / total_frames_processed_);

    std::sort(frame_latencies_ms_.begin(), frame_latencies_ms_.end());
    size_t percentile_99_index = static_cast<size_t>(std::round(total_frames_processed_ * 0.99));
    size_t percentile_95_index = static_cast<size_t>(std::round(total_frames_processed_ * 0.95));
    size_t percentile_50_index = static_cast<size_t>(std::round(total_frames_processed_ * 0.50));

    long long p99_latency_ms = frame_latencies_ms_[std::min(percentile_99_index, static_cast<size_t>(total_frames_processed_ - 1))];
    long long p95_latency_ms = frame_latencies_ms_[std::min(percentile_95_index, static_cast<size_t>(total_frames_processed_ - 1))];
    long long p50_latency_ms = frame_latencies_ms_[std::min(percentile_50_index, static_cast<size_t>(total_frames_processed_ - 1))];

    LOG_CSV("CameraCapture", "FrameCapture", p50_latency_ms, p95_latency_ms, p99_latency_ms, 0.0, average_fps);
    LOG_DEBUG("--- CameraCapture Performance Metrics (Frame Latency) ---");
    LOG_DEBUG("  Total Frames Processed: " + std::to_string(total_frames_processed_));
    LOG_DEBUG("  Average FPS: " + std::to_string(average_fps));
    LOG_DEBUG("  Average Latency: " + std::to_string(average_latency_ms) + " ms");
    LOG_DEBUG("  Latency Std Dev: " + std::to_string(std_dev_ms) + " ms");
    LOG_DEBUG("  50th Percentile Latency: " + std::to_string(p50_latency_ms) + " ms");
    LOG_DEBUG("  95th Percentile Latency: " + std::to_string(p95_latency_ms) + " ms");
    LOG_DEBUG("  99th Percentile Latency: " + std::to_string(p99_latency_ms) + " ms");
    LOG_DEBUG("---------------------------------------------------------");

    frame_latencies_ms_.clear();
    total_frames_processed_ = 0;
} // Correct closing brace for request_complete_callback

void CameraCapture::get_state() const {
    LOG_INFO("--- CameraCapture State ---");
    LOG_INFO("  Running: " + std::to_string(running_));
    LOG_INFO("  Main Stream Resolution: " + std::to_string(width_) + "x" + std::to_string(height_));
    LOG_INFO("  TPU Stream Resolution: " + std::to_string(tpu_width_) + "x" + std::to_string(tpu_height_));
    if (camera_) {
        LOG_INFO("  Camera ID: " + camera_->id());
    }
    LOG_INFO("  Actual Configured Main Stream: " + std::to_string(actual_size_.width) + "x" + std::to_string(actual_size_.height) + " " + pixelFormatToString(actual_pixel_format_));
    LOG_INFO("  Actual Configured Stride: " + std::to_string(actual_stride_));
    if (allocator_ && video_stream_ && tpu_stream_) {
        LOG_INFO("  Number of buffers (Main): " + std::to_string(allocator_->buffers(video_stream_).size()));
        LOG_INFO("  Number of buffers (TPU): " + std::to_string(allocator_->buffers(tpu_stream_).size()));
    }
    LOG_INFO("---------------------------");
}
