#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <chrono>
#include <sstream>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <algorithm>
#include <numeric>

#include "camera_capture.h"
#include "util_logging.h"

#include <opencv2/opencv.hpp>

#include <libcamera/property_ids.h>
#include <libcamera/control_ids.h>
#include <libcamera/formats.h>

#include <iostream>
#include <map>
#include <iomanip>

// Helper to convert libcamera PixelFormat to a string for logging
static std::string pixelFormatToString(const libcamera::PixelFormat& format) {
    std::stringstream ss;
    ss << "'" << std::hex << std::setfill('0') << std::setw(8) << format.fourcc() << "'";
    return ss.str();
}

static bool process_frame_buffer(const libcamera::FrameBuffer* fb,
                                 const libcamera::StreamConfiguration& cfg,
                                 std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                                 ImageQueue& queue,
                                 const char* stream_name,
                                 [[maybe_unused]] unsigned int target_width,
                                 [[maybe_unused]] unsigned int target_height,
                                 long long call_ts_epoch_ms,
                                 const libcamera::PixelFormat& actual_format,
                                 long long frame_id,
                                 [[maybe_unused]] long long exposure_ms)
{
    APP_LOG_INFO("Processing frame for " + std::string(stream_name) + " with format: " + pixelFormatToString(actual_format));
    if (fb->planes().empty()) {
        APP_LOG_ERROR(std::string(stream_name) + " FrameBuffer has no planes.");
        return false;
    }
    APP_LOG_INFO("Frame buffer has " + std::to_string(fb->planes().size()) + " planes.");

    const libcamera::FrameBuffer::Plane& plane = fb->planes()[0];
    int fd = plane.fd.get();
    size_t length = plane.length;

    // 1. Acquire a buffer from the pool.
    auto pooled_buffer = buffer_pool->acquire();
    if (!pooled_buffer) {
        APP_LOG_ERROR(std::string(stream_name) + " Failed to acquire buffer from pool.");
        return false;
    }

    // 2. Map the DMA buffer.
    void* mapped_memory = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_memory == MAP_FAILED) {
        APP_LOG_ERROR(std::string(stream_name) + " Failed to map buffer: " + std::strerror(errno));
        return false;
    }

    // 3. Copy the data.
    if (pooled_buffer->data.size() < length) {
        APP_LOG_WARNING(std::string(stream_name) + " Resizing buffer from " + std::to_string(pooled_buffer->data.size()) + " to " + std::to_string(length));
        pooled_buffer->data.resize(length);
    }
    std::memcpy(pooled_buffer->data.data(), mapped_memory, length);
    // Set the valid data size in the buffer
    pooled_buffer->size = length;

    // 4. Unmap the buffer.
    if (munmap(mapped_memory, length) == -1) {
        APP_LOG_ERROR(std::string(stream_name) + " Failed to unmap buffer: " + std::strerror(errno));
    }

    // 5. Populate the ImageData struct.
    ImageData image_data(call_ts_epoch_ms, frame_id);
    image_data.width = cfg.size.width;
    image_data.height = cfg.size.height;
    image_data.format = actual_format;
    image_data.length = length;
    // Preserve zero-copy information
    image_data.fd = fd;
    image_data.offset = plane.offset; // Add offset information

    // Handle format conversion if needed
    if (actual_format == libcamera::formats::YUYV) {
        image_data.format = libcamera::formats::YUYV;
    } else if (actual_format == libcamera::formats::RGB888 || actual_format == libcamera::formats::BGR888) {
        // This condition implies that libcamera might still output BGR even if RGB was requested.
        if (std::strcmp(stream_name, "TPU Stream") == 0) {
            APP_LOG_INFO("TPU Stream (BGR888 received), converting to RGB888...");
            // Convert BGR to RGB in place
            uint8_t* pixel_data = pooled_buffer->data.data();
            for (size_t i = 0; i < length; i += 3) {
                std::swap(pixel_data[i], pixel_data[i + 2]); // Swap R and B channels
            }
            image_data.format = libcamera::formats::RGB888; // Update format to RGB888
        } else {
            image_data.format = actual_format;
        }
    } else if (actual_format == libcamera::formats::YUYV) {
        image_data.format = libcamera::formats::YUYV;
    }

    // Attach the pooled buffer to the ImageData.
    image_data.buffer = std::move(pooled_buffer);

    // 6. Push to queue.
    if (!queue.push(std::move(image_data))) {
        APP_LOG_ERROR(std::string(stream_name) + " Failed to push image data to queue.");
        return false;
    }

    APP_LOG_INFO(std::string(stream_name) + " Frame processed and enqueued successfully.");
    return true;
}

CameraCapture::CameraCapture(unsigned int main_width, unsigned int main_height,
                             unsigned int tpu_width, unsigned int tpu_height,
                             unsigned int tpu_fps,
                             unsigned int target_tpu_width, unsigned int target_tpu_height,
                             std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool,
                             std::list<std::reference_wrapper<ImageQueue>>& main_output_queues,
                             ImageQueue& image_processor_input_queue,
                             std::chrono::seconds watchdog_timeout)
    : width_(main_width), height_(main_height),
      tpu_width_(tpu_width), tpu_height_(tpu_height),
      tpu_fps_(tpu_fps),
      target_tpu_width_(target_tpu_width), target_tpu_height_(target_tpu_height),
      main_output_queues_(main_output_queues),
      image_processor_input_queue_(image_processor_input_queue),
      image_buffer_pool_(image_buffer_pool),
      watchdog_timeout_(watchdog_timeout),
      frame_count_(0) {
    APP_LOG_INFO("CameraCapture constructor called with main=" + std::to_string(main_width) + "x" + std::to_string(main_height) +
                 ", tpu=" + std::to_string(tpu_width) + "x" + std::to_string(tpu_height) + "@" + std::to_string(tpu_fps) + "fps");
    
    // Initialize the camera manager
    camera_manager_ = std::make_unique<libcamera::CameraManager>();
    int ret = camera_manager_->start();
    if (ret) {
        APP_LOG_ERROR("Failed to start CameraManager: " + std::to_string(ret));
        camera_manager_.reset();
    }
}

CameraCapture::~CameraCapture() {
    APP_LOG_INFO("CameraCapture destructor called.");
    stop();
}

bool CameraCapture::acquire_camera() {
    if (!camera_manager_) {
        APP_LOG_ERROR("Acquire: FAILURE - CameraManager is null.");
        return false;
    }
    
    auto cameras = camera_manager_->cameras();
    if (cameras.empty()) {
        APP_LOG_ERROR("Acquire: FAILURE - No cameras found by libcamera. Check camera connection and driver.");
        return false;
    }

    // Assign the selected camera to the member variable camera_
    camera_ = cameras.front();
    if (!camera_) {
        APP_LOG_ERROR("Acquire: FAILURE - CameraManager returned a null camera pointer for first camera.");
        return false;
    }

    APP_LOG_INFO("Selected Camera ID: " + camera_->id());
    
    int ret = camera_->acquire();
    if (ret) {
        APP_LOG_ERROR("Acquire: FAILURE - Failed to acquire camera (Error: " + std::to_string(ret) + "). Is the camera in use by another process?");
        camera_.reset();
        return false;
    }
    
    APP_LOG_INFO("Camera acquired successfully.");
    return true;
}

bool CameraCapture::start() {
    APP_LOG_INFO("CameraCapture::start() called.");
    
    if (running_.load()) {
        APP_LOG_ERROR("CameraCapture is already running.");
        return false;
    }
    
    // 1. Acquire the camera 
    if (!acquire_camera()) {
        APP_LOG_ERROR("CameraCapture failed to acquire camera. Cannot proceed.");
        return false;
    }
    APP_LOG_INFO("Camera acquired successfully. Proceeding to setup.");

    // CRITICAL SECONDARY CHECK: Should not be necessary, but defensively prevents a nullptr crash.
    if (!camera_) {
        APP_LOG_ERROR("CRITICAL: Camera pointer is null immediately after successful acquire_camera(). Aborting.");
        return false;
    }
    
    // Log to confirm successful state before moving to setup
    APP_LOG_INFO("Camera acquired and confirmed to be non-null. Proceeding to setup.");

    // 2. Setup (Configure streams, allocate buffers, create requests)
    if (!setup_camera()) {
        APP_LOG_ERROR("Failed to setup camera. Releasing camera.");
        
        // Safety check before release(): Only release if the pointer is still valid.
        if (camera_) {
            camera_->release();
        }
        camera_.reset();
        return false;
    }
    APP_LOG_INFO("Camera setup completed successfully.");
    
    // 3. Connect request completed signal
    camera_->requestCompleted.connect(this, &CameraCapture::request_complete_callback);

    // 4. Start Capture
    // Prepare controls to set during camera start
    libcamera::ControlList controls_to_set;
    
    // Set frame rate to 120 FPS using FrameDurationLimits for both streams
    if (tpu_fps_ > 0) {
        int64_t frame_duration_us = 1000000 / tpu_fps_; // Calculate microseconds per frame
        controls_to_set.set(libcamera::controls::FrameDurationLimits, {frame_duration_us, frame_duration_us});
        APP_LOG_INFO("Setting FrameDurationLimits to: " + std::to_string(frame_duration_us) + " us (" + std::to_string(tpu_fps_) + " FPS)");
    }
    
    // Also set the target frame rate for the camera
    if (tpu_fps_ > 0) {
        controls_to_set.set(libcamera::controls::AeEnable, true);
        APP_LOG_INFO("Setting AE enable for target FPS: " + std::to_string(tpu_fps_));
    }

    int ret = camera_->start(&controls_to_set);
    if (ret) {
        APP_LOG_ERROR("Failed to start camera: " + std::to_string(ret));
        camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);
        camera_->release();
        camera_.reset();
        return false;
    }
    APP_LOG_INFO("Libcamera camera started successfully.");
    
    // Read back actual configured FrameDuration for logging if it was set
    const libcamera::ControlList &properties = camera_->properties();
    if (properties.get(libcamera::controls::FrameDuration)) {
        int64_t actual_frame_duration = *properties.get(libcamera::controls::FrameDuration);
        [[maybe_unused]] double actual_fps = 1000000.0 / actual_frame_duration;
        APP_LOG_INFO("Actual configured FrameDuration: " + std::to_string(actual_frame_duration) + " us (" + std::to_string(actual_fps) + " FPS)");
    } else {
        APP_LOG_INFO("FrameDuration was not configured or is not available from camera properties.");
    }

    running_ = true;
    processing_running_ = true; // Set new flag for processing thread
    request_processor_thread_ = std::thread(&CameraCapture::request_processor_thread_func, this);
    
    // 5. Queue initial requests
    frame_count_ = 0;
    last_frame_time_ = std::chrono::high_resolution_clock::now();
    
    for (auto& req_ptr : requests_) {
        if (camera_->queueRequest(req_ptr.get())) { 
            APP_LOG_ERROR("Failed to queue initial request.");
            // Stop to clean up resources if queueing fails.
            running_ = false; 
            stop(); 
            return false;
        }
    }
    APP_LOG_INFO("CameraCapture: All initial requests successfully queued.");
    return true;
}

void CameraCapture::stop() {
    // 1. Signal shutdown to threads
    if (!running_.exchange(false)) { // Signal main thread to stop
        return;
    }
    APP_LOG_INFO("Stopping CameraCapture...");

    processing_running_ = false; // Signal processing thread to stop
    request_queue_cond_var_.notify_one(); // Wake up processing thread

    // Use try-catch blocks to ensure all cleanup steps are attempted even if some fail
    try {
        if (request_processor_thread_.joinable()) {
            request_processor_thread_.join(); // Wait for processing thread to finish
        }
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception while joining request processor thread: " + std::string(e.what()));
    }

    try {
        if (camera_) {
            // 3. CRITICAL: Disconnect the callback first.
            try {
                camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);
            } catch (const std::exception& e) {
                APP_LOG_ERROR("Exception while disconnecting callback: " + std::string(e.what()));
            }

            // 4. Stop the camera. This blocks until all pending requests return (flushing).
            try {
                camera_->stop();
                APP_LOG_INFO("Libcamera camera stopped.");
            } catch (const std::exception& e) {
                APP_LOG_ERROR("Exception while stopping camera: " + std::string(e.what()));
            }
            
            try {
                camera_->release();
                APP_LOG_INFO("Libcamera camera released.");
            } catch (const std::exception& e) {
                APP_LOG_ERROR("Exception while releasing camera: " + std::string(e.what()));
            }
        }
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception in camera cleanup section: " + std::string(e.what()));
    }
    
    // 5. Free buffers
    try {
        if (allocator_) {
            try {
                if (video_stream_) allocator_->free(video_stream_);
            } catch (const std::exception& e) {
                APP_LOG_ERROR("Exception while freeing video stream buffers: " + std::string(e.what()));
            }
            
            try {
                if (tpu_stream_) allocator_->free(tpu_stream_);
            } catch (const std::exception& e) {
                APP_LOG_ERROR("Exception while freeing TPU stream buffers: " + std::string(e.what()));
            }
            
            allocator_.reset();
            APP_LOG_INFO("Libcamera buffers freed.");
        }
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception in buffer allocator cleanup: " + std::string(e.what()));
    }
    
    // 6. NOW it is safe to destroy the requests and camera shared_ptr.
    try {
        requests_.clear(); // Explicit clear for Request objects
        camera_.reset();
        APP_LOG_INFO("Requests and camera pointer cleared.");
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Exception while clearing requests or resetting camera: " + std::string(e.what()));
    }

    APP_LOG_INFO("CameraCapture stopped.");
}

bool CameraCapture::setup_camera() {
    // --- SAFETY CHECK ---
    if (!camera_) {
        APP_LOG_ERROR("CRITICAL: setup_camera called but camera_ is nullptr! Aborting setup.");
        return false;
    }
    
    APP_LOG_INFO("setup_camera: Configuring camera: " + camera_->id());
    int ret = 0; 

    // Configure dual streams: main high-res and TPU viewfinder
    std::vector<libcamera::StreamRole> roles = {
        libcamera::StreamRole::VideoRecording, // main high-res
        libcamera::StreamRole::Viewfinder      // TPU/resized
    };
    
    std::unique_ptr<libcamera::CameraConfiguration> config = camera_->generateConfiguration(roles);

    if (!config) {
        APP_LOG_ERROR("Failed to generate dual stream configuration.");
        return false;
    }

    // Check if the generated configuration contains exactly two streams
    if (config->size() < 2) {
        APP_LOG_ERROR("Generated camera configuration has less than two streams for dual-stream setup (found " + std::to_string(config->size()) + ").");
        return false;
    }

    // Configure main stream (index 0) for 1536x864 resolution to support 120 FPS
    libcamera::StreamConfiguration& mainCfg = config->at(0);
    mainCfg.pixelFormat = libcamera::formats::SRGGB10_CSI2P;  // 10-bit Bayer format as requested
    mainCfg.size.width = 1536;  // Fixed to Mode 0 resolution for 120 FPS
    mainCfg.size.height = 864;  // Fixed to Mode 0 resolution for 120 FPS
    mainCfg.bufferCount = 8; // Increase buffer count for high frame rate
    
    // Validate main stream configuration
    if (mainCfg.size.width != 1536 || mainCfg.size.height != 864) {
        APP_LOG_WARNING("Main stream configuration does not match expected 1536x864 resolution");
    }

    // Configure tpu stream (index 1) for 320x320 resolution
    libcamera::StreamConfiguration& tpuCfg = config->at(1);
    tpuCfg.pixelFormat = libcamera::formats::RGB888;
    tpuCfg.size.width = 320;   // Fixed to TPU input size
    tpuCfg.size.height = 320;  // Fixed to TPU input size
    tpuCfg.bufferCount = 8; // Increase buffer count for high frame rate
    
    // Validate TPU stream configuration
    if (tpuCfg.size.width != 320 || tpuCfg.size.height != 320) {
        APP_LOG_WARNING("TPU stream configuration does not match expected 320x320 resolution");
    }
    
    // Validate and complete the configuration
    libcamera::CameraConfiguration::Status config_status = config->validate();
    APP_LOG_INFO(std::string("CameraConfiguration validate() -> Status: ") + std::to_string(config_status));
    
    if (config_status == libcamera::CameraConfiguration::Invalid) {
        APP_LOG_ERROR("Invalid dual stream camera configuration. Check requested resolutions/formats.");
        return false;
    } else if (config_status == libcamera::CameraConfiguration::Adjusted) {
        APP_LOG_WARNING("Camera configuration adjusted by libcamera for dual streams.");
    }
    
    // Log final chosen config for each stream
    for (unsigned i = 0; i < config->size(); ++i) {
        std::string log_msg = "Final stream[" + std::to_string(i) + "] size=" +
                              std::to_string(config->at(i).size.width) + "x" +
                              std::to_string(config->at(i).size.height) + " fmt=" +
                              pixelFormatToString(config->at(i).pixelFormat);
        APP_LOG_INFO(log_msg);
    }

    // Store the actual configured stream properties for the main video stream
    actual_pixel_format_ = mainCfg.pixelFormat;
    actual_size_ = mainCfg.size;
    actual_stride_ = mainCfg.stride; // Get the stride, important for raw data.

    APP_LOG_INFO("CameraCapture: Configured main stream format: " + pixelFormatToString(actual_pixel_format_) + 
             ", size: " + std::to_string(actual_size_.width) + "x" + std::to_string(actual_size_.height) +
             ", stride: " + std::to_string(actual_stride_) + " (FOURCC: " + pixelFormatToString(actual_pixel_format_) + ")");
    
    ret = camera_->configure(config.get());
    if (ret) {
        APP_LOG_ERROR("Failed to configure camera streams (Error: " + std::to_string(ret) + ").");
        return false;
    }
    APP_LOG_INFO("Libcamera dual streams configured.");

    video_stream_ = mainCfg.stream();
    tpu_stream_ = tpuCfg.stream();

    if (video_stream_) {
        APP_LOG_INFO("Actual Video Stream Config: " + std::to_string(video_stream_->configuration().size.width) + "x" + std::to_string(video_stream_->configuration().size.height) + " format: " + pixelFormatToString(video_stream_->configuration().pixelFormat));
    } else {
        APP_LOG_ERROR("Video stream is null after configuration.");
        return false;
    }

    if (tpu_stream_) {
        APP_LOG_INFO("Actual TPU Stream Config: " + std::to_string(tpu_stream_->configuration().size.width) + "x" + std::to_string(tpu_stream_->configuration().size.height) + " format: " + pixelFormatToString(tpu_stream_->configuration().pixelFormat) + " stride: " + std::to_string(tpu_stream_->configuration().stride));
    } else {
        APP_LOG_ERROR("TPU stream is null after configuration.");
        return false;
    }

    // Allocate buffers for main video stream
    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    if (!allocator_) {
        APP_LOG_ERROR("Failed to create FrameBufferAllocator.");
        return false;
    }
    
    ret = allocator_->allocate(video_stream_);
    if (ret < 0) {
        APP_LOG_ERROR("Failed to allocate buffers for main video stream (Error: " + std::to_string(ret) + ").");
        allocator_.reset(); // Ensure allocator is cleaned up if it failed early
        return false;
    }
    APP_LOG_INFO("Libcamera buffers allocated for main video stream. Number of buffers: " + std::to_string(allocator_->buffers(video_stream_).size()));

    // Increase buffer count for high frame rate support
    if (tpu_fps_ >= 120) {
        APP_LOG_INFO("High frame rate mode detected, ensuring adequate buffer allocation");
    }

    // Allocate buffers for TPU stream
    ret = allocator_->allocate(tpu_stream_);
    if (ret < 0) {
        APP_LOG_ERROR("Failed to allocate buffers for TPU stream (Error: " + std::to_string(ret) + ").");
        allocator_.reset();
        return false;
    }
    APP_LOG_INFO("Libcamera buffers allocated for TPU stream. Number of buffers: " + std::to_string(allocator_->buffers(tpu_stream_).size()));

    // Create requests
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& video_buffers = allocator_->buffers(video_stream_);
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& tpu_buffers = allocator_->buffers(tpu_stream_);

    if (video_buffers.size() != tpu_buffers.size()) {
        APP_LOG_ERROR("Mismatched buffer counts between main video stream and TPU stream. This is unexpected.");
        return false;
    }
    
    requests_.clear(); // Ensure clean slate before populating
    
    for (unsigned int i = 0; i < video_buffers.size(); ++i) {
        std::unique_ptr<libcamera::Request> request = camera_->createRequest();
        if (!request) {
            APP_LOG_ERROR("Failed to create request.");
            return false;
        }

        // Set initial controls for high FPS
        request->controls().set(libcamera::controls::AeEnable, true);
        
        ret = request->addBuffer(video_stream_, video_buffers[i].get()); 
        if (ret) {
            APP_LOG_ERROR("Failed to add main buffer to request (Error: " + std::to_string(ret) + ").");
            return false;
        }

        ret = request->addBuffer(tpu_stream_, tpu_buffers[i].get());
        if (ret) {
            APP_LOG_ERROR("Failed to add TPU buffer to request (Error: " + std::to_string(ret) + ").");
            return false;
        }

        requests_.push_back(std::move(request));
    }
    APP_LOG_INFO("Libcamera requests created and buffers added for both streams.");
    
    // Call init_video_encoder here as it's now safe to assume streams are configured.
    if (!init_video_encoder()) {
        APP_LOG_ERROR("Failed to initialize video encoder.");
        return false;
    }

    return true;
}

void CameraCapture::request_complete_callback(libcamera::Request* request) {
    APP_LOG_INFO("CameraCapture: request_complete_callback invoked.");
    // If the main CameraCapture is stopping, or processing thread is stopping, just re-queue and return.
    // The processing_running_ flag controls the processing thread.
    if (!running_.load() || !processing_running_.load()) {
        if (request->status() != libcamera::Request::RequestCancelled) {
            request->reuse(libcamera::Request::ReuseBuffers);
            if (camera_ && camera_->queueRequest(request)) {
                APP_LOG_ERROR("Failed to re-queue request during CameraCapture shutdown.");
            }
        }
        return;
    }

    if (request->status() == libcamera::Request::RequestCancelled) {
        APP_LOG_DEBUG("CameraCapture: Request cancelled (likely flushing).");
        request->reuse(libcamera::Request::ReuseBuffers); 
        if (camera_ && camera_->queueRequest(request)) {
            APP_LOG_ERROR("Failed to re-queue cancelled request.");
        }
        return;
    }
    
    if (request->status() != libcamera::Request::RequestComplete) {
        APP_LOG_ERROR("CameraCapture: Request failed with status: " + std::to_string(request->status()));
        request->reuse(libcamera::Request::ReuseBuffers); 
        if (camera_ && camera_->queueRequest(request)) {
            APP_LOG_ERROR("Failed to re-queue failed request.");
        }
        return;
    }

    // Enqueue the request for processing by the dedicated thread
    {
        std::lock_guard<std::mutex> lock(request_queue_mutex_);
        request_queue_.push(request);
        request_queue_cond_var_.notify_one();
    }
}

void CameraCapture::request_processor_thread_func() {
    APP_LOG_INFO("CameraCapture: Request processor thread started.");
    while (processing_running_.load()) {
        std::unique_lock<std::mutex> lock(request_queue_mutex_);
        request_queue_cond_var_.wait(lock, [this] { 
            return !request_queue_.empty() || !processing_running_.load(); 
        });

        if (!processing_running_.load() && request_queue_.empty()) {
            break; // Exit if shutting down and queue is empty
        }

        if (request_queue_.empty()) {
            continue; // Spurious wakeup, wait again
        }

        libcamera::Request* request = request_queue_.front();
        request_queue_.pop();
        lock.unlock(); // Release lock before processing

        // --- Safety check for null request ---
        if (!request) {
            APP_LOG_ERROR("CameraCapture: Received null request in processor thread.");
            continue;
        }

        // --- Actual processing of the frame ---
        std::chrono::high_resolution_clock::time_point start_process_time = std::chrono::high_resolution_clock::now();

        long long produced_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count();
        long long monotonic_raw_now_ns = Logger::getInstance().get_raw_monotonic_time_ns();

        // Calculate the offset between CLOCK_REALTIME (epoch UTC) and CLOCK_MONOTONIC_RAW (nanoseconds since boot)
        long long epoch_offset_ms = produced_ts - (monotonic_raw_now_ns / 1000000);

        long long call_ts = 0;
        auto md_timestamp = request->metadata().get(libcamera::controls::SensorTimestamp);
        if (md_timestamp) {
            call_ts = (*md_timestamp / 1000000) + epoch_offset_ms;
            APP_LOG_DEBUG("SensorTimestamp (ns): " + std::to_string(*md_timestamp) + ", converted call_ts (epoch ms): " + std::to_string(call_ts));
        } else {
            APP_LOG_WARNING("CameraCapture: SensorTimestamp not found in request metadata. Using current system time.");
            call_ts = produced_ts;
        }
        
        long long frame_id = request->sequence();

        long long exposure_ms = 0;
        auto md_exposure_us = request->metadata().get(libcamera::controls::ExposureTime);
        if (md_exposure_us) {
            exposure_ms = *md_exposure_us / 1000; // Convert microseconds to milliseconds
            APP_LOG_DEBUG("ExposureTime (us): " + std::to_string(*md_exposure_us) + ", converted exposure_ms: " + std::to_string(exposure_ms));
        } else {
            APP_LOG_DEBUG("CameraCapture: ExposureTime not found in request metadata.");
        }

        // --- Safety check for buffers ---
        libcamera::Request::BufferMap captured_buffers = request->buffers();
        if (captured_buffers.empty()) {
            APP_LOG_WARNING("CameraCapture: No buffers in completed request.");
            request->reuse(libcamera::Request::ReuseBuffers); 
            if (running_.load() && camera_) {
                if (camera_->queueRequest(request)) {
                    APP_LOG_ERROR("Failed to re-queue request with no buffers.");
                }
            }
            continue;
        }

        bool processed_any_frame = false; // Track if any frame was successfully processed

        if (!main_output_queues_.empty() && captured_buffers.count(video_stream_)) {
            const libcamera::FrameBuffer* video_fb = captured_buffers.at(video_stream_);
            if (!video_fb) {
                APP_LOG_ERROR("CameraCapture: Null video frame buffer.");
            } else {
                const libcamera::StreamConfiguration& video_cfg = video_stream_->configuration();
                bool processed_main = process_frame_buffer(video_fb, video_cfg, image_buffer_pool_, main_output_queues_.front().get(), "Main Video Stream", width_, height_, call_ts, video_stream_->configuration().pixelFormat, frame_id, exposure_ms);
                if (processed_main) {
                    processed_any_frame = true;
                    // Batch logging: only log every 10th frame to reduce overhead
                    static int main_log_counter = 0;
                    main_log_counter++;
                    if (main_log_counter % 10 == 0) {
                        // Log for main video stream
                        CsvLogEntry entry;
                        entry.produced_ts_epoch_ms = produced_ts;
                        copy_to_array(entry.module, "CameraCapture");
                        entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
                        copy_to_array(entry.event, "frame_captured_main");
                        entry.call_ts_epoch_ms = call_ts;
                        entry.camera_frame_id = frame_id;
                        entry.camera_width = video_cfg.size.width;
                        entry.camera_height = video_cfg.size.height;
                        entry.camera_exposure_ms = static_cast<float>(exposure_ms);
                        Logger::getInstance().log_csv(entry);
                    }
                }
            }
        } else if (captured_buffers.count(video_stream_)) {
            APP_LOG_DEBUG("CameraCapture: Main Video Stream frame received but no output queues configured. Dropping frame.");
        } else {
            APP_LOG_WARNING("CameraCapture: Video stream buffer missing from completed request.");
        }

        if (captured_buffers.count(tpu_stream_)) {
            const libcamera::FrameBuffer* tpu_fb = captured_buffers.at(tpu_stream_);
            if (!tpu_fb) {
                APP_LOG_ERROR("CameraCapture: Null TPU frame buffer.");
            } else {
                const libcamera::StreamConfiguration& tpu_cfg = tpu_stream_->configuration();
                bool processed_tpu = this->process_tpu_raw_frame_buffer(tpu_fb, tpu_cfg, call_ts, frame_id, exposure_ms);
                if (processed_tpu) {
                    processed_any_frame = true;
                    // Log for TPU stream
                    // Batch logging: only log every 10th frame to reduce overhead
                    static int log_counter = 0;
                    log_counter++;
                    if (log_counter % 10 == 0) {
                        CsvLogEntry entry;
                        entry.produced_ts_epoch_ms = produced_ts;
                        copy_to_array(entry.module, "CameraCapture");
                        entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
                        copy_to_array(entry.event, "frame_captured_tpu");
                        entry.call_ts_epoch_ms = call_ts;
                        entry.camera_frame_id = frame_id;
                        entry.camera_width = tpu_cfg.size.width;
                        entry.camera_height = tpu_cfg.size.height;
                        entry.camera_exposure_ms = static_cast<float>(exposure_ms);
                        Logger::getInstance().log_csv(entry);
                    }
                    
                    // FPS measurement instrumentation
                    auto current_time = std::chrono::high_resolution_clock::now();
                    if (fps_measurement_frames_ == 0) {
                        first_frame_time_ = current_time;
                    } else {
                        auto interval_us = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_frame_time_).count();
                        frame_intervals_us_.push_back(interval_us);
                    }
                    last_frame_time_ = current_time;
                    fps_measurement_frames_++;
                    
                    // Update freshness indicators
                    last_frame_timestamp_ = call_ts;
                    if (frame_intervals_us_.size() > 0) {
                        long long total_interval_us = 0;
                        for (long long interval : frame_intervals_us_) {
                            total_interval_us += interval;
                        }
                        double avg_interval_us = static_cast<double>(total_interval_us) / frame_intervals_us_.size();
                        double effective_fps = 1000000.0 / avg_interval_us;
                        frame_rate_ = static_cast<int>(effective_fps);
                    }
                    
                    // Print FPS statistics every 100 frames
                    if (fps_measurement_frames_ % 100 == 0 && frame_intervals_us_.size() > 0) {
                        long long total_interval_us = 0;
                        long long min_interval_us = frame_intervals_us_[0];
                        long long max_interval_us = frame_intervals_us_[0];
                        
                        for (long long interval : frame_intervals_us_) {
                            total_interval_us += interval;
                            if (interval < min_interval_us) min_interval_us = interval;
                            if (interval > max_interval_us) max_interval_us = interval;
                        }
                        
                        double avg_interval_us = static_cast<double>(total_interval_us) / frame_intervals_us_.size();
                        double effective_fps_stats = 1000000.0 / avg_interval_us;
                        frame_rate_ = static_cast<int>(effective_fps_stats);
                        
                        APP_LOG_INFO("Camera FPS Stats (last 100 frames): Avg Interval=" + std::to_string(avg_interval_us) + "us, Min=" + std::to_string(min_interval_us) + "us, Max=" + std::to_string(max_interval_us) + "us, Effective FPS=" + std::to_string(effective_fps_stats));
                        
                        // Check if we're achieving the target 120 FPS
                        if (effective_fps_stats < 110.0) { // Allow some tolerance
                            APP_LOG_WARNING("TPU stream FPS (" + std::to_string(effective_fps_stats) + ") is below target 120 FPS");
                        } else if (effective_fps_stats >= 110.0 && effective_fps_stats <= 130.0) {
                            APP_LOG_INFO("TPU stream FPS (" + std::to_string(effective_fps_stats) + ") is within target range for 120 FPS");
                        } else {
                            APP_LOG_INFO("TPU stream FPS (" + std::to_string(effective_fps_stats) + ") is above target 120 FPS");
                        }
                    }
                }
            }
        } else {
            APP_LOG_WARNING("CameraCapture: TPU stream buffer missing from completed request.");
        }
        
        std::chrono::high_resolution_clock::time_point end_process_time = std::chrono::high_resolution_clock::now();
        [[maybe_unused]] long long total_process_us = std::chrono::duration_cast<std::chrono::microseconds>(end_process_time - start_process_time).count();
        if (processed_any_frame) {
            APP_LOG_INFO("CameraCapture: Total time to process request (frame_id=" + std::to_string(frame_id) + "): " + std::to_string(total_process_us) + " us");
        }
    
        // --- REQUEUE ---
        request->reuse(libcamera::Request::ReuseBuffers); 
        if (running_.load()) { // Check main running_ flag for libcamera interaction
            if (camera_ && camera_->queueRequest(request)) {
                APP_LOG_ERROR("Failed to re-queue request from processor thread. Stopping capture.");
                running_ = false; // Set main running flag to false
            }
        }
    }
    APP_LOG_INFO("CameraCapture: Request processor thread stopped.");
}

bool CameraCapture::init_video_encoder() {
    APP_LOG_INFO("Initializing H.264 encoder...");
    // This is a placeholder; actual encoder initialization would go here.
    // It would involve setting up encoder parameters, potentially using libavcodec or a hardware encoder.
    // For now, we'll just log that it's initialized.
    APP_LOG_INFO("H.264 encoder initialized (placeholder).");
    return true;
}

// This function will be placed in src/camera_capture.cpp
bool CameraCapture::process_tpu_raw_frame_buffer(const libcamera::FrameBuffer* fb,
                                                 const libcamera::StreamConfiguration& cfg,
                                                 long long call_ts_epoch_ms,
                                                 long long frame_id,
                                                 [[maybe_unused]] long long exposure_ms) {
    if (fb->planes().empty()) {
        APP_LOG_ERROR("TPU raw FrameBuffer has no planes.");
        return false;
    }

    const libcamera::FrameBuffer::Plane& plane = fb->planes()[0];
    int fd = plane.fd.get();
    size_t length = plane.length;

    // 1. Acquire a buffer from the pool.
    auto pooled_buffer = image_buffer_pool_->acquire();
    if (!pooled_buffer) {
        APP_LOG_ERROR("TPU raw Failed to acquire buffer from pool.");
        return false;
    }

    // 2. Map the DMA buffer.
    void* mapped_memory = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_memory == MAP_FAILED) {
        APP_LOG_ERROR(std::string("TPU raw Failed to map buffer: ") + std::strerror(errno));
        return false;
    }

    // 3. Copy the data.
    if (pooled_buffer->data.size() < length) {
        APP_LOG_WARNING("TPU raw Resizing buffer from " + std::to_string(pooled_buffer->data.size()) + " to " + std::to_string(length));
        pooled_buffer->data.resize(length);
    }
    std::memcpy(pooled_buffer->data.data(), mapped_memory, length);
    // Set the valid data size in the buffer
    pooled_buffer->size = length;

    // 4. Unmap the buffer.
    if (munmap(mapped_memory, length) == -1) {
        APP_LOG_ERROR(std::string("TPU raw Failed to unmap buffer: ") + std::strerror(errno));
    }

    // 5. Populate the ImageData struct.
    ImageData image_data(call_ts_epoch_ms, frame_id);
    image_data.width = cfg.size.width;
    image_data.height = cfg.size.height;
    image_data.format = cfg.pixelFormat;
    image_data.length = length;
    // Preserve zero-copy information
    image_data.fd = fd;
    image_data.offset = plane.offset; // Add offset information

    // Attach the pooled buffer to the ImageData.
    image_data.buffer = std::move(pooled_buffer);

    // 6. Push to TPU queue.
    if (!image_processor_input_queue_.push(std::move(image_data))) {
        APP_LOG_ERROR("TPU raw Failed to push image data to queue.");
        return false;
    }

    APP_LOG_INFO("TPU raw Frame processed and enqueued successfully.");
    return true;
}