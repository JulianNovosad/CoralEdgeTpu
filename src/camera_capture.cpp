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
                                 long long call_ts_epoch_ms, // Changed to epoch ms for consistency
                                 const libcamera::PixelFormat& actual_format,
                                 long long frame_id, // New parameter
                                 long long exposure_ms) // New parameter
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
    APP_LOG_INFO("Plane 0: fd=" + std::to_string(fd) + ", length=" + std::to_string(length));

    auto start_time_process_frame_buffer = std::chrono::high_resolution_clock::now();

    // Determine expected bytes per pixel based on the actual format
    size_t expected_bytes_per_pixel = 0;
    if (actual_format == libcamera::formats::YUYV) {
        expected_bytes_per_pixel = 2; // YUYV (YUV 4:2:2) is 2 bytes per pixel
    } else if (actual_format == libcamera::formats::RGB888 || actual_format == libcamera::formats::BGR888) {
        expected_bytes_per_pixel = 3; // RGB888/BGR888 is 3 bytes per pixel
    } else {
        APP_LOG_ERROR(std::string(stream_name) + " Unsupported pixel format " + pixelFormatToString(actual_format) + " for processing.");
        return false;
    }
    
    size_t expected_payload_size = cfg.size.width * cfg.size.height * expected_bytes_per_pixel;

    // 1. Acquire a buffer from the pool.
    auto start_time_acquire_buffer = std::chrono::high_resolution_clock::now();
    APP_LOG_INFO("Acquiring buffer from pool...");
    auto pooled_buffer = buffer_pool->acquire();
    if (!pooled_buffer) {
        APP_LOG_WARNING(std::string(stream_name) + " failed to acquire a buffer. Dropping frame.");
        return false;
    }
    auto end_time_acquire_buffer = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("Time to acquire buffer: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_time_acquire_buffer - start_time_acquire_buffer).count()) + " us");

    if (expected_payload_size > pooled_buffer->data.capacity()) {
        APP_LOG_ERROR(std::string(stream_name) + " Expected frame size (" + std::to_string(expected_payload_size) +
                  ") exceeds buffer pool capacity (" + std::to_string(pooled_buffer->data.capacity()) + "). Dropping frame.");
        return false;
    }
    
    // 2. Mmap and copy the frame data.
    auto start_time_mmap_copy = std::chrono::high_resolution_clock::now();
    APP_LOG_INFO("Mmapping plane buffer...");
    void* mmap_ptr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) {
        APP_LOG_ERROR(std::string(stream_name) + " Failed to mmap plane buffer: " + std::string(strerror(errno)));
        return false;
    }
    APP_LOG_INFO("Mmap successful. Copying data...");
    
    // Copying logic adapted for different pixel formats and stride.
    size_t bytes_per_line_src = cfg.stride;
    size_t bytes_per_line_dst = cfg.size.width * expected_bytes_per_pixel;
    
    uint8_t* dst = pooled_buffer->data.data();
    uint8_t* src = static_cast<uint8_t*>(mmap_ptr);

    if (bytes_per_line_src == bytes_per_line_dst) { // No padding per line, copy in one go
        std::memcpy(dst, src, expected_payload_size);
    } else { // Padding exists, copy line by line
        for (unsigned int i = 0; i < cfg.size.height; i++) {
            std::memcpy(dst, src, bytes_per_line_dst);
            dst += bytes_per_line_dst;
            src += bytes_per_line_src;
        }
    }
    pooled_buffer->size = expected_payload_size;

    auto copy_end_time = std::chrono::high_resolution_clock::now();
    long long copy_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end_time - start_time_mmap_copy).count();

    munmap(mmap_ptr, length);
    auto end_time_mmap_copy = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("Time for mmap and copy: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_time_mmap_copy - start_time_mmap_copy).count()) + " us");
    APP_LOG_INFO("Data copied. Pooled buffer size: " + std::to_string(pooled_buffer->size));

    // Construct ImageData directly with the new constructor
    ImageData image_data(call_ts_epoch_ms, frame_id); // Pass frame_id as well
    image_data.width = cfg.size.width;
    image_data.height = cfg.size.height;
    image_data.buffer = std::move(pooled_buffer);
    image_data.format = actual_format; // Store the actual format

    // 3. Conditional color conversion
    auto start_time_color_conversion = std::chrono::high_resolution_clock::now();
    if (actual_format == libcamera::formats::BGR888 && actual_format != libcamera::formats::RGB888) { 
        // If BGR888 is received for TPU stream, convert to RGB.
        // This condition implies that libcamera might still output BGR even if RGB was requested.
        // Or if the stream is intended for something that requires RGB and it's BGR.
        if (std::strcmp(stream_name, "TPU Stream") == 0) {
            APP_LOG_INFO("TPU Stream (BGR888 received), converting to RGB888...");
            cv::Mat bgr_image(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
            cv::cvtColor(bgr_image, bgr_image, cv::COLOR_BGR2RGB);
            image_data.format = libcamera::formats::RGB888; // Update format to RGB888
            APP_LOG_INFO("BGR888 to RGB888 conversion complete.");
        }
    } else if (actual_format == libcamera::formats::RGB888) {
        APP_LOG_INFO("RGB888 stream, no color conversion needed.");
    } else if (actual_format == libcamera::formats::YUYV) {
        APP_LOG_INFO("YUYV stream, no color conversion needed (assuming H264 takes YUYV).");
    }
    auto end_time_color_conversion = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("Time for color conversion: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_time_color_conversion - start_time_color_conversion).count()) + " us");


    // 4. Resize if necessary.
    auto start_time_resize = std::chrono::high_resolution_clock::now();
    // This resizing path currently creates cv::Mat from the buffer.
    // If the input format is YUYV, cv::resize might need special handling
    // or a conversion to BGR/RGB before resizing and then back.
    // For simplicity, for now, we assume resizing is primarily for RGB/BGR.
    // If resizing is needed for YUYV, further logic will be required.
    if (image_data.width != target_width || image_data.height != target_height) {
        APP_LOG_WARNING("Resizing of " + std::string(stream_name) + " with format " + pixelFormatToString(image_data.format) + " is not fully optimized for all formats (e.g., YUYV).");
        
        if (image_data.format == libcamera::formats::YUYV) {
            // Placeholder for YUYV resizing logic.
            // For now, if YUYV needs resizing, it's more complex as it involves
            // correct YUV scaling. We'll proceed with direct resizing for now,
            // but this might lead to quality degradation or incorrect results.
            cv::Mat src_image(image_data.height, image_data.width, CV_8UC2, image_data.buffer->data.data());
            cv::Mat resized_image;
            cv::resize(src_image, resized_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

            auto resized_pooled_buffer = buffer_pool->acquire();
            if (!resized_pooled_buffer) {
                APP_LOG_WARNING(std::string(stream_name) + " failed to acquire a buffer for resized image. Dropping frame.");
                return false;
            }

            size_t resized_size = resized_image.total() * resized_image.elemSize();
            if (resized_size > resized_pooled_buffer->data.capacity()) {
                APP_LOG_WARNING(std::string(stream_name) + " resized image size (" + std::to_string(resized_size) +
                            ") exceeds buffer capacity (" + std::to_string(resized_pooled_buffer->data.capacity()) + "). Dropping frame.");
                return false;
            }
            std::memcpy(resized_pooled_buffer->data.data(), resized_image.data, resized_size);
            resized_pooled_buffer->size = resized_size;

            image_data.width = target_width;
            image_data.height = target_height;
            image_data.buffer = std::move(resized_pooled_buffer);

        } else if (image_data.format == libcamera::formats::RGB888 || image_data.format == libcamera::formats::BGR888) {
            cv::Mat original_image(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());
            cv::Mat resized_image;
            cv::resize(original_image, resized_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

            auto resized_pooled_buffer = buffer_pool->acquire();
            if (!resized_pooled_buffer) {
                APP_LOG_WARNING(std::string(stream_name) + " failed to acquire a buffer for resized image. Dropping frame.");
                return false;
            }

            size_t resized_size = resized_image.total() * resized_image.elemSize();
            if (resized_size > resized_pooled_buffer->data.capacity()) {
                APP_LOG_WARNING(std::string(stream_name) + " resized image size (" + std::to_string(resized_size) +
                            ") exceeds buffer capacity (" + std::to_string(resized_pooled_buffer->data.capacity()) + "). Dropping frame.");
                return false;
            }
            std::memcpy(resized_pooled_buffer->data.data(), resized_image.data, resized_size);
            resized_pooled_buffer->size = resized_size;

            image_data.width = target_width;
            image_data.height = target_height;
            image_data.buffer = std::move(resized_pooled_buffer);
        } else {
            APP_LOG_ERROR("Unsupported format for resizing: " + pixelFormatToString(image_data.format));
            return false;
        }

        APP_LOG_INFO("Image resized.");
    } else {
        APP_LOG_INFO("Image already target size. No resize needed.");
    }
    auto end_time_resize = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("Time for resizing: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_time_resize - start_time_resize).count()) + " us");


    // 5. Push data to the queue.

    // 5. Push data to the queue.
    auto start_time_queue_push = std::chrono::high_resolution_clock::now();
    APP_LOG_INFO("Pushing " + std::string(stream_name) + " to queue. Final dimensions: " +
              std::to_string(image_data.width) + "x" + std::to_string(image_data.height) + 
              " format: " + pixelFormatToString(image_data.format));
    if (!queue.push(std::move(image_data))) {
        APP_LOG_WARNING(std::string(stream_name) + " queue is full. Dropping frame and returning buffer to pool.");
        // Explicitly release the buffer as the ImageData object was not moved into the queue
        image_data.buffer.reset(); 
        return false; // Indicate that the frame was dropped
    }
    auto end_time_queue_push = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("Time for queue push: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_time_queue_push - start_time_queue_push).count()) + " us");
    APP_LOG_INFO("Push successful.");

    // Log the CSV entry for this processed frame
    CsvLogEntry entry;
    entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    entry.module = "CameraCapture";
    entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    entry.event = (std::string(stream_name) == "Main Video Stream" ? "main_frame_processed" : "tpu_frame_processed");
    entry.call_ts_epoch_ms = call_ts_epoch_ms;
    entry.camera_frame_id = frame_id;
    entry.camera_width = cfg.size.width;
    entry.camera_height = cfg.size.height;
    entry.camera_exposure_ms = static_cast<float>(exposure_ms); // Cast to float
    entry.camera_copy_time_ms = static_cast<float>(copy_time_ms); // Cast to float
    Logger::getInstance().log_csv(entry);

    auto end_time_process_frame_buffer = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("Total time for process_frame_buffer: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_time_process_frame_buffer - start_time_process_frame_buffer).count()) + " us");

    return true;
}


CameraCapture::CameraCapture(unsigned int main_width, unsigned int main_height,
                             unsigned int tpu_width, unsigned int tpu_height,
                             unsigned int target_tpu_width, unsigned int target_tpu_height,
                             std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool,
                             std::list<std::reference_wrapper<ImageQueue>>& main_output_queues,
                             ImageQueue& image_processor_input_queue, // New parameter
                             std::chrono::seconds watchdog_timeout)
    : width_(main_width),
      height_(main_height),
      tpu_width_(tpu_width),
      tpu_height_(tpu_height),
      target_tpu_width_(target_tpu_width),
      target_tpu_height_(target_tpu_height),
      main_output_queues_(main_output_queues),
      image_processor_input_queue_(image_processor_input_queue), // New initializer
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
    APP_LOG_INFO("Libcamera CameraManager started.");
}

CameraCapture::~CameraCapture() {
    stop();
    if (camera_manager_) {
        camera_manager_->stop();
        APP_LOG_INFO("Libcamera CameraManager stopped.");
    }
}

/**
 * @brief Selects and acquires the camera device.
 * @return True on success, False on failure.
 */
bool CameraCapture::acquire_camera() {
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
    APP_LOG_INFO("Acquire: Camera acquired successfully.");
    return true;
}

bool CameraCapture::start() {
    APP_LOG_INFO("CameraCapture::start() called.");
    if (running_) {
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
    int ret = camera_->start();
    if (ret) {
        APP_LOG_ERROR("Failed to start camera: " + std::to_string(ret));
        camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);
        camera_->release();
        camera_.reset();
        return false;
    }
    APP_LOG_INFO("Libcamera camera started successfully.");
    APP_LOG_INFO("Libcamera camera started.");

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

    if (request_processor_thread_.joinable()) {
        request_processor_thread_.join(); // Wait for processing thread to finish
    }

    if (camera_) {
        // 3. CRITICAL: Disconnect the callback first.
        camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);

        // 4. Stop the camera. This blocks until all pending requests return (flushing).
        camera_->stop();
        APP_LOG_INFO("Libcamera camera stopped.");
        
        camera_->release();
        APP_LOG_INFO("Libcamera camera released.");
    }
    
    // 5. Free buffers
    if (allocator_) {
        if (video_stream_) allocator_->free(video_stream_);
        if (tpu_stream_) allocator_->free(tpu_stream_);
        allocator_.reset();
        APP_LOG_INFO("Libcamera buffers freed.");
    }
    
    // 6. NOW it is safe to destroy the requests and camera shared_ptr.
    requests_.clear(); // Explicit clear for Request objects
    camera_.reset();
    APP_LOG_INFO("Requests and camera pointer cleared.");

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

    // Configure main stream (index 0)
    libcamera::StreamConfiguration& mainCfg = config->at(0);
    mainCfg.pixelFormat = libcamera::formats::YUYV; // Changed from BGR888
    mainCfg.size.width = width_;
    mainCfg.size.height = height_;

    // Configure tpu stream (index 1)
    libcamera::StreamConfiguration& tpuCfg = config->at(1);
    tpuCfg.pixelFormat = libcamera::formats::RGB888; // Changed from BGR888
    tpuCfg.size.width = tpu_width_;
    tpuCfg.size.height = tpu_height_;
    
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
             ", stride: " + std::to_string(actual_stride_) + " (FOURCC: " + std::to_string(actual_pixel_format_.fourcc()) + ")");
    
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
    }

    if (tpu_stream_) {
        APP_LOG_INFO("Actual TPU Stream Config: " + std::to_string(tpu_stream_->configuration().size.width) + "x" + std::to_string(tpu_stream_->configuration().size.height) + " format: " + pixelFormatToString(tpu_stream_->configuration().pixelFormat) + " stride: " + std::to_string(tpu_stream_->configuration().stride));
    } else {
        APP_LOG_ERROR("TPU stream is null after configuration.");
    }

    // Allocate buffers for main video stream
    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    ret = allocator_->allocate(video_stream_);
    if (ret < 0) {
        APP_LOG_ERROR("Failed to allocate buffers for main video stream (Error: " + std::to_string(ret) + ").");
        allocator_.reset(); // Ensure allocator is cleaned up if it failed early
        return false;
    }
    APP_LOG_INFO("Libcamera buffers allocated for main video stream. Number of buffers: " + std::to_string(allocator_->buffers(video_stream_).size()));

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

        // Set initial controls
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
    }
    request_queue_cond_var_.notify_one(); // Notify the processing thread
    APP_LOG_DEBUG("CameraCapture: Request enqueued for processing.");

    // IMPORTANT: Do NOT call request->reuse() or queueRequest() here.
    // That will be handled by the request_processor_thread_func after processing.
}

void CameraCapture::get_performance_metrics() {
    std::lock_guard<std::mutex> lock(frame_latencies_mutex_);

    if (total_frames_processed_ == 0) {
        APP_LOG_INFO("CameraCapture: No frames processed for performance metrics.");
        return;
    }

    double average_latency_ms = 0;
    for (long long latency : frame_latencies_ms_) {
        average_latency_ms += static_cast<double>(latency); // Cast to double for accurate average
    }
    average_latency_ms /= total_frames_processed_;
    double average_fps = 1000.0 / average_latency_ms; // Inverse of average latency to get average FPS

    // double sum_sq_diff = 0; // Commented out unused variable
    // for (long long latency : frame_latencies_ms_) {
    //     sum_sq_diff += (latency - average_latency_ms) * (latency - average_latency_ms);
    // }
    // double std_dev_ms = std::sqrt(sum_sq_diff / total_frames_processed_);

    std::sort(frame_latencies_ms_.begin(), frame_latencies_ms_.end());
    size_t percentile_99_index = static_cast<size_t>(std::round(total_frames_processed_ * 0.99));
    size_t percentile_95_index = static_cast<size_t>(std::round(total_frames_processed_ * 0.95));
    size_t percentile_50_index = static_cast<size_t>(std::round(total_frames_processed_ * 0.50));

    long long p99_latency_ms = frame_latencies_ms_[std::min(percentile_99_index, static_cast<size_t>(total_frames_processed_ - 1))];
    long long p95_latency_ms = frame_latencies_ms_[std::min(percentile_95_index, static_cast<size_t>(total_frames_processed_ - 1))];
    long long p50_latency_ms = frame_latencies_ms_[std::min(percentile_50_index, static_cast<size_t>(total_frames_processed_ - 1))];

    // Populate the new CsvLogEntry fields directly
    CsvLogEntry entry; // Declare entry here
    entry.p50_latency_ms = static_cast<float>(p50_latency_ms);
    entry.p95_latency_ms = static_cast<float>(p95_latency_ms);
    entry.p99_latency_ms = static_cast<float>(p99_latency_ms);
    entry.average_fps = static_cast<float>(average_fps);
    entry.total_frames_processed_or_inferences = total_frames_processed_;
    entry.average_latency_ms = static_cast<float>(average_latency_ms);
    // Clear details field as it is now structured
    entry.details = "";

    Logger::getInstance().log_csv(entry);
    std::string header_msg = "--- CameraCapture Performance Metrics (Frame Latency) ---";
    APP_LOG_DEBUG(header_msg);
    std::string total_frames_msg = "  Total Frames Processed: " + std::to_string(total_frames_processed_);
    APP_LOG_DEBUG(total_frames_msg);
    std::string average_fps_msg = "  Average FPS: " + std::to_string(average_fps);
    APP_LOG_DEBUG(average_fps_msg);
    std::string average_latency_msg = "  Average Latency: " + std::to_string(average_latency_ms) + " ms";
    APP_LOG_DEBUG(average_latency_msg);
    std::string p50_latency_msg = "  50th Percentile Latency: " + std::to_string(p50_latency_ms) + " ms";
    APP_LOG_DEBUG(p50_latency_msg);
    APP_LOG_DEBUG("  95th Percentile Latency: " + std::to_string(p95_latency_ms) + " ms");
    APP_LOG_DEBUG("  99th Percentile Latency: " + std::to_string(p99_latency_ms) + " ms");
    APP_LOG_DEBUG("---------------------------------------------------------");

    frame_latencies_ms_.clear();
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
                                                 long long exposure_ms) {
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
        APP_LOG_WARNING("Failed to acquire a buffer for raw TPU frame. Dropping frame.");
        return false;
    }

    // This should handle YUYV, RGB888, BGR888, RGBA8888, BGRA8888 formats from libcamera
    size_t expected_bytes_per_pixel = 0;
    if (cfg.pixelFormat == libcamera::formats::YUYV) {
        expected_bytes_per_pixel = 2; // YUYV (YUV 4:2:2) is 2 bytes per pixel
    } else if (cfg.pixelFormat == libcamera::formats::RGB888 || cfg.pixelFormat == libcamera::formats::BGR888) {
        expected_bytes_per_pixel = 3; // RGB888/BGR888 is 3 bytes per pixel
    } else if (cfg.pixelFormat == libcamera::formats::RGBA8888 || cfg.pixelFormat == libcamera::formats::BGRA8888) {
        expected_bytes_per_pixel = 4; // RGBA8888/BGRA8888 is 4 bytes per pixel
    } else {
        {
        std::stringstream ss;
        ss << "Unsupported pixel format " << cfg.pixelFormat.toString().c_str() << " for raw TPU processing.";
        APP_LOG_ERROR(ss.str());
    }
        // Release the acquired buffer before returning false
        pooled_buffer.reset();
        return false;
    }
    size_t expected_payload_size = cfg.size.width * cfg.size.height * expected_bytes_per_pixel;

    if (expected_payload_size > pooled_buffer->data.capacity()) {
            {
                std::stringstream ss;
                ss << "Expected raw TPU frame size (" << expected_payload_size << ") exceeds buffer pool capacity (" << pooled_buffer->data.capacity() << "). Dropping frame.";
                APP_LOG_ERROR(ss.str());
            }        // Release the acquired buffer before returning false
        pooled_buffer.reset();
        return false;
    }

    // 2. Mmap and copy the frame data.
    void* mmap_ptr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) {
        {
        std::stringstream ss;
        ss << "Failed to mmap raw TPU plane buffer: " << strerror(errno);
        APP_LOG_ERROR(ss.str());
    }
        // Release the acquired buffer before returning false
        pooled_buffer.reset();
        return false;
    }

    size_t bytes_per_line_src = cfg.stride;
    size_t bytes_per_line_dst = cfg.size.width * expected_bytes_per_pixel;
    
    uint8_t* dst = pooled_buffer->data.data();
    uint8_t* src = static_cast<uint8_t*>(mmap_ptr);

    if (bytes_per_line_src == bytes_per_line_dst) { // No padding per line, copy in one go
        std::memcpy(dst, src, expected_payload_size);
    } else { // Padding exists, copy line by line
        for (unsigned int i = 0; i < cfg.size.height; i++) {
            std::memcpy(dst, src, bytes_per_line_dst);
            dst += bytes_per_line_dst;
            src += bytes_per_line_src;
        }
    }
    pooled_buffer->size = expected_payload_size;

    munmap(mmap_ptr, length);
    
    // Construct ImageData directly with the new constructor
    ImageData image_data(call_ts_epoch_ms, frame_id);
    image_data.width = cfg.size.width;
    image_data.height = cfg.size.height;
    image_data.buffer = std::move(pooled_buffer);
    image_data.format = cfg.pixelFormat;
    // frame_id is now set via constructor

    if (!image_processor_input_queue_.push(std::move(image_data))) {
        APP_LOG_WARNING("ImageProcessor input queue is full for raw TPU frame. Dropping frame and returning buffer to pool.");
        image_data.buffer.reset();
        return false;
    }
    {
        std::stringstream ss;
        ss << "Pushed raw TPU frame " << frame_id << " to ImageProcessor queue. Dimensions: " << image_data.width << "x" << image_data.height << ", format: " << image_data.format.toString().c_str();
        APP_LOG_DEBUG(ss.str());
    }

    return true;
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

        // --- Actual processing of the frame ---
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

        auto processing_start_time = std::chrono::high_resolution_clock::now();
        libcamera::Request::BufferMap captured_buffers = request->buffers();

        if (!main_output_queues_.empty() && captured_buffers.count(video_stream_)) {
            const libcamera::FrameBuffer* video_fb = captured_buffers.at(video_stream_);
            const libcamera::StreamConfiguration& video_cfg = video_stream_->configuration();
            process_frame_buffer(video_fb, video_cfg, image_buffer_pool_, main_output_queues_.front().get(), "Main Video Stream", width_, height_, call_ts, video_stream_->configuration().pixelFormat, frame_id, exposure_ms);
        } else if (captured_buffers.count(video_stream_)) {
            APP_LOG_DEBUG("CameraCapture: Main Video Stream frame received but no output queues configured. Dropping frame.");
        } else {
            APP_LOG_WARNING("CameraCapture: Video stream buffer missing from completed request.");
        }

        if (captured_buffers.count(tpu_stream_)) {
            const libcamera::FrameBuffer* tpu_fb = captured_buffers.at(tpu_stream_);
            const libcamera::StreamConfiguration& tpu_cfg = tpu_stream_->configuration();
            this->process_tpu_raw_frame_buffer(tpu_fb, tpu_cfg, call_ts, frame_id, exposure_ms);
        } else {
            APP_LOG_WARNING("CameraCapture: TPU stream buffer missing from completed request.");
        }
    
        // --- PERFORMANCE METRICS ---
        auto processing_end_time = std::chrono::high_resolution_clock::now();
        long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(processing_end_time - processing_start_time).count();
        {
            std::lock_guard<std::mutex> perf_lock(frame_latencies_mutex_);
            frame_latencies_ms_.push_back(duration_ms);
            total_frames_processed_++;
        }

        // --- REQUEUE ---
        request->reuse(libcamera::Request::ReuseBuffers); 
        if (running_.load()) { // Check main running_ flag for libcamera interaction
            if (camera_->queueRequest(request)) {
                APP_LOG_ERROR("Failed to re-queue request from processor thread. Stopping capture.");
                running_ = false; // Set main running flag to false
            }
        }
    }
    APP_LOG_INFO("CameraCapture: Request processor thread stopped.");
}
