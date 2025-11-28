#include "camera_capture.h"
#include "util_logging.h"

#include <opencv2/opencv.hpp> // Add this line

#include <libcamera/property_ids.h>
#include <libcamera/control_ids.h>

#include <iostream>
#include <map>
#include <iomanip>
#include <cstring> // For memcpy
#include <sys/mman.h> // For mmap, munmap
#include <unistd.h>   // For close

#include <vector> // Required for std::vector

// Helper to convert libcamera PixelFormat to a string for logging
static std::string pixelFormatToString(const libcamera::PixelFormat& format) {
    // libcamera 0.5.2 toString() provides a good description
    // Use fourcc directly as it's a stable identifier.
    std::stringstream ss;
    ss << "'" << std::hex << std::setfill('0') << std::setw(8) << format.fourcc() << "'";
    return ss.str();
}

CameraCapture::CameraCapture(unsigned int main_width, unsigned int main_height, unsigned int tpu_width, unsigned int tpu_height, std::list<std::reference_wrapper<ImageQueue>>& main_output_queues, ImageQueue& tpu_output_queue, std::chrono::seconds watchdog_timeout)
    : width_(main_width), height_(main_height), tpu_width_(tpu_width), tpu_height_(tpu_height), main_output_queues_(main_output_queues), tpu_output_queue_(tpu_output_queue), watchdog_timeout_(watchdog_timeout), last_frame_time_(std::chrono::high_resolution_clock::now()) {
    // Initialize CameraManager
    camera_manager_ = std::make_unique<libcamera::CameraManager>();
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

bool CameraCapture::start() {
    if (running_) {
        LOG_ERROR("CameraCapture is already running.");
        return false;
    }

    if (!setup_camera()) {
        LOG_ERROR("Failed to setup libcamera camera.");
        return false;
    }

    // Connect request completed signal
    camera_->requestCompleted.connect(this, &CameraCapture::request_complete_callback);

    // Start the camera
    int ret = camera_->start();
    if (ret) {
        LOG_ERROR("Failed to start camera: " + std::to_string(ret));
        camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);
        camera_->release();
        return false;
    }
    LOG_INFO("Libcamera camera started.");

    running_ = true;
    for (auto& queue_ref : main_output_queues_) {
        queue_ref.get().set_running(true);
    }
    
    // Queue initial requests
    frame_count_ = 0;
    last_frame_time_ = std::chrono::high_resolution_clock::now();
    for (std::unique_ptr<libcamera::Request>& req_ptr : requests_) { 
        if (camera_->queueRequest(req_ptr.release())) { // Release unique_ptr ownership
            LOG_ERROR("Failed to queue initial request.");
            running_ = false; // Signal failure
            stop(); // Attempt to clean up
            return false;
        }
    }
    requests_.clear(); // Unique_ptrs moved, so vector is logically empty

    LOG_INFO("CameraCapture: Initial requests queued.");
    return true;
}

void CameraCapture::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    LOG_INFO("Stopping CameraCapture...");

    for (auto& queue_ref : main_output_queues_) {
        queue_ref.get().set_running(false);
    }

    if (camera_) {
        // Disconnect the signal before stopping the camera
        camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);

        // Stop all outstanding requests, which should return them via callbacks
        camera_->stop();
        LOG_INFO("Libcamera camera stopped.");
        
        camera_->release();
        camera_.reset(); // Release shared_ptr
        LOG_INFO("Libcamera camera released.");
    }
    
    if (allocator_) {
        allocator_->free(video_stream_); // Free buffers explicitly
        if (tpu_stream_) allocator_->free(tpu_stream_);
        allocator_.reset(); // Release FrameBufferAllocator
        LOG_INFO("Libcamera buffers freed.");
    }
    
    // requests_ vector elements (unique_ptrs) will be destroyed when CameraCapture is destroyed.
    // No explicit clear needed here.
    // requests_.clear(); // Already handled by unique_ptr destructor of the vector.

    LOG_INFO("CameraCapture stopped.");
}

bool CameraCapture::setup_camera() {
    // Select the camera
    if (camera_manager_->cameras().empty()) {
        LOG_ERROR("No cameras found by libcamera.");
        return false;
    }
    camera_ = camera_manager_->cameras().front(); // Use the first camera found
    
    int ret = camera_->acquire();
    if (ret) {
        LOG_ERROR("Failed to acquire camera: " + std::to_string(ret));
        return false;
    }
    LOG_INFO("Libcamera camera acquired.");

    // Configure dual streams: main high-res and TPU viewfinder
    std::vector<libcamera::StreamRole> roles = {
        libcamera::StreamRole::VideoRecording, // main high-res
        libcamera::StreamRole::Viewfinder      // TPU/resized
    };
    std::unique_ptr<libcamera::CameraConfiguration> config = camera_->generateConfiguration(roles);

    if (!config) {
        LOG_ERROR("Failed to generate dual stream configuration.");
        camera_->release();
        return false;
    }

    // Check if the generated configuration contains exactly two streams
    if (config->size() < 2) {
        LOG_ERROR("Generated camera configuration has less than two streams for dual-stream setup.");
        camera_->release();
        return false;
    }

    // Configure main stream (index 0)
    libcamera::StreamConfiguration& mainCfg = config->at(0);
    mainCfg.pixelFormat = libcamera::formats::YUV420;
    mainCfg.size.width = width_;
    mainCfg.size.height = height_;

    // Configure tpu stream (index 1)
    libcamera::StreamConfiguration& tpuCfg = config->at(1);
    tpuCfg.pixelFormat = libcamera::formats::YUV420;
    tpuCfg.size.width = tpu_width_;
    tpuCfg.size.height = tpu_height_;
    
    // Validate and complete the configuration
    libcamera::CameraConfiguration::Status config_status = config->validate();
    LOG_INFO(std::string("CameraConfiguration validate() -> ") + std::to_string(config_status));
    
    if (config_status == libcamera::CameraConfiguration::Invalid) {
        LOG_ERROR("Invalid dual stream camera configuration.");
        camera_->release();
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
        LOG_ERROR("Failed to configure camera streams: " + std::to_string(ret));
        camera_->release();
        return false;
    }
    LOG_INFO("Libcamera dual streams configured.");
    
    video_stream_ = mainCfg.stream();
    tpu_stream_ = tpuCfg.stream();

    // Allocate buffers for main video stream
    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    ret = allocator_->allocate(video_stream_);
    if (ret < 0) {
        LOG_ERROR("Failed to allocate buffers for main video stream: " + std::to_string(ret));
        camera_->release();
        return false;
    }
    LOG_INFO("Libcamera buffers allocated for main video stream. Number of buffers: " + std::to_string(allocator_->buffers(video_stream_).size()));

    // Allocate buffers for TPU stream
    ret = allocator_->allocate(tpu_stream_);
    if (ret < 0) {
        LOG_ERROR("Failed to allocate buffers for TPU stream: " + std::to_string(ret));
        allocator_->free(video_stream_); // Free previously allocated buffers
        camera_->release();
        return false;
    }
    LOG_INFO("Libcamera buffers allocated for TPU stream. Number of buffers: " + std::to_string(allocator_->buffers(tpu_stream_).size()));

    // Create requests
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& video_buffers = allocator_->buffers(video_stream_);
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& tpu_buffers = allocator_->buffers(tpu_stream_);

    if (video_buffers.size() != tpu_buffers.size()) {
        LOG_ERROR("Mismatched buffer counts between main video stream and TPU stream.");
        camera_->release();
        return false;
    }

    for (unsigned int i = 0; i < video_buffers.size(); ++i) { // Iterate based on main stream buffer count
        std::unique_ptr<libcamera::Request> request = camera_->createRequest();
        if (!request) {
            LOG_ERROR("Failed to create request.");
            return false;
        }

        // Do not copy all controls. Start with an empty list and add specifics.
        // libcamera::ControlList& controls = request->controls();
        // controls = camera_->controls(); // Incorrect!
        request->controls().set(libcamera::controls::AeEnable, true);
        request->controls().set(libcamera::controls::AnalogueGain, 1.0f); // Set a default/starting gain
        request->controls().set(libcamera::controls::ExposureTime, 20000); // Set a default/starting exposure in microseconds

        // Add buffer for main video stream
        ret = request->addBuffer(video_stream_, video_buffers[i].get()); 
        if (ret) {
            LOG_ERROR("Failed to add buffer for main video stream to request: " + std::to_string(ret));
            return false;
        }

        // Add buffer for TPU stream
        ret = request->addBuffer(tpu_stream_, tpu_buffers[i].get());
        if (ret) {
            LOG_ERROR("Failed to add buffer for TPU stream to request: " + std::to_string(ret));
            return false;
        }

        requests_.push_back(std::move(request));
    }
    LOG_INFO("Libcamera requests created and buffers added for both streams.");
    return true;
}

void CameraCapture::request_complete_callback(libcamera::Request* request) {
    if (!running_) {
        // If not running, just re-queue the request so libcamera can clean it up.
        request->reuse();
        camera_->queueRequest(request);
        return;
    }

    if (request->status() != libcamera::Request::RequestComplete) {
        if (request->status() == libcamera::Request::RequestCancelled) {
            LOG_INFO("CameraCapture: Request cancelled, likely during shutdown.");
        } else {
            LOG_ERROR("CameraCapture: Request completion failed with status: " + std::to_string(request->status()));
        }
        request->reuse();
        camera_->queueRequest(request);
        return;
    }

    // The request is complete, so we can process the buffers.
    // We must capture the buffer map here, as reuse() will clear it from the request.
    libcamera::Request::BufferMap captured_buffers = request->buffers();

    // Process Main Video Stream
    const libcamera::FrameBuffer* video_fb = captured_buffers.at(video_stream_);
    const libcamera::StreamConfiguration& video_cfg = video_stream_->configuration();
    const libcamera::FrameBuffer::Plane& video_plane = video_fb->planes()[0];

    void* video_mmap_ptr = mmap(NULL, video_plane.length, PROT_READ, MAP_SHARED, video_plane.fd.get(), 0);
    if (video_mmap_ptr == MAP_FAILED) {
        LOG_ERROR("CameraCapture: Failed to mmap video stream frame buffer: " + std::string(strerror(errno)));
    } else {
        ImageData video_image_data;
        convert_yuv420_to_rgb888(video_fb, video_stream_->configuration(), video_image_data);
        video_image_data.timestamp = std::chrono::high_resolution_clock::now();

        for (auto& queue_ref : main_output_queues_) {
            queue_ref.get().push_and_drop_if_full(video_image_data);
        }
        munmap(video_mmap_ptr, video_plane.length);
    }

    // Process TPU Stream
    const libcamera::FrameBuffer* tpu_fb = captured_buffers.at(tpu_stream_);
    // const libcamera::StreamConfiguration& tpu_cfg = tpu_stream_->configuration(); // Not directly needed here anymore
    const libcamera::FrameBuffer::Plane& tpu_plane = tpu_fb->planes()[0]; // Still need this for mmap

    void* tpu_mmap_ptr = mmap(NULL, tpu_plane.length, PROT_READ, MAP_SHARED, tpu_plane.fd.get(), 0);
    if (tpu_mmap_ptr == MAP_FAILED) {
        LOG_ERROR("CameraCapture: Failed to mmap TPU stream frame buffer: " + std::string(strerror(errno)));
    } else {
        ImageData tpu_image_data;
        convert_yuv420_to_rgb888(tpu_fb, tpu_stream_->configuration(), tpu_image_data);
        tpu_image_data.timestamp = std::chrono::high_resolution_clock::now();
        tpu_output_queue_.push_and_drop_if_full(tpu_image_data);
        munmap(tpu_mmap_ptr, tpu_plane.length);
    }
    
    // --- FPS Calculation ---
    frame_count_++;
    if (frame_count_ % kFpsReportInterval == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - last_frame_time_;
        double fps = kFpsReportInterval / elapsed.count();
        LOG_INFO("--- FPS Report (Callback): " + std::to_string(fps) + " ---");
        last_frame_time_ = now;
    }

    // Reuse the request and re-add the buffers before re-queueing.
    request->reuse();
    for (auto const& [stream, buffer] : captured_buffers) {
        if (request->addBuffer(stream, const_cast<libcamera::FrameBuffer*>(buffer)) < 0) {
            LOG_ERROR("Failed to re-add buffer to reused request");
            // If we fail here, we should not queue the request. Let's stop to be safe.
            stop();
            return;
        }
    }
    
    camera_->queueRequest(request);
}

void CameraCapture::convert_yuv420_to_rgb888(const libcamera::FrameBuffer* buffer, const libcamera::StreamConfiguration& stream_config, ImageData& rgb_image) {
    unsigned int width = stream_config.size.width;
    unsigned int height = stream_config.size.height;
    unsigned int y_stride = stream_config.stride; // Stride for Y plane

    // Check if the buffer has enough planes for YUV420 (Y, U, V)
    if (buffer->planes().size() < 3) {
        LOG_ERROR("FrameBuffer does not have enough planes for YUV420 conversion.");
        return;
    }

    // Store mmap'd pointers and lengths for cleanup
    std::vector<void*> mmap_ptrs;
    std::vector<size_t> mmap_lengths;

    // RAII for munmap: ensure all mapped memory is unmapped on exit
    auto munmap_guard = [&]() {
        for (size_t i = 0; i < mmap_ptrs.size(); ++i) {
            if (mmap_ptrs[i] != MAP_FAILED) {
                munmap(mmap_ptrs[i], mmap_lengths[i]);
            }
        }
    };

    // Map all planes
    for (const auto& plane : buffer->planes()) {
        void* ptr = mmap(NULL, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
        if (ptr == MAP_FAILED) {
            LOG_ERROR("CameraCapture: Failed to mmap YUV plane: " + std::string(strerror(errno)));
            munmap_guard(); // Clean up already mapped memory
            return;
        }
        mmap_ptrs.push_back(ptr);
        mmap_lengths.push_back(plane.length);
    }

    // Assuming YUV420 layout: Y plane (full size), U plane (quarter size), V plane (quarter size)
    size_t y_size_bytes = width * height;
    size_t uv_width = width / 2;
    size_t uv_height = height / 2;
    size_t uv_stride = y_stride / 2; // Assuming U/V stride is half of Y stride, typical for 420

    // Create a temporary buffer to hold the combined YUV data in I420 format.
    // I420 is a planar format: Y data, then U data, then V data.
    std::vector<uint8_t> yuv_combined_data(y_size_bytes + uv_width * uv_height + uv_width * uv_height);

    // Copy Y plane data, respecting its stride
    const uint8_t* y_src_ptr = static_cast<const uint8_t*>(mmap_ptrs[0]);
    for(unsigned int r = 0; r < height; ++r) {
        std::memcpy(yuv_combined_data.data() + r * width, y_src_ptr + r * y_stride, width);
    }

    // Copy U plane data, respecting its stride
    const uint8_t* u_src_ptr = static_cast<const uint8_t*>(mmap_ptrs[1]);
    for(unsigned int r = 0; r < uv_height; ++r) {
        std::memcpy(yuv_combined_data.data() + y_size_bytes + r * uv_width, u_src_ptr + r * uv_stride, uv_width);
    }

    // Copy V plane data, respecting its stride
    const uint8_t* v_src_ptr = static_cast<const uint8_t*>(mmap_ptrs[2]);
    for(unsigned int r = 0; r < uv_height; ++r) {
        std::memcpy(yuv_combined_data.data() + y_size_bytes + uv_width * uv_height + r * uv_width, v_src_ptr + r * uv_stride, uv_width);
    }
    
    // Create a cv::Mat header for the combined YUV data.
    // The height for I420 in OpenCV is typically (height * 3 / 2) if it's a single Mat
    cv::Mat yuv_mat(height * 3 / 2, width, CV_8UC1, yuv_combined_data.data());
    
    // Create the destination RGB mat.
    cv::Mat rgb_mat;

    // Perform the color conversion.

    // Copy the data from the cv::Mat to the ImageData structure.
    rgb_image.width = rgb_mat.cols;
    rgb_image.height = rgb_mat.rows;
    size_t data_size = rgb_mat.total() * rgb_mat.elemSize();
    rgb_image.data.resize(data_size);
    std::memcpy(rgb_image.data.data(), rgb_mat.data, data_size);

    munmap_guard(); // Clean up all mapped memory
}