#include "camera_capture.h"
#include "util_logging.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iomanip> // For std::setprecision

// Define some helper functions for libcamera callbacks
static void request_complete(libcamera::Request* request) {
    // This function will be called by libcamera when a request is complete.
    // We need to manage the lifetime of the request here.
    // For now, we just indicate it's complete.

}

CameraCapture::CameraCapture(unsigned int width, unsigned int height, ThreadSafeQueue& output_queue)
    : width_(width), height_(height), output_queue_(output_queue) {
    camera_manager_ = std::make_unique<libcamera::CameraManager>();
}

CameraCapture::~CameraCapture() {
    stop();
}

bool CameraCapture::start() {
    LOG_INFO("Starting camera capture...");
    if (running_) {
        LOG_ERROR("CameraCapture is already running.");
        return false;
    }

    int ret = camera_manager_->start();
    if (ret) {
        LOG_ERROR("Failed to start camera manager: " + std::to_string(ret));
        return false;
    }
    LOG_INFO("Camera manager started.");

    if (camera_manager_->cameras().empty()) {
        LOG_ERROR("No cameras found.");
        camera_manager_->stop();
        return false;
    }
    LOG_INFO("Found " + std::to_string(camera_manager_->cameras().size()) + " cameras.");

    camera_ = camera_manager_->cameras()[0]; // Use the first camera found
    LOG_INFO("Using camera: " + camera_->id());

    ret = camera_->acquire();
    if (ret) {
        LOG_ERROR("Failed to acquire camera: " + std::to_string(ret));
        camera_manager_->stop(); // Stop manager here too
        return false;
    }
    LOG_INFO("Camera acquired.");

    config_ = camera_->generateConfiguration({libcamera::StreamRole::Viewfinder});
    if (!config_) {
        LOG_ERROR("Failed to generate camera configuration.");
        camera_->release();
        camera_manager_->stop();
        return false;
    }
    LOG_INFO("Camera configuration generated.");

    // Set resolution
    // config_->at(0).pixelFormat = libcamera::formats::BGR888; // Let libcamera choose the format
    config_->at(0).size.width = width_;
    config_->at(0).size.height = height_;
    config_->at(0).bufferCount = 4; // Number of buffers for the stream
    LOG_INFO("Set camera resolution to " + std::to_string(width_) + "x" + std::to_string(height_));

    // Validate and apply configuration
    libcamera::CameraConfiguration::Status config_status = config_->validate();
    if (config_status == libcamera::CameraConfiguration::Invalid) {
        LOG_ERROR("Failed to validate camera configuration.");
        camera_->release();
        camera_manager_->stop();
        return false;
    } else if (config_status == libcamera::CameraConfiguration::Adjusted) {
        LOG_INFO("Camera configuration was adjusted.");
    }
    LOG_INFO("Camera configuration validated.");

    ret = camera_->configure(config_.get());
    if (ret) {
        LOG_ERROR("Failed to configure camera: " + std::to_string(ret));
        camera_->release();
        camera_manager_->stop();
        return false;
    }
    LOG_INFO("Camera configured.");
    LOG_INFO("libcamera chose pixel format: " + config_->at(0).pixelFormat.toString());

    stream_ = config_->at(0).stream();
    allocator_ = new libcamera::FrameBufferAllocator(camera_);
    ret = allocator_->allocate(stream_);
    if (ret) {
        LOG_ERROR("Failed to allocate buffers: " + std::to_string(ret));
        delete allocator_; // Clean up allocator
        allocator_ = nullptr;
        camera_->release(); // Release camera
        camera_manager_->stop(); // Stop manager
        return false;
    }
    LOG_INFO("Buffers allocated.");

    for (const auto& buffer : allocator_->buffers(stream_)) {
        std::unique_ptr<libcamera::Request> request = camera_->createRequest();
        if (!request) {
            LOG_ERROR("Failed to create request.");
            return false;
        }
        ret = request->addBuffer(stream_, buffer.get());
        if (ret) {
            LOG_ERROR("Failed to add buffer to request: " + std::to_string(ret));
            return false;
        }
        camera_->queueRequest(request.release()); // libcamera takes ownership
    }
    LOG_INFO("Requests queued.");

    camera_->requestCompleted.connect(this, &CameraCapture::process_request);

    ret = camera_->start();
    if (ret) {
        LOG_ERROR("Failed to start camera: " + std::to_string(ret));
        camera_->release();
        camera_manager_->stop();
        return false;
    }
    LOG_INFO("Camera started.");

    running_ = true;
    capture_thread_ = std::thread(&CameraCapture::capture_thread_func, this);

    LOG_INFO("CameraCapture started successfully.");
    return true;
}

void CameraCapture::stop() {
    if (running_.exchange(false)) { // Atomically set running to false and check its previous value
        output_queue_.set_running(false);
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }
    }

    // Stop the camera and manager. It's generally safe to call these
    // even if they weren't fully started, but we check for existence first.
    if (camera_) {
        camera_->stop();
        camera_->release();
    }
    if (camera_manager_) {
        camera_manager_->stop();
    }

    // The allocator is a raw pointer and must be manually deleted.
    if (allocator_) {
        // In a more complex scenario, we might need to free buffers first,
        // but deleting the allocator should handle this for now.
        delete allocator_;
        allocator_ = nullptr;
    }

    LOG_INFO("CameraCapture stopped.");
}

void CameraCapture::capture_thread_func() {
    auto last_fps_time = std::chrono::high_resolution_clock::now();
    int frames_in_period = 0;

    while (running_) {
        // libcamera operates asynchronously. Requests are queued and completed via signals.
        // The main work for this thread is to manage requests and potentially report FPS.
        // The actual frame processing happens in process_request callback.

        // The camera_->requestCompleted signal handler handles the frame data.
        // We ensure that we continuously queue requests.

        // For now, this thread primarily keeps the `running_` flag active and can report FPS.
        // A more sophisticated approach might involve waiting for a certain number of requests
        // or checking queue depth, but for basic operation, the signal handler is key.

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = now - last_fps_time;

        if (diff.count() >= 1.0) {
            double fps = frames_captured_ / diff.count();
            std::cout << "Camera FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
            frames_captured_ = 0;
            last_fps_time = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Prevent busy-waiting
    }
}

void CameraCapture::process_request(libcamera::Request* request) {
    if (request->status() == libcamera::Request::RequestComplete) {
        libcamera::FrameBuffer* buffer = request->buffers().at(stream_);
        if (buffer) {
            void* mem = nullptr;
            size_t buffer_len = 0; // Declare buffer_len here

            for (const libcamera::FrameBuffer::Plane& plane : buffer->planes()) {
                buffer_len = plane.length; // Assign buffer_len here
                int fd = plane.fd.get();   // Use plane.fd.get() to extract the raw fd

                if (fd < 0) {
                    std::cerr << "Failed to get FD for buffer plane." << std::endl;
                    continue; // Skip this plane
                }

                mem = mmap(nullptr, buffer_len, PROT_READ, MAP_SHARED, fd, 0);
                if (mem == MAP_FAILED) {
                    std::cerr << "Failed to mmap buffer." << std::endl;
                    close(fd);
                    continue; // Skip this plane
                }

                ImageData image_data;
                image_data.width = width_;
                image_data.height = height_;
                image_data.timestamp = std::chrono::high_resolution_clock::now();
                image_data.data.assign(static_cast<uint8_t*>(mem), static_cast<uint8_t*>(mem) + buffer_len);

                output_queue_.push(std::move(image_data));
                frames_captured_++;

                munmap(mem, buffer_len);
                close(fd);
            }
        }
    } else {
        std::cerr << "Request completed with status: " << request->status() << std::endl;
    }
    // Re-queue the request for the next frame
    camera_->queueRequest(request); // libcamera takes ownership
}
