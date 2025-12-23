#include <libcamera/libcamera.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/stream.h>
#include <libcamera/request.h>
#include <libcamera/geometry.h>
#include <libcamera/pixel_format.h>
#include <libcamera/control_ids.h>
#include <libcamera/formats.h>

#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <condition_variable>
#include <queue>

class CameraIsolationTest {
public:
    CameraIsolationTest(unsigned int main_width, unsigned int main_height,
                       unsigned int tpu_width, unsigned int tpu_height,
                       unsigned int tpu_fps)
        : width_(main_width), height_(main_height),
          tpu_width_(tpu_width), tpu_height_(tpu_height),
          tpu_fps_(tpu_fps) {
        std::cout << "CameraIsolationTest constructor called with main=" << main_width << "x" << main_height <<
                     ", tpu=" << tpu_width << "x" << tpu_height << "@" << tpu_fps << "fps" << std::endl;
    }

    ~CameraIsolationTest() {
        std::cout << "CameraIsolationTest destructor called." << std::endl;
        stop();
    }

    bool start() {
        std::cout << "CameraIsolationTest::start() called." << std::endl;
        
        if (running_.load()) {
            std::cout << "CameraIsolationTest is already running." << std::endl;
            return false;
        }
        
        // Start the camera manager
        int ret = camera_manager_->start();
        if (ret) {
            std::cout << "Failed to start CameraManager: " << ret << std::endl;
            camera_manager_.reset();
            return false;
        }
        
        // 1. Acquire the camera 
        if (!acquire_camera()) {
            std::cout << "CameraIsolationTest failed to acquire camera. Cannot proceed." << std::endl;
            return false;
        }
        std::cout << "Camera acquired successfully. Proceeding to setup." << std::endl;

        // CRITICAL SECONDARY CHECK: Should not be necessary, but defensively prevents a nullptr crash.
        if (!camera_) {
            std::cout << "CRITICAL: Camera pointer is null immediately after successful acquire_camera(). Aborting." << std::endl;
            return false;
        }
        
        // 2. Setup (Configure streams, allocate buffers, create requests)
        if (!setup_camera()) {
            std::cout << "Failed to setup camera. Releasing camera." << std::endl;
            
            // Safety check before release(): Only release if the pointer is still valid.
            if (camera_) {
                camera_->release();
            }
            camera_.reset();
            return false;
        }
        std::cout << "Camera setup completed successfully." << std::endl;
        
        // 3. Connect request completed signal
        camera_->requestCompleted.connect(this, &CameraIsolationTest::request_complete_callback);

        // 4. Start Capture
        // Prepare controls to set during camera start
        libcamera::ControlList controls_to_set;
        
        // Set frame rate to 120 FPS using FrameDurationLimits for both streams
        if (tpu_fps_ > 0) {
            int64_t frame_duration_us = 1000000 / tpu_fps_; // Calculate microseconds per frame
            controls_to_set.set(libcamera::controls::FrameDurationLimits, {frame_duration_us, frame_duration_us});
            std::cout << "Setting FrameDurationLimits to: " << frame_duration_us << " us (" << tpu_fps_ << " FPS)" << std::endl;
        }
        
        // Also set the target frame rate for the camera
        if (tpu_fps_ > 0) {
            controls_to_set.set(libcamera::controls::AeEnable, true);
            std::cout << "Setting AE enable for target FPS: " << tpu_fps_ << std::endl;
        }

        int start_ret = camera_->start(&controls_to_set);
        if (start_ret) {
            std::cout << "Failed to start camera: " << start_ret << std::endl;
            camera_->requestCompleted.disconnect(this, &CameraIsolationTest::request_complete_callback);
            camera_->release();
            camera_.reset();
            return false;
        }
        std::cout << "Libcamera camera started successfully." << std::endl;
        
        running_ = true;
        processing_running_ = true; // Set new flag for processing thread
        request_processor_thread_ = std::thread(&CameraIsolationTest::request_processor_thread_func, this);
        
        // 5. Queue initial requests
        frame_count_ = 0;
        measurement_start_time_ = std::chrono::steady_clock::now();
        measurement_start_time_ = std::chrono::steady_clock::now();
        
        for (auto& req_ptr : requests_) {
            if (camera_->queueRequest(req_ptr.get())) { 
                std::cout << "Failed to queue initial request." << std::endl;
                // Stop to clean up resources if queueing fails.
                running_ = false; 
                stop(); 
                return false;
            }
        }
        std::cout << "CameraIsolationTest: All initial requests successfully queued." << std::endl;
        return true;
    }

    void stop() {
        // 1. Signal shutdown to threads
        if (!running_.exchange(false)) { // Signal main thread to stop
            return;
        }
        std::cout << "Stopping CameraIsolationTest..." << std::endl;

        processing_running_ = false; // Signal processing thread to stop
        request_queue_cond_var_.notify_one(); // Wake up processing thread

        // Use try-catch blocks to ensure all cleanup steps are attempted even if some fail
        try {
            if (request_processor_thread_.joinable()) {
                request_processor_thread_.join(); // Wait for processing thread to finish
            }
        } catch (const std::exception& e) {
            std::cout << "Exception while joining request processor thread: " << e.what() << std::endl;
        }

        try {
            if (camera_) {
                // 3. CRITICAL: Disconnect the callback first.
                try {
                    camera_->requestCompleted.disconnect(this, &CameraIsolationTest::request_complete_callback);
                } catch (const std::exception& e) {
                    std::cout << "Exception while disconnecting callback: " << e.what() << std::endl;
                }

                // 4. Stop the camera. This blocks until all pending requests return (flushing).
                try {
                    camera_->stop();
                    std::cout << "Libcamera camera stopped." << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "Exception while stopping camera: " << e.what() << std::endl;
                }
                
                try {
                    camera_->release();
                    std::cout << "Libcamera camera released." << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "Exception while releasing camera: " << e.what() << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cout << "Exception in camera cleanup section: " << e.what() << std::endl;
        }
        
        // 5. Free buffers
        try {
            if (allocator_) {
                try {
                    if (video_stream_) allocator_->free(video_stream_);
                } catch (const std::exception& e) {
                    std::cout << "Exception while freeing video stream buffers: " << e.what() << std::endl;
                }
                
                try {
                    if (tpu_stream_) allocator_->free(tpu_stream_);
                } catch (const std::exception& e) {
                    std::cout << "Exception while freeing TPU stream buffers: " << e.what() << std::endl;
                }
                
                allocator_.reset();
                std::cout << "Libcamera buffers freed." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Exception in buffer allocator cleanup: " << e.what() << std::endl;
        }
        
        // 6. NOW it is safe to destroy the requests and camera shared_ptr.
        try {
            requests_.clear(); // Explicit clear for Request objects
            camera_.reset();
            std::cout << "Requests and camera pointer cleared." << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Exception while clearing requests or resetting camera: " << e.what() << std::endl;
        }
        
        std::cout << "CameraIsolationTest stopped." << std::endl;
    }

    bool acquire_camera() {
        if (!camera_manager_) {
            std::cout << "Acquire: FAILURE - CameraManager is null." << std::endl;
            return false;
        }
        
        auto cameras = camera_manager_->cameras();
        if (cameras.empty()) {
            std::cout << "Acquire: FAILURE - No cameras found by libcamera. Check camera connection and driver." << std::endl;
            return false;
        }

        // Assign the selected camera to the member variable camera_
        camera_ = cameras.front();
        if (!camera_) {
            std::cout << "Acquire: FAILURE - CameraManager returned a null camera pointer for first camera." << std::endl;
            return false;
        }

        std::cout << "Selected Camera ID: " << camera_->id() << std::endl;
        
        int ret = camera_->acquire();
        if (ret) {
            std::cout << "Acquire: FAILURE - Failed to acquire camera (Error: " << ret << "). Is the camera in use by another process?" << std::endl;
            camera_.reset();
            return false;
        }
        
        std::cout << "Camera acquired successfully." << std::endl;
        return true;
    }

    bool setup_camera() {
        // --- SAFETY CHECK ---
        if (!camera_) {
            std::cout << "CRITICAL: setup_camera called but camera_ is nullptr! Aborting setup." << std::endl;
            return false;
        }
        
        std::cout << "setup_camera: Configuring camera: " << camera_->id() << std::endl;
        int ret = 0; 

        // Configure dual streams: main high-res and TPU viewfinder
        std::vector<libcamera::StreamRole> roles = {
            libcamera::StreamRole::VideoRecording, // main high-res
            libcamera::StreamRole::Viewfinder      // TPU/resized
        };
        
        std::unique_ptr<libcamera::CameraConfiguration> config = camera_->generateConfiguration(roles);

        if (!config) {
            std::cout << "Failed to generate dual stream configuration." << std::endl;
            return false;
        }

        // Check if the generated configuration contains exactly two streams
        if (config->size() < 2) {
            std::cout << "Generated camera configuration has less than two streams for dual-stream setup (found " << config->size() << ")." << std::endl;
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
            std::cout << "Main stream configuration does not match expected 1536x864 resolution" << std::endl;
        }

        // Configure tpu stream (index 1) for 320x320 resolution
        libcamera::StreamConfiguration& tpuCfg = config->at(1);
        tpuCfg.pixelFormat = libcamera::formats::RGB888;
        tpuCfg.size.width = 320;   // Fixed to TPU input size
        tpuCfg.size.height = 320;  // Fixed to TPU input size
        tpuCfg.bufferCount = 8; // Increase buffer count for high frame rate
        
        // Validate TPU stream configuration
        if (tpuCfg.size.width != 320 || tpuCfg.size.height != 320) {
            std::cout << "TPU stream configuration does not match expected 320x320 resolution" << std::endl;
        }
        
        // Validate and complete the configuration
        libcamera::CameraConfiguration::Status config_status = config->validate();
        std::cout << "CameraConfiguration validate() -> Status: " << config_status << std::endl;
        
        if (config_status == libcamera::CameraConfiguration::Invalid) {
            std::cout << "Invalid dual stream camera configuration. Check requested resolutions/formats." << std::endl;
            return false;
        } else if (config_status == libcamera::CameraConfiguration::Adjusted) {
            std::cout << "Camera configuration adjusted by libcamera for dual streams." << std::endl;
        }
        
        ret = camera_->configure(config.get());
        if (ret) {
            std::cout << "Failed to configure camera streams (Error: " << ret << ")." << std::endl;
            return false;
        }
        std::cout << "Libcamera dual streams configured." << std::endl;

        video_stream_ = mainCfg.stream();
        tpu_stream_ = tpuCfg.stream();

        if (video_stream_) {
            std::cout << "Actual Video Stream Config: " << video_stream_->configuration().size.width << "x" << video_stream_->configuration().size.height << " format" << std::endl;
        } else {
            std::cout << "Video stream is null after configuration." << std::endl;
            return false;
        }

        if (tpu_stream_) {
            std::cout << "Actual TPU Stream Config: " << tpu_stream_->configuration().size.width << "x" << tpu_stream_->configuration().size.height << " format" << std::endl;
        } else {
            std::cout << "TPU stream is null after configuration." << std::endl;
            return false;
        }

        // Allocate buffers for main video stream
        allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
        if (!allocator_) {
            std::cout << "Failed to create FrameBufferAllocator." << std::endl;
            return false;
        }
        
        ret = allocator_->allocate(video_stream_);
        if (ret < 0) {
            std::cout << "Failed to allocate buffers for main video stream (Error: " << ret << ")." << std::endl;
            allocator_.reset(); // Ensure allocator is cleaned up if it failed early
            return false;
        }
        std::cout << "Libcamera buffers allocated for main video stream. Number of buffers: " << allocator_->buffers(video_stream_).size() << std::endl;

        // Increase buffer count for high frame rate support
        if (tpu_fps_ >= 120) {
            std::cout << "High frame rate mode detected, ensuring adequate buffer allocation" << std::endl;
        }

        // Allocate buffers for TPU stream
        ret = allocator_->allocate(tpu_stream_);
        if (ret < 0) {
            std::cout << "Failed to allocate buffers for TPU stream (Error: " << ret << ")." << std::endl;
            allocator_.reset();
            return false;
        }
        std::cout << "Libcamera buffers allocated for TPU stream. Number of buffers: " << allocator_->buffers(tpu_stream_).size() << std::endl;

        // Create requests
        const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& video_buffers = allocator_->buffers(video_stream_);
        const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& tpu_buffers = allocator_->buffers(tpu_stream_);

        if (video_buffers.size() != tpu_buffers.size()) {
            std::cout << "Mismatched buffer counts between main video stream and TPU stream. This is unexpected." << std::endl;
            return false;
        }
        
        requests_.clear(); // Ensure clean slate before populating
        
        for (unsigned int i = 0; i < video_buffers.size(); ++i) {
            std::unique_ptr<libcamera::Request> request = camera_->createRequest();
            if (!request) {
                std::cout << "Failed to create request." << std::endl;
                return false;
            }

            // Set initial controls for high FPS
            request->controls().set(libcamera::controls::AeEnable, true);
            
            ret = request->addBuffer(video_stream_, video_buffers[i].get(), std::unique_ptr<libcamera::Fence>()); 
            if (ret) {
                std::cout << "Failed to add main buffer to request (Error: " << ret << ")." << std::endl;
                return false;
            }

            ret = request->addBuffer(tpu_stream_, tpu_buffers[i].get(), std::unique_ptr<libcamera::Fence>());
            if (ret) {
                std::cout << "Failed to add TPU buffer to request (Error: " << ret << ")." << std::endl;
                return false;
            }

            requests_.push_back(std::move(request));
        }
        std::cout << "Libcamera requests created and buffers added for both streams." << std::endl;
        
        return true;
    }

    void request_complete_callback(libcamera::Request* request) {
        // If the main CameraIsolationTest is stopping, or processing thread is stopping, just re-queue and return.
        // The processing_running_ flag controls the processing thread.
        if (!running_.load() || !processing_running_.load()) {
            if (request->status() != libcamera::Request::RequestCancelled) {
                request->reuse(libcamera::Request::ReuseBuffers);
                if (camera_ && camera_->queueRequest(request)) {
                    std::cout << "Failed to re-queue request during CameraIsolationTest shutdown." << std::endl;
                }
            }
            return;
        }

        if (request->status() == libcamera::Request::RequestCancelled) {
            request->reuse(libcamera::Request::ReuseBuffers); 
            if (camera_ && camera_->queueRequest(request)) {
                std::cout << "Failed to re-queue cancelled request." << std::endl;
            }
            return;
        }
        
        if (request->status() != libcamera::Request::RequestComplete) {
            std::cout << "CameraIsolationTest: Request failed with status: " << request->status() << std::endl;
            request->reuse(libcamera::Request::ReuseBuffers); 
            if (camera_ && camera_->queueRequest(request)) {
                std::cout << "Failed to re-queue failed request." << std::endl;
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

    void request_processor_thread_func() {
        std::cout << "CameraIsolationTest: Request processor thread started." << std::endl;
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
                std::cout << "CameraIsolationTest: Received null request in processor thread." << std::endl;
                continue;
            }

            // --- Count the frame ---
            frame_count_++;
            
            // Print FPS every second
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - measurement_start_time_).count();
            if (elapsed_time >= 1000) {
                double fps = (double)frame_count_ / (elapsed_time / 1000.0);
                std::cout << "=== CAMERA CAPTURE RESULTS ===" << std::endl;
                std::cout << "Frames captured per second: " << std::fixed << std::setprecision(2) << fps << std::endl;
                std::cout << "Number of in-flight requests: " << requests_.size() << std::endl;
                std::cout << "Consumer-gated: NO" << std::endl;
                std::cout << "Reason for stall: none" << std::endl;
                std::cout << "==============================" << std::endl;
                
                // Reset counters
                frame_count_ = 0;
                measurement_start_time_ = current_time;
            }

            // Immediately re-queue the request without any processing
            request->reuse(libcamera::Request::ReuseBuffers); 
            if (running_.load()) { // Check main running_ flag for libcamera interaction
                if (camera_ && camera_->queueRequest(request)) {
                    std::cout << "Failed to re-queue request from processor thread. Stopping capture." << std::endl;
                    running_ = false; // Set main running flag to false
                }
            }
        }
        std::cout << "CameraIsolationTest: Request processor thread stopped." << std::endl;
    }

private:
    unsigned int width_; ///< Breedte van de hoofdstream.
    unsigned int height_; ///< Hoogte van de hoofdstream.
    unsigned int tpu_width_; ///< Breedte van de TPU-stream.
    unsigned int tpu_height_; ///< Hoogte van de TPU-stream.
    unsigned int tpu_fps_; ///< Frame rate voor de TPU-stream.

    std::unique_ptr<libcamera::CameraManager> camera_manager_{std::make_unique<libcamera::CameraManager>()}; ///< Beheert de beschikbare camera's.
    std::shared_ptr<libcamera::Camera> camera_; ///< De geselecteerde camera.
    libcamera::Stream* video_stream_ = nullptr; ///< De hoge-resolutie videostream.
    libcamera::Stream* tpu_stream_ = nullptr; ///< De lage-resolutie TPU-stream.
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_; ///< Alloceert frame-buffers.
    std::vector<std::unique_ptr<libcamera::Request>> requests_; ///< Vector van camera requests.
    
    std::atomic<bool> running_ = false; ///< Vlag om de state van de worker thread te beheren.
    std::atomic<bool> processing_running_ = false;
    
    int frame_count_ = 0; ///< Teller voor het aantal verwerkte frames.
    std::chrono::steady_clock::time_point measurement_start_time_;
    
    // Members for dedicated request processing thread
    std::thread request_processor_thread_;
    std::queue<libcamera::Request*> request_queue_;
    std::mutex request_queue_mutex_;
    std::condition_variable request_queue_cond_var_;
};

int main() {
    std::cout << "Starting camera isolation test..." << std::endl;
    
    // Create test instance with 120 FPS configuration
    CameraIsolationTest camera_test(1536, 864, 320, 320, 120);
    
    // Start the camera
    if (!camera_test.start()) {
        std::cout << "Failed to start camera isolation test." << std::endl;
        return 1;
    }
    
    // Run for 5 seconds
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // Stop the camera
    camera_test.stop();
    
    std::cout << "Camera isolation test completed." << std::endl;
    return 0;
}