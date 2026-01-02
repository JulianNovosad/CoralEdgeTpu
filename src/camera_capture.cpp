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
#include <future>
#include <thread>

#include "camera_capture.h"
#include "util_logging.h"
#include "timing.h"  // For get_time_raw_ms()

#include <opencv2/opencv.hpp>

#include <libcamera/property_ids.h>
#include <libcamera/control_ids.h>
#include <libcamera/formats.h>

#include <iostream>
#include <map>
#include <iomanip>

#include "application.h"  // For Application counter updates

// Helper to convert libcamera PixelFormat to a string for logging
#ifdef DEBUG_MODE
static std::string pixelFormatToString(const libcamera::PixelFormat& format) {
    std::stringstream ss;
    ss << "'" << std::hex << std::setfill('0') << std::setw(8) << format.fourcc() << "'";
    return ss.str();
}
#endif

struct MappedBufferInfo {
    void* addr;
    size_t length;
};

static bool process_frame_buffer(const libcamera::FrameBuffer* fb,
                                 const libcamera::StreamConfiguration& cfg,
                                 std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                                 std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                                 ImageQueue& queue,
                                 const char* stream_name,
                                 [[maybe_unused]] unsigned int target_width,
                                 [[maybe_unused]] unsigned int target_height,
                                 std::chrono::steady_clock::time_point capture_time,
                                 uint64_t t_capture_raw_ms,
                                 const libcamera::PixelFormat& actual_format,
                                 long long frame_id,
                                 [[maybe_unused]] long long exposure_ms,
                                 std::atomic<int64_t>* main_stream_drop_counter,
                                 std::atomic<int64_t>* tpu_stream_drop_counter,
                                 std::atomic<int64_t>* frames_produced_counter,
                                 Application* app_ref,
                                 const std::map<const libcamera::FrameBuffer*, MappedBufferInfo>& mapped_buffers)
{
    auto process_start = std::chrono::steady_clock::now();
    size_t total_length = 0;
    for (const auto& plane : fb->planes()) {
        total_length += plane.length;
    }

    const libcamera::FrameBuffer::Plane& plane0 = fb->planes()[0];
    int fd = plane0.fd.get();

    auto pooled_buffer = buffer_pool->acquire();
    if (!pooled_buffer) {
        APP_LOG_ERROR(std::string(stream_name) + " Failed to acquire buffer from pool.");
        return false;
    }

    // Use persistent mapping if available, otherwise fallback to temporary mmap
    void* mapped_memory = nullptr;
    bool is_persistent = false;
    auto it = mapped_buffers.find(fb);
    if (it != mapped_buffers.end()) {
        mapped_memory = it->second.addr;
        is_persistent = true;
    } else {
        mapped_memory = mmap(nullptr, total_length, PROT_READ, MAP_SHARED, fd, 0);
    }

    if (mapped_memory == MAP_FAILED || mapped_memory == nullptr) {
        APP_LOG_ERROR(std::string(stream_name) + " Failed to map buffer: " + std::strerror(errno));
        return false;
    }

    if (pooled_buffer->data.size() < total_length) {
        pooled_buffer->data.resize(total_length);
    }
    std::memcpy(pooled_buffer->data.data(), mapped_memory, total_length);
    pooled_buffer->size = total_length;

    if (!is_persistent) {
        if (munmap(mapped_memory, total_length) == -1) {
            APP_LOG_ERROR(std::string(stream_name) + " Failed to unmap buffer: " + std::strerror(errno));
        }
    }

    auto img_data = image_data_pool->acquire();
    if (!img_data) {
        APP_LOG_ERROR(std::string(stream_name) + " Image data pool exhaust.");
        return false;
    }

    img_data->width = cfg.size.width;
    img_data->height = cfg.size.height;
    img_data->stride = cfg.stride;
    img_data->format = actual_format;
    img_data->frame_id = static_cast<int>(frame_id);
    img_data->capture_time = capture_time;
    img_data->t_capture_raw_ms = t_capture_raw_ms;
    img_data->length = total_length;
    img_data->fd = fd;
    img_data->offset = 0; // Assume 0 for simplicity or get from plane if needed

    // Population of telemetry fields
    img_data->cam_exposure_ms = static_cast<float>(exposure_ms);
    // ISP Latency: Time from capture start (capture_time) to now (host processing start)
    auto now = std::chrono::steady_clock::now();
    img_data->cam_isp_latency_ms = std::chrono::duration<float, std::chrono::milliseconds::period>(now - capture_time).count();
    // Buffer usage: Approximate based on pool status if available (simplified for now)
    img_data->cam_buffer_usage_percent = -1.0f; 

    // Also populate the pooled buffer metadata directly
    pooled_buffer->cam_exposure_ms = img_data->cam_exposure_ms;
    pooled_buffer->cam_isp_latency_ms = img_data->cam_isp_latency_ms;
    pooled_buffer->frame_id = img_data->frame_id;
    pooled_buffer->t_capture_raw_ms = img_data->t_capture_raw_ms;

    img_data->buffer = std::move(pooled_buffer); // Transfer ownership AFTER population
    
    img_data->ingest_start_time = process_start;
    img_data->ingest_end_time = std::chrono::steady_clock::now();
    
    if (app_ref) {
        app_ref->inc_cam_to_viz_produced();
    }

    if (!queue.push(img_data)) {
        // CRITICAL: Release IMMEDIATELY if push fails to prevent pool exhaustion
        if (app_ref) {
            app_ref->inc_cam_to_viz_dropped();
        }
        img_data->buffer.reset(); 
        image_data_pool->release(img_data); 
        return false;
    }
    
    // LOG THE FRAME PUSH FOR DEBUGGING
    // APP_LOG_DEBUG("Camera: Pushed frame ID " + std::to_string(img_data->frame_id) + " to queue at time " + std::to_string(get_time_raw_ms()));
    // if (std::string(stream_name) == "Main Video Stream") {
    //    std::cerr << "DEBUG: CameraCapture pushed Main frame " << img_data->frame_id << std::endl;
    // }

    return true;
}

CameraCapture::CameraCapture(unsigned int main_width, unsigned int main_height,
                             unsigned int tpu_width, unsigned int tpu_height,
                             unsigned int target_tpu_width, unsigned int target_tpu_height,
                             std::shared_ptr<BufferPool<uint8_t>> image_buffer_pool,
                             std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                             ImageQueue& image_processor_input_queue,
                             std::chrono::seconds watchdog_timeout)
    : width_(main_width), height_(main_height),
      tpu_width_(tpu_width), tpu_height_(tpu_height),
      tpu_fps_(120),
      target_tpu_width_(target_tpu_width), target_tpu_height_(target_tpu_height),
      image_processor_input_queue_(image_processor_input_queue),
      image_buffer_pool_(image_buffer_pool),
      image_data_pool_(image_data_pool),
      watchdog_timeout_(watchdog_timeout),
      frame_count_(0),
      main_output_queues_() { // Initialize as empty vector
    APP_LOG_INFO("CameraCapture constructor called.");
    
    camera_manager_ = std::make_unique<libcamera::CameraManager>();
    int ret = camera_manager_->start();
    if (ret) {
        APP_LOG_ERROR("Failed to start CameraManager: " + std::to_string(ret));
        camera_manager_.reset();
    }
}

CameraCapture::~CameraCapture() {
    stop();
}

bool CameraCapture::acquire_camera() {
    if (!camera_manager_) {
        APP_LOG_ERROR("Acquire: FAILURE - CameraManager is null.");
        return false;
    }
    
    auto cameras = camera_manager_->cameras();
    if (cameras.empty()) {
        APP_LOG_ERROR("Acquire: FAILURE - No cameras found.");
        return false;
    }

    for (const auto& cam : cameras) {
        if (cam->id().find("imx708") != std::string::npos) {
            camera_ = cam;
            APP_LOG_INFO("Explicitly selected IMX708 camera: " + camera_->id());
            break;
        }
    }

    if (!camera_) {
        APP_LOG_WARNING("IMX708 not found by ID, falling back to first available camera.");
        camera_ = cameras.front();
    }

    if (!camera_) {
        APP_LOG_ERROR("Acquire: FAILURE - Null camera pointer.");
        return false;
    }

    APP_LOG_INFO("Selected Camera ID: " + camera_->id());
    
    int ret = camera_->acquire();
    if (ret) {
        APP_LOG_ERROR("Acquire: FAILURE - Failed to acquire camera: " + std::to_string(ret));
        camera_.reset();
        return false;
    }
    
    APP_LOG_INFO("Camera acquired successfully.");
    return true;
}

bool CameraCapture::start() {
    APP_LOG_INFO("CameraCapture::start() called.");
    if (running_.load()) return false;
    if (!acquire_camera()) return false;
    if (!setup_camera()) {
        camera_->release();
        camera_.reset();
        return false;
    }
    
    camera_->requestCompleted.connect(this, &CameraCapture::request_complete_callback);

    libcamera::ControlList controls_to_set;
    // Hard FPS Lock & Exposure (120 FPS Mandate) - Removed FPS throttling to allow native max FPS
    // controls_to_set.set(libcamera::controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({frame_duration_us, frame_duration_us}));
    controls_to_set.set(libcamera::controls::AeEnable, false);
    controls_to_set.set(libcamera::controls::ExposureTime, 8000);
    
    if (camera_->start(&controls_to_set)) {
        APP_LOG_ERROR("Failed to start camera.");
        camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);
        camera_->release();
        camera_.reset();
        return false;
    }

    // Restore FrameDuration logging
    const libcamera::ControlList &properties = camera_->properties();
    auto frame_duration = properties.get(libcamera::controls::FrameDuration);
    if (frame_duration) {
        APP_LOG_INFO("Actual FrameDuration: " + std::to_string(*frame_duration) + " us (" + std::to_string(1000000.0 / *frame_duration) + " FPS)");
    }

    running_ = true;
    processing_running_ = true;
    request_processor_thread_ = std::thread(&CameraCapture::request_processor_thread_func, this);
    
    frame_count_ = 0;
    last_frame_time_ = std::chrono::steady_clock::now();
    first_frame_time_ = last_frame_time_;
    
    for (auto& req_ptr : requests_) {
        if (camera_->queueRequest(req_ptr.get())) { 
            APP_LOG_ERROR("Failed to queue initial request.");
            running_ = false; 
            stop(); 
            return false;
        }
    }
    return true;
}

void CameraCapture::stop() {
    if (!running_.exchange(false)) return;
    APP_LOG_INFO("Stopping CameraCapture...");
    processing_running_ = false;

    if (camera_) {
        camera_->requestCompleted.disconnect(this, &CameraCapture::request_complete_callback);
        camera_->stop();
        camera_->release();
        APP_LOG_INFO("Libcamera camera released.");
    }
    
    request_queue_cond_var_.notify_one();
    // Use timed join to prevent indefinite blocking
    if (request_processor_thread_.joinable()) {
        std::promise<bool> promise;
        std::future<bool> future = promise.get_future();
        
        std::thread timer_thread([this, &promise]() {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            if (request_processor_thread_.joinable()) {
                APP_LOG_WARNING("CameraCapture request processor thread did not join within timeout");
                promise.set_value(false);
            } else {
                promise.set_value(true);
            }
        });
        
        if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout) {
            timer_thread.join();
        } else {
            future.get();
            if (request_processor_thread_.joinable()) {
                request_processor_thread_.join();
            }
        }
        if (timer_thread.joinable()) {
            timer_thread.join();
        }
    }

    if (allocator_) {
        if (video_stream_) allocator_->free(video_stream_);
        if (tpu_stream_) allocator_->free(tpu_stream_);
        allocator_.reset();
    }
    
    // PERSISTENT MAPPING CLEANUP
    for (auto const& [buffer, mapping] : mapped_buffers_) {
        munmap(mapping.addr, mapping.length);
    }
    mapped_buffers_.clear();
    
    requests_.clear();
    camera_.reset();
    APP_LOG_INFO("CameraCapture stopped.");
}

bool CameraCapture::setup_camera() {
    if (!camera_) return false;
    
    // MANDATE: Exactly two streams for PiSP BCM2712_C0
    // Requesting processed formats (YUV/RGB) engages the ISP hardware.
    std::vector<libcamera::StreamRole> roles = {
        libcamera::StreamRole::VideoRecording, // Main pipeline (Encoder/Overlays)
        libcamera::StreamRole::Viewfinder      // TPU pipeline (Inference)
    };
    
    std::unique_ptr<libcamera::CameraConfiguration> config = camera_->generateConfiguration(roles);
    if (!config || config->size() != 2) {
        APP_LOG_ERROR("Failed to generate configuration for exactly two streams.");
        return false;
    }

    // --- PHASE II: SENSOR TUNING (120 FPS BINNING) ---
    // Explicitly request the 1536x864 sensor mode which supports 120 FPS.
    libcamera::SensorConfiguration sensorConfig;
    sensorConfig.bitDepth = 10;
    sensorConfig.outputSize = {1536, 864};
    config->sensorConfig = sensorConfig;

    // Main Stream (Index 0): ISP processed YUV for encoder compatibility
    libcamera::StreamConfiguration& mainCfg = config->at(0);
    mainCfg.pixelFormat = libcamera::formats::YUV420; 
    mainCfg.size.width = width_;
    mainCfg.size.height = height_;
    mainCfg.bufferCount = 6;

    // TPU Stream (Index 1): ISP processed RGB888 for Coral TPU requirements
    libcamera::StreamConfiguration& tpuCfg = config->at(1);
    tpuCfg.pixelFormat = libcamera::formats::RGB888; 
    tpuCfg.size.width = 320;
    tpuCfg.size.height = 320;
    tpuCfg.bufferCount = 32;
    
    // VALIDATION AUDIT
    libcamera::CameraConfiguration::Status validation = config->validate();
    if (validation == libcamera::CameraConfiguration::Invalid) {
        APP_LOG_ERROR("Camera configuration is INVALID and cannot be adjusted.");
        return false;
    }
    if (validation == libcamera::CameraConfiguration::Adjusted) {
        APP_LOG_WARNING("Camera configuration was ADJUSTED by libcamera to fit hardware constraints.");
    }

    if (camera_->configure(config.get())) {
        APP_LOG_ERROR("Failed to call camera_->configure(). Hardware rejected processed stream configuration.");
        return false;
    }

    video_stream_ = mainCfg.stream();
    tpu_stream_ = tpuCfg.stream();
    actual_pixel_format_ = mainCfg.pixelFormat;
    actual_size_ = mainCfg.size;
    actual_stride_ = mainCfg.stride;

    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    if (allocator_->allocate(video_stream_) < 0) {
        APP_LOG_ERROR("Failed to allocate buffers for Main (YUV) stream.");
        return false;
    }
    if (allocator_->allocate(tpu_stream_) < 0) {
        APP_LOG_ERROR("Failed to allocate buffers for TPU (RGB) stream.");
        return false;
    }

    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& video_buffers = allocator_->buffers(video_stream_);
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& tpu_buffers = allocator_->buffers(tpu_stream_);
    
    // PERSISTENT MAPPING: Map all buffers once during setup
    mapped_buffers_.clear();
    for (const auto& buffer : video_buffers) {
        size_t total_length = 0;
        for (const auto& plane : buffer->planes()) total_length += plane.length;
        void* addr = mmap(nullptr, total_length, PROT_READ, MAP_SHARED, buffer->planes()[0].fd.get(), 0);
        if (addr != MAP_FAILED) {
            mapped_buffers_[buffer.get()] = {addr, total_length};
        }
    }
    for (const auto& buffer : tpu_buffers) {
        size_t total_length = 0;
        for (const auto& plane : buffer->planes()) total_length += plane.length;
        void* addr = mmap(nullptr, total_length, PROT_READ, MAP_SHARED, buffer->planes()[0].fd.get(), 0);
        if (addr != MAP_FAILED) {
            mapped_buffers_[buffer.get()] = {addr, total_length};
        }
    }

    size_t min_buffers = std::min(video_buffers.size(), tpu_buffers.size());
    
    requests_.clear();
    for (unsigned int i = 0; i < min_buffers; ++i) {
        std::unique_ptr<libcamera::Request> request = camera_->createRequest();
        if (!request) return false;
        if (request->addBuffer(video_stream_, video_buffers[i].get()) != 0) return false;
        if (request->addBuffer(tpu_stream_, tpu_buffers[i].get()) != 0) return false;
        requests_.push_back(std::move(request));
    }
    
    APP_LOG_INFO("ISP Engaged. Camera configured with " + std::to_string(requests_.size()) + " dual-buffered YUV/RGB requests.");
    return true;
}

void CameraCapture::request_complete_callback(libcamera::Request* request) {
    if (!running_.load() || !processing_running_.load()) {
        if (request->status() != libcamera::Request::RequestCancelled) {
            request->reuse(libcamera::Request::ReuseBuffers);
            if (camera_) camera_->queueRequest(request);
        }
        return;
    }
    if (request->status() != libcamera::Request::RequestComplete) {
        APP_LOG_ERROR("Request failed status: " + std::to_string(request->status()));
        request->reuse(libcamera::Request::ReuseBuffers); 
        if (camera_) camera_->queueRequest(request);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(request_queue_mutex_);
        request_queue_.push(request);
        request_queue_cond_var_.notify_one();
    }
}

extern std::atomic<bool> g_running;

void CameraCapture::request_processor_thread_func() {
    APP_LOG_INFO("CameraCapture: Request processor thread started.");
    while (processing_running_.load() && g_running.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(request_queue_mutex_);
        if (!request_queue_cond_var_.wait_for(lock, std::chrono::milliseconds(10), [this] { 
            return !request_queue_.empty() || !processing_running_.load() || !g_running.load(); 
        })) continue;

        if ((!processing_running_.load() || !g_running.load()) && request_queue_.empty()) break;
        if (request_queue_.empty()) continue;
        
        libcamera::Request* request = request_queue_.front();
        request_queue_.pop();
        lock.unlock();

        if (!request) continue;

        auto now_mon = std::chrono::steady_clock::now();
        auto now_sys = std::chrono::system_clock::now();
        long long epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_sys.time_since_epoch()).count();
        
        // Authoritative Monotonic Raw Clock (Unified across all modules)
        uint64_t t_capture_ms = get_time_raw_ms();
        auto capture_time = now_mon; 
        
        long long frame_id = request->sequence();
        // Validate frame_id to prevent invalid values from propagating
        if (frame_id < 0) {
            static std::atomic<int> invalid_frame_counter{0};
            int counter = invalid_frame_counter.fetch_add(1);
            if (counter % 100 == 0) { // Log every 100th invalid frame to avoid log spam
                APP_LOG_WARNING("CameraCapture: Invalid frame_id " + std::to_string(frame_id) + " detected, using sequence counter instead");
            }
            // Use global frame counter as fallback
            frame_id = ImageData::global_frame_counter.fetch_add(1);
        }
        long long exposure_ms = 0;
        auto md_exposure_us = request->metadata().get(libcamera::controls::ExposureTime);
        if (md_exposure_us) exposure_ms = *md_exposure_us / 1000;

        // Log CameraCapture telemetry to unified CSV
        {
            CsvLogEntry cam_entry;
            copy_to_array(cam_entry.module, "CameraCapture");
            copy_to_array(cam_entry.event, "frame_captured");
            cam_entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            cam_entry.call_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            cam_entry.cam_frame_id = static_cast<int>(frame_id);
            cam_entry.cam_exposure_ms = static_cast<float>(exposure_ms);
            
            // ISP Latency: Time from sensor capture to now
            uint64_t sensor_ts_ns = 0;
            for (auto const& [stream, buffer] : request->buffers()) {
                sensor_ts_ns = buffer->metadata().timestamp;
                break;
            }
            if (sensor_ts_ns > 0) {
                struct timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
                cam_entry.cam_isp_latency_ms = static_cast<float>(now_ns - sensor_ts_ns) / 1000000.0f;
            } else {
                cam_entry.cam_isp_latency_ms = 0.0f;
            }
            
            if (image_buffer_pool_) {
                auto stats = image_buffer_pool_->get_buffer_stats();
                if (stats.second > 0) {
                    cam_entry.cam_buffer_usage_percent = (static_cast<float>(stats.second - stats.first) / static_cast<float>(stats.second)) * 100.0f;
                }
            }
            
            Logger::getInstance().log_csv(cam_entry);
        }
        
        // OPTIMIZATION: Extract buffers and reuse request immediately to keep libcamera pipeline full
        libcamera::Request::BufferMap captured_buffers = request->buffers();
        
        // Log SENSOR FPS frequently
        static int sensor_frame_counter = 0;
        static auto sensor_start_time = std::chrono::steady_clock::now();
        sensor_frame_counter++;
        auto sensor_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now_mon - sensor_start_time).count();
        if (sensor_duration >= 1000) {
            APP_LOG_INFO("SENSOR FPS: " + std::to_string((sensor_frame_counter * 1000.0) / sensor_duration));
            sensor_frame_counter = 0;
            sensor_start_time = now_mon;
        }

        // Processing block (independent of the Request object's lifecycle once buffers are used)
        if (captured_buffers.empty()) {
            APP_LOG_WARNING("CameraCapture: No buffers in completed request.");
            if (app_ref_) {
                app_ref_->inc_cam_to_viz_produced();
                app_ref_->inc_cam_to_viz_dropped();
                app_ref_->inc_cam_to_tpu_proc_produced();
                app_ref_->inc_cam_to_tpu_proc_dropped();
            }
            
            request->reuse(libcamera::Request::ReuseBuffers); 
            if (running_.load() && camera_) camera_->queueRequest(request);
        } else {
            // REUSE IMMEDIATELY
            request->reuse(libcamera::Request::ReuseBuffers); 
            if (running_.load() && camera_) camera_->queueRequest(request);

            bool processed_any_frame = false;
            
            // Map MappedBuffer to MappedBufferInfo for the static helper
            std::map<const libcamera::FrameBuffer*, MappedBufferInfo> compat_map;
            for(auto const& [k, v] : mapped_buffers_) compat_map[k] = {v.addr, v.length};

            // Extract sensor timestamp for latency calculation
            uint64_t sensor_ts_ns = 0;
            for (auto const& [stream, buffer] : captured_buffers) {
                sensor_ts_ns = buffer->metadata().timestamp;
                break;
            }

            if (captured_buffers.count(video_stream_)) {
                const libcamera::FrameBuffer* video_fb = captured_buffers.at(video_stream_);
                if (video_fb) {
                    // ISP MANDATE: Distribute to ALL registered consumers (Visualization, Encoder, etc.)
                    // Now iterating through the stored vector of ImageQueue pointers
                    for (auto* queue_ptr : main_output_queues_) {
                        if (queue_ptr) { // Check for nullptr in case any queue was not properly set
                            if (!process_frame_buffer(video_fb, video_stream_->configuration(), image_buffer_pool_, image_data_pool_, *queue_ptr, "Main Video Stream", width_, height_, capture_time, t_capture_ms, video_stream_->configuration().pixelFormat, frame_id, exposure_ms, &main_stream_drop_count_, &tpu_stream_drop_count_, nullptr, app_ref_, compat_map)) {
                                // Drop handled inside process_frame_buffer
                            } else {
                                processed_any_frame = true;
                            }
                        } else {
                            APP_LOG_ERROR("CameraCapture: Null pointer in main_output_queues_ vector.");
                        }
                    }
                } else {
                    APP_LOG_ERROR("CameraCapture: Null video frame buffer.");
                    if (app_ref_) {
                        app_ref_->inc_cam_to_viz_produced();
                        app_ref_->inc_cam_to_viz_dropped();
                    }
                }
            }

            if (captured_buffers.count(tpu_stream_)) {
                const libcamera::FrameBuffer* tpu_fb = captured_buffers.at(tpu_stream_);
                if (tpu_fb) {
                    if (this->process_tpu_processed_frame_buffer(tpu_fb, tpu_stream_->configuration(), capture_time, t_capture_ms, frame_id, exposure_ms, sensor_ts_ns)) {
                        fps_measurement_frames_++;
                        if (fps_measurement_frames_ > skip_initial_measurements_) {
                            auto interval_us = std::chrono::duration_cast<std::chrono::microseconds>(now_mon - last_frame_time_).count();
                            static double rolling_avg_us = 8333.33; 
                            rolling_avg_us = rolling_avg_us * 0.95 + (double)interval_us * 0.05;
                            frame_rate_.store(static_cast<int>(1000000.0 / rolling_avg_us));
                        }
                        last_frame_time_ = now_mon;
                        last_frame_timestamp_.store(epoch_ms); // Telemetry Epoch MS
                    }
                } else {
                    APP_LOG_ERROR("CameraCapture: Null TPU frame buffer.");
                    if (app_ref_) {
                        app_ref_->inc_cam_to_tpu_proc_produced();
                        app_ref_->inc_cam_to_tpu_proc_dropped();
                    }
                }
            } else {
                APP_LOG_WARNING("CameraCapture: TPU stream buffer missing.");
                if (app_ref_) {
                    app_ref_->inc_cam_to_tpu_proc_produced();
                    app_ref_->inc_cam_to_tpu_proc_dropped();
                }
            }
        }
    }
}

bool CameraCapture::init_video_encoder() { return true; }

bool CameraCapture::process_tpu_processed_frame_buffer(const libcamera::FrameBuffer* fb,
                                                 const libcamera::StreamConfiguration& cfg,
                                                 std::chrono::steady_clock::time_point capture_time,
                                                 uint64_t t_capture_raw_ms,
                                                 long long frame_id,
                                                 long long exposure_ms,
                                                 uint64_t sensor_ts_ns) {
    if (fb->planes().empty()) {
        APP_LOG_ERROR("TPU processed: No planes.");
        return false;
    }

    const libcamera::FrameBuffer::Plane& plane = fb->planes()[0];
    int fd = plane.fd.get();
    size_t length = plane.length;

    auto pooled_buffer = image_buffer_pool_->acquire();
    if (!pooled_buffer) {
        APP_LOG_ERROR("TPU processed: Buffer pool exhaustion.");
        if (app_ref_) {
            app_ref_->inc_cam_to_tpu_proc_produced();
            app_ref_->inc_cam_to_tpu_proc_dropped();
        }
        return false;
    }

    // PERSISTENT MAPPING
    void* mapped_memory = nullptr;
    bool is_persistent = false;
    // Map MappedBuffer to MappedBufferInfo for the static helper
    std::map<const libcamera::FrameBuffer*, MappedBufferInfo> compat_map;
    for(auto const& [k, v] : mapped_buffers_) compat_map[k] = {v.addr, v.length};
    
    auto it = mapped_buffers_.find(fb);
    if (it != mapped_buffers_.end()) {
        mapped_memory = it->second.addr;
        is_persistent = true;
    } else {
        mapped_memory = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    }

    if (mapped_memory == MAP_FAILED || mapped_memory == nullptr) {
        APP_LOG_ERROR("TPU processed: mmap failed.");
        if (app_ref_) {
            app_ref_->inc_cam_to_tpu_proc_produced();
            app_ref_->inc_cam_to_tpu_proc_dropped();
        }
        pooled_buffer.reset(); // CRITICAL: Release buffer to pool
        return false;
    }

    if (pooled_buffer->data.size() < length) pooled_buffer->data.resize(length);
    
    // ISP MANDATE: Copy interleaved RGB888 data directly.
    std::memcpy(pooled_buffer->data.data(), mapped_memory, length);
    pooled_buffer->size = length;
    
    if (!is_persistent) {
        munmap(mapped_memory, length);
    }

    ImageData* image_data = image_data_pool_->acquire();
    if (!image_data) {
        APP_LOG_ERROR("TPU processed: ImageData pool exhaustion.");
        if (app_ref_) {
            app_ref_->inc_cam_to_tpu_proc_produced();
            app_ref_->inc_cam_to_tpu_proc_dropped();
        }
        return false;
    }

    *image_data = ImageData(capture_time, (int)frame_id);
    image_data->width = cfg.size.width;
    image_data->height = cfg.size.height;
    image_data->stride = cfg.stride;
    image_data->format = cfg.pixelFormat; // libcamera::formats::RGB888
    image_data->length = length;
    image_data->fd = fd;
    image_data->offset = plane.offset;
    image_data->buffer = std::move(pooled_buffer);
    image_data->capture_time = capture_time;
    image_data->t_capture_raw_ms = t_capture_raw_ms;

    // Population of telemetry fields for TPU stream
    image_data->cam_exposure_ms = static_cast<float>(exposure_ms);
    
    if (sensor_ts_ns > 0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
        image_data->cam_isp_latency_ms = static_cast<float>(now_ns - sensor_ts_ns) / 1000000.0f;
    } else {
        auto now = std::chrono::steady_clock::now();
        image_data->cam_isp_latency_ms = std::chrono::duration<float, std::milli>(now - capture_time).count();
    }

    if (image_buffer_pool_) {
        auto stats = image_buffer_pool_->get_buffer_stats();
        if (stats.second > 0) {
            image_data->cam_buffer_usage_percent = (static_cast<float>(stats.second - stats.first) / static_cast<float>(stats.second)) * 100.0f;
        }
    }

    if (app_ref_) {
        app_ref_->inc_cam_to_tpu_proc_produced();
    }

    // Accounting Fix: Move increment_produced inside the lock-protected push block
    if (!image_processor_input_queue_.push(image_data)) {
        // CRITICAL: Release IMMEDIATELY if push fails to prevent pool exhaustion
        if (app_ref_) {
            app_ref_->inc_cam_to_tpu_proc_dropped();
        }
        image_data->buffer.reset();
        image_data_pool_->release(image_data);
        return false;
    }
    
    auto process_end = std::chrono::steady_clock::now();
    auto push_time = std::chrono::duration_cast<std::chrono::microseconds>(process_end - capture_time).count();
    long long current_push_avg = avg_capture_time_us_.load();
    avg_capture_time_us_.store(current_push_avg == 0 ? push_time : (long long)(current_push_avg * 0.8 + push_time * 0.2));
    
    return true;
}

// Set application reference for updating counters
