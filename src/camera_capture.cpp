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
                                 std::chrono::steady_clock::time_point capture_time,
                                 const libcamera::PixelFormat& actual_format,
                                 long long frame_id,
                                 [[maybe_unused]] long long exposure_ms,
                                 std::atomic<int64_t>* main_stream_drop_counter = nullptr,
                                 std::atomic<int64_t>* tpu_stream_drop_counter = nullptr,
                                 std::atomic<int64_t>* frames_produced_counter = nullptr)
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

    void* mapped_memory = mmap(nullptr, total_length, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_memory == MAP_FAILED) {
        APP_LOG_ERROR(std::string(stream_name) + " Failed to map buffer: " + std::strerror(errno));
        return false;
    }

    if (pooled_buffer->data.size() < total_length) {
        pooled_buffer->data.resize(total_length);
    }
    std::memcpy(pooled_buffer->data.data(), mapped_memory, total_length);
    pooled_buffer->size = total_length;

    if (munmap(mapped_memory, total_length) == -1) {
        APP_LOG_ERROR(std::string(stream_name) + " Failed to unmap buffer: " + std::strerror(errno));
    }

    ImageData image_data(capture_time, (int)frame_id);
    image_data.width = cfg.size.width;
    image_data.height = cfg.size.height;
    image_data.stride = cfg.stride;
    image_data.format = actual_format;
    image_data.length = total_length;
    image_data.fd = fd;
    image_data.offset = plane0.offset;
    
    image_data.ingest_start_time = process_start;
    image_data.ingest_end_time = std::chrono::steady_clock::now();
    image_data.capture_time = capture_time; 
    image_data.t_capture_raw_ms = get_time_raw_ms();

    image_data.buffer = std::move(pooled_buffer);
    queue.push(std::move(image_data));

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

    camera_ = cameras.front();
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
    int main_stream_fps = std::max(120u, tpu_fps_);
    int64_t main_stream_frame_duration_us = 1000000 / main_stream_fps;
    try {
        controls_to_set.set(libcamera::controls::FrameDurationLimits, {main_stream_frame_duration_us/2, main_stream_frame_duration_us*2});
    } catch (const std::exception& e) {
        APP_LOG_WARNING("FrameDurationLimits error: " + std::string(e.what()));
    }
    
    if (tpu_fps_ > 0) {
        try {
            controls_to_set.set(libcamera::controls::AeEnable, true);
        } catch (const std::exception& e) {
            APP_LOG_WARNING("AeEnable error: " + std::string(e.what()));
        }
    }

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

    // Main Stream (Index 0): ISP processed YUV for encoder compatibility
    libcamera::StreamConfiguration& mainCfg = config->at(0);
    mainCfg.pixelFormat = libcamera::formats::YUV420; 
    mainCfg.size.width = width_;
    mainCfg.size.height = height_;
    mainCfg.bufferCount = 16;

    // TPU Stream (Index 1): ISP processed RGB888 for Coral TPU requirements
    libcamera::StreamConfiguration& tpuCfg = config->at(1);
    tpuCfg.pixelFormat = libcamera::formats::RGB888; 
    tpuCfg.size.width = 320;
    tpuCfg.size.height = 320;
    tpuCfg.bufferCount = 16;
    
    // VALIDATION AUDIT
    libcamera::CameraConfiguration::Status validation = config->validate();
    if (validation == libcamera::CameraConfiguration::Invalid) {
        APP_LOG_ERROR("Camera configuration is INVALID and cannot be adjusted.");
        return false;
    }
    if (validation == libcamera::CameraConfiguration::Adjusted) {
        APP_LOG_WARNING("Camera configuration was ADJUSTED by libcamera to fit hardware constraints.");
        APP_LOG_INFO("Adjusted Main: " + std::to_string(mainCfg.size.width) + "x" + std::to_string(mainCfg.size.height) + " (" + mainCfg.pixelFormat.toString() + ")");
        APP_LOG_INFO("Adjusted TPU: " + std::to_string(tpuCfg.size.width) + "x" + std::to_string(tpuCfg.size.height) + " (" + tpuCfg.pixelFormat.toString() + ")");
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

extern std::atomic<bool> shutdown_requested;

void CameraCapture::request_processor_thread_func() {
    APP_LOG_INFO("CameraCapture: Request processor thread started.");
    while (processing_running_.load() && !shutdown_requested.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(request_queue_mutex_);
        if (!request_queue_cond_var_.wait_for(lock, std::chrono::milliseconds(10), [this] { 
            return !request_queue_.empty() || !processing_running_.load() || shutdown_requested.load(); 
        })) continue;

        if ((!processing_running_.load() || shutdown_requested.load()) && request_queue_.empty()) break;
        if (request_queue_.empty()) continue;
        
        libcamera::Request* request = request_queue_.front();
        request_queue_.pop();
        lock.unlock();

        if (!request) continue;

        auto now_mon = std::chrono::steady_clock::now();
        auto now_sys = std::chrono::system_clock::now();
        long long epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_sys.time_since_epoch()).count();
        
        // Sourced from libcamera Request metadata per instruction
        auto capture_time = now_mon;
        auto md_timestamp = request->metadata().get(libcamera::controls::SensorTimestamp);
        if (md_timestamp) {
            // We use the steady_clock::now() as base but could apply the offset if libcamera uses boot time
            capture_time = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(*md_timestamp));
        }

        long long frame_id = request->sequence();
        long long exposure_ms = 0;
        auto md_exposure_us = request->metadata().get(libcamera::controls::ExposureTime);
        if (md_exposure_us) exposure_ms = *md_exposure_us / 1000;
        
        static int sensor_frame_counter = 0;
        static auto sensor_start_time = std::chrono::steady_clock::now();
        sensor_frame_counter++;
        auto sensor_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now_mon - sensor_start_time).count();
        if (sensor_duration >= 1000) {
            APP_LOG_INFO("SENSOR FPS: " + std::to_string((sensor_frame_counter * 1000.0) / sensor_duration));
            sensor_frame_counter = 0;
            sensor_start_time = now_mon;
        }

        libcamera::Request::BufferMap captured_buffers = request->buffers();
        if (captured_buffers.empty()) {
            APP_LOG_WARNING("CameraCapture: No buffers in completed request.");
        } else {
            bool processed_any_frame = false;
            frames_produced_.fetch_add(1);
            if (app_ref_) app_ref_->increment_camera_frames_produced();

            if (!main_output_queues_.empty() && captured_buffers.count(video_stream_)) {
                const libcamera::FrameBuffer* video_fb = captured_buffers.at(video_stream_);
                if (video_fb) {
                    if (process_frame_buffer(video_fb, video_stream_->configuration(), image_buffer_pool_, main_output_queues_.front().get(), "Main Video Stream", width_, height_, capture_time, video_stream_->configuration().pixelFormat, frame_id, exposure_ms, &main_stream_drop_count_, &tpu_stream_drop_count_)) {
                        processed_any_frame = true;
                    }
                } else {
                    APP_LOG_ERROR("CameraCapture: Null video frame buffer.");
                }
            }

            if (captured_buffers.count(tpu_stream_)) {
                const libcamera::FrameBuffer* tpu_fb = captured_buffers.at(tpu_stream_);
                if (tpu_fb) {
                    if (this->process_tpu_processed_frame_buffer(tpu_fb, tpu_stream_->configuration(), capture_time, frame_id, exposure_ms)) {
                        processed_any_frame = true;
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
                }
            } else {
                APP_LOG_WARNING("CameraCapture: TPU stream buffer missing.");
            }

            if (processed_any_frame) {
                auto end_process_time = std::chrono::steady_clock::now();
                long long total_process_us = std::chrono::duration_cast<std::chrono::microseconds>(end_process_time - now_mon).count();
                long long current_avg = avg_total_loop_time_us_.load();
                avg_total_loop_time_us_.store(current_avg == 0 ? total_process_us : (long long)(current_avg * 0.9 + total_process_us * 0.1));

                // Telemetry Logging
                CsvLogEntry entry;
                entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                entry.call_ts_epoch_ms = epoch_ms;
                copy_to_array(entry.module, "Camera");
                copy_to_array(entry.event, "frame_captured");
                entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
                entry.cam_frame_id = frame_id;
                entry.cam_exposure_ms = static_cast<float>(exposure_ms);
                entry.cam_isp_latency_ms = static_cast<float>(total_process_us) / 1000.0f;
                Logger::getInstance().log_csv(entry);
            }
        }
    
        request->reuse(libcamera::Request::ReuseBuffers); 
        if (running_.load() && camera_) camera_->queueRequest(request);
    }
}

bool CameraCapture::init_video_encoder() { return true; }

bool CameraCapture::process_tpu_processed_frame_buffer(const libcamera::FrameBuffer* fb,
                                                 const libcamera::StreamConfiguration& cfg,
                                                 std::chrono::steady_clock::time_point capture_time,
                                                 long long frame_id,
                                                 [[maybe_unused]] long long exposure_ms) {
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
        return false;
    }

    void* mapped_memory = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_memory == MAP_FAILED) {
        APP_LOG_ERROR("TPU processed: mmap failed.");
        return false;
    }

    if (pooled_buffer->data.size() < length) pooled_buffer->data.resize(length);
    
    // ISP MANDATE: Copy interleaved RGB888 data directly.
    std::memcpy(pooled_buffer->data.data(), mapped_memory, length);
    pooled_buffer->size = length;
    munmap(mapped_memory, length);

    ImageData image_data(capture_time, (int)frame_id);
    image_data.width = cfg.size.width;
    image_data.height = cfg.size.height;
    image_data.stride = cfg.stride;
    image_data.format = cfg.pixelFormat; // libcamera::formats::RGB888
    image_data.length = length;
    image_data.fd = fd;
    image_data.offset = plane.offset;
    image_data.buffer = std::move(pooled_buffer);
    image_data.capture_time = capture_time;
    image_data.t_capture_raw_ms = get_time_raw_ms();

    image_processor_input_queue_.push(std::move(image_data));
    auto process_end = std::chrono::steady_clock::now();
    auto push_time = std::chrono::duration_cast<std::chrono::microseconds>(process_end - capture_time).count();
    long long current_push_avg = avg_capture_time_us_.load();
    avg_capture_time_us_.store(current_push_avg == 0 ? push_time : (long long)(current_push_avg * 0.8 + push_time * 0.2));
    
    return true;
}