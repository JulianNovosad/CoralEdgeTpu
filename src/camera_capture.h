#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <libcamera/libcamera.h>
#include <thread>
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <chrono>

// Simple structure to hold image data
struct ImageData {
    std::vector<uint8_t> data;
    size_t width;
    size_t height;
    std::chrono::high_resolution_clock::time_point timestamp;
};

// Thread-safe queue for image data
class ThreadSafeQueue {
public:
    void push(ImageData new_data) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(new_data));
        cond_var_.notify_one();
    }

    bool pop(ImageData& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]{ return !queue_.empty() || !running_; });
        if (queue_.empty()) {
            return false;
        }
        data = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool try_pop(ImageData& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        data = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void set_running(bool val) {
        running_ = val;
        if (!val) {
            cond_var_.notify_all(); // Unblock any waiting threads
        }
    }

private:
    mutable std::mutex mutex_;
    std::queue<ImageData> queue_;
    std::condition_variable cond_var_;
    std::atomic<bool> running_ = true;
};

class CameraCapture {
public:
    CameraCapture(unsigned int width, unsigned int height, ThreadSafeQueue& output_queue);
    ~CameraCapture();

    bool start();
    void stop();
    bool is_running() const { return running_; }

private:
    void capture_thread_func();
    void process_request(libcamera::Request* request);

    unsigned int width_;
    unsigned int height_;
    ThreadSafeQueue& output_queue_;

    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    std::shared_ptr<libcamera::Camera> camera_;
    std::unique_ptr<libcamera::CameraConfiguration> config_;
    libcamera::Stream* stream_ = nullptr;
    libcamera::FrameBufferAllocator* allocator_ = nullptr;

    std::atomic<bool> running_ = false;
    std::thread capture_thread_;
    std::atomic<int> frames_captured_ = 0;
};

#endif // CAMERA_CAPTURE_H
