#ifndef MJPEG_SERVER_H
#define MJPEG_SERVER_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

// Forward declare for ImageFrame (if needed, otherwise define directly)
struct ImageFrame {
    std::vector<uint8_t> jpeg_data;
    size_t width;
    size_t height;
    // Add timestamp or other metadata if necessary
};

// Thread-safe queue for MJPEG frames
class MjpegQueue {
public:
    void push(ImageFrame new_frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Keep only the latest frame for MJPEG streaming
        if (!queue_.empty()) {
            queue_.pop();
        }
        queue_.push(std::move(new_frame));
        cond_var_.notify_one();
    }

    bool pop(ImageFrame& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        // Wait for a new frame, but don't block indefinitely if stopping
        cond_var_.wait(lock, [this]{ return !queue_.empty() || !running_; });
        if (queue_.empty()) {
            return false;
        }
        frame = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // Peeks at the latest frame without removing it
    bool peek_latest(ImageFrame& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]{ return !queue_.empty() || !running_; });
        if (queue_.empty()) {
            return false;
        }
        frame = queue_.back(); // Get the latest frame
        return true;
    }

    void set_running(bool val) {
        running_ = val;
        if (!val) {
            cond_var_.notify_all(); // Unblock any waiting threads
        }
    }


private:
    mutable std::mutex mutex_;
    std::queue<ImageFrame> queue_;
    std::condition_variable cond_var_;
    std::atomic<bool> running_ = true;
};

class MjpegServer {
public:
    MjpegServer(int port, MjpegQueue& input_queue);
    ~MjpegServer();

    bool start();
    void stop();
    bool is_running() const { return running_; }

private:
    void server_thread_func();
    void handle_client(int client_sock);

    int port_;
    MjpegQueue& input_queue_;
    std::atomic<bool> running_ = false;
    std::thread server_thread_;
    int server_sock_ = -1;
};

#endif // MJPEG_SERVER_H
