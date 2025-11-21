#ifndef UDP_SENDER_H
#define UDP_SENDER_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <netinet/in.h> // For sockaddr_in

#include "inference.h" // For DetectionResult

// Thread-safe queue for detection results (to be sent via UDP)
class UdpQueue {
public:
    void push(std::vector<DetectionResult> results) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(results));
        cond_var_.notify_one();
    }

    bool pop(std::vector<DetectionResult>& results) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]{ return !queue_.empty() || !running_; });
        if (queue_.empty()) {
            return false;
        }
        results = std::move(queue_.front());
        queue_.pop();
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
    std::queue<std::vector<DetectionResult>> queue_;
    std::condition_variable cond_var_;
    std::atomic<bool> running_ = true;
};

class UdpSender {
public:
    UdpSender(const std::string& target_ip, int target_port, UdpQueue& input_queue);
    ~UdpSender();

    bool start();
    void stop();
    bool is_running() const { return running_; }

private:
    void sender_thread_func();
    std::string detection_to_json(const std::vector<DetectionResult>& results);

    std::string target_ip_;
    int target_port_;
    UdpQueue& input_queue_;
    std::atomic<bool> running_ = false;
    std::thread sender_thread_;
    int sockfd_ = -1;
    sockaddr_in server_addr_;
};

#endif // UDP_SENDER_H
