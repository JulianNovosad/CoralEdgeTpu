#ifndef LOCKFREE_QUEUE_H
#define LOCKFREE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>

/**
 * @brief Thread-safe blocking queue (replacement for boost::lockfree for debugging).
 * 
 * Uses std::queue + std::mutex + std::condition_variable.
 * Maintains the same API as the previous LockFreeQueue.
 */
template<typename T, size_t Capacity = 1024>
class LockFreeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
public:
    LockFreeQueue() {}

    bool push(const T& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= Capacity) {
            return false;
        }
        queue_.push(data);
        cv_.notify_one();
        return true;
    }

    bool pop(T& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        data = queue_.front();
        queue_.pop();
        return true;
    }

    bool wait_pop(T& data, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            data = queue_.front();
            queue_.pop();
            return true;
        }
        return false;
    }

    size_t write_available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Capacity - queue_.size();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    bool full() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size() >= Capacity;
    }
    
    size_t size_approx() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool try_pop(T& data) {
        return pop(data);
    }

    void clear(std::function<void(T&)> callback = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            T data = queue_.front();
            queue_.pop();
            if (callback) callback(data);
        }
    }
};

#endif // LOCKFREE_QUEUE_H
