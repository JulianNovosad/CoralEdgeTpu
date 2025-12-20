#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <stdexcept>
#include <functional> // For std::function

#include "util_logging.h" // Include logging utilities
#include <iostream>


// A generic buffer object that can be pooled.
template <typename T>
struct PooledBuffer {
    std::vector<T> data;
    // Add any other metadata you need, e.g., size, timestamp.
    size_t size = 0;
    
    // Zero-copy related fields
    int fd = -1;           // File descriptor for zero-copy access
    size_t offset = 0;     // Offset within the file descriptor
    size_t length = 0;     // Length of the data
};

/**
 * @brief A thread-safe, generic pool for managing reusable buffers.
 *
 * This class pre-allocates a fixed number of buffers of a specific size
 * and allows threads to acquire and release them safely. Buffers are
 * returned to the pool automatically using a custom deleter with std::shared_ptr.
 *
 * @tparam T The type of data stored in the buffer's vector (e.g., uint8_t).
 */
template <typename T>
class BufferPool {
public:
    // Constructor for fixed-size buffer types (e.g., DetectionResult)
    BufferPool(size_t pool_size, size_t buffer_data_size, const std::string& name)
        : name_(name), total_buffers_(pool_size) {
        buffers_storage_.reserve(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            buffers_storage_.emplace_back(); // Store the actual PooledBuffer object
            buffers_storage_.back().data.resize(buffer_data_size);
            available_buffers_.push(&buffers_storage_.back()); // Push raw pointer to queue
        }
    }

    // Constructor for variable-size buffer types (e.g., uint8_t for images)
    BufferPool(size_t pool_size, size_t min_buffer_size, size_t max_buffer_size, const std::string& name)
        : name_(name), total_buffers_(pool_size) {
        buffers_storage_.reserve(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            buffers_storage_.emplace_back();
            buffers_storage_.back().data.resize(min_buffer_size); // Initial allocation
            available_buffers_.push(&buffers_storage_.back()); // Push raw pointer to queue
        }
    }

    std::shared_ptr<PooledBuffer<T>> acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, std::chrono::seconds(1),
                            [this]{ return !available_buffers_.empty(); })) {
            APP_LOG_WARNING(name_ + ": Failed to acquire buffer within timeout. Available: " + std::to_string(available_buffers_.size()));
            return nullptr; // Timeout
        }
        PooledBuffer<T>* buffer_ptr = available_buffers_.front();
        available_buffers_.pop();
        lock.unlock();

        // Create a shared_ptr with a custom deleter that returns the raw pointer to the pool
        return std::shared_ptr<PooledBuffer<T>>(buffer_ptr, [this](PooledBuffer<T>* b) {
            std::unique_lock<std::mutex> local_lock(this->mutex_);
            this->available_buffers_.push(b); // Push the raw pointer back to the queue
            local_lock.unlock();
            this->cond_.notify_one();
        });
    }

    // Method to acquire a buffer with zero-copy information
    std::shared_ptr<PooledBuffer<T>> acquire_with_fd(int fd, size_t offset, size_t length) {
        auto buffer = acquire();
        if (buffer) {
            buffer->fd = fd;
            buffer->offset = offset;
            buffer->length = length;
        }
        return buffer;
    }

    // No explicit release method needed, as it's handled by the custom deleter
    
    size_t get_available_buffers() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_buffers_.size();
    }

    size_t get_total_buffers() const {
        return total_buffers_;
    }

private:
    std::string name_;
    size_t total_buffers_;
    std::queue<PooledBuffer<T>*> available_buffers_; // Stores raw pointers
    std::vector<PooledBuffer<T>> buffers_storage_; // Owns the actual buffer objects
    mutable std::mutex mutex_;
    std::condition_variable cond_;
};

#endif // BUFFER_POOL_H