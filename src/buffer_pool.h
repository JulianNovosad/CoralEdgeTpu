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


// A generic buffer object that can be pooled.
template <typename T>
struct PooledBuffer {
    std::vector<T> data;
    // Add any other metadata you need, e.g., size, timestamp.
    size_t size = 0;
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
    // Defines a shared_ptr that will automatically return the buffer to the pool.
    using PooledPtr = std::shared_ptr<PooledBuffer<T>>;

    BufferPool(size_t num_buffers, size_t buffer_size, std::string pool_name = "BufferPool")
        : name_(std::move(pool_name)) {
        if (num_buffers == 0 || buffer_size == 0) {
            throw std::invalid_argument("BufferPool: Number of buffers and buffer size must be greater than 0.");
        }
        
        for (size_t i = 0; i < num_buffers; ++i) {
            PooledBuffer<T>* buffer = new PooledBuffer<T>(); // Manually allocate raw buffer
            buffer->data.resize(buffer_size); // Pre-allocate memory
            pool_.push(buffer); // Store raw pointer in the queue
        }
    }

    // Disable copy and assignment
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    
    // Destructor to ensure all allocated raw buffers are deleted if the pool is destroyed
    ~BufferPool() {
        std::lock_guard<std::mutex> lock(mutex_); // Ensure thread safety during destruction
        while(!pool_.empty()) {
            PooledBuffer<T>* buffer = pool_.front();
            pool_.pop();
            delete buffer; // Delete the raw buffer
        }
    }

    /**
     * @brief Acquires a buffer from the pool, waiting if none are available.
     *
     * @param timeout The maximum duration to wait for a buffer.
     * @return A PooledPtr (a shared_ptr with a custom deleter) to a buffer.
     *         If a timeout occurs, the shared_ptr will be null.
     */
    PooledPtr acquire(std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cond_var_.wait_for(lock, timeout, [this] { return !pool_.empty(); })) {
            PooledBuffer<T>* raw_buffer = pool_.front();
            pool_.pop();
            APP_LOG_DEBUG(name_ + ": Acquired buffer. Available: " + std::to_string(pool_.size()));
            
            // Create a shared_ptr with a custom deleter that returns the raw buffer to the pool.
            return PooledPtr(raw_buffer, [this](PooledBuffer<T>* ptr) {
                this->release(ptr);
            });
        }
        // Timeout occurred
        APP_LOG_WARNING(name_ + ": Failed to acquire buffer within timeout. Available: " + std::to_string(pool_.size()));
        return nullptr;
    }

    /**
     * @brief Returns the number of available buffers in the pool.
     */
    size_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }

private:
    /**
     * @brief Releases a buffer, returning it to the pool.
     *
     * This method is called automatically by the custom deleter of the PooledPtr.
     *
     * @param raw_buffer The raw pointer to the buffer to be returned.
     */
    void release(PooledBuffer<T>* raw_buffer) {
        std::unique_lock<std::mutex> lock(mutex_);
        pool_.push(raw_buffer);
        APP_LOG_DEBUG(name_ + ": Released buffer. Available: " + std::to_string(pool_.size()));
        cond_var_.notify_one();
    }

    std::string name_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    std::queue<PooledBuffer<T>*> pool_; // Store raw pointers in the queue
};

#endif // BUFFER_POOL_H