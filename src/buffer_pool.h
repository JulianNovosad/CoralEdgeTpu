#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <stdexcept>

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
 * returned to the pool automatically using a custom deleter with std::unique_ptr.
 *
 * @tparam T The type of data stored in the buffer's vector (e.g., uint8_t).
 */
template <typename T>
class BufferPool {
public:
    // Defines a unique_ptr that will automatically return the buffer to the pool.
    using PooledPtr = std::unique_ptr<PooledBuffer<T>, std::function<void(PooledBuffer<T>*)>>;

    /**
     * @brief Constructs a BufferPool.
     *
     * @param num_buffers The number of buffers to pre-allocate in the pool.
     * @param buffer_size The size of each buffer to pre-allocate.
     * @param pool_name A name for logging/debugging purposes.
     */
    BufferPool(size_t num_buffers, size_t buffer_size, std::string pool_name = "BufferPool")
        : name_(std::move(pool_name)) {
        if (num_buffers == 0 || buffer_size == 0) {
            throw std::invalid_argument("BufferPool: Number of buffers and buffer size must be greater than 0.");
        }
        
        for (size_t i = 0; i < num_buffers; ++i) {
            auto buffer = std::make_unique<PooledBuffer<T>>();
            buffer->data.resize(buffer_size); // Pre-allocate memory
            pool_.push(std::move(buffer));
        }
    }

    // Disable copy and assignment
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    /**
     * @brief Acquires a buffer from the pool, waiting if none are available.
     *
     * @param timeout The maximum duration to wait for a buffer.
     * @return A PooledPtr (a unique_ptr with a custom deleter) to a buffer.
     *         If a timeout occurs, the unique_ptr will be null.
     */
    PooledPtr acquire(std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cond_var_.wait_for(lock, timeout, [this] { return !pool_.empty(); })) {
            auto buffer = std::move(pool_.front());
            pool_.pop();
            return PooledPtr(buffer.release(), [this](PooledBuffer<T>* ptr) {
                this->release(std::unique_ptr<PooledBuffer<T>>(ptr));
            });
        }
        // Timeout occurred
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
     * @param buffer The unique_ptr to the buffer to be returned.
     */
    void release(std::unique_ptr<PooledBuffer<T>> buffer) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pool_.push(std::move(buffer));
        }
        cond_var_.notify_one();
    }

    std::string name_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    std::queue<std::unique_ptr<PooledBuffer<T>>> pool_;
};

#endif // BUFFER_POOL_H
