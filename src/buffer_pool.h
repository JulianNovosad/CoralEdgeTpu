#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <set>
#include <stdexcept>
#include <functional> // For std::function

#include "util_logging.h" // Include logging utilities
#include <iostream>
#include <boost/lockfree/queue.hpp>


// A generic buffer object that can be pooled.
template <typename T>
struct PooledBuffer {
    std::vector<T> data;
    // Add any other metadata you need, e.g., size, timestamp.
    size_t size = 0;
    bool valid = true;                 // Heartbeat validity flag
    // Timestamp for video frames to enable proper PTS calculation
    long long timestamp_epoch_ms = 0;  // Timestamp in milliseconds since epoch
    uint64_t t_capture_raw_ms = 0;     // Deterministic capture time
    uint64_t t_inf_start = 0;          // Inference start
    uint64_t t_inf_end = 0;            // Inference end
    int frame_id = 0;                 // Frame ID to track frame sequence
    int64_t encoder_frame_count = -1;  // Frame count from encoder for PTS calculation
    
    // Telemetry fields
    float cam_exposure_ms = -1.0f;
    float cam_isp_latency_ms = -1.0f;
    float cam_buffer_usage_percent = -1.0f;
    float image_proc_ms = -1.0f;
    float tpu_temp_c = -1.0f;
    
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
        : name_(name), total_buffers_(pool_size), current_in_use_(0), peak_in_use_(0) {
        buffers_storage_.reserve(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            buffers_storage_.emplace_back(); // Store the actual PooledBuffer object
            buffers_storage_.back().data.resize(buffer_data_size);
            available_buffers_.push(&buffers_storage_.back()); // Push raw pointer to queue
        }
        APP_LOG_INFO("BufferPool '" + name_ + "' created with " + std::to_string(pool_size) + " buffers of fixed size " + std::to_string(buffer_data_size));
    }

    // Constructor for variable-size buffer types (e.g., uint8_t for images)
    BufferPool(size_t pool_size, size_t min_buffer_size, size_t max_buffer_size, const std::string& name)
        : name_(name), total_buffers_(pool_size), current_in_use_(0), peak_in_use_(0) {
        buffers_storage_.reserve(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            buffers_storage_.emplace_back();
            buffers_storage_.back().data.resize(min_buffer_size); // Initial allocation
            available_buffers_.push(&buffers_storage_.back()); // Push raw pointer to queue
        }
        APP_LOG_INFO("BufferPool '" + name_ + "' created with " + std::to_string(pool_size) + " buffers of variable size (min: " + std::to_string(min_buffer_size) + ", max: " + std::to_string(max_buffer_size) + ")");
    }

    std::shared_ptr<PooledBuffer<T>> acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, std::chrono::milliseconds(100), // Reduced timeout to 100ms
                            [this]{ return !available_buffers_.empty(); })) {
            // Log detailed information about the state when acquisition fails
            size_t available = available_buffers_.size();
            size_t in_use = current_in_use_.load();
            APP_LOG_WARNING(name_ + ": Failed to acquire buffer within timeout. Available: " + std::to_string(available) + 
                           ", In use: " + std::to_string(in_use) + ", Peak: " + std::to_string(peak_in_use_.load()));
            return nullptr; // Timeout
        }
        PooledBuffer<T>* buffer_ptr = available_buffers_.front();
        available_buffers_.pop();
        
        // Update usage tracking
        size_t current = current_in_use_.fetch_add(1) + 1;
        size_t peak = peak_in_use_.load();
        while (current > peak && !peak_in_use_.compare_exchange_weak(peak, current));
        
        lock.unlock();

        // Create a shared_ptr with a custom deleter that returns the raw pointer to the pool
        // Use an atomic flag to ensure the buffer is only returned once
        auto return_flag = std::make_shared<std::atomic<bool>>(false);
        return std::shared_ptr<PooledBuffer<T>>(buffer_ptr, [this, return_flag](PooledBuffer<T>* b) {
            // Use atomic compare-and-swap to ensure buffer is only returned once
            bool expected = false;
            if (return_flag->compare_exchange_strong(expected, true)) {
                // Only return the buffer if we're the first to mark it as returned
                std::unique_lock<std::mutex> local_lock(this->mutex_);
                this->available_buffers_.push(b); // Push the raw pointer back to the queue
                
                // Update usage tracking on release
                this->current_in_use_.fetch_sub(1);
                
                local_lock.unlock();
                this->cond_.notify_one();
            }
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

    /**
     * @brief Manually acquire a raw buffer pointer from the pool.
     * @return Raw pointer to a PooledBuffer, or nullptr on timeout.
     * @warning The caller is responsible for returning the buffer to the pool
     *          using release_raw().
     */
    PooledBuffer<T>* acquire_raw() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, std::chrono::milliseconds(100),
                            [this]{ return !available_buffers_.empty(); })) {
            return nullptr;
        }
        PooledBuffer<T>* buffer_ptr = available_buffers_.front();
        available_buffers_.pop();
        current_in_use_.fetch_add(1);
        return buffer_ptr;
    }

    // No explicit release method needed, as it's handled by the custom deleter
    
    size_t get_available_buffers() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_buffers_.size();
    }

    size_t get_total_buffers() const {
        return total_buffers_;
    }
    
    size_t get_current_in_use() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_in_use_;
    }
    
    size_t get_peak_in_use() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_in_use_;
    }
    
    // Method to check for potential buffer leaks
    bool has_leaked_buffers() const {
        std::lock_guard<std::mutex> lock(mutex_);
        // If available buffers plus acquired buffers don't equal total buffers, we have a leak
        // Note: This is a simplified check and assumes no buffers are currently acquired
        return available_buffers_.size() != total_buffers_;
    }
    
    // Method to get buffer pool statistics
    std::pair<size_t, size_t> get_buffer_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::make_pair(available_buffers_.size(), total_buffers_);
    }

    /**
     * @brief Manually release a raw buffer pointer back to the pool.
     * @warning Use only when the buffer was acquired from this pool and 
     *          is no longer managed by any shared_ptr.
     */
    void release_raw(PooledBuffer<T>* buffer) {
        if (!buffer) return;
        std::unique_lock<std::mutex> lock(mutex_);
        available_buffers_.push(buffer);
        current_in_use_.fetch_sub(1);
        lock.unlock();
        cond_.notify_one();
    }

private:
    std::string name_;
    size_t total_buffers_;
    mutable std::mutex mutex_;
    std::queue<PooledBuffer<T>*> available_buffers_;
    std::condition_variable cond_;
    std::vector<PooledBuffer<T>> buffers_storage_; // Store actual buffer objects
    
    // Usage tracking
    mutable std::atomic<size_t> current_in_use_;
    mutable std::atomic<size_t> peak_in_use_;
};

/**

 * @brief A simple, thread-safe pool for managing reusable objects of type T.

 */

template <typename T>

class ObjectPool {

public:

    ObjectPool(size_t size, const std::string& name) : name_(name), total_size_(size), pool_(size + 10) {

        for (size_t i = 0; i < size; ++i) {

            pool_.push(new T());

        }

        APP_LOG_INFO("ObjectPool '" + name_ + "' created with " + std::to_string(size) + " objects.");

    }



    ~ObjectPool() {

        T* obj;

        while (pool_.pop(obj)) {

            delete obj;

        }

    }



    T* acquire() {

        T* obj = nullptr;

        if (pool_.pop(obj)) {

            return obj;

        }

        return nullptr;

    }



    void release(T* obj) {

        if (obj) {

            if (!pool_.push(obj)) {

                delete obj; // Safety: if pool is somehow full (shouldn't happen with pointers)

            }

        }

    }



private:

    std::string name_;

    size_t total_size_;

    boost::lockfree::queue<T*, boost::lockfree::fixed_sized<true>> pool_;

};





#endif // BUFFER_POOL_H