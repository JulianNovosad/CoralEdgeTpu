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
        auto acquire_start = std::chrono::high_resolution_clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, std::chrono::milliseconds(100), // Reduced timeout to 100ms
                            [this]{ return !available_buffers_.empty(); })) {
            // Log detailed information about the state when acquisition fails
            size_t available = available_buffers_.size();
            size_t in_use = total_buffers_ - available;
            APP_LOG_WARNING(name_ + ": Failed to acquire buffer within timeout. Available: " + std::to_string(available) + 
                           ", In use: " + std::to_string(in_use) + ", Peak: " + std::to_string(peak_in_use_));
            return nullptr; // Timeout
        }
        PooledBuffer<T>* buffer_ptr = available_buffers_.front();
        available_buffers_.pop();
        
        // Update usage tracking
        current_in_use_ = total_buffers_ - available_buffers_.size();
        if (current_in_use_ > peak_in_use_) {
            peak_in_use_ = current_in_use_;
        }
        
        // Log acquisition with buffer address and timing
        auto acquire_end = std::chrono::high_resolution_clock::now();
        auto wait_time_us = std::chrono::duration_cast<std::chrono::microseconds>(acquire_end - acquire_start).count();
        APP_LOG_INFO(name_ + ": Acquired buffer " + std::to_string(reinterpret_cast<uintptr_t>(buffer_ptr)) + 
                    ". Available: " + std::to_string(available_buffers_.size()) + 
                    ", In use: " + std::to_string(current_in_use_) + 
                    ", Wait time: " + std::to_string(wait_time_us) + " us");
        
        lock.unlock();

        // Create a shared_ptr with a custom deleter that returns the raw pointer to the pool
        // Use an atomic flag to ensure the buffer is only returned once
        auto return_flag = std::make_shared<std::atomic<bool>>(false);
        return std::shared_ptr<PooledBuffer<T>>(buffer_ptr, [this, return_flag, acquire_end](PooledBuffer<T>* b) {
            auto release_start = std::chrono::high_resolution_clock::now();
            auto hold_time_us = std::chrono::duration_cast<std::chrono::microseconds>(release_start - acquire_end).count();
            
            // Log release with buffer address and timing
            APP_LOG_INFO(this->name_ + ": Releasing buffer " + std::to_string(reinterpret_cast<uintptr_t>(b)) + 
                        ", Hold time: " + std::to_string(hold_time_us) + " us");
            
            // Use atomic compare-and-swap to ensure buffer is only returned once
            bool expected = false;
            if (return_flag->compare_exchange_strong(expected, true)) {
                // Only return the buffer if we're the first to mark it as returned
                std::unique_lock<std::mutex> local_lock(this->mutex_);
                this->available_buffers_.push(b); // Push the raw pointer back to the queue
                
                // Update usage tracking on release
                this->current_in_use_ = this->total_buffers_ - this->available_buffers_.size();
                
                local_lock.unlock();
                this->cond_.notify_one();
            } else {
                // Buffer was already returned by another shared_ptr instance
                APP_LOG_ERROR(this->name_ + ": Attempted to return buffer " + 
                            std::to_string(reinterpret_cast<uintptr_t>(b)) + 
                            " that was already returned by another shared_ptr. Preventing double-free.");
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

private:
    std::string name_;
    size_t total_buffers_;
    mutable std::mutex mutex_;
    std::queue<PooledBuffer<T>*> available_buffers_;
    std::condition_variable cond_;
    std::vector<PooledBuffer<T>> buffers_storage_; // Store actual buffer objects
    
    // Usage tracking
    mutable size_t current_in_use_;
    mutable size_t peak_in_use_;
};

#endif // BUFFER_POOL_H