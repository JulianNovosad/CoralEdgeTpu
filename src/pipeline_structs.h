#ifndef PIPELINE_STRUCTS_H
#define PIPELINE_STRUCTS_H

#include <vector>
#include <string>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory> // For std::shared_ptr
#include <functional> // For std::function
#include "buffer_pool.h" // For BufferPool


// --- Generic Data Structures ---

/**
 * @brief Represents raw image data, now using a pooled buffer.
 *
 * Contains a shared pointer to a buffer from a pool, dimensions, and a timestamp.
 * This avoids deep copies of pixel data.
 */
struct ImageData {
    BufferPool<uint8_t>::PooledPtr buffer; ///< Shared pointer to a pooled buffer holding pixel data.
    size_t width;                                  ///< Width of the image in pixels.
    size_t height;                                 ///< Height of the image in pixels.
    std::chrono::high_resolution_clock::time_point timestamp; ///< Timestamp of image capture.
};

/**
 * @brief Represents a single object detection result.
 *
 * Stores the class ID, confidence score, and bounding box coordinates
 * for a detected object within an image frame.
 */
struct DetectionResult {
    int class_id;   ///< The ID of the detected class.
    float score;    ///< The confidence score of the detection (0.0 - 1.0).
    float xmin, ymin, xmax, ymax; ///< Bounding box coordinates (normalized 0.0 - 1.0 or pixel values).
    std::chrono::high_resolution_clock::time_point timestamp; ///< Timestamp of when the detection was made.
};

/**
 * @brief Combines ImageData with associated DetectionResults.
 *
 * This struct could be used if there's a need to pass the original image
 * along with its detections in a single unit through the pipeline.
 */
struct InferenceFrame {
    ImageData image;                    ///< The raw image data.
    std::vector<DetectionResult> detections; ///< Vector of detection results for this image.
};

// --- Thread-Safe Queue Template ---

/**
 * @brief A generic, thread-safe queue implementation.
 *
 * This template class provides a mechanism for safely passing data between
 * different threads in the application pipeline. It uses a mutex and
 * condition variable for synchronization, allowing producers to push data
 * and consumers to pop data without race conditions. It also supports
 * a specialized push for MJPEG to keep only the latest frame and a non-blocking peek.
 *
 * @tparam T The type of data stored in the queue.
 */
template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}

    // Delete copy constructor and copy assignment operator due to std::atomic member
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    /**
     * @brief Pushes a new data item into the queue.
     *
     * Locks the mutex, pushes the data, and notifies one waiting consumer.
     *
     * @param new_data The data item to be pushed.
     */
    void push(T&& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(data));
        // lock.unlock(); // Removed unlock before notify
        cond_var_.notify_one();
    }

    void push_and_drop_if_full(T&& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            queue_.pop();
        }
        queue_.push(std::move(data));
        // lock.unlock(); // Removed unlock before notify
        cond_var_.notify_one();
    }

    /**
     * @brief Specialized push operation for MJPEG frames.
     *
     * This method ensures that the queue always contains only the latest frame.
     * If the queue is not empty, it discards the old frame before pushing the new one.
     * This is useful for real-time video streams where only the most recent frame matters.
     *
     * @param new_frame The MJPEG frame to be pushed.
     */
    void push_mjpeg(T&& new_frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!queue_.empty()) {
            queue_.pop(); // Discard older frame
        }
        queue_.push(std::move(new_frame));
        cond_var_.notify_one();
    }

    /**
     * @brief Pops a data item from the front of the queue.
     *
     * Locks the mutex and waits until the queue is not empty or the queue is
     * no longer running. If the queue is running and an item is available,
     * it is retrieved and removed.
     *
     * @param data Reference to where the popped data item will be stored.
     * @return True if a data item was successfully popped, false if the queue
     *         is stopped and empty.
     */
    bool pop(T& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        // Wait until queue is not empty OR the `running_` flag is false (signaling shutdown)
        cond_var_.wait(lock, [this]{ return !queue_.empty() || !running_; });
        if (queue_.empty()) {
            return false; // Queue is empty and stopped.
        }
        data = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    /**
     * @brief Peeks at the latest item in the queue without removing it, executing a function on it.
     *
     * This is a non-blocking peek once data is available. It waits until the
     * queue is not empty or the queue is no longer running. If an item is
     * available, it executes the provided function with the latest item.
     * This avoids expensive copies under the lock.
     *
     * @param visitor A function or lambda to be executed with the latest item.
     * @return True if a data item was successfully peeked and the visitor was called,
     *         false if the queue is stopped and empty.
     */
    bool peek_latest(std::function<void(const T&)> visitor) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]{ return !queue_.empty() || !running_; });
        if (queue_.empty()) {
            return false; // Queue is empty and stopped.
        }
        visitor(queue_.back()); // Execute visitor on the latest frame
        return true;
    }
    
    /**
     * @brief Sets the running state of the queue.
     *
     * When set to false, it notifies all waiting consumers, allowing them
     * to exit their wait loops.
     *
     * @param val The new running state (true for running, false for stopped).
     */
    void set_running(bool val) {
        running_ = val;
        if (!val) {
            cond_var_.notify_all(); // Notify all waiting threads on shutdown
        }
    }

    /**
     * @brief Clears all items from the queue.
     *
     * This method locks the queue and removes all elements. It should be called
     * during shutdown to ensure proper destruction of elements before
     * BufferPools might be deallocated.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty_queue;
        std::swap(queue_, empty_queue); // Efficiently clear the queue
    }

private:
    mutable std::mutex mutex_; ///< Mutex for protecting access to the queue.
    std::queue<T> queue_;      ///< The underlying standard queue.
    std::condition_variable cond_var_; ///< Condition variable for signaling between producers/consumers.
    std::atomic<bool> running_ = true; ///< Atomic flag to indicate if the queue is active.
    size_t max_size_;
};

// --- Type aliases for all pipeline queues ---

/// @brief Type alias for a thread-safe queue holding ImageData objects.
using ImageQueue = ThreadSafeQueue<ImageData>;

// Define a type for a pooled buffer of detection results
using DetectionResultBuffer = PooledBuffer<DetectionResult>;
/// @brief Type alias for a thread-safe queue holding shared pointers to pooled detection result buffers.
using DetectionResultsQueue = ThreadSafeQueue<std::shared_ptr<DetectionResultBuffer>>;

// Define a type for a pooled buffer of H.264 NAL units
using H264Buffer = PooledBuffer<uint8_t>;
/// @brief Type alias for a thread-safe queue holding shared pointers to pooled H.264 buffers.
using H264Queue = ThreadSafeQueue<std::shared_ptr<H264Buffer>>;

#endif // PIPELINE_STRUCTS_H
