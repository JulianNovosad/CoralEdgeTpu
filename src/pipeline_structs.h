/**
 * @file pipeline_structs.h
 * @brief Defines common data structures and a thread-safe queue template
 *        used across the CoralEdgeTpu Detector application pipeline.
 *
 * This header centralizes the definitions for image data, detection results,
 * and frames, along with a robust thread-safe queue implementation essential
 * for inter-module communication in a multi-threaded real-time system.
 */

#ifndef PIPELINE_STRUCTS_H
#define PIPELINE_STRUCTS_H

#include <vector>
#include <string>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

// --- Generic Data Structures ---

/**
 * @brief Represents raw image data captured from a camera.
 *
 * Contains the raw pixel data, dimensions, and a timestamp for when the
 * frame was captured, useful for latency measurements.
 */
struct ImageData {
    std::vector<uint8_t> data; ///< Raw pixel data (e.g., BGR888 bytes).
    size_t width;              ///< Width of the image in pixels.
    size_t height;             ///< Height of the image in pixels.
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
 * @brief Represents an MJPEG encoded image frame.
 *
 * Contains the JPEG compressed data and dimensions, used for streaming
 * to clients (e.g., web browsers).
 */
struct ImageFrame {
    std::vector<uint8_t> jpeg_data; ///< JPEG compressed image data.
    size_t width;                   ///< Width of the image in pixels.
    size_t height;                  ///< Height of the image in pixels.
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
    /**
     * @brief Pushes a new data item into the queue.
     *
     * Locks the mutex, pushes the data, and notifies one waiting consumer.
     *
     * @param new_data The data item to be pushed.
     */
    void push(T new_data) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(new_data));
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
    void push_mjpeg(T new_frame) {
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
     * @brief Peeks at the latest item in the queue without removing it.
     *
     * This is a non-blocking peek once data is available. It waits until the
     * queue is not empty or the queue is no longer running. If an item is
     * available, it copies the last item. Useful for servers that only need
     * the most recent state.
     *
     * @param frame Reference to where the latest data item will be copied.
     * @return True if a data item was successfully peeked, false if the queue
     *         is stopped and empty.
     */
    bool peek_latest(T& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        // Wait until queue is not empty OR the `running_` flag is false (signaling shutdown)
        cond_var_.wait(lock, [this]{ return !queue_.empty() || !running_; });
        if (queue_.empty()) {
            return false; // Queue is empty and stopped.
        }
        frame = queue_.back(); // Get the latest frame
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

private:
    mutable std::mutex mutex_; ///< Mutex for protecting access to the queue.
    std::queue<T> queue_;      ///< The underlying standard queue.
    std::condition_variable cond_var_; ///< Condition variable for signaling between producers/consumers.
    std::atomic<bool> running_ = true; ///< Atomic flag to indicate if the queue is active.
};

// --- Type aliases for all pipeline queues ---

/// @brief Type alias for a thread-safe queue holding ImageData objects.
using ImageQueue = ThreadSafeQueue<ImageData>;
/// @brief Type alias for a thread-safe queue holding vectors of DetectionResult objects.
using UdpQueue = ThreadSafeQueue<std::vector<DetectionResult>>;
/// @brief Type alias for a thread-safe queue holding ImageFrame (MJPEG) objects.
using MjpegQueue = ThreadSafeQueue<ImageFrame>;

#endif // PIPELINE_STRUCTS_H