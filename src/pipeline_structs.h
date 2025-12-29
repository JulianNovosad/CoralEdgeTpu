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
#include <boost/lockfree/spsc_queue.hpp> // For lock-free SPSC queues
#include <libcamera/pixel_format.h> // Include for libcamera::PixelFormat

// --- Generic Data Structures ---

/**
 * @brief Represents raw image data, now using a pooled buffer.
 *
 * Contains a shared pointer to a buffer from a pool, dimensions, and a timestamp.
 * This avoids deep copies of pixel data.
 */
struct ImageData {
    std::shared_ptr<PooledBuffer<uint8_t>> buffer; ///< Shared pointer to a pooled buffer holding pixel data.
    size_t width;                                  ///< Width of the image in pixels.
    size_t height;                                 ///< Height of the image in pixels.
    libcamera::PixelFormat format;                 ///< Pixel format of the image data.
    long long timestamp_epoch_ms = 0;              ///< Timestamp of image capture in epoch milliseconds UTC.
    int frame_id = -1;                             ///< Monotonically increasing frame ID.
    
    // Zero-copy fields
    int fd = -1;                                   ///< File descriptor for zero-copy access to frame buffer.
    size_t offset = 0;                             ///< Offset within the file descriptor for zero-copy access.
    size_t length = 0;                             ///< Length of the frame data for zero-copy access.

    // Timing measurements
    std::chrono::high_resolution_clock::time_point capture_time;     ///< Time when frame was captured
    std::chrono::high_resolution_clock::time_point queue_pop_time;   ///< Time when frame was popped from queue
    std::chrono::high_resolution_clock::time_point preprocess_start_time; ///< Time when preprocessing started
    std::chrono::high_resolution_clock::time_point preprocess_end_time;   ///< Time when preprocessing ended
    std::chrono::high_resolution_clock::time_point inference_start_time;  ///< Time when inference started
    std::chrono::high_resolution_clock::time_point inference_end_time;    ///< Time when inference ended
    std::chrono::high_resolution_clock::time_point encode_start_time;     ///< Time when encoding started
    std::chrono::high_resolution_clock::time_point encode_end_time;       ///< Time when encoding ended
    std::chrono::high_resolution_clock::time_point rtsp_push_start_time;  ///< Time when RTSP push started
    std::chrono::high_resolution_clock::time_point rtsp_push_end_time;    ///< Time when RTSP push ended
    std::chrono::high_resolution_clock::time_point ingest_start_time;     ///< Time when frame ingest started
    std::chrono::high_resolution_clock::time_point ingest_end_time;       ///< Time when frame ingest ended
    std::chrono::high_resolution_clock::time_point conversion_start_time; ///< Time when format conversion started
    std::chrono::high_resolution_clock::time_point conversion_end_time;   ///< Time when format conversion ended
    std::chrono::high_resolution_clock::time_point visualization_start_time; ///< Time when visualization started
    std::chrono::high_resolution_clock::time_point visualization_end_time;   ///< Time when visualization ended
    std::chrono::high_resolution_clock::time_point display_start_time;    ///< Time when frame display started
    std::chrono::high_resolution_clock::time_point display_end_time;      ///< Time when frame display ended

    // Static member for global frame counter
    static std::atomic<int> global_frame_counter;

    // Constructor to initialize timestamp and frame_id
    ImageData(long long ts_epoch_ms = 0, int f_id = -1)
        : timestamp_epoch_ms(ts_epoch_ms), frame_id(f_id) {}
};

/**
 * @brief Represents orientation data.
 *
 * Contains yaw, pitch, and roll readings, along with a timestamp for when the data was captured.
 */
struct OrientationData {
    float yaw;   ///< Yaw angle in degrees or radians.
    float pitch; ///< Pitch angle in degrees or radians.
    float roll;  ///< Roll angle in degrees or radians.
    std::chrono::high_resolution_clock::time_point timestamp; ///< Timestamp of orientation data capture.

    // Default constructor to initialize all members to 0 or an appropriate default
    OrientationData() : yaw(0.0f), pitch(0.0f), roll(0.0f),
                        timestamp(std::chrono::high_resolution_clock::now()) {}
};

/**
 * @brief Represents a single object detection result.
 * *
 * Stores the class ID, confidence score, and bounding box coordinates
 * for a detected object within an image frame.
 */
struct DetectionResult {
    int class_id;   ///< The ID of the detected class.
    float score;    ///< The confidence score of the detection (0.0 - 1.0, normalized).
    float raw_score; ///< Raw dequantized model output for debugging.
    float xmin, ymin, xmax, ymax; ///< Bounding box coordinates (normalized 0.0 - 1.0 or pixel values).
    std::chrono::high_resolution_clock::time_point timestamp; ///< Timestamp of when the detection was made.
    int source_frame_id = -1; ///< ID of the source frame that generated this detection.
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

/// @brief Type alias for a collection of detection results.
typedef std::vector<DetectionResult> DetectionResults;

/**
 * @brief A high-performance triple buffer for asynchronous "latest-wins" data transfer.
 * 
 * Uses three distinct slots (Producer, Consumer, Latest) and atomic index swaps
 * to ensure non-blocking operation for both producer and consumer.
 */
template<typename T>
class TripleBuffer {
public:
    TripleBuffer() {
        dirty_.store(false, std::memory_order_relaxed);
        latest_index_.store(0, std::memory_order_relaxed);
        producer_index_ = 1;
        consumer_index_ = 2;
        
        // Performance: Pre-allocate vector capacity to avoid runtime heap allocations
        // only if T is DetectionResults.
        if constexpr (std::is_same_v<T, DetectionResults>) {
            buffers_[0].reserve(100);
            buffers_[1].reserve(100);
            buffers_[2].reserve(100);
        }
    }

    /**
     * @brief Gets a reference to the buffer for writing (Producer only).
     */
    T& get_write_buffer() {
        return buffers_[producer_index_];
    }

    /**
     * @brief Commits the current write buffer and swaps it with the 'Latest' slot (Producer only).
     */
    void commit_write() {
        producer_index_ = latest_index_.exchange(producer_index_, std::memory_order_acq_rel);
        dirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Updates the consumer buffer to the latest available clean buffer (Consumer only).
     * @return True if a new buffer was acquired.
     */
    bool update_consumer() {
        if (!dirty_.exchange(false, std::memory_order_acquire)) {
            return false;
        }
        consumer_index_ = latest_index_.exchange(consumer_index_, std::memory_order_acq_rel);
        return true;
    }

    /**
     * @brief Gets a reference to the current consumer buffer (Consumer only).
     */
    const T& get_read_buffer() const {
        return buffers_[consumer_index_];
    }

private:
    T buffers_[3];
    std::atomic<int> latest_index_;
    int producer_index_;
    int consumer_index_;
    std::atomic<bool> dirty_;
};

// --- Type aliases for all pipeline queues ---

/**
 * @brief A thread-safe, Multi-Producer Multi-Consumer queue.
 */
template<typename T>
class MPMCQueue {
public:
    MPMCQueue() = default;

    void push(const T& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(data);
        lock.unlock();
        cond_.notify_all();
    }

    void push(T&& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(data));
        lock.unlock();
        cond_.notify_all();
    }

    bool pop(T& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        data = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool wait_pop(T& data, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return false;
        }
        data = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    size_t read_available() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // Dummy for compatibility with spsc_queue calls
    size_t write_available() { return 100; }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};

/// @brief Type alias for a thread-safe MPMC queue holding ImageData objects.
typedef MPMCQueue<ImageData> ImageQueue;

// Define a type for a pooled buffer of detection results
using DetectionResultBuffer = PooledBuffer<DetectionResult>;
/// @brief Type alias for a thread-safe MPMC queue holding shared pointers to pooled detection result buffers.
using DetectionResultsQueue = MPMCQueue<std::shared_ptr<DetectionResultBuffer>>;

// Define a type for a pooled buffer of H.264 NAL units
using H264Buffer = PooledBuffer<uint8_t>;
/// @brief Type alias for a thread-safe MPMC queue holding shared pointers to pooled H.264 buffers.
using H264Queue = MPMCQueue<std::shared_ptr<H264Buffer>>;

#endif // PIPELINE_STRUCTS_H