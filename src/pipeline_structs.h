// Verified headers: [vector, string, chrono, queue, mutex...]
// Verification timestamp: 2026-01-06 17:08:04
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
#include "lockfree_queue.h"      // For LockFreeQueue
#include <libcamera/pixel_format.h> // Include for libcamera::PixelFormat
#include <thread>

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
    size_t stride;                                 ///< Stride (bytes per line) of the image data.
    libcamera::PixelFormat format;                 ///< Pixel format of the image data.
    int frame_id = 0;                             ///< Monotonically increasing frame ID.
    
    // Zero-copy fields
    int fd = -1;                                   ///< File descriptor for zero-copy access to frame buffer.
    size_t offset = 0;                             ///< Offset within the file descriptor for zero-copy access.
    size_t length = 0;                             ///< Length of the frame data for zero-copy access.

    // Timing measurements (Deterministic)
    uint64_t t_capture_raw_ms = 0;                  ///< Authoritative raw ms from get_time_raw_ms()
    std::chrono::steady_clock::time_point capture_time;     ///< Time when frame was captured (PRIMARY TIMESTAMP)

    // Telemetry fields inherited from Camera
    float cam_exposure_ms = -1.0f;
    float cam_isp_latency_ms = -1.0f;
    float cam_buffer_usage_percent = -1.0f;
    float image_proc_ms = -1.0f;
    float tpu_temp_c = -1.0f;

    bool isValid() const { return buffer != nullptr; }

    // Per-frame accounting fields... (truncated for brevity but I will keep them)
    std::chrono::steady_clock::time_point queue_pop_time;   ///< Time when frame was popped from queue
    std::chrono::steady_clock::time_point preprocess_start_time; ///< Time when preprocessing started
    std::chrono::steady_clock::time_point preprocess_end_time;   ///< Time when preprocessing ended
    std::chrono::steady_clock::time_point inference_start_time;  ///< Time when inference started
    std::chrono::steady_clock::time_point inference_end_time;    ///< Time when inference ended
    std::chrono::steady_clock::time_point encode_start_time;     ///< Time when encoding started
    std::chrono::steady_clock::time_point encode_end_time;       ///< Time when encoding ended
    std::chrono::steady_clock::time_point rtsp_push_start_time;  ///< Time when RTSP push started
    std::chrono::steady_clock::time_point rtsp_push_end_time;    ///< Time when RTSP push ended
    std::chrono::steady_clock::time_point ingest_start_time;     ///< Time when frame ingest started
    std::chrono::steady_clock::time_point ingest_end_time;       ///< Time when frame ingest ended
    std::chrono::steady_clock::time_point conversion_start_time; ///< Time when format conversion started
    std::chrono::steady_clock::time_point conversion_end_time;   ///< Time when format conversion ended
    std::chrono::steady_clock::time_point visualization_start_time; ///< Time when visualization started
    std::chrono::steady_clock::time_point visualization_end_time;   ///< Time when visualization ended
    std::chrono::steady_clock::time_point display_start_time;    ///< Time when frame display started
    std::chrono::steady_clock::time_point display_end_time;      ///< Time when frame display ended

    // Static member for global frame counter
    static std::atomic<int> global_frame_counter;

    // Constructor to initialize capture_time and frame_id
    ImageData(std::chrono::steady_clock::time_point ts = std::chrono::steady_clock::now(), int f_id = 0)
        : buffer(nullptr), width(0), height(0), stride(0), frame_id(f_id), fd(-1), offset(0), length(0), t_capture_raw_ms(0), capture_time(ts) {}
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
    std::chrono::steady_clock::time_point timestamp; ///< Timestamp of orientation data capture.

    // Default constructor to initialize all members to 0 or an appropriate default
    OrientationData() : yaw(0.0f), pitch(0.0f), roll(0.0f),
                        timestamp(std::chrono::steady_clock::now()) {}
};

/**
 * @brief Represents a single object detection result.
 *
 * Stores the class ID, confidence score, and bounding box coordinates
 * for a detected object within an image frame.
 */
struct DetectionResult {
    int class_id;                      ///< The ID of the detected class.
    float score;                        ///< Confidence score (0.0 - 1.0, normalized).
    float raw_score;                    ///< Raw dequantized model output for debugging.
    float xmin, ymin, xmax, ymax;       ///< Bounding box coordinates (normalized 0.0 - 1.0 or pixel values).
    std::chrono::steady_clock::time_point timestamp; ///< Timestamp of when detection was made.
    uint64_t t_capture_raw_ms = 0;      ///< Monotonic capture time inherited from ImageData (ms since boot).
    int source_frame_id = 0;            ///< ID of the source frame that generated this detection.

    // Telemetry fields carried for logging
    float cam_exposure_ms = -1.0f;
    float cam_isp_latency_ms = -1.0f;
    float cam_buffer_usage_percent = -1.0f;
    float tpu_temp_c = -1.0f;
};

/**
 * @brief Represents a ballistic impact point and safety metadata in overlay pixel coordinates.
 *
 * For visual validation only; overlay-space only (no world coords).
 * Coordinate origin: top-left of image.
 */
struct OverlayBallisticPoint {
    int x = 0;                          ///< Legacy X-coordinate in pixels.
    int y = 0;                          ///< Legacy Y-coordinate in pixels.
    bool is_valid = false;              ///< True if point is valid and should be drawn.
    int frame_id = 0;                   ///< Frame ID that this point is bound to.
    
    // Avant-Garde enhanced fields
    float impact_px_x = 0.0f;           ///< Predicted impact point X (pixels)
    float impact_px_y = 0.0f;           ///< Predicted impact point Y (pixels)
    float safety_cone_radius_px = 0.0f;  ///< Radius of safety cone in pixels
    bool safety_cone_violation = false;  ///< True if safety cone exceeds inner bounding box
    
    // Target Bounding Box / Inner Fraction (normalized 0.0-1.0)
    float inner_xmin = 0.0f;
    float inner_ymin = 0.0f;
    float inner_xmax = 0.0f;
    float inner_ymax = 0.0f;
    
    float confidence = 0.0f;            ///< Total confidence (0.0 - 1.0)
    int hit_streak = 0;                 ///< Current hit streak
};

/**
 * @brief Black Box Telemetry record for a single frame.
 */
struct TelemetryFrame {
    uint64_t frame_id;
    uint64_t t_capture;
    uint64_t t_inf_start;
    uint64_t t_inf_end;
    uint64_t t_logic_start;
    uint64_t t_logic_end;
    float target_x, target_y, target_z;
    int state;
    bool hit_scan;
};

/**
 * @brief Combines ImageData with associated DetectionResults.
 *
 * Use if passing the original image with its detections through the pipeline.
 */
struct InferenceFrame {
    ImageData image;                        ///< The raw image data.
    std::vector<DetectionResult> detections;///< Vector of detection results for this image.
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
        int old_index = latest_index_.exchange(producer_index_, std::memory_order_acq_rel);
        producer_index_ = old_index;
        if (dirty_.exchange(true, std::memory_order_release)) {
            // If it was already dirty, we just overwrote a frame that was never read
            drop_count_.fetch_add(1, std::memory_order_relaxed);
        }
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
     * @brief Gets the number of frames dropped (overwritten) in this buffer.
     */
    int64_t get_drop_count() const {
        return drop_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Checks if there is a pending (unconsumed) frame in the buffer.
     */
    bool has_pending() const {
        return dirty_.load(std::memory_order_acquire);
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
    std::atomic<int64_t> drop_count_{0};
};

// --- Type aliases for all pipeline queues ---


/// @brief Type alias for a lock-free queue holding ImageData pointers.
typedef LockFreeQueue<ImageData*, 64> ImageQueue;

// Define a type for a pooled buffer of detection results
using DetectionResultBuffer = PooledBuffer<DetectionResult>;

/**
 * @brief lifecycle token for an inference result.
 * 
 * ResultToken acts as a carrier for the shared_ptr to the detection results
 * as it moves through the lock-free queues.
 */
class ResultToken {
public:
    ResultToken() : data_(nullptr) {}

    ResultToken(std::shared_ptr<DetectionResultBuffer> data)
        : data_(std::move(data)) {}

    // Move only
    ResultToken(ResultToken&& other) noexcept 
        : data_(std::move(other.data_)) {}

    ResultToken& operator=(ResultToken&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }

    ResultToken(const ResultToken&) = delete;
    ResultToken& operator=(const ResultToken&) = delete;

    ~ResultToken() = default;

    void mark_consumed() { /* No-op: handled by external accounting */ }
    void mark_dropped() { /* No-op: handled by external accounting */ }

    const std::shared_ptr<DetectionResultBuffer>& get() const { return data_; }
    std::shared_ptr<DetectionResultBuffer>& get() { return data_; }
    bool isValid() const { return data_ != nullptr; }
    
    // Method to release buffer reference (for pooling support)
    void release_buffer() { data_.reset(); }
    
    // Method to explicitly trigger accounting before pooling (No-op now)
    void trigger_accounting() {}

private:
    std::shared_ptr<DetectionResultBuffer> data_;
};

/// @brief Type alias for a thread-safe MPMC queue holding ResultToken pointers.
using DetectionResultsQueue = LockFreeQueue<ResultToken*, 32>;

// Define a type for a pooled buffer of H.264 NAL units
using H264Buffer = PooledBuffer<uint8_t>;
/// @brief Type alias for a thread-safe MPMC queue holding pooled H.264 buffer pointers.
using H264Queue = LockFreeQueue<H264Buffer*, 16>;

#endif // PIPELINE_STRUCTS_H
