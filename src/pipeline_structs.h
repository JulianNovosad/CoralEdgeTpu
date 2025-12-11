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
    BufferPool<uint8_t>::PooledPtr buffer; ///< Shared pointer to a pooled buffer holding pixel data.
    size_t width;                                  ///< Width of the image in pixels.
    size_t height;                                 ///< Height of the image in pixels.
    libcamera::PixelFormat format;                 ///< Pixel format of the image data.
    long long timestamp_epoch_ms = 0; ///< Timestamp of image capture in epoch milliseconds UTC.
    int frame_id = -1;                             ///< Monotonically increasing frame ID.

    // Constructor to allow explicit setting of timestamp and frame_id
    ImageData(long long ts_ms = 0, int id = -1) : timestamp_epoch_ms(ts_ms), frame_id(id) {}
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

// --- Type aliases for all pipeline queues ---

/// @brief Type alias for a lock-free SPSC queue holding ImageData objects.
using ImageQueue = boost::lockfree::spsc_queue<ImageData, boost::lockfree::capacity<100>>;

// Define a type for a pooled buffer of detection results
using DetectionResultBuffer = PooledBuffer<DetectionResult>;
/// @brief Type alias for a lock-free SPSC queue holding shared pointers to pooled detection result buffers.
using DetectionResultsQueue = boost::lockfree::spsc_queue<std::shared_ptr<DetectionResultBuffer>, boost::lockfree::capacity<100>>;

// Define a type for a pooled buffer of H.264 NAL units
using H264Buffer = PooledBuffer<uint8_t>;
/// @brief Type alias for a lock-free SPSC queue holding shared pointers to pooled H.264 buffers.
using H264Queue = boost::lockfree::spsc_queue<std::shared_ptr<H264Buffer>, boost::lockfree::capacity<100>>;

#endif // PIPELINE_STRUCTS_H
