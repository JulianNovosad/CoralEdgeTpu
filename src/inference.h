#ifndef INFERENCE_H
#define INFERENCE_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h" // Required for BuiltinOpResolver
#include "tensorflow/lite/c/common.h"       // Required for TfLiteDelegate
#include "edgetpu_c.h"                      // Required for Edge TPU delegate functions

#include <vector>
#include <string>
#include <memory>   // For std::unique_ptr
#include <thread>   // For std::thread
#include <atomic>   // For std::atomic
#include <mutex>    // For std::mutex
#include <shared_mutex> // For std::shared_mutex
#include <queue>    // For std::queue (used in interpreter_pool_)

#include "pipeline_structs.h" // Use the new central header for queue types and data structures
#include "buffer_pool.h"      // For BufferPool
#include "timing.h"           // Authoritative timing

// WORKAROUND FOR EDGE TPU DELEGATE BUG:
// There appears to be a memory corruption issue in the Edge TPU delegate where it 
// occasionally fails with "Node number X (EdgeTpuDelegateForCustomOp) failed to invoke".
// This is likely caused by resource leaks or state corruption in the delegate.
// To mitigate this issue, we implement aggressive delegate recreation strategies:
// 1. Retry interpreter creation multiple times when there's an invoke error
// 2. Recreate the delegate more frequently (every 50 inferences instead of 100)
// 3. Add detailed error logging to help diagnose issues

/**
 * @brief Manages the TensorFlow Lite inference pipeline with Edge TPU acceleration.
 *
 * The InferenceEngine class loads a TFLite model, initializes multiple
 * interpreter instances (one per worker thread) with the Edge TPU delegate,
 * and processes incoming ImageData from a queue. It performs inference and
 * pushes detected objects (DetectionResult) to an output queue.
 */
class InferenceEngine {
public:
    /**
     * @brief Constructor for InferenceEngine.
     *
     * Initializes the inference engine, loads the model, and prepares internal
     * structures. It will throw `std::runtime_error` if model loading or
     * initial interpreter/delegate setup fails.
     *
     * @param model_path The filesystem path to the TensorFlow Lite model file.
     * @param input_queue Reference to the thread-safe ImageQueue from which
     *                    raw image data frames are consumed.
     * @param detection_results_for_overlay_buffer Pointer to the TripleBuffer for detection results to overlay.
     * @param detection_results_for_logic_queue Reference to the thread-safe DetectionResultsQueue to which
     *                                       detection results are pushed for logic.
     * @param detection_result_pool Reference to the buffer pool for detection results.
     * @param num_threads The number of worker threads to spawn for parallel inference.
     */
    InferenceEngine(const std::string& model_path, 
                    ImageQueue& input_queue, 
                    TripleBuffer<DetectionResults>* detection_results_for_overlay_buffer, 
                    DetectionResultsQueue& detection_results_for_logic_queue, 
                    std::shared_ptr<BufferPool<DetectionResult>> detection_result_pool,
                    std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                    std::shared_ptr<ObjectPool<ResultToken>> result_token_pool,
                    float score_threshold,
                    int num_threads = 1);

    /**
     * @brief Destructor for InferenceEngine.
     *
     * Ensures all worker threads are stopped gracefully and resources are released.
     */
    ~InferenceEngine();

    /**
     * @brief Starts the inference worker threads.
     *
     * Launches the configured number of worker threads, each running its
     * independent inference loop.
     *
     * @return True if the engine started successfully, false otherwise.
     */
    bool start();

    /**
     * @brief Stops the inference worker threads.
     *
     * Signals all worker threads to terminate and waits for them to join.
     */
    void stop();

    /**
     * @brief Checks if the inference engine is currently running.
     *
     * @return True if the engine is running, false otherwise.
     */
    bool is_running() const { return running_; }

    /**
     * @brief Retrieves the input width expected by the loaded TensorFlow Lite model.
     *
     * @return The width in pixels.
     */
    int get_input_width() const { return input_width_; }

    /**
     * @brief Retrieves the input height expected by the loaded TensorFlow Lite model.
     *
     * @return The height in pixels.
     */
    int get_input_height() const { return input_height_; }
    void get_state() const;
    
    // Timing methods for monitoring
    long long get_inference_timing_us() const { return avg_inference_time_us_; }

private:
    void worker_thread_func();
    std::unique_ptr<tflite::Interpreter> create_interpreter();
    void recreate_delegate(); // Helper to safely recreate the Edge TPU delegate
    void set_input_tensor(tflite::Interpreter* interpreter, const ImageData& image);
    std::shared_ptr<DetectionResultBuffer> get_output_tensor(tflite::Interpreter* interpreter, const ImageData& input_image);
    float get_tpu_temperature();

    std::string model_path_; ///< Path to the TensorFlow Lite model file.
    ImageQueue& input_queue_; ///< Reference to the input queue for image data.
    TripleBuffer<DetectionResults>* detection_results_for_overlay_buffer_; ///< Pointer to the triple buffer for detection results to overlay.
    DetectionResultsQueue& detection_results_for_logic_queue_; ///< Reference to the output queue for detection results to the logic module.
    std::shared_ptr<BufferPool<DetectionResult>> detection_result_pool_; ///< Pool for detection result buffers.
    std::shared_ptr<ObjectPool<ImageData>> image_data_pool_; ///< Pool for ImageData objects.
    std::shared_ptr<ObjectPool<ResultToken>> result_token_pool_; ///< Pool for ResultToken objects.
    int num_threads_; ///< Number of inference worker threads.
    float score_threshold_; ///< Confidence threshold for filtering detections.

    int input_width_ = 0; ///< Input width required by the loaded model.
    int input_height_ = 0; ///< Input height required by the loaded model.
    int input_channels_ = 0; ///< Input channels required by the loaded model (e.g., 3 for RGB/BGR).

    std::unique_ptr<tflite::FlatBufferModel> model_; ///< The loaded TensorFlow Lite model.
    tflite::ops::builtin::BuiltinOpResolver resolver_; ///< Op resolver for built-in TFLite operations.
    std::vector<std::thread> worker_threads_; ///< Vector of active inference worker threads.
    std::atomic<bool> running_ = false; ///< Atomic flag to control the running state of the inference engine.
    mutable std::shared_mutex delegate_mutex_; ///< Mutex for safe delegate recreation.
    
public:
    // Freshness indicators
    std::atomic<long long> last_inference_timestamp_{0}; ///< Timestamp of the last inference
    std::atomic<int> inference_rate_{0}; ///< Current inference rate
    std::atomic<float> tpu_temperature_{0.0f}; ///< Current TPU temperature
    
    // Timing statistics
    mutable std::atomic<long long> avg_inference_time_us_{0}; ///< Average inference time in microseconds
    
    // Public getters for drop counters to be used by Monitor
    int64_t get_overlay_queue_drop_count() const { 
        if (detection_results_for_overlay_buffer_) {
            return detection_results_for_overlay_buffer_->get_drop_count();
        }
        return overlay_queue_drop_count_.load(); 
    }
    int64_t get_logic_queue_drop_count() const { return logic_queue_drop_count_.load(); }
    
    bool has_overlay_pending() const {
        if (detection_results_for_overlay_buffer_) {
            return detection_results_for_overlay_buffer_->has_pending();
        }
        return false;
    }
    
    // Public increment methods for drop counters to be used when draining queues
    void increment_logic_queue_drop_count() { logic_queue_drop_count_.fetch_add(1); }
    void increment_overlay_queue_drop_count() { overlay_queue_drop_count_.fetch_add(1); }
    
    // Public getters for frame accounting counters
    int64_t get_frames_consumed() const { return frames_consumed_.load(); }
    int64_t get_results_produced() const { return results_produced_.load(); }
    int64_t get_results_consumed_by_logic() const { return results_consumed_by_logic_.load(); }
    int64_t get_results_consumed_by_overlay() const { return results_consumed_by_overlay_.load(); }
    
    // Method to set application reference for updating counters
    void set_application_ref(class Application* app) { app_ref_ = app; }
    
private:
    // Drop counters for proper queue accounting
    std::atomic<int64_t> overlay_queue_drop_count_{0}; ///< Count of detection results dropped from overlay queue
    std::atomic<int64_t> logic_queue_drop_count_{0};   ///< Count of detection results dropped from logic queue
    
    // Frame accounting counters
    std::atomic<int64_t> frames_consumed_{0};           ///< Count of frames consumed from input queue
    std::atomic<int64_t> results_produced_{0};          ///< Count of detection results produced
    std::atomic<int64_t> results_consumed_by_logic_{0};  ///< Count of results consumed by logic module
    std::atomic<int64_t> results_consumed_by_overlay_{0}; ///< Count of results consumed by overlay module
    
    // Application reference for updating counters
    class Application* app_ref_ = nullptr;
    
    TfLiteDelegate* edgetpu_delegate_ = nullptr; ///< The single Edge TPU delegate.

    // Telemetry state
    std::atomic<int> total_inference_count_{0};
    std::atomic<int> last_inference_count_checkpoint_{0};
    std::atomic<uint64_t> last_rate_check_ms_{0};
};

#endif // INFERENCE_H