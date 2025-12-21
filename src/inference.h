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
#include <queue>    // For std::queue (used in interpreter_pool_)

#include "pipeline_structs.h" // Use the new central header for queue types and data structures
#include "buffer_pool.h"      // For BufferPool

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
     * @param detection_results_output_queue Reference to the thread-safe DetectionResultsQueue to which
     *                                       detection results are pushed.
     * @param detection_result_pool Reference to the buffer pool for detection results.
     * @param num_threads The number of worker threads to spawn for parallel inference.
     */
    InferenceEngine(const std::string& model_path, 
                    ImageQueue& input_queue, 
                    DetectionResultsQueue& detection_results_for_overlay_queue, 
                    DetectionResultsQueue& detection_results_for_logic_queue, 
                    std::shared_ptr<BufferPool<DetectionResult>> detection_result_pool,
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

private:
    void worker_thread_func();
    std::unique_ptr<tflite::Interpreter> create_interpreter();
    void set_input_tensor(tflite::Interpreter* interpreter, const ImageData& image);
    std::shared_ptr<DetectionResultBuffer> get_output_tensor(tflite::Interpreter* interpreter);
    float get_tpu_temperature();

    std::string model_path_; ///< Path to the TensorFlow Lite model file.
    ImageQueue& input_queue_; ///< Reference to the input queue for image data.
    DetectionResultsQueue& detection_results_for_overlay_queue_; ///< Reference to the output queue for detection results to overlay.
    DetectionResultsQueue& detection_results_for_logic_queue_; ///< Reference to the output queue for detection results to the logic module.
    std::shared_ptr<BufferPool<DetectionResult>> detection_result_pool_; ///< Pool for detection result buffers.
    int num_threads_; ///< Number of inference worker threads.
    float score_threshold_; ///< Confidence threshold for filtering detections.

    int input_width_ = 0; ///< Input width required by the loaded model.
    int input_height_ = 0; ///< Input height required by the loaded model.
    int input_channels_ = 0; ///< Input channels required by the loaded model (e.g., 3 for RGB/BGR).

    std::unique_ptr<tflite::FlatBufferModel> model_; ///< The loaded TensorFlow Lite model.
    tflite::ops::builtin::BuiltinOpResolver resolver_; ///< Op resolver for built-in TFLite operations.
    std::vector<std::thread> worker_threads_; ///< Vector of active inference worker threads.
    std::atomic<bool> running_ = false; ///< Atomic flag to control the running state of the inference engine.
    
public:
    // Freshness indicators
    std::atomic<long long> last_inference_timestamp_{0}; ///< Timestamp of the last inference
    std::atomic<int> inference_rate_{0}; ///< Current inference rate
    
private:
    TfLiteDelegate* edgetpu_delegate_ = nullptr; ///< The single Edge TPU delegate.
};

#endif // INFERENCE_H