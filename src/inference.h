/**
 * @file inference.h
 * @brief Defines the InferenceEngine class for running TensorFlow Lite models
 *        with Edge TPU acceleration.
 *
 * This header provides the interface for the InferenceEngine, which manages
 * the loading, execution, and output processing of TensorFlow Lite models.
 * It integrates with the Edge TPU delegate for hardware acceleration and
 * operates within a multi-threaded pipeline, consuming image data and
 * producing detection results via thread-safe queues.
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h" // Required for BuiltinOpResolver
#include "tensorflow/lite/c/common.h"       // Required for TfLiteDelegate

#include <vector>
#include <string>
#include <memory>   // For std::unique_ptr
#include <thread>   // For std::thread
#include <atomic>   // For std::atomic
#include <mutex>    // For std::mutex
#include <queue>    // For std::queue (used in interpreter_pool_)

#include "pipeline_structs.h" // Use the new central header for queue types and data structures

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
     * @param udp_output_queue Reference to the thread-safe UdpQueue to which
     *                         detection results are pushed.
     * @param num_threads The number of worker threads to spawn for parallel inference.
     */
    InferenceEngine(const std::string& model_path, ImageQueue& input_queue, UdpQueue& udp_output_queue, int num_threads = 1);

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

private:
    /**
     * @brief The main function executed by each inference worker thread.
     *
     * This function continuously processes images from the input queue,
     * performs inference, and pushes results to the output queue.
     */
    void worker_thread_func();

    /**
     * @brief Creates and configures a TensorFlow Lite interpreter with the Edge TPU delegate.
     *
     * This method is called by each worker thread to obtain its dedicated interpreter instance.
     *
     * @return A unique_ptr to a fully initialized interpreter, or nullptr on failure.
     */
    std::unique_ptr<tflite::Interpreter> create_interpreter();

    /**
     * @brief Prepares the input tensor for inference.
     *
     * Copies image data into the interpreter's input tensor and handles
     * necessary preprocessing like color channel swapping (BGR to RGB).
     *
     * @param interpreter Pointer to the TensorFlow Lite interpreter.
     * @param image The ImageData to be processed.
     */
    void set_input_tensor(tflite::Interpreter* interpreter, const ImageData& image);

    /**
     * @brief Parses the output tensors of the interpreter to extract detection results.
     *
     * Assumes a standard output format (e.g., for SSD models) and converts
     * raw tensor data into a vector of DetectionResult objects.
     *
     * @param interpreter Pointer to the TensorFlow Lite interpreter.
     * @return A vector of detected objects.
     */
    std::vector<DetectionResult> get_output_tensor(tflite::Interpreter* interpreter);

    std::string model_path_; ///< Path to the TensorFlow Lite model file.
    ImageQueue& input_queue_; ///< Reference to the input queue for image data.
    UdpQueue& udp_output_queue_; ///< Reference to the output queue for detection results.
    int num_threads_; ///< Number of inference worker threads.

    int input_width_ = 0; ///< Input width required by the loaded model.
    int input_height_ = 0; ///< Input height required by the loaded model.
    int input_channels_ = 0; ///< Input channels required by the loaded model (e.g., 3 for RGB/BGR).

    std::unique_ptr<tflite::FlatBufferModel> model_; ///< The loaded TensorFlow Lite model.
    tflite::ops::builtin::BuiltinOpResolver resolver_; ///< Op resolver for built-in TFLite operations.
    std::vector<std::thread> worker_threads_; ///< Vector of active inference worker threads.
    std::atomic<bool> running_ = false; ///< Atomic flag to control the running state of the inference engine.
    
    // Note: The interpreter_pool_ and interpreter_pool_mutex_ are remnants from a previous
    // design where interpreters were pooled. In the current design, each worker thread
    // creates its own unique_ptr<tflite::Interpreter> instance via create_interpreter().
    // These members are kept for API compatibility with main.cpp but are not actively used
    // in the revised inference logic for sharing interpreters.
    std::mutex interpreter_pool_mutex_; ///< Mutex for protecting access to the interpreter pool (not actively used).
    std::queue<std::unique_ptr<tflite::Interpreter>> interpreter_pool_; ///< Queue for interpreter instances (not actively used in revised design).
};

#endif // INFERENCE_H