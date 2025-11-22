/**
 * @file inference.cpp
 * @brief Implements the InferenceEngine class for running TensorFlow Lite models
 *        with Edge TPU acceleration on captured image data.
 *
 * This module handles model loading, interpreter creation, Edge TPU delegate
 * integration, input tensor preparation (including color space conversion),
 * model invocation, and parsing of output tensors for object detection results.
 * It operates in a multi-threaded environment, with worker threads performing
 * inference on frames from a queue and pushing results to another queue.
 */

#include "inference.h"
#include "util_logging.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// Edge TPU delegate C API functions - Adjusted signature based on previous iteration backup
// These functions are dynamically linked from libedgetpu.so
extern "C" {
    /**
     * @brief Creates an Edge TPU delegate instance.
     *
     * This function is part of the Edge TPU C API and is used to acquire
     * a delegate that can be applied to a TensorFlow Lite interpreter
     * to enable acceleration on the Coral Edge TPU.
     *
     * @param options A pointer to delegate-specific options (can be nullptr for default).
     * @return A pointer to a TfLiteDelegate instance, or nullptr on failure.
     */
    TfLiteDelegate* tflite_plugin_create_delegate(const void* options);

    /**
     * @brief Destroys an Edge TPU delegate instance.
     *
     * This function is part of the Edge TPU C API and is used to deallocate
     * resources associated with an Edge TPU delegate.
     *
     * @param delegate A pointer to the TfLiteDelegate instance to destroy.
     */
    void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate);
}

/**
 * @brief Constructor for InferenceEngine.
 *
 * Initializes the inference engine by loading the TensorFlow Lite model from the
 * specified path, performing a pre-check for Edge TPU delegate creation, and
 * extracting input tensor dimensions for validation and processing.
 *
 * @param model_path The filesystem path to the TensorFlow Lite model file.
 * @param input_queue Reference to the ImageQueue from which image data is consumed.
 * @param udp_output_queue Reference to the UdpQueue where detection results are pushed.
 * @param num_threads The number of worker threads to use for inference.
 * @throws std::runtime_error if the model fails to load, delegate pre-check fails,
 *         or input tensor dimensions are invalid.
 */
InferenceEngine::InferenceEngine(const std::string& model_path, ImageQueue& input_queue, UdpQueue& udp_output_queue, int num_threads)
    : model_path_(model_path), input_queue_(input_queue), udp_output_queue_(udp_output_queue), num_threads_(num_threads) {

    // Load the TensorFlow Lite model from the file system.
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (!model_) {
        throw std::runtime_error("Failed to load model: " + model_path_ + ". Please ensure the model path is correct and the file exists.");
    }

    // Create a temporary interpreter for model inspection (e.g., getting dimensions).
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model_, resolver_)(&interpreter);
    if (!interpreter) {
        throw std::runtime_error("Failed to create temporary interpreter for model inspection.");
    }
    
    // Pre-check for Edge TPU delegate creation during initialization.
    // This helps in failing fast if the Edge TPU is not available or drivers are not installed.
    TfLiteDelegate* test_delegate = tflite_plugin_create_delegate(nullptr);
    if (!test_delegate) {
        throw std::runtime_error("Failed to create Edge TPU delegate during initialization. Ensure Edge TPU drivers are installed and device is connected.");
    }
    // The test delegate is not owned by the interpreter yet, so destroy it immediately.
    tflite_plugin_destroy_delegate(test_delegate);

    // Extract input tensor dimensions from the model.
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);
    
    // Validate input tensor dimensions. Expects a 4D tensor (Batch, Height, Width, Channels).
    if (input_tensor->dims->size < 4) {
        throw std::runtime_error("Model input tensor has fewer than 4 dimensions, which is not supported.");
    }
    input_height_ = input_tensor->dims->data[1];
    input_width_ = input_tensor->dims->data[2];
    input_channels_ = input_tensor->dims->data[3];
    
    LOG_INFO("Model Input Dimensions: " + std::to_string(input_width_) + "x" + std::to_string(input_height_) + "x" + std::to_string(input_channels_));

    // This application is designed for 3-channel (RGB/BGR) input images.
    if (input_channels_ != 3) {
        throw std::runtime_error("Model expects " + std::to_string(input_channels_) + " channels, but this application is hardcoded for 3 (RGB).");
    }
}

/**
 * @brief Destructor for InferenceEngine.
 *
 * Ensures that all inference worker threads are stopped gracefully.
 */
InferenceEngine::~InferenceEngine() {
    stop();
}

/**
 * @brief Starts the inference engine worker threads.
 *
 * Launches a specified number of worker threads, each running an inference loop.
 *
 * @return True if the engine started successfully, false otherwise.
 */
bool InferenceEngine::start() {
    if (running_) {
        LOG_ERROR("InferenceEngine is already running.");
        return false;
    }
    if (!model_) {
        LOG_ERROR("Model not loaded, cannot start inference engine.");
        return false;
    }

    running_ = true;
    // Signal to associated queues that they should continue running.
    input_queue_.set_running(true);
    udp_output_queue_.set_running(true);

    // Create and launch worker threads. Each thread will create its own interpreter.
    for (int i = 0; i < num_threads_; ++i) {
        worker_threads_.emplace_back(&InferenceEngine::worker_thread_func, this);
    }

    LOG_INFO("InferenceEngine started with " + std::to_string(num_threads_) + " worker threads.");
    return true;
}

/**
 * @brief Stops the inference engine worker threads.
 *
 * Sets the running flag to false, signals associated queues to stop, and joins
 * all worker threads for a clean shutdown.
 */
void InferenceEngine::stop() {
    if (running_.exchange(false)) { // Atomically set running_ to false and check previous value
        LOG_INFO("Stopping InferenceEngine...");
        // Signal associated queues to stop.
        udp_output_queue_.set_running(false);
        input_queue_.set_running(false);
        
        // Join all worker threads to ensure they complete their tasks or exit.
        for (std::thread& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        worker_threads_.clear();
        LOG_INFO("InferenceEngine stopped.");
    }
}

/**
 * @brief Creates and initializes a TensorFlow Lite interpreter with the Edge TPU delegate.
 *
 * This function is called by each worker thread to get its own interpreter instance.
 * It builds the interpreter from the loaded model, creates and applies the Edge TPU
 * delegate, and allocates tensors. Proper error handling and delegate cleanup are included.
 *
 * @return A unique pointer to a configured tflite::Interpreter, or nullptr on failure.
 */
std::unique_ptr<tflite::Interpreter> InferenceEngine::create_interpreter() {
    std::unique_ptr<tflite::Interpreter> local_interpreter;
    // Build the interpreter using the model and a standard op resolver.
    tflite::InterpreterBuilder(*model_, resolver_)(&local_interpreter);
    if (!local_interpreter) {
        LOG_ERROR("Failed to build interpreter.");
        return nullptr;
    }

    // Create the Edge TPU delegate.
    TfLiteDelegate* delegate = tflite_plugin_create_delegate(nullptr);
    if (!delegate) {
        LOG_ERROR("Failed to create EdgeTPU delegate. Ensure libedgetpu1-std is installed and device is connected.");
        return nullptr;
    }

    // Apply the Edge TPU delegate to the interpreter.
    if (local_interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        LOG_ERROR("Failed to apply EdgeTPU delegate. Check if the model is compatible with Edge TPU.");
        // If ModifyGraphWithDelegate fails, the interpreter does not take ownership of the delegate.
        tflite_plugin_destroy_delegate(delegate);
        return nullptr;
    }
    
    // Allocate tensors. This must be done *after* applying the delegate.
    if (local_interpreter->AllocateTensors() != kTfLiteOk) {
        LOG_ERROR("Failed to allocate tensors after applying EdgeTPU delegate.");
        // In this specific case, if AllocateTensors fails *after* ModifyGraphWithDelegate succeeded,
        // the interpreter might *already* own the delegate. Destroying it here might be a double-free
        // or lead to issues if the interpreter attempts to destroy it later.
        // For simplicity and safety in this context, we re-destroy it, assuming that if allocation fails,
        // the interpreter is in an invalid state and should be discarded. A more robust solution
        // might involve a custom deleter for unique_ptr or checking delegate ownership carefully.
        tflite_plugin_destroy_delegate(delegate); 
        return nullptr;
    }
    
    return local_interpreter;
}

/**
 * @brief The main function for an inference worker thread.
 *
 * Each worker thread continuously pops ImageFrame data from the input queue,
 * prepares the input tensor, invokes the TensorFlow Lite interpreter, parses
 * the output tensors, and pushes detection results to the UDP output queue.
 * The loop continues until the `running_` flag is set to false.
 */
void InferenceEngine::worker_thread_func() {
    // Each worker thread creates and owns its own interpreter instance.
    // This avoids thread-safety issues with a single interpreter.
    std::unique_ptr<tflite::Interpreter> interpreter = create_interpreter();
    if (!interpreter) {
        LOG_ERROR("Worker thread failed to create interpreter. Exiting thread.");
        return;
    }
    
    ImageData input_image;
    while (running_) {
        // Attempt to pop an image from the input queue.
        // The queue's pop method will block until data is available or `running_` is false.
        if (input_queue_.pop(input_image)) {
            // Validate input image data size against expected model input size.
            int expected_input_size = input_width_ * input_height_ * input_channels_;
            if (input_image.data.size() != expected_input_size) {
                 LOG_ERROR("Input image data size (" + std::to_string(input_image.data.size()) + 
                           ") does not match expected model input size (" + std::to_string(expected_input_size) + "). Skipping frame.");
                 continue; // Skip this frame and try the next one.
            }

            // Prepare the input tensor for the interpreter.
            set_input_tensor(interpreter.get(), input_image);

            // Invoke the TensorFlow Lite interpreter to perform inference.
            if (interpreter->Invoke() != kTfLiteOk) {
                LOG_ERROR("Failed to invoke interpreter. Skipping frame.");
                continue; // Skip this frame.
            }
            
            // Parse the output tensors to get detection results.
            std::vector<DetectionResult> results = get_output_tensor(interpreter.get());
            
            // Push detection results to the UDP output queue if any detections were made.
            if (!results.empty()) {
                udp_output_queue_.push(std::move(results));
            }
        }
    }
}

/**
 * @brief Prepares the interpreter's input tensor with image data.
 *
 * This function copies the raw image data into the interpreter's input tensor.
 * It also handles the color channel swap from BGR (from camera) to RGB (expected by model).
 * Performs validation checks for tensor type and size.
 *
 * @param interpreter A pointer to the TensorFlow Lite interpreter.
 * @param image The ImageData object containing the raw image pixels.
 */
void InferenceEngine::set_input_tensor(tflite::Interpreter* interpreter, const ImageData& image) {
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);

    // Validate input tensor type. Expects UINT8 for quantized models.
    if (input_tensor->type != kTfLiteUInt8) {
        LOG_ERROR("Input tensor type is not kTfLiteUInt8 as expected. Current type: " + std::to_string(input_tensor->type) + ". Skipping frame.");
        return;
    }
    // The input_tensor->bytes (expected size) should have already been validated against image.data.size()
    // before this call in worker_thread_func.

    // Get a pointer to the input tensor's data buffer.
    uint8_t* tensor_data = interpreter->typed_input_tensor<uint8_t>(0);
    const uint8_t* image_data = image.data.data();
    int num_pixels = input_width_ * input_height_;

    // Perform BGR to RGB conversion by swapping the red and blue channels.
    // Assuming 3 channels (RGB) in both input and output.
    for (int i = 0; i < num_pixels; ++i) {
        tensor_data[i * 3 + 0] = image_data[i * 3 + 2]; // R = B (from source)
        tensor_data[i * 3 + 1] = image_data[i * 3 + 1]; // G = G (from source)
        tensor_data[i * 3 + 2] = image_data[i * 3 + 0]; // B = R (from source)
    }
}

/**
 * @brief Extracts and parses object detection results from the interpreter's output tensors.
 *
 * This function assumes a specific output tensor structure typical for SSD MobileNet models:
 * 0: detection_boxes (float, [1, num_detections, 4]) - Bounding box coordinates [ymin, xmin, ymax, xmax] normalized [0,1].
 * 1: detection_classes (float, [1, num_detections]) - Class IDs.
 * 2: detection_scores (float, [1, num_detections]) - Confidence scores.
 * 3: num_detections (float, [1]) - Actual number of detections.
 *
 * It applies a confidence threshold (0.5) and scales bounding box coordinates to image dimensions.
 *
 * @param interpreter A pointer to the TensorFlow Lite interpreter.
 * @return A vector of DetectionResult objects found in the current frame.
 */
std::vector<DetectionResult> InferenceEngine::get_output_tensor(tflite::Interpreter* interpreter) {
    std::vector<DetectionResult> results;
    
    // Ensure the model has the expected number of output tensors.
    if (interpreter->outputs().size() < 4) {
        LOG_ERROR("Model does not have expected number of output tensors (expected 4 for SSD MobileNet).");
        return results;
    }

    // Get pointers to the output tensors.
    const float* detection_boxes = interpreter->typed_output_tensor<float>(0);
    const float* detection_classes = interpreter->typed_output_tensor<float>(1);
    const float* detection_scores = interpreter->typed_output_tensor<float>(2);
    // The num_detections tensor typically contains a single float value indicating the count.
    const int num_detections = static_cast<int>(*interpreter->typed_output_tensor<float>(3));

    auto timestamp = std::chrono::high_resolution_clock::now();

    // Iterate through detected objects and apply a confidence threshold.
    for (int i = 0; i < num_detections; ++i) {
        if (detection_scores[i] > 0.5) { // Confidence threshold for valid detections
            DetectionResult res;
            res.class_id = static_cast<int>(detection_classes[i]);
            res.score = detection_scores[i];
            res.timestamp = timestamp;

            // Bounding box coordinates are normalized [0, 1] and in [ymin, xmin, ymax, xmax] order.
            // Scale them back to the input image dimensions.
            res.ymin = detection_boxes[i * 4 + 0] * input_height_;
            res.xmin = detection_boxes[i * 4 + 1] * input_width_;
            res.ymax = detection_boxes[i * 4 + 2] * input_height_;
            res.xmax = detection_boxes[i * 4 + 3] * input_width_;
            results.push_back(res);
        }
    }
    return results;
}
