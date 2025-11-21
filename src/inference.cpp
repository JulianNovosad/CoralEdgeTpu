#include "inference.h"
#include <iostream>
#include <algorithm> // For std::sort
#include <cmath>     // For std::abs

// Edge TPU delegate C API functions
extern "C" {
    TfLiteDelegate* tflite_plugin_create_delegate(char** options_keys, char** options_values, size_t num_options);
    void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate);
}

InferenceEngine::InferenceEngine(const std::string& model_path, ThreadSafeQueue& input_queue, DetectionQueue& output_queue, int num_threads)
    : model_path_(model_path), input_queue_(input_queue), output_queue_(output_queue), num_threads_(num_threads) {

    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (!model_) {
        std::cerr << "Failed to load model: " << model_path_ << std::endl;
        // Handle error appropriately, e.g., throw exception
    }
}

InferenceEngine::~InferenceEngine() {
    stop();
}

bool InferenceEngine::start() {
    if (running_) {
        std::cerr << "InferenceEngine is already running." << std::endl;
        return false;
    }

    if (!model_) {
        std::cerr << "Model not loaded, cannot start inference engine." << std::endl;
        return false;
    }

    running_ = true;
    for (int i = 0; i < num_threads_; ++i) {
        // Create an interpreter for each thread
        std::unique_ptr<tflite::Interpreter> interpreter = create_interpreter();
        if (!interpreter) {
            std::cerr << "Failed to create interpreter for worker " << i << std::endl;
            running_ = false;
            return false;
        }
        std::lock_guard<std::mutex> lock(interpreter_pool_mutex_);
        interpreter_pool_.push(std::move(interpreter));

        worker_threads_.emplace_back(&InferenceEngine::worker_thread_func, this);
    }

    std::cout << "InferenceEngine started with " << num_threads_ << " worker threads." << std::endl;
    return true;
}

void InferenceEngine::stop() {
    if (!running_) {
        return;
    }
    running_ = false;
    output_queue_.set_running(false); // Signal consumers to stop
    // Also signal the input queue to unblock the inference threads
    input_queue_.set_running(false);
    
    for (std::thread& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    std::cout << "InferenceEngine stopped." << std::endl;
}

std::unique_ptr<tflite::Interpreter> InferenceEngine::create_interpreter() {
    std::unique_ptr<tflite::Interpreter> local_interpreter;
    tflite::InterpreterBuilder(*model_, resolver_)(&local_interpreter);

    if (!local_interpreter) {
        std::cerr << "Failed to create interpreter." << std::endl;
        return nullptr;
    }

    // Attach EdgeTPU delegate
    // Options can be passed as key-value pairs if needed
    char* options_keys[] = {};
    char* options_values[] = {};
    size_t num_options = 0;

    TfLiteDelegate* delegate = tflite_plugin_create_delegate(options_keys, options_values, num_options);
    if (!delegate) {
        std::cerr << "Failed to create EdgeTPU delegate." << std::endl;
        return nullptr;
    }

    if (local_interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        std::cerr << "Failed to apply EdgeTPU delegate." << std::endl;
        tflite_plugin_destroy_delegate(delegate);
        return nullptr;
    }

    if (local_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        tflite_plugin_destroy_delegate(delegate);
        return nullptr;
    }

    return local_interpreter;
}

void InferenceEngine::worker_thread_func() {
    std::unique_ptr<tflite::Interpreter> interpreter;

    // Get an interpreter from the pool
    {
        std::lock_guard<std::mutex> lock(interpreter_pool_mutex_);
        if (!interpreter_pool_.empty()) {
            interpreter = std::move(interpreter_pool_.front());
            interpreter_pool_.pop();
        } else {
            std::cerr << "Interpreter pool is empty for worker thread!" << std::endl;
            return; // Should not happen if pool is sized correctly
        }
    }

    ImageData input_image;
    while (running_) {
        if (input_queue_.pop(input_image)) { // Blocking call
            auto inference_start_time = std::chrono::high_resolution_clock::now();

            set_input_tensor(interpreter.get(), input_image);

            if (interpreter->Invoke() != kTfLiteOk) {
                std::cerr << "Failed to invoke interpreter." << std::endl;
                continue;
            }

            std::vector<DetectionResult> results = get_output_tensor(interpreter.get(), input_image);
            output_queue_.push(std::move(results));

            auto inference_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> inference_duration = inference_end_time - inference_start_time;
            // std::cout << "Inference time: " << inference_duration.count() << " ms" << std::endl;
        }
    }

    // Return interpreter to pool (or destroy it, if we're shutting down the pool)
    // For now, it will be destroyed when unique_ptr goes out of scope.
    // If we wanted to re-pool it, we'd push it back to a shared queue before exiting.
}

void InferenceEngine::set_input_tensor(tflite::Interpreter* interpreter, const ImageData& image) {
    // Assuming the model expects a single input tensor of type kTfLiteUInt8
    // and layout (1, height, width, channels)
    // Here we need to convert the YUV420 image to RGB or grayscale as expected by the model.
    // For now, we'll assume the model expects a single-channel grayscale image
    // and copy the Y plane directly, or if it expects RGB, we'd need conversion.

    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);

    if (input_tensor->type != kTfLiteUInt8) {
        std::cerr << "Input tensor type is not kTfLiteUInt8. Model might expect a different format." << std::endl;
        // Handle error, maybe convert type
    }

    if (input_tensor->bytes != image.data.size()) {
        std::cerr << "WARNING: Input tensor size mismatch. Expected " << input_tensor->bytes
                  << ", got " << image.data.size() << ". This will likely lead to incorrect inference results." << std::endl;
        std::cerr << "         Image resizing needs to be implemented." << std::endl;
    }

    // Copy image data to input tensor
    // This will likely cause a buffer overflow if the image is larger than the input tensor.
    // For now, we'll just copy the first `input_tensor->bytes` bytes.
    std::copy(image.data.begin(), image.data.begin() + input_tensor->bytes, interpreter->typed_input_tensor<uint8_t>(0));
}

std::vector<DetectionResult> InferenceEngine::get_output_tensor(tflite::Interpreter* interpreter, const ImageData& image) {
    std::vector<DetectionResult> results;

    // Assuming a common detection model output format:
    // Output 0: Detection boxes (1, num_boxes, 4) - [ymin, xmin, ymax, xmax]
    // Output 1: Detection classes (1, num_boxes)
    // Output 2: Detection scores (1, num_boxes)
    // Output 3: Number of detections (1)

    int num_detections = static_cast<int>(*interpreter->typed_output_tensor<float>(3));
    float* detection_boxes = interpreter->typed_output_tensor<float>(0);
    float* detection_classes = interpreter->typed_output_tensor<float>(1);
    float* detection_scores = interpreter->typed_output_tensor<float>(2);

    for (int i = 0; i < num_detections; ++i) {
        if (detection_scores[i] > 0.5) { // Threshold for detections
            DetectionResult res;
            res.class_id = static_cast<int>(detection_classes[i]);
            res.score = detection_scores[i];

            // Bounding box coordinates are normalized [0, 1]
            // Convert to absolute pixel coordinates
            res.ymin = detection_boxes[i * 4] * image.height;
            res.xmin = detection_boxes[i * 4 + 1] * image.width;
            res.ymax = detection_boxes[i * 4 + 2] * image.height;
            res.xmax = detection_boxes[i * 4 + 3] * image.width;
            res.timestamp = image.timestamp;
            results.push_back(res);
        }
    }
    return results;
}
