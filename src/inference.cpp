#include "inference.h"
#include "util_logging.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <filesystem>

// Edge TPU delegate C API functions
extern "C" {
    TfLiteDelegate* tflite_plugin_create_delegate(char** options_keys,
                                                  char** options_values,
                                                  size_t num_options,
                                                  void (*report_error)(const char*));
    void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate);
}

// Custom error reporting function for the Edge TPU delegate
void edgetpu_error_reporter(const char* msg) {
    LOG_ERROR("Edge TPU Delegate: " + std::string(msg));
}

InferenceEngine::InferenceEngine(const std::string& model_path, 
                                     ImageQueue& input_queue, 
                                     DetectionResultsQueue& detection_results_for_overlay_queue, 
                                     DetectionResultsQueue& detection_results_for_logic_queue, 
                                     std::shared_ptr<BufferPool<DetectionResult>> detection_result_pool,
                                     float score_threshold,
                                     int num_threads)
    : model_path_(model_path), 
      input_queue_(input_queue), 
      detection_results_for_overlay_queue_(detection_results_for_overlay_queue), 
      detection_results_for_logic_queue_(detection_results_for_logic_queue),
      detection_result_pool_(detection_result_pool),
      num_threads_(num_threads),
      score_threshold_(score_threshold) {

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
    
    // Extract input tensor dimensions from the model.
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);
    
    if (input_tensor->dims->size < 4) {
        throw std::runtime_error("Model input tensor has fewer than 4 dimensions, which is not supported.");
    }
    input_height_ = input_tensor->dims->data[1];
    input_width_ = input_tensor->dims->data[2];
    input_channels_ = input_tensor->dims->data[3];
    
    LOG_INFO("Model Input Dimensions: " + std::to_string(input_width_) + "x" + std::to_string(input_height_) + "x" + std::to_string(input_channels_));

    if (input_channels_ != 3) {
        throw std::runtime_error("Model expects " + std::to_string(input_channels_) + " channels, but this application is hardcoded for 3 (RGB).");
    }
}

InferenceEngine::~InferenceEngine() {
    stop();
}

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

    for (int i = 0; i < num_threads_; ++i) {
        worker_threads_.emplace_back(&InferenceEngine::worker_thread_func, this);
    }

    LOG_INFO("InferenceEngine started with " + std::to_string(num_threads_) + " worker threads.");
    return true;
}

void InferenceEngine::stop() {
    if (running_.exchange(false)) {
        LOG_INFO("Stopping InferenceEngine...");

        
        for (std::thread& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        worker_threads_.clear();
        LOG_INFO("InferenceEngine stopped.");
    }
}

std::unique_ptr<tflite::Interpreter> InferenceEngine::create_interpreter() {
    std::unique_ptr<tflite::Interpreter> local_interpreter;
    tflite::InterpreterBuilder(*model_, resolver_)(&local_interpreter);
    if (!local_interpreter) {
        LOG_ERROR("Failed to build interpreter.");
        return nullptr;
    }

    TfLiteDelegate* delegate = tflite_plugin_create_delegate(nullptr, nullptr, 0, edgetpu_error_reporter);
    if (!delegate) {
        LOG_ERROR("Failed to create EdgeTPU delegate. Ensure libedgetpu1-std is installed and device is connected.");
        return nullptr;
    }

    if (local_interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        LOG_ERROR("Failed to apply EdgeTPU delegate. Check if the model is compatible with Edge TPU.");
        tflite_plugin_destroy_delegate(delegate);
        return nullptr;
    }
    
    if (local_interpreter->AllocateTensors() != kTfLiteOk) {
        LOG_ERROR("Failed to allocate tensors after applying EdgeTPU delegate.");
        tflite_plugin_destroy_delegate(delegate); 
        return nullptr;
    }
    
    return local_interpreter;
}

void InferenceEngine::worker_thread_func() {
    std::unique_ptr<tflite::Interpreter> interpreter = create_interpreter();
    if (!interpreter) {
        LOG_ERROR("Worker thread failed to create interpreter. Exiting thread.");
        return;
    }
    
    ImageData input_image;
    while (running_) {
        if (input_queue_.pop(input_image)) {
            if (!input_image.buffer) {
                LOG_ERROR("InferenceEngine received an image with no buffer. Skipping.");
                continue;
            }
            // LOG_INFO("InferenceEngine received image - Dimensions: " + std::to_string(input_image.width) + "x" + std::to_string(input_image.height) + ", Data size: " + std::to_string(input_image.buffer->size));
            
            int expected_input_size = input_width_ * input_height_ * input_channels_;
            if (input_image.buffer->size != expected_input_size) {
                 LOG_ERROR("Input RGB image data size (" + std::to_string(input_image.buffer->size) + 
                           ") does not match expected model input size (" + std::to_string(expected_input_size) + "). Skipping frame.");
                 continue;
            }

            set_input_tensor(interpreter.get(), input_image);

            auto inference_start_time = std::chrono::high_resolution_clock::now();
            if (interpreter->Invoke() != kTfLiteOk) {
                LOG_ERROR("Failed to invoke interpreter. Skipping frame.");
                continue;
            }
            auto inference_end_time = std::chrono::high_resolution_clock::now();
            long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end_time - inference_start_time).count();

            {
                std::lock_guard<std::mutex> lock(inference_times_mutex_);
                inference_times_ms_.push_back(duration_ms);
                total_inferences_++;
            }
            
            auto results_buffer = get_output_tensor(interpreter.get());
            
            if (results_buffer && results_buffer->size > 0) {
                detection_results_for_overlay_queue_.push(results_buffer); // Push to overlay queue
                detection_results_for_logic_queue_.push(results_buffer); // Push to logic queue
            }
        }
    }
}

void InferenceEngine::set_input_tensor(tflite::Interpreter* interpreter, const ImageData& image) {
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);

    if (input_tensor->type != kTfLiteUInt8) {
        LOG_ERROR("Input tensor type is not kTfLiteUInt8 as expected. Current type: " + std::to_string(input_tensor->type) + ". Skipping frame.");
        return;
    }

    uint8_t* tensor_data = interpreter->typed_input_tensor<uint8_t>(0);
    memcpy(tensor_data, image.buffer->data.data(), image.buffer->size);
}

std::shared_ptr<DetectionResultBuffer> InferenceEngine::get_output_tensor(tflite::Interpreter* interpreter) {
    auto results_buffer = detection_result_pool_->acquire();
    if (!results_buffer) {
        LOG_WARNING("Failed to acquire a detection result buffer from the pool. No results will be reported for this frame.");
        return nullptr;
    }
    results_buffer->size = 0; // Reset size

    if (interpreter->outputs().size() < 4) {
        LOG_ERROR("Model does not have expected number of output tensors (expected 4 for SSD MobileNet).");
        return nullptr;
    }

    const float* detection_boxes = interpreter->typed_output_tensor<float>(0);
    const float* detection_classes = interpreter->typed_output_tensor<float>(1);
    const float* detection_scores = interpreter->typed_output_tensor<float>(2);
    const int num_detections = static_cast<int>(*interpreter->typed_output_tensor<float>(3));

    auto timestamp = std::chrono::high_resolution_clock::now();
    size_t result_count = 0;

    for (int i = 0; i < num_detections; ++i) {
        if (detection_scores[i] > score_threshold_) { 
            if (result_count >= results_buffer->data.size()) {
                LOG_WARNING("More detections found than space in the result buffer. Some detections will be dropped.");
                break;
            }
            DetectionResult& res = results_buffer->data[result_count];
            res.class_id = static_cast<int>(detection_classes[i]);
            res.score = detection_scores[i];
            res.timestamp = timestamp;

            res.ymin = detection_boxes[i * 4 + 0] * input_height_;
            res.xmin = detection_boxes[i * 4 + 1] * input_width_;
            res.ymax = detection_boxes[i * 4 + 2] * input_height_;
            res.xmax = detection_boxes[i * 4 + 3] * input_width_;
            result_count++;
        }
    }
    results_buffer->size = result_count;
    return results_buffer;
}

void InferenceEngine::get_performance_metrics() {
    std::lock_guard<std::mutex> lock(inference_times_mutex_);

    if (total_inferences_ == 0) {
        LOG_INFO("InferenceEngine: No inferences recorded for performance metrics.");
        return;
    }

    double average_duration_ms = 0;
    for (long long duration : inference_times_ms_) {
        average_duration_ms += duration;
    }
    average_duration_ms /= total_inferences_;
    double average_fps = 1000.0 / average_duration_ms;

    double sum_sq_diff = 0;
    for (long long duration : inference_times_ms_) {
        sum_sq_diff += (duration - average_duration_ms) * (duration - average_duration_ms);
    }
    double std_dev_ms = std::sqrt(sum_sq_diff / total_inferences_);

    std::sort(inference_times_ms_.begin(), inference_times_ms_.end());
    size_t percentile_99_index = static_cast<size_t>(std::round(total_inferences_ * 0.99));
    size_t percentile_95_index = static_cast<size_t>(std::round(total_inferences_ * 0.95));
    size_t percentile_50_index = static_cast<size_t>(std::round(total_inferences_ * 0.50));

    long long p99_latency_ms = inference_times_ms_[std::min(percentile_99_index, static_cast<size_t>(total_inferences_ - 1))];
    long long p95_latency_ms = inference_times_ms_[std::min(percentile_95_index, static_cast<size_t>(total_inferences_ - 1))];
    long long p50_latency_ms = inference_times_ms_[std::min(percentile_50_index, static_cast<size_t>(total_inferences_ - 1))];

    LOG_CSV("InferenceEngine", "Inference", p50_latency_ms, p95_latency_ms, p99_latency_ms, 0.0, average_fps);
    LOG_INFO("--- Inference Performance Metrics ---");
    LOG_INFO("  Total Inferences: " + std::to_string(total_inferences_));
    LOG_INFO("  Average FPS: " + std::to_string(average_fps));
    LOG_INFO("  Average Latency: " + std::to_string(average_duration_ms) + " ms");
    LOG_INFO("  Latency Std Dev: " + std::to_string(std_dev_ms) + " ms");
    LOG_INFO("  50th Percentile Latency: " + std::to_string(p50_latency_ms) + " ms");
    LOG_INFO("  95th Percentile Latency: " + std::to_string(p95_latency_ms) + " ms");
    LOG_INFO("  99th Percentile Latency: " + std::to_string(p99_latency_ms) + " ms");
    LOG_INFO("-------------------------------------");

    inference_times_ms_.clear();
    total_inferences_ = 0;
    performance_start_time_ = std::chrono::high_resolution_clock::now();
}

void InferenceEngine::get_state() const {
    LOG_INFO("--- InferenceEngine State ---");
    LOG_INFO("  Running: " + std::to_string(running_));
    LOG_INFO("  Model Path: " + model_path_);
    LOG_INFO("  Input Dimensions: " + std::to_string(input_width_) + "x" + std::to_string(input_height_) + "x" + std::to_string(input_channels_));
    LOG_INFO("  Number of Worker Threads: " + std::to_string(num_threads_));
    LOG_INFO("-----------------------------");
}