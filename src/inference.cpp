#include "inference.h"
#include "util_logging.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

// Edge TPU delegate C API functions
extern "C" {
    TfLiteDelegate* tflite_plugin_create_delegate(char** options_keys, char** options_values, size_t num_options, void (*report_error)(const char *));
    void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate);
}

// Custom error reporting function for the Edge TPU delegate
void edgetpu_error_reporter(const char* msg) {
    APP_LOG_ERROR("Edge TPU Delegate: " + std::string(msg));
}

float InferenceEngine::get_tpu_temperature() {
    std::ifstream temp_file("/sys/class/apex/apex_0/temp");
    if (!temp_file.is_open()) {
        // This warning can be noisy if the driver is not loaded or the device is not present.
        // It might be better to log this once at startup.
        // APP_LOG_WARNING("Could not open TPU temperature file.");
        return -1.0f;
    }
    std::string line;
    if (std::getline(temp_file, line)) {
        try {
            return std::stof(line) / 1000.0f;
        } catch (const std::invalid_argument& ia) {
            APP_LOG_ERROR("Invalid argument while parsing TPU temperature: " + std::string(ia.what()));
            return -2.0f;
        } catch (const std::out_of_range& oor) {
            APP_LOG_ERROR("Out of range while parsing TPU temperature: " + std::string(oor.what()));
            return -3.0f;
        }
    }
    return -4.0f; // Error reading line
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
    
    APP_LOG_INFO("Model Input Dimensions: " + std::to_string(input_width_) + "x" + std::to_string(input_height_) + "x" + std::to_string(input_channels_));

    if (input_channels_ != 3) {
        throw std::runtime_error("Model expects " + std::to_string(input_channels_) + " channels, but this application is hardcoded for 3 (RGB).");
    }

    // Create the Edge TPU delegate once for the entire InferenceEngine instance.
    APP_LOG_INFO("Edge TPU delegate creation starting...");
    edgetpu_delegate_ = tflite_plugin_create_delegate(nullptr, nullptr, 0, edgetpu_error_reporter);
    if (!edgetpu_delegate_) {
        APP_LOG_ERROR("Edge TPU delegate creation failed (tflite_plugin_create_delegate returned nullptr).");
        // Explicitly report error if delegate creation failed
        edgetpu_error_reporter("tflite_plugin_create_delegate returned nullptr.");
        throw std::runtime_error("Failed to create EdgeTPU delegate in constructor. Ensure libedgetpu1-std is installed and device is connected.");
    }
    APP_LOG_INFO("Edge TPU delegate created successfully in InferenceEngine constructor. Delegate address: " + std::to_string(reinterpret_cast<uintptr_t>(edgetpu_delegate_)));
}

InferenceEngine::~InferenceEngine() {
    stop();
    if (edgetpu_delegate_) {
        tflite_plugin_destroy_delegate(edgetpu_delegate_);
        APP_LOG_INFO("Edge TPU delegate destroyed.");
    }
}

bool InferenceEngine::start() {
    if (running_) {
        APP_LOG_ERROR("InferenceEngine is already running.");
        return false;
    }
    if (!model_) {
        APP_LOG_ERROR("Model not loaded, cannot start inference engine.");
        return false;
    }

    running_ = true;

    for (int i = 0; i < num_threads_; ++i) {
        worker_threads_.emplace_back(&InferenceEngine::worker_thread_func, this);
    }

    APP_LOG_INFO("InferenceEngine started with " + std::to_string(num_threads_) + " worker threads.");
    return true;
}

void InferenceEngine::stop() {
    if (running_.exchange(false)) {
        APP_LOG_INFO("Stopping InferenceEngine...");

        
        for (std::thread& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        worker_threads_.clear();
        APP_LOG_INFO("InferenceEngine stopped.");
    }
}

std::unique_ptr<tflite::Interpreter> InferenceEngine::create_interpreter() {
    std::unique_ptr<tflite::Interpreter> local_interpreter;
    tflite::InterpreterBuilder(*model_, resolver_)(&local_interpreter);
    if (!local_interpreter) {
        APP_LOG_ERROR("Failed to build interpreter.");
        return nullptr;
    }

    // Apply the pre-created EdgeTPU delegate
    if (edgetpu_delegate_ && local_interpreter->ModifyGraphWithDelegate(edgetpu_delegate_) != kTfLiteOk) {
        APP_LOG_ERROR("Failed to apply EdgeTPU delegate. Check if the model is compatible with Edge TPU.");
        return nullptr;
    }
    
    if (local_interpreter->AllocateTensors() != kTfLiteOk) {
        APP_LOG_ERROR("Failed to allocate tensors after applying EdgeTPU delegate.");
        return nullptr;
    }
    
    return local_interpreter;
}

void InferenceEngine::worker_thread_func() {
    std::unique_ptr<tflite::Interpreter> interpreter = create_interpreter();
    if (!interpreter) {
        APP_LOG_ERROR("Worker thread failed to create interpreter. Exiting thread.");
        return;
    }
    
    ImageData input_image;
    while (running_) {
        auto total_loop_start = std::chrono::high_resolution_clock::now();
        // 1. Pop from input queue
        auto pop_start = std::chrono::high_resolution_clock::now();
        if (input_queue_.pop(input_image)) {
            auto pop_end = std::chrono::high_resolution_clock::now();
            APP_LOG_DEBUG("InferenceEngine: Time to pop from queue: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(pop_end - pop_start).count()) + " us");

            if (!input_image.buffer) {
                APP_LOG_ERROR("InferenceEngine received an image with no buffer. Skipping.");
                continue;
            }
            
            long long call_ts = input_image.timestamp_epoch_ms;

            // 2. Set input tensor
            auto set_input_start = std::chrono::high_resolution_clock::now();
            set_input_tensor(interpreter.get(), input_image);
            input_image.buffer.reset(); // Explicitly release the buffer here!
            auto set_input_end = std::chrono::high_resolution_clock::now();
            APP_LOG_DEBUG("InferenceEngine: Time to set input tensor: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(set_input_end - set_input_start).count()) + " us");


            // 3. Invoke interpreter
            auto invoke_start_time = std::chrono::high_resolution_clock::now();
            if (interpreter->Invoke() != kTfLiteOk) {
                APP_LOG_ERROR("Failed to invoke interpreter. Skipping frame.");
                continue;
            }
            auto invoke_end_time = std::chrono::high_resolution_clock::now();
            long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(invoke_end_time - invoke_start_time).count();
            APP_LOG_DEBUG("InferenceEngine: Time to invoke interpreter (inference_done): " + std::to_string(duration_ms) + " ms");
            
            CsvLogEntry entry;
            entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            copy_to_array(entry.module, "InferenceEngine");
            entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
            copy_to_array(entry.event, "inference_done");
            entry.call_ts_epoch_ms = call_ts;
            entry.tpu_inference_ms = static_cast<float>(duration_ms);
            entry.tpu_input_w = input_width_;
            entry.tpu_input_h = input_height_;
            entry.tpu_temp_c = get_tpu_temperature();
            Logger::getInstance().log_csv(entry);

            {
                std::lock_guard<std::mutex> lock(inference_times_mutex_);
                inference_times_ms_.push_back(duration_ms);
                total_inferences_++;
            }
            
            // 4. Get output tensor
            auto get_output_start = std::chrono::high_resolution_clock::now();
            auto results_buffer = get_output_tensor(interpreter.get());
            auto get_output_end = std::chrono::high_resolution_clock::now();
            APP_LOG_DEBUG("InferenceEngine: Time to get output tensor: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(get_output_end - get_output_start).count()) + " us");

            // 5. Push results to output queues
            auto push_output_start = std::chrono::high_resolution_clock::now();
            if (results_buffer && results_buffer->size > 0) {
                APP_LOG_DEBUG("InferenceEngine: Pushing " + std::to_string(results_buffer->size) + " detections to overlay queue.");
                detection_results_for_overlay_queue_.push(results_buffer);
                APP_LOG_DEBUG("InferenceEngine: Pushing " + std::to_string(results_buffer->size) + " detections to logic queue.");
                detection_results_for_logic_queue_.push(results_buffer);
            } else {
                APP_LOG_WARNING("InferenceEngine: No detections to push or results_buffer is null.");
            }
            auto push_output_end = std::chrono::high_resolution_clock::now();
            APP_LOG_DEBUG("InferenceEngine: Time to push results to queues: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(push_output_end - push_output_start).count()) + " us");
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto total_loop_end = std::chrono::high_resolution_clock::now();
        APP_LOG_DEBUG("InferenceEngine: Total worker thread loop iteration time: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(total_loop_end - total_loop_start).count()) + " us");
    }
}

void InferenceEngine::set_input_tensor(tflite::Interpreter* interpreter, const ImageData& image) {
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);

    if (input_tensor->type != kTfLiteUInt8) {
        APP_LOG_ERROR("Input tensor type is not kTfLiteUInt8 as expected. Current type: " + std::to_string(input_tensor->type) + ". Skipping frame.");
        return;
    }

    uint8_t* tensor_data = interpreter->typed_input_tensor<uint8_t>(0);
    memcpy(tensor_data, image.buffer->data.data(), image.buffer->size);
}

std::shared_ptr<DetectionResultBuffer> InferenceEngine::get_output_tensor(tflite::Interpreter* interpreter) {
    auto results_buffer = detection_result_pool_->acquire();
    if (!results_buffer) {
        APP_LOG_WARNING("Failed to acquire a detection result buffer from the pool. No results will be reported for this frame.");
        return nullptr;
    }
    results_buffer->size = 0; // Reset size

    if (interpreter->outputs().size() < 4) {
        APP_LOG_ERROR("Model does not have expected number of output tensors (expected 4 for SSD MobileNet).");
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
                APP_LOG_WARNING("More detections found than space in the result buffer. Some detections will be dropped.");
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
        APP_LOG_INFO("InferenceEngine: No inferences recorded for performance metrics.");
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

    // Populate the new CsvLogEntry fields directly
    CsvLogEntry entry; // Declare entry here
    entry.p50_latency_ms = static_cast<float>(p50_latency_ms);
    entry.p95_latency_ms = static_cast<float>(p95_latency_ms);
    entry.p99_latency_ms = static_cast<float>(p99_latency_ms);
    entry.average_fps = static_cast<float>(average_fps);
    entry.total_frames_processed_or_inferences = total_inferences_;
    entry.average_latency_ms = static_cast<float>(average_duration_ms);
    // Clear details field as it is now structured
    copy_to_array(entry.details, "");
    // Preserve other specific metrics for the InferenceEngine module
    entry.tpu_input_w = input_width_;
    entry.tpu_input_h = input_height_;
    // tpu_temp_c remains -1.0f as per current capabilities
    
    Logger::getInstance().log_csv(entry);
    APP_LOG_INFO("--- Inference Performance Metrics ---");
    APP_LOG_INFO("  Total Inferences: " + std::to_string(total_inferences_));
    APP_LOG_INFO("  Average FPS: " + std::to_string(average_fps));
    APP_LOG_INFO("  Average Latency: " + std::to_string(average_duration_ms) + " ms");
    APP_LOG_INFO("  Latency Std Dev: " + std::to_string(std_dev_ms) + " ms");
    APP_LOG_INFO("  50th Percentile Latency: " + std::to_string(p50_latency_ms) + " ms");
    APP_LOG_INFO("  95th Percentile Latency: " + std::to_string(p95_latency_ms) + " ms");
    APP_LOG_INFO("  99th Percentile Latency: " + std::to_string(p99_latency_ms) + " ms");
    APP_LOG_INFO("-------------------------------------");

    inference_times_ms_.clear();
    total_inferences_ = 0;
    performance_start_time_ = std::chrono::high_resolution_clock::now();
}

void InferenceEngine::get_state() const {
    APP_LOG_INFO("--- InferenceEngine State ---");
    APP_LOG_INFO("  Running: " + std::to_string(running_));
    APP_LOG_INFO("  Model Path: " + model_path_);
    APP_LOG_INFO("  Input Dimensions: " + std::to_string(input_width_) + "x" + std::to_string(input_height_) + "x" + std::to_string(input_channels_));
    APP_LOG_INFO("  Number of Worker Threads: " + std::to_string(num_threads_));
    APP_LOG_INFO("-----------------------------");
}
