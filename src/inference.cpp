#include "inference.h"
#include "util_logging.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include "edgetpu.h"

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

    // Register Edge TPU custom op to allow loading models with Edge TPU operations
    // even when the delegate is not available (fallback to CPU)
    resolver_.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

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
    
    // List Edge TPU devices
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
    
    APP_LOG_INFO("Found " + std::to_string(num_devices) + " Edge TPU devices.");
    
    if (num_devices > 0) {
        // Use the first available device
        const auto& device = devices.get()[0];
        APP_LOG_INFO("Using Edge TPU device: " + std::string(device.path) + " (type: " + std::to_string(device.type) + ")");
        
        // Try to create delegate with options first
        APP_LOG_INFO("Creating Edge TPU delegate with options...");
        std::vector<edgetpu_option> options;
        options.push_back({"verbose", "1"});
        
        edgetpu_delegate_ = edgetpu_create_delegate(
            device.type, 
            device.path,
            options.data(), 
            options.size());
        
        if (!edgetpu_delegate_) {
            APP_LOG_ERROR("Edge TPU delegate creation failed with options.");
            // Try without options
            APP_LOG_INFO("Trying to create Edge TPU delegate without options...");
            edgetpu_delegate_ = edgetpu_create_delegate(
                device.type, 
                device.path,
                nullptr, 
                0);
                
            if (!edgetpu_delegate_) {
                APP_LOG_ERROR("Edge TPU delegate creation failed even without options.");
                // Try with USB device type explicitly
                APP_LOG_INFO("Trying to create Edge TPU delegate with USB device type...");
                edgetpu_delegate_ = edgetpu_create_delegate(
                    EDGETPU_APEX_USB, 
                    device.path,
                    nullptr, 
                    0);
                    
                if (!edgetpu_delegate_) {
                    APP_LOG_ERROR("Edge TPU delegate creation failed with USB device type.");
                    APP_LOG_WARNING("Continuing without Edge TPU delegate. Inference will run on CPU.");
                } else {
                    APP_LOG_INFO("Edge TPU delegate created successfully with USB device type.");
                }
            } else {
                APP_LOG_INFO("Edge TPU delegate created successfully without options.");
            }
        } else {
            APP_LOG_INFO("Edge TPU delegate created successfully with options.");
        }
    } else {
        APP_LOG_WARNING("No Edge TPU devices found. Inference will run on CPU.");
    }
    
    if (edgetpu_delegate_) {
        APP_LOG_INFO("Edge TPU delegate created successfully in InferenceEngine constructor. Delegate address: " + std::to_string(reinterpret_cast<uintptr_t>(edgetpu_delegate_)));
    } else {
        APP_LOG_WARNING("Edge TPU delegate not available. Inference will run on CPU.");
    }
}

InferenceEngine::~InferenceEngine() {
    stop();
    if (edgetpu_delegate_) {
        edgetpu_free_delegate(edgetpu_delegate_);
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
    
    // Counter for periodic delegate recreation
    int inference_count = 0;
    const int RECREATE_INTERVAL = 100; // Recreate delegate every 100 inferences (more aggressive)
    
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

            // Preprocessing (e.g., resizing, format conversion) is assumed to be done before data is put into the queue.
            // We log the time taken for this implicitly by measuring from pop to set_input_start.
            auto preprocessing_end = std::chrono::high_resolution_clock::now(); // Assuming pop_end is the end of preprocessing if it happened before queueing
            APP_LOG_DEBUG("InferenceEngine: Time for preprocessing (implicit from pop to set_input_start): " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(preprocessing_end - pop_end).count()) + " us");

            // 2. Set input tensor (copy data to interpreter input)
            auto set_input_start = std::chrono::high_resolution_clock::now();
            try {
                set_input_tensor(interpreter.get(), input_image);
            } catch (const std::exception& e) {
                APP_LOG_ERROR("Failed to set input tensor: " + std::string(e.what()) + ". Skipping frame.");
                input_image.buffer.reset(); // Explicitly release the buffer here!
                continue;
            }
            input_image.buffer.reset(); // Explicitly release the buffer here!
            auto set_input_end = std::chrono::high_resolution_clock::now();
            APP_LOG_DEBUG("InferenceEngine: Time to copy data to input tensor: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(set_input_end - set_input_start).count()) + " us");


            // 3. Invoke interpreter
            auto invoke_start_time = std::chrono::high_resolution_clock::now();
            TfLiteStatus invoke_status = interpreter->Invoke();
            auto invoke_end_time = std::chrono::high_resolution_clock::now();
            
            // Check if we need to recreate the interpreter due to delegate issues
            if (invoke_status != kTfLiteOk) {
                APP_LOG_ERROR("Failed to invoke interpreter with status: " + std::to_string(invoke_status) + ". Attempting to recreate interpreter.");
                
                // Try to recreate the interpreter with a fresh delegate
                interpreter = create_interpreter();
                if (!interpreter) {
                    APP_LOG_ERROR("Worker thread failed to recreate interpreter. Exiting thread.");
                    return;
                }
                APP_LOG_INFO("Successfully recreated interpreter with fresh delegate.");
                
                // Try inference again with the new interpreter
                invoke_start_time = std::chrono::high_resolution_clock::now();
                invoke_status = interpreter->Invoke();
                invoke_end_time = std::chrono::high_resolution_clock::now();
                
                if (invoke_status != kTfLiteOk) {
                    APP_LOG_ERROR("Failed to invoke interpreter even after recreation with status: " + std::to_string(invoke_status) + ". Skipping frame.");
                    continue;
                }
            }
            
            // Periodically recreate the interpreter to prevent resource accumulation
            inference_count++;
            if (inference_count % RECREATE_INTERVAL == 0) {
                APP_LOG_INFO("Periodically recreating interpreter to prevent resource accumulation. Inference count: " + std::to_string(inference_count));
                interpreter = create_interpreter();
                if (!interpreter) {
                    APP_LOG_ERROR("Worker thread failed to recreate interpreter during periodic refresh. Exiting thread.");
                    return;
                }
                APP_LOG_INFO("Successfully recreated interpreter during periodic refresh.");
            }
            
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

            // 4. Get output tensor
            auto get_output_start = std::chrono::high_resolution_clock::now();
            auto results_buffer = get_output_tensor(interpreter.get());
            auto get_output_end = std::chrono::high_resolution_clock::now();
            APP_LOG_DEBUG("InferenceEngine: Time to get output tensor: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(get_output_end - get_output_start).count()) + " us");

            // 5. Push results to output queues
            auto push_output_start = std::chrono::high_resolution_clock::now();
            if (results_buffer && results_buffer->size > 0) {
                APP_LOG_DEBUG("InferenceEngine: Pushing " + std::to_string(results_buffer->size) + " detections to overlay queue.");
                bool overlay_pushed = detection_results_for_overlay_queue_.push(results_buffer);
                APP_LOG_DEBUG("InferenceEngine: Pushing " + std::to_string(results_buffer->size) + " detections to logic queue.");
                bool logic_pushed = detection_results_for_logic_queue_.push(results_buffer);
                
                if (!overlay_pushed || !logic_pushed) {
                    APP_LOG_WARNING("InferenceEngine: Failed to push detections to one or more queues. Overlay: " + 
                                   std::to_string(overlay_pushed) + ", Logic: " + std::to_string(logic_pushed));
                }
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
    if (!image.buffer) {
        APP_LOG_ERROR("InferenceEngine::set_input_tensor received an ImageData with a null buffer. Skipping frame.");
        return;
    }

    // Invariant checks before setting input tensor
    // 1. Check that we have inputs
    if (interpreter->inputs().size() == 0) {
        APP_LOG_ERROR("Interpreter has no input tensors.");
        return;
    }

    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);

    // 2. Check tensor exists by index and name
    if (!input_tensor) {
        APP_LOG_ERROR("Input tensor at index " + std::to_string(input_tensor_idx) + " is null.");
        return;
    }

    const char* tensor_name = input_tensor->name;
    if (!tensor_name) {
        APP_LOG_ERROR("Input tensor name is null.");
        return;
    }

    std::string tensor_name_str(tensor_name);
    APP_LOG_DEBUG("Input tensor name: " + tensor_name_str);
    
    // This is the critical check - the tensor name must match what the Edge TPU delegate expects
    if (tensor_name_str != "normalized_input_image_tensor") {
        APP_LOG_ERROR("Input tensor name mismatch. Expected: normalized_input_image_tensor, Actual: " + tensor_name_str);
        // This is a critical invariant violation - throw exception for graceful handling
        throw std::runtime_error("Input tensor name mismatch. Expected: normalized_input_image_tensor, Actual: " + tensor_name_str);
    }

    // 3. Check tensor type
    if (input_tensor->type != kTfLiteUInt8) {
        APP_LOG_ERROR("Input tensor type is not kTfLiteUInt8 as expected. Current type: " + std::to_string(input_tensor->type) + ".");
        throw std::runtime_error("Input tensor type is not kTfLiteUInt8 as expected. Current type: " + std::to_string(input_tensor->type) + ".");
    }

    // 4. Check tensor dimensions
    if (input_tensor->dims->size != 4) {
        APP_LOG_ERROR("Input tensor dimensions incorrect. Expected: 4D, Actual: " + std::to_string(input_tensor->dims->size) + "D");
        throw std::runtime_error("Input tensor dimensions incorrect. Expected: 4D, Actual: " + std::to_string(input_tensor->dims->size) + "D");
    }

    // Log tensor information for debugging
    APP_LOG_DEBUG("Input tensor index: " + std::to_string(input_tensor_idx));
    APP_LOG_DEBUG("Input tensor name: " + tensor_name_str);
    APP_LOG_DEBUG("Input tensor type: " + std::to_string(input_tensor->type));
    APP_LOG_DEBUG("Input tensor dimensions: " + std::to_string(input_tensor->dims->size) + "D");
    for (int i = 0; i < input_tensor->dims->size; i++) {
        APP_LOG_DEBUG("  Dimension " + std::to_string(i) + ": " + std::to_string(input_tensor->dims->data[i]));
    }

    uint8_t* tensor_data = interpreter->typed_input_tensor<uint8_t>(0);
    
    // 5. Check tensor data pointer is valid
    if (!tensor_data) {
        APP_LOG_ERROR("Input tensor data pointer is null.");
        throw std::runtime_error("Input tensor data pointer is null.");
    }

    // Validate buffer sizes before memcpy
    size_t expected_tensor_size = input_tensor->bytes;
    size_t image_buffer_capacity = image.buffer->data.size(); // Actual vector capacity
    size_t image_buffer_valid_size = image.buffer->size; // Amount of valid data in buffer

    APP_LOG_DEBUG("Expected tensor size: " + std::to_string(expected_tensor_size) + " bytes");
    APP_LOG_DEBUG("Image buffer capacity: " + std::to_string(image_buffer_capacity) + " bytes");
    APP_LOG_DEBUG("Image buffer valid size: " + std::to_string(image_buffer_valid_size) + " bytes");

    // 6. Check that we have enough data in the buffer for the tensor
    if (image_buffer_valid_size != expected_tensor_size) {
        APP_LOG_ERROR("Mismatch in input tensor size (" + std::to_string(expected_tensor_size) + 
                      " bytes) and image buffer valid size (" + std::to_string(image_buffer_valid_size) + 
                      " bytes).");
        throw std::runtime_error("Mismatch in input tensor size (" + std::to_string(expected_tensor_size) + 
                      " bytes) and image buffer valid size (" + std::to_string(image_buffer_valid_size) + 
                      " bytes).");
    }
    
    // Additional check: Ensure the buffer has sufficient capacity
    if (image_buffer_capacity < expected_tensor_size) {
        APP_LOG_ERROR("Image buffer capacity (" + std::to_string(image_buffer_capacity) + 
                      " bytes) is less than required tensor size (" + std::to_string(expected_tensor_size) + 
                      " bytes).");
        throw std::runtime_error("Image buffer capacity (" + std::to_string(image_buffer_capacity) + 
                      " bytes) is less than required tensor size (" + std::to_string(expected_tensor_size) + 
                      " bytes).");
    }

    APP_LOG_DEBUG("Copying " + std::to_string(image.buffer->size) + " bytes from image buffer to input tensor.");
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

    APP_LOG_DEBUG("Raw model output - num_detections: " + std::to_string(num_detections));
    APP_LOG_DEBUG("Score threshold: " + std::to_string(score_threshold_));
    std::string scores_log = "Raw model output - top 5 scores: ";
    for (int i = 0; i < std::min(num_detections, 5); ++i) {
        scores_log += std::to_string(detection_scores[i]) + " ";
    }
    APP_LOG_DEBUG(scores_log);

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
            APP_LOG_DEBUG("InferenceEngine: Detected class_id: " + std::to_string(res.class_id) + " with score: " + std::to_string(res.score));
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



void InferenceEngine::get_state() const {
    APP_LOG_INFO("--- InferenceEngine State ---");
    APP_LOG_INFO("  Running: " + std::to_string(running_));
    APP_LOG_INFO("  Model Path: " + model_path_);
    APP_LOG_INFO("  Input Dimensions: " + std::to_string(input_width_) + "x" + std::to_string(input_height_) + "x" + std::to_string(input_channels_));
    APP_LOG_INFO("  Number of Worker Threads: " + std::to_string(num_threads_));
    APP_LOG_INFO("-----------------------------");
}