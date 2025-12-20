#include "inference.h"
#include "util_logging.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sys/mman.h> // For mmap, munmap
#include <errno.h>    // For errno, strerror
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
    const int RECREATE_INTERVAL = 50; // Recreate delegate every 50 inferences (even more aggressive)
                                  // WORKAROUND: This is reduced from 100 to prevent resource accumulation
                                  // and delegate issues in the Edge TPU delegate which appears to have
                                  // memory management bugs causing "Node number X (EdgeTpuDelegateForCustomOp) 
                                  // failed to invoke" errors.
    
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
            // WORKAROUND: The Edge TPU delegate occasionally fails with "Node number X (EdgeTpuDelegateForCustomOp) failed to invoke"
            // This appears to be a memory corruption bug in the delegate where it might be looking for incorrect tensor names
            // (e.g., "normalizdd_input_image_tensor" instead of "normalized_input_image_tensor").
            // To work around this issue, we retry interpreter creation multiple times and provide detailed error logging.
            if (invoke_status != kTfLiteOk) {
                APP_LOG_ERROR("Failed to invoke interpreter with status: " + std::to_string(invoke_status) + ". Attempting to recreate interpreter.");
                
                // Try to recreate the interpreter with a fresh delegate multiple times
                int max_retries = 3;
                bool recreate_success = false;
                for (int retry = 0; retry < max_retries; ++retry) {
                    APP_LOG_INFO("Attempt " + std::to_string(retry + 1) + " to recreate interpreter.");
                    
                    // Try to recreate the interpreter with a fresh delegate
                    interpreter = create_interpreter();
                    if (!interpreter) {
                        APP_LOG_ERROR("Worker thread failed to recreate interpreter on attempt " + std::to_string(retry + 1) + ". Retrying...");
                        continue;
                    }
                    APP_LOG_INFO("Successfully recreated interpreter with fresh delegate on attempt " + std::to_string(retry + 1) + ".");
                    
                    // Try inference again with the new interpreter
                    invoke_start_time = std::chrono::high_resolution_clock::now();
                    invoke_status = interpreter->Invoke();
                    invoke_end_time = std::chrono::high_resolution_clock::now();
                    
                    if (invoke_status == kTfLiteOk) {
                        recreate_success = true;
                        APP_LOG_INFO("Successfully invoked interpreter after recreation on attempt " + std::to_string(retry + 1) + ".");
                        break;
                    } else {
                        APP_LOG_ERROR("Failed to invoke interpreter on attempt " + std::to_string(retry + 1) + " with status: " + std::to_string(invoke_status) + ". Retrying...");
                    }
                }
                
                if (!recreate_success) {
                    APP_LOG_ERROR("Failed to invoke interpreter even after " + std::to_string(max_retries) + " recreation attempts with status: " + std::to_string(invoke_status) + ". Skipping frame.");
                    continue;
                }
            }
            
            // Periodically recreate the interpreter to prevent resource accumulation and delegate issues
            inference_count++;
            if (inference_count % RECREATE_INTERVAL == 0) {
                APP_LOG_INFO("Periodically recreating interpreter to prevent resource accumulation and delegate issues. Inference count: " + std::to_string(inference_count));
                interpreter = create_interpreter();
                if (!interpreter) {
                    APP_LOG_ERROR("Worker thread failed to recreate interpreter during periodic refresh. Exiting thread.");
                    return;
                }
                APP_LOG_INFO("Successfully recreated interpreter during periodic refresh.");
            }
            
            long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(invoke_end_time - invoke_start_time).count();
            long long duration_us = std::chrono::duration_cast<std::chrono::microseconds>(invoke_end_time - invoke_start_time).count();
            APP_LOG_DEBUG("InferenceEngine: Time to invoke interpreter (inference_done): " + std::to_string(duration_ms) + " ms");
            
            // Store inference timing for statistics
            static std::vector<long long> inference_times_us;
            static int inference_count = 0;
            inference_times_us.push_back(duration_us);
            inference_count++;
            
            // Print inference timing statistics every 100 inferences
            if (inference_count % 100 == 0 && inference_times_us.size() > 0) {
                long long total_time_us = 0;
                long long min_time_us = inference_times_us[0];
                long long max_time_us = inference_times_us[0];
                
                for (long long time : inference_times_us) {
                    total_time_us += time;
                    if (time < min_time_us) min_time_us = time;
                    if (time > max_time_us) max_time_us = time;
                }
                
                // Calculate 95th percentile
                std::sort(inference_times_us.begin(), inference_times_us.end());
                long long p95_index = static_cast<long long>(inference_times_us.size() * 0.95);
                long long p95_time_us = inference_times_us[p95_index];
                
                double avg_time_us = static_cast<double>(total_time_us) / inference_times_us.size();
                
                APP_LOG_INFO("Inference Timing Stats (last 100 inferences): Avg=" + std::to_string(avg_time_us) + "us, Min=" + std::to_string(min_time_us) + "us, P95=" + std::to_string(p95_time_us) + "us, Max=" + std::to_string(max_time_us) + "us");
            }
            
            // Check if interpreter is null after invocation
            if (!interpreter) {
                APP_LOG_ERROR("Interpreter is null after invocation. Skipping frame.");
                continue;
            }
            
            // Record start time for performance measurement
            auto start_time = std::chrono::high_resolution_clock::now();
            
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
            
            // Record end time and calculate duration
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            
            // Log performance metrics
            entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            copy_to_array(entry.module, "InferenceEngine");
            entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
            copy_to_array(entry.event, "performance_logged");
            entry.call_ts_epoch_ms = call_ts;
            entry.tpu_inference_ms = static_cast<float>(duration) / 1000.0f; // Convert to milliseconds
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
    if (tensor_name_str != "serving_default_input:0") {
        APP_LOG_ERROR("Input tensor name mismatch. Expected: serving_default_input:0, Actual: " + tensor_name_str);
        // This is a critical invariant violation - throw exception for graceful handling
        throw std::runtime_error("Input tensor name mismatch. Expected: serving_default_input:0, Actual: " + tensor_name_str);
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

    // Zero-copy optimization: If we have a valid file descriptor, try to map directly
    if (image.fd >= 0 && image.length > 0) {
        APP_LOG_DEBUG("Attempting zero-copy operation with fd: " + std::to_string(image.fd) + ", offset: " + std::to_string(image.offset) + ", length: " + std::to_string(image.length));
        
        // Try to mmap the frame buffer directly
        void* mmap_ptr = mmap(NULL, image.length, PROT_READ, MAP_SHARED, image.fd, image.offset);
        if (mmap_ptr != MAP_FAILED) {
            // Successfully mapped, copy directly from mapped memory
            APP_LOG_DEBUG("Zero-copy mmap successful. Copying " + std::to_string(image.length) + " bytes from mapped memory to tensor.");
            memcpy(tensor_data, static_cast<uint8_t*>(mmap_ptr), image.length);
            munmap(mmap_ptr, image.length);
            return; // Early return since we've already copied the data
        } else {
            APP_LOG_WARNING("Zero-copy mmap failed: " + std::string(strerror(errno)) + ". Falling back to buffer copy.");
        }
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

    // Access tensors using the interpreter's tensor() method
    // Model output order based on our analysis:
    // Output_0 (index 120): classes [1, 10] - UINT8
    // Output_1 (index 121): boxes [1, 10, 4] - UINT8  
    // Output_2 (index 122): num_detections [1] - UINT8
    // Output_3 (index 123): scores [1, 10] - UINT8
    
    // Map the outputs correctly based on their actual indices
    TfLiteTensor* detection_classes_tensor = interpreter->tensor(interpreter->outputs()[0]);   // index 120
    TfLiteTensor* detection_boxes_tensor = interpreter->tensor(interpreter->outputs()[1]);     // index 121
    TfLiteTensor* num_detections_tensor = interpreter->tensor(interpreter->outputs()[2]);      // index 122
    TfLiteTensor* detection_scores_tensor = interpreter->tensor(interpreter->outputs()[3]);    // index 123

    // Check for null pointers before dereferencing
    if (!detection_boxes_tensor || !detection_classes_tensor || !detection_scores_tensor || !num_detections_tensor) {
        APP_LOG_ERROR("One or more output tensors are null. Cannot process detection results.");
        return nullptr;
    }

    // Check that all tensors are UINT8 as expected
    if (detection_boxes_tensor->type != kTfLiteUInt8 || 
        detection_classes_tensor->type != kTfLiteUInt8 || 
        detection_scores_tensor->type != kTfLiteUInt8 || 
        num_detections_tensor->type != kTfLiteUInt8) {
        APP_LOG_ERROR("Unexpected tensor types. Expected all UINT8.");
        return nullptr;
    }

    // Get data pointers from tensors
    const uint8_t* detection_boxes_uint8 = reinterpret_cast<const uint8_t*>(detection_boxes_tensor->data.raw);
    const uint8_t* detection_classes_uint8 = reinterpret_cast<const uint8_t*>(detection_classes_tensor->data.raw);
    const uint8_t* detection_scores_uint8 = reinterpret_cast<const uint8_t*>(detection_scores_tensor->data.raw);
    const uint8_t* num_detections_uint8 = reinterpret_cast<const uint8_t*>(num_detections_tensor->data.raw);

    // Check for null data pointers
    if (!detection_boxes_uint8 || !detection_classes_uint8 || !detection_scores_uint8 || !num_detections_uint8) {
        APP_LOG_ERROR("One or more output tensor data pointers are null. Cannot process detection results.");
        return nullptr;
    }

    // Dequantize the num_detections value
    // For UINT8 tensors, we need to apply: float_value = (uint8_value - zero_point) * scale
    float num_detections_dequantized = 0.0f;
    if (num_detections_tensor->quantization.type == kTfLiteAffineQuantization) {
        // Get quantization parameters from the affine quantization struct
        const TfLiteAffineQuantization* quantization_params = 
            reinterpret_cast<const TfLiteAffineQuantization*>(num_detections_tensor->quantization.params);
        if (quantization_params && quantization_params->scale && quantization_params->scale->size > 0) {
            float scale = quantization_params->scale->data[0];
            int zero_point = 0;  // Default zero point for uint8
            if (quantization_params->zero_point && quantization_params->zero_point->size > 0) {
                zero_point = quantization_params->zero_point->data[0];
            }
            num_detections_dequantized = static_cast<float>(num_detections_uint8[0] - zero_point) * scale;
        } else {
            // Fallback if quantization params are not available
            num_detections_dequantized = static_cast<float>(num_detections_uint8[0]);
        }
    } else {
        // Fallback if no quantization info - assume scale=1.0, zero_point=0
        num_detections_dequantized = static_cast<float>(num_detections_uint8[0]);
    }
    
    const int num_detections = static_cast<int>(num_detections_dequantized);

    // Get quantization parameters for all tensors
    const TfLiteAffineQuantization* boxes_quant_params = nullptr;
    const TfLiteAffineQuantization* classes_quant_params = nullptr;
    const TfLiteAffineQuantization* scores_quant_params = nullptr;
    const TfLiteAffineQuantization* num_detections_quant_params = nullptr;
    
    if (detection_boxes_tensor->quantization.type == kTfLiteAffineQuantization) {
        boxes_quant_params = reinterpret_cast<const TfLiteAffineQuantization*>(detection_boxes_tensor->quantization.params);
    }
    
    if (detection_classes_tensor->quantization.type == kTfLiteAffineQuantization) {
        classes_quant_params = reinterpret_cast<const TfLiteAffineQuantization*>(detection_classes_tensor->quantization.params);
    }
    
    if (detection_scores_tensor->quantization.type == kTfLiteAffineQuantization) {
        scores_quant_params = reinterpret_cast<const TfLiteAffineQuantization*>(detection_scores_tensor->quantization.params);
    }
    
    if (num_detections_tensor->quantization.type == kTfLiteAffineQuantization) {
        num_detections_quant_params = reinterpret_cast<const TfLiteAffineQuantization*>(num_detections_tensor->quantization.params);
    }
        
    // Extract quantization parameters for boxes
    float boxes_scale = 1.0f;
    int boxes_zero_point = 0;
    if (boxes_quant_params && boxes_quant_params->scale && boxes_quant_params->scale->size > 0) {
        boxes_scale = boxes_quant_params->scale->data[0];
    }
    if (boxes_quant_params && boxes_quant_params->zero_point && boxes_quant_params->zero_point->size > 0) {
        boxes_zero_point = boxes_quant_params->zero_point->data[0];
    }
    
    // Extract quantization parameters for classes
    float classes_scale = 1.0f;
    int classes_zero_point = 0;
    if (classes_quant_params && classes_quant_params->scale && classes_quant_params->scale->size > 0) {
        classes_scale = classes_quant_params->scale->data[0];
    }
    if (classes_quant_params && classes_quant_params->zero_point && classes_quant_params->zero_point->size > 0) {
        classes_zero_point = classes_quant_params->zero_point->data[0];
    }
    
    // Extract quantization parameters for scores
    float scores_scale = 1.0f;
    int scores_zero_point = 0;
    if (scores_quant_params && scores_quant_params->scale && scores_quant_params->scale->size > 0) {
        scores_scale = scores_quant_params->scale->data[0];
    }
    if (scores_quant_params && scores_quant_params->zero_point && scores_quant_params->zero_point->size > 0) {
        scores_zero_point = scores_quant_params->zero_point->data[0];
    }
    
    // Extract quantization parameters for num_detections
    float num_detections_scale = 1.0f;
    int num_detections_zero_point = 0;
    if (num_detections_quant_params && num_detections_quant_params->scale && num_detections_quant_params->scale->size > 0) {
        num_detections_scale = num_detections_quant_params->scale->data[0];
    }
    if (num_detections_quant_params && num_detections_quant_params->zero_point && num_detections_quant_params->zero_point->size > 0) {
        num_detections_zero_point = num_detections_quant_params->zero_point->data[0];
    }
    
    // Validate quantization parameters to prevent invalid values
    if (boxes_scale <= 0.0f) boxes_scale = 1.0f;
    if (classes_scale <= 0.0f) classes_scale = 1.0f;
    if (scores_scale <= 0.0f) scores_scale = 1.0f;
    if (num_detections_scale <= 0.0f) num_detections_scale = 1.0f;
    
    // Log quantization parameters for debugging
    APP_LOG_DEBUG("Quantization params - Boxes: scale=" + std::to_string(boxes_scale) + ", zero_point=" + std::to_string(boxes_zero_point));
    APP_LOG_DEBUG("Quantization params - Classes: scale=" + std::to_string(classes_scale) + ", zero_point=" + std::to_string(classes_zero_point));
    APP_LOG_DEBUG("Quantization params - Scores: scale=" + std::to_string(scores_scale) + ", zero_point=" + std::to_string(scores_zero_point));
    APP_LOG_DEBUG("Quantization params - NumDetections: scale=" + std::to_string(num_detections_scale) + ", zero_point=" + std::to_string(num_detections_zero_point));
    
    // Log some raw UINT8 values for debugging
    std::string raw_classes_log = "Raw classes (first 5): ";
    std::string raw_scores_log = "Raw scores (first 5): ";
    for (int i = 0; i < std::min(num_detections, 5); ++i) {
        raw_classes_log += std::to_string(detection_classes_uint8[i]) + " ";
        raw_scores_log += std::to_string(detection_scores_uint8[i]) + " ";
    }
    APP_LOG_DEBUG(raw_classes_log);
    APP_LOG_DEBUG(raw_scores_log);
    
    // Dequantize the num_detections value
    // For UINT8 tensors, we need to apply: float_value = (uint8_value - zero_point) * scale
    num_detections_dequantized = static_cast<float>(num_detections_uint8[0] - num_detections_zero_point) * num_detections_scale;

    APP_LOG_DEBUG("Raw model output - num_detections: " + std::to_string(num_detections));
    APP_LOG_DEBUG("Score threshold: " + std::to_string(score_threshold_));
    std::string scores_log = "Raw model output - top 5 scores: ";
    for (int i = 0; i < std::min(num_detections, 5); ++i) {
        // Properly dequantize UINT8 score to float
        float score_float = static_cast<float>(detection_scores_uint8[i] - scores_zero_point) * scores_scale;
        scores_log += std::to_string(score_float) + " ";
    }
    APP_LOG_DEBUG(scores_log);

    // Log raw detection information for invariant verification
    for (int i = 0; i < num_detections; ++i) {
        // For class IDs, use raw UINT8 values directly as they appear to be class IDs already
        // For scores, properly dequantize UINT8 values to meaningful ranges
        int class_id = static_cast<int>(detection_classes_uint8[i]);  // Use raw value directly
        float score_float = static_cast<float>(detection_scores_uint8[i] - scores_zero_point) * scores_scale;
        APP_LOG_DEBUG("Raw detection " + std::to_string(i) + ": class=" + std::to_string(class_id) + 
                      ", score=" + std::to_string(score_float) + ", raw_class=" + std::to_string(detection_classes_uint8[i]) + 
                      ", raw_score=" + std::to_string(detection_scores_uint8[i]));
    }

    auto timestamp = std::chrono::high_resolution_clock::now();
    size_t result_count = 0;

    for (int i = 0; i < num_detections; ++i) {
        // Properly dequantize UINT8 score to float for threshold comparison
        float score_float = static_cast<float>(detection_scores_uint8[i] - scores_zero_point) * scores_scale;
        if (score_float > score_threshold_) { 
            if (result_count >= results_buffer->data.size()) {
                APP_LOG_WARNING("More detections found than space in the result buffer. Some detections will be dropped.");
                break;
            }
            DetectionResult& res = results_buffer->data[result_count];
            // For class IDs, use raw UINT8 values directly as they appear to be class IDs already
            res.class_id = static_cast<int>(detection_classes_uint8[i]);  // Use raw value directly
            res.score = score_float;
            
            // Validate class ID is within labelmap bounds (1-90)
            if (res.class_id < 1 || res.class_id > 90) {
                APP_LOG_WARNING("Invalid class ID detected: " + std::to_string(res.class_id) + ". Skipping detection.");
                continue;
            }
            
            APP_LOG_DEBUG("InferenceEngine: Detected class_id: " + std::to_string(res.class_id) + " with score: " + std::to_string(res.score));
            res.timestamp = timestamp;

            // Properly dequantize UINT8 box coordinates to normalized coordinates
            // Box format: [ymin, xmin, ymax, xmax] normalized to [0, 1]
            float ymin_norm = static_cast<float>(detection_boxes_uint8[i * 4 + 0] - boxes_zero_point) * boxes_scale;
            float xmin_norm = static_cast<float>(detection_boxes_uint8[i * 4 + 1] - boxes_zero_point) * boxes_scale;
            float ymax_norm = static_cast<float>(detection_boxes_uint8[i * 4 + 2] - boxes_zero_point) * boxes_scale;
            float xmax_norm = static_cast<float>(detection_boxes_uint8[i * 4 + 3] - boxes_zero_point) * boxes_scale;
            
            res.ymin = ymin_norm * input_height_;
            res.xmin = xmin_norm * input_width_;
            res.ymax = ymax_norm * input_height_;
            res.xmax = xmax_norm * input_width_;
            result_count++;
        }
    }
    results_buffer->size = result_count;
    
    // Log filtered detection count for invariant verification
    APP_LOG_INFO("DETECTION_INVARIANT: Raw detections=" + std::to_string(num_detections) + 
                 ", After score threshold=" + std::to_string(result_count));
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