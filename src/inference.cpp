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
#include "application.h"  // For Application counter updates

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
                                     TripleBuffer<DetectionResults>* detection_results_for_overlay_buffer, 
                                     DetectionResultsQueue& detection_results_for_logic_queue, 
                                     std::shared_ptr<BufferPool<DetectionResult>> detection_result_pool,
                                     float score_threshold,
                                     int num_threads)
    : model_path_(model_path), 
      input_queue_(input_queue), 
      detection_results_for_overlay_buffer_(detection_results_for_overlay_buffer), 
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
        
        // Use a shorter timeout or fewer retries to avoid blocking startup
        edgetpu_delegate_ = edgetpu_create_delegate(
            device.type, 
            device.path,
            options.data(), 
            options.size());
        
        if (!edgetpu_delegate_) {
            APP_LOG_ERROR("Edge TPU delegate creation failed with options. Falling back to CPU to avoid blocking startup.");
            // We skip further retries (like without options or USB type) because they often 
            // trigger the same kernel timeout (12s) which blocks the main thread.
            APP_LOG_WARNING("Continuing without Edge TPU delegate. Inference will run on CPU.");
        } else {
            APP_LOG_INFO("Edge TPU delegate created successfully.");
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
        // Push dummy data to wake up any threads blocked on pop operations
        ImageData dummy_data(std::chrono::steady_clock::now(), -1); // Initialize with default values
        dummy_data.buffer = nullptr; // Mark as dummy
        dummy_data.width = 0;
        dummy_data.height = 0;
        input_queue_.push(std::move(dummy_data));
        
        for (std::thread& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        worker_threads_.clear();
        APP_LOG_INFO("InferenceEngine stopped.");
    }
}

void InferenceEngine::recreate_delegate() {
    // Thread-safety: Use unique_lock to ensure no other thread is using the delegate
    // while we are freeing and recreating it.
    std::unique_lock<std::shared_mutex> lock(delegate_mutex_);

    // 1. Free existing delegate
    if (edgetpu_delegate_) {
        APP_LOG_INFO("Recreating Delegate: Freeing old Edge TPU delegate...");
        edgetpu_free_delegate(edgetpu_delegate_);
        edgetpu_delegate_ = nullptr;
    }

    // 2. Create new delegate
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    if (num_devices > 0) {
        const auto& device = devices.get()[0];
        std::vector<edgetpu_option> options;
        options.push_back({"verbose", "1"});
        
        edgetpu_delegate_ = edgetpu_create_delegate(device.type, device.path, options.data(), options.size());
        if (edgetpu_delegate_) {
             APP_LOG_INFO("Recreating Delegate: Success. Address: " + std::to_string(reinterpret_cast<uintptr_t>(edgetpu_delegate_)));
        } else {
             APP_LOG_ERROR("Recreating Delegate: Failed to create delegate.");
        }
    } else {
        APP_LOG_ERROR("Recreating Delegate: No Edge TPU devices found.");
    }
}

std::unique_ptr<tflite::Interpreter> InferenceEngine::create_interpreter() {
    // Safety: Take shared lock because we read edgetpu_delegate_
    std::shared_lock<std::shared_mutex> lock(delegate_mutex_);
    
    std::unique_ptr<tflite::Interpreter> local_interpreter;
    tflite::InterpreterBuilder(*model_, resolver_)(&local_interpreter);
    if (!local_interpreter) {
        APP_LOG_ERROR("Failed to build interpreter.");
        return nullptr;
    }

    // Apply the pre-created EdgeTPU delegate
    if (edgetpu_delegate_ && local_interpreter->ModifyGraphWithDelegate(edgetpu_delegate_) != kTfLiteOk) {
        APP_LOG_ERROR("Failed to apply EdgeTPU delegate.");
        return nullptr;
    }
    
    if (local_interpreter->AllocateTensors() != kTfLiteOk) {
        APP_LOG_ERROR("Failed to allocate tensors.");
        return nullptr;
    }
    
    return local_interpreter;
}

void InferenceEngine::worker_thread_func() {
    std::unique_ptr<tflite::Interpreter> interpreter = create_interpreter();
    if (!interpreter) {
        APP_LOG_ERROR("Worker thread failed to create interpreter initially. Will continue to consume and release buffers to avoid pipeline stall.");
    }
    
    // Counter for periodic delegate recreation
    int inference_count = 0;
    const int RECREATE_INTERVAL = 1000000; // Disable periodic resets (recreate only on error)
    
    // RAII FrameAccountingGuard to ensure P=C+D accounting
    struct FrameAccountingGuard {
        InferenceEngine* engine;
        bool inference_produced = false;
        
        FrameAccountingGuard(InferenceEngine* e) : engine(e) {
            // Only increment consumed counter for valid items popped from queue
            // to maintain synchronicity with producer's count (poison pills not counted)
            engine->frames_consumed_.fetch_add(1);
            if (engine->app_ref_) {
                engine->app_ref_->increment_camera_frames_consumed_by_inference();
            }
        }
        
        ~FrameAccountingGuard() {
            // Account for inference results: either produced or dropped
            if (inference_produced) {
                if (engine->app_ref_) {
                    engine->app_ref_->increment_inference_results_produced(1);
                }
                engine->results_produced_.fetch_add(1);
            } else {
                // Try to push empty results to maintain heartbeat for P=C+D with downstream consumers
                auto empty_results = engine->detection_result_pool_->acquire();
                if (empty_results) {
                    empty_results->size = 0;
                    empty_results->valid = false;
                    empty_results->t_inf_start = get_time_raw_ms();
                    empty_results->t_inf_end = empty_results->t_inf_start;
                    
                    // Push to both logic and overlay to maintain sync
                    engine->detection_results_for_logic_queue_.push(empty_results);
                    if (engine->detection_results_for_overlay_buffer_) {
                         DetectionResults& overlay_results = engine->detection_results_for_overlay_buffer_->get_write_buffer();
                         overlay_results.clear();
                         engine->detection_results_for_overlay_buffer_->commit_write();
                    }
                    
                    // Since we pushed a (heartbeat) result, this frame is PRODUCED, not dropped
                    if (engine->app_ref_) {
                        engine->app_ref_->increment_inference_results_produced(1);
                    }
                    engine->results_produced_.fetch_add(1);
                } else {
                    // Pool exhausted, this is a true drop
                    if (engine->app_ref_) {
                        engine->app_ref_->increment_inference_results_dropped();
                    }
                    engine->logic_queue_drop_count_.fetch_add(1);
                    engine->overlay_queue_drop_count_.fetch_add(1);
                }
            }
        }
    };

    while (running_) {
        ImageData input_image;
        // Use blocking wait_pop instead of polling to eliminate micro-stutter
        if (input_queue_.wait_pop(input_image, std::chrono::milliseconds(10))) {
            // Check for poison pill first before creating guard to avoid counting it
            if (!input_image.buffer) {
                if (!running_) return;
                continue;
            }
            
            FrameAccountingGuard guard(this);

            if (!interpreter) {
                // No interpreter available, just release the buffer and continue
                input_image.buffer.reset();
                continue;
            }
            
            long long call_ts = input_image.t_capture_raw_ms;

            // 2. Set input tensor
            try {
                set_input_tensor(interpreter.get(), input_image);
            } catch (...) {
                input_image.buffer.reset();
                continue;
            }
            input_image.buffer.reset();
            
            // 3. Invoke interpreter
            auto invoke_start_time = std::chrono::steady_clock::now();
            TfLiteStatus invoke_status;
            {
                std::shared_lock<std::shared_mutex> lock(delegate_mutex_);
                invoke_status = interpreter->Invoke();
            }
            auto invoke_end_time = std::chrono::steady_clock::now();
            
            if (invoke_status != kTfLiteOk) {
                // Recreate logic remains but with zero hot-path logging
                int max_retries = 3;
                bool recreate_success = false;
                for (int retry = 0; retry < max_retries; ++retry) {
                    interpreter.reset();
                    recreate_delegate();
                    interpreter = create_interpreter();
                    if (!interpreter) continue;
                    
                    invoke_start_time = std::chrono::steady_clock::now();
                    {
                        std::shared_lock<std::shared_mutex> lock(delegate_mutex_);
                        invoke_status = interpreter->Invoke();
                    }
                    invoke_end_time = std::chrono::steady_clock::now();
                    
                    if (invoke_status == kTfLiteOk) {
                        recreate_success = true;
                        break;
                    }
                }
                if (!recreate_success) continue;
            }
            
            inference_count++;
            if (inference_count % RECREATE_INTERVAL == 0) {
                interpreter.reset();
                recreate_delegate();
                interpreter = create_interpreter();
                if (!interpreter) return;
            }
            
            [[maybe_unused]] long long duration_us = std::chrono::duration_cast<std::chrono::microseconds>(invoke_end_time - invoke_start_time).count();
            
            // Update telemetry (with systolic warm-up bypass)
            last_inference_timestamp_ = call_ts;
            if (inference_count > 10) {
                avg_inference_time_us_.store(static_cast<long long>(avg_inference_time_us_.load() * 0.9 + duration_us * 0.1));
            }
            
            if (inference_count % 100 == 0) {
                // Pipe sanitized thermal sensor to telemetry
                float temp = get_tpu_temperature();
                if (temp > -10.0f) {
                    tpu_temperature_.store(temp);
                }

                // Rate calculation for watchdog/monitor using class state
                auto now = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_rate_check_).count();
                if (elapsed_ms >= 1000) {
                    int current_count = inference_count;
                    int diff = current_count - last_inference_count_.load();
                    inference_rate_.store(static_cast<int>(diff * 1000 / elapsed_ms));
                    last_inference_count_.store(current_count);
                    last_rate_check_ = now;
                }
            }
            
            // 4. Get output tensor
            auto results_buffer = get_output_tensor(interpreter.get(), input_image);

            // Telemetry Logging
            CsvLogEntry entry;
            entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            
            // Reconstruct capture epoch time
            auto now_steady = std::chrono::steady_clock::now();
            auto since_capture = std::chrono::duration_cast<std::chrono::milliseconds>(now_steady - input_image.capture_time).count();
            entry.call_ts_epoch_ms = entry.produced_ts_epoch_ms - since_capture;

            copy_to_array(entry.module, "Inference");
            copy_to_array(entry.event, "inference_done");
            entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
            entry.cam_frame_id = input_image.frame_id;
            entry.tpu_inference_ms = static_cast<float>(duration_us) / 1000.0f;
            entry.tpu_temp_c = tpu_temperature_.load();

            if (results_buffer && results_buffer->size > 0) {
                 const auto& best_det = results_buffer->data[0];
                 entry.tpu_model_score = best_det.score;
                 entry.tpu_class_id = best_det.class_id;
            }
            Logger::getInstance().log_csv(entry);

            // 5. Push results to output buffers/queues
            if (results_buffer) {
                if (detection_results_for_overlay_buffer_) {
                    DetectionResults& overlay_results = detection_results_for_overlay_buffer_->get_write_buffer();
                    if (results_buffer->size > 0) {
                        overlay_results.assign(results_buffer->data.data(), results_buffer->data.data() + results_buffer->size);
                    }
                    else {
                        overlay_results.clear();
                    }
                    detection_results_for_overlay_buffer_->commit_write();
                }
                
                detection_results_for_logic_queue_.push(results_buffer);
                guard.inference_produced = true;  // Mark as successfully produced (even if size=0, we still produced a result)
            }
        }
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
    
    // "Memory Z" Watchdog Fix:
    // The EdgeTPU delegate can corrupt the input tensor name (e.g., "normalizdd_...").
    // We stop checking for "serving_default_input:0" and trust Index 0.
    // if (tensor_name_str != "serving_default_input:0") { ... } <- REMOVED

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

std::shared_ptr<DetectionResultBuffer> InferenceEngine::get_output_tensor(tflite::Interpreter* interpreter, const ImageData& input_image) {
    auto results_buffer = detection_result_pool_->acquire();
    if (!results_buffer) {
        APP_LOG_WARNING("Failed to acquire a detection result buffer from the pool. No results will be reported for this frame.");
        return nullptr;
    }
    results_buffer->size = 0; // Reset size

    size_t num_outputs = interpreter->outputs().size();
    std::vector<DetectionResult> raw_detections;
    auto timestamp = std::chrono::steady_clock::now();

    // -------------------------------------------------------------------------
    // STRATEGY A: Standard SSD MobileNet (4 outputs)
    // -------------------------------------------------------------------------
    if (num_outputs == 4) {
        // FORENSIC FIX: Mapping adjusted based on quantization dump
        // Output 0: Scores (Scale 0.0039), Output 3: Classes (Scale 0.047)
        TfLiteTensor* detection_scores_tensor = interpreter->tensor(interpreter->outputs()[0]);
        TfLiteTensor* detection_boxes_tensor = interpreter->tensor(interpreter->outputs()[1]);
        TfLiteTensor* num_detections_tensor = interpreter->tensor(interpreter->outputs()[2]);
        TfLiteTensor* detection_classes_tensor = interpreter->tensor(interpreter->outputs()[3]);

        if (!detection_boxes_tensor || !detection_classes_tensor || !detection_scores_tensor || !num_detections_tensor) {
            APP_LOG_ERROR("One or more output tensors are null.");
            return nullptr;
        }

        // 1. Dequantize num_detections (Count)
        float num_detections_val = 0.0f;
        const uint8_t* num_det_uint8 = reinterpret_cast<const uint8_t*>(num_detections_tensor->data.raw);
        if (num_detections_tensor->quantization.type == kTfLiteAffineQuantization) {
            const TfLiteAffineQuantization* quant = reinterpret_cast<const TfLiteAffineQuantization*>(num_detections_tensor->quantization.params);
            float scale = (quant && quant->scale && quant->scale->size > 0) ? quant->scale->data[0] : 1.0f;
            int zero_point = (quant && quant->zero_point && quant->zero_point->size > 0) ? quant->zero_point->data[0] : 0;
            num_detections_val = (static_cast<float>(num_det_uint8[0]) - zero_point) * scale;
        } else {
            // Fallback to legacy params if modern quantization metadata is missing
            num_detections_val = (static_cast<float>(num_det_uint8[0]) - num_detections_tensor->params.zero_point) * num_detections_tensor->params.scale;
        }
        int num_detections = std::min(static_cast<int>(std::round(num_detections_val)), 100);

        // 2. Setup Quantization for Boxes, Classes, Scores
        auto get_robust_quant = [](TfLiteTensor* t, float& scale, int& zp) {
            if (t->quantization.type == kTfLiteAffineQuantization) {
                const TfLiteAffineQuantization* q = reinterpret_cast<const TfLiteAffineQuantization*>(t->quantization.params);
                scale = (q && q->scale && q->scale->size > 0) ? q->scale->data[0] : 1.0f;
                zp = (q && q->zero_point && q->zero_point->size > 0) ? q->zero_point->data[0] : 0;
            } else {
                scale = t->params.scale;
                zp = t->params.zero_point;
            }
            // Sanity check: if scale is still 0 (uninitialized), default to 1.0 to prevent zero-scores
            if (scale == 0.0f) scale = 1.0f;
        };

        float box_scale, class_scale, score_scale;
        int box_zp, class_zp, score_zp;
        get_robust_quant(detection_boxes_tensor, box_scale, box_zp);
        get_robust_quant(detection_classes_tensor, class_scale, class_zp);
        get_robust_quant(detection_scores_tensor, score_scale, score_zp);

        const uint8_t* boxes = reinterpret_cast<const uint8_t*>(detection_boxes_tensor->data.raw);
        const uint8_t* classes = reinterpret_cast<const uint8_t*>(detection_classes_tensor->data.raw);
        const uint8_t* scores = reinterpret_cast<const uint8_t*>(detection_scores_tensor->data.raw);

        for (int i = 0; i < num_detections; ++i) {
            float score = (static_cast<float>(scores[i]) - score_zp) * score_scale;
            if (score > score_threshold_) {
                float class_val = (static_cast<float>(classes[i]) - class_zp) * class_scale;
                
                DetectionResult res;
                res.class_id = static_cast<int>(std::round(class_val)) + 1; // 0-based -> 1-based
                res.score = score;
                res.source_frame_id = input_image.frame_id;
                res.timestamp = timestamp;
                
                res.ymin = (static_cast<float>(boxes[i * 4 + 0]) - box_zp) * box_scale;
                res.xmin = (static_cast<float>(boxes[i * 4 + 1]) - box_zp) * box_scale;
                res.ymax = (static_cast<float>(boxes[i * 4 + 2]) - box_zp) * box_scale;
                res.xmax = (static_cast<float>(boxes[i * 4 + 3]) - box_zp) * box_scale;

                // Clamp and fix
                res.ymin = std::max(0.0f, std::min(1.0f, res.ymin));
                res.xmin = std::max(0.0f, std::min(1.0f, res.xmin));
                res.ymax = std::max(0.0f, std::min(1.0f, res.ymax));
                res.xmax = std::max(0.0f, std::min(1.0f, res.xmax));
                if (res.xmin > res.xmax) std::swap(res.xmin, res.xmax);
                if (res.ymin > res.ymax) std::swap(res.ymin, res.ymax);

                raw_detections.push_back(res);
            }
        }
    } 
    // -------------------------------------------------------------------------
    // STRATEGY B: Multi-Tensor Class Output (13 outputs)
    // -------------------------------------------------------------------------
    else if (num_outputs == 13) {
        static bool logged_13_tensor = false;
        if (!logged_13_tensor) {
            APP_LOG_INFO("Detected 13-output model. Activating Global NMS Logic.");
            for (int i = 0; i < 13; ++i) {
                TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[i]);
                if (tensor) {
                     std::string dims_str = "[";
                     for (int d = 0; d < tensor->dims->size; ++d) dims_str += std::to_string(tensor->dims->data[d]) + (d < tensor->dims->size - 1 ? "," : "");
                     dims_str += "]";
                     APP_LOG_INFO("Tensor " + std::to_string(i) + " Dims: " + dims_str + " Type: " + std::to_string(tensor->type));
                }
            }
            logged_13_tensor = true;
        }

        // Iterate through all 13 tensors (assumed one per class)
        for (int i = 0; i < 13; ++i) {
            TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[i]);
            if (!tensor || tensor->type != kTfLiteUInt8) continue;

            // Get robust quantization metadata
            float scale = 1.0f; 
            int zero_point = 0;
            if (tensor->quantization.type == kTfLiteAffineQuantization) {
                 const TfLiteAffineQuantization* q = reinterpret_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
                 scale = (q && q->scale && q->scale->size > 0) ? q->scale->data[0] : 1.0f;
                 zero_point = (q && q->zero_point && q->zero_point->size > 0) ? q->zero_point->data[0] : 0;
            } else {
                 scale = tensor->params.scale;
                 zero_point = tensor->params.zero_point;
            }
            if (scale == 0.0f) scale = 1.0f;

            // Check shape: Expecting [1, N, K] where K >= 4
            if (tensor->dims->size != 3) continue;
            
            int num_candidates = tensor->dims->data[1]; // N
            int feat_dim = tensor->dims->data[2];      // K

            const uint8_t* data = reinterpret_cast<const uint8_t*>(tensor->data.raw);

            for (int j = 0; j < num_candidates; ++j) {
                int offset = j * feat_dim;
                
                // Determine Score
                float score = 0.0f;
                if (feat_dim >= 5) {
                    // Assume [y, x, y, x, score]
                    score = (static_cast<float>(data[offset + 4]) - zero_point) * scale;
                } else {
                    // If no score column, maybe implicit 1.0? 
                    // But for NMS we need scores. Skip if ambiguous.
                    continue; 
                }

                // Global Threshold: 0.6 per mandate (overriding config if lower)
                if (score < 0.6f) continue;

                DetectionResult res;
                res.class_id = i + 1; // 0-based index -> 1-based class ID (Target=12)
                res.score = score;
                res.source_frame_id = input_image.frame_id;
                res.timestamp = timestamp;
                
                res.ymin = (static_cast<float>(data[offset + 0]) - zero_point) * scale;
                res.xmin = (static_cast<float>(data[offset + 1]) - zero_point) * scale;
                res.ymax = (static_cast<float>(data[offset + 2]) - zero_point) * scale;
                res.xmax = (static_cast<float>(data[offset + 3]) - zero_point) * scale;

                // Clamp
                res.ymin = std::max(0.0f, std::min(1.0f, res.ymin));
                res.xmin = std::max(0.0f, std::min(1.0f, res.xmin));
                res.ymax = std::max(0.0f, std::min(1.0f, res.ymax));
                res.xmax = std::max(0.0f, std::min(1.0f, res.xmax));
                
                // Fix inverted
                if (res.xmin > res.xmax) std::swap(res.xmin, res.xmax);
                if (res.ymin > res.ymax) std::swap(res.ymin, res.ymax);
                
                // 10-Frame Logging for verification
                static int raw_dump_counter = 0;
                if (raw_dump_counter < 100) { // Limit total logs
                     // APP_LOG_INFO("[RAW] Class: " + std::to_string(res.class_id) + " | Score: " + std::to_string(score) + " | Box: [" + std::to_string(res.xmin) + ", " + std::to_string(res.ymin) + "]");
                     raw_dump_counter++;
                }

                raw_detections.push_back(res);
            }
        }
    } else {
        APP_LOG_ERROR("Unsupported output count: " + std::to_string(num_outputs));
        return nullptr;
    }

    // -------------------------------------------------------------------------
    // GLOBAL NMS (Unified across all classes)
    // -------------------------------------------------------------------------
    // Sort by score descending
    std::sort(raw_detections.begin(), raw_detections.end(), [](const DetectionResult& a, const DetectionResult& b) {
        return a.score > b.score;
    });

    std::vector<DetectionResult> filtered_detections;
    std::vector<bool> is_suppressed(raw_detections.size(), false);
    
    auto calculate_iou = [](const DetectionResult& a, const DetectionResult& b) {
        float x_left = std::max(a.xmin, b.xmin);
        float y_top = std::max(a.ymin, b.ymin);
        float x_right = std::min(a.xmax, b.xmax);
        float y_bottom = std::min(a.ymax, b.ymax);
        if (x_right < x_left || y_bottom < y_top) return 0.0f;
        float intersection_area = (x_right - x_left) * (y_bottom - y_top);
        float area1 = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        float area2 = (b.xmax - b.xmin) * (b.ymax - b.ymin);
        float union_area = area1 + area2 - intersection_area;
        return (union_area > 0.0f) ? (intersection_area / union_area) : 0.0f;
    };

    int pruned_count = 0;
    for (size_t i = 0; i < raw_detections.size(); ++i) {
        if (is_suppressed[i]) continue;
        filtered_detections.push_back(raw_detections[i]);
        
        for (size_t j = i + 1; j < raw_detections.size(); ++j) {
            if (is_suppressed[j]) continue;
            
            float iou = calculate_iou(raw_detections[i], raw_detections[j]);
            if (iou > 0.45f) { // IoU Threshold
                is_suppressed[j] = true;
                pruned_count++;
                
                // Diagnostic Log for significant overlaps being pruned
                static int overlap_log_limit = 0;
                if (overlap_log_limit < 5) {
                    APP_LOG_DEBUG("[DEBUG] Global NMS Pruned: Class " + std::to_string(raw_detections[j].class_id) + 
                                  " (Score " + std::to_string(raw_detections[j].score) + ") by Class " + 
                                  std::to_string(raw_detections[i].class_id) + " (Score " + std::to_string(raw_detections[i].score) + 
                                  ") IoU=" + std::to_string(iou));
                    overlap_log_limit++;
                }
            }
        }
    }

    if (!filtered_detections.empty()) {
        APP_LOG_INFO("Global NMS: Collected " + std::to_string(raw_detections.size()) + 
                     " proposals. Pruned " + std::to_string(pruned_count) + 
                     ". Outputting " + std::to_string(filtered_detections.size()) + " final boxes.");
    }

    // Fill buffer
    size_t result_count = 0;
    for (const auto& det : filtered_detections) {
        if (result_count >= results_buffer->data.size()) break;
        results_buffer->data[result_count++] = det;
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