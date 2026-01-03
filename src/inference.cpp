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
                                     std::shared_ptr<ObjectPool<ImageData>> image_data_pool,
                                     std::shared_ptr<ObjectPool<ResultToken>> result_token_pool,
                                     float score_threshold,
                                     int num_threads)
    : model_path_(model_path), 
      input_queue_(input_queue), 
      detection_results_for_overlay_buffer_(detection_results_for_overlay_buffer), 
      detection_results_for_logic_queue_(detection_results_for_logic_queue),
      detection_result_pool_(detection_result_pool),
      image_data_pool_(image_data_pool),
      result_token_pool_(result_token_pool),
      num_threads_(1), // FORCE TO 1 THREAD
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
        for (int i = 0; i < num_threads_; ++i) {
            ImageData* dummy_data = image_data_pool_->acquire();
            if (dummy_data) {
                dummy_data->buffer = nullptr; // Mark as dummy
                input_queue_.push(dummy_data);
            }
        }
        
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
    // MANDATE: If the delegate is configured but fails to apply, this is a FATAL error
    // for an Edge TPU compiled model.
    if (edgetpu_delegate_) {
        TfLiteStatus status = local_interpreter->ModifyGraphWithDelegate(edgetpu_delegate_);
        if (status != kTfLiteOk) {
            APP_LOG_ERROR("FATAL: Failed to apply EdgeTPU delegate to interpreter. Status: " + std::to_string(status));
            return nullptr;
        }
        APP_LOG_INFO("EdgeTPU delegate applied successfully to interpreter.");
    } else {
        APP_LOG_ERROR("FATAL: No EdgeTPU delegate available. Refusing to run on CPU to avoid massive latency.");
        return nullptr;
    }
    
    if (local_interpreter->AllocateTensors() != kTfLiteOk) {
        APP_LOG_ERROR("Failed to allocate tensors.");
        return nullptr;
    }
    
    return local_interpreter;
}

extern std::atomic<bool> g_running;

void InferenceEngine::worker_thread_func() {
    std::unique_ptr<tflite::Interpreter> interpreter = create_interpreter();
    if (!interpreter) {
        return; 
    }
    
    APP_LOG_INFO("InferenceEngine: Worker thread started - HEARTBEAT");
    int frames_processed = 0;
    auto last_heartbeat_time = std::chrono::steady_clock::now();
    
    while (running_ && g_running.load(std::memory_order_acquire)) {
        ImageData* input_image_ptr = nullptr;
        if (input_queue_.wait_pop(input_image_ptr, std::chrono::milliseconds(10))) {
            if (!input_image_ptr) {
                if (!running_) return;
                continue;
            }

            struct AccountingGuard {
                InferenceEngine* engine;
                ImageData* input_image_ptr;
                bool output_produced = false;
                AccountingGuard(InferenceEngine* e, ImageData* p) : engine(e), input_image_ptr(p) {}
                ~AccountingGuard() {
                    if (engine->app_ref_) {
                        // Stage 2: TPU Processor -> Inference Engine
                        engine->app_ref_->inc_proc_to_inf_consumed();
                        
                        // Stage 3: Inference Engine -> Logic/Overlay
                        // Every frame taken MUST result in either a successfully produced output
                        // or an explicitly dropped one for the next stage's Produced counter.
                        if (!output_produced) {
                            engine->app_ref_->increment_inference_results_produced(1);
                            engine->app_ref_->increment_inference_results_dropped();
                        }
                    }
                    if (input_image_ptr) {
                        engine->image_data_pool_->release(input_image_ptr);
                    }
                }
            } guard(this, input_image_ptr);

            if (!input_image_ptr->buffer) {
                continue;
            }
            
            ImageData& input_image = *input_image_ptr;
            if (!interpreter) {
                continue;
            }
            
            // 2. Set input tensor
            try {
                set_input_tensor(interpreter.get(), input_image);
            } catch (const std::exception& e) {
                APP_LOG_ERROR("InferenceEngine: Exception in set_input_tensor: " + std::string(e.what()));
                logic_queue_drop_count_.fetch_add(1); 
                continue;
            } catch (...) {
                APP_LOG_ERROR("InferenceEngine: Unknown exception in set_input_tensor");
                logic_queue_drop_count_.fetch_add(1); 
                continue;
            }
            
            // 3. Invoke interpreter
            uint64_t t_inf_start = get_time_raw_ms();
            auto invoke_start_time = std::chrono::steady_clock::now();
            TfLiteStatus invoke_status;
            
            // Phase II Mandate: No shared Invoke() mutex. 
            // Each worker thread owns its interpreter instance.
            {
                input_image.buffer.reset(); // Release image buffer ASAP
                std::shared_lock<std::shared_mutex> lock(delegate_mutex_);
                invoke_status = interpreter->Invoke();
            }
            std::cerr << "InferenceEngine: Invoke() returned " << invoke_status << " for frame " << input_image.frame_id << std::endl;
            auto invoke_end_time = std::chrono::steady_clock::now();
            uint64_t t_inf_end = get_time_raw_ms();
            long long duration_us = std::chrono::duration_cast<std::chrono::microseconds>(invoke_end_time - invoke_start_time).count();
            
            // Update telemetry
            avg_inference_time_us_.store(static_cast<long long>(avg_inference_time_us_.load() * 0.9 + duration_us * 0.1));
            last_inference_timestamp_ = input_image.t_capture_raw_ms;

            if (invoke_status != kTfLiteOk) {
                // Recreate logic remains
                int max_retries = 3;
                bool recreate_success = false;
                for (int retry = 0; retry < max_retries; ++retry) {
                    interpreter.reset();
                    recreate_delegate();
                    interpreter = create_interpreter();
                    if (!interpreter) continue;
                    
                    t_inf_start = get_time_raw_ms();
                    {
                        std::shared_lock<std::shared_mutex> lock(delegate_mutex_);
                        invoke_status = interpreter->Invoke();
                    }
                    t_inf_end = get_time_raw_ms();
                    
                    if (invoke_status == kTfLiteOk) {
                        recreate_success = true;
                        break;
                    }
                }
                if (!recreate_success) {
                    continue; 
                }
            }
            
            int current_total = total_inference_count_.fetch_add(1, std::memory_order_relaxed) + 1;
            
            if (current_total % 100 == 0) {
                float temp = get_tpu_temperature();
                if (temp > -10.0f) {
                    tpu_temperature_.store(temp);
                }

                uint64_t now_ms = get_time_raw_ms();
                // Non-blocking rate check
                uint64_t last_ms = last_rate_check_ms_.load(std::memory_order_relaxed);
                if (now_ms - last_ms >= 1000) {
                    if (last_rate_check_ms_.compare_exchange_strong(last_ms, now_ms)) {
                        int diff = current_total - last_inference_count_checkpoint_.load();
                        inference_rate_.store(static_cast<int>(diff * 1000 / (now_ms - last_ms)));
                        last_inference_count_checkpoint_.store(current_total);
                    }
                }
            }
            
            // 4. Get output tensor
            auto results_buffer = get_output_tensor(interpreter.get(), input_image);

            // 5. Push results
            if (results_buffer) {
                // Propagate metadata to buffer for Logic Module and Telemetry
                results_buffer->frame_id = input_image.frame_id;
                results_buffer->t_capture_raw_ms = input_image.t_capture_raw_ms;
                results_buffer->t_inf_start = t_inf_start;
                results_buffer->t_inf_end = t_inf_end;
                
                // Propagate camera telemetry
                results_buffer->cam_exposure_ms = input_image.cam_exposure_ms;
                results_buffer->cam_isp_latency_ms = input_image.cam_isp_latency_ms;
                results_buffer->cam_buffer_usage_percent = input_image.cam_buffer_usage_percent;
                results_buffer->tpu_temp_c = tpu_temperature_.load();
                
                // Propagate ImageProcessor telemetry
                results_buffer->image_proc_ms = input_image.image_proc_ms;

                // Log to unified CSV
                CsvLogEntry inf_entry;
                copy_to_array(inf_entry.module, "InferenceEngine");
                copy_to_array(inf_entry.event, "inference_complete");
                inf_entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                inf_entry.call_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                inf_entry.cam_frame_id = input_image.frame_id;
                inf_entry.tpu_inference_ms = static_cast<float>(duration_us) / 1000.0f;
                inf_entry.tpu_temp_c = results_buffer->tpu_temp_c;
                inf_entry.image_proc_ms = input_image.image_proc_ms;
                
                // Initialize TPU fields to non-NaN defaults
                inf_entry.tpu_model_score = 0.0f;
                inf_entry.tpu_class_id = 0;

                if (results_buffer->size > 0) {
                    inf_entry.tpu_model_score = results_buffer->data[0].score;
                    inf_entry.tpu_class_id = results_buffer->data[0].class_id;
                }
                Logger::getInstance().log_csv(inf_entry);

                if (detection_results_for_overlay_buffer_) {
                    DetectionResults& overlay_results = detection_results_for_overlay_buffer_->get_write_buffer();
                    if (results_buffer->size > 0) {
                        overlay_results.assign(results_buffer->data.data(), results_buffer->data.data() + results_buffer->size);
                    } else {
                        overlay_results.clear();
                    }
                    detection_results_for_overlay_buffer_->commit_write();
                }

                // Update timing information in the buffer before creating the token
                if (results_buffer) {
                    // Set timing information
                    results_buffer->t_inf_start = t_inf_start;
                    results_buffer->t_inf_end = t_inf_end;
                }
                
                // Create the lifecycle token from pool.
                ResultToken* token_ptr = result_token_pool_->acquire();
                if (token_ptr) {
                    *token_ptr = ResultToken(results_buffer);
                    
                    if (detection_results_for_logic_queue_.push(token_ptr)) {
                        guard.output_produced = true;
                        if (app_ref_) {
                            app_ref_->increment_inference_results_produced(1);
                        }
                    } else {
                        token_ptr->release_buffer();
                        result_token_pool_->release(token_ptr);
                    }
                }
            }
            
            // Periodic heartbeat
            frames_processed++;
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat_time).count() >= 5) {
                APP_LOG_INFO("InferenceEngine: Processed " + std::to_string(frames_processed) + " frames in last interval");
                last_heartbeat_time = now;
                frames_processed = 0;
            }
        }
    }
}

void InferenceEngine::set_input_tensor(tflite::Interpreter* interpreter, const ImageData& image) {
    if (!image.buffer) return;

    if (interpreter->inputs().size() == 0) return;

    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);

    if (!input_tensor) return;

    // MANDATE: No string constructions (std::to_string, etc.) or APP_LOG calls in this hot path.
    // Memory and logic invariants are checked by the compiler and initial setup.
    
    if (input_tensor->type != kTfLiteUInt8) {
        throw std::runtime_error("Input tensor type mismatch: expected UInt8.");
    }

    uint8_t* tensor_data = interpreter->typed_input_tensor<uint8_t>(0);
    if (!tensor_data) throw std::runtime_error("Input tensor data pointer is null.");

    // Validate buffer sizes (O(1) checks)
    if (image.buffer->size != input_tensor->bytes) {
        char err_buf[128];
        snprintf(err_buf, sizeof(err_buf), "Input tensor size mismatch: buffer=%zu, tensor=%zu", 
                 image.buffer->size, input_tensor->bytes);
        APP_LOG_ERROR(err_buf);
        throw std::runtime_error(err_buf);
    }
    
    std::memcpy(tensor_data, image.buffer->data.data(), image.buffer->size);
}

std::shared_ptr<DetectionResultBuffer> InferenceEngine::get_output_tensor(tflite::Interpreter* interpreter, const ImageData& input_image) {
    auto results_buffer = detection_result_pool_->acquire();
    if (!results_buffer) {
        APP_LOG_WARNING("Failed to acquire a detection result buffer from the pool. No results will be reported for this frame.");
        return nullptr;
    }
    std::cerr << "InferenceEngine: get_output_tensor() for frame " << input_image.frame_id << std::endl;
    results_buffer->size = 0; // Reset size

    // Manual cleanup function - call this on error paths before returning nullptr
    auto cleanup_on_error = [&results_buffer]() {
        // The shared_ptr will automatically return the buffer to the pool when it goes out of scope
        // So we just need to reset it to release the buffer immediately
        results_buffer.reset();
    };

    size_t num_outputs = interpreter->outputs().size();
    std::vector<DetectionResult> raw_detections;
    auto timestamp = std::chrono::steady_clock::now();

    // -------------------------------------------------------------------------
    // STRATEGY A: Standard SSD MobileNet (4 outputs)
    // -------------------------------------------------------------------------
    if (num_outputs == 4) {
        // Zero-copy access via typed_output_tensor
        const uint8_t* scores = interpreter->typed_output_tensor<uint8_t>(0);
        const uint8_t* boxes = interpreter->typed_output_tensor<uint8_t>(1);
        const uint8_t* num_det_uint8 = interpreter->typed_output_tensor<uint8_t>(2);
        const uint8_t* classes = interpreter->typed_output_tensor<uint8_t>(3);

        TfLiteTensor* detection_scores_tensor = interpreter->tensor(interpreter->outputs()[0]);
        TfLiteTensor* detection_boxes_tensor = interpreter->tensor(interpreter->outputs()[1]);
        TfLiteTensor* num_detections_tensor = interpreter->tensor(interpreter->outputs()[2]);
        TfLiteTensor* detection_classes_tensor = interpreter->tensor(interpreter->outputs()[3]);

        if (!scores || !boxes || !num_det_uint8 || !classes) {
            APP_LOG_ERROR("One or more output tensors are null.");
            cleanup_on_error();
            return nullptr;
        }

        // 1. Dequantize num_detections (Count)
        float num_detections_val = 0.0f;
        if (num_detections_tensor->quantization.type == kTfLiteAffineQuantization) {
            const TfLiteAffineQuantization* quant = reinterpret_cast<const TfLiteAffineQuantization*>(num_detections_tensor->quantization.params);
            float scale = (quant && quant->scale && quant->scale->size > 0) ? quant->scale->data[0] : 1.0f;
            int zero_point = (quant && quant->zero_point && quant->zero_point->size > 0) ? quant->zero_point->data[0] : 0;
            num_detections_val = (static_cast<float>(num_det_uint8[0]) - zero_point) * scale;
        } else {
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
            if (scale == 0.0f) scale = 1.0f;
        };

        float box_scale, class_scale, score_scale;
        int box_zp, class_zp, score_zp;
        get_robust_quant(detection_boxes_tensor, box_scale, box_zp);
        get_robust_quant(detection_classes_tensor, class_scale, class_zp);
        get_robust_quant(detection_scores_tensor, score_scale, score_zp);

        for (int i = 0; i < num_detections; ++i) {
            float score = (static_cast<float>(scores[i]) - score_zp) * score_scale;
            if (score > score_threshold_) {
                float class_val = (static_cast<float>(classes[i]) - class_zp) * class_scale;
                
                DetectionResult res;
                res.class_id = static_cast<int>(std::round(class_val)) + 1; // 0-based -> 1-based
                res.score = score;
                res.source_frame_id = input_image.frame_id;
                res.timestamp = timestamp;
                res.t_capture_raw_ms = input_image.t_capture_raw_ms;

                // Propagate camera telemetry from ImageData
                res.cam_exposure_ms = input_image.cam_exposure_ms;
                res.cam_isp_latency_ms = input_image.cam_isp_latency_ms;
                res.cam_buffer_usage_percent = input_image.cam_buffer_usage_percent;
                
                // Add TPU telemetry
                res.tpu_temp_c = get_tpu_temperature();
                
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
    
                // Zero-copy access
                const uint8_t* data = interpreter->typed_output_tensor<uint8_t>(i);
                if (!data) continue;
    
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
    
                    // Use configurable threshold from config
                    if (score < score_threshold_) continue;
    
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
                    
                    raw_detections.push_back(res);
                }
            }
        } 
     else {
        APP_LOG_ERROR("Unsupported output count: " + std::to_string(num_outputs));
        cleanup_on_error();
        return nullptr;
    }

    // -------------------------------------------------------------------------
    // GLOBAL NMS (Unified across all classes)
    // -------------------------------------------------------------------------
    // Sort by score descending
    std::sort(raw_detections.begin(), raw_detections.end(), [](const DetectionResult& a, const DetectionResult& b) {
        return a.score > b.score;
    });

    if (!raw_detections.empty()) {
        char raw_log[128];
        snprintf(raw_log, sizeof(raw_log), "InferenceEngine: Raw detections before NMS: %zu, max_score: %.2f", 
                 raw_detections.size(), raw_detections[0].score);
        APP_LOG_DEBUG(raw_log);
    }

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

    for (size_t i = 0; i < raw_detections.size(); ++i) {
        if (is_suppressed[i]) continue;
        filtered_detections.push_back(raw_detections[i]);
        
        for (size_t j = i + 1; j < raw_detections.size(); ++j) {
            if (is_suppressed[j]) continue;
            
            float iou = calculate_iou(raw_detections[i], raw_detections[j]);
            if (iou > 0.45f) { // IoU Threshold
                is_suppressed[j] = true;
            }
        }
    }

    // --- Step 3: Post-NMS instrumentation (AI truth) ---
    // Log: nms_box_count, min / max confidence, first box (x_min, y_min, x_max, y_max)
    char nms_log_buffer[256];
    float min_conf = 1.0f;
    float max_conf = 0.0f;
    if (!filtered_detections.empty()) {
        for (const auto& det : filtered_detections) {
            min_conf = std::min(min_conf, det.score);
            max_conf = std::max(max_conf, det.score);
        }
        const auto& first_box = filtered_detections[0];
        snprintf(nms_log_buffer, sizeof(nms_log_buffer),
                 "nms_box_count=%zu, min_confidence=%.2f, max_confidence=%.2f, first_box=(%.2f,%.2f,%.2f,%.2f)",
                 filtered_detections.size(), min_conf, max_conf,
                 first_box.xmin, first_box.ymin, first_box.xmax, first_box.ymax);
        APP_LOG_INFO(nms_log_buffer);
    } else {
        APP_LOG_INFO("NMS OUTPUT EMPTY");
    }

    // Fill buffer
    size_t result_count = 0;
    std::cerr << "InferenceEngine: filtered_detections.size()=" << filtered_detections.size() << " for frame " << input_image.frame_id << std::endl;
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