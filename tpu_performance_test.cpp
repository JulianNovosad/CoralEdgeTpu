#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "edgetpu_c.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <cstring>

int main() {
    const std::string model_path = "/home/pi/CoralEdgeTpu/detect_int8_edgetpu.tflite";
    
    // Load model
    std::cout << "Loading model: " << model_path << std::endl;
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    // List Edge TPU devices
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
    std::cout << "Found " << num_devices << " Edge TPU devices" << std::endl;
    
    if (num_devices == 0) {
        std::cerr << "No Edge TPU devices found" << std::endl;
        return -1;
    }
    
    // Create delegate
    TfLiteDelegate* edgetpu_delegate = edgetpu_create_delegate(
        devices.get()[0].type, 
        devices.get()[0].path,
        nullptr, 
        0);
        
    if (!edgetpu_delegate) {
        std::cerr << "Failed to create Edge TPU delegate" << std::endl;
        return -1;
    }
    std::cout << "Edge TPU delegate created successfully" << std::endl;

    // Create interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to build interpreter" << std::endl;
        edgetpu_free_delegate(edgetpu_delegate);
        return -1;
    }

    // Apply delegate
    if (interpreter->ModifyGraphWithDelegate(edgetpu_delegate) != kTfLiteOk) {
        std::cerr << "Failed to apply EdgeTPU delegate" << std::endl;
        edgetpu_free_delegate(edgetpu_delegate);
        return -1;
    }

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        edgetpu_free_delegate(edgetpu_delegate);
        return -1;
    }

    // Get input tensor info
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);
    std::cout << "Input tensor dimensions: ";
    for (int i = 0; i < input_tensor->dims->size; i++) {
        std::cout << input_tensor->dims->data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Input tensor type: " << input_tensor->type << std::endl;
    std::cout << "Input tensor bytes: " << input_tensor->bytes << std::endl;

    // Prepare dummy input data
    std::vector<uint8_t> dummy_input(input_tensor->bytes, 0);
    
    // Run warmup inference
    std::cout << "Running warmup inference..." << std::endl;
    uint8_t* input_data = interpreter->typed_input_tensor<uint8_t>(0);
    std::memcpy(input_data, dummy_input.data(), input_tensor->bytes);
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Warmup inference failed" << std::endl;
        edgetpu_free_delegate(edgetpu_delegate);
        return -1;
    }
    std::cout << "Warmup inference completed successfully" << std::endl;

    // Run performance test
    const int num_iterations = 100;
    std::cout << "Running " << num_iterations << " inferences for performance measurement..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        std::memcpy(input_data, dummy_input.data(), input_tensor->bytes);
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Inference " << i << " failed" << std::endl;
            edgetpu_free_delegate(edgetpu_delegate);
            return -1;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    auto avg_duration_us = total_duration.count() / num_iterations;
    auto avg_duration_ms = avg_duration_us / 1000.0;
    auto fps = 1000.0 / avg_duration_ms;
    
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Total time for " << num_iterations << " inferences: " << total_duration.count() << " microseconds" << std::endl;
    std::cout << "  Average time per inference: " << avg_duration_us << " microseconds (" << avg_duration_ms << " ms)" << std::endl;
    std::cout << "  Estimated FPS: " << fps << std::endl;

    // Cleanup
    edgetpu_free_delegate(edgetpu_delegate);
    
    return 0;
}