#include "inference.h"
#include "config_loader.h"
#include "util_logging.h"
#include "buffer_pool.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "edgetpu.h"

// Simple test to measure raw TPU inference time
int main() {
    std::cout << "Testing raw TPU inference time..." << std::endl;
    
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("/home/pi/CoralEdgeTpu/detect_int8_edgetpu.tflite");
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded successfully" << std::endl;
    
    // Create interpreter builder
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return 1;
    }
    
    // Create Edge TPU delegate
    auto* delegate = edgetpu::CreateEdgeTpuDelegate();
    if (!delegate) {
        std::cerr << "Failed to create Edge TPU delegate" << std::endl;
        return 1;
    }
    
    // Apply delegate to interpreter
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        std::cerr << "Failed to apply Edge TPU delegate" << std::endl;
        edgetpu::FreeEdgeTpuDelegate(delegate);
        return 1;
    }
    
    std::cout << "Edge TPU delegate applied successfully" << std::endl;
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        edgetpu::FreeEdgeTpuDelegate(delegate);
        return 1;
    }
    
    std::cout << "Tensors allocated successfully" << std::endl;
    
    // Get input tensor info
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);
    
    std::cout << "Input tensor dimensions: " << input_tensor->dims->data[1] << "x" << input_tensor->dims->data[2] << "x" << input_tensor->dims->data[3] << std::endl;
    
    // Fill input tensor with dummy data
    uint8_t* input_data = interpreter->typed_input_tensor<uint8_t>(0);
    size_t input_size = 320 * 320 * 3; // RGB888
    
    // Initialize with dummy data
    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = static_cast<uint8_t>(i % 256);
    }
    
    // Warm up
    for (int i = 0; i < 5; i++) {
        interpreter->Invoke();
    }
    
    // Measure inference time
    const int iterations = 100;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        interpreter->Invoke();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    double avg_time_ms = (double)duration / iterations / 1000.0;
    double fps = 1000.0 / avg_time_ms;
    
    std::cout << "Average inference time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Theoretical FPS: " << fps << std::endl;
    
    // Clean up
    edgetpu::FreeEdgeTpuDelegate(delegate);
    
    return 0;
}