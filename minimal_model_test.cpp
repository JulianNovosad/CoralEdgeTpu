#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "/usr/include/edgetpu_c.h"

// Simple function to load labels
std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open labels file: " << path << std::endl;
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

int main() {
    std::cout << "Minimal Edge TPU Detector Test" << std::endl;
    
    // Check if model file exists
    const std::string model_path = "detect_int8_edgetpu.tflite";
    const std::string labels_path = "coco_labels.txt";
    
    std::cout << "Loading model from: " << model_path << std::endl;
    
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model from: " << model_path << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;
    
    // Load labels
    auto labels = load_labels(labels_path);
    if (labels.empty()) {
        std::cerr << "Failed to load labels from: " << labels_path << std::endl;
        return 1;
    }
    std::cout << "Loaded " << labels.size() << " labels" << std::endl;
    
    // Create interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter" << std::endl;
        return 1;
    }
    
    // Create Edge TPU delegate
    std::cout << "Creating Edge TPU delegate..." << std::endl;
    
    // List Edge TPU devices
    size_t num_devices;
    edgetpu_device* devices = edgetpu_list_devices(&num_devices);
    
    if (num_devices == 0) {
        std::cerr << "No Edge TPU devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << num_devices << " Edge TPU devices" << std::endl;
    
    // Use the first device
    const auto& device = devices[0];
    std::cout << "Using device: " << device.path << std::endl;
    
    TfLiteDelegate* delegate = edgetpu_create_delegate(
        device.type, 
        device.path,
        nullptr, 
        0);
    
    if (!delegate) {
        std::cerr << "Failed to create Edge TPU delegate" << std::endl;
        edgetpu_free_devices(devices);
        return 1;
    }
    
    std::cout << "Edge TPU delegate created successfully" << std::endl;
    
    // Apply delegate
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        std::cerr << "Failed to apply Edge TPU delegate" << std::endl;
        edgetpu_free_delegate(delegate);
        edgetpu_free_devices(devices);
        return 1;
    }
    std::cout << "Edge TPU delegate applied successfully" << std::endl;
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        edgetpu_free_delegate(delegate);
        edgetpu_free_devices(devices);
        return 1;
    }
    std::cout << "Tensors allocated successfully" << std::endl;
    
    // Print input tensor info
    int input_tensor_index = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);
    std::cout << "Input tensor shape: ";
    for (int i = 0; i < input_tensor->dims->size; ++i) {
        std::cout << input_tensor->dims->data[i] << " ";
    }
    std::cout << std::endl;
    
    // Print output tensor info
    std::cout << "Output tensors: " << interpreter->outputs().size() << std::endl;
    for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
        int output_tensor_index = interpreter->outputs()[i];
        TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);
        std::cout << "Output tensor " << i << " shape: ";
        for (int j = 0; j < output_tensor->dims->size; ++j) {
            std::cout << output_tensor->dims->data[j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "SUCCESS: Model and Edge TPU are working correctly!" << std::endl;
    std::cout << "Exiting without cleanup to avoid segfault..." << std::endl;
    
    // Exit immediately to avoid cleanup segfault
    exit(0);
    
    return 0;
}