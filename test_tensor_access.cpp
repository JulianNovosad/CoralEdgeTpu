#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "/usr/include/edgetpu_c.h"
#include <opencv2/opencv.hpp>

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

// Helper function to print tensor details
void print_tensor_details(TfLiteTensor* tensor, const std::string& name, int index) {
    std::cout << "Tensor " << name << " (index: " << index << ")" << std::endl;
    std::cout << "  Shape: ";
    for (int i = 0; i < tensor->dims->size; ++i) {
        std::cout << tensor->dims->data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "  Type: " << tensor->type << " ";
    switch (tensor->type) {
        case kTfLiteFloat32:
            std::cout << "(FLOAT32)";
            break;
        case kTfLiteUInt8:
            std::cout << "(UINT8)";
            break;
        case kTfLiteInt8:
            std::cout << "(INT8)";
            break;
        default:
            std::cout << "(OTHER)";
            break;
    }
    std::cout << std::endl;
    
    std::cout << "  Bytes: " << tensor->bytes << std::endl;
    std::cout << "  Raw data ptr: " << tensor->data.raw << std::endl;
    
    // Quantization parameters
    if (tensor->quantization.type == kTfLiteAffineQuantization) {
        std::cout << "  Quantization: AFFINE" << std::endl;
        // Note: Detailed quantization params require accessing the quantization struct
    } else if (tensor->quantization.type == kTfLiteNoQuantization) {
        std::cout << "  Quantization: NONE" << std::endl;
    } else {
        std::cout << "  Quantization: UNKNOWN (" << tensor->quantization.type << ")" << std::endl;
    }
    
    // Print first few raw values
    if (tensor->data.raw && tensor->bytes > 0) {
        std::cout << "  Raw data accessible" << std::endl;
        int total_elements = 1;
        for (int i = 0; i < tensor->dims->size; ++i) {
            total_elements *= tensor->dims->data[i];
        }
        
        std::cout << "  Total elements: " << total_elements << std::endl;
        std::cout << "  First 10 values: ";
        
        // Print based on type
        if (tensor->type == kTfLiteFloat32) {
            const float* data = reinterpret_cast<const float*>(tensor->data.raw);
            for (int i = 0; i < std::min(10, total_elements); ++i) {
                std::cout << data[i] << " ";
            }
        } else if (tensor->type == kTfLiteUInt8) {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(tensor->data.raw);
            for (int i = 0; i < std::min(10, total_elements); ++i) {
                std::cout << static_cast<int>(data[i]) << " ";
            }
        } else if (tensor->type == kTfLiteInt8) {
            const int8_t* data = reinterpret_cast<const int8_t*>(tensor->data.raw);
            for (int i = 0; i < std::min(10, total_elements); ++i) {
                std::cout << static_cast<int>(data[i]) << " ";
            }
        } else {
            std::cout << "UNKNOWN TYPE";
        }
        std::cout << std::endl;
    } else {
        std::cout << "  Raw data NOT accessible" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== TFLite Model Output Tensor Analysis ===" << std::endl;
    
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
    std::cout << "\n=== Input Tensor ===" << std::endl;
    print_tensor_details(input_tensor, "Input", input_tensor_index);
    
    // Print output tensor info
    std::cout << "=== Output Tensors ===" << std::endl;
    std::cout << "Number of output tensors: " << interpreter->outputs().size() << std::endl;
    
    std::vector<std::pair<int, TfLiteTensor*>> output_tensors;
    for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
        int output_tensor_index = interpreter->outputs()[i];
        TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);
        output_tensors.push_back({static_cast<int>(i), output_tensor});
        
        std::string name = "Output_" + std::to_string(i);
        print_tensor_details(output_tensor, name, output_tensor_index);
    }
    
    // Now let's run inference with a dummy input to see actual values
    std::cout << "=== Running Inference with Dummy Input ===" << std::endl;
    
    // Fill input tensor with zeros (more realistic dummy data)
    TfLiteTensor* input_tensor_info = interpreter->tensor(interpreter->inputs()[0]);
    if (input_tensor_info && input_tensor_info->data.raw) {
        if (input_tensor_info->type == kTfLiteFloat32) {
            size_t input_size = input_tensor_info->bytes / sizeof(float);
            float* input_data = reinterpret_cast<float*>(input_tensor_info->data.raw);
            // Fill with small random-like values instead of all zeros
            for (size_t i = 0; i < input_size; ++i) {
                input_data[i] = (static_cast<float>(i % 255) / 255.0f) * 2.0f - 1.0f; // Range [-1, 1]
            }
            std::cout << "Input tensor filled with dummy data. Size: " << input_size << " floats" << std::endl;
        } else {
            std::cout << "Input tensor type not supported for dummy filling" << std::endl;
        }
    } else {
        std::cout << "Failed to access input tensor for dummy data" << std::endl;
    }
    
    // Run inference
    if (interpreter->Invoke() == kTfLiteOk) {
        std::cout << "Inference successful" << std::endl;
        
        // Print output tensor info after inference
        std::cout << "\n=== Output Tensors After Inference ===" << std::endl;
        for (const auto& pair : output_tensors) {
            int i = pair.first;
            TfLiteTensor* output_tensor = pair.second;
            std::string name = "Output_" + std::to_string(i) + "_after_inference";
            print_tensor_details(output_tensor, name, interpreter->outputs()[i]);
        }
    } else {
        std::cout << "Inference failed" << std::endl;
    }
    
    std::cout << "=== Analysis Complete ===" << std::endl;
    
    return 0;
}