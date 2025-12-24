#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_file.tflite>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    
    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = 
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    
    if (!model) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded successfully from " << model_path << std::endl;
    
    // Create an interpreter builder
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter" << std::endl;
        return 1;
    }
    
    // Print model information
    std::cout << "\n=== MODEL INFORMATION ===" << std::endl;
    std::cout << "Number of input tensors: " << interpreter->inputs().size() << std::endl;
    std::cout << "Number of output tensors: " << interpreter->outputs().size() << std::endl;
    
    // Print input tensor information
    std::cout << "\n=== INPUT TENSORS ===" << std::endl;
    for (size_t i = 0; i < interpreter->inputs().size(); ++i) {
        int tensor_index = interpreter->inputs()[i];
        TfLiteTensor* tensor = interpreter->tensor(tensor_index);
        
        std::cout << "Input #" << i << ":" << std::endl;
        std::cout << "  Index: " << tensor_index << std::endl;
        std::cout << "  Name: " << (tensor->name ? tensor->name : "(null)") << std::endl;
        std::cout << "  Type: " << tensor->type << std::endl;
        std::cout << "  Dimensions: " << tensor->dims->size << "D" << std::endl;
        
        for (int j = 0; j < tensor->dims->size; ++j) {
            std::cout << "    Dim " << j << ": " << tensor->dims->data[j] << std::endl;
        }
    }
    
    // Print output tensor information
    std::cout << "\n=== OUTPUT TENSORS ===" << std::endl;
    for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
        int tensor_index = interpreter->outputs()[i];
        TfLiteTensor* tensor = interpreter->tensor(tensor_index);
        
        std::cout << "Output #" << i << ":" << std::endl;
        std::cout << "  Index: " << tensor_index << std::endl;
        std::cout << "  Name: " << (tensor->name ? tensor->name : "(null)") << std::endl;
        std::cout << "  Type: " << tensor->type << std::endl;
        std::cout << "  Dimensions: " << tensor->dims->size << "D" << std::endl;
        
        for (int j = 0; j < tensor->dims->size; ++j) {
            std::cout << "    Dim " << j << ": " << tensor->dims->data[j] << std::endl;
        }
    }
    
    // Print all tensor information
    std::cout << "\n=== ALL TENSORS ===" << std::endl;
    int total_tensors = interpreter->tensors_size();
    std::cout << "Total number of tensors: " << total_tensors << std::endl;
    
    for (int i = 0; i < total_tensors; ++i) {
        TfLiteTensor* tensor = interpreter->tensor(i);
        if (tensor) {
            std::cout << "Tensor #" << i << ": " << (tensor->name ? tensor->name : "(null)") << std::endl;
        }
    }
    
    return 0;
}