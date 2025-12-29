/**
 * @file metadata_audit.cpp
 * @brief Forensic tool for inspecting TFLite model output quantization parameters.
 * 
 * --- HISTORICAL CONTEXT: THE "SCORE=12.0" INCIDENT ---
 * During the Aurore Mk V deployment, the system reported detection scores of 12.0+ 
 * on a 0.0-1.0 scale. Forensic analysis using this tool revealed that the 
 * TFLite runtime was occasionally failing to provide modern 'TfLiteAffineQuantization' 
 * metadata, causing the application to misinterpret raw uint8 bytes (e.g., raw value 12) 
 * as literal float scores.
 * 
 * Additionally, if the 'Count' tensor (Output 2) failed to de-quantize, a raw 255 
 * (representing 10 detections) was interpreted as 255 detections, leading to 
 * out-of-bounds reads and memory corruption.
 * 
 * --- QUANTIZATION MODES ---
 * 1. Affine Quantization (Modern): Uses tensor->quantization.params (scale/zero_point arrays).
 * 2. Legacy Params: Uses tensor->params.scale and tensor->params.zero_point (scalars).
 * 
 * This tool programmatically queries both to ensure the application logic matches 
 * the model's physical layout.
 * 
 * --- USAGE ---
 * Build: make metadata_audit
 * Run: ./metadata_audit
 */

#include <iostream>
#include <vector>
#include <string>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "edgetpu.h"

int main() {
    const std::string model_path = "/home/pi/CoralEdgeTpu/detect_int8_edgetpu.tflite";
    
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to build interpreter"
;
        return 1;
    }

    std::cout << "FORENSIC METADATA AUDIT: " << model_path << "\n";
    std::cout << "------------------------------------------\n";

    auto print_tensor_info = [&](int index, const char* label) {
        TfLiteTensor* tensor = interpreter->tensor(index);
        if (!tensor) return;

        std::cout << label << " (Index: " << index << "):\n";
        std::cout << "  Name: " << tensor->name << "\n";
        std::cout << "  Type: ";
        switch (tensor->type) {
            case kTfLiteUInt8: std::cout << "kTfLiteUInt8"; break;
            case kTfLiteInt8:  std::cout << "kTfLiteInt8"; break;
            case kTfLiteFloat32: std::cout << "kTfLiteFloat32"; break;
            default: std::cout << "Unknown (" << tensor->type << ")"; break;
        }
        std::cout << "\n";

        std::cout << "  Params Scale: " << tensor->params.scale << "\n";
        std::cout << "  Params Zero Point: " << tensor->params.zero_point << "\n";

        if (tensor->quantization.type == kTfLiteAffineQuantization) {
            auto* params = reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
            if (params && params->scale && params->scale->size > 0) {
                std::cout << "  Quant Scale: " << params->scale->data[0] << "\n";
                std::cout << "  Quant Zero Point: " << params->zero_point->data[0] << "\n";
            } else {
                std::cout << "  Quantization: Affine (but no scale/zp data)" << "\n";
            }
        } else {
            std::cout << "  Quantization Type: " << tensor->quantization.type << " (Not Affine)" << "\n";
        }
        std::cout << "  Dimensions: [";
        for (int i = 0; i < tensor->dims->size; ++i) {
            std::cout << tensor->dims->data[i] << (i == tensor->dims->size - 1 ? "" : ", ");
        }
        std::cout << "]\n\n";
    };

    std::cout << "OUTPUT TENSOR INDICES: ";
    for (int i : interpreter->outputs()) {
        std::cout << i << " ";
    }
    std::cout << "\n\n";

    std::cout << "OUTPUT TENSOR DETAILS:\n";
    int output_idx = 0;
    for (int i : interpreter->outputs()) {
        std::string label = "Output " + std::to_string(output_idx++);
        print_tensor_info(i, label.c_str());
    }

    return 0;
}
