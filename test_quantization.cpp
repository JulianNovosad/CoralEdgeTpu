#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/c/common.h"
#include "edgetpu_c.h"
#include <iostream>
#include <memory>

int main() {
    // Print the size of TfLiteAffineQuantization to understand its structure
    std::cout << "Size of TfLiteAffineQuantization: " << sizeof(TfLiteAffineQuantization) << std::endl;
    
    // Print the size of TfLiteQuantization to understand its structure
    std::cout << "Size of TfLiteQuantization: " << sizeof(TfLiteQuantization) << std::endl;
    
    return 0;
}