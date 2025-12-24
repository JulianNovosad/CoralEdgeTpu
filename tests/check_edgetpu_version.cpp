#include "edgetpu_c.h"
#include <iostream>

int main() {
    const char* version = edgetpu_version();
    if (version) {
        std::cout << "Edge TPU Runtime Version: " << version << std::endl;
    } else {
        std::cout << "Failed to get Edge TPU runtime version" << std::endl;
    }
    return 0;
}