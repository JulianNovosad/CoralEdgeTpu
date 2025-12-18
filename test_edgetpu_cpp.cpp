#include <iostream>
#include "edgetpu.h"

int main() {
    // Get the Edge TPU manager
    auto* manager = edgetpu::EdgeTpuManager::GetSingleton();
    if (!manager) {
        std::cout << "Failed to get Edge TPU manager." << std::endl;
        return 1;
    }
    
    // Enumerate devices
    auto devices = manager->EnumerateEdgeTpu();
    std::cout << "Found " << devices.size() << " Edge TPU devices." << std::endl;
    
    if (devices.empty()) {
        std::cout << "No Edge TPU devices found." << std::endl;
        return 1;
    }
    
    // Print device information
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "Device " << i << ":" << std::endl;
        std::cout << "  Type: " << static_cast<int>(devices[i].type) << std::endl;
        std::cout << "  Path: " << devices[i].path << std::endl;
    }
    
    // Try to open the first device
    auto context = manager->OpenDevice();
    if (context) {
        std::cout << "Successfully opened Edge TPU device!" << std::endl;
        std::cout << "Device version: " << manager->Version() << std::endl;
    } else {
        std::cout << "Failed to open Edge TPU device." << std::endl;
    }
    
    return 0;
}