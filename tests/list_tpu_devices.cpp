#include "edgetpu_c.h"
#include <iostream>
#include <memory>

int main() {
    size_t num_devices = 0;
    edgetpu_device* devices_ptr = edgetpu_list_devices(&num_devices);
    
    std::cout << "Found " << num_devices << " Edge TPU devices:" << std::endl;
    
    for (size_t i = 0; i < num_devices; i++) {
        const auto& device = devices_ptr[i];
        std::cout << "Device " << i << ":" << std::endl;
        std::cout << "  Path: " << device.path << std::endl;
        std::cout << "  Type: " << device.type << std::endl;
        std::cout << "  Type Name: " << (device.type == EDGETPU_APEX_PCI ? "PCI" : "USB") << std::endl;
    }
    
    edgetpu_free_devices(devices_ptr);
    
    return 0;
}