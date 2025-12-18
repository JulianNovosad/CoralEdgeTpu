#include <iostream>
#include "edgetpu_c.h"

int main() {
    // List Edge TPU devices
    size_t num_devices;
    edgetpu_device* devices = edgetpu_list_devices(&num_devices);
    
    std::cout << "Found " << num_devices << " Edge TPU devices." << std::endl;
    
    if (num_devices == 0) {
        std::cout << "No Edge TPU devices found." << std::endl;
        return 1;
    }
    
    // Print device information
    for (size_t i = 0; i < num_devices; ++i) {
        std::cout << "Device " << i << ":" << std::endl;
        std::cout << "  Type: " << devices[i].type << std::endl;
        std::cout << "  Path: " << devices[i].path << std::endl;
    }
    
    // Try to create delegate with the first device
    const auto& device = devices[0];
    std::cout << "Attempting to create delegate for device: " << device.path << std::endl;
    
    TfLiteDelegate* delegate = edgetpu_create_delegate(
        device.type, 
        device.path,
        nullptr, 
        0);
    
    if (delegate) {
        std::cout << "Successfully created Edge TPU delegate!" << std::endl;
        edgetpu_free_delegate(delegate);
    } else {
        std::cout << "Failed to create Edge TPU delegate." << std::endl;
        
        // Try with USB type explicitly
        std::cout << "Trying with USB device type..." << std::endl;
        delegate = edgetpu_create_delegate(
            EDGETPU_APEX_USB, 
            device.path,
            nullptr, 
            0);
            
        if (delegate) {
            std::cout << "Successfully created Edge TPU delegate with USB type!" << std::endl;
            edgetpu_free_delegate(delegate);
        } else {
            std::cout << "Failed to create Edge TPU delegate with USB type." << std::endl;
        }
    }
    
    // Free devices
    edgetpu_free_devices(devices);
    
    return 0;
}