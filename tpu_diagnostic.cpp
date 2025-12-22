#include <iostream>
#include <vector>
#include <memory>
#include "edgetpu_c.h"
#include <errno.h>
#include <string.h>

void print_device_info(const edgetpu_device* device, int index) {
    std::cout << "Device " << index << ":" << std::endl;
    std::cout << "  Path: " << device->path << std::endl;
    std::cout << "  Type: " << device->type << std::endl;
    
    switch(device->type) {
        case EDGETPU_APEX_PCI:
            std::cout << "  Type Name: EDGETPU_APEX_PCI" << std::endl;
            break;
        case EDGETPU_APEX_USB:
            std::cout << "  Type Name: EDGETPU_APEX_USB" << std::endl;
            break;
        default:
            std::cout << "  Type Name: UNKNOWN" << std::endl;
            break;
    }
}

int main() {
    std::cout << "=== Edge TPU Diagnostic Tool ===" << std::endl;
    
    // Step 1: List devices
    std::cout << "\n1. Listing Edge TPU devices..." << std::endl;
    size_t num_devices = 0;
    edgetpu_device* devices = edgetpu_list_devices(&num_devices);
    
    std::cout << "Found " << num_devices << " Edge TPU devices." << std::endl;
    
    if (num_devices == 0) {
        std::cout << "ERROR: No Edge TPU devices found!" << std::endl;
        return 1;
    }
    
    // Print device information
    for (size_t i = 0; i < num_devices; i++) {
        print_device_info(&devices[i], i);
    }
    
    // Step 2: Try to create delegate with different approaches
    edgetpu_device* device = &devices[0];
    std::cout << "\n2. Testing delegate creation..." << std::endl;
    
    // Approach 1: With verbose options
    std::cout << "\n2a. Creating delegate with verbose options..." << std::endl;
    std::vector<edgetpu_option> options;
    options.push_back({"verbose", "1"});
    
    TfLiteDelegate* delegate1 = edgetpu_create_delegate(
        device->type, 
        device->path,
        options.data(), 
        options.size());
    
    if (delegate1) {
        std::cout << "SUCCESS: Delegate created with verbose options." << std::endl;
        edgetpu_free_delegate(delegate1);
    } else {
        std::cout << "FAILED: Delegate creation with verbose options failed." << std::endl;
        int err = errno;
        std::cout << "  Errno: " << err << " (" << strerror(err) << ")" << std::endl;
    }
    
    // Approach 2: Without options
    std::cout << "\n2b. Creating delegate without options..." << std::endl;
    TfLiteDelegate* delegate2 = edgetpu_create_delegate(
        device->type, 
        device->path,
        nullptr, 
        0);
    
    if (delegate2) {
        std::cout << "SUCCESS: Delegate created without options." << std::endl;
        edgetpu_free_delegate(delegate2);
    } else {
        std::cout << "FAILED: Delegate creation without options failed." << std::endl;
        int err = errno;
        std::cout << "  Errno: " << err << " (" << strerror(err) << ")" << std::endl;
    }
    
    // Approach 3: With USB type explicitly
    std::cout << "\n2c. Creating delegate with USB device type..." << std::endl;
    TfLiteDelegate* delegate3 = edgetpu_create_delegate(
        EDGETPU_APEX_USB, 
        device->path,
        nullptr, 
        0);
    
    if (delegate3) {
        std::cout << "SUCCESS: Delegate created with USB device type." << std::endl;
        edgetpu_free_delegate(delegate3);
    } else {
        std::cout << "FAILED: Delegate creation with USB device type failed." << std::endl;
        int err = errno;
        std::cout << "  Errno: " << err << " (" << strerror(err) << ")" << std::endl;
    }
    
    // Approach 4: With PCI type explicitly
    std::cout << "\n2d. Creating delegate with PCI device type..." << std::endl;
    TfLiteDelegate* delegate4 = edgetpu_create_delegate(
        EDGETPU_APEX_PCI, 
        device->path,
        nullptr, 
        0);
    
    if (delegate4) {
        std::cout << "SUCCESS: Delegate created with PCI device type." << std::endl;
        edgetpu_free_delegate(delegate4);
    } else {
        std::cout << "FAILED: Delegate creation with PCI device type failed." << std::endl;
        int err = errno;
        std::cout << "  Errno: " << err << " (" << strerror(err) << ")" << std::endl;
    }
    
    // Free devices
    edgetpu_free_devices(devices);
    
    std::cout << "\n=== Diagnostic Complete ===" << std::endl;
    return 0;
}