#include <iostream>
#include <dlfcn.h> // For dlopen, dlsym, dlclose
#include <string>
#include <array>
#include <cstdio> // For popen, pclose
#include <memory> // Required for std::unique_ptr

// Define the function signature we expect to find in the library
typedef void* (*tflite_plugin_create_delegate_t)(char**, char**, size_t, void (*)(const char*));
typedef void (*tflite_plugin_destroy_delegate_t)(void*);

// Custom error reporting function for the Edge TPU delegate
void edgetpu_error_reporter(const char* msg) {
    std::cerr << "Edge TPU Delegate Error: " << msg << std::endl;
}

// Function to check if /dev/apex_0 is in use
bool is_edgetpu_in_use() {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("fuser /dev/apex_0 2>&1", "r"), pclose);
    if (!pipe) {
        std::cerr << "Error: popen failed for fuser command." << std::endl;
        return false; // Assume not in use if we can't check
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    // If fuser finds a process, it will return an exit code of 0 and print output.
    // If no process is found, it returns non-zero and prints nothing or an error.
    // We check if the result string contains any PID.
    return result.find("/dev/apex_0") != std::string::npos;
}

int main() {
    // Check if Edge TPU is in use by another process
    if (is_edgetpu_in_use()) {
        std::cerr << "Error: /dev/apex_0 (Edge TPU) is currently in use by another process." << std::endl;
        std::cerr << "Please ensure no other applications or tests are using the Edge TPU and try again." << std::endl;
        return 1;
    }

    std::cout << "Attempting to dynamically load libedgetpu.so.1.0..." << std::endl;

    // Path to libedgetpu.so.1.0 (assuming it's in /usr/lib/aarch64-linux-gnu/)
    const std::string lib_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0";

    // 1. Load the shared library
    void* handle = dlopen(lib_path.c_str(), RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library " << lib_path << ": " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "Library " << lib_path << " loaded successfully." << std::endl;

    // 2. Get a pointer to the tflite_plugin_create_delegate function
    tflite_plugin_create_delegate_t create_delegate_func = 
        (tflite_plugin_create_delegate_t)dlsym(handle, "tflite_plugin_create_delegate");
    if (!create_delegate_func) {
        std::cerr << "Failed to find symbol tflite_plugin_create_delegate: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "Symbol tflite_plugin_create_delegate found successfully." << std::endl;

    // 3. Get a pointer to the tflite_plugin_destroy_delegate function
    tflite_plugin_destroy_delegate_t destroy_delegate_func = 
        (tflite_plugin_destroy_delegate_t)dlsym(handle, "tflite_plugin_destroy_delegate");
    if (!destroy_delegate_func) {
        std::cerr << "Failed to find symbol tflite_plugin_destroy_delegate: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "Symbol tflite_plugin_destroy_delegate found successfully." << std::endl;

    // 4. Try to create a delegate using the loaded function
    std::cout << "Attempting to create delegate using dlsym'd function..." << std::endl;
    void* delegate = create_delegate_func(nullptr, nullptr, 0, edgetpu_error_reporter);

    if (!delegate) {
        std::cerr << "Failed to create delegate using dlsym'd function." << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "Delegate created successfully using dlsym'd function." << std::endl;

    // 5. Destroy the delegate
    destroy_delegate_func(delegate);
    std::cout << "Delegate destroyed successfully using dlsym'd function." << std::endl;

    // 6. Close the library
    dlclose(handle);
    std::cout << "Library " << lib_path << " closed successfully." << std::endl;

    return 0;
}