#include <iostream>
#include <dlfcn.h> // For dlopen, dlsym, dlclose
#include <string>

// Define the function signature we expect to find in the library
typedef void* (*tflite_plugin_create_delegate_t)(char**, char**, size_t, void (*)(const char*));
typedef void (*tflite_plugin_destroy_delegate_t)(void*);

int main() {
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
    // We pass nullptr for options and a simple error reporter
    void (*simple_error_reporter)(const char*) = [](const char* msg){ std::cerr << "Simple Error: " << msg << std::endl; };
    void* delegate = create_delegate_func(nullptr, nullptr, 0, simple_error_reporter);

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
