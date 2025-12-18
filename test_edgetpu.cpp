#include <iostream>
#include <dlfcn.h>
#include <string>

// Edge TPU delegate C API functions
extern "C" {
    typedef void* (*CreateDelegateFunc)(char**, char**, size_t, void (*)(const char *));
    typedef void (*DestroyDelegateFunc)(void*);
}

void error_reporter(const char* msg) {
    std::cout << "Edge TPU Error: " << msg << std::endl;
}

int main() {
    // Load the library
    void* handle = dlopen("./lib/libedgetpu.so.1", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << std::endl;
        return 1;
    }
    
    // Clear any existing error
    dlerror();
    
    // Load the symbol
    CreateDelegateFunc create_delegate = (CreateDelegateFunc) dlsym(handle, "tflite_plugin_create_delegate");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Cannot load symbol 'tflite_plugin_create_delegate': " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }
    
    std::cout << "Successfully loaded tflite_plugin_create_delegate function" << std::endl;
    
    // Try to create a delegate with error reporting
    void* delegate = create_delegate(nullptr, nullptr, 0, error_reporter);
    if (delegate) {
        std::cout << "Successfully created Edge TPU delegate" << std::endl;
        
        // Destroy the delegate
        DestroyDelegateFunc destroy_delegate = (DestroyDelegateFunc) dlsym(handle, "tflite_plugin_destroy_delegate");
        if (destroy_delegate) {
            destroy_delegate(delegate);
            std::cout << "Successfully destroyed Edge TPU delegate" << std::endl;
        }
    } else {
        std::cout << "Failed to create Edge TPU delegate" << std::endl;
    }
    
    // Close the library
    dlclose(handle);
    return 0;
}