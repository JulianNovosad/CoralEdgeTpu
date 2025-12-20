#include <iostream>
#include <string>
#include <filesystem>
#include "src/config_loader.h"

int main() {
    std::cout << "Testing configuration loading..." << std::endl;
    
    // Load configuration
    std::filesystem::path config_path = "no_encoder_config.json";
    
    ConfigLoader config_loader;
    if (!config_loader.load(config_path.string())) {
        std::cerr << "ERROR: Failed to load configuration file at " << config_path.string() << ". Exiting." << std::endl;
        return 1;
    }

    // Print some config values
    std::cout << "Model path: " << config_loader.get_model_path() << std::endl;
    std::cout << "Labels path: " << config_loader.get_labels_path() << std::endl;
    std::cout << "High res width: " << config_loader.get_high_res_width() << std::endl;
    std::cout << "High res height: " << config_loader.get_high_res_height() << std::endl;
    std::cout << "Camera FPS: " << config_loader.get_camera_fps() << std::endl;

    std::cout << "Configuration loaded successfully!" << std::endl;
    return 0;
}