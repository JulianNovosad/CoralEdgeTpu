#include "application.h"
#include "util_logging.h"
#include <filesystem>
#include <iostream>
#include <csignal>

// Forward declaration from main.cpp
std::vector<std::string> load_labels(const std::string& path);
extern std::atomic<bool> shutdown_requested;

int main(int argc, char** argv) {
    // Initialize minimal logging
    Logger::init("minimal", "./logs", {});
    Logger::getInstance().start_writer_thread();
    APP_LOG_INFO("Minimal CoralEdgeTpu Detector Starting...");

    // Load configuration
    Application app(argc, argv);
    
    // Get config loader from the application
    auto& config_loader = app.get_config_loader();
    
    // Load model and labels paths
    std::filesystem::path exe_path = argv[0];
    std::filesystem::path config_path = exe_path.parent_path() / ".." / "minimal_config.json";
    const std::string model_path = (config_path.parent_path() / config_loader.get_model_path()).string();
    const std::string labels_path = (config_path.parent_path() / config_loader.get_labels_path()).string();

    APP_LOG_INFO("Model path: " + model_path);
    APP_LOG_INFO("Labels path: " + labels_path);

    // Load labels
    auto labels = load_labels(labels_path);
    if (labels.empty()) {
        APP_LOG_ERROR("Failed to load labels from: " + labels_path);
        return 1;
    }
    APP_LOG_INFO("Loaded " + std::to_string(labels.size()) + " labels");

    // Test if model file exists
    if (!std::filesystem::exists(model_path)) {
        APP_LOG_ERROR("Model file not found: " + model_path);
        return 1;
    }
    APP_LOG_INFO("Model file found: " + model_path);

    APP_LOG_INFO("Minimal detector initialized successfully.");
    APP_LOG_INFO("Press Ctrl+C to exit.");

    // Simple loop
    while (!shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    Logger::getInstance().stop_writer_thread();
    APP_LOG_INFO("Minimal detector shutting down.");
    return 0;
}