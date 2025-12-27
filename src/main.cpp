#include "application.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream> // Added for std::cerr/cout
#include "util_logging.h"
#include <signal.h>
#include <pthread.h>
#include <termios.h>  // for terminal settings
#include <unistd.h>   // for STDIN_FILENO

// Global variable to store original terminal settings
struct termios original_termios;

// This is a global flag checked by the main loop and set by the ApplicationSupervisor.
extern std::atomic<bool> shutdown_requested;



std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    if (!file.is_open()) {
        APP_LOG_ERROR("Failed to open labels file: " + path);
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

int main(int argc, char** argv) {
    // Initialize Logger
    std::vector<SubsystemLogConfig> log_configs;
    log_configs.push_back({"CameraCapture", "camera", 3});
    log_configs.push_back({"InferenceEngine", "tpu", 3});
    log_configs.push_back({"LogicModule", "logic", 3});
    log_configs.push_back({"SystemMonitor", "system_monitor", 3});
    log_configs.push_back({"H264Encoder", "h264_encoder", 3});
    
    Logger::init("run", "logs", log_configs);
    Logger::getInstance().start_writer_thread();

    // Ensure signals are not blocked in the main thread
    sigset_t set;
    sigemptyset(&set);
    sigprocmask(SIG_SETMASK, &set, NULL);
    
    // Also unblock in the main thread specifically
    pthread_sigmask(SIG_UNBLOCK, &set, NULL);

    Application app(argc, argv);
    return app.run();
}