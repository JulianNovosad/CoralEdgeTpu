// Standard C++ Library Includes
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono> // For std::chrono::milliseconds

// C System Headers
#include <signal.h>
#include <pthread.h>
#include <termios.h>
#include <unistd.h> // For _exit

// Project-specific Headers
#include "application.h"
#include "util_logging.h"

// Global variable to store original terminal settings
struct termios original_termios;

// Global flag checked by the main loop and modules.
// Access with memory_order_acquire/release for consistency.
extern std::atomic<bool> g_running;

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
    // Save original terminal settings
    tcgetattr(STDIN_FILENO, &original_termios);

    // Initialize Logger
    Logger::init("run", "logs", nullptr);
    Logger::getInstance().start_writer_thread();

    // Ensure signals are not blocked in the main thread BEFORE any threads are spawned.
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    if (pthread_sigmask(SIG_UNBLOCK, &set, NULL) != 0) {
        std::cerr << "Failed to unblock signals" << std::endl;
    }

    // Start a hard-kill watchdog thread
    std::thread hard_kill_watchdog([]() {
        while (g_running.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // Shutdown requested, wait 10 seconds then force exit
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (!g_running.load()) {
            std::cerr << "\n[WATCHDOG] Graceful shutdown timed out (3s). Forcing termination via _exit(1)." << std::endl;
            _exit(1);
        }
    });
    hard_kill_watchdog.detach();

    // Initialize Application (this registers signal handlers via constructor)
    Application app(argc, argv);

    return app.run();
}