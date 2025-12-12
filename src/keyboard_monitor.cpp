#include "keyboard_monitor.h"
#include "util_logging.h"
#include "application_supervisor.h" // For shutdown_requested

#include <iostream>
#include <unistd.h>  // for read, STDIN_FILENO
#include <fcntl.h>   // for fcntl, F_GETFL, F_SETFL, O_NONBLOCK

// Forward declaration from application.cpp
extern std::atomic<bool> shutdown_requested;

KeyboardMonitor::KeyboardMonitor() {
    // Get original terminal settings to restore them on exit
    tcgetattr(STDIN_FILENO, &original_termios_);
}

KeyboardMonitor::~KeyboardMonitor() {
    stop();
}

void KeyboardMonitor::restore_terminal_settings() {
    // Restore original terminal settings
    tcsetattr(STDIN_FILENO, TCSANOW, &original_termios_);
}

bool KeyboardMonitor::start() {
    if (!running_.exchange(true)) {
        monitor_thread_ = std::thread(&KeyboardMonitor::monitor_thread_func, this);
        APP_LOG_INFO("KeyboardMonitor thread started.");
        return true;
    }
    return false;
}

void KeyboardMonitor::stop() {
    if (running_.exchange(false)) {
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
            APP_LOG_INFO("KeyboardMonitor thread stopped.");
        }
        restore_terminal_settings();
    }
}

bool KeyboardMonitor::is_running() const {
    return running_.load();
}

void KeyboardMonitor::monitor_thread_func() {
    set_thread_name("KeyboardMonitor");

    // Set terminal to non-canonical, non-echo mode
    struct termios new_termios = original_termios_;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);

    // Set stdin to non-blocking
    int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, old_flags | O_NONBLOCK);

    APP_LOG_INFO("Press 'o' to trigger a graceful shutdown.");

    char c;
    while (running_.load()) {
        if (read(STDIN_FILENO, &c, 1) > 0) {
            if (c == 'o') {
                APP_LOG_INFO("'o' key pressed. Initiating graceful shutdown...");
                shutdown_requested = true;
                break; // Exit the loop
            }
        }
        // Sleep for a short duration to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Clean up: restore stdin flags, terminal settings will be restored by stop() or destructor
    fcntl(STDIN_FILENO, F_SETFL, old_flags);
}
