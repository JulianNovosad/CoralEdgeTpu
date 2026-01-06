// Verified headers: [termios.h, keyboard_monitor.h, util_logging.h, application_supervisor.h, iostream...]
// Verification timestamp: 2026-01-06 17:08:04
#include <termios.h>
#include "keyboard_monitor.h"
#include "util_logging.h"
#include "application_supervisor.h" // For shutdown_requested

#include <iostream>
#include <unistd.h>  // for read, STDIN_FILENO
#include <fcntl.h>   // for fcntl, F_GETFL, F_SETFL, O_NONBLOCK

// Forward declaration from global_definitions.cpp
extern std::atomic<bool> g_running;

// External declaration for terminal settings (defined in main.cpp)
extern struct termios original_termios;

KeyboardMonitor::KeyboardMonitor() {
}

KeyboardMonitor::~KeyboardMonitor() {
    stop();
}

void KeyboardMonitor::restore_terminal_settings() {
    // Restore original terminal settings from the global authority
    tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);
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

    // Set terminal to non-canonical, non-echo mode based on global original settings
    struct termios new_termios = original_termios;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);

    // Set stdin to non-blocking
    int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, old_flags | O_NONBLOCK);

    APP_LOG_INFO("Press 'o' to trigger a graceful shutdown.");

    char c;
    while (running_.load() && g_running.load(std::memory_order_acquire)) {
        if (read(STDIN_FILENO, &c, 1) > 0) {
            if (c == 'o') {
                APP_LOG_INFO("'o' key pressed. Initiating graceful shutdown...");
                g_running.store(false, std::memory_order_release);
                break; // Exit the loop
            }
        }
        // Sleep for a short duration to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Clean up: restore stdin flags, terminal settings will be restored by stop() or destructor
    fcntl(STDIN_FILENO, F_SETFL, old_flags);
}
