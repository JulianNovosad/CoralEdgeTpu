// Verified headers: [atomic, thread, termios.h]
// Verification timestamp: 2026-01-06 17:08:04
#ifndef KEYBOARD_MONITOR_H
#define KEYBOARD_MONITOR_H

#include <atomic>
#include <thread>
#include <termios.h>


/**
 * @brief A class to monitor keyboard input in a non-blocking way.
 *
 * This class spawns a dedicated thread to listen for a specific character
 * from stdin and triggers a shutdown signal when it's detected. It handles
 * the terminal settings to allow for non-canonical, non-echoing input reading.
 */
class KeyboardMonitor {
public:
    KeyboardMonitor();
    ~KeyboardMonitor();

    /**
     * @brief Starts the keyboard monitoring thread.
     * @return True if the thread was started successfully, false otherwise.
     */
    bool start();

    /**
     * @brief Stops the keyboard monitoring thread and restores terminal settings.
     */
    void stop();

    /**
     * @brief Checks if the monitoring thread is currently running.
     * @return True if running, false otherwise.
     */
    bool is_running() const;

private:
    void monitor_thread_func();
    void restore_terminal_settings();

    std::atomic<bool> running_{false};
    std::thread monitor_thread_;
};

#endif // KEYBOARD_MONITOR_H