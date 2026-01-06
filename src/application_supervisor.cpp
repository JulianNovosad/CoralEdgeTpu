// Verified headers: [application_supervisor.h, iostream, csignal, unistd.h, termios.h...]
// Verification timestamp: 2026-01-06 17:08:04
#include "application_supervisor.h"
#include <iostream>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#include <termios.h>
#include <stdlib.h>
#include <atomic>

// External declaration of the global atomic flag defined in global_definitions.cpp
extern std::atomic<bool> g_running;

// External declaration for terminal settings (defined in main.cpp)
extern struct termios original_termios;

static void signal_handler(int signum) {
    // Restore terminal settings IMMEDIATELY when signal is received to ensure user is not left in raw mode
    tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);

    if (!g_running.load(std::memory_order_acquire)) {
        // Force exit if signal is received again during shutdown
        _exit(signum);
    }
    g_running.store(false, std::memory_order_release);

    // Start a thread to force exit if graceful shutdown takes too long
    std::thread force_exit_thread([signum]() {
        sleep(3); // 3 second grace period
        if (!g_running.load()) {
            const char* msg = "\nGraceful shutdown timed out, forcing exit...\n";
            write(STDERR_FILENO, msg, 44);
            _exit(signum);
        }
    });
    force_exit_thread.detach();
}

static void crash_handler(int signum) {
    // Restore terminal settings on crash (tcsetattr is generally signal-safe)
    tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);
    
    // POSIX async-signal-safe error reporting
    const char* msg = "\nFATAL: Program crashed with signal: ";
    write(STDERR_FILENO, msg, 37);

    if (signum == SIGSEGV) {
        write(STDERR_FILENO, "SIGSEGV (Segmentation Fault)\n", 29);
    } else if (signum == SIGABRT) {
        write(STDERR_FILENO, "SIGABRT (Abort)\n", 16);
    } else if (signum == SIGILL) {
        write(STDERR_FILENO, "SIGILL (Illegal Instruction)\n", 29);
    } else if (signum == SIGFPE) {
        write(STDERR_FILENO, "SIGFPE (Arithmetic Exception)\n", 30);
    } else {
        write(STDERR_FILENO, "UNKNOWN\n", 8);
    }
    
    _exit(signum);
}

static void exit_handler() {
    // Final terminal restore on normal exit
    tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);
}

ApplicationSupervisor::ApplicationSupervisor() {
}

ApplicationSupervisor::~ApplicationSupervisor() {
    APP_LOG_INFO("ApplicationSupervisor destroyed.");
}

void ApplicationSupervisor::register_module_stop(const std::string& module_name, std::function<void()> stop_function) {
    registered_modules_.emplace_back(module_name, stop_function);
    APP_LOG_INFO("Registered module for shutdown: " + module_name);
}

void ApplicationSupervisor::register_child_process(pid_t pid) {
    if (pid > 0) {
        child_pids_.insert(pid);
        APP_LOG_INFO("Registered child process for tracking: " + std::to_string(pid));
    }
}

void ApplicationSupervisor::setup_signal_handlers() {
    struct sigaction sa_graceful = {};
    sa_graceful.sa_handler = signal_handler;
    sigemptyset(&sa_graceful.sa_mask);
    sa_graceful.sa_flags = 0; 

    struct sigaction sa_crash = {};
    sa_crash.sa_handler = crash_handler;
    sigemptyset(&sa_crash.sa_mask);
    sa_crash.sa_flags = 0;
    
    // Register handlers for graceful shutdown signals
    sigaction(SIGINT, &sa_graceful, NULL);
    sigaction(SIGTERM, &sa_graceful, NULL);
    sigaction(SIGQUIT, &sa_graceful, NULL);
    
    // Register handlers for crash signals
    sigaction(SIGSEGV, &sa_crash, NULL);
    sigaction(SIGABRT, &sa_crash, NULL);
    
    // Register exit handler for normal termination
    atexit(exit_handler);
}

void ApplicationSupervisor::initiate_shutdown() {
    if (shutdown_in_progress_.exchange(true, std::memory_order_acq_rel)) {
        return; // Already in progress
    }
    
    APP_LOG_INFO("Initiating graceful shutdown for all registered modules...");
    
    for (auto it = registered_modules_.rbegin(); it != registered_modules_.rend(); ++it) {
        APP_LOG_INFO("Stopping module: " + it->first);
        try {
            it->second(); 
        } catch (const std::exception& e) {
            APP_LOG_ERROR("Error stopping module " + it->first + ": " + std::string(e.what()));
        }
    }
    APP_LOG_INFO("All registered modules stopped.");
}

void ApplicationSupervisor::final_cleanup() {
    if (cleanup_completed_.exchange(true, std::memory_order_acq_rel)) {
        return; // Already cleaned up
    }

    APP_LOG_INFO("Starting final cleanup of child processes...");
    
    for (pid_t pid : child_pids_) {
        if (pid > 0) {
            kill(pid, SIGTERM);
        }
    }
    
    usleep(500000); 
    
    for (auto it = child_pids_.begin(); it != child_pids_.end();) {
        pid_t pid = *it;
        if (pid > 0 && kill(pid, 0) == 0) {
            kill(pid, SIGKILL);
            usleep(100000);
        }
        it = child_pids_.erase(it);
    }
    
    APP_LOG_INFO("Final cleanup completed.");
}
