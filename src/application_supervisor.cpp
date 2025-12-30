#include "application_supervisor.h"
#include <iostream>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#include <termios.h>
#include <stdlib.h>
#include <atomic>

// External declaration of the global atomic flag defined in main.cpp
extern std::atomic<bool> shutdown_requested;

// External declaration for terminal settings (defined in main.cpp)
extern struct termios original_termios;

static void signal_handler(int signum) {
    if (shutdown_requested.load(std::memory_order_acquire)) {
        // Force exit if signal is received again during shutdown
        _exit(signum);
    }
    shutdown_requested.store(true, std::memory_order_release);
}

static void crash_handler(int signum) {
    // Restore terminal settings on crash
    tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);
    std::cerr << "\nFATAL: Program crashed with signal " << signum << std::endl;
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
