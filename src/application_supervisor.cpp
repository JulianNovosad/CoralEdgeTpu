#include "application_supervisor.h"
#include <iostream>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#include <termios.h>  // for terminal settings
#include <stdlib.h>   // for atexit

// Define the global atomic flag
std::atomic<bool> shutdown_requested(false);

// Static member definition
bool ApplicationSupervisor::shutdown_in_progress_ = false;

// External declaration for terminal settings
extern struct termios original_termios;

// Flag to track if terminal settings have been restored
static bool terminal_restored = false;

// Function to restore terminal settings
void restore_terminal_settings() {
    if (!terminal_restored) {
        tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);
        terminal_restored = true;
    }
}

// Static member function to handle signals
void signal_handler(int signal) {
    // Restore terminal settings before setting shutdown flag
    restore_terminal_settings();
    // Only set the atomic flag - do minimal work in signal handler
    if (signal == SIGINT || signal == SIGTERM || signal == SIGQUIT) {
        shutdown_requested = true;
    } else if (signal == SIGSEGV || signal == SIGABRT) {
        // For crash signals, we just restore terminal and exit
        _exit(128 + signal);
    }
}

// Exit handler for normal termination
void exit_handler() {
    restore_terminal_settings();
}

ApplicationSupervisor::ApplicationSupervisor() {
    // No modules registered yet, done in main.cpp
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
    // Register signal handlers for SIGINT (Ctrl+C), SIGTERM, and SIGQUIT
    if (std::signal(SIGINT, signal_handler) == SIG_ERR) {
        APP_LOG_ERROR("Failed to register SIGINT handler.");
    }
    if (std::signal(SIGTERM, signal_handler) == SIG_ERR) {
        APP_LOG_ERROR("Failed to register SIGTERM handler.");
    }
    if (std::signal(SIGQUIT, signal_handler) == SIG_ERR) {
        APP_LOG_ERROR("Failed to register SIGQUIT handler.");
    }
    // Register handlers for crash signals as well
    if (std::signal(SIGSEGV, signal_handler) == SIG_ERR) {
        APP_LOG_ERROR("Failed to register SIGSEGV handler.");
    }
    if (std::signal(SIGABRT, signal_handler) == SIG_ERR) {
        APP_LOG_ERROR("Failed to register SIGABRT handler.");
    }
    
    // Register exit handler for normal termination
    if (atexit(exit_handler) != 0) {
        APP_LOG_ERROR("Failed to register exit handler.");
    }
}

void ApplicationSupervisor::initiate_shutdown() {
    if (shutdown_in_progress_) {
        return; // Already in progress, avoid duplicate shutdown attempts
    }
    
    shutdown_in_progress_ = true;
    APP_LOG_INFO("Initiating graceful shutdown for all registered modules...");
    
    // Stop modules in reverse order of registration if dependency exists (e.g., consumers before producers)
    // For now, simple iteration. More advanced would track dependencies.
    for (auto it = registered_modules_.rbegin(); it != registered_modules_.rend(); ++it) {
        APP_LOG_INFO("Stopping module: " + it->first);
        try {
            it->second(); // Call the stop function
        } catch (const std::exception& e) {
            APP_LOG_ERROR("Error stopping module " + it->first + ": " + std::string(e.what()));
        }
    }
    APP_LOG_INFO("All registered modules stopped.");
}

void ApplicationSupervisor::final_cleanup() {
    APP_LOG_INFO("Starting final cleanup of child processes...");
    
    // Send SIGTERM to all tracked child processes
    for (pid_t pid : child_pids_) {
        if (pid > 0) {
            std::cout << "Sending SIGTERM to child process: " << pid << std::endl;
            APP_LOG_INFO("Sending SIGTERM to child process: " + std::to_string(pid));
            kill(pid, SIGTERM);
        }
    }
    
    // Wait briefly for processes to terminate gracefully
    usleep(500000); // Sleep for 500ms
    
    // Check which processes are still alive and send SIGKILL
    for (auto it = child_pids_.begin(); it != child_pids_.end();) {
        pid_t pid = *it;
        if (pid > 0) {
            // Check if process is still alive by sending signal 0 (doesn't actually send a signal)
            if (kill(pid, 0) == 0) {
                // Process is still alive, send SIGKILL
                std::cout << "Process " << pid << " still alive, sending SIGKILL" << std::endl;
                APP_LOG_INFO("Process " + std::to_string(pid) + " still alive, sending SIGKILL");
                kill(pid, SIGKILL);
                
                // Wait briefly for the process to be killed
                usleep(100000); // Sleep for 100ms
            }
            
            // Remove from tracking since we've attempted to kill it
            it = child_pids_.erase(it);
        } else {
            ++it;
        }
    }
    
    // Final safety net: if this is the original parent process, ensure no detector processes remain
    // Use killpg to ensure all processes in the process group are terminated
    pid_t current_pid = getpid();
    pid_t current_pgid = getpgid(current_pid);
    
    if (current_pgid == current_pid) {  // This process is the process group leader
        APP_LOG_INFO("Executing final safety net process cleanup...");
        // Additional cleanup could be added here if needed
    }
    
    APP_LOG_INFO("Final cleanup completed.");
}