#include "application_supervisor.h"
#include <iostream>

// Define the global atomic flag
std::atomic<bool> shutdown_requested(false);

// Static member function to handle signals
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        LOG_INFO("Shutdown signal (" + std::to_string(signal) + ") received. Initiating graceful shutdown.");
        shutdown_requested = true;
    }
}

ApplicationSupervisor::ApplicationSupervisor() {
    LOG_INFO("ApplicationSupervisor created.");
    // No modules registered yet, done in main.cpp
}

ApplicationSupervisor::~ApplicationSupervisor() {
    LOG_INFO("ApplicationSupervisor destroyed.");
}

void ApplicationSupervisor::register_module_stop(const std::string& module_name, std::function<void()> stop_function) {
    registered_modules_.emplace_back(module_name, stop_function);
    LOG_INFO("Registered module for shutdown: " + module_name);
}

void ApplicationSupervisor::initiate_shutdown() {
    LOG_INFO("Initiating graceful shutdown for all registered modules...");
    // Stop modules in reverse order of registration if dependency exists (e.g., consumers before producers)
    // For now, simple iteration. More advanced would track dependencies.
    for (auto it = registered_modules_.rbegin(); it != registered_modules_.rend(); ++it) {
        LOG_INFO("Stopping module: " + it->first);
        it->second(); // Call the stop function
    }
    LOG_INFO("All registered modules stopped.");
}

void ApplicationSupervisor::setup_signal_handlers() {
    // Register signal handlers for SIGINT (Ctrl+C) and SIGTERM
    if (std::signal(SIGINT, signal_handler) == SIG_ERR) {
        LOG_ERROR("Failed to register SIGINT handler.");
    }
    if (std::signal(SIGTERM, signal_handler) == SIG_ERR) {
        LOG_ERROR("Failed to register SIGTERM handler.");
    }
    LOG_INFO("Signal handlers for SIGINT and SIGTERM set up.");
}