#ifndef APPLICATION_SUPERVISOR_H
#define APPLICATION_SUPERVISOR_H

#include <atomic>
#include <vector>
#include <memory>
#include <functional>
#include <csignal> // For signal handling

#include "util_logging.h"

// Forward declarations of all modules that need to be stopped
class CameraCapture;
class InferenceEngine;
class LogicModule;
class HttpServer;
class H264Encoder;
class IMUSensor;

/**
 * @brief Global atomic flag to signal shutdown request across the application.
 */
extern std::atomic<bool> shutdown_requested;

/**
 * @brief Handles global application shutdown and orchestrates module stopping.
 *
 * The ApplicationSupervisor registers all major application modules and provides
 * a centralized mechanism to signal and manage their graceful shutdown.
 */
class ApplicationSupervisor {
public:
    ApplicationSupervisor();
    ~ApplicationSupervisor();

    /**
     * @brief Registers a module's stop function with the supervisor.
     * @param module_name A string name for the module.
     * @param stop_function A std::function that calls the module's stop() method.
     */
    void register_module_stop(const std::string& module_name, std::function<void()> stop_function);

    /**
     * @brief Initiates the graceful shutdown process for all registered modules.
     */
    void initiate_shutdown();

    /**
     * @brief Sets up signal handlers for graceful shutdown (e.g., SIGINT, SIGTERM).
     */
    void setup_signal_handlers();

private:
    std::vector<std::pair<std::string, std::function<void()>>> registered_modules_;
};

#endif // APPLICATION_SUPERVISOR_H
