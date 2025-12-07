#ifndef SYSTEM_MONITOR_H
#define SYSTEM_MONITOR_H

#include <string>
#include <thread>
#include <atomic>
#include <chrono>

#include "util_logging.h"

/**
 * @brief Monitors system resources like CPU temperature and memory usage.
 *
 * This module periodically reads system metrics and logs them using the
 * CSV logger.
 */
class SystemMonitor {
public:
    SystemMonitor(std::chrono::seconds interval_s = std::chrono::seconds(5));
    ~SystemMonitor();

    bool start();
    void stop();
    bool is_running() const { return running_; }

private:
    void worker_thread_func();
    float read_cpu_temperature();
    float read_memory_usage(); // Returns percentage of used memory

    std::atomic<bool> running_ = false;
    std::thread worker_thread_;
    std::chrono::seconds interval_s_;
};

#endif // SYSTEM_MONITOR_H
