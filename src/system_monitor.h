#ifndef SYSTEM_MONITOR_H
#define SYSTEM_MONITOR_H

#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>

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
    void get_performance_metrics();
    bool is_running() const { return running_; }

private:
    void worker_thread_func();
    float read_cpu_temperature();
    float read_memory_usage(); // Returns percentage of used memory
    float read_cpu_usage();    // Returns CPU usage percentage

    std::atomic<bool> running_ = false;
    std::mutex stop_mutex_;
    std::condition_variable stop_cv_;
    std::thread worker_thread_;
    std::chrono::seconds interval_s_;

    // Variables for CPU usage calculation
    long prev_total_cpu_time_ = 0;
    long prev_idle_cpu_time_ = 0;
};

#endif // SYSTEM_MONITOR_H