#include "system_monitor.h"
#include <fstream>
#include <sstream>
#include <string>
#include <regex> // For regex parsing of /proc/meminfo
#include <iostream>

SystemMonitor::SystemMonitor(std::chrono::seconds interval_s)
    : interval_s_(interval_s) {
    // Initialize CPU usage stats by calling it once
    read_cpu_usage(); 
    APP_LOG_INFO("SystemMonitor created with interval: " + std::to_string(interval_s_.count()) + " seconds.");
}

SystemMonitor::~SystemMonitor() {
    stop();
    APP_LOG_INFO("SystemMonitor destroyed.");
}

bool SystemMonitor::start() {
    if (running_.exchange(true)) {
        APP_LOG_ERROR("SystemMonitor is already running.");
        return false;
    }
    worker_thread_ = std::thread(&SystemMonitor::worker_thread_func, this);
    APP_LOG_INFO("SystemMonitor started.");
    return true;
}

void SystemMonitor::stop() {
    if (running_.exchange(false)) {
        APP_LOG_INFO("Stopping SystemMonitor...");
        {
            std::lock_guard<std::mutex> lock(stop_mutex_);
            stop_cv_.notify_all();
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        APP_LOG_INFO("SystemMonitor stopped.");
    }
}

void SystemMonitor::worker_thread_func() {
    APP_LOG_INFO("SystemMonitor worker thread started.");
    auto next_tick = std::chrono::steady_clock::now();
    while (running_.load(std::memory_order_acquire)) {
        float cpu_temp = read_cpu_temperature();
        float memory_usage_percent = read_memory_usage();
        float cpu_usage = read_cpu_usage();

        CsvLogEntry entry;
        entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        copy_to_array(entry.module, "SystemMonitor");
        entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        copy_to_array(entry.event, "sysmon_metrics");
        entry.call_ts_epoch_ms = entry.produced_ts_epoch_ms;
        entry.sys_cpu_temp_c = cpu_temp;
        entry.sys_cpu_usage_pct = cpu_usage;
        entry.sys_ram_usage_pct = memory_usage_percent;
        Logger::getInstance().log_csv(entry);

        next_tick += interval_s_;
        std::unique_lock<std::mutex> lock(stop_mutex_);
        if (stop_cv_.wait_until(lock, next_tick, [this] { return !running_.load(std::memory_order_acquire); })) {
            break; // Shutdown requested
        }
    }
    APP_LOG_INFO("SystemMonitor worker thread stopped.");
}

float SystemMonitor::read_cpu_temperature() {
    float temp = 0.0f;
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
    if (temp_file.is_open()) {
        int raw_temp;
        temp_file >> raw_temp;
        temp = static_cast<float>(raw_temp) / 1000.0f; // raw_temp is in milli-degrees Celsius
        temp_file.close();
    } else {
        APP_LOG_WARNING("SystemMonitor: Could not open /sys/class/thermal/thermal_zone0/temp to read CPU temperature.");
    }
    return temp;
}

float SystemMonitor::read_cpu_usage() {
    std::ifstream stat_file("/proc/stat");
    std::string line;
    float cpu_usage = 0.0f;

    if (stat_file.is_open()) {
        std::getline(stat_file, line);
        stat_file.close();

        std::stringstream ss(line);
        std::string cpu_label;
        long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;

        ss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;

        if (cpu_label == "cpu") {
            long current_idle_cpu_time = idle + iowait;
            long current_total_cpu_time = user + nice + system + idle + iowait + irq + softirq + steal;

            // Only calculate if previous values exist (not the first call)
            if (prev_total_cpu_time_ != 0 && prev_idle_cpu_time_ != 0) {
                long total_cpu_time_diff = current_total_cpu_time - prev_total_cpu_time_;
                long idle_cpu_time_diff = current_idle_cpu_time - prev_idle_cpu_time_;

                if (total_cpu_time_diff > 0) {
                    cpu_usage = 100.0f * (1.0f - static_cast<float>(idle_cpu_time_diff) / static_cast<float>(total_cpu_time_diff));
                }
            }
            // Update previous values for the next calculation
            prev_total_cpu_time_ = current_total_cpu_time;
            prev_idle_cpu_time_ = current_idle_cpu_time;
        }
    } else {
        APP_LOG_WARNING("SystemMonitor: Could not open /proc/stat to read CPU usage.");
    }
    return cpu_usage;
}

float SystemMonitor::read_memory_usage() {
    long total_memory = 0;
    long free_memory = 0;
    long buffers_memory = 0;
    long cached_memory = 0;

    std::ifstream mem_file("/proc/meminfo");
    if (mem_file.is_open()) {
        std::string line;
        std::regex total_regex(R"(MemTotal:\s*(\d+) kB)");
        std::regex free_regex(R"(MemFree:\s*(\d+) kB)");
        std::regex buffers_regex(R"(Buffers:\s*(\d+) kB)");
        std::regex cached_regex(R"(Cached:\s*(\d+) kB)");

        while (std::getline(mem_file, line)) {
            std::smatch match;
            if (std::regex_search(line, match, total_regex)) {
                total_memory = std::stol(match[1].str());
            } else if (std::regex_search(line, match, free_regex)) {
                free_memory = std::stol(match[1].str());
            } else if (std::regex_search(line, match, buffers_regex)) {
                buffers_memory = std::stol(match[1].str());
            } else if (std::regex_search(line, match, cached_regex)) {
                cached_memory = std::stol(match[1].str());
            }
        }
        mem_file.close();
    } else {
        APP_LOG_WARNING("SystemMonitor: Could not open /proc/meminfo to read memory usage.");
        return 0.0f;
    }

    // According to proc(5) man page, MemAvailable is a more accurate indicator of usable memory.
    // If not available (older kernels), a common formula for used memory is:
    // Used = Total - Free - Buffers - Cached
    long used_memory = total_memory - free_memory - buffers_memory - cached_memory;
    if (total_memory > 0) {
        return static_cast<float>(used_memory) / static_cast<float>(total_memory) * 100.0f;
    }
    return 0.0f;
}

void SystemMonitor::get_performance_metrics() {
    // This function is called by the Application main_loop to log current system stats
    float cpu_temp = read_cpu_temperature();
    float memory_usage_percent = read_memory_usage();
    float cpu_usage = read_cpu_usage();
    
    // Log using the CSV format (p50 for CPU usage, temp for CPU temp, fps for memory usage percent)
    // Log using the CSV format (p50 for CPU usage, temp for CPU temp, fps for memory usage percent)
    long long current_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count();
    CsvLogEntry entry;
    entry.produced_ts_epoch_ms = current_time_ms;
    copy_to_array(entry.module, "SystemMonitor");
    entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    copy_to_array(entry.event, "PerformanceMetrics");
    entry.call_ts_epoch_ms = current_time_ms; // Use current_time_ms as call_ts for summary metrics
    entry.sys_cpu_temp_c = cpu_temp;
    entry.sys_cpu_usage_pct = cpu_usage;
    entry.sys_ram_usage_pct = memory_usage_percent;
    // No specific 'details' string as individual metrics are now logged directly.
    Logger::getInstance().log_csv(entry);
    APP_LOG_INFO("--- SystemMonitor Performance Metrics ---");
    APP_LOG_INFO("  CPU Usage: " + std::to_string(cpu_usage) + "%");
    APP_LOG_INFO("  CPU Temperature: " + std::to_string(cpu_temp) + " C");
    APP_LOG_INFO("  Memory Usage: " + std::to_string(memory_usage_percent) + "%");
    APP_LOG_INFO("-----------------------------------------");
}
