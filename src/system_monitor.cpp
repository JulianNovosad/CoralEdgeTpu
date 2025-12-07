#include "system_monitor.h"
#include <fstream>
#include <sstream>
#include <string>
#include <regex> // For regex parsing of /proc/meminfo

SystemMonitor::SystemMonitor(std::chrono::seconds interval_s)
    : interval_s_(interval_s) {
    LOG_INFO("SystemMonitor created with interval: " + std::to_string(interval_s_.count()) + " seconds.");
}

SystemMonitor::~SystemMonitor() {
    stop();
    LOG_INFO("SystemMonitor destroyed.");
}

bool SystemMonitor::start() {
    if (running_.exchange(true)) {
        LOG_ERROR("SystemMonitor is already running.");
        return false;
    }
    worker_thread_ = std::thread(&SystemMonitor::worker_thread_func, this);
    LOG_INFO("SystemMonitor started.");
    return true;
}

void SystemMonitor::stop() {
    if (running_.exchange(false)) {
        LOG_INFO("Stopping SystemMonitor...");
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        LOG_INFO("SystemMonitor stopped.");
    }
}

void SystemMonitor::worker_thread_func() {
    LOG_INFO("SystemMonitor worker thread started.");
    while (running_) {
        float cpu_temp = read_cpu_temperature();
        float memory_usage_percent = read_memory_usage();

        // LOG_CSV(module, stage, p50, p95, p99, temp, fps)
        // For SystemMonitor, p50, p95, p99, fps are not directly applicable or meaningful in the same way.
        // We'll use 0 for these for now, and rely on the 'temp' field for CPU temp and 'fps' field for memory usage.
        LOG_CSV("SystemMonitor", "Metrics", 0.0, 0.0, 0.0, cpu_temp, memory_usage_percent);

        std::this_thread::sleep_for(interval_s_);
    }
    LOG_INFO("SystemMonitor worker thread stopped.");
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
        LOG_WARNING("SystemMonitor: Could not open /sys/class/thermal/thermal_zone0/temp to read CPU temperature.");
    }
    return temp;
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
        LOG_WARNING("SystemMonitor: Could not open /proc/meminfo to read memory usage.");
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
