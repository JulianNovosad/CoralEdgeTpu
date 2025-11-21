#include "util_logging.h"
#include <iostream>
#include <filesystem> // C++17 for creating directories
#include <sstream>

namespace fs = std::filesystem;

Logger& Logger::getInstance(const std::string& log_file_prefix, const std::string& log_dir) {
    static Logger instance(log_file_prefix, log_dir);
    return instance;
}

Logger::Logger(const std::string& log_file_prefix, const std::string& log_dir)
    : log_dir_(log_dir), log_file_prefix_(log_file_prefix), last_rotation_time_(std::chrono::system_clock::now()) {
    
    // Ensure log directory exists
    if (!fs::exists(log_dir_)) {
        fs::create_directories(log_dir_);
    }
    
    rotate_log_file(); // Create initial log file
    start_writer_thread();
}

Logger::~Logger() {
    stop_writer_thread();
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void Logger::start_writer_thread() {
    if (!running_) {
        running_ = true;
        writer_thread_ = std::thread(&Logger::writer_thread_func, this);
    }
}

void Logger::stop_writer_thread() {
    if (running_) {
        running_ = false;
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
    }
}

void Logger::log(const std::string& level, const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    log_queue_.push({std::chrono::system_clock::now(), level, message});
}

void Logger::log_json(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    // For simplicity, we'll format this as a string for now.
    // A more robust JSON logger would build a JSON object.
    std::string json_message = "{\"" + key + "\": " + value + "}";
    log_queue_.push({std::chrono::system_clock::now(), "JSON", json_message});
}

void Logger::writer_thread_func() {
    while (running_ || !log_queue_.empty()) {
        std::unique_lock<std::mutex> lock(log_mutex_);
        if (log_queue_.empty()) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep if queue is empty
            continue;
        }

        LogEntry entry = log_queue_.front();
        log_queue_.pop();
        lock.unlock();

        // Write to console
        std::cout << "[" << entry.level << "] " << entry.message << std::endl;

        // Write to file (JSON format)
        if (log_file_.is_open()) {
            log_file_ << "{\"timestamp\":\"" << get_current_iso_time() << "\", \"level\":\"" << entry.level << "\", \"message\":\"" << entry.message << "\"}" << std::endl;
        }
        
        // Basic log rotation (e.g., every hour or certain size)
        // For now, let's just make sure a new file is created on startup.
        // A more complex rotation logic would involve checking size/time and creating new files.
    }
}

void Logger::rotate_log_file() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
    
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::gmtime(&now_c); // Use gmtime for UTC

    std::ostringstream filename_ss;
    filename_ss << log_dir_ << "/" << log_file_prefix_ << "-"
                << std::put_time(now_tm, "%Y%m%d-%H%M%S") << ".json";
    
    log_file_.open(filename_ss.str(), std::ios_base::app);
    if (!log_file_.is_open()) {
        std::cerr << "Failed to open log file: " << filename_ss.str() << std::endl;
    }
    last_rotation_time_ = now;
}

std::string Logger::get_current_iso_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::gmtime(&now_c); // Use gmtime for UTC

    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}
