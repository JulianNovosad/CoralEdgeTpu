#ifndef UTIL_LOGGING_H
#define UTIL_LOGGING_H

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <queue>
#include <thread>
#include <atomic>
#include <iomanip> // For std::put_time

// Structure to hold a log entry
struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    std::string level;
    std::string message;
};

class Logger {
public:
    static Logger& getInstance(const std::string& log_file_prefix = "run", const std::string& log_dir = "logs");
    
    // Delete copy constructor and assignment operator for singleton
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void log(const std::string& level, const std::string& message);
    void log_json(const std::string& key, const std::string& value); // For specific JSON logs

    void start_writer_thread();
    void stop_writer_thread();

private:
    Logger(const std::string& log_file_prefix, const std::string& log_dir);
    ~Logger();

    void writer_thread_func();
    void rotate_log_file();
    std::string get_current_iso_time();

    std::string log_dir_;
    std::string log_file_prefix_;
    std::ofstream log_file_;
    std::mutex log_mutex_;
    std::queue<LogEntry> log_queue_;
    std::thread writer_thread_;
    std::atomic<bool> running_ = false;
    std::chrono::system_clock::time_point last_rotation_time_;
};

// Global logging macros for convenience
#define LOG_INFO(msg) Logger::getInstance().log("INFO", msg)
#define LOG_WARNING(msg) Logger::getInstance().log("WARNING", msg)
#define LOG_ERROR(msg) Logger::getInstance().log("ERROR", msg)
#define LOG_JSON(key, value) Logger::getInstance().log_json(key, value)


#endif // UTIL_LOGGING_H
