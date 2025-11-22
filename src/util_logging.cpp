/**
 * @file util_logging.cpp
 * @brief Implements a thread-safe, asynchronous logging utility for the application.
 *
 * This module provides a singleton Logger class that handles logging messages
 * to both the console (stdout) and a file. It uses a separate writer thread
 * to process log messages from a queue, minimizing the impact of logging
 * operations on the main application threads. Logs are written in JSON format
 * to a file with rotation capabilities.
 */

#include "util_logging.h"
#include <iostream>       // For std::cout, std::cerr
#include <filesystem>     // C++17 for creating directories
#include <sstream>        // For std::ostringstream
#include <iomanip>        // For std::put_time

namespace fs = std::filesystem; ///< Alias for std::filesystem for brevity.

/**
 * @brief Retrieves the singleton instance of the Logger.
 *
 * This is the access point for the Logger. The instance is created upon the first
 * call. Subsequent calls return the same instance.
 *
 * @param log_file_prefix The prefix for log filenames (e.g., "run").
 * @param log_dir The directory where log files will be stored (e.g., "logs").
 * @return A reference to the singleton Logger instance.
 */
Logger& Logger::getInstance(const std::string& log_file_prefix, const std::string& log_dir) {
    static Logger instance(log_file_prefix, log_dir);
    return instance;
}

/**
 * @brief Constructor for the Logger.
 *
 * Initializes the Logger by ensuring the log directory exists, creating an
 * initial log file, and starting the asynchronous writer thread.
 *
 * @param log_file_prefix The prefix for log filenames.
 * @param log_dir The directory for log files.
 */
Logger::Logger(const std::string& log_file_prefix, const std::string& log_dir)
    : log_dir_(log_dir), log_file_prefix_(log_file_prefix), last_rotation_time_(std::chrono::system_clock::now()) {
    
    // Ensure log directory exists, create it if not.
    if (!fs::exists(log_dir_)) {
        fs::create_directories(log_dir_);
    }
    
    rotate_log_file(); // Create the initial log file (e.g., run-YYYYMMDD-HHMMSS.json).
    start_writer_thread(); // Begin processing log messages in a separate thread.
}

/**
 * @brief Destructor for the Logger.
 *
 * Stops the writer thread and closes any open log file handles to ensure
 * all buffered messages are written and resources are released.
 */
Logger::~Logger() {
    stop_writer_thread(); // Signal writer thread to stop and wait for it.
    if (log_file_.is_open()) {
        log_file_.close(); // Close the log file.
    }
}

/**
 * @brief Starts the asynchronous writer thread.
 *
 * If the logger is not already running, this method launches a dedicated
 * thread (`writer_thread_func`) to asynchronously write log messages.
 */
void Logger::start_writer_thread() {
    if (!running_) {
        running_ = true;
        writer_thread_ = std::thread(&Logger::writer_thread_func, this);
    }
}

/**
 * @brief Stops the asynchronous writer thread.
 *
 * Sets the `running_` flag to false, which signals the writer thread to
 * exit its loop after processing any remaining messages in the queue.
 * It then waits for the thread to join, ensuring a clean shutdown.
 */
void Logger::stop_writer_thread() {
    if (running_) {
        running_ = false; // Signal the thread to stop.
        if (writer_thread_.joinable()) {
            writer_thread_.join(); // Wait for the thread to finish.
        }
    }
}

/**
 * @brief Enqueues a standard log message to be written.
 *
 * This method is thread-safe. It adds a log entry to an internal queue,
 * which is then processed by the writer thread.
 *
 * @param level The log level (e.g., "INFO", "WARNING", "ERROR").
 * @param message The log message content.
 */
void Logger::log(const std::string& level, const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex_); // Protect queue access.
    log_queue_.push({std::chrono::system_clock::now(), level, message}); // Enqueue the log entry.
}

/**
 * @brief Enqueues a structured JSON log message.
 *
 * This method specifically formats a key-value pair into a simple JSON string
 * and enqueues it with a "JSON" level.
 *
 * @param key The key for the JSON log entry.
 * @param value The value for the JSON log entry (expected to be a valid JSON string or primitive).
 */
void Logger::log_json(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(log_mutex_); // Protect queue access.
    // For simplicity, we'll format this as a string.
    // A more robust JSON logger might use a dedicated JSON library.
    std::string json_message = "{\"" + key + "\": " + value + "}";
    log_queue_.push({std::chrono::system_clock::now(), "JSON", json_message}); // Enqueue structured log.
}

/**
 * @brief The main function for the asynchronous log writer thread.
 *
 * This thread continuously dequeues log entries from `log_queue_`, writes them
 * to both the console and the log file in JSON format. It continues to run
 * as long as `running_` is true or there are messages still in the queue.
 */
void Logger::writer_thread_func() {
    // Continue running if explicitly enabled OR if there are still messages to process
    while (running_ || !log_queue_.empty()) {
        std::unique_lock<std::mutex> lock(log_mutex_);
        // If the queue is empty, release the lock and sleep briefly to avoid busy-waiting.
        if (log_queue_.empty()) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Retrieve and remove the oldest log entry.
        LogEntry entry = log_queue_.front();
        log_queue_.pop();
        lock.unlock(); // Release lock before writing to allow other threads to log.

        // Write to console (standard output).
        std::cout << "[" << entry.level << "] " << entry.message << std::endl;

        // Write to file in JSON format.
        if (log_file_.is_open()) {
            // Escape special characters in message if necessary for proper JSON,
            // but for simplicity, assuming message content is safe or will be escaped upstream.
            log_file_ << "{\"timestamp\":\"" << get_current_iso_time() << "\", \"level\":\"" << entry.level << "\", \"message\":\"" << entry.message << "\"}" << std::endl;
        }
        
        // Basic log rotation check. Currently, rotation only happens on startup.
        // To implement runtime rotation (e.g., hourly or by size), this section
        // would check `std::chrono::duration_cast<std::chrono::hours>(now - last_rotation_time_).count() > 1`
        // or file size, and then call `rotate_log_file()`.
    }
}

/**
 * @brief Rotates the log file by closing the current one and opening a new one.
 *
 * The new log file's name includes a timestamp to ensure uniqueness.
 */
void Logger::rotate_log_file() {
    if (log_file_.is_open()) {
        log_file_.close(); // Close the old log file.
    }
    
    // Generate a new filename with current timestamp.
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::gmtime(&now_c); // Use gmtime for UTC time.

    std::ostringstream filename_ss;
    filename_ss << log_dir_ << "/" << log_file_prefix_ << "-"
                << std::put_time(now_tm, "%Y%m%d-%H%M%S") << ".json";
    
    // Open the new log file in append mode.
    log_file_.open(filename_ss.str(), std::ios_base::app);
    if (!log_file_.is_open()) {
        std::cerr << "Failed to open log file: " << filename_ss.str() << std::endl;
    }
    last_rotation_time_ = now; // Update the last rotation timestamp.
}

/**
 * @brief Gets the current system time formatted as an ISO 8601 string.
 *
 * @return A string representing the current time in ISO 8601 format (e.g., "YYYY-MM-DDTHH:MM:SSZ").
 */
std::string Logger::get_current_iso_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // Note: std::put_time expects std::tm*, so we convert system_clock::time_point to std::tm.
    // This example uses gmtime for UTC time. For local time, use std::localtime.
    std::tm* now_tm = std::gmtime(&now_c); 

    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y-%m-%dT%H:%M:%SZ"); // Format as ISO 8601 with 'Z' for UTC.
    return oss.str();
}