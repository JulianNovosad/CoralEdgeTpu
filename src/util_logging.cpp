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

// Static members initialization
std::unique_ptr<Logger> Logger::instance_;
std::once_flag Logger::once_flag_;

// Implement CsvLogger methods
CsvLogger::CsvLogger(const std::string& module_name, const std::string& log_dir, int max_log_files)
    : module_name_(module_name), log_dir_(log_dir), max_log_files_(max_log_files) {
    
    // Ensure log directory exists
    if (!fs::exists(log_dir_)) {
        fs::create_directories(log_dir_);
    }
    // No initial file open here, it will be handled by rotate_log_file
}

CsvLogger::~CsvLogger() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (current_log_file_.is_open()) {
        current_log_file_.close();
    }
}

void CsvLogger::write_header() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (current_log_file_.is_open()) {
        current_log_file_ << "monotonic_time_ns,module,stage,p50,p95,p99,temp,fps\n";
        current_log_file_.flush();
    }
}

void CsvLogger::write_entry(const CsvLogEntry& entry) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (current_log_file_.is_open()) {
        current_log_file_ << entry.monotonic_time_ns << ","
                          << entry.module << ","
                          << entry.stage << ","
                          << std::fixed << std::setprecision(3) << entry.p50 << ","
                          << std::fixed << std::setprecision(3) << entry.p95 << ","
                          << std::fixed << std::setprecision(3) << entry.p99 << ","
                          << std::fixed << std::setprecision(2) << entry.temp << ","
                          << std::fixed << std::setprecision(3) << entry.fps << "\n";
        current_log_file_.flush();
    }
}

void CsvLogger::rotate_log_file() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (current_log_file_.is_open()) {
        current_log_file_.close();
    }

    // Ensure the subsystem-specific log directory exists
    if (!fs::exists(log_dir_)) {
        fs::create_directories(log_dir_);
    }

    // Generate current timestamped filename
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_c); // Use localtime for file naming

    std::ostringstream oss;
    oss << module_name_ << "_" << std::put_time(now_tm, "%Y_%m_%d_%H_%M") << ".csv";
    std::string new_log_filename = oss.str();
    
    fs::path new_log_filepath = fs::path(log_dir_) / new_log_filename;

    // Open new primary log file
    current_log_file_.open(new_log_filepath.string(), std::ios_base::out | std::ios_base::app); // Use app to append if file already exists for current minute
    if (!current_log_file_.is_open()) {
        std::cerr << "Failed to open CSV log file: " << new_log_filepath << std::endl;
    } else {
        // Only write header if file was newly created or is empty
        if (fs::file_size(new_log_filepath) == 0) {
            write_header();
        }

        // Manage old log files (keep max_log_files)
        std::vector<fs::path> log_files;
        for (const auto& entry : fs::directory_iterator(log_dir_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv" && entry.path().stem().string().rfind(module_name_, 0) == 0) {
                log_files.push_back(entry.path());
            }
        }
        std::sort(log_files.begin(), log_files.end()); // Sort by name, which should naturally sort by date/time

        while (log_files.size() > max_log_files_) {
            fs::remove(log_files.front()); // Remove the oldest log file
            log_files.erase(log_files.begin());
        }
    }
}

void Logger::init(const std::string& log_file_prefix, const std::string& base_log_dir, const std::vector<SubsystemLogConfig>& csv_configs) {
    std::call_once(once_flag_, [&]() {
        instance_.reset(new Logger(log_file_prefix, base_log_dir, csv_configs));
    });
}

Logger& Logger::getInstance() {
    if (!instance_) {
        // This should ideally not happen if init() is called at startup.
        // Provide a basic default or throw an error.
        throw std::runtime_error("Logger::init() was not called before getInstance().");
    }
    return *instance_.get();
}

/**
 * @brief Constructor for the Logger.
 *
 * Initializes the Logger by ensuring the log directory exists, creating an
 * initial log file, and starting the asynchronous writer thread.
 *
 * @param log_file_prefix The prefix for log filenames.
 * @param base_log_dir The base directory for log files.
 * @param csv_configs A vector of configurations for subsystem-specific CSV logs.
 */
Logger::Logger(const std::string& log_file_prefix, const std::string& base_log_dir, const std::vector<SubsystemLogConfig>& csv_configs)
    : base_log_dir_(base_log_dir), log_file_prefix_(log_file_prefix), last_rotation_time_(std::chrono::system_clock::now()), max_standard_log_files_(3), csv_subsystem_configs_(csv_configs) {
    
    // Ensure base log directory exists, create it if not.
    if (!fs::exists(base_log_dir_)) {
        fs::create_directories(base_log_dir_);
    }
    
    // Initialize standard log file
    rotate_standard_log_file();

    // Initialize CsvLogger instances for each subsystem
    for (const auto& config : csv_subsystem_configs_) {
        fs::path sub_log_dir = fs::path(base_log_dir_) / config.log_dir_suffix;
        csv_loggers_.try_emplace(config.name, config.name, sub_log_dir.string(), config.max_log_files);
        // Proactively create subdirectory and rotate log file to write header
        if (!fs::exists(sub_log_dir)) {
            fs::create_directories(sub_log_dir);
        }
        csv_loggers_.at(config.name).rotate_log_file();
    }
}

/**
 * @brief Destructor for the Logger.
 *
 * Stops the writer thread and closes any open log file handles to ensure
 * all buffered messages are written and resources are released.
 */
Logger::~Logger() {
    stop_writer_thread(); // Signal writer thread to stop and wait for it.
    if (standard_log_file_.is_open()) {
        standard_log_file_.close(); // Close the log file.
    }
}

/**
 * @brief Starts the asynchronous writer threads (for standard and CSV logs).
 *
 * If the logger is not already running, this method launches the dedicated
 * threads to asynchronously write log messages.
 */
void Logger::start_writer_thread() {
    if (!running_.exchange(true)) { // Atomically set to true and check old value
        standard_writer_thread_ = std::thread(&Logger::writer_thread_func, this);
        csv_writer_thread_ = std::thread(&Logger::csv_writer_thread_func, this);
    }
}

/**
 * @brief Stops the asynchronous writer threads.
 *
 * Sets the `running_` flag to false, which signals the writer threads to
 * exit their loops after processing any remaining messages in their queues.
 * It then waits for the threads to join, ensuring a clean shutdown.
 */
void Logger::stop_writer_thread() {
    if (running_.exchange(false)) { // Atomically set to false and check old value
        cond_var_.notify_all(); // Wake up standard writer thread
        csv_cond_var_.notify_all(); // Wake up CSV writer thread
        
        if (standard_writer_thread_.joinable()) {
            standard_writer_thread_.join(); // Wait for the standard thread to finish.
        }
        if (csv_writer_thread_.joinable()) {
            csv_writer_thread_.join(); // Wait for the CSV thread to finish.
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
    {
        std::lock_guard<std::mutex> lock(log_mutex_); // Protect queue access.
        log_queue_.push(LogEntry{std::chrono::system_clock::now(), level, message}); // Enqueue the log entry.
    }
    cond_var_.notify_one();
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
    {
        std::lock_guard<std::mutex> lock(log_mutex_); // Protect queue access.
        // For simplicity, we'll format this as a string.
        // A more robust JSON logger might use a dedicated JSON library.
        std::string json_message = "{\"" + key + "\": " + value + "}";
        log_queue_.push(LogEntry{std::chrono::system_clock::now(), "JSON", json_message}); // Enqueue structured log.
    }
    cond_var_.notify_one();
}

void Logger::log_csv(const CsvLogEntry& entry) {
    {
        std::lock_guard<std::mutex> lock(csv_log_mutex_);
        csv_log_queue_.push(entry);
    }
    csv_cond_var_.notify_one();
}

/**
 * @brief The main function for the asynchronous log writer thread.
 *
 * This thread continuously dequeues log entries from `log_queue_`, writes them
 * to both the console and the log file in JSON format. It continues to run
 * as long as `running_` is true or there are messages still in the queue.
 */
void Logger::writer_thread_func() {
    while (true) {
        std::unique_lock<std::mutex> lock(log_mutex_);
        // Wait until the queue is not empty or the logger is shutting down
        cond_var_.wait(lock, [this] { return !log_queue_.empty() || !running_; });

        // If shutting down and the queue is empty, exit the loop
        if (!running_ && log_queue_.empty()) {
            break;
        }

        // Retrieve and remove the oldest log entry.
        LogEntry entry = std::move(log_queue_.front());
        log_queue_.pop();
        lock.unlock(); // Release lock before writing to allow other threads to log.

        // Write to console (standard output).
        std::cout << "[" << entry.level << "] " << entry.message << std::endl;

        // Write to file in JSON format.
        if (standard_log_file_.is_open()) {
            // Escape special characters in message if necessary for proper JSON,
            // but for simplicity, assuming message content is safe or will be escaped upstream.
            standard_log_file_ << "{\"timestamp\":\"" << get_current_iso_time() << "\", \"level\":\"" << entry.level << "\", \"message\":\"" << entry.message << "\"}" << std::endl;
        }
        
        // Basic log rotation check. Currently, rotation only happens on startup.
        // To implement runtime rotation (e.g., hourly or by size), this section
        // would check `std::chrono::duration_cast<std::chrono::hours>(now - last_rotation_time_).count() > 1`
        // or file size, and then call `rotate_standard_log_file()`. // Corrected comment
    }
}

/**
 * @brief The main function executed by the asynchronous CSV log writer thread.
 *
 * This thread continuously dequeues CsvLogEntry objects and writes them to
 * their respective module-specific CSV log files, handling rotation.
 */
void Logger::csv_writer_thread_func() {
    while (true) {
        std::unique_lock<std::mutex> lock(csv_log_mutex_);
        csv_cond_var_.wait(lock, [this] { return !csv_log_queue_.empty() || !running_; });

        if (!running_ && csv_log_queue_.empty()) {
            break;
        }

        CsvLogEntry entry = std::move(csv_log_queue_.front());
        csv_log_queue_.pop();
        lock.unlock(); // Release lock before writing to allow other threads to log.

        // Retrieve the correct CsvLogger for this module
        auto it = csv_loggers_.find(entry.module);
        if (it != csv_loggers_.end()) {
            it->second.write_entry(entry);
        } else {
            std::cerr << "Warning: CsvLogger not found for module: " << entry.module << ". Entry dropped." << std::endl;
        }
    }
}

/**
 * @brief Rotates the standard log file by closing the current one and opening a new one.
 *
 * The new log file's name includes a timestamp to ensure uniqueness.
 */
void Logger::rotate_standard_log_file() {
    std::lock_guard<std::mutex> lock(log_mutex_); // Protect file operations
    if (standard_log_file_.is_open()) {
        standard_log_file_.close(); // Close the old log file.
    }

    // Delete oldest files (e.g., run-prefix.2.json -> delete)
    for (int i = max_standard_log_files_; i >= 1; --i) {
        fs::path old_path = fs::path(base_log_dir_) / (log_file_prefix_ + "." + std::to_string(i) + ".json");
        if (fs::exists(old_path)) {
            if (i == max_standard_log_files_) {
                fs::remove(old_path);
            } else {
                fs::path new_path = fs::path(base_log_dir_) / (log_file_prefix_ + "." + std::to_string(i + 1) + ".json");
                fs::rename(old_path, new_path);
            }
        }
    }

    // Rename current (un-numbered) file to .1
    fs::path current_log_filename = fs::path(base_log_dir_) / (log_file_prefix_ + ".json");
    if (fs::exists(current_log_filename)) {
        fs::path new_path = fs::path(base_log_dir_) / (log_file_prefix_ + ".1.json");
        fs::rename(current_log_filename, new_path);
    }
    
    // Open new primary log file
    standard_log_file_.open(current_log_filename.string(), std::ios_base::out | std::ios_base::trunc); // Overwrite if exists
    if (!standard_log_file_.is_open()) {
        std::cerr << "Failed to open log file: " << current_log_filename << std::endl;
    } else {
        // Optionally write a header for JSON logs, but typically not needed for structured JSON per line
    }
    last_rotation_time_ = std::chrono::system_clock::now(); // Update the last rotation timestamp with system_clock
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

/**
 * @brief Gets the current monotonic time in nanoseconds since epoch.
 *
 * This uses std::chrono::steady_clock, which is typically monotonic and
 * not subject to system clock adjustments (like NTP).
 *
 * @return A long long representing nanoseconds since the steady_clock epoch.
 */
long long Logger::get_raw_monotonic_time_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == -1) {
        // Fallback to steady_clock if CLOCK_MONOTONIC_RAW is not available or fails
        // This should not happen on Linux with modern kernels, but is a defensive measure.
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}