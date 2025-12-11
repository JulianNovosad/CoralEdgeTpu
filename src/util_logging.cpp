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
#include <algorithm>      // For std::sort

#ifdef __linux__
#include <sys/prctl.h> // For prctl(PR_SET_NAME)
#endif

namespace fs = std::filesystem; ///< Alias for std::filesystem for brevity.

// Static members initialization
std::unique_ptr<Logger> Logger::instance_;
std::once_flag Logger::once_flag_;

// Implement CsvLogger methods
CsvLogger::CsvLogger(const std::string& module_name, const std::string& log_dir, int max_log_files)
    : module_name_(module_name), log_dir_(log_dir), max_log_files_(max_log_files), current_log_minute_(-1) {
    
    // Ensure log directory exists
    if (!fs::exists(log_dir_)) {
        fs::create_directories(log_dir_);
    }
    // No initial file open here, it will be handled by rotate_log_file
}

CsvLogger::~CsvLogger() {
    std::lock_guard<std::recursive_mutex> lock(file_mutex_);
    if (current_log_file_.is_open()) {
        current_log_file_.close();
    }
}

void CsvLogger::write_header() {
    std::lock_guard<std::recursive_mutex> lock(file_mutex_);
    if (current_log_file_.is_open()) {
        // Universal header as defined in README.md and CsvLogEntry struct
        current_log_file_ << "produced_ts_epoch_ms,module,thread_id,event,call_ts_epoch_ms,"
                          << "camera_frame_id,camera_width,camera_height,camera_exposure_ms,camera_copy_time_ms,"
                          << "tpu_inference_ms,tpu_input_w,tpu_input_h,tpu_temp_c,"
                          << "encoder_encode_ms,encoder_total_encoded_frames,encoder_average_fps,"
                          << "logic_metric_ballistics,logic_metric_hit_scan,logic_metric_servo_actuation,"
                          << "sysmon_cpu_temp_c,sysmon_cpu_usage_percent,sysmon_mem_usage_percent,"
                          << "p50_latency_ms,p95_latency_ms,p99_latency_ms,average_fps,total_frames_processed_or_inferences,average_latency_ms,details\n";
        current_log_file_.flush();
    }
}

void CsvLogger::write_entry(const CsvLogEntry& entry) {
    std::lock_guard<std::recursive_mutex> lock(file_mutex_);
    if (current_log_file_.is_open()) {
        current_log_file_ << entry.produced_ts_epoch_ms << ","
                          << entry.module.data() << ","
                          << entry.thread_id << ","
                          << entry.event.data() << ","
                          << entry.call_ts_epoch_ms << ","
                          << entry.camera_frame_id << ","
                          << entry.camera_width << ","
                          << entry.camera_height << ","
                          << entry.camera_exposure_ms << ","
                          << entry.camera_copy_time_ms << ","
                          << entry.tpu_inference_ms << ","
                          << entry.tpu_input_w << ","
                          << entry.tpu_input_h << ","
                          << entry.tpu_temp_c << ","
                          << entry.encoder_encode_ms << ","
                          << entry.encoder_total_encoded_frames << ","
                          << entry.encoder_average_fps << ","
                          << entry.logic_metric_ballistics << ","
                          << entry.logic_metric_hit_scan << ","
                          << entry.logic_metric_servo_actuation << ","
                          << entry.sysmon_cpu_temp_c << ","
                          << entry.sysmon_cpu_usage_percent << ","
                          << entry.sysmon_mem_usage_percent << ","
                          << entry.p50_latency_ms << ","
                          << entry.p95_latency_ms << ","
                          << entry.p99_latency_ms << ","
                          << entry.average_fps << ","
                          << entry.total_frames_processed_or_inferences << ","
                          << entry.average_latency_ms << ","
                          << entry.details.data() << "\n"; // Ensure details is the last field
        current_log_file_.flush();
    }
}


void CsvLogger::rotate_log_file() {
    std::lock_guard<std::recursive_mutex> lock(file_mutex_);
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
        current_log_minute_ = -1; // Indicate no file is open
    } else {
        // Only write header if file was newly created or is empty
        if (fs::file_size(new_log_filepath) == 0) {
            write_header();
        }

        // Update current_log_minute_
        auto now_for_minute = std::chrono::system_clock::now();
        std::time_t now_c_for_minute = std::chrono::system_clock::to_time_t(now_for_minute);
        std::tm* now_tm_for_minute = std::localtime(&now_c_for_minute);
        current_log_minute_ = now_tm_for_minute->tm_min;

        // Manage old log files (keep max_log_files)
        std::vector<fs::path> log_files;
        try {
            for (const auto& entry : fs::directory_iterator(log_dir_)) {
                if (entry.is_regular_file() && entry.path().extension() == ".csv" && entry.path().stem().string().rfind(module_name_, 0) == 0) {
                    log_files.push_back(entry.path());
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Filesystem error during directory iteration for " << module_name_ << ": " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "General error during directory iteration for " << module_name_ << ": " << e.what() << std::endl;
        }
        std::sort(log_files.begin(), log_files.end()); // Sort by name, which should naturally sort by date/time

        while (log_files.size() > static_cast<size_t>(max_log_files_)) {
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
        static Logger dummy_logger; // Use the now-public default constructor
        return dummy_logger;
    }
    return *instance_.get();
}

// Definition for the public default constructor
Logger::Logger()
    : base_log_dir_("."), log_file_prefix_("dummy_log"), last_rotation_time_(std::chrono::system_clock::now()), max_standard_log_files_(1), running_(false) {
    // No-op, primarily for the dummy logger during early initialization
    // No need to create directories or start threads for a dummy.
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
        APP_LOG_INFO("Logger: Creating standard writer thread.");
        standard_writer_thread_ = std::thread(&Logger::writer_thread_func, this);
        APP_LOG_INFO("Logger: Creating CSV writer thread.");
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
    log_queue_.push(std::move(LogEntry{std::chrono::system_clock::now(), level, message})); // Enqueue the log entry.
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
    std::string json_message = "{\"" + key + "\": " + value + "}";
    log_queue_.push(std::move(LogEntry{std::chrono::system_clock::now(), "JSON", json_message})); // Enqueue structured log.
}

void Logger::log_csv(const CsvLogEntry& entry) {
    csv_log_queue_.push(std::move(entry));
}

/**
 * @brief The main function for the asynchronous log writer thread.
 *
 * This thread continuously dequeues log entries from `log_queue_`, writes them
 * to both the console and the log file in JSON format. It continues to run
 * as long as `running_` is true or there are messages still in the queue.
 */
void Logger::writer_thread_func() {
    APP_LOG_INFO("Logger: Standard writer thread started.");
    LogEntry entry; // Declare outside loop to avoid re-allocation
    while (running_.load()) { // Loop while running_ is true
        // Try to pop an entry from the queue
        if (log_queue_.pop(entry)) { // Use non-blocking pop
            // Write to console (standard output).
            std::cout << "[" << entry.level.data() << "] " << entry.message.data() << std::endl;

            // Write to file in JSON format.
            if (standard_log_file_.is_open()) {
                standard_log_file_ << "{\"timestamp\":\"" << get_current_iso_time() << "\", \"level\":\"" << entry.level.data() << "\", \"message\":\"" << entry.message.data() << "\"}" << std::endl;
            }
            // Basic log rotation check. Currently, rotation only happens on startup.
            // To implement runtime rotation (e.g., hourly or by size), this section
            // would check `std::chrono::duration_cast<std::chrono::hours>(now - last_rotation_time_).count() > 1`
            // or file size, and then call `rotate_standard_log_file()`.
        } else {
            // If queue is empty, sleep for a short period to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    // Process any remaining messages in the queue before exiting
    while (log_queue_.pop(entry)) {
        std::cout << "[" << entry.level.data() << "] " << entry.message.data() << std::endl;
        if (standard_log_file_.is_open()) {
            standard_log_file_ << "{\"timestamp\":\"" << get_current_iso_time() << "\", \"level\":\"" << entry.level.data() << "\", \"message\":\"" << entry.message.data() << "\"}" << std::endl;
        }
    }
    APP_LOG_INFO("Logger: Standard writer thread finished.");
}

/**
 * @brief The main function executed by the asynchronous CSV log writer thread.
 *
 * This thread continuously dequeues CsvLogEntry objects and writes them to
 * their respective module-specific CSV log files, handling rotation.
 */
void Logger::csv_writer_thread_func() {
    APP_LOG_INFO("Logger: CSV writer thread started.");
    CsvLogEntry entry; // Declare outside loop to avoid re-allocation
    while (running_.load()) {
        if (csv_log_queue_.pop(entry)) {
            // Retrieve the correct CsvLogger for this module
            APP_LOG_DEBUG("Attempting to find CsvLogger for module: [" + std::string(entry.module.data()) + "]");
            auto it = csv_loggers_.find(std::string(entry.module.data()));
            if (it != csv_loggers_.end()) {
                // Check for log rotation based on current minute
                auto now_check = std::chrono::system_clock::now();
                std::time_t now_c_check = std::chrono::system_clock::to_time_t(now_check);
                std::tm* now_tm_check = std::localtime(&now_c_check);
                
                // Use getter methods to access CsvLogger members
                if (now_tm_check->tm_min != it->second.get_current_log_minute() || !it->second.is_file_open()) {
                    it->second.rotate_log_file();
                }
                
                it->second.write_entry(entry);
            } else {
                APP_LOG_ERROR("CsvLogger not found for module: [" + std::string(entry.module.data()) + "]. Entry dropped.");
            }
        } else {
            // If queue is empty, sleep for a short period to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    // Process any remaining messages in the queue before exiting
    while (csv_log_queue_.pop(entry)) {
        APP_LOG_DEBUG("Processing remaining CsvLogEntry for module: [" + std::string(entry.module.data()) + "]");
        auto it = csv_loggers_.find(std::string(entry.module.data()));
        if (it != csv_loggers_.end()) {
            auto now_check = std::chrono::system_clock::now();
            std::time_t now_c_check = std::chrono::system_clock::to_time_t(now_check);
            std::tm* now_tm_check = std::localtime(&now_c_check);
            
            if (now_tm_check->tm_min != it->second.get_current_log_minute() || !it->second.is_file_open()) {
                it->second.rotate_log_file();
            }
            it->second.write_entry(entry);
        } else {
            APP_LOG_ERROR("CsvLogger not found for module: [" + std::string(entry.module.data()) + "]. Entry dropped.");
        }
    }
    APP_LOG_INFO("Logger: CSV writer thread finished.");
}

/**
 * @brief Rotates the standard log file by closing the current one and opening a new one.
 *
 * The new log file's name includes a timestamp to ensure uniqueness.
 */
void Logger::rotate_standard_log_file() {

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

// set_thread_name implementation
void set_thread_name(const std::string& name) {
#ifdef __linux__
    // Only set if running on Linux and name fits PR_SET_NAME limit (16 chars including null terminator)
    if (name.length() < 16) {
        prctl(PR_SET_NAME, name.c_str(), 0, 0, 0);
    }
#else
    // No-op on other platforms
    (void)name; // Suppress unused parameter warning
#endif // __linux__
}