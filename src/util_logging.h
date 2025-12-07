/**
 * @file util_logging.h
 * @brief Defines a thread-safe, asynchronous logging utility for the application.
 *
 * This header provides the Logger singleton class, which allows various modules
 * to log messages with different severity levels (INFO, WARNING, ERROR) and
 * also supports structured JSON logging. Messages are processed in a separate
 * thread to minimize impact on application performance and are written to
 * both the console and a timestamped JSON log file.
 */

#ifndef UTIL_LOGGING_H
#define UTIL_LOGGING_H

#include <sys/time.h> // For clock_gettime, CLOCK_MONOTONIC_RAW
#include <time.h>     // For timespec
#include <string>
#include <fstream>
#include <mutex>          // For std::mutex
#include <condition_variable> // Required for std::condition_variable
#include <chrono>         // For std::chrono::system_clock, time_point
#include <queue>          // For std::queue<LogEntry>
#include <thread>         // For std::thread
#include <atomic>         // For std::atomic<bool>
#include <iomanip>        // For std::put_time
#include <map>            // For std::map<std::string, CsvLogger>
#include <filesystem>     // For std::filesystem
// Forward declare CsvLogger to avoid circular dependency
class CsvLogger;

/**
 * @brief Structure to hold a single log entry.
 *
 * Each log entry consists of a timestamp, severity level, and the message content.
 */
struct LogEntry {
    std::chrono::system_clock::time_point timestamp; ///< The exact time the log entry was created.
    std::string level;                               ///< The severity level of the log (e.g., "INFO", "WARNING", "ERROR", "JSON").
    std::string message;                             ///< The actual message content to be logged.
};

/**
 * @brief Structure to hold a single structured CSV log entry for performance metrics.
 *
 * Each entry includes module, stage, and various performance statistics.
 */
struct CsvLogEntry {
    long long monotonic_time_ns; ///< Monotonic timestamp in nanoseconds using CLOCK_MONOTONIC_RAW.
    std::string module;          ///< Name of the module (e.g., "Camera", "Inference", "Logic").
    std::string stage;           ///< Name of the stage (e.g., "Frame Capture", "Inference", "Prediction").
    double p50;                  ///< 50th percentile latency in ms.
    double p95;                  ///< 95th percentile latency in ms.
    double p99;                  ///< 99th percentile latency in ms.
    double temp;                 ///< Temperature reading (e.g., CPU temperature).
    double fps;                  ///< Frames per second or equivalent throughput.
};

/**
 * @brief Manages a single CSV log file for a specific module, including rotation.
 */
class CsvLogger {
public:
    CsvLogger(const std::string& module_name, const std::string& log_dir, int max_log_files);
    ~CsvLogger();

    void write_header();
    void write_entry(const CsvLogEntry& entry);
    void rotate_log_file();

private:
    std::string module_name_;
    std::string log_dir_;
    int max_log_files_;
    std::ofstream current_log_file_;
    std::mutex file_mutex_; // Protects access to the file
};


/**
 * @brief A singleton class for thread-safe, asynchronous logging.
 *
 * The Logger class ensures that only one instance exists throughout the application.
 * It provides methods to enqueue log messages, which are then processed and written
 * to console and a file by a dedicated background thread.
 */
class Logger {
public:
    /**
     * @brief Retrieves the singleton instance of the Logger.
     *
     * This is the primary access point for the Logger. The instance is created
     * upon the first call with default or provided parameters. Subsequent calls
     * return the same instance.
     *
     * @param log_file_prefix The prefix for log filenames (e.g., "run").
     * @param log_dir The directory where log files will be stored (e.g., "logs").
     * @return A reference to the singleton Logger instance.
     */
    static Logger& getInstance(const std::string& log_file_prefix = "run", const std::string& log_dir = "logs");
    
    // Delete copy constructor and assignment operator to enforce singleton pattern.
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief Enqueues a standard log message with a specified level.
     *
     * This method is thread-safe and non-blocking, as messages are added
     * to a queue for asynchronous processing.
     *
     * @param level The severity level of the log message (e.g., "INFO", "WARNING", "ERROR").
     * @param message The content of the log message.
     */
    void log(const std::string& level, const std::string& message);

    /**
     * @brief Enqueues a structured JSON log message.
     *
     * This method is used for logging key-value pairs, which are formatted
     * into a simple JSON string before being enqueued.
     *
     * @param key The key for the JSON log entry.
     * @param value The value for the JSON log entry (expected to be a JSON string or primitive).
     */
    void log_json(const std::string& key, const std::string& value);

    /**
     * @brief Enqueues a structured CSV log message.
     *
     * This method is thread-safe and non-blocking, as messages are added
     * to a separate queue for asynchronous CSV processing.
     *
     * @param entry The structured CsvLogEntry to be logged.
     */
    void log_csv(const CsvLogEntry& entry);

    /**
     * @brief Starts the asynchronous writer threads (for standard and CSV logs).
     *
     * If the logger is not already running, this method launches the background
     * threads responsible for writing log messages to their destinations.
     */
    void start_writer_thread();

    /**
     * @brief Stops the asynchronous writer threads.
     *
     * Gracefully signals the writer threads to terminate and waits for them to
     * finish processing any remaining log messages in their queues.
     */
    void stop_writer_thread();

    /**
     * @brief Gets the current monotonic time in nanoseconds since epoch using CLOCK_MONOTONIC_RAW.
     *
     * This uses `clock_gettime` with `CLOCK_MONOTONIC_RAW` for a strictly monotonic timestamp.
     *
     * @return A long long representing nanoseconds since the `CLOCK_MONOTONIC_RAW` epoch.
     */
    long long get_raw_monotonic_time_ns();

private:
    /**
     * @brief Private constructor for the Logger (singleton pattern).
     *
     * Initializes the log directory, file prefix, and sets up the initial
     * log file.
     *
     * @param log_file_prefix The prefix for log filenames.
     * @param log_dir The directory for log files.
     */
    Logger(const std::string& log_file_prefix, const std::string& log_dir);

    /**
     * @brief Private destructor for the Logger.
     *
     * Ensures proper shutdown of the writer threads and closes log files.
     */
    ~Logger();

    /**
     * @brief The main function executed by the asynchronous standard log writer thread.
     *
     * This thread continuously dequeues LogEntry objects and writes them to the
     * console and the standard log file.
     */
    void writer_thread_func();

    /**
     * @brief The main function executed by the asynchronous CSV log writer thread.
     *
     * This thread continuously dequeues CsvLogEntry objects and writes them to
     * their respective module-specific CSV log files, handling rotation.
     */
    void csv_writer_thread_func();

    /**
     * @brief Rotates the standard log file.
     *
     * Closes the current standard log file and opens a new one with a timestamped name.
     */
    void rotate_standard_log_file();

    /**
     * @brief Gets the current system time formatted as an ISO 8601 string.
     *
     * @return A string representing the current time in ISO 8601 format.
     */
    std::string get_current_iso_time();


    // Standard log members
    std::string log_dir_;                        ///< The directory where log files are stored.
    std::string log_file_prefix_;                ///< The prefix used for standard log filenames.
    std::ofstream standard_log_file_;            ///< Output file stream for standard logs.
    std::mutex log_mutex_;                       ///< Mutex to protect access to the standard log queue.
    std::condition_variable cond_var_;           ///< Condition variable for standard writer thread signaling.
    std::queue<LogEntry> log_queue_;             ///< Queue for asynchronous standard log processing.
    std::thread standard_writer_thread_;         ///< Dedicated thread for writing standard log messages.
    std::chrono::system_clock::time_point last_rotation_time_; ///< Timestamp of the last standard log file rotation.
    int max_standard_log_files_;

    // CSV log members
    std::mutex csv_log_mutex_;                   ///< Mutex to protect access to the CSV log queue.
    std::condition_variable csv_cond_var_;       ///< Condition variable for CSV writer thread signaling.
    std::queue<CsvLogEntry> csv_log_queue_;      ///< Queue for asynchronous CSV log processing.
    std::thread csv_writer_thread_;              ///< Dedicated thread for writing CSV log messages.
    std::map<std::string, CsvLogger> csv_loggers_; ///< Map to manage CsvLogger instances per module.

    std::atomic<bool> running_ = false;          ///< Atomic flag to control writer threads' running state.
};

// --- Global Logging Macros for Convenience ---

/// @brief Logs an informational message.
#define LOG_INFO(msg) Logger::getInstance().log("INFO", msg)
/// @brief Logs a warning message.
#define LOG_WARNING(msg) Logger::getInstance().log("WARNING", msg)
/// @brief Logs an error message.
#define LOG_ERROR(msg) Logger::getInstance().log("ERROR", msg)
/// @brief Logs a structured JSON message.
#define LOG_JSON(key, value) Logger::getInstance().log_json(key, value)

// Macro for convenience to log CSV performance metrics
#define LOG_CSV(module, stage, p50, p95, p99, temp, fps) \
    Logger::getInstance().log_csv({Logger::getInstance().get_raw_monotonic_time_ns(), module, stage, p50, p95, p99, temp, fps})


#endif // UTIL_LOGGING_H
