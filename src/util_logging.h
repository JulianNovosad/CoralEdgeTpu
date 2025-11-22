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

#include <string>
#include <fstream>
#include <mutex>          // For std::mutex
#include <chrono>         // For std::chrono::system_clock, time_point
#include <queue>          // For std::queue<LogEntry>
#include <thread>         // For std::thread
#include <atomic>         // For std::atomic<bool>
#include <iomanip>        // For std::put_time

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
     * @brief Starts the asynchronous writer thread.
     *
     * If the logger is not already running, this method launches the background
     * thread responsible for writing log messages to their destinations.
     */
    void start_writer_thread();

    /**
     * @brief Stops the asynchronous writer thread.
     *
     * Gracefully signals the writer thread to terminate and waits for it to
     * finish processing any remaining log messages in the queue.
     */
    void stop_writer_thread();

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
     * Ensures proper shutdown of the writer thread and closes the log file.
     */
    ~Logger();

    /**
     * @brief The main function executed by the asynchronous writer thread.
     *
     * This thread continuously dequeues log entries and writes them to the
     * console and the log file.
     */
    void writer_thread_func();

    /**
     * @brief Rotates the log file.
     *
     * Closes the current log file and opens a new one with a timestamped name.
     */
    void rotate_log_file();

    /**
     * @brief Gets the current system time formatted as an ISO 8601 string.
     *
     * @return A string representing the current time in ISO 8601 format.
     */
    std::string get_current_iso_time();

    std::string log_dir_;                        ///< The directory where log files are stored.
    std::string log_file_prefix_;                ///< The prefix used for log filenames.
    std::ofstream log_file_;                     ///< Output file stream for logging to a file.
    std::mutex log_mutex_;                       ///< Mutex to protect access to the log queue.
    std::queue<LogEntry> log_queue_;             ///< Queue for asynchronous log message processing.
    std::thread writer_thread_;                  ///< Dedicated thread for writing log messages.
    std::atomic<bool> running_ = false;          ///< Atomic flag to control the writer thread's running state.
    std::chrono::system_clock::time_point last_rotation_time_; ///< Timestamp of the last log file rotation.
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


#endif // UTIL_LOGGING_H