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
#include <chrono>         // For std::chrono::system_clock, time_point
#include <thread>         // For std::thread
#include <atomic>         // For std::atomic<bool>
#include <array>          // For std::array
#include <iomanip>        // For std::put_time
#include <map>            // For std::map<std::string, CsvLogger>
#include <filesystem>     // For std::filesystem
#include <vector>         // For std::vector
#include <execinfo.h>     // For backtrace
#include <cxxabi.h>       // For __cxa_demangle
#include <cstring>        // For strncpy
#include <mutex>          // For std::recursive_mutex and std::once_flag

#include <boost/lockfree/spsc_queue.hpp> // For lock-free SPSC queues

// --- Logging Configuration Struct ---
struct SubsystemLogConfig {
    std::string name;
    std::string log_dir_suffix;
    int max_log_files;
};

// Forward declare CsvLogger to avoid circular dependency
class CsvLogger;

// Function to set thread name (for easier debugging)
void set_thread_name(const std::string& name);

// --- Global Logging Macros for Convenience ---

/// @brief Logs an informational message.
#define APP_LOG_INFO(msg) Logger::getInstance().log("INFO", msg)
/// @brief Logs a warning message.
#define APP_LOG_WARNING(msg) Logger::getInstance().log("WARNING", msg)
/// @brief Logs an error message.
#define APP_LOG_ERROR(msg) Logger::getInstance().log("ERROR", msg)
#ifdef DEBUG_MODE
/// @brief Logs a debug message.
#define APP_LOG_DEBUG(msg) Logger::getInstance().log("DEBUG", msg)
#else
#define APP_LOG_DEBUG(msg) (void)0 // No-op in release mode
#endif
/// @brief Logs a structured JSON message.
#define APP_LOG_JSON(key, value) Logger::getInstance().log_json(key, value)




// Helper function to safely copy a C-style string to a std::array<char, N>
template<size_t N>
void copy_to_array(std::array<char, N>& destination, const char* source) {
    if (source == nullptr) {
        destination[0] = '\0';
        return;
    }
    strncpy(destination.data(), source, destination.size() - 1);
    destination[destination.size() - 1] = '\0'; // Ensure null-termination
}

/**
 * @brief Structure to hold a single log entry.
 *
 * Each log entry consists of a timestamp, severity level, and the message content.
 */
struct LogEntry {
    std::chrono::system_clock::time_point timestamp; ///< The exact time the log entry was created.
    std::array<char, 16> level;                      ///< The severity level of the log (e.g., "INFO", "WARNING", "ERROR", "JSON").
    std::array<char, 256> message;                   ///< The actual message content to be logged.

    LogEntry() = default; // Default constructor

    // Constructor to safely copy strings into fixed-size arrays
    LogEntry(std::chrono::system_clock::time_point ts, const std::string& lvl, const std::string& msg)
        : timestamp(ts) {
        strncpy(level.data(), lvl.c_str(), level.size() - 1);
        level[level.size() - 1] = '\0'; // Ensure null-termination
        strncpy(message.data(), msg.c_str(), message.size() - 1);
        message[message.size() - 1] = '\0'; // Ensure null-termination
    }
};

/**
 * @brief Structure to hold a single structured CSV log entry for performance metrics.
 *
 * Each entry includes module, stage, and various performance statistics.
 */
struct CsvLogEntry {
    long long produced_ts_epoch_ms; ///< Timestamp when this log line was produced (epoch ms, UTC)
    std::array<char, 32> module;    ///< Module name: camera|tpu|encoder|logic|sysmon
    long long thread_id;            ///< Numeric OS thread id (TID)
    std::array<char, 64> event;     ///< Short label (e.g. frame_captured, inference_done, encode_done)
    long long call_ts_epoch_ms;     ///< Timestamp when the module was *called/issued* to start work (epoch ms, UTC)

    // Camera-specific metrics
    long long camera_frame_id = -1;
    int camera_width = -1;
    int camera_height = -1;
    float camera_exposure_ms = -1.0f;
    float camera_copy_time_ms = -1.0f;

    // TPU-specific metrics
    float tpu_inference_ms = -1.0f;
    int tpu_input_w = -1;
    int tpu_input_h = -1;
    float tpu_temp_c = -1.0f;

    // Encoder-specific metrics
    float encoder_encode_ms = -1.0f;
    long long encoder_total_encoded_frames = -1;
    float encoder_average_fps = -1.0f;

    // Logic-specific metrics (placeholders for now)
    float logic_metric_ballistics = -1.0f;
    float logic_metric_hit_scan = -1.0f;
    float logic_metric_servo_actuation = -1.0f;

    // System Monitor-specific metrics
    float sysmon_cpu_temp_c = -1.0f;
    float sysmon_cpu_usage_percent = -1.0f;
    float sysmon_mem_usage_percent = -1.0f;

    // Generic Performance Metrics (for PerformanceMetrics event type)
    float p50_latency_ms = -1.0f;
    float p95_latency_ms = -1.0f;
    float p99_latency_ms = -1.0f;
    float average_fps = -1.0f;
    long long total_frames_processed_or_inferences = -1;
    float average_latency_ms = -1.0f;

    std::array<char, 1024> details; ///< Optional: Any additional details as a string (e.g., error messages, JSON if necessary for very complex data)

    CsvLogEntry() = default; // Default constructor

    // Constructor to safely copy string parts
    CsvLogEntry(long long ts_prod, const std::string& mod, long long tid, const std::string& evt, long long ts_call)
        : produced_ts_epoch_ms(ts_prod), thread_id(tid), call_ts_epoch_ms(ts_call) {
        strncpy(module.data(), mod.c_str(), module.size() - 1);
        module[module.size() - 1] = '\0';
        strncpy(event.data(), evt.c_str(), event.size() - 1);
        event[event.size() - 1] = '\0';
        details[0] = '\0'; // Initialize details to empty string
    }

    // Full constructor for convenience (can be extended as needed)
    CsvLogEntry(long long ts_prod, const std::string& mod, long long tid, const std::string& evt, long long ts_call,
                long long cam_frame_id, int cam_w, int cam_h, float cam_exp_ms, float cam_copy_ms,
                float tpu_inf_ms, int tpu_in_w, int tpu_in_h, float tpu_temp,
                float enc_ms, long long enc_frames, float enc_fps,
                float logic_ballistics, float logic_hit_scan, float logic_servo,
                float sysmon_cpu_temp, float sysmon_cpu_usage, float sysmon_mem_usage,
                float p50_lat, float p95_lat, float p99_lat, float avg_fps, long long total_frames, float avg_lat,
                const std::string& det)
        : produced_ts_epoch_ms(ts_prod), thread_id(tid), call_ts_epoch_ms(ts_call),
          camera_frame_id(cam_frame_id), camera_width(cam_w), camera_height(cam_h),
          camera_exposure_ms(cam_exp_ms), camera_copy_time_ms(cam_copy_ms),
          tpu_inference_ms(tpu_inf_ms), tpu_input_w(tpu_in_w), tpu_input_h(tpu_in_h),
          tpu_temp_c(tpu_temp), encoder_encode_ms(enc_ms), encoder_total_encoded_frames(enc_frames),
          encoder_average_fps(enc_fps), logic_metric_ballistics(logic_ballistics),
          logic_metric_hit_scan(logic_hit_scan), logic_metric_servo_actuation(logic_servo),
          sysmon_cpu_temp_c(sysmon_cpu_temp), sysmon_cpu_usage_percent(sysmon_cpu_usage),
          sysmon_mem_usage_percent(sysmon_mem_usage), p50_latency_ms(p50_lat),
          p95_latency_ms(p95_lat), p99_latency_ms(p99_lat), average_fps(avg_fps),
          total_frames_processed_or_inferences(total_frames), average_latency_ms(avg_lat) {
        strncpy(module.data(), mod.c_str(), module.size() - 1);
        module[module.size() - 1] = '\0';
        strncpy(event.data(), evt.c_str(), event.size() - 1);
        event[event.size() - 1] = '\0';
        strncpy(details.data(), det.c_str(), details.size() - 1);
        details[details.size() - 1] = '\0';
    }
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

    int get_current_log_minute() const { return current_log_minute_; }
    bool is_file_open() const { return current_log_file_.is_open(); }

private:
    std::string module_name_;
    std::string log_dir_;
    int max_log_files_;
    std::ofstream current_log_file_;
    std::recursive_mutex file_mutex_; // Protects access to the file
    int current_log_minute_; // New member to store the minute of the current log file

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
     * @brief Initializes the singleton Logger instance with logging parameters.
     *
     * This method should be called once at the application's startup.
     *
     * @param log_file_prefix The prefix for standard log filenames (e.g., "run").
     * @param base_log_dir The base directory where log files will be stored (e.g., "/home/pi/logs").
     * @param csv_configs A vector of configurations for subsystem-specific CSV logs.
     */
    static void init(const std::string& log_file_prefix, const std::string& base_log_dir, const std::vector<SubsystemLogConfig>& csv_configs);

    /**
     * @brief Retrieves the singleton instance of the Logger.
     *
     * This is the primary access point for the Logger. `init()` must have been
     * called prior to the first call of this method.
     *
     * @return A reference to the singleton Logger instance.
     */
    static Logger& getInstance();
    
    // Public default constructor. Used only for the "dummy" logger instance
    // that is returned by getInstance() when init() has not been called,
    // to prevent crashes. Logs a warning and potentially a stack trace.
    Logger(); 

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
     * @param base_log_dir The base directory for log files.
     * @param csv_configs A vector of configurations for subsystem-specific CSV logs.
     */
    Logger(const std::string& log_file_prefix, const std::string& base_log_dir, const std::vector<SubsystemLogConfig>& csv_configs);

public: // Changed to public
    /**
     * @brief Destructor for the Logger.
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


    // Static member for the singleton instance
    static std::unique_ptr<Logger> instance_;
    static std::once_flag once_flag_;

    // Standard log members
    std::string base_log_dir_;                   ///< The base directory where log files are stored.
    std::string log_file_prefix_;                ///< The prefix used for standard log filenames.
    std::ofstream standard_log_file_;            ///< Output file stream for standard logs.


    boost::lockfree::spsc_queue<LogEntry, boost::lockfree::capacity<100>> log_queue_;             ///< Queue for asynchronous standard log processing.
    std::thread standard_writer_thread_;         ///< Dedicated thread for writing standard log messages.
    std::chrono::system_clock::time_point last_rotation_time_; ///< Timestamp of the last standard log file rotation.
    int max_standard_log_files_;

    // CSV log members


    boost::lockfree::spsc_queue<CsvLogEntry, boost::lockfree::capacity<100>> csv_log_queue_;      ///< Queue for asynchronous CSV log processing.
    std::thread csv_writer_thread_;              ///< Dedicated thread for writing CSV log messages.
    std::map<std::string, CsvLogger> csv_loggers_; ///< Map to manage CsvLogger instances per module.
    std::vector<SubsystemLogConfig> csv_subsystem_configs_; ///< Store CSV subsystem configurations

    std::atomic<bool> running_ = false;          ///< Atomic flag to control writer threads' running state.
};

// --- Global Logging Macros for Convenience ---

/// @brief Logs an informational message.
#define APP_LOG_INFO(msg) Logger::getInstance().log("INFO", msg)
/// @brief Logs a warning message.
#define APP_LOG_WARNING(msg) Logger::getInstance().log("WARNING", msg)
/// @brief Logs an error message.
#define APP_LOG_ERROR(msg) Logger::getInstance().log("ERROR", msg)
#ifdef DEBUG_MODE
/// @brief Logs a debug message.
#define APP_LOG_DEBUG(msg) Logger::getInstance().log("DEBUG", msg)
#else
#define APP_LOG_DEBUG(msg) (void)0 // No-op in release mode
#endif
#endif // UTIL_LOGGING_H
