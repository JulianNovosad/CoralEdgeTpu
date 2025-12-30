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
class ConfigLoader;
struct TelemetryFrame;

// Telemetry CSV Janitor
void write_telemetry_trace(const TelemetryFrame* buffer, size_t start_idx, size_t end_idx);

// Function to set thread name (for easier debugging)
void set_thread_name(const std::string& name);






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
 * @brief Unified "Wide Row" CSV Log Entry Schema.
 *
 * Populates Common fields for every event. Subsystem fields are sparsely populated (defaults to -1).
 */
struct CsvLogEntry {
    // --- A. Common (System & Time) ---
    long long produced_ts_epoch_ms = -1;  ///< Wall clock time when log was created
    long long call_ts_epoch_ms = -1;      ///< Trigger time (e.g. frame capture time)
    std::array<char, 32> module = {};     ///< camera|tpu|logic|encoder|sysmon
    std::array<char, 64> event = {};      ///< frame_captured, inference_done, etc.
    long long thread_id = -1;             ///< OS Thread ID

    // --- B. Camera & ISP (Input) ---
    long long cam_frame_id = -1;
    float cam_exposure_ms = -1.0f;
    float cam_isp_latency_ms = -1.0f;     ///< Copy/Conversion time
    float cam_buffer_usage_percent = -1.0f;

    // --- C. Inference (TPU) ---
    float tpu_inference_ms = -1.0f;
    float tpu_temp_c = -1.0f;
    float tpu_model_score = -1.0f;        ///< Confidence
    int tpu_class_id = -1;

    // --- D. Logic & Ballistics ---
    float logic_target_dist_m = -1.0f;
    float logic_ballistic_drop_m = -1.0f;
    float logic_windage_m = -1.0f;
    float logic_servo_x_cmd = -1.0f;      ///< Normalized 0.0-1.0
    float logic_servo_y_cmd = -1.0f;      ///< Normalized 0.0-1.0
    float logic_solution_time_ms = -1.0f;

    // --- E. Encoder (Output) ---
    float enc_process_ms = -1.0f;
    float enc_bitrate_mbps = -1.0f;
    int enc_queue_depth = -1;

    // --- F. System Health ---
    float sys_cpu_temp_c = -1.0f;
    float sys_cpu_usage_pct = -1.0f;
    float sys_ram_usage_pct = -1.0f;
    float sys_voltage_v = -1.0f;          ///< Optional

    CsvLogEntry() = default; // Default constructor sets defaults
};

/**
 * @brief Manages the single Unified Session Log file.
 */
class CsvLogger {
public:
    CsvLogger(const std::string& log_file_path);
    ~CsvLogger();

    void write_header();
    void write_entry(const CsvLogEntry& entry);
    void flush_buffer_to_disk();

private:
    std::string log_file_path_;
    std::ofstream current_log_file_;
    std::recursive_mutex file_mutex_; 
    std::vector<CsvLogEntry> buffer_; 
    std::mutex buffer_mutex_; 
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
     * @brief Initializes the singleton Logger instance with logging parameters and config for metadata.
     *
     * @param log_file_prefix The prefix for standard log filenames (e.g., "run").
     * @param base_log_dir The base directory where log files will be stored (e.g., "/home/pi/logs").
     * @param config_loader Pointer to ConfigLoader for dumping metadata.json (optional/can be passed here).
     */
    static void init(const std::string& log_file_prefix, const std::string& base_log_dir, const ConfigLoader* config_loader = nullptr);

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
     * This uses `clock_get_time` with `CLOCK_MONOTONIC_RAW` for a strictly monotonic timestamp.
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
     * @param config_loader Pointer to ConfigLoader for dumping metadata.json.
     */
    Logger(const std::string& log_file_prefix, const std::string& base_log_dir, const ConfigLoader* config_loader);

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
     * @brief The main function executed by the asynchronous log flusher thread.
     *
     * This thread periodically flushes the RAM buffers of all CsvLogger instances to disk.
     */
    void log_flusher_thread_func();

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
    
    /**
     * @brief Prunes old session directories to keep only the latest ones.
     */
    void prune_session_directories();

    /**
     * @brief Writes the metadata.json file to the current session directory.
     *
     * @param config The ConfigLoader instance to get metadata from.
     */
    void write_metadata_json(const ConfigLoader* config);


    // Static member for the singleton instance
    static std::unique_ptr<Logger> instance_;
    static std::once_flag once_flag_;

    // Standard log members
    std::string base_log_dir_;                   ///< The base directory where log files are stored.
    std::string current_session_dir_;            ///< The dedicated directory for this run (logs/session_XXX)
    std::string log_file_prefix_;                ///< The prefix used for standard log filenames.
    std::ofstream standard_log_file_;            ///< Output file stream for standard logs.


    boost::lockfree::spsc_queue<LogEntry, boost::lockfree::capacity<100>> log_queue_;             ///< Queue for asynchronous standard log processing.
    std::thread standard_writer_thread_;         ///< Dedicated thread for writing standard log messages.
    std::chrono::system_clock::time_point last_rotation_time_; ///< Timestamp of the last standard log file rotation.
    int max_standard_log_files_;

    // CSV log members
    std::thread log_flusher_thread_;              ///< Dedicated thread for periodically flushing CSV log messages.
    std::unique_ptr<CsvLogger> unified_logger_;   ///< Single Unified Logger instance
    
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