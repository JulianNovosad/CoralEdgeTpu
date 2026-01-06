// Verified headers: [string, vector, mutex, array, atomic...]
// Verification timestamp: 2026-01-06 17:08:04
#ifndef UTIL_LOGGING_H
#define UTIL_LOGGING_H

#include <string>
#include <vector>
#include <mutex>
#include <array>
#include <atomic>
#include <thread>
#include <fstream>
#include "lockfree_queue.h"
struct TelemetryFrame;

// Fixed-size string for hot-path logging to avoid allocations
template<size_t N>
struct FixedString {
    std::array<char, N> buffer;
    
    FixedString() { buffer.fill(0); }
    FixedString(const char* str) {
        buffer.fill(0);
        if (str) {
            for (size_t i = 0; i < N - 1 && str[i] != '\0'; ++i) {
                buffer[i] = str[i];
            }
        }
    }
    const char* data() const { return buffer.data(); }
    void assign(const char* str) {
        buffer.fill(0);
        if (str) {
            for (size_t i = 0; i < N - 1 && str[i] != '\0'; ++i) {
                buffer[i] = str[i];
            }
        }
    }
};

/**
 * @brief Telemetry entry for unified.csv (Wide Row format)
 */
struct CsvLogEntry {
    long long produced_ts_epoch_ms = 0;
    long long call_ts_epoch_ms = 0;
    FixedString<32> module;
    FixedString<64> event;
    long long thread_id = 0;

    // Camera
    int cam_frame_id = -1;
    float cam_exposure_ms = -1.0f;
    float cam_isp_latency_ms = -1.0f;
    float cam_buffer_usage_percent = -1.0f;

    // ImageProcessor
    float image_proc_ms = -1.0f;

    // TPU
    float tpu_inference_ms = -1.0f;
    float tpu_temp_c = -1.0f;
    float tpu_model_score = -1.0f;
    int tpu_class_id = -1;

    // Logic
    float logic_target_dist_m = -1.0f;
    float logic_ballistic_drop_m = -1.0f;
    float logic_windage_m = -1.0f;
    float logic_servo_x_cmd = -1.0f;
    float logic_servo_y_cmd = -1.0f;
    float logic_solution_time_ms = -1.0f;

    // Encoder
    float enc_process_ms = -1.0f;
    float enc_bitrate_mbps = -1.0f;
    int enc_queue_depth = -1;

    // System
    float sys_cpu_temp_c = -1.0f;
    float sys_cpu_usage_pct = -1.0f;
    float sys_ram_usage_pct = -1.0f;
    float sys_voltage_v = -1.0f;
};

class CsvLogger {
public:
    CsvLogger(const std::string& log_file_path);
    ~CsvLogger();
    void write_entry(const CsvLogEntry& entry);
    void flush_buffer_to_disk();

    /// @brief Get the program start time in epoch milliseconds (for relative produced_ts calculation)
    static long long get_start_time_epoch_ms() { return start_time_epoch_ms_; }
    
    /// @brief Get the program start time in monotonic milliseconds (for relative call_ts calculation)
    static uint64_t get_start_time_monotonic_ms() { return start_time_monotonic_ms_; }

private:
    void write_header();
    std::string log_file_path_;
    std::ofstream current_log_file_;
    std::vector<CsvLogEntry> buffer_;
    std::mutex buffer_mutex_;
    std::recursive_mutex file_mutex_;
    
    /// @brief Program start time in epoch milliseconds (set once, never changes) - for produced_ts
    static long long start_time_epoch_ms_;
    /// @brief Program start time in monotonic milliseconds (set once, never changes) - for call_ts
    static uint64_t start_time_monotonic_ms_;
    static std::once_flag start_time_init_flag_;
};

class ConfigLoader; // Forward declaration

class Logger {
public:
    static void init(const std::string& log_file_prefix, const std::string& base_log_dir, const ConfigLoader* config_loader = nullptr);
    static Logger& getInstance();

    void start_writer_thread();
    void stop_writer_thread();

    void log(const std::string& level, const std::string& message);
    void log_json(const std::string& key, const std::string& value);
    void log_csv(const CsvLogEntry& entry);

private:
    Logger();
    Logger(const std::string& log_file_prefix, const std::string& base_log_dir, const ConfigLoader* config_loader);
    ~Logger();

    void writer_thread_func();
    void log_flusher_thread_func();
    void prune_session_directories();
    void write_metadata_json(const ConfigLoader* config);
    void rotate_standard_log_file();
    std::string get_current_iso_time();
    long long get_raw_monotonic_time_ns();

    std::string base_log_dir_;
    std::string log_file_prefix_;
    std::string current_session_dir_;
    std::ofstream standard_log_file_;
    std::unique_ptr<CsvLogger> unified_logger_;
    
    std::atomic<bool> running_;
    std::thread standard_writer_thread_;
    std::thread log_flusher_thread_;
    
    struct LogEntry {
        std::chrono::system_clock::time_point timestamp;
        FixedString<16> level;
        FixedString<256> message;
    };
    LockFreeQueue<LogEntry, 8192> log_queue_;
    
    std::chrono::system_clock::time_point last_rotation_time_;
    int max_standard_log_files_;

    static std::unique_ptr<Logger> instance_;
    static std::once_flag once_flag_;
    
    friend class std::default_delete<Logger>;
};

// Global logging macros
#define APP_LOG_INFO(msg) Logger::getInstance().log("INFO", msg)
#define APP_LOG_WARNING(msg) Logger::getInstance().log("WARNING", msg)
#define APP_LOG_ERROR(msg) Logger::getInstance().log("ERROR", msg)
#define APP_LOG_DEBUG(msg) Logger::getInstance().log("DEBUG", msg)

// Utility functions
void set_thread_name(const std::string& name);
void set_realtime_priority(std::thread& thread, int priority, int cpu_core = -1);


inline void copy_to_array(FixedString<32>& target, const char* source) { target.assign(source); }
inline void copy_to_array(FixedString<64>& target, const char* source) { target.assign(source); }

#endif // UTIL_LOGGING_H
