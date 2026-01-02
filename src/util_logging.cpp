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
#include "config_loader.h" 
#include <iostream>       // For std::cout, std::cerr
#include <filesystem>     // C++17 for creating directories
#include <sstream>        // For std::ostringstream
#include <iomanip>        // For std::put_time
#include <algorithm>      // For std::sort
#include <cerrno>         // For errno
#include <string>         // Explicitly include <string> for std::string
#include "pipeline_structs.h"

#ifdef __linux__
#include <sys/prctl.h> // For prctl(PR_SET_NAME)
#endif

// set_thread_name implementation
void write_telemetry_trace(const TelemetryFrame* buffer, size_t start_idx, size_t end_idx) {
    if (!buffer) return;

    FILE* f = fopen("/tmp/aurore_trace.csv", "a");
    if (!f) {
        std::cerr << "[ERROR] Failed to open /tmp/aurore_trace.csv for writing: " << strerror(errno) << std::endl;
        return;
    }

    // Check if file is empty to write header
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "Frame,Capture,InfTime,LogicTime,X,Y,Z,State,Hit\n");
    }

    for (size_t i = start_idx; i < end_idx; ++i) {
        const TelemetryFrame& frame = buffer[i % 10000]; // Fixed size 10000
        
        // Calculate durations
        long long inf_time = (long long)(frame.t_inf_end - frame.t_inf_start);
        long long logic_time = (long long)(frame.t_logic_end - frame.t_logic_start);

        fprintf(f, "%lu,%lu,%lld,%lld,%.2f,%.2f,%.2f,%d,%d\n",
                frame.frame_id,
                frame.t_capture,
                inf_time,
                logic_time,
                frame.target_x,
                frame.target_y,
                frame.target_z,
                frame.state,
                frame.hit_scan ? 1 : 0);
    }

    fclose(f);
}

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

namespace fs = std::filesystem; ///< Alias for std::filesystem for brevity. 

// Static members initialization
std::unique_ptr<Logger> Logger::instance_;
std::once_flag Logger::once_flag_;

// CsvLogger static member initialization
long long CsvLogger::start_time_epoch_ms_ = 0;
uint64_t CsvLogger::start_time_monotonic_ms_ = 0;
std::once_flag CsvLogger::start_time_init_flag_;

// =============================================================================
// CsvLogger Implementation (Unified Session Log)
// =============================================================================

CsvLogger::CsvLogger(const std::string& log_file_path)
    : log_file_path_(log_file_path) {
    
    // Initialize program start times (once, thread-safe)
    std::call_once(start_time_init_flag_, []() {
        // Epoch time for produced_ts (system_clock)
        start_time_epoch_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Monotonic time for call_ts (CLOCK_MONOTONIC_RAW)
        struct timespec ts;
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0) {
            start_time_monotonic_ms_ = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
        } else {
            // Fallback to steady_clock
            auto now = std::chrono::steady_clock::now();
            start_time_monotonic_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();
        }
    });
    
    // Ensure parent directory exists (should be handled by Logger::init, but safety first)
    fs::path p(log_file_path_);
    if (!fs::exists(p.parent_path())) {
        fs::create_directories(p.parent_path());
    }

    // Open in append mode
    current_log_file_.open(log_file_path_, std::ios_base::out | std::ios_base::app);
    if (!current_log_file_.is_open()) {
        std::cerr << "Failed to open Unified CSV log file: " << log_file_path_ << " - Reason: " << strerror(errno) << std::endl;
    } else {
        // Write header if file is empty
        if (fs::file_size(log_file_path_) == 0) {
            write_header();
        }
    }
}

CsvLogger::~CsvLogger() {
    std::cerr << "CsvLogger: Destructor called, flushing final buffer..." << std::endl;
    flush_buffer_to_disk(); // Ensure data is saved before closing
    std::lock_guard<std::recursive_mutex> lock(file_mutex_);
    if (current_log_file_.is_open()) {
        current_log_file_.close();
    }
}

void CsvLogger::write_header() {
    std::lock_guard<std::recursive_mutex> lock(file_mutex_);
    if (current_log_file_.is_open()) {
        // Unified "Wide Row" Header
        current_log_file_ << "produced_ts_epoch_ms,call_ts_epoch_ms,module,event,thread_id,"
                          << "cam_frame_id,cam_exposure_ms,cam_isp_latency_ms,cam_buffer_usage_percent,"
                          << "image_proc_ms,"
                          << "tpu_inference_ms,tpu_temp_c,tpu_model_score,tpu_class_id,"
                          << "logic_target_dist_m,logic_ballistic_drop_m,logic_windage_m,logic_servo_x_cmd,logic_servo_y_cmd,logic_solution_time_ms,"
                          << "enc_process_ms,enc_bitrate_mbps,enc_queue_depth,"
                          << "sys_cpu_temp_c,sys_cpu_usage_pct,sys_ram_usage_pct,sys_voltage_v\n";
        current_log_file_.flush();
    }
}

void CsvLogger::write_entry(const CsvLogEntry& entry) {
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        buffer_.push_back(entry);
    }
    if (buffer_.size() % 10 == 0) {
        std::cerr << "CsvLogger: Buffer size reached " << buffer_.size() << ", forcing flush." << std::endl;
        flush_buffer_to_disk();
    }
}

void CsvLogger::flush_buffer_to_disk() {
    std::lock_guard<std::mutex> buffer_lock(buffer_mutex_);
    if (buffer_.empty()) {
        return;
    }

    size_t count = buffer_.size();
    std::cerr << "CsvLogger: Flushing " << count << " entries to " << log_file_path_ << std::endl;

    std::lock_guard<std::recursive_mutex> file_lock(file_mutex_);
    if (!current_log_file_.is_open()) {
        // Try to re-open
        current_log_file_.open(log_file_path_, std::ios_base::out | std::ios_base::app);
        if (!current_log_file_.is_open()) {
            std::cerr << "Failed to reopen CSV log file. Entries dropped." << std::endl;
            buffer_.clear();
            return;
        }
    }

    for (const auto& e : buffer_) {
        // Convert timestamps to relative milliseconds from program start
        // produced_ts comes from system_clock (epoch), so subtract epoch start
        long long relative_produced_ts = e.produced_ts_epoch_ms - start_time_epoch_ms_;
        // call_ts comes from CLOCK_MONOTONIC_RAW, so subtract monotonic start
        long long relative_call_ts = e.call_ts_epoch_ms - static_cast<long long>(start_time_monotonic_ms_);
        
        // Ensure non-negative (in case of clock issues or entries from before start time was set)
        if (relative_produced_ts < 0) relative_produced_ts = 0;
        if (relative_call_ts < 0) relative_call_ts = 0;
        
        // Format: Common (Always present)
        current_log_file_ << relative_produced_ts << ","
                          << relative_call_ts << ","
                          << e.module.data() << ","
                          << e.event.data() << ","
                          << e.thread_id << ",";
        
        // Format: Camera
        if (e.cam_frame_id != -1) { current_log_file_ << e.cam_frame_id << ","; } else { current_log_file_ << "NaN,"; }
        if (e.cam_exposure_ms != -1.0f) { current_log_file_ << e.cam_exposure_ms << ","; } else { current_log_file_ << "NaN,"; }
        if (e.cam_isp_latency_ms != -1.0f) { current_log_file_ << e.cam_isp_latency_ms << ","; } else { current_log_file_ << "NaN,"; }
        if (e.cam_buffer_usage_percent != -1.0f) { current_log_file_ << e.cam_buffer_usage_percent << ","; } else { current_log_file_ << "NaN,"; }

        // Format: ImageProcessor
        if (e.image_proc_ms != -1.0f) { current_log_file_ << e.image_proc_ms << ","; } else { current_log_file_ << "NaN,"; }

        // Format: TPU
        if (e.tpu_inference_ms != -1.0f) { current_log_file_ << e.tpu_inference_ms << ","; } else { current_log_file_ << "NaN,"; }
        if (e.tpu_temp_c != -1.0f) { current_log_file_ << e.tpu_temp_c << ","; } else { current_log_file_ << "NaN,"; }
        if (e.tpu_model_score != -1.0f) { current_log_file_ << e.tpu_model_score << ","; } else { current_log_file_ << "NaN,"; }
        if (e.tpu_class_id != -1) { current_log_file_ << e.tpu_class_id << ","; } else { current_log_file_ << "NaN,"; }

        // Format: Logic
        if (e.logic_target_dist_m != -1.0f) { current_log_file_ << e.logic_target_dist_m << ","; } else { current_log_file_ << "NaN,"; }
        if (e.logic_ballistic_drop_m != -1.0f) { current_log_file_ << e.logic_ballistic_drop_m << ","; } else { current_log_file_ << "NaN,"; }
        if (e.logic_windage_m != -1.0f) { current_log_file_ << e.logic_windage_m << ","; } else { current_log_file_ << "NaN,"; }
        if (e.logic_servo_x_cmd != -1.0f) { current_log_file_ << e.logic_servo_x_cmd << ","; } else { current_log_file_ << "NaN,"; }
        if (e.logic_servo_y_cmd != -1.0f) { current_log_file_ << e.logic_servo_y_cmd << ","; } else { current_log_file_ << "NaN,"; }
        if (e.logic_solution_time_ms != -1.0f) { current_log_file_ << e.logic_solution_time_ms << ","; } else { current_log_file_ << "NaN,"; }

        // Format: Encoder
        if (e.enc_process_ms != -1.0f) { current_log_file_ << e.enc_process_ms << ","; } else { current_log_file_ << "NaN,"; }
        if (e.enc_bitrate_mbps != -1.0f) { current_log_file_ << e.enc_bitrate_mbps << ","; } else { current_log_file_ << "NaN,"; }
        if (e.enc_queue_depth != -1) { current_log_file_ << e.enc_queue_depth << ","; } else { current_log_file_ << "NaN,"; }

        // Format: System
        if (e.sys_cpu_temp_c != -1.0f) { current_log_file_ << e.sys_cpu_temp_c << ","; } else { current_log_file_ << "NaN,"; }
        if (e.sys_cpu_usage_pct != -1.0f) { current_log_file_ << e.sys_cpu_usage_pct << ","; } else { current_log_file_ << "NaN,"; }
        if (e.sys_ram_usage_pct != -1.0f) { current_log_file_ << e.sys_ram_usage_pct << ","; } else { current_log_file_ << "NaN,"; }
        if (e.sys_voltage_v != -1.0f) { current_log_file_ << e.sys_voltage_v; } else { current_log_file_ << "NaN"; }
        
        current_log_file_ << "\n";
    }
    
    current_log_file_.flush();
    buffer_.clear();
}

// =============================================================================
// Logger Implementation
// =============================================================================

// Definition for the public default constructor
Logger::Logger()
    : base_log_dir_("."), log_file_prefix_("dummy_log"), last_rotation_time_(std::chrono::system_clock::now()), max_standard_log_files_(1), running_(false) {
}

void Logger::init(const std::string& log_file_prefix, const std::string& base_log_dir, const ConfigLoader* config_loader) {
    std::call_once(once_flag_, [&]() {
        instance_.reset(new Logger(log_file_prefix, base_log_dir, config_loader));
    });
}

Logger& Logger::getInstance() {
    if (!instance_) {
        static Logger dummy_logger; 
        return dummy_logger;
    }
    return *instance_.get();
}

Logger::Logger(const std::string& log_file_prefix, const std::string& base_log_dir, const ConfigLoader* config_loader)
    : base_log_dir_(base_log_dir), log_file_prefix_(log_file_prefix), last_rotation_time_(std::chrono::system_clock::now()), max_standard_log_files_(3) {
    
    // 1. Ensure Base Directory Exists
    if (!fs::exists(base_log_dir_)) {
        fs::create_directories(base_log_dir_);
    }
    
    // 2. Prune Old Sessions
    prune_session_directories();

    // 3. Create New Session Directory
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_c);
    std::ostringstream oss;
    oss << "session_" << std::put_time(now_tm, "%Y%m%d_%H%M%S");
    std::string session_name = oss.str();
    
    fs::path session_path = fs::path(base_log_dir_) / session_name;
    fs::create_directories(session_path);
    current_session_dir_ = session_path.string();
    APP_LOG_INFO("Logger: Created session directory: " + current_session_dir_);

    // 4. Initialize Standard Log in Session Dir
    fs::path std_log_path = session_path / (log_file_prefix_ + ".json");
    standard_log_file_.open(std_log_path.string(), std::ios_base::out | std::ios_base::trunc);
    if (!standard_log_file_.is_open()) {
        std::cerr << "Failed to open standard log file: " << std_log_path << std::endl;
    }

    // 5. Write Metadata Sidecar
    if (config_loader) {
        write_metadata_json(config_loader);
    } else {
        APP_LOG_WARNING("Logger: No ConfigLoader provided. Skipping metadata.json generation.");
    }

    // 6. Initialize Unified CSV Logger
    fs::path unified_csv_path = session_path / "unified.csv";
    unified_logger_ = std::make_unique<CsvLogger>(unified_csv_path.string());
    std::cerr << "Logger: Initialized in directory: " << current_session_dir_ << std::endl;
}

Logger::~Logger() {
    stop_writer_thread(); 
    if (standard_log_file_.is_open()) {
        standard_log_file_.close(); 
    }
}

void Logger::start_writer_thread() {
    if (!running_.exchange(true)) { 
        APP_LOG_INFO("Logger: Creating standard writer thread.");
        standard_writer_thread_ = std::thread(&Logger::writer_thread_func, this);
        APP_LOG_INFO("Logger: Creating CSV log flusher thread.");
        log_flusher_thread_ = std::thread(&Logger::log_flusher_thread_func, this);
    }
}

void Logger::stop_writer_thread() {
    if (running_.exchange(false)) { 
        if (standard_writer_thread_.joinable()) {
            standard_writer_thread_.join(); 
        }
        if (log_flusher_thread_.joinable()) {
            log_flusher_thread_.join(); 
        }
        if (unified_logger_) {
            unified_logger_->flush_buffer_to_disk();
        }
    }
}

void Logger::log(const std::string& level, const std::string& message) {
    log_queue_.push(std::move(LogEntry{std::chrono::system_clock::now(), level.c_str(), message.c_str()})); 
}

void Logger::log_json(const std::string& key, const std::string& value) {
    std::string json_message = "{\"" + key + "\": " + value + "}";
    log_queue_.push(std::move(LogEntry{std::chrono::system_clock::now(), "JSON", json_message.c_str()})); 
}

void Logger::log_csv(const CsvLogEntry& entry) {
    if (unified_logger_) {
        unified_logger_->write_entry(entry);
    }
}

void Logger::writer_thread_func() {
    set_thread_name("StdLogWriter");
    // APP_LOG_INFO("Logger: Standard writer thread started."); // Can't log here, infinite recursion potentially if buffer full?
    // Actually safe to log to queue, but let's avoid noise.

    LogEntry entry; 
    while (running_.load(std::memory_order_acquire)) { 
        if (log_queue_.pop(entry)) { 
            std::cerr << "[" << entry.level.data() << "] " << entry.message.data() << std::endl;

            if (standard_log_file_.is_open()) {
                standard_log_file_ << "{\"timestamp\":\"" << get_current_iso_time() << "\", \"level\":\"" << entry.level.data() << "\", \"message\":\"" << entry.message.data() << "\"}" << std::endl;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    while (log_queue_.pop(entry)) {
        std::cout << "[" << entry.level.data() << "] " << entry.message.data() << std::endl;
        if (standard_log_file_.is_open()) {
            standard_log_file_ << "{\"timestamp\":\"" << get_current_iso_time() << "\", \"level\":\"" << entry.level.data() << "\", \"message\":\"" << entry.message.data() << "\"}" << std::endl;
        }
    }
}

void Logger::rotate_standard_log_file() {
    // Session based rotation logic handles this mostly, but if we wanted to split large standard logs within a session:
    // Implemented simplistic version here or assume session is short enough.
    // For now, no-op or simple close/reopen if needed.
    // Current requirement: 3 latest run folders. The logic is in the constructor.
}

std::string Logger::get_current_iso_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::gmtime(&now_c); 
    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

long long Logger::get_raw_monotonic_time_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == -1) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

void Logger::log_flusher_thread_func() {
    set_thread_name("CsvFlusher");
    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
        if (unified_logger_) {
            unified_logger_->flush_buffer_to_disk();
        }
    }
    if (unified_logger_) {
        unified_logger_->flush_buffer_to_disk();
    }
}

void Logger::prune_session_directories() {
    std::vector<fs::path> sessions;
    try {
        for (const auto& entry : fs::directory_iterator(base_log_dir_)) {
            if (entry.is_directory() && entry.path().filename().string().find("session_") == 0) {
                sessions.push_back(entry.path());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Logger: Error iterating log dir for pruning: " << e.what() << std::endl;
    }

    std::sort(sessions.begin(), sessions.end()); // Lexicographical sort works for YYYYMMDD_HHMMSS

    // Keep latest 2 (so we can add 1 more to make 3)
    // Actually mandate says: "maintain exactly the 3 latest run folders... pruning the oldest on startup"
    // So if we have 3, delete 1. If we have 10, delete 8.
    // We want to end up with 3 *after* creation. So delete until we have 2.
    
    while (sessions.size() >= 3) {
        APP_LOG_INFO("Logger: Pruning old session: " + sessions.front().string());
        fs::remove_all(sessions.front());
        sessions.erase(sessions.begin());
    }
}

void Logger::write_metadata_json(const ConfigLoader* config) {
    if (!config) return; 
    
    fs::path meta_path = fs::path(current_session_dir_) / "metadata.json";
    std::ofstream meta_file(meta_path);
    if (meta_file.is_open()) {
        nlohmann::json meta;
        
        // System Info
        meta["compile_timestamp"] = __DATE__ " " __TIME__;
        // meta["git_hash"] = GIT_HASH; // Needs build system support, optional
        
        // Configuration
        meta["config"] = config->get_json_config();
        
        // Specific Ballistics Context ("Diameter of the Sun")
        meta["ballistics"]["muzzle_velocity_mps"] = config->get_muzzle_velocity_mps();
        meta["ballistics"]["ballistic_coefficient"] = config->get_ballistic_coefficient_si();
        meta["ballistics"]["zero_distance_m"] = config->get_zero_distance_m();
        meta["ballistics"]["sight_height_m"] = config->get_sight_height_m();
        meta["ballistics"]["env_temp_c"] = config->get_temperature_c();
        meta["ballistics"]["env_pressure_pa"] = config->get_air_pressure_pa();

        meta_file << meta.dump(4);
    } else {
        std::cerr << "Logger: Failed to write metadata.json" << std::endl;
    }
}