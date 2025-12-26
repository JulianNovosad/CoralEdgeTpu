#ifndef MONITOR_H
#define MONITOR_H

#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

// Forward declaration to avoid circular dependency
class Application;

class Monitor {
public:
    Monitor(Application& app);
    ~Monitor();
    
    void start();
    void stop();
    
private:
    void monitor_thread_func();
    
    Application& app_;
    std::atomic<bool> running_;
    std::thread monitor_thread_;
    
    // Run identification and time tracking
    std::string run_id_;
    std::chrono::steady_clock::time_point run_start_time_;
    
    // Variables to track previous state for throughput calculations
    long long prev_last_frame_timestamp_{0};
    long long prev_last_inference_timestamp_{0};
    long long prev_last_logic_timestamp_{0};
    
    // Previous queue depths for calculating in/out rates
    size_t prev_raw_image_queue_depth_{0};
    size_t prev_tpu_inference_queue_depth_{0};
    size_t prev_detection_overlay_queue_depth_{0};
    size_t prev_detection_logic_queue_depth_{0};
    size_t prev_overlaid_video_queue_depth_{0};
    size_t prev_h264_output_queue_depth_{0};
    
    // Flags to track first update
    bool first_update_{true};
    bool pipeline_initialized_{false}; // Track when pipeline has established flow
    int initialization_counter_{0}; // Counter to track initialization phase
    static const int INITIALIZATION_DELAY = 2; // Wait 2 seconds before showing full status
};

#endif // MONITOR_H