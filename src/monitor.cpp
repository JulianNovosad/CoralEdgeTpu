#include "monitor.h"
#include "application.h"  // Include the full definition
#include "util_logging.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>

Monitor::Monitor(Application& app) : app_(app), running_(false) {}

Monitor::~Monitor() {
    stop();
}

void Monitor::start() {
    if (running_.exchange(true)) {
        return; // Already running
    }
    
    monitor_thread_ = std::thread(&Monitor::monitor_thread_func, this);
}

void Monitor::stop() {
    if (running_.exchange(false)) {
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
}

void Monitor::monitor_thread_func() {
    // Give the application some time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    while (running_) {
        // Clear screen and move cursor to top-left
        std::cout << "\033[2J\033[1;1H";
        
        // Print header
        std::cout << "==========================================\n";
        std::cout << "    CoralEdgeTpu System Monitor\n";
        std::cout << "==========================================\n";
        std::cout << std::endl;
        
        // Get current time
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::cout << "Current Time: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << std::endl;
        std::cout << std::endl;
        
        // Camera Module Status
        std::cout << "[Camera Module]" << std::endl;
        if (app_.get_primary_camera() && app_.get_primary_camera()->is_running()) {
            std::cout << "  Status: RUNNING" << std::endl;
            std::cout << "  Frame Rate: " << app_.get_primary_camera()->frame_rate_ << " FPS" << std::endl;
            std::cout << "  Last Frame: " << app_.get_primary_camera()->last_frame_timestamp_ << " ms" << std::endl;
        } else {
            std::cout << "  Status: STOPPED" << std::endl;
        }
        std::cout << std::endl;
        
        // Inference Module Status
        std::cout << "[Inference Module]" << std::endl;
        if (app_.get_inference_engine() && app_.get_inference_engine()->is_running()) {
            std::cout << "  Status: RUNNING" << std::endl;
            std::cout << "  Inference Rate: " << app_.get_inference_engine()->inference_rate_ << " IPS" << std::endl;
            std::cout << "  Last Inference: " << app_.get_inference_engine()->last_inference_timestamp_ << " ms" << std::endl;
        } else {
            std::cout << "  Status: STOPPED" << std::endl;
        }
        std::cout << std::endl;
        
        // Logic Module Status
        std::cout << "[Logic Module]" << std::endl;
        if (app_.get_logic_module() && app_.get_logic_module()->is_running()) {
            std::cout << "  Status: RUNNING" << std::endl;
            std::cout << "  Logic Rate: " << app_.get_logic_module()->logic_rate_ << " CPS" << std::endl;
            std::cout << "  Last Logic: " << app_.get_logic_module()->last_logic_timestamp_ << " ms" << std::endl;
        } else {
            std::cout << "  Status: STOPPED" << std::endl;
        }
        std::cout << std::endl;
        
        // Queue Depths
        std::cout << "[Queue Depths]" << std::endl;
        // Raw image for processor queue
        size_t raw_image_queue_depth = app_.raw_image_for_processor_queue_.read_available();
        size_t raw_image_queue_capacity = app_.raw_image_for_processor_queue_.write_available() + raw_image_queue_depth;
        std::cout << "  Raw Image Queue: " << raw_image_queue_depth << "/" << raw_image_queue_capacity << std::endl;
        
        // TPU inference queue
        size_t tpu_inference_queue_depth = app_.tpu_inference_queue_.read_available();
        size_t tpu_inference_queue_capacity = app_.tpu_inference_queue_.write_available() + tpu_inference_queue_depth;
        std::cout << "  TPU Inference Queue: " << tpu_inference_queue_depth << "/" << tpu_inference_queue_capacity << std::endl;
        
        // Detection results for overlay queue
        size_t detection_overlay_queue_depth = app_.detection_results_for_overlay_queue_.read_available();
        size_t detection_overlay_queue_capacity = app_.detection_results_for_overlay_queue_.write_available() + detection_overlay_queue_depth;
        std::cout << "  Detection Overlay Queue: " << detection_overlay_queue_depth << "/" << detection_overlay_queue_capacity << std::endl;
        
        // Detection results for logic queue
        size_t detection_logic_queue_depth = app_.detection_results_for_logic_queue_.read_available();
        size_t detection_logic_queue_capacity = app_.detection_results_for_logic_queue_.write_available() + detection_logic_queue_depth;
        std::cout << "  Detection Logic Queue: " << detection_logic_queue_depth << "/" << detection_logic_queue_capacity << std::endl;
        
        // Overlaid video queue
        size_t overlaid_video_queue_depth = app_.overlaid_video_queue_.read_available();
        size_t overlaid_video_queue_capacity = app_.overlaid_video_queue_.write_available() + overlaid_video_queue_depth;
        std::cout << "  Overlaid Video Queue: " << overlaid_video_queue_depth << "/" << overlaid_video_queue_capacity << std::endl;
        
        // H264 output queue
        size_t h264_output_queue_depth = app_.h264_output_queue_.read_available();
        size_t h264_output_queue_capacity = app_.h264_output_queue_.write_available() + h264_output_queue_depth;
        std::cout << "  H264 Output Queue: " << h264_output_queue_depth << "/" << h264_output_queue_capacity << std::endl;
        std::cout << std::endl;
        
        // Orientation Sensor Status
        std::cout << "[Orientation Sensor]" << std::endl;
        if (app_.get_orientation_sensor() && app_.get_orientation_sensor()->is_running()) {
            std::cout << "  Status: RUNNING" << std::endl;
            auto imu_data = app_.get_orientation_sensor()->get_latest_orientation_data();
            std::cout << "  Yaw: " << std::fixed << std::setprecision(2) << imu_data.yaw << " deg" << std::endl;
            std::cout << "  Pitch: " << std::fixed << std::setprecision(2) << imu_data.pitch << " deg" << std::endl;
            std::cout << "  Roll: " << std::fixed << std::setprecision(2) << imu_data.roll << " deg" << std::endl;
        } else {
            std::cout << "  Status: STOPPED" << std::endl;
        }
        std::cout << std::endl;
        
        // System Monitor Status
        std::cout << "[System Monitor]" << std::endl;
        if (app_.get_system_monitor() && app_.get_system_monitor()->is_running()) {
            std::cout << "  Status: RUNNING" << std::endl;
        } else {
            std::cout << "  Status: STOPPED" << std::endl;
        }
        std::cout << std::endl;
        
        // H264 Encoder Status
        std::cout << "[H264 Encoder]" << std::endl;
        if (app_.get_h264_encoder() && app_.get_h264_encoder()->is_running()) {
            std::cout << "  Status: RUNNING" << std::endl;
        } else {
            std::cout << "  Status: STOPPED" << std::endl;
        }
        std::cout << std::endl;
        
        // Image Processor Status
        std::cout << "[Image Processor]" << std::endl;
        if (app_.get_image_processor() && app_.get_image_processor()->is_running()) {
            std::cout << "  Status: RUNNING" << std::endl;
        } else {
            std::cout << "  Status: STOPPED" << std::endl;
        }
        std::cout << std::endl;
        
        // Keyboard Monitor Status
        std::cout << "[Keyboard Monitor]" << std::endl;
        if (app_.get_keyboard_monitor() && app_.get_keyboard_monitor()->is_running()) {
            std::cout << "  Status: RUNNING" << std::endl;
        } else {
            std::cout << "  Status: STOPPED" << std::endl;
        }
        std::cout << std::endl;
        
        // Pipeline Timing Information
        std::cout << "[Pipeline Timing]" << std::endl;
        // TODO: Add timing measurements for each stage of the pipeline
        std::cout << "  Capture: TODO us" << std::endl;
        std::cout << "  Queue pop: TODO us" << std::endl;
        std::cout << "  Pre-processing: TODO us" << std::endl;
        std::cout << "  Inference: TODO us" << std::endl;
        std::cout << "  Encoding: TODO us" << std::endl;
        std::cout << "  NAL handling: TODO us" << std::endl;
        std::cout << "  Total loop: TODO us" << std::endl;
        std::cout << std::endl;
        
        // Wait before next update
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}