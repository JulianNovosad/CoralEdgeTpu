#include "logic.h"
#include "util_logging.h"
#include "imu_sensor.h" // Include for full definition of IMUSensor
#include <algorithm> // For std::find_if, std::min, std::max
#include <cmath> // For M_PI, atan2, sqrt

// Initialize the static member
long LogicModule::next_track_id_ = 0;

float LogicModule::calculate_iou(const DetectionResult& det1, const DetectionResult& det2) {
    // Determine the coordinates of the intersection rectangle
    float x_left = std::max(det1.xmin, det2.xmin);
    float y_top = std::max(det1.ymin, det2.ymin);
    float x_right = std::min(det1.xmax, det2.xmax);
    float y_bottom = std::min(det1.ymax, det2.ymax);

    // If no intersection, return 0
    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    // Calculate intersection area
    float intersection_area = (x_right - x_left) * (y_bottom - y_top);

    // Calculate union area
    float area1 = (det1.xmax - det1.xmin) * (det1.ymax - det1.ymin);
    float area2 = (det2.xmax - det2.xmin) * (det2.ymax - det2.ymin);
    float union_area = area1 + area2 - intersection_area;

    // Return IoU
    return intersection_area / union_area;
}

LogicModule::LogicModule(DetectionResultsQueue& detection_input_queue, std::shared_ptr<IMUSensor> imu_sensor)
    : detection_input_queue_(detection_input_queue), imu_sensor_(imu_sensor),
      current_fallback_mode_(NORMAL_OPERATION) {
    LOG_INFO("LogicModule created. This module centralizes ballistics, object tracking, and safety logic.");
    performance_start_time_ = std::chrono::high_resolution_clock::now();
}

LogicModule::~LogicModule() {
    stop();
    LOG_INFO("LogicModule destroyed.");
}

bool LogicModule::start() {
    if (running_.exchange(true)) {
        LOG_ERROR("LogicModule is already running.");
        return false;
    }
    // detection_input_queue_.set_running(true); // boost::lockfree::spsc_queue does not have set_running
    worker_thread_ = std::thread(&LogicModule::worker_thread_func, this);
    LOG_INFO("LogicModule started.");
    return true;
}

void LogicModule::stop() {
    if (running_.exchange(false)) {
        LOG_INFO("Stopping LogicModule...");
        // detection_input_queue_.set_running(false); // boost::lockfree::spsc_queue does not have set_running
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        LOG_INFO("LogicModule stopped.");
    }
}

void LogicModule::worker_thread_func() {
    while (running_) {
        std::shared_ptr<DetectionResultBuffer> detections_buffer;
        if (detection_input_queue_.pop(detections_buffer)) {
            if (detections_buffer && detections_buffer->size > 0) {
                // Get the latest IMU data from the sensor module
                IMUData current_imu_data = imu_sensor_->get_latest_imu_data();

                // Call the actual processing logic
                process(detections_buffer->data, current_imu_data);
            }
        }
    }
}

void LogicModule::process(const std::vector<DetectionResult>& detections, const IMUData& imu_data) {
    auto process_start_time = std::chrono::high_resolution_clock::now();
    
    char log_buffer[256];
    
    // --- 1. Sensor Fusion (Basic Placeholder) ---
    // Use IMU data to potentially refine device orientation or movement estimates.
    snprintf(log_buffer, sizeof(log_buffer), "IMU Data: Accel(%.2f, %.2f, %.2f), Gyro(%.2f, %.2f, %.2f)",
             imu_data.accel_x, imu_data.accel_y, imu_data.accel_z,
             imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z);
    LOG_INFO(log_buffer);

    // --- 2. Object Tracking ---
    // A. Mark all existing tracks as unassociated for the current frame
    for (auto& track : active_tracks_) {
        track.associated_this_frame = false;
    }

    // B. Associate new detections with existing tracks (IoU-based)
    for (const auto& new_detection : detections) {
        float best_iou = 0.0f;
        TrackedObject* best_match_track = nullptr;

        for (auto& track : active_tracks_) {
            // Only consider tracks not yet associated in this frame
            if (!track.associated_this_frame) {
                float iou = calculate_iou(new_detection, track.last_detection);
                if (iou > best_iou && iou >= 0.3f) { // IoU threshold
                    best_iou = iou;
                    best_match_track = &track;
                }
            }
        }

        if (best_match_track) {
            // Update existing track
            best_match_track->last_detection = new_detection;
            // Mock a distance for now, actual distance will come from other sensors
            best_match_track->pos_z = 5.0f; // Placeholder distance
            // Update pos_x, pos_y based on new detection and distance - requires camera intrinsics
            best_match_track->last_update_time = new_detection.timestamp;
            best_match_track->hit_streak++;
            best_match_track->missed_frames = 0;
            best_match_track->associated_this_frame = true;
            snprintf(log_buffer, sizeof(log_buffer), "Track ID %ld updated (IoU: %.2f). Hit streak: %d", best_match_track->id, best_iou, best_match_track->hit_streak);
            LOG_INFO(log_buffer);
        } else {
            // Create new track
            // Mock a distance for now
            active_tracks_.emplace_back(++next_track_id_, new_detection, 5.0f); // Placeholder distance
            snprintf(log_buffer, sizeof(log_buffer), "New track ID %ld created.", active_tracks_.back().id);
            LOG_INFO(log_buffer);
        }
    }

    // C. Manage missed tracks and remove old tracks
    // Iterate in reverse to allow safe removal
    for (auto it = active_tracks_.rbegin(); it != active_tracks_.rend(); ++it) {
        if (!it->associated_this_frame) {
            it->missed_frames++;
            snprintf(log_buffer, sizeof(log_buffer), "Track ID %ld missed. Missed frames: %d", it->id, it->missed_frames);
            LOG_INFO(log_buffer);

            if (it->missed_frames > 5) { // Threshold for track removal
                snprintf(log_buffer, sizeof(log_buffer), "Track ID %ld removed due to excessive missed frames.", it->id);
                LOG_INFO(log_buffer);
                active_tracks_.erase(std::next(it).base()); // Erase from the original vector
            }
        }
    }

    // --- 3. Ballistics Calculation (using tracked object's estimated state) ---
    // Iterate through active tracks and try to predict an impact point.
    for (auto& track : active_tracks_) {
        float impact_x, impact_y, impact_z;
        // Use the actual imu_data for prediction
        if (predict_impact_point(track, imu_data, impact_x, impact_y, impact_z)) {
            snprintf(log_buffer, sizeof(log_buffer), "Predicted Impact Point for Track ID %ld: (%.2f, %.2f, %.2f)",
                     track.id, impact_x, impact_y, impact_z);
            LOG_INFO(log_buffer);
        } else {
            snprintf(log_buffer, sizeof(log_buffer), "No Impact Point Predicted for Track ID %ld.", track.id);
            LOG_INFO(log_buffer);
        }
    }

    // --- 4. Uncertainty Propagation & Safety Checks ---
    float predicted_impact_uncertainty = 0.5f; // Placeholder value
    std::string safety_message;
    
    // Iterate through active tracks and perform safety checks
    for (auto& track : active_tracks_) {
        SafetyStatus safety_status = perform_safety_and_uncertainty_checks(track, predicted_impact_uncertainty, safety_message);
        
        switch (safety_status) {
            case SAFETY_OK:
                if (current_fallback_mode_ != NORMAL_OPERATION) {
                    LOG_INFO("Returning to NORMAL_OPERATION.");
                    current_fallback_mode_ = NORMAL_OPERATION;
                }
                snprintf(log_buffer, sizeof(log_buffer), "Safety check PASSED for Track ID %ld: %s", track.id, safety_message.c_str());
                LOG_INFO(log_buffer);
                // Proceed with servo actuation if needed
                float impact_x, impact_y, impact_z;
                if (predict_impact_point(track, imu_data, impact_x, impact_y, impact_z)) {
                    issue_servo_commands(impact_x, impact_y, impact_z); // Issue commands if safe
                }
                break;
            case SAFETY_WARNING_UNCERTAINTY:
            case SAFETY_WARNING_TRACK_UNSTABLE:
                if (current_fallback_mode_ != FALLBACK_A_REDUCED_PERFORMANCE) {
                    LOG_WARNING("Activating FALLBACK_A_REDUCED_PERFORMANCE due to warning: " + safety_message);
                    current_fallback_mode_ = FALLBACK_A_REDUCED_PERFORMANCE;
                }
                snprintf(log_buffer, sizeof(log_buffer), "Safety check WARNING for Track ID %ld: %s", track.id, safety_message.c_str());
                LOG_WARNING(log_buffer);
                // Reduced performance action: e.g., only log, do not issue commands
                break;
            case SAFETY_CRITICAL_UNCERTAINTY:
            case SAFETY_CRITICAL_OTHER:
                if (current_fallback_mode_ < FALLBACK_B_WARNING_STATE) { // Promote to higher fallback if less severe mode
                    LOG_ERROR("Activating FALLBACK_B_WARNING_STATE due to critical issue: " + safety_message);
                    current_fallback_mode_ = FALLBACK_B_WARNING_STATE;
                }
                snprintf(log_buffer, sizeof(log_buffer), "Safety check CRITICAL for Track ID %ld: %s", track.id, safety_message.c_str());
                LOG_ERROR(log_buffer);
                // Critical action: e.g., halt all operations, wait for manual override
                break;
            // Add other cases for FALLBACK_C if specific conditions lead directly to it
        }
    }
    
    auto process_end_time = std::chrono::high_resolution_clock::now();
    long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(process_end_time - process_start_time).count();
    {
        std::lock_guard<std::mutex> lock(prediction_times_mutex_);
        prediction_times_ms_.push_back(duration_ms);
        total_predictions_++;
    }
}

// Constants for ballistics (placeholders for now)
const float GRAVITY = 9.81f; // m/s^2
const float MUZZLE_VELOCITY = 100.0f; // m/s (placeholder)

bool LogicModule::predict_impact_point(const TrackedObject& target, const IMUData& current_imu_data, float& out_x, float& out_y, float& out_z) {
    // This is a simplified 2D projectile motion model (vertical plane only for y, z)
    // assuming target is stationary in X-axis (for simplicity of baseline)
    // We will consider the target's current estimated Z (distance) as the range.

    // Step 1: Calculate time to reach target distance (Z)
    // Assuming projectile is fired horizontally with MUZZLE_VELOCITY
    // Time = Distance / Velocity
    float time_to_target = target.pos_z / MUZZLE_VELOCITY;

    // Step 2: Calculate vertical drop due to gravity during time_to_target
    float vertical_drop = 0.5f * GRAVITY * time_to_target * time_to_target;

    // Step 3: Predicted impact point (relative to fire point, in a simplified frame)
    // For simplicity, we assume the target's x and y are relative to the camera's center view
    // and the projectile starts at (0,0) in this frame.
    
    // out_x and out_y would be target's original x and y, but adjusted for projectile's flight.
    // For a simple model, we assume target's X and Y are unaffected by projectile motion and directly map.
    out_x = target.pos_x; 
    out_y = target.pos_y - vertical_drop; // Adjust Y for gravity drop
    out_z = target.pos_z; // Target's distance is the impact distance

    // In a real scenario, this would involve:
    // 1. Projecting target's 2D image coordinates to 3D using camera intrinsics and estimated distance.
    // 2. Integrating IMU data to get current device orientation and stabilize target position, to determine firing solution.
    // 3. Extrapolating target position based on estimated velocity.
    // 4. Solving ballistics equations to find intersection of projectile trajectory and target path.
    // 5. Considering wind, air resistance, projectile properties.

    return true; // Always "predict" for now
}

// Constants for safety checks (placeholders for now)
const int MIN_HIT_STREAK = 3; // Minimum consecutive detections for a stable track
const float MAX_PREDICTED_UNCERTAINTY = 0.75f; // Max allowed uncertainty for engagement (e.g., in meters)
const float WARNING_PREDICTED_UNCERTAINTY_THRESHOLD = 0.5f; // Uncertainty level triggering a warning
const int WARNING_HIT_STREAK_THRESHOLD = 2; // Hit streak level triggering a warning

SafetyStatus LogicModule::perform_safety_and_uncertainty_checks(const TrackedObject& target, float predicted_impact_uncertainty, std::string& safety_status_message) {
    // Check for critical uncertainty first
    if (predicted_impact_uncertainty > MAX_PREDICTED_UNCERTAINTY) {
        safety_status_message = "CRITICAL: High predicted impact uncertainty (" + std::to_string(predicted_impact_uncertainty) + "m). Exceeds " + std::to_string(MAX_PREDICTED_UNCERTAINTY) + "m.";
        return SAFETY_CRITICAL_UNCERTAINTY;
    }

    // Check for critical track instability
    if (target.hit_streak < MIN_HIT_STREAK) {
        safety_status_message = "CRITICAL: Track is unstable (hit streak: " + std::to_string(target.hit_streak) + ", required: " + std::to_string(MIN_HIT_STREAK) + ").";
        // This could be CRITICAL if below MIN_HIT_STREAK, or a WARNING if just below a higher threshold.
        // For simplicity, making it critical if it's below MIN_HIT_STREAK for now.
        return SAFETY_CRITICAL_OTHER; // Placeholder for now, refine later
    }

    // Check for warning level uncertainty
    if (predicted_impact_uncertainty > WARNING_PREDICTED_UNCERTAINTY_THRESHOLD) {
        safety_status_message = "WARNING: Predicted impact uncertainty (" + std::to_string(predicted_impact_uncertainty) + "m) above warning threshold " + std::to_string(WARNING_PREDICTED_UNCERTAINTY_THRESHOLD) + "m.";
        return SAFETY_WARNING_UNCERTAINTY;
    }

    // Check for warning level track instability
    if (target.hit_streak < WARNING_HIT_STREAK_THRESHOLD) { // E.g., if MIN_HIT_STREAK is 3, a streak of 1 or 2 is a warning
        safety_status_message = "WARNING: Track stability low (hit streak: " + std::to_string(target.hit_streak) + ", warning threshold: " + std::to_string(WARNING_HIT_STREAK_THRESHOLD) + ").";
        return SAFETY_WARNING_TRACK_UNSTABLE;
    }
    
    // Add more checks as needed, e.g., target class, distance, environmental factors
    // if (target.last_detection.class_id == FRIENDLY_CLASS_ID) {
    //     safety_status_message = "Target identified as friendly.";
    //     return SAFETY_CRITICAL_OTHER; // Or a specific SAFETY_CRITICAL_FRIENDLY_FIRE
    // }

    safety_status_message = "All safety checks passed.";
    return SAFETY_OK;
}

void LogicModule::issue_servo_commands(float target_x, float target_y, float target_z) {
    // This is a placeholder for actual servo/actuator control.
    // In a real system, this would translate target_x, target_y, target_z into
    // specific angles or pulse widths for connected servos/motors,
    // potentially through a hardware abstraction layer (e.g., I2C communication with PCA9685).

    char log_buffer[256];
    snprintf(log_buffer, sizeof(log_buffer), "Issuing servo commands for target: (%.2f, %.2f, %.2f)",
             target_x, target_y, target_z);
    LOG_INFO(log_buffer);

    // Placeholder: Convert 3D target to 2D servo angles (very simplified)
    float pan_angle = atan2(target_x, target_z) * 180.0f / M_PI; // Pan based on X and Z
    float tilt_angle = atan2(target_y, target_z) * 180.0f / M_PI; // Tilt based on Y and Z

    snprintf(log_buffer, sizeof(log_buffer), "Simulated servo angles: Pan=%.2f deg, Tilt=%.2f deg",
             pan_angle, tilt_angle);
    LOG_INFO(log_buffer);

    // Further implementation would involve:
    // - Inverse kinematics to convert 3D point to joint angles.
    // - Sending commands via I2C to a PWM controller (e.g., PCA9685).
    // - Incorporating PID control for smooth and accurate movements.
}

void LogicModule::get_performance_metrics() {
    std::lock_guard<std::mutex> lock(prediction_times_mutex_);

    if (total_predictions_ == 0) {
        LOG_INFO("LogicModule: No predictions recorded for performance metrics.");
        return;
    }

    double average_duration_ms = 0;
    for (long long duration : prediction_times_ms_) {
        average_duration_ms += duration;
    }
    average_duration_ms /= total_predictions_;
    double average_fps = 1000.0 / average_duration_ms; // Using FPS for predictions as an analogy

    double sum_sq_diff = 0;
    for (long long duration : prediction_times_ms_) {
        sum_sq_diff += (duration - average_duration_ms) * (duration - average_duration_ms);
    }
    double std_dev_ms = std::sqrt(sum_sq_diff / total_predictions_);

    std::sort(prediction_times_ms_.begin(), prediction_times_ms_.end());
    size_t percentile_99_index = static_cast<size_t>(std::round(total_predictions_ * 0.99));
    size_t percentile_95_index = static_cast<size_t>(std::round(total_predictions_ * 0.95));
    size_t percentile_50_index = static_cast<size_t>(std::round(total_predictions_ * 0.50));

    long long p99_latency_ms = prediction_times_ms_[std::min(percentile_99_index, static_cast<size_t>(total_predictions_ - 1))];
    long long p95_latency_ms = prediction_times_ms_[std::min(percentile_95_index, static_cast<size_t>(total_predictions_ - 1))];
    long long p50_latency_ms = prediction_times_ms_[std::min(percentile_50_index, static_cast<size_t>(total_predictions_ - 1))];

    LOG_CSV("LogicModule", "Prediction", p50_latency_ms, p95_latency_ms, p99_latency_ms, 0.0, average_fps);
    LOG_INFO("--- LogicModule Performance Metrics (Time-to-Prediction) ---");
    LOG_INFO("  Total Predictions: " + std::to_string(total_predictions_));
    LOG_INFO("  Average Prediction Rate (FPS equivalent): " + std::to_string(average_fps));
    LOG_INFO("  Average Latency: " + std::to_string(average_duration_ms) + " ms");
    LOG_INFO("  Latency Std Dev: " + std::to_string(std_dev_ms) + " ms");
    LOG_INFO("  50th Percentile Latency: " + std::to_string(p50_latency_ms) + " ms");
    LOG_INFO("  95th Percentile Latency: " + std::to_string(p95_latency_ms) + " ms");
    LOG_INFO("  99th Percentile Latency: " + std::to_string(p99_latency_ms) + " ms");
    LOG_INFO("----------------------------------------------------------");

    prediction_times_ms_.clear();
    total_predictions_ = 0;
}