#include "config_loader.h"
#include "logic.h"
#include "util_logging.h"
#include "orientation_sensor.h"
#include <algorithm>
#include <cmath>
#include <iostream> // Added for std::cerr/cout

// --- Fysieke en wiskundige constanten ---
constexpr float GRAVITY_CONST = 9.81f;      // Zwaartekrachtversnelling in m/s^2
constexpr float R_DRY_AIR = 287.058f;   // Specifieke gasconstante voor droge lucht in J/(kg·K)
constexpr float PI = 3.14159265358979323846f;

// Initializeer de statische leden
long LogicModule::next_track_id_ = 0;

// --- Implementatie van BallisticsSolver ---

BallisticsSolver::BallisticsSolver(const BallisticProfile& profile) : profile_(profile) {
    zero_pitch_rad_ = calculate_zero_pitch();
    APP_LOG_INFO("BallisticsSolver created and zero pitch calculated.");
}

float BallisticsSolver::get_air_density() const {
    float temp_kelvin = profile_.temperature_c + 273.15f;
    return profile_.air_pressure_pa / (R_DRY_AIR * temp_kelvin);
}

Vec3 BallisticsSolver::drag_force(const Vec3& velocity, float air_density) {
    float v = velocity.magnitude();
    if (v < 1e-6) return {0.0f, 0.0f, 0.0f}; // Vermijd deling door nul
    float drag_magnitude = 0.5f * air_density * v * v * (profile_.bullet_mass_kg / profile_.ballistic_coefficient_si);
    return velocity * (-drag_magnitude / v);
}

BallisticState BallisticsSolver::derivatives(const BallisticState& state, float air_density) {
    Vec3 gravitational_force = {0.0f, -GRAVITY_CONST * profile_.bullet_mass_kg, 0.0f};
    Vec3 drag = drag_force(state.velocity, air_density);
    Vec3 total_force = gravitational_force + drag;
    Vec3 acceleration = total_force * (1.0f / profile_.bullet_mass_kg);

    return {{state.velocity}, {acceleration}};
}

BallisticState BallisticsSolver::rk4_step(const BallisticState& state, float dt, float air_density) {
    BallisticState k1 = derivatives(state, air_density);
    BallisticState k2 = derivatives({state.position + k1.position * (dt / 2.0f), state.velocity + k1.velocity * (dt / 2.0f)}, air_density);
    BallisticState k3 = derivatives({state.position + k2.position * (dt / 2.0f), state.velocity + k2.velocity * (dt / 2.0f)}, air_density);
    BallisticState k4 = derivatives({state.position + k3.position * dt, state.velocity + k3.velocity * dt}, air_density);

    Vec3 pos_next = state.position + (k1.position + k2.position*2.0f + k3.position*2.0f + k4.position) * (dt / 6.0f);
    Vec3 vel_next = state.velocity + (k1.velocity + k2.velocity*2.0f + k3.velocity*2.0f + k4.velocity) * (dt / 6.0f);
    
    return {pos_next, vel_next};
}

std::vector<BallisticState> BallisticsSolver::calculate_trajectory(float initial_pitch, float max_distance, float time_step_override) {
    std::vector<BallisticState> trajectory;
    float air_density = get_air_density();

    // Determine an appropriate time_step if not overridden
    float actual_time_step = time_step_override;
    if (actual_time_step == 0.0f) { // Using 0.0f to indicate default, as 0.01f is an actual value
        // Calculate time_step to ensure max_distance is covered in a reasonable number of steps.
        // Let's target roughly 500 steps for the given max_distance.
        float target_steps = 500.0f;
        actual_time_step = max_distance / (profile_.muzzle_velocity_mps * target_steps);
        
        // Ensure time_step is within a reasonable range for stability and performance
        actual_time_step = std::max(0.0001f, std::min(0.1f, actual_time_step)); // min 0.1ms, max 100ms
    }

    BallisticState current_state;
    current_state.position = {0, -profile_.sight_height_m, 0};
    current_state.velocity = {
        profile_.muzzle_velocity_mps * std::cos(initial_pitch),
        profile_.muzzle_velocity_mps * std::sin(initial_pitch),
        0.0f
    };
    trajectory.push_back(current_state);

    while (current_state.position.x < max_distance && current_state.position.y >= -profile_.sight_height_m) {
        current_state = rk4_step(current_state, actual_time_step, air_density); // Use actual_time_step
        trajectory.push_back(current_state);
        if (trajectory.size() > 20000) { // Safety stop, verhoogd naar 20000
            APP_LOG_WARNING("Trajectory calculation exceeded 20000 steps. Max distance: " + std::to_string(max_distance) + "m, Time step: " + std::to_string(actual_time_step) + "s. Last Pos Y: " + std::to_string(current_state.position.y) + "m");
            break;
        }
    }
    return trajectory;
}

float BallisticsSolver::calculate_zero_pitch() {
    APP_LOG_INFO("Calculating zero pitch for " + std::to_string(profile_.zero_distance_m) + "m...");

    float low_angle_rad = -0.05f; // -~3 degrees, a reasonable lower bound
    float high_angle_rad = 0.05f; // +~3 degrees, a reasonable upper bound
    
    constexpr int max_iterations = 30;
    constexpr float tolerance_m = 0.001f; // 1 mm

    for (int i = 0; i < max_iterations; ++i) {
        float mid_angle_rad = (low_angle_rad + high_angle_rad) / 2.0f;
        // Use profile_.zero_distance_m directly as max_distance
        auto trajectory = calculate_trajectory(mid_angle_rad, profile_.zero_distance_m);

        if (trajectory.empty()) {
            APP_LOG_ERROR("Failed to calculate trajectory during zero pitch calculation for mid_angle_rad: " + std::to_string(mid_angle_rad));
            // If trajectory is empty, something is wrong. Return 0.0f.
            return 0.0f; 
        }

        float height_at_zero = 0.0f;
        bool reached_zero_distance = false;
        for (size_t j = 1; j < trajectory.size(); ++j) {
            if (trajectory[j].position.x >= profile_.zero_distance_m) {
                // Lineaire interpolatie voor de hoogte
                const auto& p1 = trajectory[j-1].position;
                const auto& p2 = trajectory[j].position;
                float t = (profile_.zero_distance_m - p1.x) / (p2.x - p1.x);
                height_at_zero = p1.y + t * (p2.y - p1.y);
                reached_zero_distance = true;
                break;
            }
        }
        
        // If zero_distance_m was not reached within the trajectory,
        // it means the angle was too low (or max_distance too small).
        // We need to adjust binary search accordingly.
        if (!reached_zero_distance) {
            // If it didn't reach zero distance, it means it hit the ground too early or too flat.
            // This implies the angle was too low, so treat it as if height_at_zero was < 0.
            low_angle_rad = mid_angle_rad;
            continue; // Skip to next iteration
        }

        // Vergelijk de hoogte met de zichtlijn (die op y=0 ligt in ons coördinatensysteem)
        if (std::abs(height_at_zero) < tolerance_m) {
            APP_LOG_INFO("Zero pitch found after " + std::to_string(i + 1) + " iterations: " + std::to_string(mid_angle_rad) + " rad");
            return mid_angle_rad;
        }

        if (height_at_zero < 0) { // Kogel te laag, dus hoek moet omhoog
            low_angle_rad = mid_angle_rad;
        } else { // Kogel te hoog, dus hoek moet omlaag
            high_angle_rad = mid_angle_rad;
        }
    }

    APP_LOG_WARNING("Zero pitch calculation did not converge within " + std::to_string(max_iterations) + " iterations. Best estimate: " + std::to_string((low_angle_rad + high_angle_rad) / 2.0f) + " rad.");
    return (low_angle_rad + high_angle_rad) / 2.0f; // Geef de beste schatting terug
}

// --- Implementatie van LogicModule ---

float LogicModule::calculate_iou(const DetectionResult& det1, const DetectionResult& det2) {
    float x_left = std::max(det1.xmin, det2.xmin);
    float y_top = std::max(det1.ymin, det2.ymin);
    float x_right = std::min(det1.xmax, det2.xmax);
    float y_bottom = std::min(det1.ymax, det2.ymax);
    if (x_right < x_left || y_bottom < y_top) return 0.0f;
    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float area1 = (det1.xmax - det1.xmin) * (det1.ymax - det1.ymin);
    float area2 = (det2.xmax - det2.xmin) * (det2.ymax - det2.ymin);
    float union_area = area1 + area2 - intersection_area;
    return intersection_area / union_area;
}

LogicModule::LogicModule(DetectionResultsQueue& detection_input_queue, std::shared_ptr<OrientationSensor> orientation_sensor, const ConfigLoader& config)
    : detection_input_queue_(detection_input_queue),
      orientation_sensor_(orientation_sensor),
      config_(config),
      max_active_tracks_(config.get_max_active_tracks()),
      track_iou_threshold_(config.get_track_iou_threshold()),
      track_missed_frames_threshold_(config.get_track_missed_frames_threshold()),
      min_track_confidence_(config.get_min_track_confidence()),
      current_fallback_mode_(NORMAL_OPERATION) {
    
    BallisticProfile profile = {
        .muzzle_velocity_mps = config.get_muzzle_velocity_mps(),
        .bullet_mass_kg = config.get_bullet_mass_kg(),
        .ballistic_coefficient_si = config.get_ballistic_coefficient_si(),
        .sight_height_m = config.get_sight_height_m(),
        .zero_distance_m = config.get_zero_distance_m(),
        .air_pressure_pa = config.get_air_pressure_pa(),
        .temperature_c = config.get_temperature_c()
    };
    ballistics_solver_ = std::make_unique<BallisticsSolver>(profile);

    APP_LOG_INFO("LogicModule created with 3D Ballistics Solver, configured from file.");
}

LogicModule::~LogicModule() { stop(); APP_LOG_INFO("LogicModule destroyed."); }
bool LogicModule::start() {
    if (running_.exchange(true)) { APP_LOG_ERROR("LogicModule is already running."); return false; }
    worker_thread_ = std::thread(&LogicModule::worker_thread_func, this);
    APP_LOG_INFO("LogicModule started.");
    return true;
}
void LogicModule::stop() {
    if (running_.exchange(false)) {
        APP_LOG_INFO("Stopping LogicModule...");
        if (worker_thread_.joinable()) worker_thread_.join();
        APP_LOG_INFO("LogicModule stopped.");
    }
}

void LogicModule::worker_thread_func() {
    APP_LOG_INFO("LogicModule worker thread started.");
    while (running_) {
        std::shared_ptr<DetectionResultBuffer> detections_buffer;
        if (detection_input_queue_.pop(detections_buffer)) {
            APP_LOG_INFO("LogicModule: Pop successful. Detections buffer received.");
            long long call_ts = 0;
            if (detections_buffer && detections_buffer->size > 0) {
                APP_LOG_INFO("LogicModule: Detections buffer is valid with size: " + std::to_string(detections_buffer->size));
                // Use the timestamp of the first detection as the call_ts
                call_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            detections_buffer->data[0].timestamp.time_since_epoch()).count();
            } else {
                APP_LOG_WARNING("LogicModule: Pop successful, but detections_buffer is null or empty. Using current time for call_ts.");
                call_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count();
            }

            if (detections_buffer && detections_buffer->size > 0) {
                OrientationData current_imu_data = orientation_sensor_->get_latest_orientation_data();
                
                auto t_start = std::chrono::high_resolution_clock::now();
                process(detections_buffer->data, current_imu_data);
                auto t_end = std::chrono::high_resolution_clock::now();
                
                long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
                
                CsvLogEntry entry;
                entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                copy_to_array(entry.module, "LogicModule");
                entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
                copy_to_array(entry.event, "logic_cycle_done"); // Event changed to reflect per-cycle
                entry.call_ts_epoch_ms = call_ts;
                entry.logic_metric_ballistics = static_cast<float>(duration_ms); // Total processing time for this cycle
                entry.logic_metric_hit_scan = static_cast<float>(current_hit_scan_count_);
                entry.logic_metric_servo_actuation = static_cast<float>(current_servo_command_count_);
                Logger::getInstance().log_csv(entry);
            }
        } else {
            // Log that pop failed, and sleep to prevent busy-waiting
            if (running_) { // Only log if still intended to be running
                APP_LOG_DEBUG("LogicModule: Pop failed (queue empty or stopped). Sleeping for 10ms.");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    APP_LOG_INFO("LogicModule worker thread stopped.");
}

void LogicModule::process(const std::vector<DetectionResult>& detections, const OrientationData& imu_data) {
    auto total_process_start = std::chrono::high_resolution_clock::now();

    auto start_sensor_fusion = std::chrono::high_resolution_clock::now();
    perform_sensor_fusion(imu_data);
    auto end_sensor_fusion = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time for sensor fusion: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_sensor_fusion - start_sensor_fusion).count()) + " us");

    auto start_update_tracks = std::chrono::high_resolution_clock::now();
    update_object_tracks(detections);
    auto end_update_tracks = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time to update object tracks: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_update_tracks - start_update_tracks).count()) + " us");

    auto start_ballistics = std::chrono::high_resolution_clock::now();
    calculate_ballistics_for_tracks(imu_data);
    auto end_ballistics = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time for ballistic calculations: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_ballistics - start_ballistics).count()) + " us");

    auto start_safety = std::chrono::high_resolution_clock::now();
    perform_safety_and_actuation(imu_data);
    auto end_safety = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time for safety and actuation: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_safety - start_safety).count()) + " us");

    auto total_process_end = std::chrono::high_resolution_clock::now(); // Declaration added here
    APP_LOG_DEBUG("LogicModule: Total time for process function: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(total_process_end - total_process_start).count()) + " us");
}

bool LogicModule::predict_impact_point(const TrackedObject& target, const OrientationData& /*current_imu_data*/, Vec3& out_impact_point) {
    if (!ballistics_solver_) return false;

    float target_distance = target.position.x;
    float target_height = target.position.y;
    float angle_to_target = std::atan2(target_height, target_distance);
    
    auto trajectory = ballistics_solver_->calculate_trajectory(angle_to_target, target_distance + 50.0f);
    if (trajectory.empty()) return false;

    for (const auto& state : trajectory) {
        if (state.position.x >= target_distance) {
            out_impact_point = state.position;
            return true;
        }
    }
    out_impact_point = trajectory.back().position;
    return true;
}

void LogicModule::calculate_ballistics_for_tracks(const OrientationData& imu_data) {
    char log_buffer[256];
    for (auto& track : active_tracks_) {
        Vec3 impact_point;
        if (predict_impact_point(track, imu_data, impact_point)) {
            snprintf(log_buffer, sizeof(log_buffer), "Predicted Impact Point for Track ID %ld: (x:%.2f, y:%.2f, z:%.2f)",
                     track.id, impact_point.x, impact_point.y, impact_point.z);
            APP_LOG_INFO(log_buffer);
        } else {
            snprintf(log_buffer, sizeof(log_buffer), "No Impact Point Predicted for Track ID %ld.", track.id);
            APP_LOG_INFO(log_buffer);
        }
    }
}

void LogicModule::perform_safety_and_actuation(const OrientationData& imu_data) {
    char log_buffer[256];
    Vec3 impact_point; // Moved declaration outside the switch
    // --- 4. Uncertainty Propagation & Safety Checks ---
    float predicted_impact_uncertainty = 0.5f; 
    std::string safety_message;
    
    // Iterate through active tracks and perform safety checks
    for (auto& track : active_tracks_) {
        SafetyStatus safety_status = perform_safety_and_uncertainty_checks(track, predicted_impact_uncertainty, safety_message);
        
        switch (safety_status) {
            case SAFETY_OK:
                if (current_fallback_mode_ != NORMAL_OPERATION) {
                    APP_LOG_INFO("Returning to NORMAL_OPERATION.");
                    current_fallback_mode_ = NORMAL_OPERATION;
                }
                snprintf(log_buffer, sizeof(log_buffer), "Safety check PASSED for Track ID %ld: %s", track.id, safety_message.c_str());
                APP_LOG_INFO(log_buffer);
                
                // Calculate impact point and send as telemetry instead of servo commands
                if (predict_impact_point(track, imu_data, impact_point)) {
                    current_hit_scan_count_++; // Increment hit scan count
                    // Format the impact point as JSON for telemetry
                    // Example JSON: {"track_id": 1, "impact_point": {"x": 10.5, "y": 2.1, "z": 150.7}}
                    std::string telemetry_message = "{\"track_id\": " + std::to_string(track.id) + 
                                                    ", \"impact_point\": {\"x\": " + std::to_string(impact_point.x) +
                                                    ", \"y\": " + std::to_string(impact_point.y) +
                                                    ", \"z\": " + std::to_string(impact_point.z) + "}}";
                    // In a real system, this telemetry_message would be sent over ZeroMQ to tcp://*:6000
                    APP_LOG_INFO("Telemetry (simulated): Sending impact point data: " + telemetry_message);
                }
                break;
            case SAFETY_WARNING_UNCERTAINTY:
            case SAFETY_WARNING_TRACK_UNSTABLE:
                if (current_fallback_mode_ != FALLBACK_A_REDUCED_PERFORMANCE) {
                    APP_LOG_WARNING("Activating FALLBACK_A_REDUCED_PERFORMANCE due to warning: " + safety_message);
                    current_fallback_mode_ = FALLBACK_A_REDUCED_PERFORMANCE;
                }
                snprintf(log_buffer, sizeof(log_buffer), "Safety check WARNING for Track ID %ld: %s", track.id, safety_message.c_str());
                APP_LOG_WARNING(log_buffer);
                // Reduced performance action: e.g., only log, do not issue commands
                break;
            case SAFETY_CRITICAL_UNCERTAINTY:
            case SAFETY_CRITICAL_OTHER:
                if (current_fallback_mode_ < FALLBACK_B_WARNING_STATE) { // Promote to higher fallback if less severe mode
                    APP_LOG_ERROR("Activating FALLBACK_B_WARNING_STATE due to critical issue: " + safety_message);
                    current_fallback_mode_ = FALLBACK_B_WARNING_STATE;
                }
                snprintf(log_buffer, sizeof(log_buffer), "Safety check CRITICAL for Track ID %ld: %s", track.id, safety_message.c_str());
                APP_LOG_ERROR(log_buffer);
                // Critical action: e.g., halt all operations, wait for manual override
                break;
            default:
                snprintf(log_buffer, sizeof(log_buffer), "Safety check: Unhandled SafetyStatus for Track ID %ld: %d", track.id, static_cast<int>(safety_status));
                APP_LOG_ERROR(log_buffer);
                break;
        }
    }
}

void LogicModule::update_object_tracks(const std::vector<DetectionResult>& detections) {
    for (auto& track : active_tracks_) track.associated_this_frame = false;

    for (const auto& new_detection : detections) {
        float best_iou = 0.0f;
        TrackedObject* best_match_track = nullptr;

        for (auto& track : active_tracks_) {
            if (!track.associated_this_frame) {
                float iou = calculate_iou(new_detection, track.last_detection);
                if (iou > best_iou && iou >= track_iou_threshold_) {
                    best_iou = iou;
                    best_match_track = &track;
                }
            }
        }

        if (best_match_track) {
            best_match_track->last_detection = new_detection;
            best_match_track->position.z = config_.get_zero_distance_m(); // Use configured zero distance
            best_match_track->last_update_time = new_detection.timestamp;
            best_match_track->hit_streak++;
            best_match_track->missed_frames = 0;
            best_match_track->associated_this_frame = true;
        } else {
            // Only create a new track if the detection score is above the minimum confidence threshold
            if (new_detection.score >= min_track_confidence_) {
                if (active_tracks_.size() < static_cast<size_t>(max_active_tracks_)) {
                    active_tracks_.emplace_back(++next_track_id_, new_detection, config_.get_zero_distance_m()); // Use configured zero distance
                } else {
                    APP_LOG_WARNING("Max active tracks reached (" + std::to_string(max_active_tracks_) + "). New detection ignored.");
                }
            }
        }
    }

    active_tracks_.erase(std::remove_if(active_tracks_.begin(), active_tracks_.end(), 
        [&](TrackedObject& track) {
            if (!track.associated_this_frame) track.missed_frames++;
            return track.missed_frames > track_missed_frames_threshold_;
        }), active_tracks_.end());
}

// --- Onveranderde of licht aangepaste functies ---
const int MIN_HIT_STREAK = 1;
const float MAX_PREDICTED_UNCERTAINTY = 0.75f;

SafetyStatus LogicModule::perform_safety_and_uncertainty_checks(const TrackedObject& target, float predicted_impact_uncertainty, std::string& safety_status_message) {
    if (predicted_impact_uncertainty > MAX_PREDICTED_UNCERTAINTY) {
        safety_status_message = "CRITICAL: High predicted impact uncertainty.";
        return SAFETY_CRITICAL_UNCERTAINTY;
    }
    if (target.hit_streak < MIN_HIT_STREAK) {
        safety_status_message = "CRITICAL: Track is unstable.";
        return SAFETY_CRITICAL_OTHER;
    }
    safety_status_message = "All safety checks passed.";
    return SAFETY_OK;
}

void LogicModule::issue_servo_commands(float target_x, float target_y, float target_z) {
    char log_buffer[256];
    snprintf(log_buffer, sizeof(log_buffer), "Issuing servo commands for target: (%.2f, %.2f, %.2f)", target_x, target_y, target_z);
    APP_LOG_INFO(log_buffer);
    current_servo_command_count_++; // Increment servo command count
}

void LogicModule::perform_sensor_fusion(const OrientationData& /*imu_data*/) { /* Implement sensor fusion logic */ }