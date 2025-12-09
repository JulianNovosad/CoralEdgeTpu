#include "config_loader.h"
#include "logic.h"
#include "util_logging.h"
#include "orientation_sensor.h"
#include <algorithm>
#include <cmath>

// --- Fysieke en wiskundige constanten ---
constexpr float GRAVITY_CONST = 9.81f;      // Zwaartekrachtversnelling in m/s^2
constexpr float R_DRY_AIR = 287.058f;   // Specifieke gasconstante voor droge lucht in J/(kg·K)
constexpr float PI = 3.14159265358979323846f;

// Initializeer de statische leden
long LogicModule::next_track_id_ = 0;

// --- Implementatie van BallisticsSolver ---

BallisticsSolver::BallisticsSolver(const BallisticProfile& profile) : profile_(profile) {
    zero_pitch_rad_ = calculate_zero_pitch();
    LOG_INFO("BallisticsSolver created and zero pitch calculated.");
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

std::vector<BallisticState> BallisticsSolver::calculate_trajectory(float initial_pitch, float max_distance, float time_step) {
    std::vector<BallisticState> trajectory;
    float air_density = get_air_density();

    BallisticState current_state;
    current_state.position = {0, -profile_.sight_height_m, 0};
    current_state.velocity = {
        profile_.muzzle_velocity_mps * std::cos(initial_pitch),
        profile_.muzzle_velocity_mps * std::sin(initial_pitch),
        0.0f
    };
    trajectory.push_back(current_state);

    while (current_state.position.x < max_distance) {
        current_state = rk4_step(current_state, time_step, air_density);
        trajectory.push_back(current_state);
        if (trajectory.size() > 5000) { // Veiligheidsstop
            LOG_WARNING("Trajectory calculation exceeded 5000 steps.");
            break;
        }
    }
    return trajectory;
}

float BallisticsSolver::calculate_zero_pitch() {
    LOG_INFO("Calculating zero pitch for " + std::to_string(profile_.zero_distance_m) + "m...");

    float low_angle_rad = -0.05f; // -~3 degrees, a reasonable lower bound
    float high_angle_rad = 0.05f; // +~3 degrees, a reasonable upper bound
    
    constexpr int max_iterations = 30;
    constexpr float tolerance_m = 0.001f; // 1 mm

    for (int i = 0; i < max_iterations; ++i) {
        float mid_angle_rad = (low_angle_rad + high_angle_rad) / 2.0f;
        auto trajectory = calculate_trajectory(mid_angle_rad, profile_.zero_distance_m + 1.0f);

        if (trajectory.empty()) {
            LOG_ERROR("Failed to calculate trajectory during zero pitch calculation.");
            return 0.0f; // Fout
        }

        // Interpoleer om de exacte hoogte op de zero-afstand te vinden
        float height_at_zero = 0.0f;
        for (size_t j = 1; j < trajectory.size(); ++j) {
            if (trajectory[j].position.x >= profile_.zero_distance_m) {
                // Lineaire interpolatie voor de hoogte
                const auto& p1 = trajectory[j-1].position;
                const auto& p2 = trajectory[j].position;
                float t = (profile_.zero_distance_m - p1.x) / (p2.x - p1.x);
                height_at_zero = p1.y + t * (p2.y - p1.y);
                break;
            }
        }
        
        // Vergelijk de hoogte met de zichtlijn (die op y=0 ligt in ons coördinatensysteem)
        if (std::abs(height_at_zero) < tolerance_m) {
            LOG_INFO("Zero pitch found after " + std::to_string(i + 1) + " iterations: " + std::to_string(mid_angle_rad) + " rad");
            return mid_angle_rad;
        }

        if (height_at_zero < 0) { // Kogel te laag, dus hoek moet omhoog
            low_angle_rad = mid_angle_rad;
        } else { // Kogel te hoog, dus hoek moet omlaag
            high_angle_rad = mid_angle_rad;
        }
    }

    LOG_WARNING("Zero pitch calculation did not converge within " + std::to_string(max_iterations) + " iterations.");
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

    LOG_INFO("LogicModule created with 3D Ballistics Solver, configured from file.");
    performance_start_time_ = std::chrono::high_resolution_clock::now();
}

LogicModule::~LogicModule() { stop(); LOG_INFO("LogicModule destroyed."); }
bool LogicModule::start() {
    if (running_.exchange(true)) { LOG_ERROR("LogicModule is already running."); return false; }
    worker_thread_ = std::thread(&LogicModule::worker_thread_func, this);
    LOG_INFO("LogicModule started.");
    return true;
}
void LogicModule::stop() {
    if (running_.exchange(false)) {
        LOG_INFO("Stopping LogicModule...");
        if (worker_thread_.joinable()) worker_thread_.join();
        LOG_INFO("LogicModule stopped.");
    }
}

void LogicModule::worker_thread_func() {
    while (running_) {
        std::shared_ptr<DetectionResultBuffer> detections_buffer;
        if (detection_input_queue_.pop(detections_buffer)) {
            if (detections_buffer && detections_buffer->size > 0) {
                OrientationData current_imu_data = orientation_sensor_->get_latest_orientation_data();
                process(detections_buffer->data, current_imu_data);
            }
        }
    }
}

void LogicModule::process(const std::vector<DetectionResult>& detections, const OrientationData& imu_data) {
    auto t_start = std::chrono::high_resolution_clock::now();
    perform_sensor_fusion(imu_data);
    update_object_tracks(detections);
    calculate_ballistics_for_tracks(imu_data);
    perform_safety_and_actuation(imu_data);
    auto t_end = std::chrono::high_resolution_clock::now();
    long long dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::lock_guard<std::mutex> lock(prediction_times_mutex_);
    prediction_times_ms_.push_back(dur_ms);
    total_predictions_++;
}

bool LogicModule::predict_impact_point(const TrackedObject& target, const OrientationData& current_imu_data, Vec3& out_impact_point) {
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
            LOG_INFO(log_buffer);
        } else {
            snprintf(log_buffer, sizeof(log_buffer), "No Impact Point Predicted for Track ID %ld.", track.id);
            LOG_INFO(log_buffer);
        }
    }
}

void LogicModule::perform_safety_and_actuation(const OrientationData& imu_data) {
    char log_buffer[256];
    float predicted_impact_uncertainty = 0.5f; 
    std::string safety_message;
    
    for (auto& track : active_tracks_) {
        SafetyStatus safety_status = perform_safety_and_uncertainty_checks(track, predicted_impact_uncertainty, safety_message);
        
        if (safety_status == SAFETY_OK) {
            Vec3 impact_point;
            if (predict_impact_point(track, imu_data, impact_point)) {
                issue_servo_commands(impact_point.x, impact_point.y, impact_point.z);
            }
        }
        // ... (andere safety logic)
    }
}

void LogicModule::update_object_tracks(const std::vector<DetectionResult>& detections) {
    char log_buffer[256];
    for (auto& track : active_tracks_) track.associated_this_frame = false;

    for (const auto& new_detection : detections) {
        float best_iou = 0.0f;
        TrackedObject* best_match_track = nullptr;

        for (auto& track : active_tracks_) {
            if (!track.associated_this_frame) {
                float iou = calculate_iou(new_detection, track.last_detection);
                if (iou > best_iou && iou >= 0.3f) {
                    best_iou = iou;
                    best_match_track = &track;
                }
            }
        }

        if (best_match_track) {
            best_match_track->last_detection = new_detection;
            best_match_track->position.x = 50.0f; // Placeholder: afstand is nu X
            best_match_track->last_update_time = new_detection.timestamp;
            best_match_track->hit_streak++;
            best_match_track->missed_frames = 0;
            best_match_track->associated_this_frame = true;
        } else {
            active_tracks_.emplace_back(++next_track_id_, new_detection, 50.0f);
        }
    }

    active_tracks_.erase(std::remove_if(active_tracks_.begin(), active_tracks_.end(), 
        [&](TrackedObject& track) {
            if (!track.associated_this_frame) track.missed_frames++;
            return track.missed_frames > 5;
        }), active_tracks_.end());
}

// --- Onveranderde of licht aangepaste functies ---
const int MIN_HIT_STREAK = 3;
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
    LOG_INFO(log_buffer);
}

void LogicModule::get_performance_metrics() { /* ... onveranderd ... */ }
void LogicModule::perform_sensor_fusion(const OrientationData& imu_data) { /* ... onveranderd ... */ }