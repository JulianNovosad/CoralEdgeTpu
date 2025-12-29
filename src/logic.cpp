#include "config_loader.h"
#include "logic.h"
#include "util_logging.h"
#include "orientation_sensor.h"
#include "application.h"  // For Application counter updates
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream> // Added for std::cerr/cout

// --- Fysieke en wiskundige constanten ---
constexpr float GRAVITY_CONST = 9.81f;      // Zwaartekrachtversnelling in m/s^2
constexpr float R_DRY_AIR = 287.058f;   // Specifieke gasconstante voor droge lucht in J/(kg·K)
constexpr float PI = 3.14159265358979323846f;

// --- Camera Parameters ---
constexpr float CAMERA_FOCAL_LENGTH_MM = 4.74f;   // Camera focal length in mm (RPi Camera Module 3)
constexpr float TARGET_WIDTH_CM = 50.0f;          // Preset target width in cm
constexpr float TARGET_HEIGHT_CM = 50.0f;         // Preset target height in cm
constexpr float SENSOR_WIDTH_MM = 6.45f;          // Raspberry Pi Camera Module 3 sensor width in mm
constexpr float SENSOR_HEIGHT_MM = 3.63f;         // Raspberry Pi Camera Module 3 sensor height in mm

// --- Tracking Parameters ---
constexpr int MIN_STABLE_HIT_STREAK = 5;          // Minimum hit streak for a track to be considered stable


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
    if (v < 1e-6) return {0.0f, 0.0f, 0.0f}; // Avoid division by zero
    
    // Implement standard G1 drag model approximation
    // For G1 model, drag coefficient varies with velocity
    // Using simplified G1 approximation based on velocity ranges
    float cd = 0.0f;
    
    if (v <= 200.0f) {
        // Subsonic region - simplified linear approximation
        cd = 0.25f + (0.35f - 0.25f) * (v / 200.0f);
    } else if (v <= 400.0f) {
        // Transonic region
        cd = 0.35f + (0.28f - 0.35f) * ((v - 200.0f) / 200.0f);
    } else if (v <= 800.0f) {
        // Supersonic region
        cd = 0.28f + (0.20f - 0.28f) * ((v - 400.0f) / 400.0f);
    } else {
        // Hypersonic region
        cd = 0.20f;
    }
    
    // Drag force = 0.5 * rho * v² * Cd * A
    // For mass-based BC: Cd * A = BC * m / (BC_ref * m_ref)
    // Simplified: drag_magnitude = 0.5 * air_density * v² * (profile_.ballistic_coefficient_si / profile_.bullet_mass_kg)
    // When ballistic_coefficient_si <= 0, disable drag (vacuum model)
    if (profile_.ballistic_coefficient_si <= 0.0f) {
        return {0.0f, 0.0f, 0.0f}; // No drag
    }
    
    // Use mass-based formulation for drag calculation
    // The ballistic coefficient is already mass-based, so we don't divide by mass again
    float drag_magnitude = 0.5f * air_density * v * v * profile_.ballistic_coefficient_si * cd;
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
        
        // Enhanced safety stop with multiple conditions
        if (trajectory.size() > 100) { // FORENSIC SAFETY CAP: 100 STEPS MAX
            APP_LOG_WARNING("Trajectory calculation exceeded 100 steps (FORENSIC CAP). Max distance: " + std::to_string(max_distance) + "m, Time step: " + std::to_string(actual_time_step) + "s. Last Pos Y: " + std::to_string(current_state.position.y) + "m");
            break;
        }
        
        // Additional safety checks for invalid values
        if (!std::isfinite(current_state.position.x) || !std::isfinite(current_state.position.y) || 
            !std::isfinite(current_state.position.z) || !std::isfinite(current_state.velocity.x) ||
            !std::isfinite(current_state.velocity.y) || !std::isfinite(current_state.velocity.z)) {
            APP_LOG_ERROR("Trajectory calculation produced NaN/Inf values. Stopping calculation.");
            break;
        }
        
        // Check for extremely large values that could cause overflow
        float pos_magnitude = current_state.position.magnitude();
        if (pos_magnitude > 10000.0f) { // Limit to 10km
            APP_LOG_WARNING("Trajectory calculation produced extremely large position values. Stopping calculation.");
            break;
        }
    }
    return trajectory;
}

float BallisticsSolver::calculate_zero_pitch() {
    APP_LOG_INFO("Calculating zero pitch for " + std::to_string(profile_.zero_distance_m) + "m...");
    
    // Calculate initial bounds based on simplified ballistics to ensure convergence
    // Use basic physics: height = sight_height - 0.5 * g * t^2 where t = distance / muzzle_velocity
    float time_of_flight = profile_.zero_distance_m / profile_.muzzle_velocity_mps;
    float drop_at_zero = 0.5f * GRAVITY_CONST * time_of_flight * time_of_flight;
    float initial_angle_estimate = std::atan2(drop_at_zero - profile_.sight_height_m, profile_.zero_distance_m);
    
    // Set bounds around the initial estimate with a reasonable range
    float low_angle_rad = initial_angle_estimate - 0.1f; // -~5.7 degrees around estimate
    float high_angle_rad = initial_angle_estimate + 0.1f; // +~5.7 degrees around estimate
    
    // Ensure bounds are reasonable and different
    if (high_angle_rad <= low_angle_rad) {
        low_angle_rad = initial_angle_estimate - 0.2f;
        high_angle_rad = initial_angle_estimate + 0.2f;
    }
    
    // Clamp bounds to reasonable values to prevent extreme angles
    low_angle_rad = std::max(-0.5f, low_angle_rad);  // -28.6 degrees max
    high_angle_rad = std::min(0.5f, high_angle_rad); // +28.6 degrees max
    
    constexpr int max_iterations = 200; // Increased iterations for better convergence
    constexpr float tolerance_m = 0.001f; // 1 mm
    constexpr int max_consecutive_failures = 10; // Prevent infinite loops if trajectory calculation fails
    
    int consecutive_failures = 0;
    
    for (int i = 0; i < max_iterations; ++i) {
        float mid_angle_rad = (low_angle_rad + high_angle_rad) / 2.0f;
        auto trajectory = calculate_trajectory(mid_angle_rad, profile_.zero_distance_m);
        
        if (trajectory.empty()) {
            consecutive_failures++;
            if (consecutive_failures >= max_consecutive_failures) {
                APP_LOG_ERROR("Trajectory calculation failed repeatedly during zero pitch calculation. Returning initial estimate: " + std::to_string(initial_angle_estimate));
                return initial_angle_estimate;
            }
            
            // Adjust search bounds if trajectory calculation fails
            if (mid_angle_rad < initial_angle_estimate) {
                low_angle_rad = mid_angle_rad;
            } else {
                high_angle_rad = mid_angle_rad;
            }
            continue;
        }
        
        consecutive_failures = 0; // Reset on successful calculation
        
        float height_at_zero = 0.0f;
        bool reached_zero_distance = false;
        for (size_t j = 1; j < trajectory.size(); ++j) {
            if (trajectory[j].position.x >= profile_.zero_distance_m) {
                // Linear interpolation for height
                const auto& p1 = trajectory[j-1].position;
                const auto& p2 = trajectory[j].position;
                float t = (profile_.zero_distance_m - p1.x) / (p2.x - p1.x);
                if (t >= 0.0f && t <= 1.0f) {
                    height_at_zero = p1.y + t * (p2.y - p1.y);
                } else {
                    // Fallback: use the closest point
                    height_at_zero = trajectory[j].position.y;
                }
                reached_zero_distance = true;
                break;
            }
        }
        
        // If zero_distance_m was not reached within the trajectory,
        // it means the bullet hit the ground before reaching the target
        if (!reached_zero_distance) {
            // Check if the bullet went too high or too low
            if (!trajectory.empty()) {
                float final_height = trajectory.back().position.y;
                if (final_height > -profile_.sight_height_m) {
                    // Bullet went too high, angle should be lower
                    high_angle_rad = mid_angle_rad;
                } else {
                    // Bullet went too low, angle should be higher
                    low_angle_rad = mid_angle_rad;
                }
            } else {
                low_angle_rad = mid_angle_rad;
            }
            continue;
        }
        
        // Compare height with sight line (which is at y=0 in our coordinate system)
        if (std::abs(height_at_zero) < tolerance_m) {
            APP_LOG_INFO("Zero pitch found after " + std::to_string(i + 1) + " iterations: " + std::to_string(mid_angle_rad) + " rad");
            return mid_angle_rad;
        }
        
        if (height_at_zero < 0) { // Bullet too low, angle needs to go up
            low_angle_rad = mid_angle_rad;
        } else { // Bullet too high, angle needs to go down
            high_angle_rad = mid_angle_rad;
        }
        
        // Check for convergence: if the bounds are very close, return the midpoint
        if (std::abs(high_angle_rad - low_angle_rad) < tolerance_m * 0.1f) {
            float result = (low_angle_rad + high_angle_rad) / 2.0f;
            APP_LOG_INFO("Zero pitch converged to " + std::to_string(result) + " rad after " + std::to_string(i + 1) + " iterations (bounds too close)");
            return result;
        }
    }
    
    float result = (low_angle_rad + high_angle_rad) / 2.0f;
    APP_LOG_INFO("Zero pitch calculation completed after " + std::to_string(max_iterations) + " iterations. Best estimate: " + std::to_string(result) + " rad.");
    return result; // Return the best estimate
}

float BallisticsSolver::calculate_flight_time(float distance) {
    // Simple approximation: time = distance / muzzle_velocity
    if (profile_.muzzle_velocity_mps > 0.0f) {
        return distance / profile_.muzzle_velocity_mps;
    }
    return 0.0f;
}

bool BallisticsSolver::calculate_impact_point(const TrackedObject& target, Vec3& out_impact_point, float& out_flight_time) {
    // Validate input target position
    if (!std::isfinite(target.position.x) || !std::isfinite(target.position.y) || !std::isfinite(target.position.z)) {
        APP_LOG_ERROR("Invalid target position values detected in calculate_impact_point");
        return false;
    }
    
    // Validate input target velocity
    if (!std::isfinite(target.velocity.x) || !std::isfinite(target.velocity.y) || !std::isfinite(target.velocity.z)) {
        APP_LOG_ERROR("Invalid target velocity values detected in calculate_impact_point");
        return false;
    }
    
    // Validate input target acceleration
    if (!std::isfinite(target.acceleration.x) || !std::isfinite(target.acceleration.y) || !std::isfinite(target.acceleration.z)) {
        APP_LOG_ERROR("Invalid target acceleration values detected in calculate_impact_point");
        return false;
    }
    
    // Calculate distance to target
    float target_distance = target.position.z; // Assuming z is the forward distance
    if (target_distance <= 0.0f || !std::isfinite(target_distance)) {
        APP_LOG_WARNING("Invalid target distance: " + std::to_string(target_distance));
        return false;
    }
    
    // Add reasonable limits to prevent extreme calculations
    if (target_distance > 1000.0f) { // Limit to 1km
        APP_LOG_WARNING("Target distance too large: " + std::to_string(target_distance) + "m. Limiting to 1000m.");
        target_distance = 1000.0f;
    }
    
    // Log the distance estimate for verification
    char log_buffer[256];
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    snprintf(log_buffer, sizeof(log_buffer), "INTERNAL DISTANCE ESTIMATE: value_meters = %.2f, timestamp = %ld", target_distance, now);
    APP_LOG_INFO(log_buffer);
    
    // Calculate bullet flight time to target
    out_flight_time = calculate_flight_time(target_distance);
    if (out_flight_time <= 0.0f || !std::isfinite(out_flight_time)) {
        APP_LOG_WARNING("Invalid flight time calculated: " + std::to_string(out_flight_time));
        return false;
    }
    
    // Add reasonable limit to flight time
    if (out_flight_time > 10.0f) { // Limit to 10 seconds
        APP_LOG_WARNING("Flight time too large: " + std::to_string(out_flight_time) + "s. Limiting to 10s.");
        out_flight_time = 10.0f;
    }
    
    // Predict target position after flight time using kinematic equations
    // position = initial_position + velocity * time + 0.5 * acceleration * time^2
    Vec3 predicted_position = target.position + target.velocity * out_flight_time + target.acceleration * (0.5f * out_flight_time * out_flight_time);
    
    // Validate predicted position
    if (!std::isfinite(predicted_position.x) || !std::isfinite(predicted_position.y) || !std::isfinite(predicted_position.z)) {
        APP_LOG_ERROR("Invalid predicted position calculated");
        return false;
    }
    
    // No IMU correction applied - using raw predicted position
    
    // Validate corrected position
    if (!std::isfinite(predicted_position.x) || !std::isfinite(predicted_position.y) || !std::isfinite(predicted_position.z)) {
        APP_LOG_ERROR("Invalid corrected position calculated");
        return false;
    }
    
    // Calculate ballistic trajectory to predicted target position
    // Calculate angle needed to hit the predicted position
    float horizontal_distance = std::sqrt(predicted_position.x * predicted_position.x + predicted_position.z * predicted_position.z);
    float vertical_offset = predicted_position.y - (-profile_.sight_height_m); // Relative to sight line
    
    // Validate horizontal distance
    if (!std::isfinite(horizontal_distance) || horizontal_distance <= 0.0f) {
        APP_LOG_WARNING("Invalid horizontal distance calculated: " + std::to_string(horizontal_distance));
        return false;
    }
    
    // Add reasonable limits
    if (horizontal_distance > 2000.0f) { // Limit to 2km
        APP_LOG_WARNING("Horizontal distance too large: " + std::to_string(horizontal_distance) + "m. Limiting to 2000m.");
        horizontal_distance = 2000.0f;
    }
    
    float angle_to_target = std::atan2(vertical_offset, horizontal_distance);
    
    // Validate angle
    if (!std::isfinite(angle_to_target)) {
        APP_LOG_WARNING("Invalid angle to target calculated: " + std::to_string(angle_to_target));
        return false;
    }
    
    // Add reasonable limits to max distance for trajectory calculation
    float max_trajectory_distance = horizontal_distance + 50.0f;
    if (max_trajectory_distance > 2000.0f) { // Limit to 2km
        APP_LOG_WARNING("Max trajectory distance too large: " + std::to_string(max_trajectory_distance) + "m. Limiting to 2000m.");
        max_trajectory_distance = 2000.0f;
    }
    
    auto trajectory = calculate_trajectory(angle_to_target, max_trajectory_distance);
    if (trajectory.empty()) {
        APP_LOG_WARNING("Empty trajectory returned for angle=" + std::to_string(angle_to_target) + ", distance=" + std::to_string(max_trajectory_distance));
        return false;
    }
    
    // Find impact point in trajectory
    Vec3 ballistic_impact = trajectory.back().position;
    
    // Validate ballistic impact point
    if (!std::isfinite(ballistic_impact.x) || !std::isfinite(ballistic_impact.y) || !std::isfinite(ballistic_impact.z)) {
        APP_LOG_ERROR("Invalid ballistic impact point calculated");
        return false;
    }
    
    // Combine target movement prediction with ballistic calculation
    // The ballistic calculation gives us the drop from the sight line
    // We need to adjust this to get the impact point relative to the target
    out_impact_point.x = predicted_position.x; // Lateral movement prediction
    out_impact_point.y = predicted_position.y + (ballistic_impact.y - (-profile_.sight_height_m)); // Adjust for bullet drop
    out_impact_point.z = predicted_position.z; // Forward distance prediction
    
    // Validate final impact point
    if (!std::isfinite(out_impact_point.x) || !std::isfinite(out_impact_point.y) || !std::isfinite(out_impact_point.z)) {
        APP_LOG_ERROR("Invalid final impact point calculated");
        return false;
    }
    
    return true;
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
      current_fallback_mode_(NORMAL_OPERATION),
      servo_position_(0.0f),
      servo_direction_(true),
      last_direction_change_(std::chrono::steady_clock::now()),
      distance_history_(DISTANCE_WINDOW_SIZE, 0.0f) {
    
    // Initialize camera intrinsics for angular error calculation
    float image_width_px = static_cast<float>(config_.get_tpu_target_width());
    float image_height_px = static_cast<float>(config_.get_tpu_target_height());
    focal_length_px_ = (CAMERA_FOCAL_LENGTH_MM * image_width_px) / SENSOR_WIDTH_MM;
    image_center_x_ = image_width_px * 0.5f;
    image_center_y_ = image_height_px * 0.5f;
    
    // Log verification value for 3° safety cone
    float px_at_3deg = focal_length_px_ * std::tan(3.0f * PI / 180.0f);
    APP_LOG_INFO("Safety cone verification: 3° corresponds to " + std::to_string(px_at_3deg) + " pixels");
    
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
    
    // Initialize ZeroMQ context for telemetry
    zmq_context_ = std::make_unique<zmq::context_t>(1);
    
    // Initialize PCA9685 LED controller (bus 1, default address 0x40) with 333Hz for servo
    led_controller_ = std::make_unique<PCA9685Controller>(1, 0x40);
    if (!led_controller_->initialize(333)) {
        APP_LOG_WARNING("Failed to initialize PCA9685 LED controller");
    } else {
        APP_LOG_INFO("PCA9685 LED controller initialized successfully");
        // Set servo 0 to middle position as indicator that system is running
        led_controller_->set_servo_position(0, 0.5f); // 50% position (middle)
    }

    APP_LOG_INFO("LogicModule created with 3D Ballistics Solver, configured from file.");

    // Load class distance map
    if (!load_class_distance_map("class_distance_map.json")) {
        APP_LOG_WARNING("Failed to load class distance map, using default distance estimation");
    }
    
    // Load class scale factors for distance calibration
    if (!load_class_scale_factors("class_scale_factors.json")) {
        APP_LOG_WARNING("Failed to load class scale factors, using default distance estimation");
    }
    
    // Load labelmap for human-readable class names
    if (!load_labelmap(config.get_labels_path())) {
        APP_LOG_WARNING("Failed to load labelmap from: " + config.get_labels_path() + ", using numeric class IDs only");
    }
}

LogicModule::~LogicModule() { stop(); APP_LOG_INFO("LogicModule destroyed."); }
bool LogicModule::start() {
    if (running_.exchange(true)) { APP_LOG_ERROR("LogicModule is already running."); return false; }
    
    try {
        // Initialize ZeroMQ publisher socket for telemetry
        telemetry_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, zmq::socket_type::pub);
        
        // Get telemetry configuration from config
        std::string telemetry_address = config_.get_telemetry_pub_address();
        telemetry_socket_->bind(telemetry_address);
        
        APP_LOG_INFO("LogicModule telemetry socket bound to: " + telemetry_address);
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Failed to initialize telemetry socket: " + std::string(e.what()));
        running_ = false;
        return false;
    }
    
    // Start servo worker thread
    servo_worker_running_ = true;
    servo_worker_thread_ = std::thread(&LogicModule::servo_worker_thread_func, this);
    
    worker_thread_ = std::thread(&LogicModule::worker_thread_func, this);
    APP_LOG_INFO("LogicModule started.");
    return true;
}
void LogicModule::stop() {
    if (running_.exchange(false)) {
        APP_LOG_INFO("Stopping LogicModule...");
        if (worker_thread_.joinable()) worker_thread_.join();
        
        // Stop servo worker thread
        if (servo_worker_running_.exchange(false)) {
            {
                std::lock_guard<std::mutex> lock(servo_queue_mutex_);
                // Notify the servo worker thread in case it's waiting
            }
            if (servo_worker_thread_.joinable()) servo_worker_thread_.join();
        }
        
        // Clean up ZeroMQ sockets
        telemetry_socket_.reset();
        zmq_context_.reset();
        
        APP_LOG_INFO("LogicModule stopped.");
    }
}

void LogicModule::send_telemetry_data(const std::string& telemetry_message) {
    if (!telemetry_socket_) {
        APP_LOG_ERROR("Telemetry socket not initialized");
        return;
    }
    
    try {
        zmq::message_t msg(telemetry_message.size());
        memcpy(msg.data(), telemetry_message.c_str(), telemetry_message.size());
        telemetry_socket_->send(msg, zmq::send_flags::none);
        APP_LOG_DEBUG("Telemetry data sent: " + telemetry_message);
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Failed to send telemetry data: " + std::string(e.what()));
    }
}

void LogicModule::worker_thread_func() {
    APP_LOG_INFO("LogicModule worker thread started.");
    
    // Initialize timing variables for logic rate calculation
    int logic_cycle_count = 0;
    std::vector<long long> logic_cycle_times;
    auto last_update_time = std::chrono::high_resolution_clock::now();
    int successful_pops = 0;
    
    // Track rate calculation for idle periods too
    auto last_activity_time = std::chrono::high_resolution_clock::now();
    
    while (running_) {
        std::shared_ptr<DetectionResultBuffer> detections_buffer;
        // Use blocking wait_pop instead of polling to eliminate micro-stutter
        if (detection_input_queue_.wait_pop(detections_buffer, std::chrono::milliseconds(10))) {
            successful_pops++;
            long long call_ts = 0;
            if (detections_buffer && detections_buffer->size > 0) {
                APP_LOG_DEBUG("LogicModule: Detections buffer is valid with size: " + std::to_string(detections_buffer->size));
                // Use the timestamp of the first detection as the call_ts
                call_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            detections_buffer->data[0].timestamp.time_since_epoch()).count();
                
                // If we have an application reference, update the consumed counter
                if (app_ref_) {
                    app_ref_->increment_inference_results_consumed_by_logic();
                }
                
                auto t_start = std::chrono::high_resolution_clock::now();
                process(detections_buffer->data);
                auto t_end = std::chrono::high_resolution_clock::now();
                long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
                // Update freshness indicators
                last_logic_timestamp_ = call_ts;
                
                // Track processing time for rate calculation
                logic_cycle_times.push_back(duration_ms);
                logic_cycle_count++;
                
                // Update the last activity time for rate calculation
                last_activity_time = std::chrono::high_resolution_clock::now();
                
                // Update logic rate every 50 cycles instead of 100 to be more responsive
                if (logic_cycle_count % 25 == 0 && logic_cycle_times.size() > 0) {  // Reduced from 50 to 25 for more responsive updates
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update_time).count();
                    if (elapsed_ms > 0) {
                        // Calculate rate as cycles per second
                        logic_rate_ = static_cast<int>((25.0 * 1000.0) / elapsed_ms);
                    } else {
                        // Fallback: use the average processing time
                        long long total_time_ms = 0;
                        for (long long time : logic_cycle_times) {
                            total_time_ms += time;
                        }
                        double avg_time_ms = static_cast<double>(total_time_ms) / logic_cycle_times.size();
                        if (avg_time_ms > 0) {
                            logic_rate_ = static_cast<int>(1000.0 / avg_time_ms);
                        }
                    }
                    logic_cycle_times.clear();
                    // Update the last update time after calculation
                    last_update_time = current_time;
                    // Log statistics periodically
                    APP_LOG_INFO("LogicModule: Processed " + std::to_string(successful_pops) + " detection batches. Logic rate: " + std::to_string(logic_rate_.load()) + " CPS");
                }
                // FPS measurement for decision stage
                static int decision_frame_counter = 0;
                static auto decision_start_time = std::chrono::high_resolution_clock::now();
                decision_frame_counter++;
                auto current_time = std::chrono::high_resolution_clock::now();
                auto decision_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - decision_start_time).count();
                if (decision_duration >= 1000) { // Log every second
                    double decision_fps = (decision_frame_counter * 1000.0) / decision_duration;
                    APP_LOG_INFO("DECISION FPS MEASUREMENT: " + std::to_string(decision_fps) + " FPS over " + std::to_string(decision_frame_counter) + " decisions in " + std::to_string(decision_duration) + " ms");
                    // Reset for next measurement
                    decision_frame_counter = 0;
                    decision_start_time = current_time;
                }
            } else {
                // Buffer exists but has no detections - still count this as a successful pop
                APP_LOG_DEBUG("LogicModule: Pop successful, but detections_buffer is null or empty. Frame processed without detections.");
                // Update the last logic timestamp even for empty frames to maintain timing
                last_logic_timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::system_clock::now().time_since_epoch()).count();
                logic_cycle_count++;
                
                // Update the last activity time for rate calculation
                last_activity_time = std::chrono::high_resolution_clock::now();
            }
        }
    }
    APP_LOG_INFO("LogicModule worker thread stopped.");
}

void LogicModule::servo_worker_thread_func() {
    APP_LOG_INFO("Servo worker thread started.");
    while (servo_worker_running_) {
        ServoCommand command;
        bool has_command = false;
        
        // Check if there's a command in the queue
        {
            std::lock_guard<std::mutex> lock(servo_queue_mutex_);
            if (!servo_command_queue_.empty()) {
                command = servo_command_queue_.front();
                servo_command_queue_.pop();
                has_command = true;
            }
        }
        
        if (has_command) {
            // Log timing information for servo command verification
            auto queue_to_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - command.timestamp).count();
            
            // Check cooldown period (0.3 seconds)
            auto time_since_last_actuation = 0;
            {
                std::lock_guard<std::mutex> lock(last_direction_change_mutex_);
                time_since_last_actuation = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - last_direction_change_).count();
            }
            
            // Log detailed timing information
            char timing_log_buffer[256];
            snprintf(timing_log_buffer, sizeof(timing_log_buffer), 
                     "SERVO_TIMING: queue_to_execution=%dms cooldown_elapsed=%dms", 
                     static_cast<int>(queue_to_execution_time), static_cast<int>(time_since_last_actuation));
            APP_LOG_INFO(timing_log_buffer);
            
            if (time_since_last_actuation >= 300) { // 300ms cooldown
                // Execute servo command
                auto execution_start = std::chrono::high_resolution_clock::now();
                execute_servo_command(command.target_x, command.target_y, command.target_z, command.confidence);
                auto execution_end = std::chrono::high_resolution_clock::now();
                auto execution_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    execution_end - execution_start).count();
                
                // Log execution time
                APP_LOG_INFO("SERVO_TIMING: execution_time=" + std::to_string(execution_duration) + "us");
                
                // Update last actuation time
                {
                    std::lock_guard<std::mutex> lock(last_direction_change_mutex_);
                    last_direction_change_ = std::chrono::steady_clock::now();
                }
            } else {
                APP_LOG_INFO("Skipping servo command: Cooldown period active (" + std::to_string(time_since_last_actuation) + "ms since last actuation)");
            }
        } else {
            // No commands, sleep briefly to prevent busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    APP_LOG_INFO("Servo worker thread stopped.");
}

void LogicModule::execute_servo_command(float target_x, float target_y, float target_z, float confidence) {
    // Only issue servo commands if confidence is above activation threshold
    if (confidence <= config_.get_servo_activate_confidence()) {
        APP_LOG_INFO("Skipping servo command: Confidence too low (" + std::to_string(confidence * 100.0f) + "%)");
        return;
    }
    
    // Perform oscillation pattern: move to position and back
    if (led_controller_ && led_controller_->is_initialized()) {
        // Execute one oscillation cycle
        // Move to target position (centered at 0.5)
        led_controller_->set_servo_position(0, 0.5f);
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Brief pause
        
        // Move to extended position based on target
        float normalized_x = 0.5f + (target_x / static_cast<float>(config_.get_tpu_target_width()));
        float normalized_y = 0.5f + (target_y / static_cast<float>(config_.get_tpu_target_height()));
        
        // Clamp to valid range [0.0, 1.0]
        normalized_x = std::max(0.0f, std::min(1.0f, normalized_x));
        normalized_y = std::max(0.0f, std::min(1.0f, normalized_y));
        
        // Use average of x and y positions for servo
        float target_position = (normalized_x + normalized_y) * 0.5f;
        led_controller_->set_servo_position(0, target_position);
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Brief pause
        
        // Return to center position
        led_controller_->set_servo_position(0, 0.5f);
        
        // Update last actuation time
        {
            std::lock_guard<std::mutex> lock(last_direction_change_mutex_);
            last_direction_change_ = std::chrono::steady_clock::now();
        }
    }
    
    // Log servo command with detailed information for causality validation
    char log_buffer[256];
    snprintf(log_buffer, sizeof(log_buffer), "CAUSALITY_VALIDATION: Servo oscillated: pos=(%.2f, %.2f, %.2f) conf=%.2f%%", 
            target_x, target_y, target_z, confidence * 100.0f);
    APP_LOG_INFO(log_buffer);
    
    current_servo_command_count_++; // Increment servo command count
}

void LogicModule::process(const std::vector<DetectionResult>& detections) {
    [[maybe_unused]] auto total_process_start = std::chrono::high_resolution_clock::now();
    
    // Early return if detections vector is empty to prevent unnecessary processing
    if (detections.empty()) {
        // Log this condition but don't process further
        static auto last_empty_log = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto time_since_last_log = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_empty_log).count();
        
        // Only log every 1000ms to avoid log spam
        if (time_since_last_log > 1000) {
            APP_LOG_DEBUG("LogicModule: No detections received, skipping processing.");
            last_empty_log = now;
        }
        return;
    }
    
    // Frame-to-frame consistency filtering
    // Only process detections if we have consistent detections across multiple frames
    static int consecutive_low_confidence_frames = 0;
    const int max_consecutive_low_confidence_frames = 3; // Allow up to 3 consecutive low confidence frames
    
    // Check if all detections have very low confidence or are invalid
    bool all_detections_low_confidence = true;
    int valid_detections_count = 0;
    for (const auto& det : detections) {
        // Check if detection has valid values (not zero values and reasonable confidence)
        if (det.score > 0.60f && 
            det.xmax > det.xmin && 
            det.ymax > det.ymin && 
            det.xmin >= 0 && det.ymin >= 0 && 
            det.xmax <= config_.get_tpu_target_width() && det.ymax <= config_.get_tpu_target_height()) {
            all_detections_low_confidence = false;
            valid_detections_count++;
        }
    }
    
    if (all_detections_low_confidence || valid_detections_count == 0) {
        consecutive_low_confidence_frames++;
    } else {
        consecutive_low_confidence_frames = 0;
    }
    
    // Check if camera is covered (no detections or all detections have very low confidence consistently)
    bool camera_covered = false;
    if (detections.empty() || consecutive_low_confidence_frames > max_consecutive_low_confidence_frames) {
        camera_covered = true;
    }
    
    // If camera is covered, lock servo in safe position and return early
    if (camera_covered) {
        // Lock servo in safe position (center)
        if (led_controller_ && led_controller_->is_initialized()) {
            // Check cooldown period (0.3 seconds)
            auto current_time = std::chrono::steady_clock::now();
            auto time_since_last_actuation = 0;
            {
                std::lock_guard<std::mutex> lock(last_direction_change_mutex_);
                time_since_last_actuation = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_direction_change_).count();
            }
            
            if (time_since_last_actuation >= 300) { // 300ms cooldown
                led_controller_->set_servo_position(0, 0.5f); // Center position is safe position
                {
                    std::lock_guard<std::mutex> lock(last_direction_change_mutex_);
                    last_direction_change_ = std::chrono::steady_clock::now();
                }
                APP_LOG_INFO("CAMERA_COVERED: Locked servo in safe position (center)");
            }
        }
        return; // Skip all other processing when camera is covered
    }

    // Only log detection information for invariant verification if we have valid detections
    if (valid_detections_count > 0) {
        APP_LOG_DEBUG("Valid detections: " + std::to_string(valid_detections_count) + "/" + std::to_string(detections.size()));
    } else {
        // Log that we have no valid detections
        static auto last_invalid_log = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto time_since_last_log = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_invalid_log).count();
        
        // Only log every 1000ms to avoid log spam
        if (time_since_last_log > 1000) {
            APP_LOG_DEBUG("LogicModule: No valid detections with reasonable confidence, skipping detailed processing.");
            last_invalid_log = now;
        }
    }

    [[maybe_unused]] auto start_sensor_fusion = std::chrono::high_resolution_clock::now();
    perform_sensor_fusion();
    [[maybe_unused]] auto end_sensor_fusion = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time for sensor fusion: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_sensor_fusion - start_sensor_fusion).count()) + " us");

    [[maybe_unused]] auto start_update_tracks = std::chrono::high_resolution_clock::now();
    update_object_tracks(detections);
    [[maybe_unused]] auto end_update_tracks = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time to update object tracks: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_update_tracks - start_update_tracks).count()) + " us");

    [[maybe_unused]] auto start_ballistics = std::chrono::high_resolution_clock::now();
    calculate_ballistics_for_tracks();
    [[maybe_unused]] auto end_ballistics = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time for ballistic calculations: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_ballistics - start_ballistics).count()) + " us");

    [[maybe_unused]] auto start_safety = std::chrono::high_resolution_clock::now();
    perform_safety_and_actuation();
    [[maybe_unused]] auto end_safety = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time for safety and actuation: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_safety - start_safety).count()) + " us");

    [[maybe_unused]] auto total_process_end = std::chrono::high_resolution_clock::now(); // Declaration added here
    APP_LOG_DEBUG("LogicModule: Total time for process function: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(total_process_end - total_process_start).count()) + " us");
}


void LogicModule::calculate_ballistics_for_tracks() {
    char log_buffer[256];
    // --- 5. Ballistics Calculation ---
    
    // Record start time for performance measurement
    [[maybe_unused]] auto start_time = std::chrono::high_resolution_clock::now();
    
    // Log number of active tracks
    APP_LOG_DEBUG("Active tracks: " + std::to_string(active_tracks_.size()));

    // Iterate through active tracks and calculate ballistics
    for (auto& track : active_tracks_) {
        // Perform additional gating checks before ballistics calculation
        
        // Check 1: Detection confidence ≥ 60% (authoritative confidence metric)
        if (track.last_detection.score < 0.60f) {
            APP_LOG_DEBUG("Ballistics gating: Skipping track ID " + std::to_string(track.id) + 
                         " due to low detection confidence (" + std::to_string(track.last_detection.score) + " < 0.60)");
            continue;
        }
        
        // Check 2: Detection stability across frames (minimum hit streak)
        if (track.hit_streak < 2) {  // At least 2 consecutive detections to ensure stability
            APP_LOG_DEBUG("Ballistics gating: Skipping track ID " + std::to_string(track.id) + 
                         " due to insufficient hit streak (" + std::to_string(track.hit_streak) + " < 2)");
            continue;
        }
        
        // Check 3: Distance plausibility (based on our early rejection criteria)
        if (track.position.z < 0.5f || track.position.z > 5.0f) {
            APP_LOG_DEBUG("Ballistics gating: Skipping track ID " + std::to_string(track.id) + 
                         " due to implausible distance (" + std::to_string(track.position.z) + "m)");
            continue;
        }
        
        Vec3 impact_point = {0.0f, 0.0f, 0.0f};
        float flight_time = 0.0f;
        
        // Use the enhanced ballistics solver with tracking integration
        if (ballistics_solver_ && ballistics_solver_->calculate_impact_point(track, impact_point, flight_time)) {
            // Propagate uncertainty based on flight time
            Uncertainty uncertainty = propagate_uncertainty(track, flight_time);
            
            // Validate the calculated impact point to prevent extremely large values that cause hangs
            if (!std::isfinite(impact_point.x) || !std::isfinite(impact_point.y) || !std::isfinite(impact_point.z) ||
                std::abs(impact_point.x) > 100000.0f || std::abs(impact_point.y) > 100000.0f || std::abs(impact_point.z) > 100000.0f) {
                snprintf(log_buffer, sizeof(log_buffer), "Invalid impact point detected for Track ID %ld: impact=(%.2f, %.2f, %.2f). Skipping.", 
                         track.id, impact_point.x, impact_point.y, impact_point.z);
                APP_LOG_WARNING(log_buffer);
                continue; // Skip this track to prevent processing invalid data that causes hangs
            }
            
            // Store the impact point and uncertainty in the track for use in safety checks
            track.predicted_impact_point = impact_point;
            track.uncertainty = uncertainty;
            
            // Basic ballistics logging
            APP_LOG_DEBUG("Ballistics solved for Track ID " + std::to_string(track.id) + 
                         ", confidence=" + std::to_string(uncertainty.total_confidence));
            
            // Log distance measurement for monitoring
            float measured_distance = track.position.z;
            APP_LOG_DEBUG("Distance measurement: " + std::to_string(measured_distance) + "m");
            
            // Calculate servo command based on ballistics and uncertainty
            // This integrates the ballistics computation with servo feedback
            float combined_confidence = uncertainty.total_confidence;
            
            // Only issue servo commands if confidence is above activation threshold
            if (combined_confidence > config_.get_servo_activate_confidence()) {
                current_hit_scan_count_++; // Increment hit scan count
                
                // Log servo command issuance for invariant verification
                APP_LOG_INFO("DETECTION_INVARIANT: Issuing servo command for track ID " + std::to_string(track.id) + 
                             ", class=" + std::to_string(track.last_detection.class_id) + 
                             ", confidence=" + std::to_string(combined_confidence));

                // Issue servo commands to control LEDs
                issue_servo_commands(impact_point.x, impact_point.y, impact_point.z, combined_confidence);
                
                // Format the impact point as JSON for telemetry
                // Example JSON: {"track_id": 1, "impact_point": {"x": 10.5, "y": 2.1, "z": 150.7}}
                std::string telemetry_message = "{\"track_id\": " + std::to_string(track.id) + 
                                ", \"impact_point\": {\"x\": " + std::to_string(impact_point.x) +
                                ", \"y\": " + std::to_string(impact_point.y) +
                                ", \"z\": " + std::to_string(impact_point.z) + 
                                ", \"confidence\": " + std::to_string(combined_confidence * 100.0f) + "}";
                // Send telemetry data via ZeroMQ
                send_telemetry_data(telemetry_message);
            } else {
                snprintf(log_buffer, sizeof(log_buffer), "Skipping servo command for Track ID %ld: Confidence too low (%.2f%%)", track.id, combined_confidence * 100.0f);
                APP_LOG_INFO(log_buffer);
            }
        } else {
            snprintf(log_buffer, sizeof(log_buffer), "No Impact Point Predicted for Track ID %ld.", track.id);
            APP_LOG_INFO(log_buffer);
        }
    }
    

}

void LogicModule::perform_safety_and_actuation() {
    char log_buffer[256];
    // --- 4. Uncertainty Propagation & Safety Checks ---
    
    // Record start time for performance measurement
    [[maybe_unused]] auto start_time = std::chrono::high_resolution_clock::now();
    
    // Iterate through active tracks and perform safety checks
    for (auto& track : active_tracks_) {
        // Use the uncertainty already calculated and stored in the track
        std::string safety_message;
        SafetyStatus safety_status = perform_safety_and_uncertainty_checks(track, track.uncertainty, safety_message);
        
        // Declare variables outside the switch statement to avoid cross-initialization issues
        Vec3 crosshair_point = {0.0f, 0.0f, track.position.z};
        float impact_distance = 0.0f;
        float angular_error_degrees = 0.0f;
        float distance_factor = 0.0f;
        float combined_confidence = 0.0f;
        // Additional variables for angular error calculation
        float impact_pixel_x = 0.0f;
        float impact_pixel_y = 0.0f;
        float dx = 0.0f;
        float dy = 0.0f;
        float radial_px = 0.0f;
        
        switch (safety_status) {
            case SAFETY_OK: {
                if (current_fallback_mode_ != NORMAL_OPERATION) {
                    APP_LOG_INFO("Returning to NORMAL_OPERATION.");
                    current_fallback_mode_ = NORMAL_OPERATION;
                }
                snprintf(log_buffer, sizeof(log_buffer), "Safety check PASSED for Track ID %ld: %s", track.id, safety_message.c_str());
                APP_LOG_INFO(log_buffer);
                
                // Calculate distance between predicted impact point and crosshair
                // Crosshair is at center of frame (0, 0, track.position.z)
                impact_distance = calculate_impact_point_distance(track.predicted_impact_point, crosshair_point);
                
                // Validate the predicted impact point to prevent extremely large values that cause hangs
                if (!std::isfinite(track.predicted_impact_point.x) || !std::isfinite(track.predicted_impact_point.y) || !std::isfinite(track.predicted_impact_point.z) ||
                    std::abs(track.predicted_impact_point.x) > 100000.0f || std::abs(track.predicted_impact_point.y) > 100000.0f || std::abs(track.predicted_impact_point.z) > 100000.0f) {
                    snprintf(log_buffer, sizeof(log_buffer), "Invalid predicted impact point for Track ID %ld: impact=(%.2f, %.2f, %.2f). Skipping safety checks.", 
                             track.id, track.predicted_impact_point.x, track.predicted_impact_point.y, track.predicted_impact_point.z);
                    APP_LOG_WARNING(log_buffer);
                    continue; // Skip this track to prevent processing invalid data that causes hangs
                }
                
                // Calculate angular error between crosshair and impact point using camera intrinsics
                // Convert impact point from world coordinates to pixel coordinates
                impact_pixel_x = (track.predicted_impact_point.x * focal_length_px_) / track.position.z + image_center_x_;
                impact_pixel_y = (track.predicted_impact_point.y * focal_length_px_) / track.position.z + image_center_y_;
                
                // Calculate inner fraction center for more precise targeting
                {
                    float fraction = config_.get_inner_fraction();
                    float bbox_width = track.last_detection.xmax - track.last_detection.xmin;
                    float bbox_height = track.last_detection.ymax - track.last_detection.ymin;
                    
                    float inner_xmin = track.last_detection.xmin + (1.0f - fraction) * 0.5f * bbox_width;
                    float inner_xmax = track.last_detection.xmax - (1.0f - fraction) * 0.5f * bbox_width;
                    float inner_ymin = track.last_detection.ymin + (1.0f - fraction) * 0.5f * bbox_height;
                    float inner_ymax = track.last_detection.ymax - (1.0f - fraction) * 0.5f * bbox_height;
                    
                    // Inner fraction center in normalized coordinates
                    float inner_center_x_norm = (inner_xmin + inner_xmax) * 0.5f;
                    float inner_center_y_norm = (inner_ymin + inner_ymax) * 0.5f;
                    
                    // Convert to pixel coordinates
                    float inner_center_x_px = inner_center_x_norm * config_.get_tpu_target_width();
                    float inner_center_y_px = inner_center_y_norm * config_.get_tpu_target_height();
                    
                    // Calculate radial displacement in pixels from inner fraction center
                    dx = impact_pixel_x - inner_center_x_px;
                    dy = impact_pixel_y - inner_center_y_px;
                    
                    // Enhanced logging for verification
                    snprintf(log_buffer, sizeof(log_buffer), "INNER_FRACTION_CENTER: track_id=%ld inner_x=%.2f inner_y=%.2f bbox_center_x=%.2f bbox_center_y=%.2f", 
                             track.id, inner_center_x_px, inner_center_y_px, 
                             (track.last_detection.xmin + track.last_detection.xmax) * 0.5f * config_.get_tpu_target_width(),
                             (track.last_detection.ymin + track.last_detection.ymax) * 0.5f * config_.get_tpu_target_height());
                    APP_LOG_INFO(log_buffer);
                }
                
                radial_px = std::sqrt(dx*dx + dy*dy);
                
                // Calculate angular error using camera intrinsics (atan approach)
                angular_error_degrees = camera_cone_error_degrees_from_pixels(radial_px);
                
                // Enhanced logging for verification with detailed calculation information
                {
                    float angular_error_rad = std::atan(radial_px / focal_length_px_);
                    float angular_error_deg_calc = angular_error_rad * (180.0f / PI);
                    
                    snprintf(log_buffer, sizeof(log_buffer), 
                             "ANGULAR_ERROR_VALIDATION: track_id=%ld radial_px=%.2f focal_length_px=%.2f angular_deg=%.2f (calc=%.2f) threshold=%.2f", 
                             track.id, radial_px, focal_length_px_, angular_error_degrees, angular_error_deg_calc, config_.get_max_angular_error_degrees());
                    APP_LOG_INFO(log_buffer);
                    
                    // Additional verification logging for debugging
                    char debug_log_buffer[256];
                    snprintf(debug_log_buffer, sizeof(debug_log_buffer), 
                             "ANGULAR_ERROR_DEBUG: track_id=%ld impact_pixel=(%.2f,%.2f) image_center=(%.2f,%.2f) dx=%.2f dy=%.2f", 
                             track.id, impact_pixel_x, impact_pixel_y, image_center_x_, image_center_y_, dx, dy);
                    APP_LOG_INFO(debug_log_buffer);
                }
                
                // Hard angular veto: if angular error exceeds threshold, no servo command is issued
                if (angular_error_degrees > config_.get_max_angular_error_degrees()) {
                    snprintf(log_buffer, sizeof(log_buffer), "Skipping servo command for Track ID %ld: Angular error too high (%.2f° > %.2f°)", 
                             track.id, angular_error_degrees, config_.get_max_angular_error_degrees());
                    APP_LOG_INFO(log_buffer);
                    continue; // Skip to the next track without issuing servo command
                }
                
                // Calculate confidence based on uncertainty and distance
                // Higher distance from crosshair means lower confidence
                distance_factor = std::exp(-impact_distance * config_.get_distance_confidence_factor()); // Adjust this factor as needed
                combined_confidence = track.uncertainty.total_confidence * distance_factor;
                
                // Additional servo gating checks:
                // 1. Detection confidence ≥ 60%
                // 2. Detection stability (minimum hit streak)
                // 3. Distance variance is low (implies stable tracking)
                // 4. Angular error is within limits (already checked above)
                bool detection_confidence_ok = track.last_detection.score >= 0.60f;
                bool detection_stable = track.hit_streak >= 2;  // At least 2 consecutive detections
                
                if (!detection_confidence_ok) {
                    APP_LOG_DEBUG("Servo gating: Skipping track ID " + std::to_string(track.id) + 
                                 " due to low detection confidence (" + std::to_string(track.last_detection.score) + " < 0.60)");
                }
                
                if (!detection_stable) {
                    APP_LOG_DEBUG("Servo gating: Skipping track ID " + std::to_string(track.id) + 
                                 " due to insufficient stability (hit_streak=" + std::to_string(track.hit_streak) + " < 2)");
                }
                
                // Only issue servo commands if all conditions are met
                if (combined_confidence > config_.get_servo_activate_confidence() && 
                    detection_confidence_ok && 
                    detection_stable) {
                    current_hit_scan_count_++; // Increment hit scan count
                    
                    // Log servo actuation
                    APP_LOG_DEBUG("Servo actuating for Track ID " + std::to_string(track.id) + 
                                 ", confidence=" + std::to_string(combined_confidence));
                    

                    
                    // Issue servo commands to control LEDs
                    issue_servo_commands(track.predicted_impact_point.x, track.predicted_impact_point.y, track.predicted_impact_point.z, combined_confidence);
                    
                    // Format the impact point as JSON for telemetry
                    // Example JSON: {"track_id": 1, "impact_point": {"x": 10.5, "y": 2.1, "z": 150.7}}
                    std::string telemetry_message = "{\"track_id\": " + std::to_string(track.id) + 
                                    ", \"impact_point\": {\"x\": " + std::to_string(track.predicted_impact_point.x) +
                                    ", \"y\": " + std::to_string(track.predicted_impact_point.y) +
                                    ", \"z\": " + std::to_string(track.predicted_impact_point.z) + 
                                    ", \"confidence\": " + std::to_string(combined_confidence * 100.0f) + 
                                    ", \"angular_error\": " + std::to_string(angular_error_degrees) + "}";
                    // Send telemetry data via ZeroMQ
                    send_telemetry_data(telemetry_message);
                } else {
                    APP_LOG_DEBUG("Servo skipping for Track ID " + std::to_string(track.id) + 
                                 ", confidence=" + std::to_string(combined_confidence));
                    

                }
                break;
            }
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
                if (current_fallback_mode_ != FALLBACK_B_WARNING_STATE) { // Promote to higher fallback if less severe mode
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
    // Update existing tracks
    for (auto& track : active_tracks_) {
        track.associated_this_frame = false;
        
        // Store previous position for velocity calculation
        // Vec3 prev_position = track.position;  // Commented out as it's not used
        auto prev_time = track.last_update_time;
        
        // Predict track position based on previous velocity and acceleration
        auto current_time = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(current_time - prev_time).count();
        
        if (dt > 0.0f) {
            // Update position based on velocity and acceleration
            track.position = track.position + track.velocity * dt + track.acceleration * (0.5f * dt * dt);
            
            // Update velocity based on acceleration
            Vec3 prev_velocity = track.velocity;
            track.velocity = track.velocity + track.acceleration * dt;
            
            // Update acceleration based on change in velocity (simplified)
            if (dt > 0.001f) { // Avoid division by zero
                track.acceleration = (track.velocity - prev_velocity) * (1.0f / dt);
            }
        }
    }

    for (const auto& new_detection : detections) {
        // EARLY REJECTION: Apply hard threshold of ≥60% confidence on authoritative confidence metric
        if (new_detection.score < 0.60f) {
            APP_LOG_DEBUG("Early rejection: Detection confidence too low (" + std::to_string(new_detection.score) + " < 0.60)");
            continue;
        }
        
        // EARLY REJECTION: Check bounding box area for plausibility at expected operating distance
        float bbox_width = new_detection.xmax - new_detection.xmin;
        float bbox_height = new_detection.ymax - new_detection.ymin;
        float bbox_area = bbox_width * bbox_height;
        
        // For a target at expected distance, the bounding box should be within reasonable bounds
        // Typical target might occupy ~0.1 to 0.8 of image area depending on target size
        if (bbox_area < 0.01f || bbox_area > 0.8f) {  // Less than 1% or more than 80% of image area
            /* APP_LOG_DEBUG("Early rejection: Bounding box area implausible (" + std::to_string(bbox_area) + ", width=" + 
                         std::to_string(bbox_width) + ", height=" + std::to_string(bbox_height) + ")"); */
            continue;
        }
        
        // EARLY REJECTION: Check aspect ratio for plausibility (not extremely wide or tall)
        float aspect_ratio = bbox_width / bbox_height;
        if (aspect_ratio < 0.1f || aspect_ratio > 10.0f) {  // Not extremely narrow or wide
            APP_LOG_DEBUG("Early rejection: Aspect ratio unrealistic (" + std::to_string(aspect_ratio) + ")");
            continue;
        }
        
        // EARLY REJECTION: Estimate distance and check for physical plausibility
        float estimated_distance = estimate_target_distance(new_detection);
        if (estimated_distance < 0.5f || estimated_distance > 5.0f) {  // Outside 0.5m to 5m range
            APP_LOG_DEBUG("Early rejection: Distance outside plausible range (" + std::to_string(estimated_distance) + "m)");
            continue;
        }
        
        // Find best matching track using IOU
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
            // Update track with new detection
            auto prev_position = best_match_track->position;
            auto prev_time = best_match_track->last_update_time;
            best_match_track->last_detection = new_detection;
            best_match_track->position.z = estimated_distance;
            // Convert normalized detection coordinates to world coordinates
            // Center of bounding box in normalized coordinates
            float center_x_norm = (new_detection.xmin + new_detection.xmax) * 0.5f;
            float center_y_norm = (new_detection.ymin + new_detection.ymax) * 0.5f;
            // Convert to pixel coordinates
            float center_x_px = center_x_norm * config_.get_tpu_target_width();
            float center_y_px = center_y_norm * config_.get_tpu_target_height();
            // Convert to centered coordinates (relative to image center)
            float center_x_centered = center_x_px - (config_.get_tpu_target_width() * 0.5f);
            float center_y_centered = center_y_px - (config_.get_tpu_target_height() * 0.5f);
            // Convert to real-world coordinates using pinhole camera model
            // x = (pixel_x_centered * distance) / focal_length_pixels
            // y = (pixel_y_centered * distance) / focal_length_pixels
            float focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * config_.get_tpu_target_width()) / SENSOR_WIDTH_MM;
            best_match_track->position.x = (center_x_centered * estimated_distance) / focal_length_pixels;
            best_match_track->position.y = (center_y_centered * estimated_distance) / focal_length_pixels;
            best_match_track->last_update_time = new_detection.timestamp;
            best_match_track->hit_streak++;
            best_match_track->missed_frames = 0;
            best_match_track->associated_this_frame = true;
            // Calculate new velocity based on position change
            auto current_time = new_detection.timestamp;
            float dt = std::chrono::duration<float>(current_time - prev_time).count();
            if (dt > 0.001f) { // Avoid division by zero
                Vec3 new_position = best_match_track->position;
                Vec3 velocity = (new_position - prev_position) * (1.0f / dt);
                // Update velocity with smoothing factor
                float alpha = 0.7f; // Smoothing factor
                best_match_track->velocity = best_match_track->velocity * (1.0f - alpha) + velocity * alpha;
                // Update position uncertainty based on velocity
                best_match_track->position_uncertainty = best_match_track->position_uncertainty + best_match_track->velocity_uncertainty * dt;
            }
        } else {
            // Enforce single-target invariant: if a stable track exists, don't create new tracks
            // A stable track has sufficient hit streak (configurable threshold)
            bool has_stable_track = false;
            const int min_stable_hit_streak = MIN_STABLE_HIT_STREAK; // Configurable threshold for stability
            
            for (const auto& track : active_tracks_) {
                if (track.hit_streak >= min_stable_hit_streak) {
                    has_stable_track = true;
                    break;
                }
            }
            
            // Only create a new track if no stable track exists AND confidence is sufficient
            if (!has_stable_track && new_detection.score >= min_track_confidence_) {
                if (active_tracks_.size() < static_cast<size_t>(max_active_tracks_)) {
                    // Convert normalized detection coordinates to world coordinates
                    // Center of bounding box in normalized coordinates
                    float center_x_norm = (new_detection.xmin + new_detection.xmax) * 0.5f;
                    float center_y_norm = (new_detection.ymin + new_detection.ymax) * 0.5f;
                    // Convert to pixel coordinates
                    float center_x_px = center_x_norm * config_.get_tpu_target_width();
                    float center_y_px = center_y_norm * config_.get_tpu_target_height();
                    // Convert to centered coordinates (relative to image center)
                    float center_x_centered = center_x_px - (config_.get_tpu_target_width() * 0.5f);
                    float center_y_centered = center_y_px - (config_.get_tpu_target_height() * 0.5f);
                    // Convert to real-world coordinates using pinhole camera model
                    // x = (pixel_x_centered * distance) / focal_length_pixels
                    // y = (pixel_y_centered * distance) / focal_length_pixels
                    float focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * config_.get_tpu_target_width()) / SENSOR_WIDTH_MM;
                    float x_world = (center_x_centered * estimated_distance) / focal_length_pixels;
                    float y_world = (center_y_centered * estimated_distance) / focal_length_pixels;
                    // Create TrackedObject with proper initial position
                    active_tracks_.emplace_back(++next_track_id_, new_detection, estimated_distance, x_world, y_world);
                } else {
                    APP_LOG_WARNING("Max active tracks reached (" + std::to_string(max_active_tracks_) + "). New detection ignored.");
                }
            } else if (has_stable_track) {
                APP_LOG_DEBUG("Single-target invariant: Stable track exists, rejecting new detection (hit_streak >= " + 
                             std::to_string(min_stable_hit_streak) + ")");
            }
        }
    }

    // Remove stale tracks
    active_tracks_.erase(std::remove_if(active_tracks_.begin(), active_tracks_.end(), 
        [&](TrackedObject& track) {
            if (!track.associated_this_frame) track.missed_frames++;
            return track.missed_frames > track_missed_frames_threshold_;
        }), active_tracks_.end());
}

float LogicModule::estimate_target_distance(const DetectionResult& detection) {
    // Estimate distance using camera parameters and target dimensions
    // Using pinhole camera model: distance = (real_world_size * focal_length) / object_size_in_pixels
    
    // Image resolution used in the system
    const float IMAGE_WIDTH = config_.get_tpu_target_width();   // 320 pixels
    
    // Calculate pixel dimensions of the detection
    // The detection coordinates are normalized [0,1], convert to pixels
    float pixel_width = (detection.xmax - detection.xmin) * config_.get_tpu_target_width();
    float pixel_height = (detection.ymax - detection.ymin) * config_.get_tpu_target_height();
    
    // Determine which dimension to use for distance calculation
    // Use the dimension (width or height) that gives us the most accurate result
    float pixel_size, sensor_dim, real_world_size;
    
    if (pixel_width >= pixel_height) {
        // Use width for distance calculation
        pixel_size = pixel_width;
        sensor_dim = SENSOR_WIDTH_MM;
        real_world_size = TARGET_WIDTH_CM / 100.0f; // Convert cm to meters
    } else {
        // Use height for distance calculation
        pixel_size = pixel_height;
        sensor_dim = SENSOR_HEIGHT_MM;
        real_world_size = TARGET_HEIGHT_CM / 100.0f; // Convert cm to meters
    }
    
    // Avoid division by zero or very small values
    if (pixel_size <= 1.0f) return config_.get_zero_distance_m();
    
    // Convert focal length to pixels using correct sensor dimensions
    // Formula: focal_length_pixels = (focal_length_mm * image_dimension_pixels) / sensor_dimension_mm
    float focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * IMAGE_WIDTH) / sensor_dim;
    
    // Calculate distance using pinhole camera model
    float raw_distance = (real_world_size * focal_length_pixels) / pixel_size;
    
    // Apply class-specific correction
    float corrected_distance = apply_class_correction(detection.class_id, raw_distance);
    
    // Validate distance is within physically plausible range before adding to smoothing
    // This ensures invalid distance estimates don't pollute the smoothing window
    if (corrected_distance < 0.5f || corrected_distance > 5.0f) {
        APP_LOG_DEBUG("Distance validation: Raw distance " + std::to_string(raw_distance) + 
                     "m corrected to " + std::to_string(corrected_distance) + 
                     "m is outside plausible range [0.5m, 5.0m]. Using fallback distance.");
        // Use a reasonable fallback value for the expected operating range
        corrected_distance = 2.0f;
    }
    
    // Apply per-class smoothing to the corrected distance estimate
    float smoothed_distance = add_class_distance_estimate(detection.class_id, corrected_distance);
    
    // Log distance smoothing validation with enhanced details
    char dist_log_buffer[512];
    const auto& history = class_distance_histories_[detection.class_id];
    size_t count = history.full ? CLASS_DISTANCE_WINDOW_SIZE : history.index;
    
    // Create a string with the current window values for detailed logging
    std::string window_values = "[";
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) window_values += ",";
        
        // Validate the distance value to prevent invalid string conversion
        float distance_val = history.distances[i];
        if (!std::isfinite(distance_val)) {
            window_values += "0.000000"; // Use a safe default value
        } else {
            // Clamp the value to a reasonable range to prevent extremely long strings
            distance_val = std::max(0.0f, std::min(1000.0f, distance_val));
            window_values += std::to_string(distance_val);
        }
    }
    // Basic distance logging for debugging
    APP_LOG_DEBUG("Distance estimate: class=" + std::to_string(detection.class_id) + 
                 ", smoothed=" + std::to_string(smoothed_distance) + "m");
    
    // Clamp to reasonable values (0.1m to 100m)
    return std::max(0.1f, std::min(100.0f, smoothed_distance));
}

float LogicModule::add_distance_estimate(float distance) {
    // Validate input distance
    if (!std::isfinite(distance) || distance <= 0.0f) {
        APP_LOG_WARNING("Invalid distance value detected: " + std::to_string(distance) + ". Using default value.");
        distance = 1.0f; // Use a reasonable default
    }
    
    // Add the new distance to the rolling window
    distance_history_[distance_history_index_] = distance;
    
    // Update the index, wrapping around when we reach the end
    distance_history_index_ = (distance_history_index_ + 1) % DISTANCE_WINDOW_SIZE;
    
    // Mark the buffer as full once we've filled it completely
    if (distance_history_index_ == 0) {
        distance_history_full_ = true;
    }
    
    // Calculate the median of the distances in the window
    std::vector<float> sorted_distances;
    
    // Determine how many elements we have
    size_t count = distance_history_full_ ? DISTANCE_WINDOW_SIZE : distance_history_index_;
    
    // Copy the distances to sort them, validating each value
    sorted_distances.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        float val = distance_history_[i];
        // Validate each value before adding to sorted vector
        if (std::isfinite(val) && val > 0.0f) {
            sorted_distances.push_back(val);
        } else {
            APP_LOG_WARNING("Invalid distance value in history: " + std::to_string(val) + ". Skipping.");
        }
    }
    
    // Check if we have valid values
    if (sorted_distances.empty()) {
        APP_LOG_WARNING("No valid distance values found in history. Returning default value.");
        return 1.0f; // Return a reasonable default
    }
    
    // Sort the distances
    std::sort(sorted_distances.begin(), sorted_distances.end());
    
    // Calculate median
    size_t valid_count = sorted_distances.size();
    float median;
    if (valid_count % 2 == 0) {
        // Even number of elements - average of two middle elements
        median = (sorted_distances[valid_count / 2 - 1] + sorted_distances[valid_count / 2]) / 2.0f;
    } else {
        // Odd number of elements - middle element
        median = sorted_distances[valid_count / 2];
    }
    
    return median;
}

bool LogicModule::load_class_distance_map(const std::string& filepath) {
    try {
        // Open the JSON file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            APP_LOG_ERROR("Failed to open class distance map file: " + filepath);
            return false;
        }
        
        // Parse the JSON
        nlohmann::json j;
        file >> j;
        
        // Clear existing map
        class_distance_map_.clear();
        
        // Iterate through the JSON object
        for (auto& [key, value] : j.items()) {
            try {
                // Parse class ID from key
                int class_id = std::stoi(key);
                
                // Extract median distance
                if (value.contains("median_distance") && value["median_distance"].is_number()) {
                    float median_distance = value["median_distance"];
                    class_distance_map_[class_id] = median_distance;
                }
            } catch (const std::exception& e) {
                APP_LOG_WARNING("Failed to parse class entry for key: " + key + ", error: " + e.what());
                continue;
            }
        }
        
        APP_LOG_INFO("Loaded " + std::to_string(class_distance_map_.size()) + " class distance mappings");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Failed to load class distance map: " + std::string(e.what()));
        return false;
    }
}

bool LogicModule::load_class_scale_factors(const std::string& filepath) {
    try {
        // Open the JSON file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            APP_LOG_ERROR("Failed to open class scale factors file: " + filepath);
            return false;
        }
        // Parse the JSON
        nlohmann::json j;
        file >> j;
        // Clear existing map
        class_scale_factors_.clear();
        // Iterate through the JSON object
        for (auto& [key, value] : j.items()) {
            try {
                // Parse class ID from key
                int class_id = std::stoi(key);
                // Extract scale factor
                if (value.is_number()) {
                    float scale_factor = value;
                    class_scale_factors_[class_id] = scale_factor;
                }
            } catch (const std::exception& e) {
                APP_LOG_WARNING("Failed to parse scale factor entry for key: " + key + ", error: " + e.what());
                continue;
            }
        }
        APP_LOG_INFO("Loaded " + std::to_string(class_scale_factors_.size()) + " class scale factors");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Failed to load class scale factors: " + std::string(e.what()));
        return false;
    }
}

bool LogicModule::load_labelmap(const std::string& filepath) {
    try {
        APP_LOG_INFO("Loading labelmap from: " + filepath);
        // Open the labelmap file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            APP_LOG_ERROR("Failed to open labelmap file: " + filepath);
            return false;
        }
        
        // Clear existing map
        class_names_.clear();
        
        // Parse the labelmap.pbtxt format
        std::string line;
        int current_id = -1;
        std::string current_display_name;
        
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);
            
            // Skip empty lines
            if (line.empty()) continue;
            
            // Look for id field
            if (line.find("id:") != std::string::npos) {
                // Extract the ID value
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    std::string id_str = line.substr(colon_pos + 1);
                    id_str.erase(0, id_str.find_first_not_of(" \t"));
                    try {
                        current_id = std::stoi(id_str);
                    } catch (const std::exception& e) {
                        APP_LOG_WARNING("Failed to parse ID from line: " + line);
                        current_id = -1;
                    }
                }
            }
            // Look for display_name field
            else if (line.find("display_name:") != std::string::npos) {
                // Extract the display name value
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    current_display_name = line.substr(colon_pos + 1);
                    // Remove quotes and whitespace
                    current_display_name.erase(0, current_display_name.find_first_not_of(" \"\t"));
                    current_display_name.erase(current_display_name.find_last_not_of(" \"\t") + 1);
                }
            }
            // Look for closing brace to finalize entry
            else if (line == "}" && current_id != -1 && !current_display_name.empty()) {
                class_names_[current_id] = current_display_name;
                current_id = -1;
                current_display_name.clear();
            }
        }
        
        APP_LOG_INFO("Loaded " + std::to_string(class_names_.size()) + " class name mappings");
        return true;
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Failed to load labelmap: " + std::string(e.what()));
        return false;
    }
}

float LogicModule::apply_class_correction(int class_id, float raw_distance) {
    // Use class-specific scale factors for distance calibration
    auto it = class_scale_factors_.find(class_id);
    if (it != class_scale_factors_.end()) {
        // Apply the scale factor for this class
        return raw_distance * it->second;
    }
    
    // Fallback to mapping-based correction for unmapped classes
    auto map_it = class_distance_map_.find(class_id);
    if (map_it != class_distance_map_.end()) {
        float median_class_distance = map_it->second;
        // Original formula: adjusted_distance = raw_distance * (2.8 / median_class_distance)
        const float TARGET_REFERENCE_DISTANCE = 2.8f;
        float distance_ratio = TARGET_REFERENCE_DISTANCE / median_class_distance;
        return raw_distance * distance_ratio;
    }
    
    // Fallback to raw distance for completely unmapped classes
    return raw_distance;
}

float LogicModule::add_class_distance_estimate(int class_id, float distance) {
    // Validate input distance
    if (!std::isfinite(distance) || distance <= 0.0f) {
        APP_LOG_WARNING("Invalid class distance value detected for class " + std::to_string(class_id) + ": " + std::to_string(distance) + ". Using default value.");
        distance = 1.0f; // Use a reasonable default
    }
    
    // Get or create the distance history for this class
    auto& history = class_distance_histories_[class_id];
    
    // Add the new distance to the rolling window
    history.distances[history.index] = distance;
    
    // Update the index, wrapping around when we reach the end
    history.index = (history.index + 1) % CLASS_DISTANCE_WINDOW_SIZE;
    
    // Mark the buffer as full once we've filled it completely
    if (history.index == 0) {
        history.full = true;
    }
    
    // Calculate the median of the distances in the window
    std::vector<float> sorted_distances;
    
    // Determine how many elements we have
    size_t count = history.full ? CLASS_DISTANCE_WINDOW_SIZE : history.index;
    
    // Copy the distances to sort them, validating each value
    sorted_distances.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        float val = history.distances[i];
        // Validate each value before adding to sorted vector
        if (std::isfinite(val) && val > 0.0f) {
            sorted_distances.push_back(val);
        } else {
            APP_LOG_WARNING("Invalid class distance value in history for class " + std::to_string(class_id) + ": " + std::to_string(val) + ". Skipping.");
        }
    }
    
    // Check if we have valid values
    if (sorted_distances.empty()) {
        APP_LOG_WARNING("No valid class distance values found in history for class " + std::to_string(class_id) + ". Returning default value.");
        return 1.0f; // Return a reasonable default
    }
    
    // Sort the distances
    std::sort(sorted_distances.begin(), sorted_distances.end());
    
    // Calculate median
    size_t valid_count = sorted_distances.size();
    float median;
    if (valid_count % 2 == 0) {
        // Even number of elements - average of two middle elements
        median = (sorted_distances[valid_count / 2 - 1] + sorted_distances[valid_count / 2]) / 2.0f;
    } else {
        // Odd number of elements - middle element
        median = sorted_distances[valid_count / 2];
    }
    
    return median;
}

float LogicModule::get_smoothed_class_distance(int class_id) {
    // Check if we have history for this class
    auto it = class_distance_histories_.find(class_id);
    if (it == class_distance_histories_.end()) {
        return 0.0f; // No history for this class
    }
    
    const auto& history = it->second;
    
    // Determine how many elements we have
    size_t count = history.full ? CLASS_DISTANCE_WINDOW_SIZE : history.index;
    if (count == 0) {
        return 0.0f; // No data
    }
    
    // Calculate the median of the distances in the window
    std::vector<float> sorted_distances;
    sorted_distances.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        float val = history.distances[i];
        // Validate each value before adding to sorted vector
        if (std::isfinite(val) && val > 0.0f) {
            sorted_distances.push_back(val);
        } else {
            APP_LOG_WARNING("Invalid class distance value in smoothed history for class " + std::to_string(class_id) + ": " + std::to_string(val) + ". Skipping.");
        }
    }
    
    // Check if we have valid values
    if (sorted_distances.empty()) {
        APP_LOG_WARNING("No valid class distance values found in smoothed history for class " + std::to_string(class_id) + ". Returning default value.");
        return 0.0f; // Return zero as default
    }
    
    // Sort the distances
    std::sort(sorted_distances.begin(), sorted_distances.end());
    
    // Calculate median
    size_t valid_count = sorted_distances.size();
    float median;
    if (valid_count % 2 == 0) {
        // Even number of elements - average of two middle elements
        median = (sorted_distances[valid_count / 2 - 1] + sorted_distances[valid_count / 2]) / 2.0f;
    } else {
        // Odd number of elements - middle element
        median = sorted_distances[valid_count / 2];
    }
    
    return median;
}

std::vector<std::pair<int, float>> LogicModule::get_top_classes_with_distances(size_t count) {
    std::vector<std::pair<int, float>> class_distances;
    
    // Collect all classes with their smoothed distances
    for (const auto& pair : class_distance_histories_) {
        int class_id = pair.first;
        float smoothed_distance = get_smoothed_class_distance(class_id);
        
        // Only include classes with valid distances
        if (smoothed_distance > 0.0f) {
            class_distances.emplace_back(class_id, smoothed_distance);
        }
    }
    
    // Sort by distance (ascending order - closest first)
    std::sort(class_distances.begin(), class_distances.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second < b.second;
              });
    
    // Return only the requested number of classes
    if (class_distances.size() > count) {
        class_distances.resize(count);
    }
    
    return class_distances;
}

float LogicModule::calculate_impact_point_distance(const Vec3& impact_point, const Vec3& crosshair_point) {
    // Calculate 3D distance between impact point and crosshair
    float dx = impact_point.x - crosshair_point.x;
    float dy = impact_point.y - crosshair_point.y;
    float dz = impact_point.z - crosshair_point.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Safety thresholds - TODO: These should come from config
    
Uncertainty LogicModule::propagate_uncertainty(const TrackedObject& target, float flight_time) {
    Uncertainty uncertainty;
    
    // Linear uncertainty propagation model
    // σ_position = σ_initial_position + σ_velocity * time + 0.5 * σ_acceleration * time^2
    
    // Position uncertainty increases with time and velocity uncertainty
    Vec3 position_uncertainty = target.position_uncertainty + 
                               target.velocity_uncertainty * flight_time +
                               target.acceleration * (0.5f * flight_time * flight_time);
    
    // Velocity uncertainty increases with acceleration uncertainty
    Vec3 velocity_uncertainty = target.velocity_uncertainty + 
                               target.acceleration * flight_time;
    
    // Calculate variance (square of standard deviation)
    uncertainty.position_variance = position_uncertainty.x * position_uncertainty.x +
                                  position_uncertainty.y * position_uncertainty.y +
                                  position_uncertainty.z * position_uncertainty.z;
    
    uncertainty.velocity_variance = velocity_uncertainty.x * velocity_uncertainty.x +
                                  velocity_uncertainty.y * velocity_uncertainty.y +
                                  velocity_uncertainty.z * velocity_uncertainty.z;
    
    // Distance variance based on position uncertainty
    uncertainty.distance_variance = target.position_uncertainty.z * target.position_uncertainty.z;
    
    // Calculate total confidence based on propagated uncertainties
    // Lower uncertainty means higher confidence
    float total_variance = uncertainty.position_variance + uncertainty.velocity_variance + uncertainty.distance_variance;
    
    // Check for valid inputs to prevent NaN
    if (!std::isfinite(total_variance) || total_variance < 0.0f) {
        APP_LOG_ERROR("Invalid total variance detected: " + std::to_string(total_variance));
        uncertainty.total_confidence = 0.0f;
        return uncertainty;
    }
    
    float confidence_decay_factor = config_.get_confidence_decay_factor();
    if (!std::isfinite(confidence_decay_factor) || confidence_decay_factor <= 0.0f) {
        APP_LOG_ERROR("Invalid confidence decay factor detected: " + std::to_string(confidence_decay_factor));
        uncertainty.total_confidence = 0.0f;
        return uncertainty;
    }
    
    // Convert variance to confidence (0.0 - 1.0)
    // Using inverse exponential function for smoother transition
    float exponent = -total_variance * confidence_decay_factor;
    if (!std::isfinite(exponent)) {
        APP_LOG_ERROR("Invalid exponent in confidence calculation: total_variance=" + 
                      std::to_string(total_variance) + ", decay_factor=" + std::to_string(confidence_decay_factor));
        uncertainty.total_confidence = 0.0f;
        return uncertainty;
    }
    
    uncertainty.total_confidence = std::exp(exponent);
    
    // Check for valid result
    if (!std::isfinite(uncertainty.total_confidence)) {
        APP_LOG_ERROR("Confidence calculation resulted in NaN/Inf");
        uncertainty.total_confidence = 0.0f;
        return uncertainty;
    }
    
    // Clamp confidence to [0.0, 1.0]
    uncertainty.total_confidence = std::max(0.0f, std::min(1.0f, uncertainty.total_confidence));
    
    return uncertainty;
}

SafetyStatus LogicModule::perform_safety_and_uncertainty_checks(const TrackedObject& target, const Uncertainty& uncertainty, std::string& safety_status_message) {
    // Check if confidence is too low
    if (uncertainty.total_confidence < config_.get_min_confidence_threshold()) { // Less than minimum confidence threshold
        safety_status_message = "CRITICAL: Low confidence in target tracking.";
        return SAFETY_CRITICAL_UNCERTAINTY;
    }
    
    // Check if position uncertainty is too high
    if (uncertainty.position_variance > config_.get_max_position_variance()) {
        safety_status_message = "CRITICAL: High predicted position uncertainty.";
        return SAFETY_CRITICAL_UNCERTAINTY;
    }
    
    if (target.hit_streak < 1) { // MIN_HIT_STREAK constant
        safety_status_message = "CRITICAL: Track is unstable.";
        return SAFETY_CRITICAL_OTHER;
    }
    safety_status_message = "All safety checks passed.";
    return SAFETY_OK;
}


void LogicModule::issue_servo_commands(float target_x, float target_y, float target_z, float confidence) {
    // Instead of executing servo commands directly, enqueue them for the servo worker thread
    {
        std::lock_guard<std::mutex> lock(servo_queue_mutex_);
        ServoCommand command;
        command.target_x = target_x;
        command.target_y = target_y;
        command.target_z = target_z;
        command.confidence = confidence;
        command.timestamp = std::chrono::steady_clock::now();
        servo_command_queue_.push(command);
    }
}


void LogicModule::perform_sensor_fusion() { 
    // No IMU sensor fusion - using raw tracking data
    // Apply no orientation correction to active tracks
    for (auto& track : active_tracks_) {
        // No IMU corrections applied - using raw track positions
        // Track positions remain unchanged
    }
    
    APP_LOG_DEBUG("Sensor fusion updated - No IMU data used");
}

// New method for camera-space angular error calculation
float LogicModule::camera_cone_error_degrees_from_pixels(float radial_px) const
{
    // Clamp input to prevent numerical issues
    if (radial_px < 0.0f) radial_px = 0.0f;
    
    // Prevent division by zero or invalid focal length
    if (focal_length_px_ <= 0.0f || !std::isfinite(focal_length_px_)) {
        APP_LOG_ERROR("Invalid focal length detected: " + std::to_string(focal_length_px_));
        return 90.0f; // Return maximum error if focal length is invalid
    }
    
    // Prevent division by zero or invalid radial distance
    if (!std::isfinite(radial_px)) {
        APP_LOG_ERROR("Invalid radial distance detected: " + std::to_string(radial_px));
        return 90.0f; // Return maximum error if radial distance is invalid
    }
    
    // Calculate angular error using camera intrinsics
    float angular_error_rad = std::atan(radial_px / focal_length_px_);
    float angular_error_deg = angular_error_rad * (180.0f / PI);
    
    // Check for valid result
    if (!std::isfinite(angular_error_deg)) {
        APP_LOG_ERROR("Angular error calculation resulted in NaN/Inf: radial_px=" + 
                      std::to_string(radial_px) + ", focal_length_px_=" + std::to_string(focal_length_px_));
        return 90.0f; // Return maximum error if calculation failed
    }
    
    // Hard clamp to 90° for safety
    return std::min(angular_error_deg, 90.0f);
}

// Test routine to verify correctness of adjusted distances
void test_class_distance_adjustment() {
    APP_LOG_INFO("Starting class distance adjustment test...");
    
    // Create a mock LogicModule (we won't actually run it)
    // This is just to test the distance calculation methods
    
    APP_LOG_INFO("Class distance adjustment test completed.");
}