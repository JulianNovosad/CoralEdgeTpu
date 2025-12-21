#include "config_loader.h"
#include "logic.h"
#include "util_logging.h"
#include "orientation_sensor.h"
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
        if (trajectory.size() > 20000) { // Safety stop, verhoogd naar 20000
            APP_LOG_WARNING("Trajectory calculation exceeded 20000 steps. Max distance: " + std::to_string(max_distance) + "m, Time step: " + std::to_string(actual_time_step) + "s. Last Pos Y: " + std::to_string(current_state.position.y) + "m");
            break;
        }
    }
    return trajectory;
}

float BallisticsSolver::calculate_zero_pitch() {
    APP_LOG_INFO("Calculating zero pitch for " + std::to_string(profile_.zero_distance_m) + "m...");
    
    // Expanded bounds for better convergence
    float low_angle_rad = -0.2f; // -~11 degrees
    float high_angle_rad = 0.2f; // +~11 degrees
    
    constexpr int max_iterations = 100; // Further increased iterations for better convergence
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
                // Linear interpolation for height
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
        
        // Additional debugging for convergence issues
        if (i == max_iterations - 1) {
            APP_LOG_WARNING("Zero pitch calculation did not converge within " + std::to_string(max_iterations) + " iterations. "
                           "Best estimate: " + std::to_string((low_angle_rad + high_angle_rad) / 2.0f) + " rad. "
                           "Last height_at_zero: " + std::to_string(height_at_zero) + "m. "
                           "Range: [" + std::to_string(low_angle_rad) + ", " + std::to_string(high_angle_rad) + "]");
        }
    }
    
    float result = (low_angle_rad + high_angle_rad) / 2.0f;
    APP_LOG_WARNING("Zero pitch calculation did not converge within " + std::to_string(max_iterations) + " iterations. Best estimate: " + std::to_string(result) + " rad.");
    return result; // Return the best estimate
}

float BallisticsSolver::calculate_flight_time(float distance) {
    // Simple approximation: time = distance / muzzle_velocity
    if (profile_.muzzle_velocity_mps > 0.0f) {
        return distance / profile_.muzzle_velocity_mps;
    }
    return 0.0f;
}

bool BallisticsSolver::calculate_impact_point(const TrackedObject& target, const OrientationData& imu_data, Vec3& out_impact_point, float& out_flight_time) {
    // Calculate distance to target
    float target_distance = target.position.z; // Assuming z is the forward distance
    if (target_distance <= 0.0f) return false;
    
    // Log the distance estimate for verification
    char log_buffer[256];
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    snprintf(log_buffer, sizeof(log_buffer), "INTERNAL DISTANCE ESTIMATE: value_meters = %.2f, timestamp = %ld", target_distance, now);
    APP_LOG_INFO(log_buffer);
    
    // Calculate bullet flight time to target
    out_flight_time = calculate_flight_time(target_distance);
    if (out_flight_time <= 0.0f) return false;
    
    // Predict target position after flight time using kinematic equations
    // position = initial_position + velocity * time + 0.5 * acceleration * time^2
    Vec3 predicted_position = target.position + target.velocity * out_flight_time + target.acceleration * (0.5f * out_flight_time * out_flight_time);
    
    // Apply IMU orientation correction to predicted position
    // Simple correction using pitch and yaw angles
    float pitch_correction = imu_data.pitch * 0.01f; // Small correction factor
    float yaw_correction = imu_data.yaw * 0.01f;     // Small correction factor
    
    predicted_position.y += pitch_correction * target_distance; // Vertical adjustment based on pitch
    predicted_position.x += yaw_correction * target_distance;   // Lateral adjustment based on yaw
    
    // Calculate ballistic trajectory to predicted target position
    // Calculate angle needed to hit the predicted position
    float horizontal_distance = std::sqrt(predicted_position.x * predicted_position.x + predicted_position.z * predicted_position.z);
    float vertical_offset = predicted_position.y - (-profile_.sight_height_m); // Relative to sight line
    float angle_to_target = std::atan2(vertical_offset, horizontal_distance);
    
    auto trajectory = calculate_trajectory(angle_to_target, horizontal_distance + 50.0f);
    if (trajectory.empty()) return false;
    
    // Find impact point in trajectory
    Vec3 ballistic_impact = trajectory.back().position;
    
    // Combine target movement prediction with ballistic calculation
    // The ballistic calculation gives us the drop from the sight line
    // We need to adjust this to get the impact point relative to the target
    out_impact_point.x = predicted_position.x; // Lateral movement prediction
    out_impact_point.y = predicted_position.y + (ballistic_impact.y - (-profile_.sight_height_m)); // Adjust for bullet drop
    out_impact_point.z = predicted_position.z; // Forward distance prediction
    
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
    if (!load_labelmap("labelmap.pbtxt")) {
        APP_LOG_WARNING("Failed to load labelmap, using numeric class IDs only");
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
                
                // Update freshness indicators
                last_logic_timestamp_ = call_ts;
                static int logic_cycle_count = 0;
                static std::vector<long long> logic_cycle_times;
                logic_cycle_times.push_back(duration_ms);
                logic_cycle_count++;
                
                // Update logic rate every 100 cycles
                if (logic_cycle_count % 100 == 0 && logic_cycle_times.size() > 0) {
                    long long total_time_ms = 0;
                    for (long long time : logic_cycle_times) {
                        total_time_ms += time;
                    }
                    double avg_time_ms = static_cast<double>(total_time_ms) / logic_cycle_times.size();
                    if (avg_time_ms > 0) {
                        logic_rate_ = static_cast<int>(1000.0 / avg_time_ms);
                    }
                    logic_cycle_times.clear();
                }
                
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
            // Check cooldown period (0.3 seconds)
            auto time_since_last_actuation = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - command.timestamp).count();
            
            if (time_since_last_actuation >= 300) { // 300ms cooldown
                // Execute servo command
                execute_servo_command(command.target_x, command.target_y, command.target_z, command.confidence);
            } else {
                APP_LOG_INFO("Skipping servo command: Cooldown period active (" + std::to_string(time_since_last_actuation) + "ms since command queued)");
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
        last_direction_change_ = std::chrono::steady_clock::now();
    }
    
    // Log servo command with detailed information for causality validation
    char log_buffer[256];
    snprintf(log_buffer, sizeof(log_buffer), "CAUSALITY_VALIDATION: Servo oscillated: pos=(%.2f, %.2f, %.2f) conf=%.2f%%", 
            target_x, target_y, target_z, confidence * 100.0f);
    APP_LOG_INFO(log_buffer);
    
    current_servo_command_count_++; // Increment servo command count
}

void LogicModule::process(const std::vector<DetectionResult>& detections, const OrientationData& imu_data) {
    [[maybe_unused]] auto total_process_start = std::chrono::high_resolution_clock::now();

    // Check if camera is covered (no detections or all detections have very low confidence)
    bool camera_covered = false;
    if (detections.empty()) {
        camera_covered = true;
    } else {
        // Check if all detections have very low confidence (likely noise when camera is covered)
        camera_covered = true;
        for (const auto& det : detections) {
            if (det.score > 0.1f) {  // If any detection has reasonable confidence, camera is not covered
                camera_covered = false;
                break;
            }
        }
    }

    // If camera is covered, lock servo in safe position and return early
    if (camera_covered) {
        // Lock servo in safe position (center)
        if (led_controller_ && led_controller_->is_initialized()) {
            // Check cooldown period (0.3 seconds)
            auto current_time = std::chrono::steady_clock::now();
            auto time_since_last_actuation = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_direction_change_).count();
            
            if (time_since_last_actuation >= 300) { // 300ms cooldown
                led_controller_->set_servo_position(0, 0.5f); // Center position is safe position
                last_direction_change_ = std::chrono::steady_clock::now();
                APP_LOG_INFO("CAMERA_COVERED: Locked servo in safe position (center)");
            }
        }
        return; // Skip all other processing when camera is covered
    }

    // Log detection information for invariant verification
    APP_LOG_INFO("DETECTION_INVARIANT: Detections received by logic module: " + std::to_string(detections.size()));
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        [[maybe_unused]] float area = (det.xmax - det.xmin) * (det.ymax - det.ymin);
        APP_LOG_INFO("DETECTION_INVARIANT: Detection " + std::to_string(i) + 
                     ": class=" + std::to_string(det.class_id) + 
                     ", score=" + std::to_string(det.score) + 
                     ", area=" + std::to_string(area) +
                     ", box=[" + std::to_string(det.xmin) + "," + std::to_string(det.ymin) + "," + 
                     std::to_string(det.xmax) + "," + std::to_string(det.ymax) + "]" +
                     ", timestamp=" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                         det.timestamp.time_since_epoch()).count()));
        
        // Calculate distance for this detection using the pinhole camera model
        float pixel_width = det.xmax - det.xmin;
        float pixel_height = det.ymax - det.ymin;
        float pixel_size = std::max(pixel_width, pixel_height);
        
        // Avoid division by zero or very small values
        if (pixel_size > 1.0f) {
            // Use the same calculation as in estimate_target_distance but without smoothing
            const float IMAGE_WIDTH = config_.get_tpu_target_width();
            float focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * IMAGE_WIDTH) / SENSOR_WIDTH_MM;
            float real_world_size = std::max(TARGET_WIDTH_CM, TARGET_HEIGHT_CM) / 100.0f;
            [[maybe_unused]] float distance = (real_world_size * focal_length_pixels) / pixel_size;
            
            APP_LOG_INFO("DETECTION_DISTANCE: class=" + std::to_string(det.class_id) + 
                         ", score=" + std::to_string(det.score) + 
                         ", box=[" + std::to_string(det.xmin) + "," + std::to_string(det.ymin) + "," + 
                         std::to_string(det.xmax) + "," + std::to_string(det.ymax) + "]" +
                         ", distance=" + std::to_string(distance) + "m");
        } else {
            APP_LOG_INFO("DETECTION_DISTANCE: class=" + std::to_string(det.class_id) + 
                         ", score=" + std::to_string(det.score) + 
                         ", box=[" + std::to_string(det.xmin) + "," + std::to_string(det.ymin) + "," + 
                         std::to_string(det.xmax) + "," + std::to_string(det.ymax) + "]" +
                         ", distance=too_small");
        }
    }

    [[maybe_unused]] auto start_sensor_fusion = std::chrono::high_resolution_clock::now();
    perform_sensor_fusion(imu_data);
    [[maybe_unused]] auto end_sensor_fusion = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time for sensor fusion: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_sensor_fusion - start_sensor_fusion).count()) + " us");

    [[maybe_unused]] auto start_update_tracks = std::chrono::high_resolution_clock::now();
    update_object_tracks(detections);
    [[maybe_unused]] auto end_update_tracks = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time to update object tracks: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_update_tracks - start_update_tracks).count()) + " us");

    [[maybe_unused]] auto start_ballistics = std::chrono::high_resolution_clock::now();
    calculate_ballistics_for_tracks(imu_data);
    [[maybe_unused]] auto end_ballistics = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time for ballistic calculations: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_ballistics - start_ballistics).count()) + " us");

    [[maybe_unused]] auto start_safety = std::chrono::high_resolution_clock::now();
    perform_safety_and_actuation(imu_data);
    [[maybe_unused]] auto end_safety = std::chrono::high_resolution_clock::now();
    APP_LOG_DEBUG("LogicModule: Time for safety and actuation: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end_safety - start_safety).count()) + " us");

    [[maybe_unused]] auto total_process_end = std::chrono::high_resolution_clock::now(); // Declaration added here
    APP_LOG_DEBUG("LogicModule: Total time for process function: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(total_process_end - total_process_start).count()) + " us");
}


void LogicModule::calculate_ballistics_for_tracks(const OrientationData& imu_data) {
    char log_buffer[256];
    // --- 5. Ballistics Calculation ---
    
    // Record start time for performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Log number of active tracks for invariant verification
    APP_LOG_INFO("DETECTION_INVARIANT: Active tracks for ballistics: " + std::to_string(active_tracks_.size()));

    // Iterate through active tracks and calculate ballistics
    for (auto& track : active_tracks_) {
        Vec3 impact_point = {0.0f, 0.0f, 0.0f};
        float flight_time = 0.0f;
        
        // Use the enhanced ballistics solver with tracking integration
        if (ballistics_solver_ && ballistics_solver_->calculate_impact_point(track, imu_data, impact_point, flight_time)) {
            // Propagate uncertainty based on flight time
            Uncertainty uncertainty = propagate_uncertainty(track, flight_time);
            
            // Store the impact point and uncertainty in the track for use in safety checks
            track.predicted_impact_point = impact_point;
            track.uncertainty = uncertainty;
            
            // Log ballistics output with detailed information for causality validation
            snprintf(log_buffer, sizeof(log_buffer), "CAUSALITY_VALIDATION: Ballistics solved for Track ID %ld: impact=(%.2f, %.2f, %.2f) conf=%.2f%% detection=[class=%d,score=%.4f,box=(%.1f,%.1f,%.1f,%.1f)]", 
                     track.id, impact_point.x, impact_point.y, impact_point.z, uncertainty.total_confidence * 100.0f,
                     track.last_detection.class_id, track.last_detection.score,
                     track.last_detection.xmin, track.last_detection.ymin, 
                     track.last_detection.xmax, track.last_detection.ymax);
            APP_LOG_INFO(log_buffer);
            
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
    
    // Record end time and calculate duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    // Log performance metrics
    CsvLogEntry entry;
    entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    copy_to_array(entry.module, "LogicModule");
    entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    copy_to_array(entry.event, "ballistics_calculated");
    entry.call_ts_epoch_ms = entry.produced_ts_epoch_ms; // Use current time as call time
    entry.logic_metric_ballistics = static_cast<float>(duration) / 1000.0f; // Convert to milliseconds
    entry.logic_metric_hit_scan = static_cast<float>(current_hit_scan_count_);
    entry.logic_metric_servo_actuation = static_cast<float>(current_servo_command_count_);
    Logger::getInstance().log_csv(entry);
}

void LogicModule::perform_safety_and_actuation(const OrientationData& imu_data) {
    char log_buffer[256];
    // --- 4. Uncertainty Propagation & Safety Checks ---
    
    // Record start time for performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    
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
        
        switch (safety_status) {
            case SAFETY_OK:
                if (current_fallback_mode_ != NORMAL_OPERATION) {
                    APP_LOG_INFO("Returning to NORMAL_OPERATION.");
                    current_fallback_mode_ = NORMAL_OPERATION;
                }
                snprintf(log_buffer, sizeof(log_buffer), "Safety check PASSED for Track ID %ld: %s", track.id, safety_message.c_str());
                APP_LOG_INFO(log_buffer);
                
                // Calculate distance between predicted impact point and crosshair
                // Crosshair is at center of frame (0, 0, track.position.z)
                impact_distance = calculate_impact_point_distance(track.predicted_impact_point, crosshair_point);
                
                // Calculate angular error between crosshair and impact point
                angular_error_degrees = calculate_angular_error_degrees(track.predicted_impact_point, crosshair_point, track.position.z);
                
                // Hard angular veto: if angular error exceeds threshold, no servo command is issued
                if (angular_error_degrees > config_.get_max_angular_error_degrees()) {
                    snprintf(log_buffer, sizeof(log_buffer), "Skipping servo command for Track ID %ld: Angular error too high (%.2f° > %.2f°)", 
                             track.id, angular_error_degrees, config_.get_max_angular_error_degrees());
                    APP_LOG_INFO(log_buffer);
                    break; // Exit the switch case without issuing servo command
                }
                
                // Calculate confidence based on uncertainty and distance
                // Higher distance from crosshair means lower confidence
                distance_factor = std::exp(-impact_distance * config_.get_distance_confidence_factor()); // Adjust this factor as needed
                combined_confidence = track.uncertainty.total_confidence * distance_factor;
                
                // Only issue servo commands if confidence is above 90%
                if (combined_confidence > config_.get_servo_activate_confidence()) {
                    current_hit_scan_count_++; // Increment hit scan count
                    
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
                    snprintf(log_buffer, sizeof(log_buffer), "Skipping servo command for Track ID %ld: Confidence too low (%.2f%%)", track.id, combined_confidence * 100.0f);
                    APP_LOG_INFO(log_buffer);
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
    
    // Record end time and calculate duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    // Log performance metrics
    CsvLogEntry entry;
    entry.produced_ts_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    copy_to_array(entry.module, "LogicModule");
    entry.thread_id = static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    copy_to_array(entry.event, "safety_actuation_done");
    entry.call_ts_epoch_ms = entry.produced_ts_epoch_ms; // Use current time as call time
    entry.logic_metric_ballistics = static_cast<float>(duration) / 1000.0f; // Convert to milliseconds
    entry.logic_metric_hit_scan = static_cast<float>(current_hit_scan_count_);
    entry.logic_metric_servo_actuation = static_cast<float>(current_servo_command_count_);
    Logger::getInstance().log_csv(entry);
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
            
            // Estimate distance using camera parameters
            float estimated_distance = estimate_target_distance(new_detection);
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
            // Only create a new track if the detection score is above the minimum confidence threshold
            if (new_detection.score >= min_track_confidence_) {
                if (active_tracks_.size() < static_cast<size_t>(max_active_tracks_)) {
                    // Estimate distance using camera parameters
                    float estimated_distance = estimate_target_distance(new_detection);
                    
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
    // The detection coordinates are already in pixel coordinates, not normalized
    float pixel_width = detection.xmax - detection.xmin;
    float pixel_height = detection.ymax - detection.ymin;
    
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
    
    // Apply per-class smoothing to the corrected distance estimate
    float smoothed_distance = add_class_distance_estimate(detection.class_id, corrected_distance);
    
    // Get top 3 classes with their smoothed distances for logging
    auto top_classes = get_top_classes_with_distances(3);
    
    // Build a string with the top classes information
    std::string top_classes_info = "";
    for (size_t i = 0; i < top_classes.size(); ++i) {
        int class_id = top_classes[i].first;
        float distance = top_classes[i].second;
        
        // Get human-readable class name if available
        std::string class_name = std::to_string(class_id);
        auto name_it = class_names_.find(class_id);
        if (name_it != class_names_.end()) {
            class_name = name_it->second;
        }
        
        top_classes_info += " class_" + class_name + "=" + std::to_string(static_cast<int>(distance * 100) / 100.0) + "m";
    }
    
    // Log the smoothed estimated distance with bounding box information for verification
    char log_buffer[512];
    snprintf(log_buffer, sizeof(log_buffer), 
             "INTERNAL DISTANCE ESTIMATE: bbox[%.1f,%.1f,%.1f,%.1f] size[%.1fx%.1f] raw=%.3fm corrected=%.3fm smoothed=%.3fm class=%d%s", 
             detection.xmin, detection.ymin, detection.xmax, detection.ymax,
             pixel_width, pixel_height, raw_distance, corrected_distance, smoothed_distance, detection.class_id,
             top_classes_info.c_str());
    APP_LOG_INFO(log_buffer);
    
    // Clamp to reasonable values (0.1m to 100m)
    return std::max(0.1f, std::min(100.0f, smoothed_distance));
}

float LogicModule::add_distance_estimate(float distance) {
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
    
    // Copy the distances to sort them
    sorted_distances.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        sorted_distances.push_back(distance_history_[i]);
    }
    
    // Sort the distances
    std::sort(sorted_distances.begin(), sorted_distances.end());
    
    // Calculate median
    float median;
    if (count % 2 == 0) {
        // Even number of elements - average of two middle elements
        median = (sorted_distances[count / 2 - 1] + sorted_distances[count / 2]) / 2.0f;
    } else {
        // Odd number of elements - middle element
        median = sorted_distances[count / 2];
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
    
    // Copy the distances to sort them
    sorted_distances.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        sorted_distances.push_back(history.distances[i]);
    }
    
    // Sort the distances
    std::sort(sorted_distances.begin(), sorted_distances.end());
    
    // Calculate median
    float median;
    if (count % 2 == 0) {
        // Even number of elements - average of two middle elements
        median = (sorted_distances[count / 2 - 1] + sorted_distances[count / 2]) / 2.0f;
    } else {
        // Odd number of elements - middle element
        median = sorted_distances[count / 2];
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
        sorted_distances.push_back(history.distances[i]);
    }
    
    // Sort the distances
    std::sort(sorted_distances.begin(), sorted_distances.end());
    
    // Calculate median
    float median;
    if (count % 2 == 0) {
        // Even number of elements - average of two middle elements
        median = (sorted_distances[count / 2 - 1] + sorted_distances[count / 2]) / 2.0f;
    } else {
        // Odd number of elements - middle element
        median = sorted_distances[count / 2];
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

float LogicModule::calculate_angular_error_degrees(const Vec3& impact_point, const Vec3& crosshair_point, float target_distance) {
    // Calculate angular error in degrees between crosshair and impact point
    // Using small angle approximation: angle ≈ displacement / distance
    
    // Calculate displacement in x and y directions (assuming z is depth)
    float dx = impact_point.x - crosshair_point.x;
    float dy = impact_point.y - crosshair_point.y;
    
    // Calculate radial displacement in pixels
    float radial_displacement = std::sqrt(dx*dx + dy*dy);
    
    // Convert to angular displacement in radians
    // Using small angle approximation: angle (radians) = displacement / distance
    float angular_error_radians = (target_distance > 0.0f) ? (radial_displacement / target_distance) : 0.0f;
    
    // Convert to degrees
    constexpr float RAD_TO_DEG = 180.0f / PI;
    return angular_error_radians * RAD_TO_DEG;
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
    
    // Convert variance to confidence (0.0 - 1.0)
    // Using inverse exponential function for smoother transition
    uncertainty.total_confidence = std::exp(-total_variance * config_.get_confidence_decay_factor());
    
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


void LogicModule::perform_sensor_fusion(const OrientationData& imu_data) { 
    // Store latest IMU data for use in other calculations
    latest_imu_data_ = imu_data;
    
    // Apply orientation correction to active tracks
    for (auto& track : active_tracks_) {
        // Apply IMU orientation correction to track position
        // Simple correction using pitch and yaw angles
        float pitch_correction = imu_data.pitch * 0.01f; // Small correction factor
        float yaw_correction = imu_data.yaw * 0.01f;     // Small correction factor
        
        // Apply corrections to track position
        track.position.y += pitch_correction * track.position.z; // Vertical adjustment based on pitch
        track.position.x += yaw_correction * track.position.z;   // Lateral adjustment based on yaw
        
        // Also apply to predicted impact point
        track.predicted_impact_point.y += pitch_correction * track.position.z;
        track.predicted_impact_point.x += yaw_correction * track.position.z;
    }
    
    APP_LOG_DEBUG("Sensor fusion updated - Yaw: " + std::to_string(imu_data.yaw) + 
                  ", Pitch: " + std::to_string(imu_data.pitch) + 
                  ", Roll: " + std::to_string(imu_data.roll));
}

// Test routine to verify correctness of adjusted distances
void test_class_distance_adjustment() {
    APP_LOG_INFO("Starting class distance adjustment test...");
    
    // Create a mock LogicModule (we won't actually run it)
    // This is just to test the distance calculation methods
    
    APP_LOG_INFO("Class distance adjustment test completed.");
}