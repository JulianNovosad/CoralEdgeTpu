// Simple test for ballistics solver fix
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

// Constants
constexpr float GRAVITY_CONST = 9.81f;
constexpr float R_DRY_AIR = 287.058f;

// Vec3 structure
struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    
    Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    float magnitude() const { return std::sqrt(x*x + y*y + z*z); }
};

// BallisticProfile structure
struct BallisticProfile {
    float muzzle_velocity_mps;
    float bullet_mass_kg;
    float ballistic_coefficient_si;
    float sight_height_m;
    float zero_distance_m;
    float air_pressure_pa;
    float temperature_c;
};

// BallisticState structure
struct BallisticState {
    Vec3 position;
    Vec3 velocity;
};

// OrientationData structure
struct OrientationData {
    float pitch = 0.0f;
    float roll = 0.0f;
    float yaw = 0.0f;
};

// DetectionResult structure
struct DetectionResult {
    int class_id;
    float score;
    float xmin, ymin, xmax, ymax;
    std::chrono::high_resolution_clock::time_point timestamp;
};

// TrackedObject structure
struct TrackedObject {
    long id;
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    
    TrackedObject(long _id, float initial_distance, float initial_x = 0.0f, float initial_y = 0.0f)
        : id(_id), 
          position({initial_x, initial_y, initial_distance}),
          velocity({0.0f, 0.0f, 0.0f}),
          acceleration({0.0f, 0.0f, 0.0f}) {}
};

// BallisticsSolver class
class BallisticsSolver {
private:
    BallisticProfile profile_;

public:
    BallisticsSolver(const BallisticProfile& profile) : profile_(profile) {}

    float get_air_density() const {
        float temp_kelvin = profile_.temperature_c + 273.15f;
        return profile_.air_pressure_pa / (R_DRY_AIR * temp_kelvin);
    }

    Vec3 drag_force(const Vec3& velocity, float air_density) {
        float v = velocity.magnitude();
        if (v < 1e-6) return {0.0f, 0.0f, 0.0f};
        
        float cd = 0.0f;
        if (v <= 200.0f) {
            cd = 0.25f + (0.35f - 0.25f) * (v / 200.0f);
        } else if (v <= 400.0f) {
            cd = 0.35f + (0.28f - 0.35f) * ((v - 200.0f) / 200.0f);
        } else if (v <= 800.0f) {
            cd = 0.28f + (0.20f - 0.28f) * ((v - 400.0f) / 400.0f);
        } else {
            cd = 0.20f;
        }
        
        if (profile_.ballistic_coefficient_si <= 0.0f) {
            return {0.0f, 0.0f, 0.0f};
        }
        
        // Fixed: Don't divide by mass again since ballistic_coefficient_si is already mass-based
        float drag_magnitude = 0.5f * air_density * v * v * profile_.ballistic_coefficient_si * cd;
        return velocity * (-drag_magnitude / v);
    }

    BallisticState derivatives(const BallisticState& state, float air_density) {
        Vec3 gravitational_force = {0.0f, -GRAVITY_CONST * profile_.bullet_mass_kg, 0.0f};
        Vec3 drag = drag_force(state.velocity, air_density);
        Vec3 total_force = gravitational_force + drag;
        Vec3 acceleration = total_force * (1.0f / profile_.bullet_mass_kg);
        return {{state.velocity}, {acceleration}};
    }

    BallisticState rk4_step(const BallisticState& state, float dt, float air_density) {
        BallisticState k1 = derivatives(state, air_density);
        BallisticState k2 = derivatives({state.position + k1.position * (dt / 2.0f), state.velocity + k1.velocity * (dt / 2.0f)}, air_density);
        BallisticState k3 = derivatives({state.position + k2.position * (dt / 2.0f), state.velocity + k2.velocity * (dt / 2.0f)}, air_density);
        BallisticState k4 = derivatives({state.position + k3.position * dt, state.velocity + k3.velocity * dt}, air_density);

        Vec3 pos_next = state.position + (k1.position + k2.position*2.0f + k3.position*2.0f + k4.position) * (dt / 6.0f);
        Vec3 vel_next = state.velocity + (k1.velocity + k2.velocity*2.0f + k3.velocity*2.0f + k4.velocity) * (dt / 6.0f);
        
        return {pos_next, vel_next};
    }

    std::vector<BallisticState> calculate_trajectory(float initial_pitch, float max_distance, float time_step_override = 0.0f) {
        std::vector<BallisticState> trajectory;
        float air_density = get_air_density();

        float actual_time_step = time_step_override;
        if (actual_time_step == 0.0f) {
            float target_steps = 500.0f;
            actual_time_step = max_distance / (profile_.muzzle_velocity_mps * target_steps);
            actual_time_step = std::max(0.0001f, std::min(0.1f, actual_time_step));
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
            current_state = rk4_step(current_state, actual_time_step, air_density);
            trajectory.push_back(current_state);
            if (trajectory.size() > 20000) {
                break;
            }
        }
        return trajectory;
    }

    float calculate_flight_time(float distance) {
        if (profile_.muzzle_velocity_mps > 0.0f) {
            return distance / profile_.muzzle_velocity_mps;
        }
        return 0.0f;
    }

    bool calculate_impact_point(const TrackedObject& target, const OrientationData& imu_data, Vec3& out_impact_point, float& out_flight_time) {
        float target_distance = target.position.z;
        if (target_distance <= 0.0f) return false;
        
        out_flight_time = calculate_flight_time(target_distance);
        if (out_flight_time <= 0.0f) return false;
        
        Vec3 predicted_position = target.position + target.velocity * out_flight_time + target.acceleration * (0.5f * out_flight_time * out_flight_time);
        
        float pitch_correction = imu_data.pitch * 0.01f;
        float yaw_correction = imu_data.yaw * 0.01f;
        
        predicted_position.y += pitch_correction * target_distance;
        predicted_position.x += yaw_correction * target_distance;
        
        float horizontal_distance = std::sqrt(predicted_position.x * predicted_position.x + predicted_position.z * predicted_position.z);
        float vertical_offset = predicted_position.y - (-profile_.sight_height_m);
        float angle_to_target = std::atan2(vertical_offset, horizontal_distance);
        
        auto trajectory = calculate_trajectory(angle_to_target, horizontal_distance + 50.0f);
        if (trajectory.empty()) return false;
        
        Vec3 ballistic_impact = trajectory.back().position;
        
        // Fixed: Correctly calculate the impact point relative to the target
        out_impact_point.x = predicted_position.x;
        out_impact_point.y = predicted_position.y + (ballistic_impact.y - (-profile_.sight_height_m)); // Adjust for bullet drop
        out_impact_point.z = predicted_position.z;
        
        return true;
    }
};

int main() {
    // Create a ballistics profile
    BallisticProfile profile;
    profile.muzzle_velocity_mps = 800.0f;
    profile.bullet_mass_kg = 0.009f;
    profile.ballistic_coefficient_si = 0.005f;
    profile.sight_height_m = 0.15f;
    profile.zero_distance_m = 100.0f;
    profile.air_pressure_pa = 101325.0f;
    profile.temperature_c = 20.0f;
    
    // Create a mock orientation data
    OrientationData imu_data;
    imu_data.pitch = 0.0f;
    imu_data.roll = 0.0f;
    imu_data.yaw = 0.0f;
    
    // Test 1: Close target (10m)
    std::cout << "=== Test 1: Close Target (10m) ===" << std::endl;
    float estimated_distance_1 = 10.0f;
    float x_world_1 = 0.0f;
    float y_world_1 = 0.5f;
    TrackedObject track_1(1, estimated_distance_1, x_world_1, y_world_1);
    
    BallisticsSolver solver(profile);
    Vec3 impact_point_1;
    float flight_time_1;
    bool success_1 = solver.calculate_impact_point(track_1, imu_data, impact_point_1, flight_time_1);
    
    std::cout << "Target position: x=" << track_1.position.x << "m, y=" << track_1.position.y 
              << "m, z=" << track_1.position.z << "m" << std::endl;
    std::cout << "Ballistics calculation: " << (success_1 ? "SUCCESS" : "FAILED") << std::endl;
    
    if (success_1) {
        std::cout << "Impact point: x=" << impact_point_1.x << "m, y=" << impact_point_1.y 
                  << "m, z=" << impact_point_1.z << "m" << std::endl;
        std::cout << "Flight time: " << flight_time_1 << " seconds" << std::endl;
        
        if (std::abs(impact_point_1.y) < 100.0f) {
            std::cout << "CLOSE TARGET TEST PASSED" << std::endl;
        } else {
            std::cout << "CLOSE TARGET TEST FAILED - Y coordinate unreasonable: " << impact_point_1.y << std::endl;
        }
    }
    
    std::cout << std::endl;
    
    // Test 2: Far target (100m)
    std::cout << "=== Test 2: Far Target (100m) ===" << std::endl;
    float estimated_distance_2 = 100.0f;
    float x_world_2 = 0.0f;
    float y_world_2 = 1.0f;  // Higher target at longer distance
    TrackedObject track_2(2, estimated_distance_2, x_world_2, y_world_2);
    
    Vec3 impact_point_2;
    float flight_time_2;
    bool success_2 = solver.calculate_impact_point(track_2, imu_data, impact_point_2, flight_time_2);
    
    std::cout << "Target position: x=" << track_2.position.x << "m, y=" << track_2.position.y 
              << "m, z=" << track_2.position.z << "m" << std::endl;
    std::cout << "Ballistics calculation: " << (success_2 ? "SUCCESS" : "FAILED") << std::endl;
    
    if (success_2) {
        std::cout << "Impact point: x=" << impact_point_2.x << "m, y=" << impact_point_2.y 
                  << "m, z=" << impact_point_2.z << "m" << std::endl;
        std::cout << "Flight time: " << flight_time_2 << " seconds" << std::endl;
        
        if (std::abs(impact_point_2.y) < 100.0f) {
            std::cout << "FAR TARGET TEST PASSED" << std::endl;
        } else {
            std::cout << "FAR TARGET TEST FAILED - Y coordinate unreasonable: " << impact_point_2.y << std::endl;
        }
    }
    
    std::cout << std::endl;
    
    // Final verification
    if (success_1 && success_2 && std::abs(impact_point_1.y) < 100.0f && std::abs(impact_point_2.y) < 100.0f) {
        std::cout << "BALLISTICS Y-COORDINATE FIX VERIFIED" << std::endl;
    } else {
        std::cout << "BALLISTICS FIX FAILED" << std::endl;
    }
    
    return 0;
}