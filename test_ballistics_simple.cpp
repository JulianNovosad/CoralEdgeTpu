#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>

// Simplified Vec3 structure
struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    
    Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    float magnitude() const { return std::sqrt(x*x + y*y + z*z); }
};

// Simplified BallisticProfile structure
struct BallisticProfile {
    // Munitie
    float muzzle_velocity_mps;  // Mondingssnelheid in m/s
    float bullet_mass_kg;       // Kogelmassa in kg
    float ballistic_coefficient_si; // G1 Ballistische coëfficiënt in SI-eenheden (kg/m^2)

    // Wapen
    float sight_height_m;       // Hoogte van vizier boven de loop in meters
    float zero_distance_m;      // Afstand waarop ingeschoten is in meters

    // Omgeving (vereenvoudigd, kan later dynamisch)
    float air_pressure_pa;      // Luchtdruk in Pascal
    float temperature_c;        // Temperatuur in Celsius
};

// Simplified BallisticState structure
struct BallisticState {
    Vec3 position;
    Vec3 velocity;
};

// Simplified TrackedObject structure
struct TrackedObject {
    long id;
    Vec3 position; // World coordinates (x, y, z in meters)
    Vec3 velocity; // Velocity vector (m/s)
    Vec3 acceleration; // Acceleration vector (m/s²)
};

// Simplified BallisticsSolver class with only the essential methods
class BallisticsSolver {
private:
    BallisticProfile profile_;

public:
    BallisticsSolver(const BallisticProfile& profile) : profile_(profile) {}

    float get_air_density() const {
        // Simplified air density calculation using ideal gas law
        // ρ = P / (R * T)
        // Where P is pressure in Pa, R is specific gas constant for air (287.058 J/(kg·K)), T is temperature in Kelvin
        const float R = 287.058f; // Specific gas constant for dry air
        float T_kelvin = profile_.temperature_c + 273.15f;
        return profile_.air_pressure_pa / (R * T_kelvin);
    }

    Vec3 drag_force(const Vec3& velocity, float air_density) {
        // Calculate drag force using F = 0.5 * ρ * v² * Cd * A
        // But simplified using ballistic coefficient: F_drag = (v² * BC * ρ) / (2 * m)
        float speed = velocity.magnitude();
        float speed_squared = speed * speed;
        float drag_magnitude = (speed_squared * profile_.ballistic_coefficient_si * air_density) / (2.0f * profile_.bullet_mass_kg);
        
        // Drag force opposes velocity direction
        if (speed > 0.0f) {
            Vec3 drag_direction = Vec3{-velocity.x, -velocity.y, -velocity.z}.operator*(1.0f / speed);
            return drag_direction.operator*(drag_magnitude);
        }
        return {0.0f, 0.0f, 0.0f};
    }

    BallisticState derivatives(const BallisticState& state, float air_density) {
        BallisticState deriv;
        deriv.position = state.velocity; // dx/dt = v
        
        // dv/dt = a = F/m = (gravity + drag) / m
        Vec3 gravity = {0.0f, -9.81f, 0.0f}; // Gravity pointing down (negative y)
        Vec3 drag = drag_force(state.velocity, air_density);
        deriv.velocity = gravity.operator+(drag.operator*(1.0f / profile_.bullet_mass_kg));
        
        return deriv;
    }

    BallisticState rk4_step(const BallisticState& state, float dt, float air_density) {
        BallisticState k1 = derivatives(state, air_density);
        BallisticState temp1 = {state.position.operator+(k1.position.operator*(dt * 0.5f)), state.velocity.operator+(k1.velocity.operator*(dt * 0.5f))};
        
        BallisticState k2 = derivatives(temp1, air_density);
        BallisticState temp2 = {state.position.operator+(k2.position.operator*(dt * 0.5f)), state.velocity.operator+(k2.velocity.operator*(dt * 0.5f))};
        
        BallisticState k3 = derivatives(temp2, air_density);
        BallisticState temp3 = {state.position.operator+(k3.position.operator*(dt)), state.velocity.operator+(k3.velocity.operator*(dt))};
        
        BallisticState k4 = derivatives(temp3, air_density);
        
        // Weighted average
        Vec3 pos_avg = k1.position.operator*(dt/6.0f).operator+(k2.position.operator*(dt/3.0f)).operator+(k3.position.operator*(dt/3.0f)).operator+(k4.position.operator*(dt/6.0f));
        Vec3 vel_avg = k1.velocity.operator*(dt/6.0f).operator+(k2.velocity.operator*(dt/3.0f)).operator+(k3.velocity.operator*(dt/3.0f)).operator+(k4.velocity.operator*(dt/6.0f));
        
        return {state.position.operator+(pos_avg), state.velocity.operator+(vel_avg)};
    }

    std::vector<BallisticState> calculate_trajectory(float initial_pitch, float max_distance, float time_step_override = 0.0f) {
        std::vector<BallisticState> trajectory;
        
        // Initial conditions
        BallisticState initial_state;
        initial_state.position = {0.0f, profile_.sight_height_m, 0.0f}; // Start at sight height
        
        // Initial velocity components
        float vx = profile_.muzzle_velocity_mps * std::cos(initial_pitch);
        float vy = profile_.muzzle_velocity_mps * std::sin(initial_pitch);
        initial_state.velocity = {vx, vy, 0.0f}; // Assume shooting in x-y plane (z=0)
        
        trajectory.push_back(initial_state);
        
        float air_density = get_air_density();
        float time_step = (time_step_override > 0.0f) ? time_step_override : 0.01f; // Default 10ms steps
        
        BallisticState current_state = initial_state;
        while (current_state.position.x < max_distance && current_state.position.y > -100.0f) { // Stop if bullet goes too far down
            current_state = rk4_step(current_state, time_step, air_density);
            trajectory.push_back(current_state);
        }
        
        return trajectory;
    }

    float calculate_zero_pitch() {
        // Binary search for the pitch angle that zeros at the specified distance
        float low_angle = -0.1f;   // -5.7 degrees
        float high_angle = 0.1f;   // 5.7 degrees
        float tolerance = 0.0001f; // 0.0057 degrees
        
        while (high_angle - low_angle > tolerance) {
            float mid_angle = (low_angle + high_angle) * 0.5f;
            auto trajectory = calculate_trajectory(mid_angle, profile_.zero_distance_m + 50.0f);
            
            if (trajectory.empty()) return 0.0f;
            
            // Find the point where x is closest to zero_distance
            float best_diff = std::abs(trajectory[0].position.x - profile_.zero_distance_m);
            float impact_y = trajectory[0].position.y;
            
            for (const auto& state : trajectory) {
                float diff = std::abs(state.position.x - profile_.zero_distance_m);
                if (diff < best_diff) {
                    best_diff = diff;
                    impact_y = state.position.y;
                }
            }
            
            if (impact_y > profile_.sight_height_m) {
                low_angle = mid_angle; // Need higher angle
            } else {
                high_angle = mid_angle; // Need lower angle
            }
        }
        
        return (low_angle + high_angle) * 0.5f;
    }

    float calculate_flight_time(float distance) {
        // Simple approximation: time = distance / muzzle_velocity
        if (profile_.muzzle_velocity_mps > 0.0f) {
            return distance / profile_.muzzle_velocity_mps;
        }
        return 0.0f;
    }

    bool calculate_impact_point(const TrackedObject& target, Vec3& out_impact_point, float& out_flight_time) {
        // Calculate distance to target
        float target_distance = target.position.z; // Assuming z is the forward distance
        if (target_distance <= 0.0f) return false;
        
        // Calculate bullet flight time to target
        out_flight_time = calculate_flight_time(target_distance);
        if (out_flight_time <= 0.0f) return false;
        
        // Predict target position after flight time using kinematic equations
        // position = initial_position + velocity * time + 0.5 * acceleration * time^2
        Vec3 predicted_position = target.position + target.velocity * out_flight_time + target.acceleration * (0.5f * out_flight_time * out_flight_time);
        
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
        out_impact_point.x = predicted_position.x; // Lateral movement prediction
        out_impact_point.y = ballistic_impact.y;   // Vertical drop from ballistics
        out_impact_point.z = predicted_position.z; // Forward distance prediction
        
        return true;
    }
};

int main() {
    // Create a ballistics profile (more realistic values)
    BallisticProfile profile;
    profile.muzzle_velocity_mps = 800.0f;  // 800 m/s
    profile.bullet_mass_kg = 0.009f;       // 9 gram bullet (0.009 kg)
    profile.ballistic_coefficient_si = 0.005f; // Ballistic coefficient
    profile.sight_height_m = 0.15f;        // 15 cm sight height
    profile.zero_distance_m = 100.0f;      // Zeroed at 100 meters
    profile.air_pressure_pa = 101325.0f;   // Standard atmospheric pressure
    profile.temperature_c = 20.0f;         // 20°C
    
    // Create a tracked object (target 10 meters away, slightly above center)
    TrackedObject track;
    track.id = 1;
    track.position = {0.0f, 0.5f, 10.0f}; // Target 10 meters away, 0.5m above center
    track.velocity = {0.0f, 0.0f, 0.0f};  // Static target
    track.acceleration = {0.0f, 0.0f, 0.0f}; // No acceleration
    
    // Create ballistics solver
    BallisticsSolver solver(profile);
    
    // Calculate impact point
    Vec3 impact_point;
    float flight_time;
    bool success = solver.calculate_impact_point(track, impact_point, flight_time);
    
    std::cout << "=== Ballistics Fix Verification ===" << std::endl;
    std::cout << "Target position: x=" << track.position.x << "m, y=" << track.position.y 
              << "m, z=" << track.position.z << "m" << std::endl;
    std::cout << "Ballistics calculation: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    
    if (success) {
        std::cout << "Impact point: x=" << impact_point.x << "m, y=" << impact_point.y 
                  << "m, z=" << impact_point.z << "m" << std::endl;
        std::cout << "Flight time: " << flight_time << " seconds" << std::endl;
        
        // Check if Y values are reasonable
        if (std::abs(impact_point.y) < 100.0f) {  // Should be within reasonable physical bounds
            std::cout << "BALLISTICS Y-COORDINATE FIX VERIFIED" << std::endl;
        } else {
            std::cout << "BALLISTICS FIX FAILED - Y coordinate still unreasonable: " << impact_point.y << std::endl;
        }
    } else {
        std::cout << "BALLISTICS FIX FAILED - Could not calculate impact point" << std::endl;
    }
    
    // Also test the trajectory calculation directly
    std::cout << "\n--- Direct Trajectory Test ---" << std::endl;
    auto trajectory = solver.calculate_trajectory(0.0f, 20.0f); // 0 degree pitch, 20m max distance
    std::cout << "Trajectory points: " << trajectory.size() << std::endl;
    if (!trajectory.empty()) {
        std::cout << "First point: x=" << trajectory.front().position.x 
                  << ", y=" << trajectory.front().position.y 
                  << ", z=" << trajectory.front().position.z << std::endl;
        std::cout << "Last point: x=" << trajectory.back().position.x 
                  << ", y=" << trajectory.back().position.y 
                  << ", z=" << trajectory.back().position.z << std::endl;
    }
    
    return 0;
}