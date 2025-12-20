#include "src/logic.h"
#include "src/config_loader.h"
#include <iostream>
#include <vector>

int main() {
    // Create a mock configuration
    ConfigLoader config;
    
    // Create a mock detection result (static target at center of frame)
    DetectionResult detection;
    detection.class_id = 0;  // person
    detection.score = 0.95f;
    detection.xmin = 0.4f;   // Bounding box slightly left of center
    detection.ymin = 0.4f;   // Bounding box slightly above center
    detection.xmax = 0.6f;   // Bounding box slightly right of center
    detection.ymax = 0.6f;   // Bounding box slightly below center
    detection.timestamp = std::chrono::high_resolution_clock::now();
    
    // Create a mock orientation data
    OrientationData imu_data;
    imu_data.pitch = 0.0f;
    imu_data.roll = 0.0f;
    imu_data.yaw = 0.0f;
    
    // Create a tracked object with proper world coordinates
    float estimated_distance = 10.0f; // 10 meters away
    float x_world = 0.0f;  // Center of frame
    float y_world = 0.5f;  // Slightly above center
    
    TrackedObject track(1, detection, estimated_distance, x_world, y_world);
    
    // Create a ballistics profile
    BallisticProfile profile;
    profile.muzzle_velocity_mps = 800.0f;  // 800 m/s
    profile.bullet_mass_kg = 0.009f;       // 9 gram bullet (0.009 kg)
    profile.ballistic_coefficient_si = 0.005f; // Ballistic coefficient
    profile.sight_height_m = 0.15f;        // 15 cm sight height
    profile.zero_distance_m = 100.0f;      // Zeroed at 100 meters
    profile.air_pressure_pa = 101325.0f;   // Standard atmospheric pressure
    profile.temperature_c = 20.0f;         // 20°C
    
    // Create ballistics solver
    BallisticsSolver solver(profile);
    
    // Calculate impact point
    Vec3 impact_point;
    float flight_time;
    bool success = solver.calculate_impact_point(track, imu_data, impact_point, flight_time);
    
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
    
    return 0;
}