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
    
    // Create a tracked object
    float estimated_distance = 10.0f; // 10 meters away
    
    // Convert normalized detection coordinates to world coordinates
    float center_x_norm = (detection.xmin + detection.xmax) * 0.5f;
    float center_y_norm = (detection.ymin + detection.ymax) * 0.5f;
    
    // Convert to pixel coordinates (assuming 640x480 resolution)
    float center_x_px = center_x_norm * 640.0f;
    float center_y_px = center_y_norm * 480.0f;
    
    // Convert to centered coordinates (relative to image center)
    float center_x_centered = center_x_px - (640.0f * 0.5f);
    float center_y_centered = center_y_px - (480.0f * 0.5f);
    
    // Convert to real-world coordinates using pinhole camera model
    // Using the same constants as in logic.cpp
    constexpr float CAMERA_FOCAL_LENGTH_CM = 4.74f;
    constexpr float SENSOR_WIDTH_CM = 0.64f;
    float focal_length_pixels = (CAMERA_FOCAL_LENGTH_CM * 640.0f) / SENSOR_WIDTH_CM;
    float x_world = (center_x_centered * estimated_distance) / focal_length_pixels;
    float y_world = (center_y_centered * estimated_distance) / focal_length_pixels;
    
    // Create TrackedObject with proper initial position
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
    std::cout << "Input detection: box=(" << detection.xmin << "," << detection.ymin << "," 
              << detection.xmax << "," << detection.ymax << ")" << std::endl;
    std::cout << "Estimated distance: " << estimated_distance << " meters" << std::endl;
    std::cout << "World coordinates: x=" << x_world << "m, y=" << y_world << "m, z=" << estimated_distance << "m" << std::endl;
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