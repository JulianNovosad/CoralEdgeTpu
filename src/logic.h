#ifndef LOGIC_H
#define LOGIC_H

#include "pipeline_structs.h"
#include "orientation_sensor.h"
#include "config_loader.h" // Include de config loader
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <cmath>

// --- Nieuwe 3D Ballistiek Structuren ---

/**
 * @brief Een eenvoudige 3D-vectorstructuur.
 */
struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    float magnitude() const { return std::sqrt(x*x + y*y + z*z); }
};

/**
 * @brief Profiel voor wapen- en munitie-eigenschappen.
 */
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

/**
 * @brief Toestandsvector voor de RK4-solver.
 */
struct BallisticState {
    Vec3 position;
    Vec3 velocity;
};

// --- Einde Nieuwe 3D Ballistiek Structuren ---


/**
 * @brief Representeert een enkel gevolgd object.
 */
struct TrackedObject {
    long id;
    DetectionResult last_detection;
    
    Vec3 position; // Gebruikt nu Vec3
    Vec3 velocity; // Gebruikt nu Vec3

    std::chrono::high_resolution_clock::time_point last_update_time;
    int hit_streak;
    int missed_frames;
    bool associated_this_frame;

    TrackedObject(long _id, const DetectionResult& detection, float initial_distance)
        : id(_id), last_detection(detection), 
          position({0.0f, 0.0f, initial_distance}), // Init positie
          velocity({0.0f, 0.0f, 0.0f}),
          last_update_time(detection.timestamp),
          hit_streak(1), missed_frames(0), associated_this_frame(true) {}
};

/**
 * @brief Enumeratie voor de veiligheidsstatus van het systeem.
 */
enum SafetyStatus {
    SAFETY_OK,
    SAFETY_WARNING_UNCERTAINTY,
    SAFETY_WARNING_TRACK_UNSTABLE,
    SAFETY_CRITICAL_UNCERTAINTY,
    SAFETY_CRITICAL_OTHER
};

/**
 * @brief Enumeratie voor de fallback-modi van het systeem.
 */
enum FallbackMode {
    NORMAL_OPERATION,
    FALLBACK_A_REDUCED_PERFORMANCE,
    FALLBACK_B_WARNING_STATE,
    FALLBACK_C_SAFE_SHUTDOWN
};


/**
 * @brief Klasse voor het uitvoeren van 3D ballistische berekeningen met RK4.
 */
class BallisticsSolver {
public:
    BallisticsSolver(const BallisticProfile& profile);
    
    std::vector<BallisticState> calculate_trajectory(float initial_pitch, float max_distance, float time_step = 0.0f);
    float calculate_zero_pitch();

private:
    BallisticProfile profile_;
    float zero_pitch_rad_ = 0.0f;
    
    Vec3 drag_force(const Vec3& velocity, float air_density);
    BallisticState derivatives(const BallisticState& state, float air_density);
    BallisticState rk4_step(const BallisticState& state, float dt, float air_density);
    float get_air_density() const;
};


/**
 * @brief De centrale logica-module.
 */
class LogicModule {
public:
    LogicModule(DetectionResultsQueue& detection_input_queue, std::shared_ptr<OrientationSensor> orientation_sensor, const ConfigLoader& config);
    ~LogicModule();

    bool start();
    void stop();
    bool is_running() const { return running_; }

private:
    void worker_thread_func();
    void process(const std::vector<DetectionResult>& detections, const OrientationData& imu_data);
    void update_object_tracks(const std::vector<DetectionResult>& detections);
    SafetyStatus perform_safety_and_uncertainty_checks(const TrackedObject& target, float predicted_impact_uncertainty, std::string& safety_status_message);
    void issue_servo_commands(float target_x, float target_y, float target_z);
    bool predict_impact_point(const TrackedObject& target, const OrientationData& current_imu_data, Vec3& out_impact_point);
    float calculate_iou(const DetectionResult& det1, const DetectionResult& det2);
    void perform_sensor_fusion(const OrientationData& imu_data);
    void calculate_ballistics_for_tracks(const OrientationData& imu_data);
    void perform_safety_and_actuation(const OrientationData& imu_data);


    DetectionResultsQueue& detection_input_queue_;
    std::atomic<bool> running_ = false;
    std::thread worker_thread_;
    std::shared_ptr<OrientationSensor> orientation_sensor_;

    std::vector<TrackedObject> active_tracks_;
    static long next_track_id_;

    std::unique_ptr<BallisticsSolver> ballistics_solver_;

    // New configuration parameters for tracking
    int max_active_tracks_;
    float track_iou_threshold_;
    int track_missed_frames_threshold_;

    FallbackMode current_fallback_mode_ = NORMAL_OPERATION;
    long current_hit_scan_count_ = 0;
    long current_servo_command_count_ = 0;

};

#endif // LOGIC_H
