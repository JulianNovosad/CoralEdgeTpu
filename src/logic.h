#ifndef LOGIC_H
#define LOGIC_H

#include "pipeline_structs.h"
#include "orientation_sensor.h"
#include "config_loader.h" // Include de config loader
#include "pca9685_controller.h" // PCA9685 LED controller
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <cmath>
#include <zmq.hpp> // ZeroMQ C++ bindings
#include <map>
#include <fstream>
#include "json.hpp" // nlohmann/json header

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
    float dot(const Vec3& other) const { return x*other.x + y*other.y + z*other.z; }
    Vec3 cross(const Vec3& other) const { return {y*other.z - z*other.y, z*other.x - x*other.z, x*other.y - y*other.x}; }
    Vec3 normalize() const { float mag = magnitude(); return mag > 0 ? Vec3{x/mag, y/mag, z/mag} : Vec3{0,0,0}; }
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

/**
 * @brief Enumeratie voor de fallback-modi van het systeem.
 */

/**
 * @brief Structuur voor het bijhouden van onzekerheid in metingen.
 */
struct Uncertainty {
    float position_variance;     // Variantie in positie (m^2)
    float velocity_variance;     // Variantie in snelheid (m^2/s^2)
    float distance_variance;     // Variantie in afstand (m^2)
    float total_confidence;     // Totale betrouwbaarheid (0.0 - 1.0)
};

/**
 * @brief Representeert een enkel gevolgd object.
 */
struct TrackedObject {
    long id;
    DetectionResult last_detection;
    
    Vec3 position; // Gebruikt nu Vec3
    Vec3 velocity; // Gebruikt nu Vec3
    Vec3 acceleration; // Nieuw: versnelling voor betere voorspellingen

    std::chrono::high_resolution_clock::time_point last_update_time;
    int hit_streak;
    int missed_frames;
    bool associated_this_frame;
    
    // Nieuw: onzekerheid in positie en snelheid
    Vec3 position_uncertainty;
    Vec3 velocity_uncertainty;
    
    // Nieuw: voorspelde impact punt en onzekerheid
    Vec3 predicted_impact_point;
    Uncertainty uncertainty;

    TrackedObject(long _id, const DetectionResult& detection, float initial_distance, float initial_x = 0.0f, float initial_y = 0.0f)
        : id(_id), last_detection(detection), 
          position({initial_x, initial_y, initial_distance}), // Init positie
          velocity({0.0f, 0.0f, 0.0f}),
          acceleration({0.0f, 0.0f, 0.0f}),
          last_update_time(detection.timestamp),
          hit_streak(1), missed_frames(0), associated_this_frame(true),
          position_uncertainty({0.1f, 0.1f, 0.1f}), // Initiële onzekerheid
          velocity_uncertainty({0.1f, 0.1f, 0.1f}),
          predicted_impact_point({0.0f, 0.0f, 0.0f}),
          uncertainty() {}
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
    FALLBACK_C_CRITICAL_STATE
};


/**
 * @brief Klasse voor het uitvoeren van 3D ballistische berekeningen met RK4.
 */
class BallisticsSolver {
public:
    BallisticsSolver(const BallisticProfile& profile);
    
    std::vector<BallisticState> calculate_trajectory(float initial_pitch, float max_distance, float time_step = 0.0f);
    float calculate_zero_pitch();
    float calculate_flight_time(float distance);
    
    // New method for calculating impact point based on target movement and orientation
    bool calculate_impact_point(const TrackedObject& target, const OrientationData& imu_data, Vec3& out_impact_point, float& out_flight_time);

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
    void servo_worker_thread_func(); // New servo worker thread function
    void process(const std::vector<DetectionResult>& detections, const OrientationData& imu_data);
    void update_object_tracks(const std::vector<DetectionResult>& detections);
    SafetyStatus perform_safety_and_uncertainty_checks(const TrackedObject& target, const Uncertainty& uncertainty, std::string& safety_status_message);
    void issue_servo_commands(float target_x, float target_y, float target_z, float confidence);
    void execute_servo_command(float target_x, float target_y, float target_z, float confidence); // New function to actually execute servo commands
    float calculate_iou(const DetectionResult& det1, const DetectionResult& det2);
    void perform_sensor_fusion(const OrientationData& imu_data);
    void calculate_ballistics_for_tracks(const OrientationData& imu_data);
    void perform_safety_and_actuation(const OrientationData& imu_data);
    void send_telemetry_data(const std::string& telemetry_message);
    
    // New methods for enhanced ballistics and uncertainty
    Uncertainty propagate_uncertainty(const TrackedObject& target, float flight_time);
    float estimate_target_distance(const DetectionResult& detection);
    float calculate_impact_point_distance(const Vec3& impact_point, const Vec3& crosshair_point);
    
    // Removed incorrect function:
    // float calculate_angular_error_degrees(const Vec3& impact_point, const Vec3& crosshair_point, float target_distance);

    DetectionResultsQueue& detection_input_queue_;
    std::atomic<bool> running_ = false;
    
    // Freshness indicators
public:
    std::atomic<long long> last_logic_timestamp_{0}; ///< Timestamp of the last logic processing
    std::atomic<int> logic_rate_{0}; ///< Current logic processing rate
private:
    std::thread worker_thread_;
    std::shared_ptr<OrientationSensor> orientation_sensor_;
    const ConfigLoader& config_;

    std::vector<TrackedObject> active_tracks_;
    static long next_track_id_;

    std::unique_ptr<BallisticsSolver> ballistics_solver_;

    // New configuration parameters for tracking
    int max_active_tracks_;
    float track_iou_threshold_;
    int track_missed_frames_threshold_;
    float min_track_confidence_;

    FallbackMode current_fallback_mode_ = NORMAL_OPERATION;
    long current_hit_scan_count_ = 0;
    long current_servo_command_count_ = 0;
    
    // ZeroMQ socket for sending telemetry data
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::unique_ptr<zmq::socket_t> telemetry_socket_;
    
    // PCA9685 LED controller
    std::unique_ptr<PCA9685Controller> led_controller_;
    
    // Servo control variables for thread safety
    float servo_position_ = 0.0f;
    bool servo_direction_ = true;
    std::chrono::steady_clock::time_point last_direction_change_;
    
    // Servo command queue for decoupled actuation
    struct ServoCommand {
        float target_x;
        float target_y;
        float target_z;
        float confidence;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::queue<ServoCommand> servo_command_queue_;
    std::mutex servo_queue_mutex_;
    std::thread servo_worker_thread_;
    std::atomic<bool> servo_worker_running_ = false;
    
    // Storage for latest IMU data
    OrientationData latest_imu_data_;
    
    // Per-class distance tracking for multi-class smoothing
    static const size_t CLASS_DISTANCE_WINDOW_SIZE = 10;
    struct ClassDistanceHistory {
        std::vector<float> distances;
        size_t index;
        bool full;
        ClassDistanceHistory() : distances(CLASS_DISTANCE_WINDOW_SIZE, 0.0f), index(0), full(false) {}
    };
    std::map<int, ClassDistanceHistory> class_distance_histories_;
    std::map<int, std::string> class_names_;  // Map class IDs to human-readable names
    
    // Global distance tracking (existing)
    static const size_t DISTANCE_WINDOW_SIZE = 10;
    std::vector<float> distance_history_;
    size_t distance_history_index_ = 0;
    bool distance_history_full_ = false;
    
    // Class distance map for corrections
    std::map<int, float> class_distance_map_;
    
    // Class scale factors for distance calibration
    std::map<int, float> class_scale_factors_;
    
    // Methods for class-specific distance handling
    bool load_class_distance_map(const std::string& filepath);
    bool load_class_scale_factors(const std::string& filepath);
    bool load_labelmap(const std::string& filepath);
    float apply_class_correction(int class_id, float raw_distance);
    float add_class_distance_estimate(int class_id, float distance);
    float get_smoothed_class_distance(int class_id);
    std::vector<std::pair<int, float>> get_top_classes_with_distances(size_t count = 3);
    
    // Method to add a distance estimate to the rolling window and get smoothed value
    float add_distance_estimate(float distance);

    // Camera intrinsics for angular error calculation
    float focal_length_px_;  // Focal length in pixels
    float image_center_x_;   // Image center X coordinate in pixels
    float image_center_y_;   // Image center Y coordinate in pixels
    
    // New method for camera-space angular error calculation
    float camera_cone_error_degrees_from_pixels(float radial_px) const;

};

#endif // LOGIC_H