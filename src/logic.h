#ifndef LOGIC_H
#define LOGIC_H

#include "pipeline_structs.h"
#include <chrono> // For std::chrono
#include <thread> // For std::thread

/**
 * @brief Represents a single tracked object.
 *
 * This struct holds the state of a dynamically tracked object, including its unique ID,
 * the last known detection, estimated 3D position and velocity, and a history
 * for more robust tracking.
 */
struct TrackedObject {
    long id;                               ///< Unique identifier for this tracked object.
    DetectionResult last_detection;        ///< The last detection associated with this track.
    
    // Estimated 3D state (simplified for now)
    float pos_x, pos_y, pos_z;             ///< Estimated 3D position (e.g., in camera frame, meters).
    float vel_x, vel_y, vel_z;             ///< Estimated 3D velocity (e.g., in meters/second).

    std::chrono::high_resolution_clock::time_point last_update_time; ///< Timestamp of the last update.
    int hit_streak;                        ///< Number of consecutive frames this object has been detected.
    int missed_frames;                     ///< Number of consecutive frames this object has been missed.
    bool associated_this_frame;            ///< Flag to indicate if the track was associated in the current frame.

    TrackedObject(long _id, const DetectionResult& detection, float initial_distance)
        : id(_id), last_detection(detection), 
          pos_x(0.0f), pos_y(0.0f), pos_z(initial_distance), // Assuming initial distance provides Z
          vel_x(0.0f), vel_y(0.0f), vel_z(0.0f),
          last_update_time(detection.timestamp),
          hit_streak(1), missed_frames(0), associated_this_frame(true) {
              // A more sophisticated initialization would convert 2D bbox to 3D position
              // using camera intrinsics and the initial_distance.
          }
};

// Forward declaration for IMUSensor
class IMUSensor;

enum SafetyStatus {
    SAFETY_OK,
    SAFETY_WARNING_UNCERTAINTY,
    SAFETY_WARNING_TRACK_UNSTABLE,
    SAFETY_CRITICAL_UNCERTAINTY,
    SAFETY_CRITICAL_OTHER // For other critical failures
};

enum FallbackMode {
    NORMAL_OPERATION,
    FALLBACK_A_REDUCED_PERFORMANCE,
    FALLBACK_B_WARNING_STATE,
    FALLBACK_C_SAFE_SHUTDOWN
};

class LogicModule {
public:
    LogicModule(DetectionResultsQueue& detection_input_queue, std::shared_ptr<IMUSensor> imu_sensor);
    ~LogicModule();

    bool start();
    void stop();
    bool is_running() const { return running_; }

    void get_performance_metrics();

private:
    void worker_thread_func();

    // The main function for the centralized logic module
    void process(const std::vector<DetectionResult>& detections, const IMUData& imu_data);

    DetectionResultsQueue& detection_input_queue_;
    std::atomic<bool> running_ = false;
    std::thread worker_thread_;
    std::shared_ptr<IMUSensor> imu_sensor_;

    std::vector<TrackedObject> active_tracks_; ///< Currently active tracked objects.
    static long next_track_id_;                ///< Counter for generating unique track IDs.

    static float calculate_iou(const DetectionResult& det1, const DetectionResult& det2);

    // Helper method for hit-scan/predicted impact point calculation
    // Returns true if an impact point can be predicted, false otherwise
    bool predict_impact_point(const TrackedObject& target, const IMUData& current_imu_data, float& out_x, float& out_y, float& out_z);

    // Helper method for safety and uncertainty propagation
    SafetyStatus perform_safety_and_uncertainty_checks(const TrackedObject& target, float predicted_impact_uncertainty, std::string& safety_status_message);

    // Helper method for servo actuation API
    void issue_servo_commands(float target_x, float target_y, float target_z);
    
    // Performance measurement members
    std::vector<long long> prediction_times_ms_;
    std::mutex prediction_times_mutex_;
    long long total_predictions_ = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> performance_start_time_;
    FallbackMode current_fallback_mode_;
};

#endif // LOGIC_H