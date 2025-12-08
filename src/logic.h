/**
 * @file logic.h
 * @brief Definieert de hoofdlogica-module voor ballistische berekeningen,
 * object-tracking en veiligheidscontroles.
 */
#ifndef LOGIC_H
#define LOGIC_H

#include "pipeline_structs.h"
#include "orientation_sensor.h"
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>

/**
 * @brief Representeert een enkel gevolgd object.
 *
 * Deze struct bevat de status van een dynamisch gevolgd object, inclusief zijn unieke ID,
 * de laatst bekende detectie, geschatte 3D-positie en -snelheid, en een historie
 * voor robuustere tracking.
 */
struct TrackedObject {
    long id;                               ///< Unieke identifier voor dit gevolgde object.
    DetectionResult last_detection;        ///< De laatste detectie die aan dit spoor is gekoppeld.
    
    // Geschatte 3D-toestand (vereenvoudigd)
    float pos_x, pos_y, pos_z;             ///< Geschatte 3D-positie (bijv. in cameracoördinaten, meters).
    float vel_x, vel_y, vel_z;             ///< Geschatte 3D-snelheid (bijv. in meters/seconde).

    std::chrono::high_resolution_clock::time_point last_update_time; ///< Tijdstip van de laatste update.
    int hit_streak;                        ///< Aantal opeenvolgende frames dat dit object is gedetecteerd.
    int missed_frames;                     ///< Aantal opeenvolgende frames dat dit object is gemist.
    bool associated_this_frame;            ///< Vlag om aan te geven of het spoor in het huidige frame is geassocieerd.

    TrackedObject(long _id, const DetectionResult& detection, float initial_distance)
        : id(_id), last_detection(detection), 
          pos_x(0.0f), pos_y(0.0f), pos_z(initial_distance), // Aanname dat initiële afstand Z levert
          vel_x(0.0f), vel_y(0.0f), vel_z(0.0f),
          last_update_time(detection.timestamp),
          hit_streak(1), missed_frames(0), associated_this_frame(true) {}
};

/**
 * @brief Enumeratie voor de veiligheidsstatus van het systeem.
 */
enum SafetyStatus {
    SAFETY_OK,                          ///< Systeem functioneert normaal.
    SAFETY_WARNING_UNCERTAINTY,         ///< Waarschuwing: onzekerheid in voorspelling is hoog.
    SAFETY_WARNING_TRACK_UNSTABLE,      ///< Waarschuwing: object-track is onstabiel.
    SAFETY_CRITICAL_UNCERTAINTY,        ///< Kritiek: onzekerheid overschrijdt veilige drempels.
    SAFETY_CRITICAL_OTHER               ///< Kritiek: andere kritieke fout.
};

/**
 * @brief Enumeratie voor de fallback-modi van het systeem.
 */
enum FallbackMode {
    NORMAL_OPERATION,                   ///< Normale werking.
    FALLBACK_A_REDUCED_PERFORMANCE,     ///< Terugvalmodus A: verminderde prestaties.
    FALLBACK_B_WARNING_STATE,           ///< Terugvalmodus B: waarschuwingsstatus.
    FALLBACK_C_SAFE_SHUTDOWN            ///< Terugvalmodus C: veilige uitschakeling.
};

/**
 * @brief De centrale logica-module.
 *
 * Verwerkt detectieresultaten, fuseert sensordata, volgt objecten,
 * berekent ballistiek en voert veiligheidscontroles uit.
 */
class LogicModule {
public:
    /**
     * @brief Constructor voor de LogicModule.
     * @param detection_input_queue Wachtrij voor inkomende detectieresultaten.
     * @param orientation_sensor Gedeelde pointer naar de oriëntatiesensor.
     */
    LogicModule(DetectionResultsQueue& detection_input_queue, std::shared_ptr<OrientationSensor> orientation_sensor);
    ~LogicModule();

    /**
     * @brief Start de worker thread van de logica-module.
     * @return True bij succes, anders false.
     */
    bool start();

    /**
     * @brief Stopt de worker thread van de logica-module.
     */
    void stop();

    /**
     * @brief Controleert of de module draait.
     * @return True als de module draait, anders false.
     */
    bool is_running() const { return running_; }

    /**
     * @brief Berekent en logt prestatie-indicatoren.
     */
    void get_performance_metrics();

private:
    void worker_thread_func();

    // De hoofdverwerkingsfunctie
    void process(const std::vector<DetectionResult>& detections, const OrientationData& imu_data);

    DetectionResultsQueue& detection_input_queue_;
    std::atomic<bool> running_ = false;
    std::thread worker_thread_;
    std::shared_ptr<OrientationSensor> orientation_sensor_;

    std::vector<TrackedObject> active_tracks_; ///< Huidige actieve gevolgde objecten.
    static long next_track_id_;                ///< Teller voor het genereren van unieke track-ID's.

    // Hulpmethode voor het berekenen van Intersection over Union (IoU).
    static float calculate_iou(const DetectionResult& det1, const DetectionResult& det2);

    // Hulpmethode voor het voorspellen van het inslagpunt.
    bool predict_impact_point(const TrackedObject& target, const OrientationData& current_imu_data, float& out_x, float& out_y, float& out_z);

    // Hulpmethode voor veiligheids- en onzekerheidscontroles.
    SafetyStatus perform_safety_and_uncertainty_checks(const TrackedObject& target, float predicted_impact_uncertainty, std::string& safety_status_message);

    // Hulpmethode voor servo-aansturing.
    void issue_servo_commands(float target_x, float target_y, float target_z);
    
    // Leden voor prestatiemetingen
    std::vector<long long> prediction_times_ms_;
    std::mutex prediction_times_mutex_;
    long long total_predictions_ = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> performance_start_time_;
    FallbackMode current_fallback_mode_;

    // Private hulpmethoden voor het refactoren van de process-functie
    void perform_sensor_fusion(const OrientationData& imu_data);
    void update_object_tracks(const std::vector<DetectionResult>& detections);
    void calculate_ballistics_for_tracks(const OrientationData& imu_data);
    void perform_safety_and_actuation(const OrientationData& imu_data);
    void update_process_metrics(std::chrono::high_resolution_clock::time_point processing_start_time, std::chrono::high_resolution_clock::time_point processing_end_time);
};

#endif // LOGIC_H