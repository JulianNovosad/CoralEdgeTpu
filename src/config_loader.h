/**
 * @file config_loader.h
 * @brief Laadt en beheert de applicatieconfiguratie vanuit een JSON-bestand.
 */
#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include <chrono>
#include "json.hpp" // nlohmann/json header
#include <libcamera/pixel_format.h> // For libcamera::PixelFormat
#include <libcamera/formats.h>    // For libcamera::formats

/**
 * @brief Laadt configuratie-instellingen uit een JSON-bestand.
 *
 * Deze klasse parseert een JSON-bestand en biedt C++-vriendelijke getter-methoden
 * om toegang te krijgen tot configuratieparameters.
 */
class ConfigLoader {
public:
    /**
     * @brief Laadt de configuratie uit het opgegeven JSON-bestand.
     * @param config_file_path Het pad naar het JSON-configuratiebestand.
     * @return True als het laden succesvol was, anders false.
     */
    bool load(const std::string& config_file_path);

    // --- Getters voor configuratiewaarden ---

    std::string get_model_path() const;
    std::string get_labels_path() const;
    unsigned int get_high_res_width() const;
    unsigned int get_high_res_height() const;
    std::chrono::seconds get_camera_watchdog_timeout() const;
    int get_inference_worker_threads() const;
    int get_jpeg_quality() const;
    double get_camera_fps() const;
    float get_detection_score_threshold() const;
    std::string get_log_path() const;

    // --- TPU Stream Configuration ---
    unsigned int get_tpu_stream_width() const;
    unsigned int get_tpu_stream_height() const;
    unsigned int get_tpu_stream_fps() const; // New: Get TPU stream FPS
    unsigned int get_tpu_target_width() const;
    unsigned int get_tpu_target_height() const;
    libcamera::PixelFormat get_tpu_stream_pixel_format() const;

    // --- Video & Telemetry ---
    std::string get_video_stream_protocol() const;
    std::string get_video_stream_address() const;
    unsigned short get_video_stream_port() const;
    std::string get_telemetry_protocol() const;
    std::string get_telemetry_pub_address() const;
    
    // --- Ballistiek ---
    /** @return Mondingssnelheid in m/s. Standaard: 850.0. */
    float get_muzzle_velocity_mps() const;
    /** @return Kogelmassa in kg. Standaard: 0.008. */
    float get_bullet_mass_kg() const;
    /** @return Ballistische coëfficiënt (G1 SI). Standaard: 0.25. */
    float get_ballistic_coefficient_si() const;
    /** @return Hoogte van vizier boven loop in meters. Standaard: 0.05. */
    float get_sight_height_m() const;
    /** @return Afstand waarop ingeschoten is in meters. Standaard: 100.0. */
    float get_zero_distance_m() const;
    /** @return Luchtdruk in Pascal. Standaard: 101325.0. */
    float get_air_pressure_pa() const;
    /** @return Temperatuur in Celsius. Standaard: 15.0. */
    float get_temperature_c() const;
    /** @return Drag model type (G1 or G7). Standaard: "G1". */
    std::string get_drag_model() const;
    /** @return Reference drag coefficient for the selected model. Standaard: 0.25. */
    float get_drag_coefficient_reference() const;

    // --- Tracking ---
    /** @return Maximale aantal actieve tracks. Standaard: 100. */
    int get_max_active_tracks() const;
    /** @return Minimale Intersection over Union (IoU) voor trackassociatie. Standaard: 0.3. */
    float get_track_iou_threshold() const;
    /** @return Aantal gemiste frames voordat een track wordt verwijderd. Standaard: 5. */
    int get_track_missed_frames_threshold() const;
    /** @return Minimale detectiebetrouwbaarheid om een nieuwe track aan te maken. Standaard: 0.6. */
    float get_min_track_confidence() const;

    // --- Safety Thresholds ---
    /** @return Minimum confidence threshold for safety checks. Standaard: 0.1. */
    float get_min_confidence_threshold() const;
    /** @return Maximum allowable position variance. Standaard: 0.75. */
    float get_max_position_variance() const;
    /** @return Confidence threshold for servo activation. Standaard: 0.9. */
    float get_servo_activate_confidence() const;
    /** @return Decay factor for confidence calculation. Standaard: 0.1. */
    float get_confidence_decay_factor() const;
    /** @return Factor for distance-based confidence. Standaard: 0.1. */
    float get_distance_confidence_factor() const;
    /** @return Maximum allowable angular error in degrees. Standaard: 1.0. */
    float get_max_angular_error_degrees() const;

    // --- Netwerkconfiguratie ---
    std::string get_listen_address() const;
    unsigned short get_phone_orientation_yaw_port() const;
    unsigned short get_phone_orientation_pitch_port() const;
    unsigned short get_phone_orientation_roll_port() const;

    /**
     * @brief Provides read-only access to the raw JSON configuration data.
     * @return A const reference to the nlohmann::json object containing the configuration.
     */
    const nlohmann::json& get_json_config() const;

private:
    nlohmann::json config_data_; ///< De geparste JSON-configuratiedata.
};

#endif // CONFIG_LOADER_H