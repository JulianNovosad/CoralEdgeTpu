/**
 * @file config_loader.h
 * @brief Laadt en beheert de applicatieconfiguratie vanuit een JSON-bestand.
 */
#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include <chrono>
#include "json.hpp" // nlohmann/json header

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