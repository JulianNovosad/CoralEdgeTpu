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

    /** @return Pad naar het TFLite-modelbestand. Standaard: "model.tflite". */
    std::string get_model_path() const;

    /** @return Pad naar het labels-bestand. Standaard: "labels.txt". */
    std::string get_labels_path() const;

    /** @return Breedte van de hoge-resolutie videostream. Standaard: 1920. */
    unsigned int get_high_res_width() const;

    /** @return Hoogte van de hoge-resolutie videostream. Standaard: 1080. */
    unsigned int get_high_res_height() const;

    /** @return Timeout voor de camera-watchdog in seconden. Standaard: 10. */
    std::chrono::seconds get_camera_watchdog_timeout() const;

    /** @return Aantal worker threads voor inferentie. Standaard: 1. */
    int get_inference_worker_threads() const;

    /** @return JPEG-kwaliteit voor de HTTP-stream. Standaard: 90. */
    int get_jpeg_quality() const;

    /** @return Doel-FPS voor de camera. Standaard: 30.0. */
    double get_camera_fps() const;

    /** @return Minimale betrouwbaarheidsscore voor detecties. Standaard: 0.5. */
    float get_detection_score_threshold() const;

    /** @return Pad voor logbestanden. Standaard: "/home/pi/CoralEdgeTpu/logs". */
    std::string get_log_path() const;

    /** @return Protocol voor videostreaming (e.g., "RTP", "HTTP_WEBSOCKET"). Standaard: "HTTP_WEBSOCKET". */
    std::string get_video_stream_protocol() const;

    /** @return Adres voor videostreaming. Standaard: "0.0.0.0". */
    std::string get_video_stream_address() const;

    /** @return Poort voor videostreaming. Standaard: 5000. */
    unsigned short get_video_stream_port() const;

    /** @return Protocol voor telemetriestreaming (e.g., "ZeroMQ", "HTTP_WEBSOCKET"). Standaard: "HTTP_WEBSOCKET". */
    std::string get_telemetry_protocol() const;

    /** @return Publicatie-adres voor telemetriestreaming (ZeroMQ). Standaard: "tcp://*:6000". */
    std::string get_telemetry_pub_address() const;

    // --- Netwerkconfiguratie ---

    /** @return Het IP-adres waarop de server moet luisteren. Standaard: "0.0.0.0". */
    std::string get_listen_address() const;

    /** @return TCP-poort voor de H.264 videostream. Standaard: 1001. */
    unsigned short get_livestream_video_port() const;

    /** @return TCP-poort voor de bounding box-stream. Standaard: 1002. */
    unsigned short get_bounding_box_stream_port() const;

    /** @return TCP-poort voor het richtpunt-coördinaat. Standaard: 1003. */
    unsigned short get_reticle_coordinate_port() const;

    /** @return TCP-poort voor de status/telemetrie-stream. Standaard: 1004. */
    unsigned short get_status_telemetry_port() const;

    /** @return UDP-poort voor oriëntatie (yaw) van de telefoon. Standaard: 2001. */
    unsigned short get_phone_orientation_yaw_port() const;

    /** @return UDP-poort voor oriëntatie (pitch) van de telefoon. Standaard: 2002. */
    unsigned short get_phone_orientation_pitch_port() const;

    /** @return UDP-poort voor oriëntatie (roll) van de telefoon. Standaard: 2003. */
    unsigned short get_phone_orientation_roll_port() const;

private:
    nlohmann::json config_data_; ///< De geparste JSON-configuratiedata.
};

#endif // CONFIG_LOADER_H