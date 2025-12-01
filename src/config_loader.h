#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include "json.hpp" // nlohmann/json header

class ConfigLoader {
public:
    // Loads configuration from the specified JSON file.
    bool load(const std::string& config_file_path);

    // Getters for configuration values
    std::string get_model_path() const;
    std::string get_labels_path() const;
    unsigned int get_high_res_width() const;
    unsigned int get_high_res_height() const;
    int get_udp_raw_video_port() const;
    int get_udp_bounding_box_port() const;
    int get_http_overlaid_video_port() const;
    std::string get_mobile_app_ip() const;
    std::chrono::seconds get_camera_watchdog_timeout() const;
    int get_inference_worker_threads() const;
    int get_jpeg_quality() const;
    double get_camera_fps() const;

private:
    nlohmann::json config_data_;
};

#endif // CONFIG_LOADER_H