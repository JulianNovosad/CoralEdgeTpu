#include "config_loader.h"
#include "util_logging.h"
#include <fstream>
#include <stdexcept>

bool ConfigLoader::load(const std::string& config_file_path) {
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open config file: " + config_file_path);
        return false;
    }

    try {
        file >> config_data_;
    } catch (const nlohmann::json::parse_error& e) {
        LOG_ERROR("Failed to parse config file: " + std::string(e.what()));
        return false;
    }
    LOG_INFO("Configuration loaded successfully from " + config_file_path);
    return true;
}

std::string ConfigLoader::get_model_path() const {
    return config_data_.value("/application/model_path"_json_pointer, "model.tflite");
}

std::string ConfigLoader::get_labels_path() const {
    return config_data_.value("/application/labels_path"_json_pointer, "labels.txt");
}

unsigned int ConfigLoader::get_high_res_width() const {
    return config_data_.value("/application/high_res_width"_json_pointer, 1920);
}

unsigned int ConfigLoader::get_high_res_height() const {
    return config_data_.value("/application/high_res_height"_json_pointer, 1080);
}

int ConfigLoader::get_udp_raw_video_port() const {
    return config_data_.value("/application/udp_raw_video_port"_json_pointer, 12345);
}

int ConfigLoader::get_udp_bounding_box_port() const {
    return config_data_.value("/application/udp_bounding_box_port"_json_pointer, 12346);
}



std::string ConfigLoader::get_mobile_app_ip() const {
    return config_data_.value("/application/mobile_app_ip"_json_pointer, "127.0.0.1");
}

std::string ConfigLoader::get_ip_address() const {
    return get_mobile_app_ip();
}

std::chrono::seconds ConfigLoader::get_camera_watchdog_timeout() const {
    return std::chrono::seconds(config_data_.value("/application/camera_watchdog_timeout_seconds"_json_pointer, 10));
}

int ConfigLoader::get_inference_worker_threads() const {
    return config_data_.value("/application/inference_worker_threads"_json_pointer, 1);
}

int ConfigLoader::get_jpeg_quality() const {
    return config_data_.value("/application/jpeg_quality"_json_pointer, 90);
}

double ConfigLoader::get_camera_fps() const {
    return config_data_.value("/application/camera_fps"_json_pointer, 30.0);
}

// New Network Port Getters
unsigned short ConfigLoader::get_livestream_video_port() const {
    return config_data_.value("/application/network_ports/livestream_video_port/port"_json_pointer, 1001);
}

unsigned short ConfigLoader::get_bounding_box_stream_port() const {
    return config_data_.value("/application/network_ports/bounding_box_stream_port/port"_json_pointer, 1002);
}

unsigned short ConfigLoader::get_reticle_coordinate_port() const {
    return config_data_.value("/application/network_ports/reticle_coordinate_port/port"_json_pointer, 1003);
}

unsigned short ConfigLoader::get_status_telemetry_port() const {
    return config_data_.value("/application/network_ports/status_telemetry_port/port"_json_pointer, 1004);
}

unsigned short ConfigLoader::get_phone_orientation_yaw_port() const {
    return config_data_.value("/application/network_ports/phone_orientation_yaw_port/port"_json_pointer, 2001);
}

unsigned short ConfigLoader::get_phone_orientation_pitch_port() const {
    return config_data_.value("/application/network_ports/phone_orientation_pitch_port/port"_json_pointer, 2002);
}

unsigned short ConfigLoader::get_phone_orientation_roll_port() const {
    return config_data_.value("/application/network_ports/phone_orientation_roll_port/port"_json_pointer, 2003);
}