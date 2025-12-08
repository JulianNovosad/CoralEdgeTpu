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

std::string ConfigLoader::get_listen_address() const {
    return config_data_.value("/application/listen_address"_json_pointer, "0.0.0.0");
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

float ConfigLoader::get_detection_score_threshold() const {
    return config_data_.value("/application/detection_score_threshold"_json_pointer, 0.5f);
}

std::string ConfigLoader::get_log_path() const {
    return config_data_.value("/application/log_path"_json_pointer, "/home/pi/CoralEdgeTpu/logs");
}

std::string ConfigLoader::get_video_stream_protocol() const {
    return config_data_.value("/application/video_stream/protocol"_json_pointer, "HTTP_WEBSOCKET");
}

std::string ConfigLoader::get_video_stream_address() const {
    return config_data_.value("/application/video_stream/address"_json_pointer, "0.0.0.0");
}

unsigned short ConfigLoader::get_video_stream_port() const {
    return config_data_.value("/application/video_stream/port"_json_pointer, 5000);
}

std::string ConfigLoader::get_telemetry_protocol() const {
    return config_data_.value("/application/telemetry/protocol"_json_pointer, "HTTP_WEBSOCKET");
}

std::string ConfigLoader::get_telemetry_pub_address() const {
    return config_data_.value("/application/telemetry/pub_address"_json_pointer, "tcp://*:6000");
}

// New Network Port Getters
unsigned short ConfigLoader::get_phone_orientation_yaw_port() const {
    return config_data_.value("/application/network_ports/phone_orientation_yaw"_json_pointer, 2001);
}

unsigned short ConfigLoader::get_phone_orientation_pitch_port() const {
    return config_data_.value("/application/network_ports/phone_orientation_pitch"_json_pointer, 2002);
}

unsigned short ConfigLoader::get_phone_orientation_roll_port() const {
    return config_data_.value("/application/network_ports/phone_orientation_roll"_json_pointer, 2003);
}