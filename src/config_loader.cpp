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
    return config_data_["application"]["model_path"].get<std::string>();
}

std::string ConfigLoader::get_labels_path() const {
    return config_data_["application"]["labels_path"].get<std::string>();
}

unsigned int ConfigLoader::get_high_res_width() const {
    return config_data_["application"]["high_res_width"].get<unsigned int>();
}

unsigned int ConfigLoader::get_high_res_height() const {
    return config_data_["application"]["high_res_height"].get<unsigned int>();
}

int ConfigLoader::get_udp_raw_video_port() const {
    return config_data_["application"]["udp_raw_video_port"].get<int>();
}

int ConfigLoader::get_udp_bounding_box_port() const {
    return config_data_["application"]["udp_bounding_box_port"].get<int>();
}

int ConfigLoader::get_http_overlaid_video_port() const {
    return config_data_["application"]["http_overlaid_video_port"].get<int>();
}

std::string ConfigLoader::get_mobile_app_ip() const {
    return config_data_["application"]["mobile_app_ip"].get<std::string>();
}

std::chrono::seconds ConfigLoader::get_camera_watchdog_timeout() const {
    return std::chrono::seconds(config_data_["application"]["camera_watchdog_timeout_seconds"].get<int>());
}

int ConfigLoader::get_inference_worker_threads() const {
    return config_data_["application"]["inference_worker_threads"].get<int>();
}

int ConfigLoader::get_jpeg_quality() const {
    return config_data_["application"]["jpeg_quality"].get<int>();
}