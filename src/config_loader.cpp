#include "config_loader.h"
#include "util_logging.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

bool ConfigLoader::load(const std::string& config_file_path) {
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open config file: " << config_file_path << std::endl;
        return false;
    }

    try {
        file >> config_data_;
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "ERROR: Failed to parse config file: " << e.what() << std::endl;
        return false;
    }
    std::cerr << "INFO: Configuration loaded successfully from " << config_file_path << std::endl;
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

std::string ConfigLoader::get_telemetry_protocol() const {
    return config_data_.value("/application/telemetry/protocol"_json_pointer, "HTTP_WEBSOCKET");
}

std::string ConfigLoader::get_telemetry_pub_address() const {
    return config_data_.value("/application/telemetry/pub_address"_json_pointer, "tcp://*:6000");
}

// --- Ballistics Getters ---
float ConfigLoader::get_muzzle_velocity_mps() const {
    return config_data_.value("/application/ballistics/muzzle_velocity_mps"_json_pointer, 850.0f);
}

float ConfigLoader::get_bullet_mass_kg() const {
    return config_data_.value("/application/ballistics/bullet_mass_kg"_json_pointer, 0.008f);
}

float ConfigLoader::get_ballistic_coefficient_si() const {
    return config_data_.value("/application/ballistics/ballistic_coefficient_si"_json_pointer, 0.25f);
}

float ConfigLoader::get_sight_height_m() const {
    return config_data_.value("/application/ballistics/sight_height_m"_json_pointer, 0.05f);
}

float ConfigLoader::get_zero_distance_m() const {
    return config_data_.value("/application/ballistics/zero_distance_m"_json_pointer, 100.0f);
}

float ConfigLoader::get_air_pressure_pa() const {
    return config_data_.value("/application/ballistics/air_pressure_pa"_json_pointer, 101325.0f);
}

float ConfigLoader::get_temperature_c() const {
    return config_data_.value("/application/ballistics/temperature_c"_json_pointer, 15.0f);
}

std::string ConfigLoader::get_drag_model() const {
    return config_data_.value("/application/ballistics/drag_model"_json_pointer, "G1");
}

float ConfigLoader::get_drag_coefficient_reference() const {
    return config_data_.value("/application/ballistics/drag_coefficient_reference"_json_pointer, 0.25f);
}

// --- Tracking ---
int ConfigLoader::get_max_active_tracks() const {
    return config_data_.value("/application/tracking/max_active_tracks"_json_pointer, 100);
}

float ConfigLoader::get_track_iou_threshold() const {
    return config_data_.value("/application/tracking/track_iou_threshold"_json_pointer, 0.3f);
}

int ConfigLoader::get_track_missed_frames_threshold() const {
    return config_data_.value("/application/tracking/track_missed_frames_threshold"_json_pointer, 5);
}

float ConfigLoader::get_min_track_confidence() const {
    return config_data_.value("/application/tracking/min_track_confidence"_json_pointer, 0.6f);
}

// --- Safety Thresholds ---
float ConfigLoader::get_min_confidence_threshold() const {
    return config_data_.value("/application/safety/min_confidence_threshold"_json_pointer, 0.1f);
}

float ConfigLoader::get_max_position_variance() const {
    return config_data_.value("/application/safety/max_position_variance"_json_pointer, 0.75f);
}

float ConfigLoader::get_servo_activate_confidence() const {
    return config_data_.value("/application/safety/servo_activate_confidence"_json_pointer, 0.9f);
}

float ConfigLoader::get_confidence_decay_factor() const {
    return config_data_.value("/application/safety/confidence_decay_factor"_json_pointer, 0.1f);
}

float ConfigLoader::get_distance_confidence_factor() const {
    return config_data_.value("/application/safety/distance_confidence_factor"_json_pointer, 0.1f);
}

float ConfigLoader::get_max_angular_error_degrees() const {
    return config_data_.value("/application/safety/max_angular_error_degrees"_json_pointer, 1.0f);
}


// --- Network Port Getters (Corrected) ---
unsigned short ConfigLoader::get_phone_orientation_yaw_port() const {
    return config_data_.value("/application/network_ports/2001/port"_json_pointer, 2001);
}

unsigned short ConfigLoader::get_phone_orientation_pitch_port() const {
    return config_data_.value("/application/network_ports/2002/port"_json_pointer, 2002);
}

unsigned short ConfigLoader::get_phone_orientation_roll_port() const {
    return config_data_.value("/application/network_ports/2003/port"_json_pointer, 2003);
}

unsigned short ConfigLoader::get_video_stream_port() const {
    // Check if the port is defined in the network_ports section
    if (config_data_.contains("/application/network_ports/1001"_json_pointer)) {
        const auto& port_config = config_data_["/application/network_ports/1001"_json_pointer];
        if (port_config.contains("port")) {
            return port_config["port"].get<unsigned short>();
        }
    }
    // Fallback to default value
    return 1001;
}

// RTSP-specific configuration getters
int ConfigLoader::get_rtsp_port() const {
    return config_data_.value("/application/video_stream/rtsp_port"_json_pointer, 8554);
}

std::string ConfigLoader::get_rtsp_mount_point() const {
    return config_data_.value("/application/video_stream/rtsp_mount_point"_json_pointer, "/live");
}

std::string ConfigLoader::get_rtsp_username() const {
    return config_data_.value("/application/video_stream/rtsp_username"_json_pointer, "");
}

std::string ConfigLoader::get_rtsp_password() const {
    return config_data_.value("/application/video_stream/rtsp_password"_json_pointer, "");
}

// --- TPU Stream Configuration Getters ---
unsigned int ConfigLoader::get_tpu_stream_width() const {
    return config_data_.value("/application/tpu_stream/width"_json_pointer, 300);
}

unsigned int ConfigLoader::get_tpu_stream_height() const {
    return config_data_.value("/application/tpu_stream/height"_json_pointer, 300);
}

unsigned int ConfigLoader::get_tpu_stream_fps() const {
    return config_data_.value("/application/tpu_stream/fps"_json_pointer, 30); // Default to 30 FPS
}

unsigned int ConfigLoader::get_tpu_target_width() const {
    return config_data_.value("/application/tpu_target/width"_json_pointer, 300);
}

unsigned int ConfigLoader::get_tpu_target_height() const {
    return config_data_.value("/application/tpu_target/height"_json_pointer, 300);
}

libcamera::PixelFormat ConfigLoader::get_tpu_stream_pixel_format() const {
    std::string format_str = "BGR888"; // Default value
    auto pixel_format_path = "/application/tpu_stream/pixel_format"_json_pointer;

    if (config_data_.contains(pixel_format_path)) {
        format_str = config_data_[pixel_format_path].get<std::string>();
        APP_LOG_INFO("ConfigLoader: Found TPU stream pixel format in config: '" + format_str + "'");
    } else {
        APP_LOG_WARNING("ConfigLoader: TPU stream pixel format not found in config. Using default: BGR888");
    }

    if (format_str == "BGR888") return libcamera::formats::BGR888;
    if (format_str == "RGB888") return libcamera::formats::RGB888;
    if (format_str == "BGRA8888") return libcamera::formats::BGRA8888;
    if (format_str == "RGBA8888") return libcamera::formats::RGBA8888;
    if (format_str == "YUYV") return libcamera::formats::YUYV;
    // Add other formats as needed
    {
        std::stringstream ss;
        ss << "Configured TPU stream pixel format '" << format_str << "' not recognized by logic. Defaulting to BGR888.";
        APP_LOG_WARNING(ss.str());
    }
    return libcamera::formats::BGR888; // Default
}


const nlohmann::json& ConfigLoader::get_json_config() const {
    return config_data_;
}