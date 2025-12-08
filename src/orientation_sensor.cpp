#include "orientation_sensor.h"
#include "util_logging.h"

OrientationSensor::OrientationSensor(unsigned short yaw_port, unsigned short pitch_port, unsigned short roll_port)
    : yaw_port_(yaw_port), pitch_port_(pitch_port), roll_port_(roll_port) {
    LOG_INFO("OrientationSensor created (mock data provider). Yaw Port: " + std::to_string(yaw_port) +
               ", Pitch Port: " + std::to_string(pitch_port) + ", Roll Port: " + std::to_string(roll_port));
    // Initialize mock data
    latest_orientation_data_.yaw = 0.0f;
    latest_orientation_data_.pitch = 0.0f;
    latest_orientation_data_.roll = 0.0f;
    latest_orientation_data_.timestamp = std::chrono::high_resolution_clock::now();
}

OrientationSensor::~OrientationSensor() {
    stop();
    LOG_INFO("OrientationSensor destroyed.");
}

bool OrientationSensor::start() {
    if (running_.exchange(true)) {
        LOG_ERROR("OrientationSensor is already running.");
        return false;
    }
    // TODO: Initialize UDP sockets for yaw_port_, pitch_port_, roll_port_ here.
    // For example:
    // yaw_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    // bind(yaw_socket_, ...);
    // Add error handling for socket creation and binding.

    worker_thread_ = std::thread(&OrientationSensor::worker_thread_func, this);
    LOG_INFO("OrientationSensor started.");
    return true;
}

void OrientationSensor::stop() {
    if (running_.exchange(false)) {
        LOG_INFO("Stopping OrientationSensor...");
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        LOG_INFO("OrientationSensor stopped.");
    }
}

void OrientationSensor::worker_thread_func() {
    // In a real implementation, this thread would read from the IMU hardware.
    // For now, it just periodically updates the timestamp of the mock data.
    LOG_INFO("OrientationSensor worker thread started (mocking data).");
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate sensor read interval
        {
            std::lock_guard<std::mutex> lock(orientation_data_mutex_);
            latest_orientation_data_.timestamp = std::chrono::high_resolution_clock::now();
            // In a real scenario, actual sensor values would be updated here.
            // For now, only the timestamp changes.
        }
    }
    LOG_INFO("OrientationSensor worker thread stopped.");
}

OrientationData OrientationSensor::get_latest_orientation_data() const {
    std::lock_guard<std::mutex> lock(orientation_data_mutex_); // Updated mutex name
    return latest_orientation_data_; // Updated member variable name
}
