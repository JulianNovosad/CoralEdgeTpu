#include "orientation_sensor.h"
#include "util_logging.h"

OrientationSensor::OrientationSensor() {
    LOG_INFO("OrientationSensor created (mock data provider).");
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
