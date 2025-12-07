#include "imu_sensor.h"
#include "util_logging.h"

IMUSensor::IMUSensor() {
    LOG_INFO("IMUSensor created (mock data provider).");
    // Initialize mock data
    latest_imu_data_.accel_x = 0.1f; latest_imu_data_.accel_y = 0.05f; latest_imu_data_.accel_z = 9.81f;
    latest_imu_data_.gyro_x = 0.01f; latest_imu_data_.gyro_y = 0.02f; latest_imu_data_.gyro_z = 0.005f;
    latest_imu_data_.mag_x = 25.0f; latest_imu_data_.mag_y = 30.0f; latest_imu_data_.mag_z = -10.0f;
    latest_imu_data_.timestamp = std::chrono::high_resolution_clock::now();
}

IMUSensor::~IMUSensor() {
    stop();
    LOG_INFO("IMUSensor destroyed.");
}

bool IMUSensor::start() {
    if (running_.exchange(true)) {
        LOG_ERROR("IMUSensor is already running.");
        return false;
    }
    worker_thread_ = std::thread(&IMUSensor::worker_thread_func, this);
    LOG_INFO("IMUSensor started.");
    return true;
}

void IMUSensor::stop() {
    if (running_.exchange(false)) {
        LOG_INFO("Stopping IMUSensor...");
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        LOG_INFO("IMUSensor stopped.");
    }
}

void IMUSensor::worker_thread_func() {
    // In a real implementation, this thread would read from the IMU hardware.
    // For now, it just periodically updates the timestamp of the mock data.
    LOG_INFO("IMUSensor worker thread started (mocking data).");
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate sensor read interval
        {
            std::lock_guard<std::mutex> lock(imu_data_mutex_);
            latest_imu_data_.timestamp = std::chrono::high_resolution_clock::now();
            // In a real scenario, actual sensor values would be updated here.
            // For now, only the timestamp changes.
        }
    }
    LOG_INFO("IMUSensor worker thread stopped.");
}

IMUData IMUSensor::get_latest_imu_data() const {
    std::lock_guard<std::mutex> lock(imu_data_mutex_);
    return latest_imu_data_;
}
