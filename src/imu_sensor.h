#ifndef IMU_SENSOR_H
#define IMU_SENSOR_H

#include "pipeline_structs.h" // For IMUData

#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory> // For std::shared_ptr

/**
 * @brief Interface for an IMU sensor module.
 *
 * This class provides methods to start/stop the IMU reading, and to retrieve
 * the latest IMU data. For now, it will provide mock data.
 */
class IMUSensor {
public:
    IMUSensor();
    ~IMUSensor();

    bool start();
    void stop();
    bool is_running() const { return running_; }

    /**
     * @brief Retrieves the latest IMU data.
     * @return An IMUData struct containing the latest sensor readings.
     */
    IMUData get_latest_imu_data() const;

private:
    void worker_thread_func();

    std::atomic<bool> running_ = false;
    std::thread worker_thread_;
    mutable std::mutex imu_data_mutex_; // Mutable to allow const get_latest_imu_data to lock
    IMUData latest_imu_data_; // Store the latest IMU data
};

#endif // IMU_SENSOR_H
