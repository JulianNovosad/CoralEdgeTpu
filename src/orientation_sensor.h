#ifndef ORIENTATION_SENSOR_H
#define ORIENTATION_SENSOR_H

#include "pipeline_structs.h" // For OrientationData

#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory> // For std::shared_ptr

/**
 * @brief Interface for an Orientation sensor module.
 *
 * This class provides methods to start/stop the orientation sensor reading, and to retrieve
 * the latest orientation data. For now, it will provide mock data.
 */
class OrientationSensor {
public:
    OrientationSensor(unsigned short yaw_port, unsigned short pitch_port, unsigned short roll_port);
    ~OrientationSensor();

    bool start();
    void stop();
    bool is_running() const { return running_; }

    /**
     * @brief Retrieves the latest orientation data.
     * @return An OrientationData struct containing the latest sensor readings.
     */
    OrientationData get_latest_orientation_data() const;

private:
    void worker_thread_func();

    unsigned short yaw_port_;
    unsigned short pitch_port_;
    unsigned short roll_port_;

    std::atomic<bool> running_ = false;
    std::thread worker_thread_;
    mutable std::mutex orientation_data_mutex_; // Mutable to allow const get_latest_orientation_data to lock
    OrientationData latest_orientation_data_; // Store the latest orientation data
};

#endif // ORIENTATION_SENSOR_H
