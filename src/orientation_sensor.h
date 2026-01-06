// Verified headers: [string, memory, atomic, thread, mutex...]
// Verification timestamp: 2026-01-06 17:08:04
#ifndef ORIENTATION_SENSOR_H
#define ORIENTATION_SENSOR_H

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <zmq.hpp>
#include "pipeline_structs.h"

/**
 * @brief Receives orientation data from an external sensor (e.g., Android app) via ZeroMQ.
 */
class OrientationSensor {
public:
    /**
     * @brief Constructor for OrientationSensor.
     * @param yaw_port Port for yaw data.
     * @param pitch_port Port for pitch data.
     * @param roll_port Port for roll data.
     * @note Legacy constructor, currently uses ports.
     */
    OrientationSensor(int yaw_port, int pitch_port, int roll_port);
    ~OrientationSensor();

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    /**
     * @brief Gets the latest received orientation data.
     * @return The most recent OrientationData.
     */
    OrientationData get_latest_orientation_data();

    /**
     * @brief Sets the source address and port for orientation data.
     * @param ip The source IP address.
     * @param port The source port.
     */
    void set_source(const std::string& ip, int port);

private:
    void receiver_thread_func();

    int yaw_port_;
    int pitch_port_;
    int roll_port_;
    
    std::string source_ip_;
    int source_port_;

    std::atomic<bool> running_;
    std::thread receiver_thread_;
    
    std::mutex data_mutex_;
    OrientationData latest_data_;
    
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::unique_ptr<zmq::socket_t> zmq_socket_;
};

#endif // ORIENTATION_SENSOR_H
