#include "orientation_sensor.h"
#include "util_logging.h"
#include <iostream>
#include <sstream>

OrientationSensor::OrientationSensor(unsigned short yaw_port, unsigned short pitch_port, unsigned short roll_port)
    : yaw_port_(yaw_port), pitch_port_(pitch_port), roll_port_(roll_port) {
    APP_LOG_INFO("OrientationSensor created. Yaw Port: " + std::to_string(yaw_port) +
               ", Pitch Port: " + std::to_string(pitch_port) + ", Roll Port: " + std::to_string(roll_port));
    
    // Initialize ZeroMQ context
    zmq_context_ = std::make_unique<zmq::context_t>(1);
    
    // Initialize orientation data
    latest_orientation_data_.yaw = 0.0f;
    latest_orientation_data_.pitch = 0.0f;
    latest_orientation_data_.roll = 0.0f;
    latest_orientation_data_.timestamp = std::chrono::high_resolution_clock::now();
}

OrientationSensor::~OrientationSensor() {
    stop();
    APP_LOG_INFO("OrientationSensor destroyed.");
}

bool OrientationSensor::start() {
    if (running_.exchange(true)) {
        APP_LOG_ERROR("OrientationSensor is already running.");
        return false;
    }
    
    try {
        // Initialize ZeroMQ subscriber sockets for each orientation component
        yaw_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, zmq::socket_type::sub);
        pitch_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, zmq::socket_type::sub);
        roll_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, zmq::socket_type::sub);
        
        // Connect to the ports
        std::string yaw_address = "tcp://*:" + std::to_string(yaw_port_);
        std::string pitch_address = "tcp://*:" + std::to_string(pitch_port_);
        std::string roll_address = "tcp://*:" + std::to_string(roll_port_);
        
        yaw_socket_->bind(yaw_address);
        pitch_socket_->bind(pitch_address);
        roll_socket_->bind(roll_address);
        
        // Subscribe to all messages (empty subscription means subscribe to all)
        yaw_socket_->set(zmq::sockopt::subscribe, "");
        pitch_socket_->set(zmq::sockopt::subscribe, "");
        roll_socket_->set(zmq::sockopt::subscribe, "");
        
        APP_LOG_INFO("OrientationSensor ZeroMQ sockets bound: Yaw=" + yaw_address + 
                     ", Pitch=" + pitch_address + ", Roll=" + roll_address);
    } catch (const std::exception& e) {
        APP_LOG_ERROR("Failed to initialize ZeroMQ sockets: " + std::string(e.what()));
        running_ = false;
        return false;
    }

    worker_thread_ = std::thread(&OrientationSensor::worker_thread_func, this);
    APP_LOG_INFO("OrientationSensor started.");
    return true;
}

void OrientationSensor::stop() {
    if (running_.exchange(false)) {
        APP_LOG_INFO("Stopping OrientationSensor...");
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        
        // Clean up ZeroMQ sockets
        yaw_socket_.reset();
        pitch_socket_.reset();
        roll_socket_.reset();
        zmq_context_.reset();
        
        APP_LOG_INFO("OrientationSensor stopped.");
    }
}

void OrientationSensor::worker_thread_func() {
    APP_LOG_INFO("OrientationSensor worker thread started.");
    
    // Poll items for checking incoming messages
    zmq::pollitem_t items[] = {
        { static_cast<void*>(*yaw_socket_), 0, ZMQ_POLLIN, 0 },
        { static_cast<void*>(*pitch_socket_), 0, ZMQ_POLLIN, 0 },
        { static_cast<void*>(*roll_socket_), 0, ZMQ_POLLIN, 0 }
    };
    
    while (running_) {
        try {
            // Poll for incoming messages with a timeout
            zmq::poll(items, 3, std::chrono::milliseconds(100));
            
            bool data_updated = false;
            
            // Check for yaw data
            if (items[0].revents & ZMQ_POLLIN) {
                zmq::message_t msg;
                auto result = yaw_socket_->recv(msg, zmq::recv_flags::dontwait);
                if (result) {
                    std::string yaw_str(static_cast<char*>(msg.data()), msg.size());
                    float yaw_val = std::stof(yaw_str);
                    
                    {
                        std::lock_guard<std::mutex> lock(orientation_data_mutex_);
                        latest_orientation_data_.yaw = yaw_val;
                        data_updated = true;
                    }
                }
            }
            
            // Check for pitch data
            if (items[1].revents & ZMQ_POLLIN) {
                zmq::message_t msg;
                auto result = pitch_socket_->recv(msg, zmq::recv_flags::dontwait);
                if (result) {
                    std::string pitch_str(static_cast<char*>(msg.data()), msg.size());
                    float pitch_val = std::stof(pitch_str);
                    
                    {
                        std::lock_guard<std::mutex> lock(orientation_data_mutex_);
                        latest_orientation_data_.pitch = pitch_val;
                        data_updated = true;
                    }
                }
            }
            
            // Check for roll data
            if (items[2].revents & ZMQ_POLLIN) {
                zmq::message_t msg;
                auto result = roll_socket_->recv(msg, zmq::recv_flags::dontwait);
                if (result) {
                    std::string roll_str(static_cast<char*>(msg.data()), msg.size());
                    float roll_val = std::stof(roll_str);
                    
                    {
                        std::lock_guard<std::mutex> lock(orientation_data_mutex_);
                        latest_orientation_data_.roll = roll_val;
                        data_updated = true;
                    }
                }
            }
            
            // Update timestamp if any data was received
            if (data_updated) {
                std::lock_guard<std::mutex> lock(orientation_data_mutex_);
                latest_orientation_data_.timestamp = std::chrono::high_resolution_clock::now();
                APP_LOG_DEBUG("Orientation data updated - Yaw: " + std::to_string(latest_orientation_data_.yaw) +
                              ", Pitch: " + std::to_string(latest_orientation_data_.pitch) +
                              ", Roll: " + std::to_string(latest_orientation_data_.roll));
            }
            
        } catch (const std::exception& e) {
            APP_LOG_ERROR("Error in OrientationSensor worker thread: " + std::string(e.what()));
            // Continue running unless stopped
        }
    }
    
    APP_LOG_INFO("OrientationSensor worker thread stopped.");
}

OrientationData OrientationSensor::get_latest_orientation_data() const {
    std::lock_guard<std::mutex> lock(orientation_data_mutex_); // Updated mutex name
    return latest_orientation_data_; // Updated member variable name
}