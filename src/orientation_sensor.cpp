// Verified headers: [orientation_sensor.h, util_logging.h, json.hpp, future]
// Verification timestamp: 2026-01-06 17:08:04
#include "orientation_sensor.h"
#include "util_logging.h"
#include "json.hpp"
#include <future>

OrientationSensor::OrientationSensor(int yaw_port, int pitch_port, int roll_port)
    : yaw_port_(yaw_port), pitch_port_(pitch_port), roll_port_(roll_port), 
      source_port_(0), running_(false) {
    zmq_context_ = std::make_unique<zmq::context_t>(1);
}

OrientationSensor::~OrientationSensor() {
    stop();
}

bool OrientationSensor::start() {
    if (running_.load()) return true;
    
    running_.store(true);
    receiver_thread_ = std::thread(&OrientationSensor::receiver_thread_func, this);
    
    APP_LOG_INFO("OrientationSensor started.");
    return true;
}

void OrientationSensor::stop() {
    if (!running_.load()) return;
    
    running_.store(false);
    if (zmq_socket_) {
        zmq_socket_->close();
    }
    
    if (receiver_thread_.joinable()) {
        auto shared_promise = std::make_shared<std::promise<void>>();
        std::future<void> future = shared_promise->get_future();
        std::thread joiner_thread([this, shared_promise]() {
            try {
                if (receiver_thread_.joinable()) {
                    receiver_thread_.join();
                }
                shared_promise->set_value();
            } catch (...) {}
        });
        if (future.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
            APP_LOG_WARNING("[SHUTDOWN] OrientationSensor receiver thread did not join within 3s, detaching.");
            if (receiver_thread_.joinable()) receiver_thread_.detach();
            joiner_thread.detach();
        } else {
            if (joiner_thread.joinable()) joiner_thread.join();
        }
    }
    
    APP_LOG_INFO("OrientationSensor stopped.");
}

OrientationData OrientationSensor::get_latest_orientation_data() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return latest_data_;
}

void OrientationSensor::set_source(const std::string& ip, int port) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    source_ip_ = ip;
    source_port_ = port;
    
    // If running, we might need to reconnect. 
    // For now, let the thread handle reconnection if socket is reset or on next check.
    APP_LOG_INFO("OrientationSensor source set to " + ip + ":" + std::to_string(port));
}

void OrientationSensor::receiver_thread_func() {
    while (running_.load()) {
        std::string current_ip;
        int current_port = 0;
        
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            current_ip = source_ip_;
            current_port = source_port_;
        }
        
        if (current_ip.empty() || current_port == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        
        try {
            zmq_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, zmq::socket_type::sub);
            std::string endpoint = "tcp://" + current_ip + ":" + std::to_string(current_port);
            zmq_socket_->connect(endpoint);
            zmq_socket_->set(zmq::sockopt::subscribe, "");
            zmq_socket_->set(zmq::sockopt::rcvtimeo, 1000); // 1s timeout
            
            APP_LOG_INFO("OrientationSensor connected to " + endpoint);
            
            while (running_.load()) {
                // Check if source changed
                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    if (source_ip_ != current_ip || source_port_ != current_port) {
                        break; // Reconnect
                    }
                }
                
                zmq::message_t msg;
                auto res = zmq_socket_->recv(msg, zmq::recv_flags::none);
                
                if (res) {
                    try {
                        auto j = nlohmann::json::parse(std::string(static_cast<char*>(msg.data()), msg.size()));
                        
                        OrientationData data;
                        data.yaw = j.at("yaw").get<float>();
                        data.pitch = j.at("pitch").get<float>();
                        data.roll = j.at("roll").get<float>();
                        data.timestamp = std::chrono::steady_clock::now();
                        
                        {
                            std::lock_guard<std::mutex> lock(data_mutex_);
                            latest_data_ = data;
                        }
                    } catch (const std::exception& e) {
                        APP_LOG_ERROR("OrientationSensor: Failed to parse JSON: " + std::string(e.what()));
                    }
                }
            }
            
            zmq_socket_->close();
            zmq_socket_.reset();
            
        } catch (const std::exception& e) {
            APP_LOG_ERROR("OrientationSensor receiver error: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
}
