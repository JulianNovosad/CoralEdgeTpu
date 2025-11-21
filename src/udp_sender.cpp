#include "udp_sender.h"
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>

UdpSender::UdpSender(const std::string& target_ip, int target_port, UdpQueue& input_queue)
    : target_ip_(target_ip), target_port_(target_port), input_queue_(input_queue) {

    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0) {
        std::cerr << "Failed to create UDP socket." << std::endl;
        // Handle error, e.g., throw exception
    }

    server_addr_.sin_family = AF_INET;
    server_addr_.sin_port = htons(target_port_);
    if (inet_pton(AF_INET, target_ip_.c_str(), &server_addr_.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported: " << target_ip_ << std::endl;
        close(sockfd_);
        sockfd_ = -1;
        // Handle error
    }
}

UdpSender::~UdpSender() {
    stop();
    if (sockfd_ != -1) {
        close(sockfd_);
    }
}

bool UdpSender::start() {
    if (running_) {
        std::cerr << "UdpSender is already running." << std::endl;
        return false;
    }
    if (sockfd_ == -1) {
        std::cerr << "UDP socket not initialized, cannot start sender." << std::endl;
        return false;
    }

    running_ = true;
    input_queue_.set_running(true);
    sender_thread_ = std::thread(&UdpSender::sender_thread_func, this);
    std::cout << "UDP sender started for target: " << target_ip_ << ":" << target_port_ << std::endl;
    return true;
}

void UdpSender::stop() {
    if (!running_) {
        return;
    }
    running_ = false;
    input_queue_.set_running(false);
    if (sender_thread_.joinable()) {
        sender_thread_.join();
    }
    std::cout << "UDP sender stopped." << std::endl;
}

void UdpSender::sender_thread_func() {
    std::vector<DetectionResult> results;
    while (running_) {
        if (input_queue_.pop(results)) { // Blocking call
            std::string json_data = detection_to_json(results);
            if (!json_data.empty()) {
                ssize_t sent_bytes = sendto(sockfd_, json_data.c_str(), json_data.length(), 0,
                                            (const sockaddr*)&server_addr_, sizeof(server_addr_));
                if (sent_bytes < 0) {
                    std::cerr << "Failed to send UDP packet to " << target_ip_ << ":" << target_port_ << std::endl;
                } else {
                    // std::cout << "Sent " << sent_bytes << " bytes via UDP." << std::endl;
                }
            }
        }
    }
}

std::string UdpSender::detection_to_json(const std::vector<DetectionResult>& results) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& res = results[i];
        oss << "{";
        oss << "{\"class_id\":" << res.class_id << ",";
        oss << "\"score\":" << res.score << ",";
        oss << "\"xmin\":" << res.xmin << ",";
        oss << "\"ymin\":" << res.ymin << ",";
        oss << "\"xmax\":" << res.xmax << ",";
        oss << "\"ymax\":" << res.ymax << ",";
        oss << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(res.timestamp.time_since_epoch()).count();
        oss << "}";
        if (i < results.size() - 1) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}
