/**
 * @file udp_sender.cpp
 * @brief Implements the UdpSender class for transmitting object detection results via UDP.
 *
 * This module provides the concrete implementation for establishing a UDP socket,
 * serializing detection results into JSON format, and sending them to a
 * configured network endpoint. It operates in a dedicated thread, consuming
 * data from a thread-safe queue.
 */

#include "udp_sender.h"
#include "util_logging.h" // For LOG_INFO, LOG_ERROR
#include <iostream>       // For std::cerr (temporarily before full LOG_ERROR conversion)
#include <sys/socket.h>   // For socket, sendto
#include <arpa/inet.h>    // For inet_pton
#include <unistd.h>       // For close
#include <sstream>        // For std::ostringstream

/**
 * @brief Constructor for UdpSender.
 *
 * Initializes the UDP socket and configures the target server address structure.
 *
 * @param target_ip The IP address of the target UDP receiver.
 * @param target_port The UDP port number of the target receiver.
 * @param input_queue Reference to the thread-safe UdpQueue providing detection results.
 */
UdpSender::UdpSender(const std::string& target_ip, int target_port, UdpQueue& input_queue)
    : target_ip_(target_ip), target_port_(target_port), input_queue_(input_queue) {

    // Create UDP socket.
    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0) {
        LOG_ERROR("Failed to create UDP socket.");
        // In a real application, proper error handling (e.g., throwing an exception)
        // would be necessary to prevent module startup if socket creation fails.
    }

    // Configure server address structure.
    server_addr_.sin_family = AF_INET; // IPv4
    server_addr_.sin_port = htons(target_port_); // Convert port to network byte order
    // Convert IP address from string to binary form.
    if (inet_pton(AF_INET, target_ip_.c_str(), &server_addr_.sin_addr) <= 0) {
        LOG_ERROR("Invalid address or address not supported: " + target_ip_);
        if (sockfd_ != -1) {
            close(sockfd_);
            sockfd_ = -1; // Mark socket as invalid
        }
    }
}

/**
 * @brief Destructor for UdpSender.
 *
 * Ensures that the sender thread is gracefully stopped and closes the UDP socket.
 */
UdpSender::~UdpSender() {
    stop(); // Ensure sender thread is stopped
    if (sockfd_ != -1) {
        close(sockfd_); // Close the socket
    }
}

/**
 * @brief Starts the UDP sender.
 *
 * Launches the dedicated sender thread. Verifies that the UDP socket is valid
 * before attempting to start.
 *
 * @return True if the sender started successfully, false otherwise.
 */
bool UdpSender::start() {
    if (running_) {
        LOG_ERROR("UdpSender is already running.");
        return false;
    }
    if (sockfd_ == -1) {
        LOG_ERROR("UDP socket not initialized or invalid, cannot start sender.");
        return false;
    }

    running_ = true;
    input_queue_.set_running(true); // Signal input queue to be active
    sender_thread_ = std::thread(&UdpSender::sender_thread_func, this);
    LOG_INFO("UDP sender started for target: " + target_ip_ + ":" + std::to_string(target_port_));
    return true;
}

/**
 * @brief Stops the UDP sender.
 *
 * Signals the sender thread to terminate and waits for it to finish. Also
 * signals the input queue to stop.
 */
void UdpSender::stop() {
    if (!running_.exchange(false)) { // Atomically set to false and check previous value
        return; // Already stopped
    }
    LOG_INFO("Stopping UDP sender...");
    input_queue_.set_running(false); // Signal input queue to stop
    if (sender_thread_.joinable()) {
        sender_thread_.join(); // Wait for the sender thread to complete
    }
    LOG_INFO("UDP sender stopped.");
}

/**
 * @brief The main loop for the UDP sender thread.
 *
 * Continuously retrieves vectors of `DetectionResult` from the input queue,
 * converts them to a JSON string, and transmits them via the UDP socket.
 * The loop runs as long as the `running_` flag is true.
 */
void UdpSender::sender_thread_func() {
    std::vector<DetectionResult> results;
    while (running_) {
        // Pop detection results from the queue. This call will block until data
        // is available or the queue is signaled to stop.
        if (input_queue_.pop(results)) {
            // Convert the detection results to a JSON string.
            std::string json_data = detection_to_json(results);
            if (!json_data.empty()) {
                // Send the JSON data via UDP.
                ssize_t sent_bytes = sendto(sockfd_, json_data.c_str(), json_data.length(), 0,
                                            (const sockaddr*)&server_addr_, sizeof(server_addr_));
                if (sent_bytes < 0) {
                    LOG_ERROR("Failed to send UDP packet to " + target_ip_ + ":" + std::to_string(target_port_) + ": " + strerror(errno));
                } else {
                    // For debugging: LOG_INFO("Sent " + std::to_string(sent_bytes) + " bytes via UDP.");
                }
            }
        }
    }
}

/**
 * @brief Converts a vector of `DetectionResult` objects into a JSON string.
 *
 * The output JSON is an array of objects, where each object represents
 * a single detection including its class ID, score, bounding box coordinates,
 * and timestamp.
 *
 * @param results A constant reference to a vector of `DetectionResult` objects.
 * @return A JSON formatted string representing the detection results.
 */
std::string UdpSender::detection_to_json(const std::vector<DetectionResult>& results) {
    std::ostringstream oss;
    oss << "["; // Start of JSON array
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& res = results[i];
        oss << "{"; // Start of JSON object for a single detection
        oss << "\"class_id\":" << res.class_id << ",";
        oss << "\"score\":" << res.score << ",";
        oss << "\"xmin\":" << res.xmin << ",";
        oss << "\"ymin\":" << res.ymin << ",";
        oss << "\"xmax\":" << res.xmax << ",";
        oss << "\"ymax\":" << res.ymax << ",";
        // Convert high_resolution_clock time_point to milliseconds since epoch for JSON
        oss << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(res.timestamp.time_since_epoch()).count();
        oss << "}"; // End of JSON object for a single detection
        if (i < results.size() - 1) {
            oss << ","; // Add comma between objects
        }
    }
    oss << "]"; // End of JSON array
    return oss.str();
}