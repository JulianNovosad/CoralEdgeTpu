/**
 * @file udp_video_sender.cpp
 * @brief Implements the UdpVideoSender class for transmitting MJPEG video frames via UDP.
 *
 * This module provides the concrete implementation for establishing a UDP socket,
 * sending MJPEG frames to a configured network endpoint. It operates in a
 * dedicated thread, consuming data from a thread-safe queue. It handles
 * fragmentation of large MJPEG frames to fit within typical UDP packet size limits.
 */

#include "udp_video_sender.h"
#include "util_logging.h" // For LOG_INFO, LOG_ERROR
#include <iostream>       // For std::cerr (temporarily before full LOG_ERROR conversion)
#include <sys/socket.h>   // For socket, sendto
#include <arpa/inet.h>    // For inet_pton
#include <unistd.h>       // For close
#include <sstream>        // For std::ostringstream
#include <cstring>        // For strerror

/**
 * @brief Constructor for UdpVideoSender.
 *
 * Initializes the UDP socket and configures the target server address structure.
 *
 * @param target_ip The IP address of the target UDP receiver (mobile app).
 * @param target_port The UDP port number of the target receiver (e.g., 50000).
 * @param input_queue Reference to the thread-safe MjpegQueue providing MJPEG frames.
 */
UdpVideoSender::UdpVideoSender(const std::string& target_ip, int target_port, MjpegQueue& input_queue)
    : target_ip_(target_ip), target_port_(target_port), input_queue_(input_queue) {

    // Create UDP socket.
    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0) {
        LOG_ERROR("UdpVideoSender: Failed to create UDP socket.");
        // In a real application, proper error handling (e.g., throwing an exception)
        // would be necessary to prevent module startup if socket creation fails.
    }

    // Configure server address structure.
    server_addr_.sin_family = AF_INET; // IPv4
    server_addr_.sin_port = htons(target_port_); // Convert port to network byte order
    // Convert IP address from string to binary form.
    if (inet_pton(AF_INET, target_ip_.c_str(), &server_addr_.sin_addr) <= 0) {
        LOG_ERROR("UdpVideoSender: Invalid address or address not supported: " + target_ip_);
        if (sockfd_ != -1) {
            close(sockfd_);
            sockfd_ = -1; // Mark socket as invalid
        }
    }
}

/**
 * @brief Destructor for UdpVideoSender.
 *
 * Ensures that the sender thread is gracefully stopped and closes the UDP socket.
 */
UdpVideoSender::~UdpVideoSender() {
    stop(); // Ensure sender thread is stopped
    if (sockfd_ != -1) {
        close(sockfd_); // Close the socket
    }
}

/**
 * @brief Starts the UDP video sender.
 *
 * Launches the dedicated sender thread. Verifies that the UDP socket is valid
 * before attempting to start.
 *
 * @return True if the sender started successfully, false otherwise.
 */
bool UdpVideoSender::start() {
    if (running_) {
        LOG_ERROR("UdpVideoSender is already running.");
        return false;
    }
    if (sockfd_ == -1) {
        LOG_ERROR("UdpVideoSender: UDP socket not initialized or invalid, cannot start sender.");
        return false;
    }

    running_ = true;
    input_queue_.set_running(true); // Signal input queue to be active
    sender_thread_ = std::thread(&UdpVideoSender::sender_thread_func, this);
    LOG_INFO("UdpVideoSender started for target: " + target_ip_ + ":" + std::to_string(target_port_));
    return true;
}

/**
 * @brief Stops the UDP video sender.
 *
 * Signals the sender thread to terminate and waits for it to finish. Also
 * signals the input queue to stop.
 */
void UdpVideoSender::stop() {
    if (!running_.exchange(false)) { // Atomically set to false and check previous value
        return; // Already stopped
    }
    LOG_INFO("Stopping UdpVideoSender...");
    input_queue_.set_running(false); // Signal input queue to stop
    if (sender_thread_.joinable()) {
        sender_thread_.join(); // Wait for the sender thread to complete
    }
    LOG_INFO("UdpVideoSender stopped.");
}

/**
 * @brief The main loop for the UDP video sender thread.
 *
 * Continuously retrieves `ImageFrame` objects (containing MJPEG data) from
 * the input queue and sends them to the configured UDP target.
 * The loop runs as long as the `running_` flag is true.
 */
void UdpVideoSender::sender_thread_func() {
    ImageFrame frame;
    while (running_) {
        // Pop MJPEG frames from the queue. This call will block until data
        // is available or the queue is signaled to stop.
        if (input_queue_.pop(frame)) {
            send_mjpeg_frame(frame);
        }
    }
}

/**
 * @brief Sends a single MJPEG frame, potentially fragmented, over UDP.
 *
 * This function fragments the MJPEG data into chunks if it exceeds
 * `MAX_UDP_PAYLOAD_SIZE` to ensure it fits within typical UDP packet limits.
 * Each fragment is sent with a simple header for reassembly.
 *
 * @param frame The ImageFrame containing the MJPEG data.
 */
void UdpVideoSender::send_mjpeg_frame(const ImageFrame& frame) {
    const uint8_t* data_ptr = frame.jpeg_data.data();
    size_t total_size = frame.jpeg_data.size();
    size_t bytes_sent = 0;
    
    // Simple fragmentation header: [sequence_number (2 bytes), total_fragments (2 bytes), fragment_index (2 bytes)]
    // This is a very basic header for demonstration. A robust solution would need a more complex protocol.
    const size_t FRAG_HEADER_SIZE = 6;
    static uint16_t sequence_number = 0; // Incremented for each full frame.

    if (total_size == 0) {
        LOG_WARNING("UdpVideoSender: Attempted to send empty MJPEG frame.");
        return;
    }

    // Determine number of fragments needed
    size_t payload_per_fragment = MAX_UDP_PAYLOAD_SIZE - FRAG_HEADER_SIZE;
    size_t num_fragments = (total_size + payload_per_fragment - 1) / payload_per_fragment;

    if (num_fragments > 0xFFFF) { // Ensure total_fragments fits in 2 bytes
        LOG_ERROR("UdpVideoSender: MJPEG frame too large to fragment. Skipping frame.");
        return;
    }

    sequence_number++; // Increment sequence for a new frame.

    for (size_t i = 0; i < num_fragments; ++i) {
        size_t offset = i * payload_per_fragment;
        size_t current_fragment_size = std::min(payload_per_fragment, total_size - offset);

        // Prepare the packet buffer with header + payload
        std::vector<uint8_t> packet_buffer(FRAG_HEADER_SIZE + current_fragment_size);
        
        // Write fragmentation header
        packet_buffer[0] = (sequence_number >> 8) & 0xFF; // Sequence high byte
        packet_buffer[1] = sequence_number & 0xFF;        // Sequence low byte
        packet_buffer[2] = (num_fragments >> 8) & 0xFF;    // Total fragments high byte
        packet_buffer[3] = num_fragments & 0xFF;          // Total fragments low byte
        packet_buffer[4] = (i >> 8) & 0xFF;                // Fragment index high byte
        packet_buffer[5] = i & 0xFF;                      // Fragment index low byte

        // Copy payload data
        std::memcpy(packet_buffer.data() + FRAG_HEADER_SIZE, data_ptr + offset, current_fragment_size);

        // Send the UDP packet
        ssize_t sent_bytes_current_frag = sendto(sockfd_, (const char*)packet_buffer.data(), packet_buffer.size(), 0,
                                                 (const sockaddr*)&server_addr_, sizeof(server_addr_));
        if (sent_bytes_current_frag < 0) {
            LOG_ERROR("UdpVideoSender: Failed to send UDP fragment " + std::to_string(i) + " for frame " + std::to_string(sequence_number) + ": " + strerror(errno));
            // Break on error, as subsequent fragments for this frame will also likely fail.
            break; 
        }
        bytes_sent += sent_bytes_current_frag;
    }
    // For debugging: LOG_INFO("UdpVideoSender: Sent MJPEG frame (seq " + std::to_string(sequence_number) + ") in " + std::to_string(num_fragments) + " fragments, total " + std::to_string(bytes_sent) + " bytes.");
}