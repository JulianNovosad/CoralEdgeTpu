#include "udp_video_sender.h"
#include "util_logging.h"
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>
#include <cstring>
#include <algorithm> // For std::max, std::min
#include <cmath>     // For std::round

namespace {
    // In-place YUV to BGR conversion
    void convert_yuv_to_bgr(const ImageData& yuv_image, std::vector<uint8_t>& bgr_data) {
        int width = yuv_image.width;
        int height = yuv_image.height;
        int num_pixels = width * height;
        bgr_data.resize(num_pixels * 3);

        const uint8_t* y_plane = yuv_image.data.data();
        const uint8_t* u_plane = y_plane + num_pixels;
        const uint8_t* v_plane = u_plane + (num_pixels / 4);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int y = y_plane[i * width + j];
                int u = u_plane[(i / 2) * (width / 2) + (j / 2)];
                int v = v_plane[(i / 2) * (width / 2) + (j / 2)];

                // YUV to RGB conversion
                int r = y + 1.402 * (v - 128);
                int g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128);
                int b = y + 1.772 * (u - 128);

                // Clamp values to [0, 255]
                r = std::max(0, std::min(255, r));
                g = std::max(0, std::min(255, g));
                b = std::max(0, std::min(255, b));

                int bgr_index = (i * width + j) * 3;
                bgr_data[bgr_index + 0] = static_cast<uint8_t>(b);
                bgr_data[bgr_index + 1] = static_cast<uint8_t>(g);
                bgr_data[bgr_index + 2] = static_cast<uint8_t>(r);
            }
        }
    }
}

UdpVideoSender::UdpVideoSender(const std::string& target_ip, int target_port, ImageQueue& input_queue)
    : target_ip_(target_ip), target_port_(target_port), input_queue_(input_queue) {

    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0) {
        LOG_ERROR("UdpVideoSender: Failed to create UDP socket.");
    }

    server_addr_.sin_family = AF_INET;
    server_addr_.sin_port = htons(target_port_);
    if (inet_pton(AF_INET, target_ip_.c_str(), &server_addr_.sin_addr) <= 0) {
        LOG_ERROR("UdpVideoSender: Invalid address or address not supported: " + target_ip_);
        if (sockfd_ != -1) {
            close(sockfd_);
            sockfd_ = -1;
        }
    }
}

UdpVideoSender::~UdpVideoSender() {
    stop();
    if (sockfd_ != -1) {
        close(sockfd_);
    }
}

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
    input_queue_.set_running(true);
    sender_thread_ = std::thread(&UdpVideoSender::sender_thread_func, this);
    LOG_INFO("UdpVideoSender started for target: " + target_ip_ + ":" + std::to_string(target_port_));
    return true;
}

void UdpVideoSender::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    LOG_INFO("Stopping UdpVideoSender...");
    input_queue_.set_running(false);
    if (sender_thread_.joinable()) {
        sender_thread_.join();
    }
    LOG_INFO("UdpVideoSender stopped.");
}

void UdpVideoSender::sender_thread_func() {
    ImageData raw_image;
    while (running_) {
        if (input_queue_.pop(raw_image)) {
            try {
                // The input to compress_image is now directly RGB888.
                std::vector<uint8_t> mjpeg_data = jpeg_compressor_.compress_image(
                    raw_image.data.data(),
                    raw_image.width,
                    raw_image.height,
                    80,
                    JCS_RGB
                );

                ImageFrame mjpeg_frame;
                mjpeg_frame.jpeg_data = std::move(mjpeg_data);
                send_mjpeg_frame(mjpeg_frame);

            } catch (const std::runtime_error& e) {
                LOG_ERROR("UdpVideoSender: JPEG compression failed: " + std::string(e.what()));
            }
        }
    }
}

void UdpVideoSender::send_mjpeg_frame(const ImageFrame& frame) {
    const uint8_t* data_ptr = frame.jpeg_data.data();
    size_t total_size = frame.jpeg_data.size();
    size_t bytes_sent = 0;
    
    const size_t FRAG_HEADER_SIZE = 6;
    static uint16_t sequence_number = 0;

    if (total_size == 0) {
        LOG_WARNING("UdpVideoSender: Attempted to send empty MJPEG frame.");
        return;
    }

    size_t payload_per_fragment = MAX_UDP_PAYLOAD_SIZE - FRAG_HEADER_SIZE;
    size_t num_fragments = (total_size + payload_per_fragment - 1) / payload_per_fragment;

    if (num_fragments > 0xFFFF) {
        LOG_ERROR("UdpVideoSender: MJPEG frame too large to fragment. Skipping frame.");
        return;
    }

    sequence_number++;

    for (size_t i = 0; i < num_fragments; ++i) {
        size_t offset = i * payload_per_fragment;
        size_t current_fragment_size = std::min(payload_per_fragment, total_size - offset);

        std::vector<uint8_t> packet_buffer(FRAG_HEADER_SIZE + current_fragment_size);
        
        packet_buffer[0] = (sequence_number >> 8) & 0xFF;
        packet_buffer[1] = sequence_number & 0xFF;
        packet_buffer[2] = (num_fragments >> 8) & 0xFF;
        packet_buffer[3] = num_fragments & 0xFF;
        packet_buffer[4] = (i >> 8) & 0xFF;
        packet_buffer[5] = i & 0xFF;

        std::memcpy(packet_buffer.data() + FRAG_HEADER_SIZE, data_ptr + offset, current_fragment_size);

        ssize_t sent_bytes_current_frag = sendto(sockfd_, (const char*)packet_buffer.data(), packet_buffer.size(), 0,
                                                 (const sockaddr*)&server_addr_, sizeof(server_addr_));
        if (sent_bytes_current_frag < 0) {
            LOG_ERROR("UdpVideoSender: Failed to send UDP fragment " + std::to_string(i) + " for frame " + std::to_string(sequence_number) + ": " + strerror(errno));
            break; 
        }
    }
}

void UdpVideoSender::get_state() const {
    LOG_INFO("--- UdpVideoSender State (Raw Video) ---");
    LOG_INFO("  Running: " + std::to_string(running_));
    LOG_INFO("  Target IP: " + target_ip_);
    LOG_INFO("  Target Port: " + std::to_string(target_port_));
    if (sockfd_ != -1) {
        LOG_INFO("  Socket FD: " + std::to_string(sockfd_));
    } else {
        LOG_INFO("  Socket: Not initialized or closed.");
    }
    LOG_INFO("  Max UDP Payload Size: " + std::to_string(MAX_UDP_PAYLOAD_SIZE));
    LOG_INFO("----------------------------------------");
}