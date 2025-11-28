#ifndef UDP_VIDEO_SENDER_H
#define UDP_VIDEO_SENDER_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <netinet/in.h> // For sockaddr_in

#include "pipeline_structs.h" // For ImageQueue
#include "jpeg_wrapper.h"   // For JpegCompressGuard

class UdpVideoSender {
public:
    UdpVideoSender(const std::string& dest_ip, int dest_port, ImageQueue& input_queue, int jpeg_quality); // Added jpeg_quality parameter
    ~UdpVideoSender();
    bool start();
    void stop();
    bool is_running() const { return running_; }
    void get_state() const;

private:
    void sender_thread_func();
    void send_mjpeg_frame(const ImageFrame& frame);

    std::string target_ip_;
    int target_port_;
    ImageQueue& input_queue_;
    int sockfd_ = -1;
    struct sockaddr_in server_addr_;
    std::thread sender_thread_;
    std::atomic<bool> running_{false};
    JpegCompressGuard jpeg_compressor_;
    int jpeg_quality_; // Added to store JPEG quality

    static const size_t MAX_UDP_PAYLOAD_SIZE = 65507;
};

#endif // UDP_VIDEO_SENDER_H