#ifndef UDP_STREAMER_H
#define UDP_STREAMER_H

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include "pipeline_structs.h"

/**
 * @brief Handles UDP streaming of H264 video data using GStreamer.
 */
class UDPStreamer {
public:
    UDPStreamer(int width, int height, double fps);
    ~UDPStreamer();

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    bool is_pipeline_ready() const { return pipeline_ready_.load(); }

    void pushH264Data(std::shared_ptr<H264Buffer> buffer);

    /**
     * @brief Sets the destination IP and port for the UDP stream.
     * @param ip Destination IP address.
     * @param port Destination UDP port.
     */
    void set_destination(const std::string& ip, int port);

private:
    void setup_pipeline();
    void initialization_worker();

    GstElement* pipeline_;
    GstElement* appsrc_;
    GMainLoop* loop_;
    
    std::thread server_thread_;
    std::thread init_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> initializing_;
    std::atomic<bool> pipeline_ready_;
    
    int width_;
    int height_;
    double fps_;
    
    std::string destination_ip_;
    int destination_port_;
    
    std::mutex appsrc_mutex_;
    
    // Timing for PTS calculation
    uint64_t base_time_;
    uint64_t last_pts_;
};

#endif // UDP_STREAMER_H
