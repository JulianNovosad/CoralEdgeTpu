// Verified headers: [memory, thread, atomic, mutex, string...]
// Verification timestamp: 2026-01-06 17:08:04
#ifndef MPEGTS_SERVER_H
#define MPEGTS_SERVER_H

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include "pipeline_structs.h"

/**
 * @brief Handles MPEG-TS over TCP streaming of H264 video data using GStreamer.
 */
class MpegTsServer {
public:
    MpegTsServer(int width, int height, double fps, const std::string& default_address, unsigned short default_port);
    ~MpegTsServer();

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    bool is_pipeline_ready() const { return pipeline_ready_.load(); }

    void pushH264Data(std::shared_ptr<H264Buffer> buffer);

    /**
     * @brief Sets the destination IP for orientation data.
     * @param ip Destination IP address.
     */
    void set_destination(const std::string& ip);

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
    unsigned short default_port_; // Added to store the default port
    GstElement* udpsink_;
    void update_udpsink_host(const std::string& host);
    
    std::mutex appsrc_mutex_;
    
    // Timing for PTS calculation
    uint64_t base_time_;
    uint64_t last_pts_;
};

#endif // MPEGTS_SERVER_H
