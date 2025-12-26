#ifndef RTSP_SERVER_H
#define RTSP_SERVER_H

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/app/gstappsrc.h>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <queue>
#include "pipeline_structs.h"

class RTSPServerWrapper {
public:
    RTSPServerWrapper(int rtspPort, const std::string& streamName);
    ~RTSPServerWrapper();
    
    bool start();
    void stop();
    bool isRunning() const { return running_; }
    
    // Function to push H.264 NAL units to the stream
    void pushH264Data(std::shared_ptr<H264Buffer> buffer);
    
    // Method to set the appsrc from the callback
    void set_appsrc(GstElement* appsrc) { appsrc_ = appsrc; }
    GstElement* get_appsrc() const { return appsrc_; }

private:
    void serverThread();
    
public:
    void flush_pending_buffers(GstElement* appsrc);
    void send_sps_pps_headers(GstElement* appsrc);
    
private:
    void extract_and_store_headers(std::shared_ptr<H264Buffer> buffer);
    void send_latest_keyframe(GstElement* appsrc);
    
    // GStreamer components
    GstRTSPServer* server_;
    GstRTSPMountPoints* mounts_;
    GstRTSPMediaFactory* factory_;
    GMainLoop* loop_;
    GSource* timeout_source_; // For more responsive shutdown
    
    // Server configuration
    int rtspPort_;
    std::string streamName_;
    
    // Threading
    std::atomic<bool> running_;
    std::thread server_thread_;
    
    // Latest frame for RTSP streaming
    std::mutex latest_mutex_;
    std::shared_ptr<H264Buffer> latest_buffer_;
    
    // Queue for buffering frames
    std::queue<std::shared_ptr<H264Buffer>> frame_queue_;
    std::mutex queue_mutex_;
    
    // Buffer for frames when appsrc is not ready (for SPS/PPS handling)
    std::vector<std::shared_ptr<H264Buffer>> pending_buffers_;
    std::mutex pending_buffers_mutex_;
    
    // Store SPS and PPS NAL units for new clients
    std::vector<uint8_t> sps_buffer_;
    std::vector<uint8_t> pps_buffer_;
    std::mutex sps_pps_mutex_;
    
    // Store the latest keyframe for immediate delivery to new clients
    std::vector<uint8_t> latest_keyframe_buffer_;
    std::mutex latest_keyframe_mutex_;
    
    // GStreamer appsrc for feeding frames
    GstElement* appsrc_;
};

#endif // RTSP_SERVER_H