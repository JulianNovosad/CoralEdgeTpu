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
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include "pipeline_structs.h"

class RTSPServerWrapper {
public:
    RTSPServerWrapper(int rtspPort, const std::string& streamName, int width = 1536, int height = 864, double fps = 40.0);
    ~RTSPServerWrapper();
    
    bool start();
    void stop();
    bool isRunning() const { return running_; }
    
    // Function to push H.264 NAL units to the stream
    void pushH264Data(std::shared_ptr<H264Buffer> buffer);
    
    // Thread-safe appsrc management
    void set_appsrc(GstElement* appsrc);
    GstElement* get_appsrc() const;  // Returns ref'd pointer - caller must unref
    
    // Callbacks
    static void client_connected_cb(GstRTSPServer *server, GstRTSPClient *client, gpointer user_data);
    static void client_closed_cb(GstRTSPClient *client, gpointer user_data);
    
    // Public methods for use in callbacks
    void send_sps_pps_headers();
    void send_latest_keyframe();
    std::vector<GstRTSPClient*> take_pending_clients();
    void flush_pending_buffers();
    bool is_appsrc_ready() const;    // Check if appsrc is ready to receive data

private:
    void serverThread();
    void extract_and_store_headers(std::shared_ptr<H264Buffer> buffer);
    
    // GStreamer components
    GstRTSPServer* server_;
    GstRTSPMountPoints* mounts_;
    GstRTSPMediaFactory* factory_;
    GMainLoop* loop_;
    GSource* timeout_source_; // For more responsive shutdown
    
    // Server configuration
    int rtspPort_;
    std::string streamName_;
    int width_;
    int height_;
    double fps_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    

    
    // Buffer for frames when appsrc is not ready (for SPS/PPS handling)
    std::vector<std::shared_ptr<H264Buffer>> pending_buffers_;
    mutable std::mutex pending_buffers_mutex_;
    
    // Store SPS and PPS NAL units for new clients
    std::vector<uint8_t> sps_buffer_;
    std::vector<uint8_t> pps_buffer_;
    mutable std::mutex sps_pps_mutex_;
    
    // Store the latest keyframe for immediate delivery to new clients
    std::vector<uint8_t> latest_keyframe_buffer_;
    mutable std::mutex latest_keyframe_mutex_;
    
    // GStreamer appsrc for feeding frames
    GstElement* appsrc_;
    mutable std::mutex appsrc_mutex_;
    
    // Throughput counters
    std::atomic<uint64_t> frames_in_{0};
    std::atomic<uint64_t> frames_out_{0};
    std::atomic<uint64_t> bytes_in_{0};
    std::atomic<uint64_t> bytes_out_{0};
    
    // Binary lock for port binding
    static std::mutex port_binding_mutex_;
    static std::atomic<bool> port_in_use_;
    
    // Check if port is available
    static bool is_port_available(int port);
    
    // Client connection management
    std::vector<GstRTSPClient*> pending_clients_;
    mutable std::mutex pending_clients_mutex_;
    
    // Client management methods
    void manage_client_connection(GstRTSPClient* client);
    
    // Thread-safe pending client management
    void add_pending_client(GstRTSPClient* client);
    size_t pending_client_count() const;
    
    // Resource management
    static std::mutex camera_access_mutex_;
    static std::atomic<bool> camera_in_use_;
    static std::atomic<int> active_client_count_;
    static const int MAX_SIMULTANEOUS_CLIENTS = 10; // Limit simultaneous clients
    
    // Resource monitoring
    void monitor_resources();
    bool is_resource_usage_acceptable();
    
    // Internal cleanup
    void internal_cleanup();
    
    // Timing synchronization
    std::atomic<long long> base_time_{0};
    std::atomic<long long> first_pts_{0};
    std::atomic<GstClockTime> last_pts_{0};
};

#endif // RTSP_SERVER_H