#ifndef RTSP_SERVER_WRAPPER_H
#define RTSP_SERVER_WRAPPER_H

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <memory>
#include "pipeline_structs.h"

/**
 * @brief Wrapper for GStreamer RTSP Server.
 */
class RTSPServerWrapper {
public:
    RTSPServerWrapper(int rtspPort, const std::string& streamName, int width, int height, double fps);
    ~RTSPServerWrapper();

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    void pushH264Data(std::shared_ptr<H264Buffer> buffer);
    void set_appsrc(GstElement* appsrc);
    GstElement* get_appsrc() const;
    
    void flush_pending_buffers();
    void send_sps_pps_headers();
    void send_latest_keyframe();
    void manage_client_connection(GstRTSPClient* client);

private:
    static void media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data);
    static void client_connected_cb(GstRTSPServer *server, GstRTSPClient *client, gpointer user_data);
    static void client_closed_cb(GstRTSPClient *client, gpointer user_data);

    void add_pending_client(GstRTSPClient* client);
    std::vector<GstRTSPClient*> take_pending_clients();
    bool is_appsrc_ready() const;
    void internal_cleanup();
    bool is_port_available(int port);
    size_t pending_client_count() const;
    void serverThread();
    void monitor_resources();
    bool is_resource_usage_acceptable();
    void extract_and_store_headers(std::shared_ptr<H264Buffer> buffer);

    GstRTSPServer *server_;
    GstRTSPMountPoints *mounts_;
    GstRTSPMediaFactory *factory_;
    GMainLoop *loop_;
    GSource *timeout_source_;

    int rtspPort_;
    std::string streamName_;
    int width_;
    int height_;
    double fps_;

    std::atomic<bool> running_;
    std::thread server_thread_;
    
    mutable std::mutex appsrc_mutex_;
    GstElement *appsrc_;

    std::mutex pending_buffers_mutex_;
    std::vector<std::shared_ptr<H264Buffer>> pending_buffers_;

    std::mutex sps_pps_mutex_;
    std::vector<uint8_t> sps_buffer_;
    std::vector<uint8_t> pps_buffer_;

    std::mutex latest_keyframe_mutex_;
    std::vector<uint8_t> latest_keyframe_buffer_;

    mutable std::mutex pending_clients_mutex_;
    std::vector<GstRTSPClient*> pending_clients_;

    static std::mutex port_binding_mutex_;
    static std::atomic<bool> port_in_use_;
    static std::mutex camera_access_mutex_;
    static std::atomic<bool> camera_in_use_;
    static std::atomic<int> active_client_count_;
    
    uint64_t base_time_ = 0;
    uint64_t last_pts_ = 0;
};

#endif // RTSP_SERVER_WRAPPER_H
