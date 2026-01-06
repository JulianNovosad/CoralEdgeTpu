// Verified headers: [rtsp_server.h, util_logging.h, chrono, cstring, iostream...]
// Verification timestamp: 2026-01-06 17:08:04
#include "rtsp_server.h"
#include "util_logging.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <glib.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <atomic>
#include <sstream>

extern std::atomic<bool> g_running;

// Static member definitions
std::mutex RTSPServerWrapper::port_binding_mutex_;
std::atomic<bool> RTSPServerWrapper::port_in_use_{false};
std::mutex RTSPServerWrapper::camera_access_mutex_;
std::atomic<bool> RTSPServerWrapper::camera_in_use_{false};
std::atomic<int> RTSPServerWrapper::active_client_count_{0};

// Static callback function to handle media configuration
void RTSPServerWrapper::media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data) {
    RTSPServerWrapper *server_wrapper = static_cast<RTSPServerWrapper*>(user_data);
    APP_LOG_INFO("RTSP: media_configure callback triggered");
    
    GstElement *pipeline = gst_rtsp_media_get_element(media);
    if (!pipeline) {
        APP_LOG_ERROR("RTSP: Failed to get pipeline from media");
        return;
    }

    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "video_source");
    if (appsrc) {
        APP_LOG_INFO("RTSP: Found appsrc 'video_source'");
        server_wrapper->set_appsrc(appsrc);
        
        g_object_set(G_OBJECT(appsrc),
            "is-live", TRUE,
            "format", GST_FORMAT_TIME,
            "do-timestamp", FALSE,
            NULL);
            
        server_wrapper->flush_pending_buffers();
        server_wrapper->send_sps_pps_headers();
        server_wrapper->send_latest_keyframe();
    } else {
        APP_LOG_ERROR("RTSP: Failed to get appsrc element from RTSP media pipeline");
    }
    
    gst_rtsp_media_set_reusable(media, TRUE);
    gst_rtsp_media_set_shared(media, TRUE);
    gst_object_unref(pipeline);
}

void RTSPServerWrapper::client_connected_cb(GstRTSPServer *server, GstRTSPClient *client, gpointer user_data) {
    RTSPServerWrapper *server_wrapper = static_cast<RTSPServerWrapper*>(user_data);
    APP_LOG_INFO("RTSP: New client attempting connection");
    
    if (server_wrapper) {
        server_wrapper->manage_client_connection(client);
    }
    g_signal_connect(client, "closed", G_CALLBACK(client_closed_cb), user_data);
}

void RTSPServerWrapper::client_closed_cb(GstRTSPClient *client, gpointer user_data) {
    active_client_count_--;
    APP_LOG_INFO("RTSP: Client disconnected");
}

RTSPServerWrapper::RTSPServerWrapper(int rtspPort, const std::string& streamName, int width, int height, double fps)
    : server_(nullptr), mounts_(nullptr), factory_(nullptr), loop_(nullptr),
      timeout_source_(nullptr), rtspPort_(rtspPort), streamName_(streamName),
      width_(width), height_(height), fps_(fps),
      running_(false), server_thread_(), appsrc_(nullptr) {
}

void RTSPServerWrapper::set_appsrc(GstElement* appsrc) {
    std::lock_guard<std::mutex> lock(appsrc_mutex_);
    if (appsrc_ != nullptr) {
        gst_object_unref(appsrc_);
    }
    appsrc_ = appsrc;
    if (appsrc_ != nullptr) {
        gst_object_ref(appsrc);
    }
}

GstElement* RTSPServerWrapper::get_appsrc() const {
    std::lock_guard<std::mutex> lock(appsrc_mutex_);
    if (appsrc_ != nullptr) {
        gst_object_ref(appsrc_);
        return appsrc_;
    }
    return nullptr;
}

RTSPServerWrapper::~RTSPServerWrapper() {
    stop();
    set_appsrc(nullptr);
}

bool RTSPServerWrapper::start() {
    if (running_) return true;
    internal_cleanup();
    
    if (!is_port_available(rtspPort_)) return false;
    
    std::lock_guard<std::mutex> lock(port_binding_mutex_);
    if (port_in_use_.load()) return false;
    port_in_use_.store(true);
    
    if (!gst_is_initialized()) gst_init(nullptr, nullptr);
    
    server_ = gst_rtsp_server_new();
    gst_rtsp_server_set_address(server_, "0.0.0.0");
    gst_rtsp_server_set_service(server_, std::to_string(rtspPort_).c_str());
    
    mounts_ = gst_rtsp_server_get_mount_points(server_);
    factory_ = gst_rtsp_media_factory_new();
    
    std::stringstream pipeline_ss;
    pipeline_ss << "( appsrc name=video_source is-live=true format=time "
                << "caps=video/x-h264,stream-format=byte-stream,alignment=au,"
                << "width=" << width_ << ",height=" << height_ << ",framerate=" << (int)fps_ << "/1 ! "
                << "h264parse config-interval=1 ! "
                << "rtph264pay name=pay0 pt=96 )";
    
    gst_rtsp_media_factory_set_launch(factory_, pipeline_ss.str().c_str());
    
    gst_rtsp_media_factory_set_shared(factory_, TRUE);
    g_signal_connect(factory_, "media-configure", G_CALLBACK(media_configure), this);
    
    gst_rtsp_mount_points_add_factory(mounts_, streamName_.c_str(), factory_);
    g_signal_connect(server_, "client-connected", G_CALLBACK(client_connected_cb), this);
    g_object_unref(mounts_);
    
    loop_ = g_main_loop_new(NULL, FALSE);
    if (gst_rtsp_server_attach(server_, NULL) == 0) return false;
    
    running_ = true;
    server_thread_ = std::thread([this]() { g_main_loop_run(loop_); });
    
    APP_LOG_INFO("RTSP SERVER STARTED on port " + std::to_string(rtspPort_));
    return true;
}

void RTSPServerWrapper::stop() {
    if (!running_) return;
    running_ = false;
    if (loop_) g_main_loop_quit(loop_);
    if (server_thread_.joinable()) server_thread_.join();
    if (loop_) {
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }
    if (server_) {
        g_object_unref(server_);
        server_ = nullptr;
    }
    port_in_use_.store(false);
}

void RTSPServerWrapper::pushH264Data(std::shared_ptr<H264Buffer> buffer) {
    if (!buffer) return;
    
    GstElement* appsrc_element = get_appsrc();
    if (!appsrc_element) {
        std::lock_guard<std::mutex> lock(pending_buffers_mutex_);
        if (pending_buffers_.size() > 10) pending_buffers_.erase(pending_buffers_.begin());
        pending_buffers_.push_back(buffer);
        return;
    }
    
    GstBuffer* gst_buffer = gst_buffer_new_and_alloc(buffer->size);
    if (!gst_buffer) {
        gst_object_unref(appsrc_element);
        return;
    }
    
    GstMapInfo map_info;
    if (gst_buffer_map(gst_buffer, &map_info, GST_MAP_WRITE)) {
        memcpy(map_info.data, buffer->data.data(), buffer->size);
        gst_buffer_unmap(gst_buffer, &map_info);
    }
    
    GstState state;
    gst_element_get_state(appsrc_element, &state, NULL, 0);
    if (state < GST_STATE_PAUSED) {
        gst_buffer_unref(gst_buffer);
        gst_object_unref(appsrc_element);
        return;
    }
    
    if (base_time_ == 0) base_time_ = buffer->timestamp_epoch_ms;
    
    extract_and_store_headers(buffer);
    
    GstClockTime pts = 0;
    if (buffer->timestamp_epoch_ms >= base_time_) {
        pts = (GstClockTime)(buffer->timestamp_epoch_ms - base_time_) * GST_MSECOND;
    }
    if (pts <= last_pts_ && last_pts_ != 0) pts = last_pts_ + 1 * GST_MSECOND;
    last_pts_ = pts;
    
    GST_BUFFER_PTS(gst_buffer) = pts;
    GST_BUFFER_DTS(gst_buffer) = pts;
    GST_BUFFER_DURATION(gst_buffer) = (GstClockTime)(1000 / 40) * GST_MSECOND;
    
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_element), gst_buffer);
    if (ret != GST_FLOW_OK) {
        APP_LOG_DEBUG("RTSP: push failed: " + std::to_string(ret));
    }
    
    gst_object_unref(appsrc_element);
}

void RTSPServerWrapper::extract_and_store_headers(std::shared_ptr<H264Buffer> buffer) {
    if (!buffer || buffer->size < 5) return;
    const uint8_t* data = buffer->data.data();
    size_t size = buffer->size;
    size_t pos = 0;
    while (pos + 3 < size) {
        size_t start_code_len = 0;
        if (data[pos] == 0 && data[pos+1] == 0 && data[pos+2] == 0 && data[pos+3] == 1) start_code_len = 4;
        else if (data[pos] == 0 && data[pos+1] == 0 && data[pos+2] == 1) start_code_len = 3;
        if (start_code_len > 0) {
            size_t start = pos;
            uint8_t nal_type = data[pos + start_code_len] & 0x1F;
            pos += start_code_len;
            size_t next_pos = pos;
            size_t end = size;
            while (next_pos + 2 < size) {
                if (data[next_pos] == 0 && data[next_pos+1] == 0 && (data[next_pos+2] == 1 || (next_pos + 3 < size && data[next_pos+2] == 0 && data[next_pos+3] == 1))) {
                    end = next_pos;
                    break;
                }
                next_pos++;
            }
            if (nal_type == 7) {
                std::lock_guard<std::mutex> lock(sps_pps_mutex_);
                sps_buffer_.assign(data + start, data + end);
            } else if (nal_type == 8) {
                std::lock_guard<std::mutex> lock(sps_pps_mutex_);
                pps_buffer_.assign(data + start, data + end);
            } else if (nal_type == 5) {
                std::lock_guard<std::mutex> lock(latest_keyframe_mutex_);
                latest_keyframe_buffer_.assign(data + start, data + end);
            }
            pos = end;
        } else pos++;
    }
}

void RTSPServerWrapper::send_sps_pps_headers() {
    std::lock_guard<std::mutex> lock(sps_pps_mutex_);
    GstElement* appsrc_element = get_appsrc();
    if (!appsrc_element) return;
    if (!sps_buffer_.empty()) {
        GstBuffer* b = gst_buffer_new_and_alloc(sps_buffer_.size());
        GstMapInfo m;
        gst_buffer_map(b, &m, GST_MAP_WRITE);
        memcpy(m.data, sps_buffer_.data(), sps_buffer_.size());
        gst_buffer_unmap(b, &m);
        gst_app_src_push_buffer(GST_APP_SRC(appsrc_element), b);
    }
    if (!pps_buffer_.empty()) {
        GstBuffer* b = gst_buffer_new_and_alloc(pps_buffer_.size());
        GstMapInfo m;
        gst_buffer_map(b, &m, GST_MAP_WRITE);
        memcpy(m.data, pps_buffer_.data(), pps_buffer_.size());
        gst_buffer_unmap(b, &m);
        gst_app_src_push_buffer(GST_APP_SRC(appsrc_element), b);
    }
    gst_object_unref(appsrc_element);
}

void RTSPServerWrapper::send_latest_keyframe() {
    std::lock_guard<std::mutex> lock(latest_keyframe_mutex_);
    GstElement* appsrc_element = get_appsrc();
    if (!appsrc_element || latest_keyframe_buffer_.empty()) {
        if (appsrc_element) gst_object_unref(appsrc_element);
        return;
    }
    GstBuffer* b = gst_buffer_new_and_alloc(latest_keyframe_buffer_.size());
    GstMapInfo m;
    gst_buffer_map(b, &m, GST_MAP_WRITE);
    memcpy(m.data, latest_keyframe_buffer_.data(), latest_keyframe_buffer_.size());
    gst_buffer_unmap(b, &m);
    gst_app_src_push_buffer(GST_APP_SRC(appsrc_element), b);
    gst_object_unref(appsrc_element);
}

void RTSPServerWrapper::manage_client_connection(GstRTSPClient* client) {
    active_client_count_++;
    if (is_appsrc_ready()) {
        send_sps_pps_headers();
        send_latest_keyframe();
    } else add_pending_client(client);
}

void RTSPServerWrapper::add_pending_client(GstRTSPClient* client) {
    std::lock_guard<std::mutex> lock(pending_clients_mutex_);
    gst_object_ref(client);
    pending_clients_.push_back(client);
}

std::vector<GstRTSPClient*> RTSPServerWrapper::take_pending_clients() {
    std::lock_guard<std::mutex> lock(pending_clients_mutex_);
    auto c = std::move(pending_clients_);
    pending_clients_.clear();
    return c;
}

void RTSPServerWrapper::flush_pending_buffers() {
    std::vector<std::shared_ptr<H264Buffer>> b;
    {
        std::lock_guard<std::mutex> lock(pending_buffers_mutex_);
        b = std::move(pending_buffers_);
        pending_buffers_.clear();
    }
    for (auto& buf : b) pushH264Data(buf);
}

bool RTSPServerWrapper::is_appsrc_ready() const {
    GstElement* e = get_appsrc();
    if (!e) return false;
    GstState s;
    gst_element_get_state(e, &s, NULL, 0);
    gst_object_unref(e);
    return s >= GST_STATE_PAUSED;
}

void RTSPServerWrapper::internal_cleanup() {
    APP_LOG_INFO("Internal cleanup completed");
}

bool RTSPServerWrapper::is_port_available(int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return false;
    int f = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &f, sizeof(f));
    struct sockaddr_in a;
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons(port);
    int r = bind(s, (struct sockaddr*)&a, sizeof(a));
    close(s);
    return r == 0;
}

size_t RTSPServerWrapper::pending_client_count() const {
    std::lock_guard<std::mutex> lock(pending_clients_mutex_);
    return pending_clients_.size();
}

void RTSPServerWrapper::serverThread() {
    g_main_loop_run(loop_);
}

void RTSPServerWrapper::monitor_resources() {}
bool RTSPServerWrapper::is_resource_usage_acceptable() { return true; }
