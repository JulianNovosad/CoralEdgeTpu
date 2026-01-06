// Verified headers: [mpegts_server.h, util_logging.h, sstream, cstring, arpa/inet.h]
// Verification timestamp: 2026-01-06 17:08:04
#include "mpegts_server.h"
#include "util_logging.h"
#include <glib.h>
#include <sstream>
#include <cstring>
#include <arpa/inet.h>

extern std::atomic<bool> g_running;

MpegTsServer::MpegTsServer(int width, int height, double fps, const std::string& default_address, unsigned short default_port)
    : pipeline_(nullptr), appsrc_(nullptr), loop_(nullptr),
      running_(false), initializing_(false), pipeline_ready_(false),
      width_(width), height_(height), fps_(fps),
      destination_ip_(default_address), 
      default_port_(default_port), 
      base_time_(0), last_pts_(0) {
}

void MpegTsServer::update_udpsink_host(const std::string& host) {
    std::string actual_host = host.empty() ? destination_ip_ : host;

    if (destination_ip_ == actual_host && pipeline_ready_.load() && udpsink_) {
        return;
    }

    APP_LOG_INFO("MpegTsServer: Attempting to set multiudpsink destination to " + actual_host);
    destination_ip_ = actual_host;

    if (pipeline_ready_.load() && udpsink_) {
        std::string clients = actual_host + ":" + std::to_string(default_port_);
        g_object_set(udpsink_, "clients", clients.c_str(), NULL);
        APP_LOG_INFO("MpegTsServer: Successfully updated multiudpsink clients to " + clients);
    } else if (!pipeline_ready_.load()) {
        APP_LOG_DEBUG("MpegTsServer: Pipeline not ready, will update host when it is.");
    } else {
        APP_LOG_DEBUG("MpegTsServer: udpsink_ is not available, will attempt update when it becomes available.");
    }
}

bool MpegTsServer::start() {
    if (running_.load() || initializing_.load()) return true;
    
    initializing_.store(true);
    running_.store(true);
    
    init_thread_ = std::thread(&MpegTsServer::initialization_worker, this);
    
    return true;
}

void MpegTsServer::initialization_worker() {
    APP_LOG_INFO("MpegTsServer: Starting background initialization...");
    
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
    
    setup_pipeline();
    
    if (!pipeline_) {
        APP_LOG_ERROR("MpegTsServer: Failed to setup pipeline");
        initializing_.store(false);
        running_.store(false);
        return;
    }

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        APP_LOG_ERROR("MpegTsServer: Failed to set pipeline to PLAYING state");
        initializing_.store(false);
        running_.store(false);
        return;
    } else if (ret == GST_STATE_CHANGE_ASYNC) {
        GstState current, pending;
        ret = gst_element_get_state(pipeline_, &current, &pending, 200 * GST_MSECOND);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            APP_LOG_ERROR("MpegTsServer: Async state change failed after timeout");
            initializing_.store(false);
            running_.store(false);
            return;
        }
    }
    
    server_thread_ = std::thread([this]() {
        if (loop_) {
            GstBus* bus = gst_element_get_bus(pipeline_);
            gst_bus_add_watch(bus, [](GstBus* bus, GstMessage* msg, gpointer data) -> gboolean {
                switch (GST_MESSAGE_TYPE(msg)) {
                    case GST_MESSAGE_ERROR: {
                        GError* err;
                        gchar* debug;
                        gst_message_parse_error(msg, &err, &debug);
                        APP_LOG_ERROR("MpegTsServer GStreamer Error: " + std::string(err->message));
                        g_error_free(err);
                        g_free(debug);
                        break;
                    }
                    case GST_MESSAGE_WARNING: {
                        GError* err;
                        gchar* debug;
                        gst_message_parse_warning(msg, &err, &debug);
                        APP_LOG_WARNING("MpegTsServer GStreamer Warning: " + std::string(err->message));
                        g_error_free(err);
                        g_free(debug);
                        break;
                    }
                    default:
                        break;
                }
                return TRUE;
            }, nullptr);
            gst_object_unref(bus);
            g_main_loop_run(loop_);
        }
    });
    
    pipeline_ready_.store(true);
    initializing_.store(false);

    if (!destination_ip_.empty() && udpsink_) {
        update_udpsink_host(destination_ip_);
    }
    
    APP_LOG_INFO("MpegTsServer: Background initialization complete. Streaming RTP/H264 on port " + std::to_string(default_port_) + " (UDP) to " + destination_ip_);
}

void MpegTsServer::setup_pipeline() {
    std::stringstream ss;
    std::string host_to_use = destination_ip_;
    if (host_to_use == "0.0.0.0") {
        host_to_use = "127.0.0.1";
        APP_LOG_WARNING("MpegTsServer: udpsink destination was 0.0.0.0, defaulting to 127.0.0.1 for initial setup.");
    }
    
    std::string clients = host_to_use + ":" + std::to_string(default_port_);
    
    ss << "appsrc name=src is-live=true format=time ! "
       << "queue leaky=downstream max-size-buffers=50 ! "
       << "h264parse ! "
       << "rtph264pay name=pay0 pt=96 ! "
       << "multiudpsink name=myudpsink clients=" << clients << " sync=false";
    
    std::string pipeline_string = ss.str();

    GError* error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_string.c_str(), &error);

    if (error) {
        APP_LOG_ERROR("MpegTsServer: Failed to parse pipeline: " + std::string(error->message));
        g_error_free(error);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(appsrc_mutex_);
        appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
        if (!appsrc_) {
            APP_LOG_ERROR("MpegTsServer: Failed to get appsrc from pipeline");
            return;
        }

        udpsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "myudpsink");
        if (!udpsink_) {
            APP_LOG_ERROR("MpegTsServer: Failed to get multiudpsink from pipeline by name 'myudpsink'");
            return;
        }

        std::stringstream caps_ss;
        caps_ss << "video/x-h264,width=" << width_ << ",height=" << height_ << ",framerate=" << (int)fps_ << "/1,stream-format=byte-stream,alignment=nal";
        GstCaps* caps = gst_caps_from_string(caps_ss.str().c_str());
        gst_app_src_set_caps(GST_APP_SRC(appsrc_), caps);
        gst_caps_unref(caps);
    }

    loop_ = g_main_loop_new(NULL, FALSE);
}

MpegTsServer::~MpegTsServer() {
    stop();
}

void MpegTsServer::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    
    APP_LOG_INFO("MpegTsServer: Stopping...");

    if (init_thread_.joinable()) {
        init_thread_.join();
    }
    
    pipeline_ready_.store(false);
    initializing_.store(false);

    if (loop_) {
        g_main_loop_quit(loop_);
    }
    
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        GstState current, pending;
        gst_element_get_state(pipeline_, &current, &pending, 100 * GST_MSECOND);
    }
    
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    if (loop_) {
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }
    
    {
        std::lock_guard<std::mutex> lock(appsrc_mutex_);
        if (appsrc_) {
            gst_object_unref(appsrc_);
            appsrc_ = nullptr;
        }
    }

    if (udpsink_) {
        gst_object_unref(udpsink_);
        udpsink_ = nullptr;
    }

    if (pipeline_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    
    APP_LOG_INFO("MpegTsServer: Stopped");
}

void MpegTsServer::pushH264Data(std::shared_ptr<H264Buffer> buffer) {
    if (!pipeline_ready_.load()) {
        return;
    }
    if (!buffer || !buffer->data.data() || buffer->size == 0) {
        return;
    }
    
    GstElement* src = nullptr;
    {
        std::lock_guard<std::mutex> lock(appsrc_mutex_);
        if (appsrc_) {
            src = appsrc_;
            gst_object_ref(src);
        }
    }
    
    if (!src) return;
    
    GstBuffer* gst_buffer = gst_buffer_new_and_alloc(buffer->size);
    if (!gst_buffer) {
        gst_object_unref(src);
        return;
    }
    
    GstMapInfo map_info;
    if (gst_buffer_map(gst_buffer, &map_info, GST_MAP_WRITE)) {
        memcpy(map_info.data, buffer->data.data(), buffer->size);
        gst_buffer_unmap(gst_buffer, &map_info);
    }
    
    // Set timestamp
    GstClockTime pts = (GstClockTime)(buffer->timestamp_epoch_ms % 1000000) * GST_MSECOND;
    if (base_time_ == 0) base_time_ = pts;
    pts = pts - base_time_;
    if (pts <= last_pts_ && last_pts_ != 0) pts = last_pts_ + 1 * GST_MSECOND;
    last_pts_ = pts;
    
    GST_BUFFER_PTS(gst_buffer) = pts;
    GST_BUFFER_DTS(gst_buffer) = pts;
    GST_BUFFER_DURATION(gst_buffer) = (GstClockTime)(1000.0 / fps_) * GST_MSECOND;
    
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(src), gst_buffer);
    if (ret != GST_FLOW_OK) {
        APP_LOG_WARNING("MpegTsServer: push failed: " + std::to_string(ret));
    } else {
        static int log_counter = 0;
        if (log_counter++ % 30 == 0) {
             std::string hex_dump = "";
             for (int i = 0; i < std::min((int)buffer->size, 16); i++) {
                 char buf[4];
                 sprintf(buf, "%02X ", buffer->data[i]);
                 hex_dump += buf;
             }
             APP_LOG_INFO("MpegTsServer: Successfully pushed buffer to pipeline. PTS=" + std::to_string(pts) + " size=" + std::to_string(buffer->size) + " data=" + hex_dump);
        }
    }
    
    gst_object_unref(src);
}

void MpegTsServer::set_destination(const std::string& ip) {
    update_udpsink_host(ip);
}