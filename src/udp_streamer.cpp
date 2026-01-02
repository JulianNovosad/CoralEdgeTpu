#include "udp_streamer.h"
#include "util_logging.h"
#include <glib.h>
#include <sstream>
#include <cstring>
#include <arpa/inet.h>

extern std::atomic<bool> g_running;

UDPStreamer::UDPStreamer(int width, int height, double fps)
    : pipeline_(nullptr), appsrc_(nullptr), loop_(nullptr),
      running_(false), initializing_(false), pipeline_ready_(false),
      width_(width), height_(height), fps_(fps), 
      destination_ip_("255.255.255.255"), destination_port_(5000),
      base_time_(0), last_pts_(0) {
}

UDPStreamer::~UDPStreamer() {
    stop();
}

bool UDPStreamer::start() {
    if (running_.load() || initializing_.load()) return true;
    
    initializing_.store(true);
    running_.store(true);
    
    // Spawn initialization in a background thread to avoid blocking main loop
    init_thread_ = std::thread(&UDPStreamer::initialization_worker, this);
    
    return true;
}

void UDPStreamer::initialization_worker() {
    APP_LOG_INFO("UDPStreamer: Starting background initialization...");
    
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
    
    // Pre-flight check for IP address format
    struct sockaddr_in sa;
    if (inet_pton(AF_INET, destination_ip_.c_str(), &(sa.sin_addr)) != 1) {
        APP_LOG_ERROR("UDPStreamer: Invalid destination IP format: " + destination_ip_);
        initializing_.store(false);
        running_.store(false);
        return;
    }

    setup_pipeline();
    
    if (!pipeline_) {
        APP_LOG_ERROR("UDPStreamer: Failed to setup pipeline");
        initializing_.store(false);
        running_.store(false);
        return;
    }

    // Attempt to set pipeline to PLAYING state with timeout
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        APP_LOG_ERROR("UDPStreamer: Failed to set pipeline to PLAYING state");
        initializing_.store(false);
        running_.store(false);
        return;
    } else if (ret == GST_STATE_CHANGE_ASYNC) {
        // Wait for state change with a 200ms timeout
        GstState current, pending;
        ret = gst_element_get_state(pipeline_, &current, &pending, 200 * GST_MSECOND);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            APP_LOG_ERROR("UDPStreamer: Async state change failed after timeout");
            initializing_.store(false);
            running_.store(false);
            return;
        }
    }
    
    // Start the GLib main loop in a thread
    server_thread_ = std::thread([this]() {
        if (loop_) {
            g_main_loop_run(loop_);
        }
    });
    
    pipeline_ready_.store(true);
    initializing_.store(false);
    
    APP_LOG_INFO("UDPStreamer: Background initialization complete. Streaming to " + destination_ip_ + ":" + std::to_string(destination_port_));
}

void UDPStreamer::setup_pipeline() {
    // Pipeline: appsrc (h264) -> queue -> h264parse -> mpegtsmux -> tcpserversink
    // Restored MPEG-TS for VLC compatibility (URL: tcp://<PI_IP>:5000)
    std::stringstream ss;
    ss << "appsrc name=src is-live=true format=time ! "
       << "queue leaky=downstream max-size-buffers=50 ! "
       << "h264parse ! "
       << "mpegtsmux ! "
       << "tcpserversink host=0.0.0.0 port=5000 sync=false";
    
    std::string pipeline_string = ss.str();

    GError* error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_string.c_str(), &error);

    if (error) {
        APP_LOG_ERROR("UDPStreamer: Failed to parse pipeline: " + std::string(error->message));
        g_error_free(error);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(appsrc_mutex_);
        appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
        if (!appsrc_) {
            APP_LOG_ERROR("UDPStreamer: Failed to get appsrc from pipeline");
            return;
        }

        // Set caps on appsrc to indicate H.264 video
        std::stringstream caps_ss;
        caps_ss << "video/x-h264,width=" << width_ << ",height=" << height_ << ",framerate=" << (int)fps_ << "/1,stream-format=byte-stream,alignment=au";
        GstCaps* caps = gst_caps_from_string(caps_ss.str().c_str());
        gst_app_src_set_caps(GST_APP_SRC(appsrc_), caps);
        gst_caps_unref(caps);
    }

    // Create main loop
    loop_ = g_main_loop_new(NULL, FALSE);
}

void UDPStreamer::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    
    APP_LOG_INFO("UDPStreamer: Stopping...");

    // Wait for initialization thread if it's still running
    if (init_thread_.joinable()) {
        init_thread_.join();
    }
    
    pipeline_ready_.store(false);
    initializing_.store(false);

    if (loop_) {
        g_main_loop_quit(loop_);
    }
    
    if (pipeline_) {
        // Safe transition to NULL with timeout
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

    if (pipeline_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    
    APP_LOG_INFO("UDPStreamer: Stopped");
}

void UDPStreamer::pushH264Data(std::shared_ptr<H264Buffer> buffer) {
    // Mandate: Drop frames if pipeline is not ready to avoid blocking main loop
    if (!pipeline_ready_.load()) {
        // Limit debug spam
        static int drop_log_count = 0;
        if (drop_log_count++ % 100 == 0) std::cerr << "DEBUG: UDPStreamer dropping frame (pipeline not ready)" << std::endl;
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
    GstClockTime pts = 0;
    if (base_time_ == 0) base_time_ = static_cast<GstClockTime>(buffer->timestamp_epoch_ms);
    
    if (static_cast<GstClockTime>(buffer->timestamp_epoch_ms) >= base_time_) {
        pts = (static_cast<GstClockTime>(buffer->timestamp_epoch_ms) - base_time_) * GST_MSECOND;
    }
    if (pts <= last_pts_ && last_pts_ != 0) pts = last_pts_ + 1 * GST_MSECOND;
    last_pts_ = pts;
    
    GST_BUFFER_PTS(gst_buffer) = pts;
    GST_BUFFER_DTS(gst_buffer) = pts;
    GST_BUFFER_DURATION(gst_buffer) = (GstClockTime)(1000.0 / fps_) * GST_MSECOND;
    
    // Non-blocking push to appsrc
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(src), gst_buffer);
    if (ret != GST_FLOW_OK) {
        APP_LOG_DEBUG("UDPStreamer: push failed: " + std::to_string(ret));
    } else {
        // push OK
    }
    
    gst_object_unref(src);
}

void UDPStreamer::set_destination(const std::string& ip, int port) {
    if (destination_ip_ == ip && destination_port_ == port) {
        return;
    }
    
    APP_LOG_INFO("UDPStreamer: Destination change requested to " + ip + ":" + std::to_string(port));
    
    // Non-blocking restart
    stop();
    
    destination_ip_ = ip;
    destination_port_ = port;
    
    start();
}