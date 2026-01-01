#include "udp_streamer.h"
#include "util_logging.h"
#include <glib.h>
#include <sstream>
#include <cstring>

extern std::atomic<bool> g_running;

UDPStreamer::UDPStreamer(int width, int height, double fps)
    : pipeline_(nullptr), appsrc_(nullptr), loop_(nullptr),
      running_(false), width_(width), height_(height), fps_(fps), base_time_(0), last_pts_(0) {
}

UDPStreamer::~UDPStreamer() {
    stop();
}

bool UDPStreamer::start() {
    if (running_.load()) return true;
    
    if (!gst_is_initialized()) gst_init(nullptr, nullptr);
    
    setup_pipeline();
    
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        APP_LOG_ERROR("UDPStreamer: Failed to set pipeline to PLAYING state");
        return false;
    }
    
    running_.store(true);
    
    // Start the main loop in a thread
    server_thread_ = std::thread([this]() {
        g_main_loop_run(loop_);
    });
    
    APP_LOG_INFO("UDPStreamer: Started UDP stream on port 5000");
    return true;
}

void UDPStreamer::setup_pipeline() {
    // Pipeline: appsrc (h264) -> h264parse -> mpegtsmux -> udpsink
    std::string pipeline_string = 
        "appsrc name=src is-live=true format=time ! "
        "queue leaky=downstream max-size-buffers=2 ! "
        "h264parse ! "
        "mpegtsmux ! "
        "udpsink host=192.168.178.255 port=5000 sync=false";

    APP_LOG_INFO("UDPStreamer: Pipeline: " + pipeline_string);

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
    
    APP_LOG_INFO("UDPStreamer: Pipeline configured successfully (Raw -> x264enc -> UDP)");
}

void UDPStreamer::stop() {
    if (!running_.load()) return;
    
    running_.store(false);
    
    if (loop_) {
        g_main_loop_quit(loop_);
    }
    
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
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
    if (!running_.load() || !buffer || !buffer->data.data() || buffer->size == 0) {
        return;
    }
    
    APP_LOG_DEBUG("UDPStreamer: Pushing H264 frame, size=" + std::to_string(buffer->size));
    
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
    
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(src), gst_buffer);
    if (ret != GST_FLOW_OK) {
        APP_LOG_DEBUG("UDPStreamer: push failed: " + std::to_string(ret));
    } else {
        APP_LOG_DEBUG("UDPStreamer: Successfully pushed buffer to GStreamer, size=" + std::to_string(buffer->size));
    }
    
    gst_object_unref(src);
}