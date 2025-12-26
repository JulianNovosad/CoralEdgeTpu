#include "rtsp_server.h"
#include "util_logging.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <glib.h>

// Static callback function to handle media configuration
static void media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data) {
    RTSPServerWrapper *server_wrapper = static_cast<RTSPServerWrapper*>(user_data);
    
    APP_LOG_INFO("RTSP MEDIA CONFIGURE CALLBACK TRIGGERED - Pipeline construction successful, new client connecting");
    
    // Get the pipeline from the media
    GstElement *pipeline = gst_rtsp_media_get_element(media);
    
    // Get the appsrc element from the pipeline
    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "video_source");
    
    if (appsrc) {
        // Set the appsrc in the wrapper
        server_wrapper->set_appsrc(appsrc);
        
        // Configure appsrc properties to allow SDP generation
        // Use static caps to allow pipeline to preroll for SDP generation
        g_object_set(G_OBJECT(appsrc),
            "is-live", FALSE,  // Set to FALSE to allow pipeline preroll for SDP
            "format", GST_FORMAT_TIME,
            "stream-type", GST_APP_STREAM_TYPE_STREAM,
            "do-timestamp", TRUE,
            "block", FALSE,
            "min-latency", 0,
            "max-latency", 0,
            "produce-lateness", FALSE,
            NULL);
        
        // Set the caps for H.264 video with proper stream-format to match encoder output
        GstCaps *caps = gst_caps_new_simple("video/x-h264",
            "stream-format", G_TYPE_STRING, "avc",
            "alignment", G_TYPE_STRING, "au",
            "profile", G_TYPE_STRING, "constrained-baseline",
            "width", G_TYPE_INT, 1280,
            "height", G_TYPE_INT, 720,
            "framerate", GST_TYPE_FRACTION, 30, 1,
            NULL);
        g_object_set(G_OBJECT(appsrc), "caps", caps, NULL);
        gst_caps_unref(caps);
        
        // Skip dummy frame pushing - we'll rely on real frames from the application
        // The pipeline will preroll when the first real frame arrives
        APP_LOG_INFO("RTSP appsrc configured for live streaming - waiting for real frames from application");
        
        // Log caps negotiation details
        APP_LOG_INFO("RTSP appsrc configured with caps: video/x-h264, stream-format=avc, alignment=au, profile=constrained-baseline");
        
        APP_LOG_INFO("RTSP media configured with appsrc for client connection, preparing for SPS/PPS headers");
        
        // Log timestamp for first few client connections
        static int client_connection_counter = 0;
        if (client_connection_counter < 5) {
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            APP_LOG_INFO("CLIENT CONNECTION #" + std::to_string(client_connection_counter) + 
                        ": Timestamp=" + std::to_string(timestamp) + "ms");
            client_connection_counter++;
        }
        
        // First, send SPS/PPS headers to the new client if available
        server_wrapper->send_sps_pps_headers(appsrc);
        
        // Then flush any pending buffers that were queued before appsrc was available
        server_wrapper->flush_pending_buffers(appsrc);
        
        // Now switch appsrc to live mode for actual streaming
        g_object_set(G_OBJECT(appsrc), "is-live", TRUE, NULL);
        APP_LOG_INFO("RTSP appsrc switched to live mode for streaming");
        
        // Do not manually set pipeline state - let the RTSP server handle state transitions
        // The RTSP server will handle the state changes needed for SDP generation
        APP_LOG_INFO("Pipeline state management delegated to RTSP server for proper DESCRIBE handling");
    } else {
        APP_LOG_ERROR("Failed to get appsrc element from RTSP media pipeline");
    }
    
    // Set the media as reusable to allow proper pipeline setup for DESCRIBE
    gst_rtsp_media_set_reusable(media, TRUE);
    
    // Also make sure the media is properly configured for SDP generation
    gst_rtsp_media_set_shared(media, TRUE);
    
    // Release the pipeline reference
    gst_object_unref(pipeline);
}

RTSPServerWrapper::RTSPServerWrapper(int rtspPort, const std::string& streamName)
    : rtspPort_(rtspPort), streamName_(streamName), 
      server_(nullptr), mounts_(nullptr), factory_(nullptr), loop_(nullptr),
      timeout_source_(nullptr), appsrc_(nullptr), running_(false) {
}

RTSPServerWrapper::~RTSPServerWrapper() {
    stop();
}

bool RTSPServerWrapper::start() {
    if (running_) {
        APP_LOG_WARNING("RTSP server is already running");
        return true;
    }
    
    // Initialize GStreamer if not already initialized
    if (!gst_is_initialized()) {
        // Enable GStreamer debugging for RTSP
        gst_init(nullptr, nullptr);
        
        // Optionally enable debugging for specific categories
        gst_debug_set_active(true);
        gst_debug_set_threshold_from_string("rtspserver:5,rtph264pay:5,h264parse:5", true);
    }
    
    // Create the RTSP server
    server_ = gst_rtsp_server_new();
    if (!server_) {
        APP_LOG_ERROR("Failed to create RTSP server");
        return false;
    }
    
    // Set the port
    g_object_set(server_, "service", std::to_string(rtspPort_).c_str(), nullptr);
    
    // Set session timeout to a higher value to prevent connection drops
    // Default is typically around 30 seconds, setting to 0 means no timeout
    g_object_set(server_, "session-timeout", 0, nullptr);
    
    // Set maximum number of connections to handle multiple clients
    g_object_set(server_, "max-connections", 10, nullptr);
    
    // Enable socket reuse to prevent "Address already in use" errors
    g_object_set(server_, "reuse-socket", TRUE, nullptr);
    
    // Get the mounts
    mounts_ = gst_rtsp_server_get_mount_points(server_);
    if (!mounts_) {
        APP_LOG_ERROR("Failed to get mount points");
        return false;
    }
    
    // Create a media factory
    factory_ = gst_rtsp_media_factory_new();
    if (!factory_) {
        APP_LOG_ERROR("Failed to create media factory");
        return false;
    }
    
    // Set the launch string for the pipeline with better VLC compatibility
    // Use stream-format=avc for AVC1 format which is more compatible with VLC
    // config-interval=1 ensures SPS/PPS headers are sent with keyframes
    // aggregate-mode=zero-latency for immediate header delivery
    // mtu=1400 for proper packet sizing
    // pt=96 for RTP payload type
    APP_LOG_INFO("RTSP pipeline configured with: video/x-h264,stream-format=byte-stream,alignment=au,profile=constrained-baseline");
    APP_LOG_INFO("h264parse config-interval=1 for SPS/PPS delivery with keyframes");
    APP_LOG_INFO("rtph264pay config-interval=1,aggregate-mode=zero-latency,mtu=1400 for immediate RTP delivery");
    
    gst_rtsp_media_factory_set_launch(factory_, 
        "( appsrc name=video_source block=false format=GST_FORMAT_TIME do-timestamp=true stream-type=stream min-latency=0 max-latency=200000000 "
        "caps=video/x-h264,stream-format=avc,alignment=au,profile=constrained-baseline ! "
        "queue max-size-buffers=5 max-size-time=100000000 max-size-bytes=524288 leaky=downstream silent=false ! "
        "h264parse config-interval=1 parse-if-needed=true ! rtph264pay config-interval=1 aggregate-mode=zero-latency pt=96 mtu=1400 name=pay0 )");
    
    // Configure the factory
    gst_rtsp_media_factory_set_shared(factory_, TRUE); // Allow multiple clients
    gst_rtsp_media_factory_set_transport_mode(factory_, GST_RTSP_TRANSPORT_MODE_PLAY);
    
    // Set additional properties to prevent timeouts
    g_object_set(factory_, "stop-on-eos", FALSE, nullptr);
    g_object_set(factory_, "eos-on-shutdown", TRUE, nullptr);
    
    // Connect the media-configure signal to our callback
    g_signal_connect(factory_, "media-configure", G_CALLBACK(media_configure), this);
    
    // Set the media factory on the mount points
    gst_rtsp_mount_points_add_factory(mounts_, streamName_.c_str(), factory_);
    
    // Configure the factory with additional settings for mobile compatibility
    gst_rtsp_media_factory_set_shared(factory_, TRUE); // Allow multiple clients
    gst_rtsp_media_factory_set_transport_mode(factory_, GST_RTSP_TRANSPORT_MODE_PLAY);
    
    // Make media reusable to allow proper pipeline setup for DESCRIBE
    gst_rtsp_media_factory_set_shared(factory_, TRUE);
    
    // Use default protocols - TCP and UDP are typically supported by default

    // Release references
    g_object_unref(mounts_);
    
    // Create the main loop in the main thread
    loop_ = g_main_loop_new(nullptr, FALSE);
    if (!loop_) {
        APP_LOG_ERROR("Failed to create GMainLoop for RTSP server");
        return false;
    }
    
    // Attach the RTSP server to our main loop
    auto id = gst_rtsp_server_attach(server_, g_main_loop_get_context(loop_));
    if (id == 0) {
        APP_LOG_ERROR("Failed to attach RTSP server to main loop context");
        g_main_loop_unref(loop_);
        loop_ = nullptr;
        return false;
    }
    
    // Add a keep-alive timeout to prevent the server from entering a state where it recreates the socket
    timeout_source_ = g_timeout_source_new_seconds(10); // Add timeout every 10 seconds
    g_source_set_callback(timeout_source_, [](gpointer data) -> gboolean {
        // Keep the main loop active and prevent server timeout issues
        APP_LOG_DEBUG("RTSP server keep-alive ping");
        return G_SOURCE_CONTINUE; // Continue the timeout
    }, nullptr, nullptr);
    
    g_source_attach(timeout_source_, g_main_loop_get_context(loop_));
    
    // Start the server thread
    running_ = true;
    server_thread_ = std::thread(&RTSPServerWrapper::serverThread, this);
    
    APP_LOG_INFO("GStreamer RTSP server started successfully on port " + std::to_string(rtspPort_) + ", stream URL: rtsp://127.0.0.1:" + std::to_string(rtspPort_) + streamName_);
    APP_LOG_INFO("RTSP server is now accepting connections on rtsp://<your_ip>:" + std::to_string(rtspPort_) + streamName_);
    return true;
}

void RTSPServerWrapper::stop() {
    if (!running_) {
        return;
    }
    
    APP_LOG_INFO("Initiating RTSP server shutdown sequence...");
    running_ = false;
    
    // Stop and clean up the timeout source
    if (timeout_source_) {
        g_source_destroy(timeout_source_);
        g_source_unref(timeout_source_);
        timeout_source_ = nullptr;
    }
    
    // Stop the main loop if it exists
    if (loop_) {
        APP_LOG_INFO("Stopping GMainLoop...");
        g_main_loop_quit(loop_);
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }
    
    // Stop the server thread
    if (server_thread_.joinable()) {
        APP_LOG_INFO("Waiting for RTSP server thread to join...");
        server_thread_.join();
        APP_LOG_INFO("RTSP server thread joined successfully");
    }
    
    // Push a dummy buffer to appsrc to ensure clean shutdown if appsrc exists
    if (appsrc_) {
        // Push a small dummy buffer to unblock any waiting operations
        GstBuffer* dummy_buffer = gst_buffer_new_allocate(NULL, 1, NULL);
        if (dummy_buffer) {
            GstMapInfo map;
            gst_buffer_map(dummy_buffer, &map, GST_MAP_WRITE);
            memset(map.data, 0, 1);  // Fill with zeros
            gst_buffer_unmap(dummy_buffer, &map);
            
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), dummy_buffer);
            APP_LOG_INFO("Pushed dummy buffer to appsrc during shutdown, return: " + std::to_string(ret));
        }
        
        // Send end-of-stream to appsrc to signal the end
        GstFlowReturn eos_ret = gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
        APP_LOG_INFO("Sent end-of-stream to appsrc during shutdown, return: " + std::to_string(eos_ret));
    }
    
    // Transition all GStreamer elements to NULL state before unref
    if (appsrc_) {
        GstStateChangeReturn ret = gst_element_set_state(GST_ELEMENT(appsrc_), GST_STATE_NULL);
        APP_LOG_INFO("Appsrc state change to NULL: " + std::to_string(ret));
        gst_object_unref(appsrc_);
        appsrc_ = nullptr;
    }
    
    // Clean up GStreamer resources after thread stops
    if (factory_) {
        g_object_unref(factory_);
        factory_ = nullptr;
    }
    
    if (server_) {
        // Detach the server from the main context
        gst_rtsp_server_attach(server_, nullptr); // Detach from main context
        g_object_unref(server_);
        server_ = nullptr;
    }
    
    APP_LOG_INFO("GStreamer RTSP server stopped");
}

void RTSPServerWrapper::flush_pending_buffers(GstElement* appsrc) {
    if (!appsrc) {
        return;
    }
    
    std::vector<std::shared_ptr<H264Buffer>> local_pending_buffers;
    
    // Move pending buffers to local vector to minimize lock time
    {
        std::lock_guard<std::mutex> lock(pending_buffers_mutex_);
        local_pending_buffers = std::move(pending_buffers_);
        pending_buffers_.clear();
    }
    
    APP_LOG_INFO("Flushing " + std::to_string(local_pending_buffers.size()) + " pending buffers to appsrc");
    
    for (const auto& buffer : local_pending_buffers) {
        if (buffer && buffer->data.data() && buffer->size > 0) {
            // Create a GStreamer buffer from the H.264 data
            GstBuffer* gst_buffer = gst_buffer_new_allocate(nullptr, buffer->size, nullptr);
            if (gst_buffer) {
                GstMapInfo map;
                gst_buffer_map(gst_buffer, &map, GST_MAP_WRITE);
                memcpy(map.data, buffer->data.data(), buffer->size);
                gst_buffer_unmap(gst_buffer, &map);
                
                // Push the buffer to appsrc
                GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(appsrc), gst_buffer);
                if (ret != GST_FLOW_OK && ret != GST_FLOW_FLUSHING) {
                    APP_LOG_DEBUG("Failed to flush pending buffer to appsrc: " + std::to_string(ret));
                } else {
                    APP_LOG_DEBUG("Successfully flushed pending buffer of size " + std::to_string(buffer->size) + " to appsrc");
                }
            }
        }
    }
}

void RTSPServerWrapper::serverThread() {
    APP_LOG_INFO("GStreamer RTSP server thread started");
    
    // Run the main loop that was created in the start method
    if (loop_) {
        APP_LOG_INFO("GMainLoop exists, entering run loop...");
        g_main_loop_run(loop_);
        APP_LOG_INFO("GMainLoop run finished");
        // Note: The loop_ is unrefed in the stop() method
    } else {
        APP_LOG_ERROR("No GMainLoop available for RTSP server");
    }
    
    APP_LOG_INFO("GStreamer RTSP server thread stopped");
}

// Helper function to identify NAL unit type
static int get_nal_unit_type(const uint8_t* data, size_t size) {
    if (size < 5) return -1;  // Not enough data for a NAL unit header
    
    // Look for start code (0x00000001 or 0x000001)
    size_t start_offset = 0;
    if (size >= 4 && data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x00 && data[3] == 0x01) {
        start_offset = 4;
    } else if (size >= 3 && data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x01) {
        start_offset = 3;
    } else {
        return -1;  // No start code found
    }
    
    // NAL unit type is in the lower 5 bits of the byte after start code
    if (start_offset < size) {
        return (data[start_offset] & 0x1F);
    }
    return -1;
}

void RTSPServerWrapper::extract_and_store_headers(std::shared_ptr<H264Buffer> buffer) {
    if (!buffer || buffer->size < 5) return;
    
    int nal_type = get_nal_unit_type(buffer->data.data(), buffer->size);
    
    if (nal_type == 7) { // SPS (Sequence Parameter Set)
        std::lock_guard<std::mutex> lock(sps_pps_mutex_);
        sps_buffer_.clear();
        sps_buffer_.assign(buffer->data.data(), buffer->data.data() + buffer->size);
        APP_LOG_INFO("Stored SPS header of size " + std::to_string(buffer->size) + 
                    " at buffer addr: " + std::to_string(reinterpret_cast<uintptr_t>(buffer->data.data())));
        
        // Log first few bytes of SPS for verification
        if (buffer->size >= 8) {
            std::string sps_bytes = "";
            for (int i = 0; i < std::min(8, static_cast<int>(buffer->size)); i++) {
                char byte_str[4];
                snprintf(byte_str, sizeof(byte_str), "%02X ", static_cast<unsigned char>(buffer->data.data()[i]));
                sps_bytes += byte_str;
            }
            APP_LOG_DEBUG("SPS header bytes: " + sps_bytes);
        }
    } else if (nal_type == 8) { // PPS (Picture Parameter Set)
        std::lock_guard<std::mutex> lock(sps_pps_mutex_);
        pps_buffer_.clear();
        pps_buffer_.assign(buffer->data.data(), buffer->data.data() + buffer->size);
        APP_LOG_INFO("Stored PPS header of size " + std::to_string(buffer->size) + 
                    " at buffer addr: " + std::to_string(reinterpret_cast<uintptr_t>(buffer->data.data())));
        
        // Log first few bytes of PPS for verification
        if (buffer->size >= 8) {
            std::string pps_bytes = "";
            for (int i = 0; i < std::min(8, static_cast<int>(buffer->size)); i++) {
                char byte_str[4];
                snprintf(byte_str, sizeof(byte_str), "%02X ", static_cast<unsigned char>(buffer->data.data()[i]));
                pps_bytes += byte_str;
            }
            APP_LOG_DEBUG("PPS header bytes: " + pps_bytes);
        }
    } else if (nal_type == 5) { // IDR (Keyframe)
        std::lock_guard<std::mutex> lock(latest_keyframe_mutex_);
        latest_keyframe_buffer_.clear();
        latest_keyframe_buffer_.assign(buffer->data.data(), buffer->data.data() + buffer->size);
        APP_LOG_INFO("Stored latest keyframe (IDR) of size " + std::to_string(buffer->size));
        
        // Log first few bytes of keyframe for verification
        if (buffer->size >= 8) {
            std::string keyframe_bytes = "";
            for (int i = 0; i < std::min(8, static_cast<int>(buffer->size)); i++) {
                char byte_str[4];
                snprintf(byte_str, sizeof(byte_str), "%02X ", static_cast<unsigned char>(buffer->data.data()[i]));
                keyframe_bytes += byte_str;
            }
            APP_LOG_DEBUG("Keyframe (IDR) bytes: " + keyframe_bytes);
        }
    }
}

void RTSPServerWrapper::send_latest_keyframe(GstElement* appsrc) {
    if (!appsrc) {
        APP_LOG_WARNING("send_latest_keyframe called with null appsrc");
        return;
    }
    
    std::lock_guard<std::mutex> lock(latest_keyframe_mutex_);
    
    if (!latest_keyframe_buffer_.empty()) {
        GstBuffer* keyframe_buffer = gst_buffer_new_allocate(nullptr, latest_keyframe_buffer_.size(), nullptr);
        if (keyframe_buffer) {
            GstMapInfo map;
            gst_buffer_map(keyframe_buffer, &map, GST_MAP_WRITE);
            memcpy(map.data, latest_keyframe_buffer_.data(), latest_keyframe_buffer_.size());
            gst_buffer_unmap(keyframe_buffer, &map);
            
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(appsrc), keyframe_buffer);
            if (ret != GST_FLOW_OK && ret != GST_FLOW_FLUSHING) {
                APP_LOG_ERROR("Failed to push latest keyframe to appsrc: " + std::to_string(ret));
            } else {
                static int packet_sequence_counter = 2; // SPS=0, PPS=1, IDR=2 (assuming we already sent SPS/PPS)
                APP_LOG_INFO("RTP PACKET #" + std::to_string(packet_sequence_counter) + 
                           ": Successfully pushed latest keyframe of size " + std::to_string(latest_keyframe_buffer_.size()) + " to new client");
                packet_sequence_counter++;
                
                // Check if this is an IDR frame by examining NAL unit type
                if (latest_keyframe_buffer_.size() >= 5) {
                    uint8_t nal_type = 0;
                    // Look for start code and extract NAL type
                    size_t start_offset = 0;
                    if (latest_keyframe_buffer_.size() >= 4 && 
                        latest_keyframe_buffer_.data()[0] == 0x00 && latest_keyframe_buffer_.data()[1] == 0x00 && 
                        latest_keyframe_buffer_.data()[2] == 0x00 && latest_keyframe_buffer_.data()[3] == 0x01) {
                        start_offset = 4;
                    } else if (latest_keyframe_buffer_.size() >= 3 && 
                               latest_keyframe_buffer_.data()[0] == 0x00 && latest_keyframe_buffer_.data()[1] == 0x00 && 
                               latest_keyframe_buffer_.data()[2] == 0x01) {
                        start_offset = 3;
                    } else {
                        // No start code, assume first byte is NAL header
                        start_offset = 0;
                    }
                    
                    if (start_offset < latest_keyframe_buffer_.size()) {
                        nal_type = latest_keyframe_buffer_.data()[start_offset] & 0x1F;  // Get lower 5 bits for NAL unit type
                    }
                    
                    const char* nal_type_str = "Unknown";
                    bool is_idr = false;
                    switch (nal_type) {
                        case 5: 
                            nal_type_str = "IDR-Slice"; 
                            is_idr = true;
                            break;
                        case 1: nal_type_str = "P-Slice"; break;
                        case 7: nal_type_str = "SPS"; break;
                        case 8: nal_type_str = "PPS"; break;
                        case 6: nal_type_str = "SEI"; break;
                        default: break;
                    }
                    
                    APP_LOG_INFO("Keyframe is " + std::string(nal_type_str) + " (type " + std::to_string(nal_type) + "), IDR: " + std::string(is_idr ? "YES" : "NO"));
                    
                    // Verify start codes in IDR frame
                    bool has_start_code = false;
                    if (latest_keyframe_buffer_.size() >= 4) {
                        if (latest_keyframe_buffer_.data()[0] == 0x00 && latest_keyframe_buffer_.data()[1] == 0x00 && 
                            latest_keyframe_buffer_.data()[2] == 0x00 && latest_keyframe_buffer_.data()[3] == 0x01) {
                            has_start_code = true;
                        } else if (latest_keyframe_buffer_.size() >= 3 && 
                                   latest_keyframe_buffer_.data()[0] == 0x00 && latest_keyframe_buffer_.data()[1] == 0x00 && 
                                   latest_keyframe_buffer_.data()[2] == 0x01) {
                            has_start_code = true;
                        }
                    }
                    APP_LOG_INFO("IDR frame has valid start code: " + std::string(has_start_code ? "YES" : "NO"));
                    
                    // Log complete IDR frame header for verification (first 32 bytes)
                    std::string idr_full = "";
                    size_t log_size = std::min(static_cast<size_t>(32), latest_keyframe_buffer_.size());
                    for (size_t i = 0; i < log_size; i++) {
                        char byte_str[4];
                        snprintf(byte_str, sizeof(byte_str), "%02X ", static_cast<unsigned char>(latest_keyframe_buffer_.data()[i]));
                        idr_full += byte_str;
                    }
                    APP_LOG_DEBUG("Complete IDR frame header (first " + std::to_string(log_size) + " bytes): " + idr_full);
                }
            }
        } else {
            APP_LOG_ERROR("Failed to allocate GstBuffer for keyframe");
        }
    } else {
        APP_LOG_WARNING("No keyframe available to send to new client - client may experience delayed playback until first annotated keyframe arrives");
        // Do NOT send dummy keyframes - only send real annotated frames from the detection/annotation pipeline
        // This ensures clients receive actual annotated video content, not dummy frames
        APP_LOG_INFO("Waiting for first real annotated keyframe from detection pipeline before sending to new client");
    }
}

void RTSPServerWrapper::send_sps_pps_headers(GstElement* appsrc) {
    if (!appsrc) {
        APP_LOG_WARNING("send_sps_pps_headers called with null appsrc");
        return;
    }
    
    std::lock_guard<std::mutex> lock(sps_pps_mutex_);
    
    APP_LOG_INFO("Sending SPS/PPS headers to new client...");
    
    // Track the sequence of RTP packets for the new client
    static int packet_sequence_counter = 0;
    
    // Send SPS first
    if (!sps_buffer_.empty()) {
        GstBuffer* sps_buffer = gst_buffer_new_allocate(nullptr, sps_buffer_.size(), nullptr);
        if (sps_buffer) {
            GstMapInfo map;
            gst_buffer_map(sps_buffer, &map, GST_MAP_WRITE);
            memcpy(map.data, sps_buffer_.data(), sps_buffer_.size());
            gst_buffer_unmap(sps_buffer, &map);
            
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(appsrc), sps_buffer);
            if (ret != GST_FLOW_OK && ret != GST_FLOW_FLUSHING) {
                APP_LOG_ERROR("Failed to push SPS header to appsrc: " + std::to_string(ret));
            } else {
                APP_LOG_INFO("RTP PACKET #" + std::to_string(packet_sequence_counter) + 
                           ": Successfully pushed SPS header of size " + std::to_string(sps_buffer_.size()) + " to new client");
                packet_sequence_counter++;
                
                // Log complete SPS header for verification (first 32 bytes to see full header)
                std::string sps_full = "";
                size_t log_size = std::min(static_cast<size_t>(32), sps_buffer_.size());
                for (size_t i = 0; i < log_size; i++) {
                    char byte_str[4];
                    snprintf(byte_str, sizeof(byte_str), "%02X ", static_cast<unsigned char>(sps_buffer_.data()[i]));
                    sps_full += byte_str;
                }
                APP_LOG_DEBUG("Complete SPS header (first " + std::to_string(log_size) + " bytes): " + sps_full);
                
                // Verify start codes in SPS
                if (sps_buffer_.size() >= 4) {
                    bool has_start_code = false;
                    if (sps_buffer_.data()[0] == 0x00 && sps_buffer_.data()[1] == 0x00 && sps_buffer_.data()[2] == 0x00 && sps_buffer_.data()[3] == 0x01) {
                        has_start_code = true;
                    } else if (sps_buffer_.size() >= 3 && sps_buffer_.data()[0] == 0x00 && sps_buffer_.data()[1] == 0x00 && sps_buffer_.data()[2] == 0x01) {
                        has_start_code = true;
                    }
                    APP_LOG_INFO("SPS header has valid start code: " + std::string(has_start_code ? "YES" : "NO"));
                }
            }
        } else {
            APP_LOG_ERROR("Failed to allocate GstBuffer for SPS header");
        }
    } else {
        APP_LOG_WARNING("No SPS header available to send to new client");
    }
    
    // Send PPS next
    if (!pps_buffer_.empty()) {
        GstBuffer* pps_buffer = gst_buffer_new_allocate(nullptr, pps_buffer_.size(), nullptr);
        if (pps_buffer) {
            GstMapInfo map;
            gst_buffer_map(pps_buffer, &map, GST_MAP_WRITE);
            memcpy(map.data, pps_buffer_.data(), pps_buffer_.size());
            gst_buffer_unmap(pps_buffer, &map);
            
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(appsrc), pps_buffer);
            if (ret != GST_FLOW_OK && ret != GST_FLOW_FLUSHING) {
                APP_LOG_ERROR("Failed to push PPS header to appsrc: " + std::to_string(ret));
            } else {
                APP_LOG_INFO("RTP PACKET #" + std::to_string(packet_sequence_counter) + 
                           ": Successfully pushed PPS header of size " + std::to_string(pps_buffer_.size()) + " to new client");
                packet_sequence_counter++;
                
                // Log complete PPS header for verification (first 32 bytes to see full header)
                std::string pps_full = "";
                size_t log_size = std::min(static_cast<size_t>(32), pps_buffer_.size());
                for (size_t i = 0; i < log_size; i++) {
                    char byte_str[4];
                    snprintf(byte_str, sizeof(byte_str), "%02X ", static_cast<unsigned char>(pps_buffer_.data()[i]));
                    pps_full += byte_str;
                }
                APP_LOG_DEBUG("Complete PPS header (first " + std::to_string(log_size) + " bytes): " + pps_full);
                
                // Verify start codes in PPS
                if (pps_buffer_.size() >= 4) {
                    bool has_start_code = false;
                    if (pps_buffer_.data()[0] == 0x00 && pps_buffer_.data()[1] == 0x00 && pps_buffer_.data()[2] == 0x00 && pps_buffer_.data()[3] == 0x01) {
                        has_start_code = true;
                    } else if (pps_buffer_.size() >= 3 && pps_buffer_.data()[0] == 0x00 && pps_buffer_.data()[1] == 0x00 && pps_buffer_.data()[2] == 0x01) {
                        has_start_code = true;
                    }
                    APP_LOG_INFO("PPS header has valid start code: " + std::string(has_start_code ? "YES" : "NO"));
                }
            }
        } else {
            APP_LOG_ERROR("Failed to allocate GstBuffer for PPS header");
        }
    } else {
        APP_LOG_WARNING("No PPS header available to send to new client");
    }
    
    // Send the latest keyframe after headers to ensure immediate playback
    // This is crucial for VLC to start decoding immediately
    send_latest_keyframe(appsrc);
    
    if (sps_buffer_.empty() && pps_buffer_.empty()) {
        APP_LOG_ERROR("No SPS or PPS headers available! New client will not be able to decode video.");
    }
}

void RTSPServerWrapper::pushH264Data(std::shared_ptr<H264Buffer> buffer) {
    if (!running_ || !buffer || !buffer->data.data() || buffer->size == 0) {
        return;
    }
    
    // Track frame rate
    static auto last_log_time = std::chrono::steady_clock::now();
    static int frame_count = 0;
    frame_count++;
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count();
    if (elapsed >= 1000) { // Log every second
        APP_LOG_INFO("RTSP Server: Pushing " + std::to_string(frame_count) + " ANNOTATED frames in last second");
        frame_count = 0;
        last_log_time = now;
    }
    
    // Extract and store SPS/PPS headers if present
    extract_and_store_headers(buffer);
    
    // Log NAL unit information
    if (buffer->size >= 5) {  // Need at least 5 bytes to check NAL header
        uint8_t nal_type = 0;
        // Look for start code (0x00000001 or 0x000001)
        size_t start_offset = 0;
        if (buffer->size >= 4 && 
            buffer->data[0] == 0x00 && buffer->data[1] == 0x00 && 
            buffer->data[2] == 0x00 && buffer->data[3] == 0x01) {
            start_offset = 4;
        } else if (buffer->size >= 3 && 
                   buffer->data[0] == 0x00 && buffer->data[1] == 0x00 && 
                   buffer->data[2] == 0x01) {
            start_offset = 3;
        } else {
            // No start code, assume first byte is NAL header
            start_offset = 0;
        }
        
        if (start_offset < buffer->size) {
            nal_type = buffer->data[start_offset] & 0x1F;  // Get lower 5 bits for NAL unit type
        }
        
        const char* nal_type_str = "Unknown";
        switch (nal_type) {
            case 1: nal_type_str = "P-Slice"; break;
            case 5: nal_type_str = "IDR-Slice"; break;  // Keyframe
            case 7: nal_type_str = "SPS"; break;        // Sequence Parameter Set
            case 8: nal_type_str = "PPS"; break;        // Picture Parameter Set
            case 6: nal_type_str = "SEI"; break;
            default: break;
        }
        
        // Log every frame type with details
        (void)nal_type_str; // Suppress unused variable warning
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        (void)timestamp; // Suppress unused variable warning
        APP_LOG_DEBUG("RTSP Push: NAL type: " + std::string(nal_type_str) + 
                     " (" + std::to_string(nal_type) + ")" +
                     ", Size: " + std::to_string(buffer->size) + 
                     ", Timestamp: " + std::to_string(timestamp) + "ms");
        
        // Save first few NAL units to file for inspection
        static int nal_dump_counter = 0;
        if (nal_dump_counter < 5) {  // Save first 5 NAL units
            std::string filename = "/tmp/nal_unit_" + std::to_string(nal_dump_counter) + "_" + nal_type_str + ".h264";
            FILE* file = fopen(filename.c_str(), "wb");
            if (file) {
                fwrite(buffer->data.data(), 1, buffer->size, file);
                fclose(file);
                APP_LOG_INFO("Saved NAL unit to file: " + filename);
            }
            nal_dump_counter++;
        }
    }
    
#ifdef DEBUG_MODE
    // Record RTSP push start time
    auto rtsp_push_start_time = std::chrono::high_resolution_clock::now();
#endif
    
    GstElement* current_appsrc = get_appsrc();
    if (current_appsrc) {
        // Validate buffer size before pushing to appsrc
        if (buffer->size == 0) {
            APP_LOG_WARNING("Skipping zero-size buffer push to appsrc");
            return;
        }
        
        // Create a GStreamer buffer from the H.264 data
        GstBuffer* gst_buffer = gst_buffer_new_allocate(nullptr, buffer->size, nullptr);
        if (gst_buffer) {
            GstMapInfo map;
            gst_buffer_map(gst_buffer, &map, GST_MAP_WRITE);
            memcpy(map.data, buffer->data.data(), buffer->size);
            gst_buffer_unmap(gst_buffer, &map);
            
            // Set buffer timestamp and duration if available in the original data
            // For now, we'll log the current time as a reference
            GstClockTime pts = GST_CLOCK_TIME_NONE;
            GstClockTime dts = GST_CLOCK_TIME_NONE;
            
            // Set PTS if available (using current time as a reference)
            pts = gst_util_uint64_scale(static_cast<uint64_t>(g_get_monotonic_time()) * GST_USECOND, 1, 1);
            GST_BUFFER_PTS(gst_buffer) = pts;
            GST_BUFFER_DTS(gst_buffer) = dts;  // DTS is typically not used for H.264
            
            // Log detailed buffer information before pushing
            APP_LOG_INFO("H264 BUFFER PUSH: Size=" + std::to_string(buffer->size) + 
                        " bytes, PTS=" + std::to_string(pts) + 
                        " (timestamp: " + std::to_string(GST_TIME_AS_MSECONDS(pts)) + "ms)");
            
            // Get appsrc caps to verify format
            GstCaps* caps = gst_app_src_get_caps(GST_APP_SRC_CAST(current_appsrc));
            if (caps) {
                gchar* caps_str = gst_caps_to_string(caps);
                APP_LOG_DEBUG("APP SRC CAPS: " + std::string(caps_str));
                g_free(caps_str);
                gst_caps_unref(caps);
            } else {
                APP_LOG_DEBUG("APP SRC CAPS: None");
            }
            
            // Push the buffer to appsrc
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(current_appsrc), gst_buffer);
            if (ret != GST_FLOW_OK && ret != GST_FLOW_FLUSHING) {
                APP_LOG_ERROR("Failed to push buffer to appsrc: " + std::to_string(ret));
            } else {
                APP_LOG_INFO("SUCCESS: Pushed buffer of size " + std::to_string(buffer->size) + 
                           " bytes to appsrc, return: " + std::to_string(ret));
            }
        }
    } else {
        // No appsrc available yet, store in pending buffers for when a client connects
        std::lock_guard<std::mutex> lock(pending_buffers_mutex_);
        pending_buffers_.push_back(buffer);
        
        // Limit the size of pending buffers to prevent memory issues
        if (pending_buffers_.size() > 100) {  // Keep only the most recent 100 frames
            pending_buffers_.erase(pending_buffers_.begin());
        }
        
        APP_LOG_DEBUG("No appsrc available, buffering frame of size " + std::to_string(buffer->size) + " (pending buffer size: " + std::to_string(pending_buffers_.size()) + ")");
    }
    
#ifdef DEBUG_MODE
    // Record RTSP push end time
    auto rtsp_push_end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(rtsp_push_end_time - rtsp_push_start_time).count();
    APP_LOG_DEBUG("RTSP push time: " + std::to_string(duration_us) + " us");
#endif
}