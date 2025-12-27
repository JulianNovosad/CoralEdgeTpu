#include "rtsp_server.h"
#include "util_logging.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <glib.h>

#include <unistd.h>
#include <arpa/inet.h>
#include <atomic>

extern std::atomic<bool> shutdown_requested;

// Static helper to wait for port to be listening
static bool wait_for_port_ready(int port, int timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();
    while (!shutdown_requested.load()) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return false;

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        // Use loopback to check
        if (inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) <= 0) {
            close(sock);
            return false;
        }

        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            close(sock);
            return true;
        }
        close(sock);

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() > timeout_ms) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

// Static member definitions
std::mutex RTSPServerWrapper::port_binding_mutex_;
std::atomic<bool> RTSPServerWrapper::port_in_use_{false};
std::mutex RTSPServerWrapper::camera_access_mutex_;
std::atomic<bool> RTSPServerWrapper::camera_in_use_{false};
std::atomic<int> RTSPServerWrapper::active_client_count_{0};

// Static callback function to handle media configuration
static void media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data) {
    RTSPServerWrapper *server_wrapper = static_cast<RTSPServerWrapper*>(user_data);
    
    std::cerr << "RTSP: media_configure callback triggered" << std::endl;
    APP_LOG_INFO("RTSP MEDIA CONFIGURE CALLBACK TRIGGERED - Pipeline construction successful, new client connecting");
    APP_LOG_INFO("RTSP MEDIA CONFIGURE: Client RTSP session established, preparing pipeline");
    
    // Get the pipeline from the media
    GstElement *pipeline = gst_rtsp_media_get_element(media);
    if (!pipeline) {
        std::cerr << "RTSP: Failed to get pipeline from media" << std::endl;
        return;
    }

    g_signal_connect(pipeline, "deep-notify", G_CALLBACK(+[](GstObject *self, GstObject *prop_parent, GParamSpec *pspec, gpointer user_data) {
        if (std::string(pspec->name) == "state") {
            GstState state;
            GstState pending;
            g_object_get(prop_parent, "state", &state, "pending", &pending, NULL);
            std::string state_name = gst_element_state_get_name(state);
            std::string pending_name = gst_element_state_get_name(pending);
            APP_LOG_INFO("RTSP Pipeline Element State Change: " + std::string(GST_OBJECT_NAME(prop_parent)) + 
                        " -> " + state_name + " (pending: " + pending_name + ")");
        }
    }), server_wrapper);  // Pass server_wrapper as user_data
    
    // Get the appsrc element from the pipeline
    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "video_source");
    
    if (appsrc) {
        APP_LOG_INFO("RTSP: Found appsrc 'video_source'");
        // Set the appsrc in the wrapper
        server_wrapper->set_appsrc(appsrc);
        
        // Configure appsrc properties for live streaming with proper timing
        g_object_set(G_OBJECT(appsrc),
            "is-live", TRUE,
            "format", GST_FORMAT_TIME,
            "stream-type", GST_APP_STREAM_TYPE_STREAM,
            "do-timestamp", TRUE,
            "min-latency", 0,
            "max-latency", 200000000, // 200ms max latency
            NULL);
            
        // Explicitly set caps on appsrc to match encoder settings
        // This ensures h264parse knows the resolution/framerate even if SPS/PPS are delayed
        GstCaps *caps = gst_caps_new_simple("video/x-h264",
            "stream-format", G_TYPE_STRING, "byte-stream",
            "alignment", G_TYPE_STRING, "nal",
            "parsed", G_TYPE_BOOLEAN, TRUE,
            "profile", G_TYPE_STRING, "constrained-baseline",
            "width", G_TYPE_INT, 1536,
            "height", G_TYPE_INT, 864,
            "framerate", GST_TYPE_FRACTION, 40, 1,
            NULL);
        gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
        gst_caps_unref(caps);
        
        // Flush any pending buffers that accumulated before the appsrc was ready
        server_wrapper->flush_pending_buffers();
        

        
        // Note: Caps are already set in the pipeline string, but we set them here too for appsrc
        // The pipeline string now forces byte-stream/nal after h264parse
        APP_LOG_INFO("RTSP: appsrc configured with is-live=TRUE, caps are set to byte-stream/nal");
        
        // Get and configure h264parse element for proper SPS/PPS delivery
        GstElement *h264parse = nullptr;
        GstIterator *it = gst_bin_iterate_elements(GST_BIN(pipeline));
        GValue value = G_VALUE_INIT;
        
        while (gst_iterator_next(it, &value)) {
            GstElement *child = GST_ELEMENT(g_value_get_object(&value));
            const gchar *name = GST_OBJECT_NAME(child);
            if (name && g_str_has_prefix(name, "h264parse")) {
                h264parse = child;
                gst_object_ref(h264parse); // Add ref since we'll unref later
                break;
            }
            g_value_reset(&value);
        }
        g_value_unset(&value);
        gst_iterator_free(it);
        
        if (h264parse) {
            g_object_set(h264parse, "config-interval", 1, nullptr);
            APP_LOG_INFO("RTSP: h264parse configured with config-interval=1");
            gst_object_unref(h264parse);
        } else {
            APP_LOG_WARNING("RTSP: Could not find h264parse element to configure for SPS/PPS delivery");
        }
        
        // Process any pending clients now that appsrc is ready
        std::vector<GstRTSPClient*> pending_clients = server_wrapper->take_pending_clients();
        for (GstRTSPClient* client : pending_clients) {
            APP_LOG_INFO("Processing pending client after appsrc is ready");
            // Send SPS/PPS headers to the new client
            server_wrapper->send_sps_pps_headers();
            // Send the latest keyframe to allow immediate video playback
            server_wrapper->send_latest_keyframe();
            // Unref the client since we took ownership from the queue
            gst_object_unref(client);
        }
    } else {
        std::cerr << "RTSP ERROR: Failed to get appsrc element from RTSP media pipeline" << std::endl;
        APP_LOG_ERROR("Failed to get appsrc element from RTSP media pipeline");
    }
    
    // Set the media as reusable to allow proper pipeline setup for DESCRIBE
    gst_rtsp_media_set_reusable(media, TRUE);
    
    // Also make sure the media is properly configured for SDP generation
    gst_rtsp_media_set_shared(media, TRUE);
    
    // Release the pipeline reference
    gst_object_unref(pipeline);
}

void RTSPServerWrapper::client_connected_cb(GstRTSPServer *server, GstRTSPClient *client, gpointer user_data) {
    APP_LOG_INFO("RTSP CLIENT_CONNECTED_CB: New client attempting connection");
    GstRTSPConnection *connection = gst_rtsp_client_get_connection(client);
    if (connection) {
        const gchar *ip = gst_rtsp_connection_get_ip(connection);
        APP_LOG_INFO("RTSP CLIENT_CONNECTED_CB: Client from IP=" + std::string(ip ? ip : "Unknown"));
    }
    
    RTSPServerWrapper *server_wrapper = static_cast<RTSPServerWrapper*>(user_data);
    if (server_wrapper) {
        server_wrapper->manage_client_connection(client);
    }
    
    // Connect to client closed signal
    g_signal_connect(client, "closed", G_CALLBACK(client_closed_cb), user_data);
}

void RTSPServerWrapper::client_closed_cb(GstRTSPClient *client, gpointer user_data) {
    RTSPServerWrapper *server_wrapper = static_cast<RTSPServerWrapper*>(user_data);
    (void)server_wrapper; // Unused for now

    GstRTSPConnection *connection = gst_rtsp_client_get_connection(client);
    if (connection) {
        const gchar *ip = gst_rtsp_connection_get_ip(connection);
        std::cerr << "RTSP: unmanage_client (disconnected) from " << (ip ? ip : "Unknown") << std::endl;
        APP_LOG_INFO("RTSP UNMANAGE_CLIENT (DISCONNECTED): IP=" + std::string(ip ? ip : "Unknown"));
    } else {
        std::cerr << "RTSP: unmanage_client (disconnected) from Unknown Connection" << std::endl;
        APP_LOG_INFO("RTSP UNMANAGE_CLIENT (DISCONNECTED): Unknown Connection");
    }
}

RTSPServerWrapper::RTSPServerWrapper(int rtspPort, const std::string& streamName)
    : rtspPort_(rtspPort), streamName_(streamName), 
      server_(nullptr), mounts_(nullptr), factory_(nullptr), loop_(nullptr),
      timeout_source_(nullptr), appsrc_(nullptr), running_(false) {
}

void RTSPServerWrapper::set_appsrc(GstElement* appsrc) {
    std::lock_guard<std::mutex> lock(appsrc_mutex_);
    if (appsrc_ != nullptr) {
        gst_object_unref(appsrc_);  // Release old reference
    }
    appsrc_ = appsrc;
    if (appsrc_ != nullptr) {
        gst_object_ref(appsrc);  // Take ownership reference
    }
}

GstElement* RTSPServerWrapper::get_appsrc() const {
    std::lock_guard<std::mutex> lock(appsrc_mutex_);
    if (appsrc_ != nullptr) {
        gst_object_ref(appsrc_);  // Caller must unref
        return appsrc_;
    }
    return nullptr;
}

RTSPServerWrapper::~RTSPServerWrapper() {
    stop();
    // Clean up appsrc reference
    {
        std::lock_guard<std::mutex> lock(appsrc_mutex_);
        if (appsrc_) {
            gst_object_unref(appsrc_);
            appsrc_ = nullptr;
        }
    }
}

bool RTSPServerWrapper::start() {
    if (running_) {
        APP_LOG_WARNING("RTSP server is already running");
        return true;
    }
    
    // Internal cleanup: stop any running detector or RTSP server before starting
    internal_cleanup();
    
    // Check if port is available using binary lock
    if (!is_port_available(rtspPort_)) {
        APP_LOG_ERROR("RTSP server port " + std::to_string(rtspPort_) + " is already in use by another process");
        return false;
    }
    
    // Acquire the port binding lock
    std::lock_guard<std::mutex> lock(port_binding_mutex_);
    if (port_in_use_.load()) {
        APP_LOG_ERROR("RTSP server port " + std::to_string(rtspPort_) + " is already in use by another instance");
        return false;
    }
    port_in_use_.store(true);
    
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
    
    // Set the address and port explicitly
    gst_rtsp_server_set_address(server_, "0.0.0.0");
    gst_rtsp_server_set_service(server_, std::to_string(rtspPort_).c_str());
    
    // Configure session management for reliable client handling
    GstRTSPSessionPool *session_pool = gst_rtsp_server_get_session_pool(server_);
    if (session_pool) {
        gst_rtsp_session_pool_set_max_sessions(session_pool, 10);  // Allow up to 10 concurrent sessions
        g_object_unref(session_pool);
    }
    
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
    // Use explicit caps to ensure stable negotiation between appsrc and h264parse.
    APP_LOG_INFO("RTSP pipeline configured with: appsrc ! h264parse ! caps ! rtph264pay");
    
    gst_rtsp_media_factory_set_launch(factory_, 
        "( appsrc name=video_source is-live=true block=false format=GST_FORMAT_TIME do-timestamp=true stream-type=stream ! "
        "video/x-h264,stream-format=byte-stream,alignment=nal,parsed=true ! "
        "h264parse ! "
        "video/x-h264,stream-format=byte-stream,alignment=nal ! "
        "rtph264pay name=pay0 config-interval=1 pt=96 )");
    

    
    // Configure the factory
    gst_rtsp_media_factory_set_shared(factory_, TRUE); // Allow multiple clients
    gst_rtsp_media_factory_set_suspend_mode(factory_, GST_RTSP_SUSPEND_MODE_NONE); // Keep pipeline running
    gst_rtsp_media_factory_set_transport_mode(factory_, GST_RTSP_TRANSPORT_MODE_PLAY);
    
    // Set supported transport protocols for maximum client compatibility
    gst_rtsp_media_factory_set_protocols(factory_, 
        (GstRTSPLowerTrans)(GST_RTSP_LOWER_TRANS_TCP | GST_RTSP_LOWER_TRANS_UDP));
    
    // Connect the media-configure signal to our callback
    g_signal_connect(factory_, "media-configure", G_CALLBACK(media_configure), this);
    
    // Set the media factory on the mount points
    gst_rtsp_mount_points_add_factory(mounts_, streamName_.c_str(), factory_);
    
    // Connect the client-connected signal
    g_signal_connect(server_, "client-connected", G_CALLBACK(client_connected_cb), this);
    
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
    
    APP_LOG_INFO("RTSP server successfully attached to main loop with ID: " + std::to_string(id));
    APP_LOG_INFO("RTSP SERVER: GMainLoop running, server ready to accept connections on port " + std::to_string(rtspPort_));
    
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
        // Don't unref here yet, wait for thread to finish using it
    }
    
    // Stop the server thread
    if (server_thread_.joinable()) {
        APP_LOG_INFO("Waiting for RTSP server thread to join...");
        server_thread_.join();
        APP_LOG_INFO("RTSP server thread joined successfully");
    }

    // Now it's safe to unref the loop
    if (loop_) {
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }
    
    // Push a dummy buffer to appsrc to ensure clean shutdown if appsrc exists
    GstElement* current_appsrc = get_appsrc();
    if (current_appsrc) {
        // Push a small dummy buffer to unblock any waiting operations
        GstBuffer* dummy_buffer = gst_buffer_new_allocate(NULL, 1, NULL);
        if (dummy_buffer) {
            GstMapInfo map;
            gst_buffer_map(dummy_buffer, &map, GST_MAP_WRITE);
            memset(map.data, 0, 1);  // Fill with zeros
            gst_buffer_unmap(dummy_buffer, &map);
            
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(current_appsrc), dummy_buffer);
            APP_LOG_INFO("Pushed dummy buffer to appsrc during shutdown, return: " + std::to_string(ret));
        }
        
        // Send end-of-stream to appsrc to signal the end
        GstFlowReturn eos_ret = gst_app_src_end_of_stream(GST_APP_SRC(current_appsrc));
        APP_LOG_INFO("Sent end-of-stream to appsrc during shutdown, return: " + std::to_string(eos_ret));
        
        gst_object_unref(current_appsrc); // Release the reference from get_appsrc()
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
    
    // Release the port binding lock
    port_in_use_.store(false);
    
    APP_LOG_INFO("GStreamer RTSP server stopped");
}

void RTSPServerWrapper::flush_pending_buffers() {
    GstElement* current_appsrc = get_appsrc();
    if (!current_appsrc) {
        APP_LOG_INFO("flush_pending_buffers: No appsrc available, skipping flush");
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
                
                // Check the current state of the appsrc for logging purposes
                GstState appsrc_state, pending_state;
                GstStateChangeReturn state_ret = gst_element_get_state(current_appsrc, &appsrc_state, &pending_state, GST_CLOCK_TIME_NONE);
                std::string state_info = "";
                if (state_ret == GST_STATE_CHANGE_SUCCESS) {
                    state_info = " (appsrc state: " + std::string(gst_element_state_get_name(appsrc_state)) + ")";
                }
                
                // Push the buffer to appsrc
                GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(current_appsrc), gst_buffer);
                if (ret != GST_FLOW_OK && ret != GST_FLOW_FLUSHING) {
                    APP_LOG_DEBUG("Failed to flush pending buffer to appsrc: " + std::to_string(ret) + state_info);
                } else {
                    APP_LOG_DEBUG("Successfully flushed pending buffer of size " + std::to_string(buffer->size) + " to appsrc" + state_info);
                }
            }
        }
    }
    
    gst_object_unref(current_appsrc); // Release the reference from get_appsrc()
}

void RTSPServerWrapper::serverThread() {
    APP_LOG_INFO("GStreamer RTSP server thread started");
    
    // Run the main loop that was created in the start method
    if (loop_) {
        APP_LOG_INFO("GMainLoop exists, entering run loop...");
        APP_LOG_INFO("CRITICAL: GMainLoop is now RUNNING.");
        
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

void RTSPServerWrapper::send_latest_keyframe() {
    GstElement* current_appsrc = get_appsrc();
    if (!current_appsrc) {
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
            
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(current_appsrc), keyframe_buffer);
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
    
    gst_object_unref(current_appsrc); // Release the reference from get_appsrc()
}

void RTSPServerWrapper::send_sps_pps_headers() {
    GstElement* current_appsrc = get_appsrc();
    if (!current_appsrc) {
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
            
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(current_appsrc), sps_buffer);
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
            
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(current_appsrc), pps_buffer);
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
    send_latest_keyframe();
    
    if (sps_buffer_.empty() && pps_buffer_.empty()) {
        APP_LOG_ERROR("No SPS or PPS headers available! New client will not be able to decode video.");
    }
    
    gst_object_unref(current_appsrc); // Release the reference from get_appsrc()
}

bool RTSPServerWrapper::is_port_available(int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        return false;
    }
    
    // Set socket options to allow reuse
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    int result = bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    close(sock);
    
    return (result == 0);
}

bool RTSPServerWrapper::is_appsrc_ready() const {
    std::lock_guard<std::mutex> lock(appsrc_mutex_);
    return appsrc_ != nullptr;
}

void RTSPServerWrapper::add_pending_client(GstRTSPClient* client) {
    if (!client) return;
    std::lock_guard<std::mutex> lock(pending_clients_mutex_);
    g_object_ref(client);  // Take reference
    pending_clients_.push_back(client);
}

std::vector<GstRTSPClient*> RTSPServerWrapper::take_pending_clients() {
    std::lock_guard<std::mutex> lock(pending_clients_mutex_);
    std::vector<GstRTSPClient*> clients;
    std::swap(clients, pending_clients_);  // Move ownership
    return clients;
}

size_t RTSPServerWrapper::pending_client_count() const {
    std::lock_guard<std::mutex> lock(pending_clients_mutex_);
    return pending_clients_.size();
}

void RTSPServerWrapper::manage_client_connection(GstRTSPClient* client) {
    GstRTSPConnection *connection = gst_rtsp_client_get_connection(client);
    if (connection) {
        const gchar *ip = gst_rtsp_connection_get_ip(connection);
        std::string client_ip = ip ? ip : "Unknown";
        APP_LOG_INFO("RTSP MANAGE_CLIENT: Client from " + std::string(client_ip));
        
        // Add all clients to pending queue - they will be processed when media is configured
        // This ensures all clients get proper SPS/PPS headers when the media pipeline is ready
        std::lock_guard<std::mutex> lock(pending_clients_mutex_);
        g_object_ref(client);  // Take reference since we're storing the client
        pending_clients_.push_back(client);
        APP_LOG_INFO("RTSP CLIENT: Adding client from " + client_ip + " to pending queue (queue size: " + std::to_string(pending_clients_.size()) + ")");
    }
}

void RTSPServerWrapper::pushH264Data(std::shared_ptr<H264Buffer> buffer) {
    if (!buffer || !buffer->data.data() || buffer->size == 0) {
        return;
    }
    
    // Update IN counters
    frames_in_++;
    bytes_in_ += buffer->size;
    
    // Track frame rate and throughput
    static auto last_log_time = std::chrono::steady_clock::now();
    static uint64_t last_frames_in = 0;
    static uint64_t last_frames_out = 0;
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count();
    if (elapsed_ms >= 1000) { // Log every second
        uint64_t current_frames_in = frames_in_.load();
        uint64_t current_frames_out = frames_out_.load();
        
        double fps_in = (current_frames_in - last_frames_in) * 1000.0 / elapsed_ms;
        double fps_out = (current_frames_out - last_frames_out) * 1000.0 / elapsed_ms;
        
        APP_LOG_INFO("RTSP_THROUGHPUT: In=" + std::to_string(fps_in) + 
                    " fps, Out=" + std::to_string(fps_out) + " fps (interval: " + std::to_string(elapsed_ms) + "ms)");
        
        last_frames_in = current_frames_in;
        last_frames_out = current_frames_out;
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
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        APP_LOG_INFO("RTSP Push: NAL type: " + std::string(nal_type_str) + 
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
            gst_object_unref(current_appsrc); // Release the reference from get_appsrc()
            return;
        }
        
        // Check for start code (0x00 0x00 0x00 0x01 or 0x00 0x00 0x01)
        bool has_start_code = false;
        if (buffer->size >= 4 && 
            buffer->data[0] == 0x00 && buffer->data[1] == 0x00 && 
            buffer->data[2] == 0x00 && buffer->data[3] == 0x01) {
            has_start_code = true;
        } else if (buffer->size >= 3 && 
                   buffer->data[0] == 0x00 && buffer->data[1] == 0x00 && 
                   buffer->data[2] == 0x01) {
            has_start_code = true;
        }

        GstBuffer* gst_buffer = nullptr;
        size_t pushed_size = 0;
        
        // Log first 16 bytes of input buffer for debugging
        char hex_dump[64];
        char* ptr = hex_dump;
        for (size_t i = 0; i < std::min(static_cast<size_t>(16), buffer->size); ++i) {
            ptr += sprintf(ptr, "%02X ", buffer->data.data()[i]);
        }
        APP_LOG_INFO("RTSP PUSH HEX: " + std::string(hex_dump) + " (Total size: " + std::to_string(buffer->size) + ")");

        if (has_start_code) {
            // Start code present: Wrap existing buffer using gst_buffer_new_wrapped_full
            // Create a new shared_ptr on the heap to keep the buffer alive
            auto* user_data = new std::shared_ptr<H264Buffer>(buffer);
            
            gst_buffer = gst_buffer_new_wrapped_full(
                GST_MEMORY_FLAG_READONLY,
                buffer->data.data(),
                buffer->size,
                0,
                buffer->size,
                user_data,
                [](gpointer data) {
                    delete static_cast<std::shared_ptr<H264Buffer>*>(data);
                }
            );
            pushed_size = buffer->size;
        } else {
            // Missing start code: Assume AVCC format (4-byte length prefix)
            // and convert to Annex-B by replacing the length with a start code.
            // This is required because the encoder may output AVCC but RTSP needs Annex-B.
            size_t new_size = buffer->size; 
            gst_buffer = gst_buffer_new_allocate(nullptr, new_size, nullptr);
            if (gst_buffer) {
                GstMapInfo map;
                gst_buffer_map(gst_buffer, &map, GST_MAP_WRITE);
                // Replace first 4 bytes (AVCC length) with Annex-B start code
                map.data[0] = 0x00;
                map.data[1] = 0x00;
                map.data[2] = 0x00;
                map.data[3] = 0x01;
                if (buffer->size > 4) {
                    memcpy(map.data + 4, buffer->data.data() + 4, buffer->size - 4);
                }
                gst_buffer_unmap(gst_buffer, &map);
                pushed_size = new_size;
            }
        }
        
        if (gst_buffer) {
            // Set PTS, DTS and Duration to NONE. 
            // appsrc is configured with do-timestamp=true, so it will provide correct timestamps for the live stream.
            // Using GST_CLOCK_TIME_NONE for duration improves VLC compatibility as requested.
            GST_BUFFER_PTS(gst_buffer) = GST_CLOCK_TIME_NONE;
            GST_BUFFER_DTS(gst_buffer) = GST_CLOCK_TIME_NONE;
            GST_BUFFER_DURATION(gst_buffer) = GST_CLOCK_TIME_NONE;
            
            // Log detailed buffer information before pushing
            APP_LOG_INFO("H264 BUFFER PUSH: Size=" + std::to_string(pushed_size) + 
                        " bytes (Start code: " + (has_start_code ? "Existing" : "Added") + 
                        "), PTS=NONE (do-timestamp=true), Duration=NONE");
            
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
            
            // Check the current state of the appsrc for logging purposes
            GstState appsrc_state, pending_state;
            GstStateChangeReturn state_ret = gst_element_get_state(current_appsrc, &appsrc_state, &pending_state, GST_CLOCK_TIME_NONE);
            std::string state_info = "";
            if (state_ret == GST_STATE_CHANGE_SUCCESS) {
                state_info = " (appsrc state: " + std::string(gst_element_state_get_name(appsrc_state)) + ")";
            }
            
            // Push the buffer to appsrc
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC_CAST(current_appsrc), gst_buffer);
            if (ret != GST_FLOW_OK && ret != GST_FLOW_FLUSHING) {
                APP_LOG_ERROR("Failed to push buffer to appsrc: " + std::to_string(ret) + state_info);
            } else {
                // Update OUT counters
                frames_out_++;
                bytes_out_ += pushed_size;
                
                APP_LOG_INFO("SUCCESS: Pushed buffer of size " + std::to_string(pushed_size) + 
                           " bytes to appsrc, return: " + std::to_string(ret) + state_info);
            }
        } else {
            APP_LOG_ERROR("Failed to create GstBuffer for RTSP push");
        }
        
        gst_object_unref(current_appsrc); // Release the reference from get_appsrc()
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
}

void RTSPServerWrapper::monitor_resources() {
    // This method can be called periodically to monitor resource usage
    // For now, we'll just log the current resource usage
    APP_LOG_INFO("RTSP Resource Monitor: Active clients: " + std::to_string(active_client_count_.load()) + 
                 ", Port in use: " + std::to_string(port_in_use_.load()) + 
                 ", Camera in use: " + std::to_string(camera_in_use_.load()));
}

bool RTSPServerWrapper::is_resource_usage_acceptable() {
    // Check if we're within resource limits
    int current_clients = active_client_count_.load();
    if (current_clients > MAX_SIMULTANEOUS_CLIENTS) {
        APP_LOG_WARNING("RTSP Resource Monitor: Too many active clients (" + std::to_string(current_clients) + 
                       " > " + std::to_string(MAX_SIMULTANEOUS_CLIENTS) + ")");
        return false;
    }
    
    return true;
}

void RTSPServerWrapper::internal_cleanup() {
    APP_LOG_INFO("Performing internal cleanup of stale resources...");
    
    // Additional cleanup could include removing stale FIFOs, temp files, etc.
    APP_LOG_INFO("Internal cleanup completed");
}