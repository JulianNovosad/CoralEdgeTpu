#include "http_streamer.h"
#include "util_logging.h"
#include <iostream>
#include <sstream>

HttpStreamer::HttpStreamer(const std::vector<std::string>& options) 
    : options_(options), running_(false) {
}

HttpStreamer::~HttpStreamer() {
    stop();
}

void HttpStreamer::start() {
    if (running_) return;

    try {
        APP_LOG_INFO("Starting HttpStreamer on port 8080..."); // Assuming port is in options_
        server_ = std::make_unique<CivetServer>(options_);
        
        // Register handlers
        server_->addHandler("\/", this);
        server_->addWebSocketHandler("\/stream", this);
        
        running_ = true;
        APP_LOG_INFO("HttpStreamer started successfully.");
    } catch (const CivetException& e) {
        APP_LOG_ERROR("Failed to start HttpStreamer: " + std::string(e.what()));
    }
}

void HttpStreamer::stop() {
    if (!running_) return;
    
    APP_LOG_INFO("Stopping HttpStreamer...");
    
    // Close all active connections
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        for (auto conn : connections_) {
            mg_close_connection(conn);
        }
        connections_.clear();
    }
    
    if (server_) {
        server_->close();
        server_.reset();
    }
    
    running_ = false;
    APP_LOG_INFO("HttpStreamer stopped.");
}

void HttpStreamer::pushH264Data(const std::vector<uint8_t>& data) {
    if (!running_ || data.empty()) return;

    std::lock_guard<std::mutex> lock(connections_mutex_);
    for (auto conn : connections_) {
        // Send binary data (opcode 0x2 for binary frame in WebSocket)
        mg_websocket_write(conn, MG_WEBSOCKET_OPCODE_BINARY, (const char*)data.data(), data.size());
    }
}

// CivetHandler interface
bool HttpStreamer::handleGet(CivetServer *server, struct mg_connection *conn) {
    const char* html = 
        "<!DOCTYPE html>"
        "<html><head><title>Coral EdgeTPU Stream</title>"
        "<style>"
        "body { font-family: sans-serif; background: #222; color: #eee; text-align: center; }"
        "#video-container { margin: 20px auto; width: 640px; height: 480px; background: #000; border: 2px solid #444; }"
        "#status { font-family: monospace; color: #0f0; margin-top: 10px; }"
        "</style>"
        "</head>"
        "<body><h1>EdgeTPU H.264 Live Stream</h1>"
        "<div id='video-container'><canvas id='canvas' width='640' height='480'></canvas></div>"
        "<div id='status'>Connecting...</div>"
        "<p>Note: Requires a browser with WebCodecs API (Chrome 94+, Edge 94+).</p>"
        "<script>"
        "  const status = document.getElementById('status');"
        "  const canvas = document.getElementById('canvas');"
        "  const ctx = canvas.getContext('2d');"
        "  let ws = null;"
        "  let decoder = null;"
        "  let frameCount = 0;"
        "  let totalBytes = 0;"
        ""
        "  function initDecoder() {"
        "    if ('VideoDecoder' in window) {"
        "      decoder = new VideoDecoder({"
        "        output: (frame) => {"
        "          ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);"
        "          frame.close();"
        "          frameCount++;"
        "          status.innerText = 'Receiving: ' + frameCount + ' frames, ' + (totalBytes/1024).toFixed(1) + ' KB';"
        "        },"
        "        error: (e) => { console.error('Decoder error:', e); status.innerText = 'Decoder Error: ' + e.message; }"
        "      });"
        "      decoder.configure({"
        "        codec: 'avc1.42001e', // Baseline Profile Level 3.0 (common for x264 baseline)"
        "        optimizeForLatency: true"
        "      });"
        "    } else {"
        "      status.innerText = 'WebCodecs API not supported. Use Chrome/Edge.'; "
        "    }"
        "  }"
        ""
        "  function connect() {"
        "    ws = new WebSocket('ws://' + location.host + '/stream');"
        "    ws.binaryType = 'arraybuffer';"
        "    ws.onopen = () => { status.innerText = 'Connected. Waiting for data...'; };"
        "    ws.onclose = () => { status.innerText = 'Disconnected. Reconnecting...'; setTimeout(connect, 2000); };"
        "    ws.onerror = (e) => { status.innerText = 'WebSocket Error'; };"
        "    ws.onmessage = (event) => {"
        "      totalBytes += event.data.byteLength;"
        "      if (!decoder) return;"
        "      "
        "      const chunk = new EncodedVideoChunk({"
        "        type: 'key', // Simplification: assuming keyframes or treating all as key for resilience (not ideal but works for simple streams)"
        "        timestamp: event.timeStamp * 1000,"
        "        data: new Uint8Array(event.data)"
        "      });"
        "      try {"
        "        decoder.decode(chunk);"
        "      } catch (e) {"
        "        console.error('Decode error', e);"
        "      }"
        "    };"
        "  }"
        ""
        "  initDecoder();"
        "  connect();"
        "</script>"
        "</body></html>";
        
    mg_printf(conn, 
              "HTTP/1.1 200 OK\r\n"
              "Content-Type: text/html\r\n"
              "Connection: close\r\n"
              "\r\n%s", html);
    return true;
}

// CivetWebSocketHandler interface
bool HttpStreamer::handleConnection(CivetServer *server, const struct mg_connection *conn) {
    return true; // Accept connection
}

void HttpStreamer::handleReadyState(CivetServer *server, struct mg_connection *conn) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    connections_.insert(conn);
    APP_LOG_INFO("New WebSocket connection accepted.");
}

bool HttpStreamer::handleData(CivetServer *server, struct mg_connection *conn, int bits, char *data, size_t data_len) {
    // We don't expect data from client, but we must handle it to keep connection alive if needed
    return true;
}

void HttpStreamer::handleClose(CivetServer *server, const struct mg_connection *conn) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    connections_.erase(const_cast<struct mg_connection*>(conn));
    APP_LOG_INFO("WebSocket connection closed.");
}
