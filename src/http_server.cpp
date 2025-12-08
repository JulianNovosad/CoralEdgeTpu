#include "http_server.h"
#include "util_logging.h"
extern "C" {
#include "civetweb.h"
}
#include <vector>
#include <cstring>

HttpServer::HttpServer(const std::string& ip_address, 
               unsigned short livestream_video_port,
               unsigned short bounding_box_stream_port,
               unsigned short reticle_coordinate_port,
               unsigned short status_telemetry_port,
               H264Queue& h264_input_queue)
    : ip_address_(ip_address), 
      livestream_video_port_(livestream_video_port), 
      bounding_box_stream_port_(bounding_box_stream_port), 
      reticle_coordinate_port_(reticle_coordinate_port), 
      status_telemetry_port_(status_telemetry_port), 
      h264_input_queue_(h264_input_queue), running_(false), ctx_(nullptr) {
    LOG_INFO("HttpServer created for WebSocket streaming.");
}

HttpServer::~HttpServer() {
    stop();
    LOG_INFO("HttpServer destroyed.");
}

bool HttpServer::start() {
    if (running_) {
        LOG_ERROR("HttpServer is already running.");
        return false;
    }
    
    const char* options[] = {
        "listening_ports", (ip_address_ + ":" + std::to_string(livestream_video_port_)).c_str(),
        "websocket_timeout_ms", "5000",
        0
    };

    LOG_INFO("HttpServer: Attempting to start CivetWeb server on " + address_);
    ctx_ = mg_start(NULL, 0, options);
    if (ctx_ == NULL) {
        LOG_ERROR("HttpServer: Failed to start CivetWeb server. mg_start returned NULL.");
        return false;
    }

    // After mg_start, verify if the port is actually bound
    struct mg_server_port ports[32]; // Max 32 ports
    int num_ports = mg_get_server_ports(ctx_, 32, ports);
    bool port_found = false;
    int configured_port = livestream_video_port_;

    if (num_ports > 0) {
        for (int i = 0; i < num_ports; ++i) {
            if (ports[i].port == configured_port) {
                port_found = true;
                break;
            }
        }
    }

    if (!port_found) {
        LOG_ERROR("HttpServer: CivetWeb server failed to bind to configured port " + configured_port_str + ". mg_get_server_ports shows no listener.");
        mg_stop(ctx_); // Stop CivetWeb if it failed to bind
        ctx_ = nullptr;
        return false;
    }


    mg_set_websocket_handler(ctx_, "/stream",
                             websocket_connect_handler,
                             websocket_ready_handler,
                             NULL, // data handler (not needed for server-push)
                             websocket_close_handler,
                             this);

    running_ = true;
    distributor_thread_ = std::thread(&HttpServer::distributor_thread_func, this);
    LOG_INFO("HttpServer started with WebSocket endpoint /stream");
    return true;
}

void HttpServer::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    LOG_INFO("Stopping HttpServer...");
    
    if (distributor_thread_.joinable()) {
        distributor_thread_.join();
    }
    
    // Disconnect all clients
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (auto const& [conn, client] : clients_) {
            client->is_active = false;
            if(client->writer_thread.joinable()) {
                client->writer_thread.join();
            }
        }
        clients_.clear();
    }

    if (ctx_) {
        mg_stop(ctx_);
        ctx_ = nullptr;
    }
    LOG_INFO("HttpServer stopped.");
}

void HttpServer::distributor_thread_func() {
    LOG_INFO("H.264 distributor thread started.");
    while (running_) {
        std::shared_ptr<H264Buffer> buffer;
        if (h264_input_queue_.pop(buffer)) {
            if (buffer) {
                broadcast_packet(buffer);
            }
        } else {
            if (!running_) break;
        }
    }
    LOG_INFO("H.264 distributor thread stopped.");
}

void HttpServer::broadcast_packet(std::shared_ptr<H264Buffer> buffer) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    // LOG_INFO("Broadcasting packet of size " + std::to_string(buffer->size) + " to " + std::to_string(clients_.size()) + " clients.");
    for (auto const& [conn, client] : clients_) {
        client->queue->push(std::move(buffer)); // Push the shared_ptr, avoids copy
    }
}

int HttpServer::websocket_connect_handler(const struct mg_connection* conn, void* cbdata) {
    HttpServer* self = static_cast<HttpServer*>(cbdata);
    LOG_INFO("WebSocket client connected.");
    // Returning 0 signifies acceptance of the connection.
    return 0;
}

void HttpServer::websocket_ready_handler(struct mg_connection* conn, void* cbdata) {
    HttpServer* self = static_cast<HttpServer*>(cbdata);
    self->add_client(conn);
}

void HttpServer::websocket_close_handler(const struct mg_connection* conn, void* cbdata) {
    HttpServer* self = static_cast<HttpServer*>(cbdata);
    self->remove_client(const_cast<struct mg_connection*>(conn));
}

void HttpServer::add_client(struct mg_connection* c) {
    auto client = std::make_shared<Client>();
    
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        clients_[c] = client;
    }

    // Start a dedicated writer thread for this client
    client->writer_thread = std::thread(&HttpServer::client_writer_thread, this, c);
    LOG_INFO("Added new client. Total clients: " + std::to_string(clients_.size()));
}

void HttpServer::remove_client(struct mg_connection* c) {
    std::shared_ptr<Client> client;
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        if (clients_.count(c)) {
            client = clients_[c];
            clients_.erase(c);
        }
    }

    if (client) {
        client->is_active = false;
        if (client->writer_thread.joinable()) {
            client->writer_thread.join();
        }
    }
    LOG_INFO("Removed client. Total clients: " + std::to_string(clients_.size()));
}


void HttpServer::client_writer_thread(struct mg_connection* c) {
    LOG_INFO("Client writer thread started.");
    std::shared_ptr<Client> client;

    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        if (clients_.count(c) == 0) {
            LOG_ERROR("Client not found in map at start of writer thread.");
            return;
        }
        client = clients_[c];
    }
    
    std::shared_ptr<H264Buffer> h264_buffer;
    while (client->is_active) {
        if (client->queue->pop(h264_buffer)) {
            if (h264_buffer && h264_buffer->size > 0) {
                int ret = mg_websocket_write(c, MG_WEBSOCKET_OPCODE_BINARY, 
                                             reinterpret_cast<const char*>(h264_buffer->data.data()), 
                                             h264_buffer->size);
                if (ret <= 0) {
                    LOG_ERROR("mg_websocket_write failed. Closing connection.");
                    break; 
                }
            }
        } else {
            // Pop returned false, means queue is stopping
            if (!client->is_active) {
                break;
            }
        }
    }
    LOG_INFO("Client writer thread stopped.");
}
