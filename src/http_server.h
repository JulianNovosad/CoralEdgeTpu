#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

#include "pipeline_structs.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <memory>

// Forward declaration
struct mg_connection;
struct mg_context;

class HttpServer {
public:
    HttpServer(const std::string& address, H264Queue& h264_input_queue);
    ~HttpServer();

    bool start();
    void stop();

    void broadcast_packet(std::shared_ptr<H264Buffer> buffer);

private:
    // Holds state for each connected WebSocket client
    struct Client {
        std::unique_ptr<H264Queue> queue;
        std::thread writer_thread;
        std::atomic<bool> is_active;

        Client() : queue(std::make_unique<H264Queue>(50)), is_active(true) {}
    };

    // WebSocket event handlers
    static int websocket_connect_handler(const struct mg_connection *conn, void *cbdata);
    static void websocket_ready_handler(struct mg_connection *conn, void *cbdata);
    static void websocket_close_handler(const struct mg_connection *conn, void *cbdata);

    // Thread functions
    void client_writer_thread(struct mg_connection* c);
    void distributor_thread_func();

    // Client management
    void add_client(struct mg_connection* c);
    void remove_client(struct mg_connection* c);

    std::string address_;
    H264Queue& h264_input_queue_;
    std::atomic<bool> running_;
    struct mg_context* ctx_ = nullptr;
    std::thread distributor_thread_;

    std::map<const struct mg_connection*, std::shared_ptr<Client>> clients_;
    std::mutex clients_mutex_;
};

#endif // HTTP_SERVER_H
