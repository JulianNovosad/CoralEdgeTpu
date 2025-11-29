#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

#include "pipeline_structs.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>

// Forward declaration
struct mg_connection;
struct mg_context;

class HttpServer {
public:
    HttpServer(const std::string& address, const std::string& document_root, ImageQueue& input_queue);
    ~HttpServer();

    bool start();
    void stop();

private:
    static int event_handler(struct mg_connection* c, void* fn_data);
    void handle_stream(struct mg_connection* c);

    std::string address_;
    std::string document_root_;
    ImageQueue& input_queue_;
    std::atomic<bool> running_;
    struct mg_context* ctx_ = nullptr;
};

#endif // HTTP_SERVER_H
