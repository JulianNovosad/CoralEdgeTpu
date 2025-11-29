#include "http_server.h"
#include "util_logging.h"
extern "C" {
#include "civetweb.h"
}
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

HttpServer::HttpServer(const std::string& address, const std::string& document_root, ImageQueue& input_queue)
    : address_(address), document_root_(document_root), input_queue_(input_queue), running_(false) {
    LOG_INFO("HttpServer created.");
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
        "document_root", document_root_.c_str(),
        "listening_ports", address_.c_str(),
        0
    };

    ctx_ = mg_start(NULL, 0, options);
    if (ctx_ == NULL) {
        LOG_ERROR("Failed to start civetweb server.");
        running_ = false;
        return false; // Return false on failure
    }

    mg_set_request_handler(ctx_, "/stream", event_handler, this);

    running_ = true;
    LOG_INFO("HttpServer started on address " + address_);
    return true; // Return true on success
}

void HttpServer::stop() {
    if (running_.exchange(false)) {
        LOG_INFO("Stopping HttpServer...");
        if (ctx_) {
            mg_stop(ctx_);
            ctx_ = nullptr;
        }
        LOG_INFO("HttpServer stopped.");
    }
}

int HttpServer::event_handler(struct mg_connection* c, void* fn_data) {
    HttpServer* self = static_cast<HttpServer*>(fn_data);
    self->handle_stream(c);
    return 1;
}

void HttpServer::handle_stream(struct mg_connection* c) {
    mg_printf(c, "HTTP/1.1 200 OK\r\n"
                 "Content-Type: multipart/x-mixed-replace; boundary=--frame\r\n"
                 "\r\n");

    ImageData image_data;
    while (running_) {
        if (input_queue_.pop(image_data)) {
            cv::Mat image(image_data.height, image_data.width, CV_8UC3, image_data.data.data());
            std::vector<uchar> buf;
            cv::imencode(".jpg", image, buf);

            mg_printf(c, "--frame\r\n"
                         "Content-Type: image/jpeg\r\n"
                         "Content-Length: %zu\r\n"
                         "\r\n", buf.size());
            mg_write(c, buf.data(), buf.size());
            mg_printf(c, "\r\n");
        }
    }
}
