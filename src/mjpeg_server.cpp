#include "mjpeg_server.h"
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sstream>
#include <iomanip> // For std::hex, std::setfill, std::setw

// Boundary string for MJPEG stream
const std::string MJPEG_BOUNDARY = "opencv_boundary";

MjpegServer::MjpegServer(int port, MjpegQueue& input_queue)
    : port_(port), input_queue_(input_queue) {}

MjpegServer::~MjpegServer() {
    stop();
}

bool MjpegServer::start() {
    if (running_) {
        std::cerr << "MjpegServer is already running." << std::endl;
        return false;
    }

    server_sock_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock_ < 0) {
        std::cerr << "Failed to create server socket." << std::endl;
        return false;
    }

    int optval = 1;
    setsockopt(server_sock_, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_);

    if (bind(server_sock_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to bind server socket to port " << port_ << std::endl;
        close(server_sock_);
        return false;
    }

    if (listen(server_sock_, 5) < 0) { // Max 5 pending connections
        std::cerr << "Failed to listen on server socket." << std::endl;
        close(server_sock_);
        return false;
    }

    running_ = true;
    input_queue_.set_running(true);
    server_thread_ = std::thread(&MjpegServer::server_thread_func, this);
    std::cout << "MJPEG server started on port " << port_ << std::endl;
    return true;
}

void MjpegServer::stop() {
    if (!running_) {
        return;
    }
    running_ = false;
    input_queue_.set_running(false);

    if (server_sock_ != -1) {
        // Shut down the socket to unblock accept()
        shutdown(server_sock_, SHUT_RDWR);
        close(server_sock_);
        server_sock_ = -1;
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    std::cout << "MJPEG server stopped." << std::endl;
}

void MjpegServer::server_thread_func() {
    while (running_) {
        sockaddr_in client_addr{};
        socklen_t client_addr_len = sizeof(client_addr);
        int client_sock = accept(server_sock_, (struct sockaddr*)&client_addr, &client_addr_len);

        if (client_sock < 0) {
            if (running_) { // Only report error if we're still running
                std::cerr << "Failed to accept client connection." << std::endl;
            }
            continue;
        }
        std::thread client_handler(&MjpegServer::handle_client, this, client_sock);
        client_handler.detach(); // Detach to allow handling multiple clients
    }
}

void MjpegServer::handle_client(int client_sock) {
    char buffer[2048];
    ssize_t bytes_received = recv(client_sock, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received <= 0) {
        close(client_sock);
        return;
    }
    buffer[bytes_received] = '\0';
    // std::cout << "Received request:\n" << buffer << std::endl;

    // We expect a GET request for /, typically
    if (std::string(buffer).rfind("GET / ", 0) != 0 && std::string(buffer).rfind("GET /stream ", 0) != 0) {
        // Simple 404 for other requests
        std::string response = "HTTP/1.0 404 Not Found\r\nContent-Length: 13\r\n\r\n404 Not Found";
        send(client_sock, response.c_str(), response.length(), 0);
        close(client_sock);
        return;
    }

    // Send MJPEG stream headers
    std::ostringstream header_ss;
    header_ss << "HTTP/1.0 200 OK\r\n";
    header_ss << "Cache-Control: no-cache\r\n";
    header_ss << "Pragma: no-cache\r\n";
    header_ss << "Connection: close\r\n";
    header_ss << "Content-Type: multipart/x-mixed-replace; boundary=" << MJPEG_BOUNDARY << "\r\n";
    header_ss << "\r\n";
    std::string headers = header_ss.str();
    send(client_sock, headers.c_str(), headers.length(), 0);

    ImageFrame frame;
    while (running_ && input_queue_.peek_latest(frame)) { // Peek the latest frame
        std::ostringstream content_ss;
        content_ss << "--" << MJPEG_BOUNDARY << "\r\n";
        content_ss << "Content-Type: image/jpeg\r\n";
        content_ss << "Content-Length: " << frame.jpeg_data.size() << "\r\n";
        content_ss << "\r\n";
        std::string content_headers = content_ss.str();

        // Send content headers
        if (send(client_sock, content_headers.c_str(), content_headers.length(), 0) < 0) {
            std::cerr << "Failed to send content headers to client." << std::endl;
            break;
        }

        // Send JPEG data
        if (send(client_sock, (const char*)frame.jpeg_data.data(), frame.jpeg_data.size(), 0) < 0) {
            std::cerr << "Failed to send JPEG data to client." << std::endl;
            break;
        }

        // Send boundary
        if (send(client_sock, "\r\n", 2, 0) < 0) {
            std::cerr << "Failed to send boundary to client." << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // Avoid burning CPU, simulate frame rate
    }

    close(client_sock);
    // std::cout << "Client disconnected." << std::endl;
}
