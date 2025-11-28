/**
 * @file mjpeg_server.cpp
 * @brief Implements the MjpegServer class for streaming MJPEG video over HTTP.
 *
 * This module provides the concrete implementation for a simple HTTP server
 * that serves MJPEG video frames. It handles socket creation, binding, listening,
 * client connections, and sending multipart/x-mixed-replace responses with
 * JPEG frames retrieved from a thread-safe queue.
 */

#include "mjpeg_server.h"
#include "util_logging.h" // For LOG_INFO, LOG_ERROR
#include <iostream>       // For std::cerr (temporarily before full LOG_ERROR conversion)
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <sys/socket.h>   // For socket, bind, listen, accept, send, recv
#include <netinet/in.h>   // For sockaddr_in, INADDR_ANY
#include <unistd.h>       // For close
#include <sstream>        // For std::ostringstream
#include <iomanip>        // For std::hex, std::setfill, std::setw
#include <cstring>        // For strerror
#include <opencv2/opencv.hpp>

// Boundary string for MJPEG multipart stream.
// This string separates individual JPEG frames within the HTTP response.
const std::string MJPEG_BOUNDARY = "opencv_boundary";


/**
 * @brief Constructor for MjpegServer.
 *
 * Initializes the server with the specified listening port and a reference
 * to the queue from which MJPEG frames will be obtained.
 *
 * @param port The TCP port number on which the server will listen.
 * @param input_queue Reference to the thread-safe MjpegQueue providing MJPEG frames.
 */
MjpegServer::MjpegServer(int port, MjpegQueue& input_queue)
    : port_(port), input_queue_(input_queue) {}

/**
 * @brief Destructor for MjpegServer.
 *
 * Ensures the server is stopped and its resources are properly cleaned up.
 */
MjpegServer::~MjpegServer() {
    stop(); // Call stop to ensure graceful shutdown and thread joining.
}

/**
 * @brief Starts the MJPEG server.
 *
 * Creates a TCP socket, binds it to the specified port, and starts listening
 * for incoming client connections. Launches a dedicated `server_thread_func`
 * to handle client connections asynchronously.
 *
 * @return True if the server started successfully, false otherwise.
 */
bool MjpegServer::start() {
    if (running_) {
        LOG_ERROR("MjpegServer is already running.");
        return false;
    }

    // Create a TCP socket.
    server_sock_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock_ < 0) {
        LOG_ERROR("Failed to create server socket: " + std::string(strerror(errno)));
        return false;
    }

    // Set socket option to reuse address, preventing "Address already in use" errors.
    int optval = 1;
    if (setsockopt(server_sock_, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
        LOG_ERROR("Failed to set SO_REUSEADDR option on server socket: " + std::string(strerror(errno)));
        close(server_sock_);
        return false;
    }

    // Configure server address structure.
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;           // IPv4
    server_addr.sin_addr.s_addr = INADDR_ANY;   // Listen on all available network interfaces.
    server_addr.sin_port = htons(port_);        // Convert port to network byte order.

    // Bind the socket to the specified port and address.
    if (bind(server_sock_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        LOG_ERROR("Failed to bind server socket to port " + std::to_string(port_) + ": " + std::string(strerror(errno)));
        close(server_sock_);
        return false;
    }

    // Start listening for incoming connections. Max 5 pending connections in queue.
    if (listen(server_sock_, 5) < 0) {
        LOG_ERROR("Failed to listen on server socket: " + std::string(strerror(errno)));
        close(server_sock_);
        return false;
    }

    LOG_INFO("MjpegServer is now listening for connections on port " + std::to_string(port_));

    std::promise<bool> server_ready_promise;
    std::future<bool> server_ready_future = server_ready_promise.get_future();

    running_ = true;
    input_queue_.set_running(true);
    server_thread_ = std::thread(&MjpegServer::server_thread_func, this, std::move(server_ready_promise));

    // Wait for the server_thread_func to signal that it has fully started its accept loop
    server_ready_future.wait();
    if (!server_ready_future.get()) {
        LOG_ERROR("MjpegServer thread failed to signal readiness.");
        running_ = false;
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        close(server_sock_);
        return false;
    }
    
    LOG_INFO("MjpegServer is fully operational on port " + std::to_string(port_));
    return true;
}

/**
 * @brief Stops the MJPEG server.
 *
 * Signals the server thread to terminate, gracefully shuts down the server socket
 * to unblock any `accept()` calls, closes the socket, and joins the server thread.
 */
void MjpegServer::stop() {
    if (!running_.exchange(false)) { // Atomically set to false and check previous value.
        return; // Already stopped.
    }
    LOG_INFO("Stopping MJPEG server...");
    input_queue_.set_running(false); // Signal input queue to stop.

    // Shut down and close the server socket to unblock accept() in the server_thread_func.
    if (server_sock_ != -1) {
        shutdown(server_sock_, SHUT_RDWR); // Stop all I/O on the socket.
        close(server_sock_);
        server_sock_ = -1;
    }
    // Wait for the server thread to finish its execution.
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    LOG_INFO("MJPEG server stopped.");
}

/**
 * @brief The main loop for the MJPEG server thread.
 *
 * This function continuously accepts new client connections. For each accepted
 * client, it detaches a new thread to handle the client's request, allowing
 * the server to serve multiple clients concurrently.
 */
void MjpegServer::server_thread_func(std::promise<bool> server_ready_promise) {
    server_ready_promise.set_value(true); // Signal that the server thread is ready and its accept loop will start.

    while (running_) {
        sockaddr_in client_addr{};
        socklen_t client_addr_len = sizeof(client_addr);
        int client_sock = accept(server_sock_, (struct sockaddr*)&client_addr, &client_addr_len);

        if (client_sock < 0) {
            if (running_) { // Only report an error if the server is still expected to be running.
                LOG_ERROR("Failed to accept client connection: " + std::string(strerror(errno)));
            } else {
                LOG_INFO("Server stopping, accept() call interrupted."); // Expected behavior on shutdown.
            }
            continue; // Continue to the next iteration, waiting for another client.
        }
        // Launch a new thread to handle the client connection. Detach it so the server doesn't wait for it.
        std::thread client_handler(&MjpegServer::handle_client, this, client_sock);
        client_handler.detach();
    }
}

void MjpegServer::handle_client(int client_sock) {
    // Send the initial HTTP response header for a multipart MJPEG stream.
    std::string header = "HTTP/1.0 200 OK\r\n";
    header += "Content-Type: multipart/x-mixed-replace; boundary=--" + MJPEG_BOUNDARY + "\r\n";
    header += "Cache-Control: no-cache\r\n";
    header += "Pragma: no-cache\r\n";
    header += "Connection: close\r\n";
    header += "\r\n"; // End of headers
    
    if (send(client_sock, header.c_str(), header.length(), MSG_NOSIGNAL) < 0) {
        LOG_ERROR("MjpegServer: Failed to send initial HTTP headers to client (sock: " + std::to_string(client_sock) + "): " + std::string(strerror(errno)));
        close(client_sock);
        return;
    }
    LOG_INFO("MjpegServer: Sent initial HTTP headers to client (sock: " + std::to_string(client_sock) + ")");

    // Loop to continuously send MJPEG frames while the server is running and frames are available.
    LOG_INFO("MjpegServer: Starting frame sending loop for client (sock: " + std::to_string(client_sock) + ")");
    MjpegFrame mjpeg_frame;
    while (running_) {
        if (!input_queue_.pop(mjpeg_frame)) {
            // If pop fails (e.g., queue empty and shutting down), break loop
            break;
        }

        std::ostringstream content_ss;
        content_ss << "--" << MJPEG_BOUNDARY << "\r\n";        // Start of a new part with boundary.
        content_ss << "Content-Type: image/jpeg\r\n";          // Content type of the part.
        content_ss << "Content-Length: " << mjpeg_frame.jpeg_data.size() << "\r\n"; // Size of the JPEG data.
        content_ss << "\r\n"; // End of content headers for this part.
        std::string content_headers = content_ss.str();

        // Send content headers for the current JPEG frame.
        if (send(client_sock, content_headers.c_str(), content_headers.length(), MSG_NOSIGNAL) < 0) {
            LOG_ERROR("MjpegServer: Failed to send content headers for frame (size: " + std::to_string(mjpeg_frame.jpeg_data.size()) + ", sock: " + std::to_string(client_sock) + "): " + std::string(strerror(errno)));
            break; // Exit loop if send fails.
        }

        // Send the actual JPEG data.
        if (send(client_sock, reinterpret_cast<const char*>(mjpeg_frame.jpeg_data.data()), mjpeg_frame.jpeg_data.size(), MSG_NOSIGNAL) < 0) {
            LOG_ERROR("MjpegServer: Failed to send JPEG data (size: " + std::to_string(mjpeg_frame.jpeg_data.size()) + ", sock: " + std::to_string(client_sock) + "): " + std::string(strerror(errno)));
            break; // Exit loop if send fails.
        }

        // Send the boundary delimiter after the JPEG data.
        if (send(client_sock, "\r\n", 2, MSG_NOSIGNAL) < 0) {
            LOG_ERROR("MjpegServer: Failed to send boundary delimiter (sock: " + std::to_string(client_sock) + "): " + std::string(strerror(errno)));
            break; // Exit loop if send fails.
        }
    }

    LOG_INFO("MjpegServer: Client disconnected (sock: " + std::to_string(client_sock) + ")");
    close(client_sock); // Close client socket after stream ends or error.
}

void MjpegServer::get_state() const {
    LOG_INFO("--- MjpegServer State ---");
    LOG_INFO("  Running: " + std::to_string(running_));
    LOG_INFO("  Listening Port: " + std::to_string(port_));
    if (server_sock_ != -1) {
        LOG_INFO("  Server Socket FD: " + std::to_string(server_sock_));
    } else {
        LOG_INFO("  Server Socket: Not initialized or closed.");
    }
    LOG_INFO("-------------------------");
}