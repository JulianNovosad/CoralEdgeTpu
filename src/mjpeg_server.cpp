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

    running_ = true;
    input_queue_.set_running(true); // Signal input queue to be active.
    server_thread_ = std::thread(&MjpegServer::server_thread_func, this);
    LOG_INFO("MJPEG server started on port " + std::to_string(port_));
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
void MjpegServer::server_thread_func() {
    while (running_) {
        sockaddr_in client_addr{}; // Structure to hold client address information.
        socklen_t client_addr_len = sizeof(client_addr);
        // Accept a new client connection. This call blocks until a client connects or the socket is shut down.
        int client_sock = accept(server_sock_, (struct sockaddr*)&client_addr, &client_addr_len);

        if (client_sock < 0) {
            if (running_) { // Only report an error if the server is still expected to be running.
                LOG_ERROR("Failed to accept client connection: " + std::string(strerror(errno)));
            }
            continue; // Continue to the next iteration, waiting for another client.
        }
        // Launch a new thread to handle the client connection. Detach it so the server doesn't wait for it.
        std::thread client_handler(&MjpegServer::handle_client, this, client_sock);
        client_handler.detach();
    }
}

/**
 * @brief Handles an individual client connection for MJPEG streaming.
 *
 * This function reads the client's HTTP request, sends appropriate MJPEG stream
 * headers, and then continuously retrieves the latest MJPEG frames from the
 * input queue, sending them to the connected client as a multipart/x-mixed-replace
 * HTTP response.
 *
 * @param client_sock The socket file descriptor for the connected client.
 */
void MjpegServer::handle_client(int client_sock) {
    char buffer[2048]; // Buffer to read incoming HTTP request.
    ssize_t bytes_received = recv(client_sock, buffer, sizeof(buffer) - 1, 0); // Read client request.
    if (bytes_received <= 0) {
        LOG_WARNING("Client disconnected or error receiving request.");
        close(client_sock);
        return;
    }
    buffer[bytes_received] = '\0'; // Null-terminate the received data.
    // LOG_INFO("Received request from client:\n" + std::string(buffer)); // For debugging requests.

    // Basic request parsing: check for GET / or GET /stream
    if (std::string(buffer).rfind("GET / ", 0) != 0 && std::string(buffer).rfind("GET /stream ", 0) != 0) {
        // For any other request, send a simple 404 Not Found response.
        std::string response = "HTTP/1.0 404 Not Found\r\nContent-Length: 13\r\n\r\n404 Not Found";
        send(client_sock, response.c_str(), response.length(), 0);
        close(client_sock);
        return;
    }

    // Send MJPEG stream HTTP headers for multipart/x-mixed-replace.
    std::ostringstream header_ss;
    header_ss << "HTTP/1.0 200 OK\r\n";
    header_ss << "Cache-Control: no-cache\r\n";       // Prevent caching.
    header_ss << "Pragma: no-cache\r\n";             // For older HTTP/1.0 clients.
    header_ss << "Connection: close\r\n";            // Connection will be closed by server.
    header_ss << "Content-Type: multipart/x-mixed-replace; boundary=" << MJPEG_BOUNDARY << "\r\n"; // Main content type.
    header_ss << "\r\n"; // End of headers.
    std::string headers = header_ss.str();
    if (send(client_sock, headers.c_str(), headers.length(), 0) < 0) {
        LOG_ERROR("Failed to send MJPEG stream headers to client: " + std::string(strerror(errno)));
        close(client_sock);
        return;
    }

    ImageFrame frame;
    // Loop to continuously send MJPEG frames while the server is running and frames are available.
    while (running_ && input_queue_.peek_latest(frame)) { // Peek the latest frame from the queue.
        std::ostringstream content_ss;
        content_ss << "--" << MJPEG_BOUNDARY << "\r\n";        // Start of a new part with boundary.
        content_ss << "Content-Type: image/jpeg\r\n";          // Content type of the part.
        content_ss << "Content-Length: " << frame.jpeg_data.size() << "\r\n"; // Size of the JPEG data.
        content_ss << "\r\n"; // End of content headers for this part.
        std::string content_headers = content_ss.str();

        // Send content headers for the current JPEG frame.
        if (send(client_sock, content_headers.c_str(), content_headers.length(), 0) < 0) {
            LOG_ERROR("Failed to send content headers for MJPEG frame to client: " + std::string(strerror(errno)));
            break; // Exit loop if send fails.
        }

        // Send the actual JPEG data.
        if (send(client_sock, (const char*)frame.jpeg_data.data(), frame.jpeg_data.size(), 0) < 0) {
            LOG_ERROR("Failed to send JPEG data to client: " + std::string(strerror(errno)));
            break; // Exit loop if send fails.
        }

        // Send the boundary delimiter after the JPEG data.
        if (send(client_sock, "\r\n", 2, 0) < 0) {
            LOG_ERROR("Failed to send boundary delimiter to client: " + std::string(strerror(errno)));
            break; // Exit loop if send fails.
        }
        // Small delay to control frame rate and prevent excessive CPU usage.
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    close(client_sock); // Close client socket after stream ends or error.
    // LOG_INFO("Client disconnected."); // For debugging client connections.
}