/**
 * @file mjpeg_server.h
 * @brief Defines the MjpegServer class for streaming MJPEG video over HTTP.
 *
 * This class implements a simple HTTP server that serves MJPEG video frames
 * to connected clients. It retrieves MJPEG frames from a thread-safe queue
 * populated by a camera capture module and streams them to web browsers or
 * other compatible clients.
 */

#ifndef MJPEG_SERVER_H
#define MJPEG_SERVER_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <sys/socket.h> // For socket programming types

#include "pipeline_structs.h" // Use the new central header

/**
 * @brief A simple HTTP server for streaming MJPEG video.
 *
 * The MjpegServer class creates a TCP/IP server that listens for incoming
 * HTTP connections. Upon a client connection, it continuously reads MJPEG
 * frames from an input queue and streams them as a multipart/x-mixed-replace
 * HTTP response, suitable for displaying live video in web browsers.
 */
class MjpegServer {
public:
    /**
     * @brief Constructor for MjpegServer.
     *
     * Initializes the MJPEG server with the specified port and a reference
     * to the input queue from which MJPEG frames will be retrieved.
     *
     * @param port The TCP port number on which the server will listen.
     * @param input_queue Reference to the thread-safe MjpegQueue providing MJPEG frames.
     */
    MjpegServer(int port, MjpegQueue& input_queue);

    /**
     * @brief Destructor for MjpegServer.
     *
     * Ensures that the server thread is gracefully stopped and resources are released.
     */
    ~MjpegServer();

    /**
     * @brief Starts the MJPEG server.
     *
     * Creates and launches the main server thread that listens for client connections.
     *
     * @return True if the server started successfully, false otherwise.
     */
    bool start();

    /**
     * @brief Stops the MJPEG server.
     *
     * Signals the server thread to terminate, closes the server socket, and joins
     * the server thread for a clean shutdown.
     */
    void stop();

    /**
     * @brief Checks if the MJPEG server is currently running.
     *
     * @return True if the server is running, false otherwise.
     */
    bool is_running() const { return running_; }

private:
    /**
     * @brief The main loop for the MJPEG server thread.
     *
     * This function binds to the specified port, listens for incoming connections,
     * accepts clients, and then detaches a new thread to handle each client.
     */
    void server_thread_func();

    /**
     * @brief Handles an individual client connection for MJPEG streaming.
     *
     * This function sends the HTTP multipart header to the client and then
     * continuously reads MJPEG frames from the input queue, sending each frame
     * with its boundary delimiter to the connected client.
     *
     * @param client_sock The socket file descriptor for the connected client.
     */
    void handle_client(int client_sock);

    int port_; ///< The TCP port number the server listens on.
    MjpegQueue& input_queue_; ///< Reference to the queue from which MJPEG frames are retrieved.
    std::atomic<bool> running_ = false; ///< Atomic flag to control the server's running state.
    std::thread server_thread_; ///< The main thread running the server_thread_func.
    int server_sock_ = -1; ///< The socket file descriptor for the listening server socket.
};

#endif // MJPEG_SERVER_H