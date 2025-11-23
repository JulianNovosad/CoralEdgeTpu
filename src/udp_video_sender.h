#ifndef UDP_VIDEO_SENDER_H
#define UDP_VIDEO_SENDER_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <netinet/in.h> // For sockaddr_in

#include "pipeline_structs.h" // For MjpegQueue, ImageFrame

/**
 * @brief Sends raw MJPEG video frames via UDP.
 *
 * This class implements a UDP client that sends MJPEG frames to a specified
 * IP address and port. It retrieves MJPEG frames from a thread-safe queue
 * and sends them directly as UDP packets. Each frame may be fragmented
 * if its size exceeds the UDP packet size limit.
 */
class UdpVideoSender {
public:
    /**
     * @brief Constructor for UdpVideoSender.
     *
     * Initializes the UDP sender with the target IP address and port, and a
     * reference to the input queue from which MJPEG frames will be retrieved.
     *
     * @param target_ip The IP address of the target UDP receiver (mobile app).
     * @param target_port The UDP port number of the target receiver (e.g., 50000).
     * @param input_queue Reference to the thread-safe MjpegQueue providing MJPEG frames.
     */
    UdpVideoSender(const std::string& target_ip, int target_port, MjpegQueue& input_queue);

    /**
     * @brief Destructor for UdpVideoSender.
     *
     * Ensures that the sender thread is gracefully stopped and resources are released.
     */
    ~UdpVideoSender();

    /**
     * @brief Starts the UDP video sender.
     *
     * Creates and launches the main sender thread that retrieves and transmits MJPEG data.
     *
     * @return True if the sender started successfully, false otherwise.
     */
    bool start();

    /**
     * @brief Stops the UDP video sender.
     *
     * Signals the sender thread to terminate, closes the UDP socket, and joins
     * the sender thread for a clean shutdown.
     */
    void stop();

    /**
     * @brief Checks if the UDP video sender is currently running.
     *
     * @return True if the sender is running, false otherwise.
     */
    bool is_running() const { return running_; }

private:
    /**
     * @brief The main loop for the UDP video sender thread.
     *
     * This function continuously retrieves MJPEG frames from the input queue
     * and sends them to the configured UDP target. Frames larger than
     * the maximum UDP payload size are fragmented.
     */
    void sender_thread_func();

    /**
     * @brief Sends a single MJPEG frame, possibly fragmented, over UDP.
     *
     * @param frame The ImageFrame containing the MJPEG data.
     */
    void send_mjpeg_frame(const ImageFrame& frame);

    std::string target_ip_; ///< The IP address of the target UDP receiver.
    int target_port_; ///< The UDP port number of the target receiver.
    MjpegQueue& input_queue_; ///< Reference to the queue providing MJPEG frames.
    std::atomic<bool> running_ = false; ///< Atomic flag to control the sender's running state.
    std::thread sender_thread_; ///< The main thread running the sender_thread_func.
    int sockfd_ = -1; ///< The socket file descriptor for the UDP socket.
    sockaddr_in server_addr_; ///< Structure holding the target server's address information.

    static const size_t MAX_UDP_PAYLOAD_SIZE = 1400; ///< Maximum payload size for UDP (considering Ethernet MTU).
};

#endif // UDP_VIDEO_SENDER_H