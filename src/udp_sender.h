/**
 * @file udp_sender.h
 * @brief Defines the UdpSender class for sending object detection results via UDP.
 *
 * This class implements a UDP client that sends serialized object detection
 * results to a specified IP address and port. It retrieves detection data
 * from a thread-safe queue populated by the inference engine and converts
 * them into a JSON format for transmission.
 */

#ifndef UDP_SENDER_H
#define UDP_SENDER_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <netinet/in.h> // For sockaddr_in structure

#include "pipeline_structs.h" // Use the new central header

// Define the queue type for this module
/// @brief Type alias for the specific UdpQueue used by UdpSender.
using UdpQueue = ThreadSafeQueue<std::vector<DetectionResult>>;

/**
 * @brief Sends object detection results over UDP.
 *
 * The UdpSender class establishes a UDP connection to a target IP address and port.
 * It runs a dedicated thread that continuously retrieves `std::vector<DetectionResult>`
 * objects from an input queue, serializes them into a JSON string, and sends
 * the JSON payload via UDP.
 */
class UdpSender {
public:
    /**
     * @brief Constructor for UdpSender.
     *
     * Initializes the UDP sender with the target IP address and port, and a
     * reference to the input queue from which detection results will be retrieved.
     *
     * @param target_ip The IP address of the target UDP receiver.
     * @param target_port The UDP port number of the target receiver.
     * @param input_queue Reference to the thread-safe UdpQueue providing detection results.
     */
    UdpSender(const std::string& target_ip, int target_port, UdpQueue& input_queue);

    /**
     * @brief Destructor for UdpSender.
     *
     * Ensures that the sender thread is gracefully stopped and resources are released.
     */
    ~UdpSender();

    /**
     * @brief Starts the UDP sender.
     *
     * Creates and launches the main sender thread that retrieves and transmits data.
     *
     * @return True if the sender started successfully, false otherwise.
     */
    bool start();

    /**
     * @brief Stops the UDP sender.
     *
     * Signals the sender thread to terminate, closes the UDP socket, and joins
     * the sender thread for a clean shutdown.
     */
    void stop();

    /**
     * @brief Checks if the UDP sender is currently running.
     *
     * @return True if the sender is running, false otherwise.
     */
    bool is_running() const { return running_; }

private:
    /**
     * @brief The main loop for the UDP sender thread.
     *
     * This function continuously retrieves detection results from the input queue,
     * converts them to a JSON string, and sends them to the configured UDP target.
     */
    void sender_thread_func();

    /**
     * @brief Converts a vector of DetectionResult objects into a JSON string.
     *
     * The JSON format typically includes an array of objects, where each object
     * represents a detection with its class ID, score, and bounding box coordinates.
     *
     * @param results A constant reference to a vector of DetectionResult objects.
     * @return A JSON formatted string representing the detection results.
     */
    std::string detection_to_json(const std::vector<DetectionResult>& results);

    std::string target_ip_; ///< The IP address of the target UDP receiver.
    int target_port_; ///< The UDP port number of the target receiver.
    UdpQueue& input_queue_; ///< Reference to the queue providing detection results.
    std::atomic<bool> running_ = false; ///< Atomic flag to control the sender's running state.
    std::thread sender_thread_; ///< The main thread running the sender_thread_func.
    int sockfd_ = -1; ///< The socket file descriptor for the UDP socket.
    sockaddr_in server_addr_; ///< Structure holding the target server's address information.
};

#endif // UDP_SENDER_H