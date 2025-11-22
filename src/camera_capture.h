/**
 * @file camera_capture.h
 * @brief Defines the CameraCapture class for managing a dedicated raw image stream
 *        using rpicam-vid subprocess and robust supervision.
 *
 * This class is responsible for launching and supervising an rpicam-vid subprocess
 * that captures a raw image stream (e.g., BGR888). It reads raw image data from the
 * subprocess's stdout pipe, converts it into ImageData objects, and pushes them
 * into a thread-safe queue for consumption by an inference engine. It leverages
 * the ProcessSupervisor for robust process management, including restart policies
 * and watchdog functionality.
 */

#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include <chrono>
#include <memory> // For std::unique_ptr

#include "pipeline_structs.h"
#include "process_supervisor.hpp" // Include the new supervisor

/**
 * @brief Manages a dedicated raw image camera stream using an rpicam-vid subprocess.
 *
 * This class encapsulates the logic for capturing a raw video stream from the
 * camera via an rpicam-vid subprocess. It utilizes the ProcessSupervisor to
 * ensure the stability and reliability of the subprocess. Raw image data is
 * read from the subprocess's pipe, converted, and then pushed into a shared queue.
 */
class CameraCapture {
public:
    /**
     * @brief Constructor for CameraCapture.
     *
     * Initializes the camera capture module with specified dimensions and a reference
     * to the output queue where parsed image data will be pushed. It also sets up
     * the ProcessSupervisor for robust subprocess management.
     *
     * @param width The desired width of the raw image stream.
     * @param height The desired height of the raw image stream.
     * @param output_queue Reference to the thread-safe ImageQueue for output.
     * @param watchdog_timeout The duration after which a lack of activity in the
     *                         stream will trigger a subprocess restart (e.g., 5 seconds).
     */
    CameraCapture(unsigned int width, unsigned int height, ImageQueue& output_queue, std::chrono::seconds watchdog_timeout);

    /**
     * @brief Destructor for CameraCapture.
     *
     * Ensures that the capture process and associated threads are gracefully stopped.
     */
    ~CameraCapture();

    /**
     * @brief Starts the camera capture process and its supervisor.
     *
     * Launches the rpicam-vid subprocess and initiates the monitoring and
     * pipe reading threads.
     *
     * @return True if the module started successfully, false otherwise.
     */
    bool start();

    /**
     * @brief Stops the camera capture process and its supervisor.
     *
     * Sends termination signals to the subprocess and joins all associated threads
     * for a clean shutdown.
     */
    void stop();

    /**
     * @brief Checks if the camera capture module is currently running.
     *
     * @return True if the module is running, false otherwise.
     */
    bool is_running() const; // Delegates to supervisor's is_running()

private:
    unsigned int width_;  ///< The width of the captured raw image stream.
    unsigned int height_; ///< The height of the captured raw image stream.
    ImageQueue& output_queue_; ///< Reference to the queue where processed ImageData objects are pushed.
    
    /**
     * @brief Generates the command-line arguments for the rpicam-vid subprocess.
     *
     * Constructs a vector of strings representing the arguments to be passed to
     * `/usr/bin/rpicam-vid` for raw image capture (e.g., BGR888). Includes
     * parameters for codec, dimensions, and output to stdout. Logs the configuration as JSON.
     *
     * @return A vector of strings containing the rpicam-vid command arguments.
     */
    std::vector<std::string> get_command_args();

    /**
     * @brief Parses raw byte data from the pipe into complete ImageData objects.
     *
     * This function is passed as a callback to the ProcessSupervisor. It accumulates
     * bytes in the buffer and, once a full frame's worth of data is received, it
     * constructs an ImageData object and pushes it to the output queue. Consumed
     * bytes are removed from the buffer.
     *
     * @param buffer A reference to the buffer accumulating raw byte data.
     * @param bytes_read The number of new bytes read in the current read operation (not directly used for parsing,
     *                   the entire buffer content is processed).
     * @param queue The output ImageQueue to push parsed ImageData objects.
     * @return True if at least one complete frame was parsed and pushed, false otherwise.
     */
    bool parse_frame_data(std::vector<uint8_t>& buffer, size_t bytes_read, ImageQueue& queue);

    /// Unique pointer to the ProcessSupervisor instance managing the rpicam-vid subprocess.
    std::unique_ptr<ProcessSupervisor<ImageQueue, ImageData>> supervisor_;
};

#endif // CAMERA_CAPTURE_H
