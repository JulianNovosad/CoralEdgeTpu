/**
 * @file mjpeg_capture.h
 * @brief Defines the MjpegCapture class for managing a dedicated MJPEG camera stream
 *        using rpicam-vid subprocess and robust supervision.
 *
 * This class is responsible for launching and supervising an rpicam-vid subprocess
 * that captures an MJPEG video stream. It reads MJPEG frames from the subprocess's
 * stdout pipe, parses them, and pushes them into a thread-safe queue for consumption
 * by an MJPEG web server or other preview components. It leverages the ProcessSupervisor
 * for robust process management, including restart policies and watchdog functionality.
 */

#ifndef MJPEG_CAPTURE_H
#define MJPEG_CAPTURE_H

#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include <chrono>
#include <memory> // For std::unique_ptr

#include "pipeline_structs.h"
#include "process_supervisor.hpp" // Include the new supervisor

/**
 * @brief Manages a dedicated MJPEG camera stream using an rpicam-vid subprocess.
 *
 * This class encapsulates the logic for capturing an MJPEG video stream from the
 * camera via an rpicam-vid subprocess. It utilizes the ProcessSupervisor to
 * ensure the stability and reliability of the subprocess. MJPEG frames are
 * read from the subprocess's pipe, parsed, and then pushed into a shared queue.
 */
class MjpegCapture {
public:
    /**
     * @brief Constructor for MjpegCapture.
     *
     * Initializes the MJPEG capture module with specified dimensions, frame rate,
     * and a reference to the output queue where parsed MJPEG frames will be pushed.
     * It also sets up the ProcessSupervisor for robust subprocess management.
     *
     * @param width The desired width of the MJPEG stream.
     * @param height The desired height of the MJPEG stream.
     * @param fps The desired frames per second of the MJPEG stream.
     * @param output_queue Reference to the thread-safe MjpegQueue for output.
     * @param watchdog_timeout The duration after which a lack of activity in the
     *                         stream will trigger a subprocess restart (e.g., 5 seconds).
     */
    MjpegCapture(unsigned int width, unsigned int height, unsigned int fps, MjpegQueue& output_queue, std::chrono::seconds watchdog_timeout);

    /**
     * @brief Destructor for MjpegCapture.
     *
     * Ensures that the capture process and associated threads are gracefully stopped.
     */
    ~MjpegCapture();

    /**
     * @brief Starts the MJPEG camera capture process and its supervisor.
     *
     * Launches the rpicam-vid subprocess and initiates the monitoring and
     * pipe reading threads.
     *
     * @return True if the module started successfully, false otherwise.
     */
    bool start();

    /**
     * @brief Stops the MJPEG camera capture process and its supervisor.
     *
     * Sends termination signals to the subprocess and joins all associated threads
     * for a clean shutdown.
     */
    void stop();

    /**
     * @brief Checks if the MJPEG capture module is currently running.
     *
     * @return True if the module is running, false otherwise.
     */
    bool is_running() const; // Delegates to supervisor's is_running()

private:
    unsigned int width_;  ///< The width of the captured MJPEG stream.
    unsigned int height_; ///< The height of the captured MJPEG stream.
    unsigned int fps_;    ///< The frames per second of the captured MJPEG stream.
    MjpegQueue& output_queue_; ///< Reference to the queue where processed MJPEG frames are pushed.
    
    /**
     * @brief Generates the command-line arguments for the rpicam-vid subprocess.
     *
     * Constructs a vector of strings representing the arguments to be passed to
     * `/usr/bin/rpicam-vid` for MJPEG capture. Includes parameters for codec,
     * dimensions, framerate, and output to stdout. Logs the configuration as JSON.
     *
     * @return A vector of strings containing the rpicam-vid command arguments.
     */
    std::vector<std::string> get_command_args();

    /**
     * @brief Parses raw byte data from the pipe into complete MJPEG ImageFrame objects.
     *
     * This function is passed as a callback to the ProcessSupervisor. It accumulates
     * bytes in the buffer and searches for JPEG Start of Image (FF D8) and End of Image
     * (FF D9) markers to extract complete frames. Found frames are pushed to the
     * output queue. Consumed bytes are removed from the buffer.
     *
     * @param buffer A reference to the buffer accumulating raw byte data.
     * @param bytes_read The number of new bytes read in the current read operation (not directly used for parsing,
     *                   the entire buffer content is processed).
     * @param queue The output MjpegQueue to push parsed ImageFrame objects.
     * @return True if at least one complete frame was parsed and pushed, false otherwise.
     */
    bool parse_frame_data(std::vector<uint8_t>& buffer, size_t bytes_read, MjpegQueue& queue);

    /// Unique pointer to the ProcessSupervisor instance managing the rpicam-vid subprocess.
    std::unique_ptr<ProcessSupervisor<MjpegQueue, ImageFrame>> supervisor_;
};

#endif // MJPEG_CAPTURE_H
