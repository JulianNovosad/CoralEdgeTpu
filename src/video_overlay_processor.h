#ifndef VIDEO_OVERLAY_PROCESSOR_H
#define VIDEO_OVERLAY_PROCESSOR_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory> // For std::unique_ptr

#include "pipeline_structs.h"
#include "jpeg_wrapper.h" // For JpegCompressGuard

/**
 * @brief Processes raw image data and overlays object detection bounding boxes.
 *
 * This class consumes raw ImageData and corresponding DetectionResults,
 * draws the bounding boxes and labels onto the image data, and then
 * compresses the overlaid image into MJPEG format, pushing the result
 * to an output MjpegQueue.
 */
class VideoOverlayProcessor {
public:
    /**
     * @brief Constructor for VideoOverlayProcessor.
     *
     * Initializes the processor with references to input queues for image data
     * and detection results, and an output queue for overlaid MJPEG frames.
     *
     * @param inference_image_queue Reference to the ImageQueue providing raw image data.
     * @param detection_results_queue Reference to the UdpQueue providing detection results.
     * @param overlaid_mjpeg_output_queue Reference to the MjpegQueue for overlaid MJPEG frames.
     * @param labels Reference to a vector of strings for class labels.
     */
    VideoOverlayProcessor(ImageQueue& inference_image_queue,
                          UdpQueue& detection_results_queue,
                          MjpegQueue& overlaid_mjpeg_output_queue,
                          const std::vector<std::string>& labels);

    /**
     * @brief Destructor for VideoOverlayProcessor.
     *
     * Ensures the processing thread is gracefully stopped.
     */
    ~VideoOverlayProcessor();

    /**
     * @brief Starts the video overlay processing thread.
     *
     * @return True if the thread started successfully, false otherwise.
     */
    bool start();

    /**
     * @brief Stops the video overlay processing thread.
     */
    void stop();

    /**
     * @brief Checks if the processor is currently running.
     *
     * @return True if running, false otherwise.
     */
    bool is_running() const { return running_; }

private:
    /**
     * @brief The main loop for the video overlay processing thread.
     *
     * This function continuously retrieves image data and detection results,
     * performs overlay, and pushes the resulting MJPEG frame to the output queue.
     */
    void processing_thread_func();

    /**
     * @brief Overlays bounding boxes and labels onto the raw image data.
     *
     * This method uses basic pixel manipulation to draw rectangles and text
     * directly onto the `ImageData` buffer.
     *
     * @param image_data The raw image data to draw on.
     * @param detections The detection results to overlay.
     */
    void draw_overlays(ImageData& image_data, const std::vector<DetectionResult>& detections);

    /**
     * @brief Converts RGB pixel data to YCbCr (YUV) as often expected by JPEG.
     *
     * @param r Red component (0-255).
     * @param g Green component (0-255).
     * @param b Blue component (0-255).
     * @param y Output Y component.
     * @param cb Output Cb component.
     * @param cr Output Cr component.
     */
    void rgb_to_ycbcr(uint8_t r, uint8_t g, uint8_t b, uint8_t& y, uint8_t& cb, uint8_t& cr);


    ImageQueue& inference_image_queue_; ///< Input queue for raw image data.
    UdpQueue& detection_results_queue_; ///< Input queue for detection results.
    MjpegQueue& overlaid_mjpeg_output_queue_; ///< Output queue for overlaid MJPEG frames.
    const std::vector<std::string>& labels_; ///< Reference to class labels for text overlay.

    std::atomic<bool> running_ = false; ///< Atomic flag to control the thread's running state.
    std::thread processing_thread_; ///< The dedicated thread for video overlay processing.
    
    // JpegCompressGuard for converting overlaid raw images to MJPEG
    JpegCompressGuard jpeg_compressor_;
};

#endif // VIDEO_OVERLAY_PROCESSOR_H