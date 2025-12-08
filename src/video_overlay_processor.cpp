#include "video_overlay_processor.h"
#include "util_logging.h" // For LOG_INFO, LOG_ERROR etc.

VideoOverlayProcessor::VideoOverlayProcessor(ImageQueue& main_camera_output_queue,
                                             DetectionResultsQueue& detection_results_for_overlay_queue,
                                             ImageQueue& overlaid_video_queue,
                                             const std::vector<std::string>& labels)
    : main_camera_output_queue_(main_camera_output_queue),
      detection_results_for_overlay_queue_(detection_results_for_overlay_queue),
      overlaid_video_queue_(overlaid_video_queue),
      labels_(labels),
      running_(false) {
    LOG_INFO("VideoOverlayProcessor initialized.");
}

bool VideoOverlayProcessor::start() {
    if (running_.exchange(true)) {
        LOG_WARNING("VideoOverlayProcessor is already running.");
        return true;
    }
    LOG_INFO("VideoOverlayProcessor starting...");
    // Add actual startup logic here if needed, e.g., thread creation
    return true;
}

bool VideoOverlayProcessor::is_running() const {
    return running_.load();
}

void VideoOverlayProcessor::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    LOG_INFO("VideoOverlayProcessor stopping...");
    // Add actual shutdown logic here if needed, e.g., joining threads
}

void VideoOverlayProcessor::processOverlay(ImageData& image, const DetectionResult& detection) {
    // Process overlay implementation - currently a stub
    // This function will eventually draw bounding boxes, reticles, etc.
    // For now, it just passes the image through.
    if (overlaid_video_queue_.push(std::move(image))) {
        LOG_DEBUG("VideoOverlayProcessor: Pushed image to overlaid_video_queue.");
    } else {
        LOG_WARNING("VideoOverlayProcessor: Failed to push image to overlaid_video_queue (queue full).");
    }
}