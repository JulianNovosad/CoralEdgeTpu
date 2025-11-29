#include "video_overlay_processor.h"
#include "util_logging.h"
#include <opencv2/opencv.hpp>

VideoOverlayProcessor::VideoOverlayProcessor(ImageQueue& input_video_queue,
                                         DetectionResultsQueue& input_detection_queue,
                                         ImageQueue& output_video_queue)
    : input_video_queue_(input_video_queue),
      input_detection_queue_(input_detection_queue),
      output_queue_(output_video_queue),
      running_(false) {
    // Constructor logic
}

VideoOverlayProcessor::~VideoOverlayProcessor() {
    stop();
    LOG_INFO("VideoOverlayProcessor destroyed.");
}

bool VideoOverlayProcessor::start() {
    if (running_) {
        LOG_ERROR("VideoOverlayProcessor is already running.");
        return false;
    }
    running_ = true;
    input_video_queue_.set_running(true);
    input_detection_queue_.set_running(true); 
    output_queue_.set_running(true);
    worker_thread_ = std::thread(&VideoOverlayProcessor::worker_thread_func, this);
    LOG_INFO("VideoOverlayProcessor started.");
    return true;
}

void VideoOverlayProcessor::stop() {
    if (running_.exchange(false)) {
        LOG_INFO("Stopping VideoOverlayProcessor...");
        input_video_queue_.set_running(false);
        input_detection_queue_.set_running(false); // Also set running for the new queue
        output_queue_.set_running(false);
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        LOG_INFO("VideoOverlayProcessor stopped.");
    }
}

void VideoOverlayProcessor::worker_thread_func() {
    ImageData image_data;
    std::vector<DetectionResult> latest_detections;

    while (running_) {
        if (input_video_queue_.pop(image_data)) {
            // Peek latest detections. If none available, use an empty vector.
            // Using peek_latest to not consume detections if multiple frames arrive before new detections.
            if (!input_detection_queue_.peek_latest(latest_detections)) {
                latest_detections.clear(); // Clear old detections if no new ones are available or queue is empty
            }

            // Convert ImageData to cv::Mat
            cv::Mat frame(image_data.height, image_data.width, CV_8UC3, image_data.data.data());

            // Draw bounding boxes and labels
            for (const auto& detection : latest_detections) {
                // Assuming bounding box coordinates are normalized [0,1]
                int xmin = static_cast<int>(detection.xmin * image_data.width);
                int ymin = static_cast<int>(detection.ymin * image_data.height);
                int xmax = static_cast<int>(detection.xmax * image_data.width);
                int ymax = static_cast<int>(detection.ymax * image_data.height);

                cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2); // Green rectangle
                
                // For label, we'd need access to labels vector, which VideoOverlayProcessor doesn't have.
                // For now, just class ID and score.
                std::string label = "Class: " + std::to_string(detection.class_id) + " Score: " + std::to_string(static_cast<int>(detection.score * 100)) + "%";
                cv::putText(frame, label, cv::Point(xmin, ymin - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }

            // Convert cv::Mat back to ImageData
            // No need to copy if image_data.data is directly modified by cv::Mat (CV_8UC3, image_data.data.data())
            // But if frame is re-allocated (e.g., resizing or new Mat after drawing), we need to copy back
            // For drawing, frame operates on image_data.data directly, so no copy back for data is strictly needed unless size changed.
            // However, it's safer to ensure the ImageData is fully updated.
            image_data.data.assign(frame.data, frame.data + (frame.total() * frame.elemSize()));
            
            output_queue_.push(image_data);
        }
    }
}
