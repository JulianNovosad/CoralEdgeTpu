#include "video_overlay_processor.h"
#include "util_logging.h"
#include <opencv2/opencv.hpp>

VideoOverlayProcessor::VideoOverlayProcessor(ImageQueue& input_video_queue,
                                         DetectionResultsQueue& input_detection_queue,
                                         ImageQueue& output_video_queue,
                                         const std::vector<std::string>& labels)
    : input_video_queue_(input_video_queue),
      input_detection_queue_(input_detection_queue),
      output_queue_(output_video_queue),
      labels_(labels),
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
    worker_thread_ = std::thread(&VideoOverlayProcessor::worker_thread_func, this);
    LOG_INFO("VideoOverlayProcessor started.");
    return true;
}

void VideoOverlayProcessor::stop() {
    if (running_.exchange(false)) {
        LOG_INFO("Stopping VideoOverlayProcessor...");
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        LOG_INFO("VideoOverlayProcessor stopped.");
    }
}

void VideoOverlayProcessor::worker_thread_func() {
    ImageData image_data;
    std::shared_ptr<DetectionResultBuffer> detections_buffer;

    while (running_) {
        if (input_video_queue_.pop(image_data)) {
            // Attempt to get the latest detection results. If none available, latest_detections_ will remain empty.
            if (input_detection_queue_.pop(detections_buffer)) {
                if (detections_buffer && detections_buffer->size > 0) {
                    latest_detections_.assign(detections_buffer->data.begin(), detections_buffer->data.begin() + detections_buffer->size);
                } else {
                    latest_detections_.clear(); // No detections or empty buffer
                }
            } else {
                latest_detections_.clear(); // No new detections popped
            }

            // Convert ImageData to cv::Mat, wrapping the data from the pooled buffer
            cv::Mat frame(image_data.height, image_data.width, CV_8UC3, image_data.buffer->data.data());

            // Draw bounding boxes and labels
            for (const auto& detection : latest_detections_) {
                // Coordinates are now absolute pixels from the inference engine
                int xmin = static_cast<int>(detection.xmin);
                int ymin = static_cast<int>(detection.ymin);
                int xmax = static_cast<int>(detection.xmax);
                int ymax = static_cast<int>(detection.ymax);

                cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2); // Green rectangle
                
                char label_buffer[256];
                if (detection.class_id >= 0 && static_cast<size_t>(detection.class_id) < labels_.size()) {
                    snprintf(label_buffer, sizeof(label_buffer), "%s: %d%%", labels_[detection.class_id].c_str(), static_cast<int>(detection.score * 100));
                } else {
                    snprintf(label_buffer, sizeof(label_buffer), "ID %d: %d%%", detection.class_id, static_cast<int>(detection.score * 100));
                }
                cv::putText(frame, label_buffer, cv::Point(xmin, ymin - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }

            // The drawing is done in-place on the buffer. Just move the original ImageData object
            // with its shared_ptr to the next queue.
            output_queue_.push(std::move(image_data));
        }
    }
}
