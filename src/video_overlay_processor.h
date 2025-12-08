#ifndef VIDEO_OVERLAY_PROCESSOR_H
#define VIDEO_OVERLAY_PROCESSOR_H

#include "pipeline_structs.h" // Assuming this contains necessary data structures
#include <vector>
#include <string>
#include <atomic>

class VideoOverlayProcessor {
public:
    VideoOverlayProcessor(ImageQueue& main_camera_output_queue,
                          DetectionResultsQueue& detection_results_for_overlay_queue,
                          ImageQueue& overlaid_video_queue,
                          const std::vector<std::string>& labels);
    
    bool start();
    bool is_running() const;
    void stop();

    void processOverlay(ImageData& image, const DetectionResult& detection);

private:
    ImageQueue& main_camera_output_queue_;
    DetectionResultsQueue& detection_results_for_overlay_queue_;
    ImageQueue& overlaid_video_queue_;
    const std::vector<std::string>& labels_;

    std::atomic<bool> running_;
};

#endif // VIDEO_OVERLAY_PROCESSOR_H