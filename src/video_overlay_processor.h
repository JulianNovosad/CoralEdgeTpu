#ifndef VIDEO_OVERLAY_PROCESSOR_H
#define VIDEO_OVERLAY_PROCESSOR_H

#include "pipeline_structs.h"

class VideoOverlayProcessor {
public:
    VideoOverlayProcessor(ImageQueue& input_video_queue,
                          DetectionResultsQueue& input_detection_queue,
                          ImageQueue& output_video_queue);
    ~VideoOverlayProcessor();

    bool start();
    void stop();

private:
    void worker_thread_func();

    ImageQueue& input_video_queue_;
    DetectionResultsQueue& input_detection_queue_;
    ImageQueue& output_queue_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
};

#endif // VIDEO_OVERLAY_PROCESSOR_H
