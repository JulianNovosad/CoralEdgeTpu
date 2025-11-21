#ifndef OUTPUT_PROCESSOR_H
#define OUTPUT_PROCESSOR_H

#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

#include "inference.h" // For DetectionQueue and DetectionResult
#include "mjpeg_server.h" // For MjpegQueue and ImageFrame
#include "jpeg_wrapper.h" // For JpegCompressGuard

// Forward declare for ImageFrame (if needed, otherwise define directly)
struct ImageData; // From camera_capture.h

class OutputProcessor {
public:
    OutputProcessor(DetectionQueue& input_detection_queue, MjpegQueue& output_mjpeg_queue, const std::vector<std::string>& labels);
    ~OutputProcessor();

    bool start();
    void stop();
    bool is_running() const { return running_; }

private:
    void process_thread_func();
    std::vector<uint8_t> draw_boxes_and_encode_jpeg(const ImageData& original_image, const std::vector<DetectionResult>& detections);

    DetectionQueue& input_detection_queue_;
    MjpegQueue& output_mjpeg_queue_;
    const std::vector<std::string>& labels_; // Store labels

    std::atomic<bool> running_ = false;
    std::thread process_thread_;
    JpegCompressGuard jpeg_compressor_;
};

#endif // OUTPUT_PROCESSOR_H
