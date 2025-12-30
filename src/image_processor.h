#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <opencv2/opencv.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include "pipeline_structs.h"
#include <libcamera/pixel_format.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <map>

class ImageProcessor {
public:
        // Constructor for processors that apply detection overlays
        ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                       TripleBuffer<DetectionResults>* detection_buffer,
                       std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                       libcamera::PixelFormat input_pixel_format,
                       int output_width, int output_height);
        
        // Constructor for processors that only do basic processing (like for TPU inference)
        ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                       std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                       libcamera::PixelFormat input_pixel_format,
                       int output_width, int output_height);
        ~ImageProcessor();
    
        bool start();
            void stop();
            bool is_running() const;
        
            // Set skip factor (process only every Nth frame)
            void set_skip_factor(int skip_factor) { skip_factor_ = skip_factor; }
        
            // Timing methods for monitoring
            long long get_queue_pop_timing_us() const { return avg_queue_pop_time_us_; }
            long long get_preprocess_timing_us() const { return avg_preprocess_time_us_; }
        
        // Method to set application reference for updating counters
        void set_application_ref(class Application* app) { app_ref_ = app; }
    
    private:
        void worker_thread_func();
        void apply_detections_to_frame(cv::Mat& frame, const DetectionResults& detections);
    
        ImageQueue& input_queue_;
        ImageQueue& output_queue_;
        TripleBuffer<DetectionResults>* detection_buffer_ptr_;  // Pointer to triple buffer (null for non-overlay processors)
        std::shared_ptr<BufferPool<uint8_t>> buffer_pool_;
        libcamera::PixelFormat input_pixel_format_;
        int output_width_;
        int output_height_;
        int skip_factor_ = 1;
        uint64_t frame_counter_ = 0;
    
        std::atomic<bool> running_{false};
        std::thread worker_thread_;
        
        // Timing statistics
        mutable std::atomic<long long> avg_queue_pop_time_us_{0};
        mutable std::atomic<long long> avg_preprocess_time_us_{0};
    
        // Caching for sticky detections
        DetectionResults last_detections_;
        std::chrono::steady_clock::time_point last_detection_time_;

        // Zero-Copy FD Cache
        struct MappedBuffer {
            void* start;
            size_t length;
            int internal_fd; // Duplicated FD to ensure ownership and validity
        };
        
        struct BufferKey {
            dev_t dev;
            ino_t ino;
            bool operator<(const BufferKey& other) const {
                if (dev != other.dev) return dev < other.dev;
                return ino < other.ino;
            }
        };

        std::map<BufferKey, MappedBuffer> fd_map_; 
        std::mutex fd_map_mutex_;

        // Application reference for updating counters
        class Application* app_ref_ = nullptr;
    };
    