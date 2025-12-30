#include "inference.h"
#include "config_loader.h"
#include "util_logging.h"
#include "buffer_pool.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

// Simple test to measure TPU inference rate without logging
int main() {
    // Load configuration
    ConfigLoader config_loader;
    if (!config_loader.load("/home/pi/CoralEdgeTpu/config.json")) {
        std::cerr << "Failed to load configuration" << std::endl;
        return 1;
    }
    
    // Create buffer pools
    auto image_pool = std::make_shared<BufferPool<uint8_t>>(10, 320 * 320 * 3, 320 * 320 * 3, "image_pool"); // RGB888 format
    auto detection_pool = std::make_shared<BufferPool<DetectionResult>>(10, 100, "detection_pool"); // Up to 100 detections
    
    // Create queues
    ImageQueue tpu_inference_queue;
    TripleBuffer<DetectionResults> detection_results_for_overlay_buffer;
    DetectionResultsQueue detection_results_for_logic_queue;
    
    // Create inference engine
    InferenceEngine inference_engine(
        "/home/pi/CoralEdgeTpu/detect_int8_edgetpu.tflite",
        tpu_inference_queue,
        &detection_results_for_overlay_buffer,
        detection_results_for_logic_queue,
        detection_pool,
        config_loader.get_detection_score_threshold(),
        config_loader.get_inference_worker_threads()
    );
    
    // Start inference engine
    if (!inference_engine.start()) {
        std::cerr << "Failed to start inference engine" << std::endl;
        return 1;
    }
    
    std::cout << "Inference engine started. Feeding dummy frames..." << std::endl;
    
    // Feed dummy frames for 10 seconds
    auto start_time = std::chrono::high_resolution_clock::now();
    std::atomic<int> frame_count(0);
    
    // Producer thread to feed dummy frames
    std::thread producer([&]() {
        while (std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::high_resolution_clock::now() - start_time).count() < 10) {
            
            // Create dummy image data
            ImageData dummy_frame;
            dummy_frame.width = 320;
            dummy_frame.height = 320;
            dummy_frame.format = libcamera::formats::RGB888;
            dummy_frame.timestamp_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            dummy_frame.frame_id = frame_count.load();
            
            // Acquire buffer and fill with dummy data
            auto buffer = image_pool->acquire();
            if (buffer) {
                size_t buffer_size = 320 * 320 * 3; // RGB888
                buffer->data.resize(buffer_size);
                // Fill with dummy data
                memset(buffer->data.data(), 0, buffer_size);
                buffer->size = buffer_size;
                dummy_frame.buffer = buffer;
                
                // Push to queue
                if (tpu_inference_queue.push(std::move(dummy_frame))) {
                    frame_count++;
                }
            }
            
            // No delay to maximize frame arrival rate
            // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    // Let it run for 10 seconds
    producer.join();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Test completed in " << duration << " ms" << std::endl;
    std::cout << "Frames processed: " << frame_count.load() << std::endl;
    std::cout << "Effective FPS: " << (frame_count.load() * 1000.0 / duration) << std::endl;
    
    // Stop inference engine
    inference_engine.stop();
    
    return 0;
}