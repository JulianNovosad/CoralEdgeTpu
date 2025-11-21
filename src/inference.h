#ifndef INFERENCE_H
#define INFERENCE_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <chrono>

#include "camera_capture.h" // For ImageData

// Define a structure to hold detection results
struct DetectionResult {
    int class_id;
    float score;
    float xmin, ymin, xmax, ymax; // Bounding box coordinates
    std::chrono::high_resolution_clock::time_point timestamp;
};

// Thread-safe queue for inference results
class DetectionQueue {
public:
    void push(std::vector<DetectionResult> results) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(results));
        cond_var_.notify_one();
    }

    bool pop(std::vector<DetectionResult>& results) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]{ return !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        results = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void set_running(bool val) {
        running_ = val;
        if (!val) {
            cond_var_.notify_all(); // Unblock any waiting threads
        }
    }

private:
    mutable std::mutex mutex_;
    std::queue<std::vector<DetectionResult>> queue_;
    std::condition_variable cond_var_;
    std::atomic<bool> running_ = true;
};

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path, ThreadSafeQueue& input_queue, DetectionQueue& output_queue, int num_threads = 1);
    ~InferenceEngine();

    bool start();
    void stop();
    bool is_running() const { return running_; }

private:
    void worker_thread_func();
    std::unique_ptr<tflite::Interpreter> create_interpreter();
    void set_input_tensor(tflite::Interpreter* interpreter, const ImageData& image);
    std::vector<DetectionResult> get_output_tensor(tflite::Interpreter* interpreter, const ImageData& image);

    std::string model_path_;
    ThreadSafeQueue& input_queue_;
    DetectionQueue& output_queue_;
    int num_threads_;

    std::unique_ptr<tflite::FlatBufferModel> model_;
    tflite::ops::builtin::BuiltinOpResolver resolver_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_ = false;
    std::mutex interpreter_pool_mutex_;
    std::queue<std::unique_ptr<tflite::Interpreter>> interpreter_pool_;
};

#endif // INFERENCE_H
