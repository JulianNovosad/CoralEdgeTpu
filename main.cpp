// camera_producer_host.cpp
// A multi-threaded camera producer that provides two streams:
// 1. 100 FPS raw frames to a Unix socket for a local TPU.
// 2. 45 FPS JPEG-encoded frames to a TCP socket for a remote client (phone).
//
// Compilation:
// g++ -o camera_producer_host camera_producer_host.cpp -O3 -std=c++17 -lcamera -pthread

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/op_resolver.h>
#include "edgetpu/edgetpu_delegate_for_custom_op.h"
#include <tensorflow/lite/c/common.h> // For TfLiteDelegate definition
#include <tensorflow/lite/builtin_op_data.h>
#include <fstream> // Required for ReadLabels

#include <iostream>
#include <vector>
#include <string>
#include <csignal>
#include <thread>
#include <chrono>
#include <atomic>
#include <stdexcept>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/file.h> // For flock
#include <vector>
#include <ostream>
#include <iostream>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <sys/select.h>
#include <sys/time.h>
#include <numeric> // For std::accumulate

#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/stream.h>
#include <sys/mman.h>

using namespace libcamera;

// Global libcamera objects
static std::unique_ptr<CameraManager> camera_manager;
static std::shared_ptr<Camera> camera_obj;
static std::unique_ptr<FrameBufferAllocator> allocator;
static Stream *raw_stream = nullptr;
static std::vector<std::unique_ptr<Request>> libcamera_requests;

// --- Configuration ---
// --- Configuration ---
// const char* UNIX_SOCK_DIR = "/tmp/coral_ipc"; // Removed after UDS refactor
// const char* UNIX_SOCK_PATH = "/tmp/coral_ipc/camera.sock"; // Removed after UDS refactor
const char* LOCK_FILE = "/tmp/coral_ipc/camera_producer.lock"; // Prevent multiple instances
const char* PHONE_IP = "192.168.37.27";
const int PHONE_PORT = 8080; // TCP port for phone frames
const int UDP_PHONE_PORT = 9090; // UDP port for phone detections

// Frame dimensions
const int RAW_FRAME_WIDTH = 300;
const int RAW_FRAME_HEIGHT = 300;
const int RAW_FRAME_SIZE = RAW_FRAME_WIDTH * RAW_FRAME_HEIGHT * 3; // For BGR888

// Placeholder for JPEG frame size (actual size will vary)
const size_t JPEG_FRAME_MAX_SIZE = 300 * 1024; // 300 KB
const int MAX_QUEUE_SIZE = 10; // Max number of frames to buffer in memory

// --- Globals ---
std::atomic<bool> running(true);
std::atomic<uint32_t> tpu_frame_counter(0); // Frame ID counter for TPU thread
std::atomic<uint32_t> phone_frame_counter(0); // Frame ID counter for Phone thread

// --- Thread-safe Queue for Frames ---
template<typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= MAX_QUEUE_SIZE) {
            queue_.pop(); // Drop the oldest frame
        }
        queue_.push(std::move(value));
        cond_.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{ return !queue_.empty() || !running; });
        if (!running && queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void stop() {
        cond_.notify_all();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};

// Frame data structures
using RawFrame = std::vector<uint8_t>;
using JpegFrame = std::vector<uint8_t>;

ThreadSafeQueue<RawFrame> inference_queue;
ThreadSafeQueue<JpegFrame> phone_queue;

#include <iomanip>
#include <ctime>
#include <sstream>

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}

// --- Resource Cleanup ---
void cleanup_resources() {
    std::cout << "[" << get_timestamp() << "] Cleaning up resources..." << std::endl;
    
    // Remove Unix socket file if it exists (Removed after UDS refactor)
    // if (access(UNIX_SOCK_PATH, F_OK) == 0) {
    //     if (unlink(UNIX_SOCK_PATH) == 0) {
    //         std::cout << "[" << get_timestamp() << "] Removed Unix socket file." << std::endl;
    //     }
    // }
    
    // Remove lock file if it exists
    if (access(LOCK_FILE, F_OK) == 0) {
        if (unlink(LOCK_FILE) == 0) {
            std::cout << "[" << get_timestamp() << "] Removed lock file." << std::endl;
        }
    }
}

// --- Signal Handler ---
void signal_handler(int signum) {
    std::cout << "\n[" << get_timestamp() << "] Caught signal " << signum << ", shutting down..." << std::endl;
    running = false;
    cleanup_resources(); // Ensure cleanup on signal
}

// --- Helper function to send all data --- (Removed as UDS is no longer used)
// bool send_all(int sock, const void* data, size_t size) {
//     const char* buffer = static_cast<const char*>(data);
//     while (size > 0) {
//         ssize_t sent = send(sock, buffer, size, MSG_NOSIGNAL);
//         if (sent <= 0) {
//             if (errno == EINTR) continue; // Interrupted by signal, retry
//             perror("send");
//             return false; // Connection closed or error
//         }
//         buffer += sent;
//         size -= sent;
//     }
//     return true;
// }

// --- Image Resizing Utility (Nearest Neighbor) ---
RawFrame resize_image_rgb(const RawFrame& input_frame, int input_width, int input_height, int output_width, int output_height) {
    RawFrame output_frame(output_width * output_height * 3); // 3 for RGB
    for (int y_out = 0; y_out < output_height; ++y_out) {
        for (int x_out = 0; x_out < output_width; ++x_out) {
            // Calculate corresponding input pixel coordinates
            int x_in = static_cast<int>(static_cast<float>(x_out) / output_width * input_width);
            int y_in = static_cast<int>(static_cast<float>(y_out) / output_height * input_height);

            // Ensure coordinates are within bounds
            if (x_in >= input_width) x_in = input_width - 1;
            if (y_in >= input_height) y_in = input_height - 1;

            // Copy RGB channels
            for (int c = 0; c < 3; ++c) {
                output_frame[(y_out * output_width + x_out) * 3 + c] =
                    input_frame[(y_in * input_width + x_in) * 3 + c];
            }
        }
    }
    return output_frame;
}

extern "C" {
TfLiteDelegate* tflite_plugin_create_delegate(const void* options);
void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate);
}




// --- Thread 2: TPU Inference Thread ---
void tpu_inference_thread(const std::string& model_path, const std::string& labels_path) {
    std::cout << "[" << get_timestamp() << "] [TPU Thread] Started. Initializing TensorFlow Lite interpreter..." << std::endl;

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "[" << get_timestamp() << "] [TPU Thread] Failed to load model: " << model_path << std::endl;
        running = false;
        return;
    }

    // Build the interpreter with the Edge TPU delegate
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "[" << get_timestamp() << "] [TPU Thread] Failed to construct interpreter." << std::endl;
        running = false;
        return;
    }

    // Create the Edge TPU delegate using the C API
    TfLiteDelegate* delegate = tflite_plugin_create_delegate(nullptr); // No options for now
    if (!delegate) {
        std::cerr << "[" << get_timestamp() << "] [TPU Thread] Failed to create Edge TPU delegate using tflite_plugin_create_delegate. Ensure Edge TPU is connected and drivers are installed." << std::endl;
        running = false;
        return;
    }
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        std::cerr << "[" << get_timestamp() << "] [TPU Thread] Failed to apply Edge TPU delegate." << std::endl;
        // The delegate is owned by the interpreter after ModifyGraphWithDelegate if successful.
        // If it fails, we should free it.
        tflite_plugin_destroy_delegate(delegate);
        running = false;
        return;
    }
    // The delegate is owned by the interpreter after ModifyGraphWithDelegate if successful.
    // So, we don't need to explicitly free it here if it succeeds.

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "[" << get_timestamp() << "] [TPU Thread] Failed to allocate tensors." << std::endl;
        running = false;
        return;
    }

    // Add logging here
    TfLiteIntArray* input_dims = interpreter->input_tensor(0)->dims;
    int input_height = input_dims->data[1];
    int input_width = input_dims->data[2];
    int input_channels = input_dims->data[3];
    std::cout << "[" << get_timestamp() << "] [TPU Thread] Model expected input tensor: "
              << input_width << "x" << input_height << "x" << input_channels
              << " (" << interpreter->input_tensor(0)->bytes << " bytes)" << std::endl;

    // Read labels
    auto ReadLabels = [](const std::string& filename) -> std::vector<std::string> {
        std::vector<std::string> labels;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open labels file: " << filename << std::endl;
            return labels;
        }
        std::string line;
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        return labels;
    };
    std::vector<std::string> labels = ReadLabels(labels_path);
    if (labels.empty()) {
        std::cerr << "[" << get_timestamp() << "] [TPU Thread] Failed to load labels or labels file is empty." << std::endl;
        running = false;
        return;
    }

    // --- Benchmarking variables ---
    const int NUM_BENCHMARK_FRAMES = 1000;
    std::vector<double> inference_latencies; // in milliseconds
    inference_latencies.reserve(NUM_BENCHMARK_FRAMES);
    auto benchmark_start_time = std::chrono::high_resolution_clock::now();
    int frames_processed = 0;

    RawFrame frame;
    while (running && inference_queue.pop(frame)) {
        // Log the actual frame size from camera
        std::cout << "[" << get_timestamp() << "] [TPU Thread] Raw frame received (size: " << frame.size() << " bytes, "
                  << RAW_FRAME_WIDTH << "x" << RAW_FRAME_HEIGHT << " RGB)" << std::endl;

        // Resize the frame
        RawFrame resized_frame = resize_image_rgb(frame, RAW_FRAME_WIDTH, RAW_FRAME_HEIGHT, input_width, input_height);

        if (resized_frame.size() != interpreter->input_tensor(0)->bytes) {
            std::cerr << "[" << get_timestamp() << "] [TPU Thread] Error: Resized frame size mismatch with tensor size. Expected: " << interpreter->input_tensor(0)->bytes << ", Got: " << resized_frame.size() << std::endl;
            continue;
        }
        memcpy(interpreter->input_tensor(0)->data.uint8, resized_frame.data(), resized_frame.size());

        auto start = std::chrono::high_resolution_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "[" << get_timestamp() << "] [TPU Thread] Failed to invoke interpreter." << std::endl;
            continue;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        double current_latency_ms = diff.count() * 1000;
        inference_latencies.push_back(current_latency_ms);
        std::cout << "[" << get_timestamp() << "] [TPU Thread] Inference took " << current_latency_ms << " ms" << std::endl;

        // Get output tensor details
        TfLiteIntArray* output_dims = interpreter->output_tensor(0)->dims;
        int output_size = output_dims->data[output_dims->size - 1];

        // Get top-1 result (assuming classification model)
        const uint8_t* output = interpreter->output_tensor(0)->data.uint8;
        int top_index = 0;
        uint8_t max_score = 0;
        for (int i = 0; i < output_size; ++i) {
            if (output[i] > max_score) {
                max_score = output[i];
                top_index = i;
            }
        }

        if (top_index < labels.size()) {
            std::cout << "[" << get_timestamp() << "] [TPU Thread] Top-1 result: " << labels[top_index] << " (score: " << static_cast<int>(max_score) << ")" << std::endl;
        } else {
            std::cout << "[" << get_timestamp() << "] [TPU Thread] Top-1 index out of bounds for labels. Index: " << top_index << ", Labels size: " << labels.size() << std::endl;
        }

        frames_processed++;
        if (frames_processed >= NUM_BENCHMARK_FRAMES) {
            std::cout << "[" << get_timestamp() << "] [TPU Thread] Reached " << NUM_BENCHMARK_FRAMES << " frames. Stopping for benchmark report." << std::endl;
            running = false; // Stop the main loop
            break;
        }
    }

    // --- Benchmark Report ---
    if (!inference_latencies.empty()) {
        auto benchmark_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time_s = benchmark_end_time - benchmark_start_time;
        double total_time_ms = total_time_s.count() * 1000;

        std::sort(inference_latencies.begin(), inference_latencies.end());

        double min_latency = inference_latencies.front();
        double max_latency = inference_latencies.back();
        double avg_latency = std::accumulate(inference_latencies.begin(), inference_latencies.end(), 0.0) / inference_latencies.size();

        // Calculate percentiles
        auto get_percentile = [](const std::vector<double>& sorted_data, double percentile) {
            if (sorted_data.empty()) return 0.0;
            size_t index = static_cast<size_t>(std::ceil(percentile / 100.0 * sorted_data.size())) - 1;
            if (index >= sorted_data.size()) index = sorted_data.size() - 1; // Cap to max index
            return sorted_data[index];
        };

        double p50_latency = get_percentile(inference_latencies, 50);
        double p95_latency = get_percentile(inference_latencies, 95);
        double p99_latency = get_percentile(inference_latencies, 99);
        
        double fps = frames_processed / total_time_s.count();

        // --- Console Output (existing) ---
        std::cout << "\n[" << get_timestamp() << "] --- TPU Inference Benchmark Report ---" << std::endl;
        std::cout << "  Frames Processed: " << frames_processed << std::endl;
        std::cout << "  Total Benchmark Time: " << total_time_ms << " ms" << std::endl;
        std::cout << "  Average FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
        std::cout << "  Latency Statistics (ms):" << std::endl;
        std::cout << "    Min: " << std::fixed << std::setprecision(3) << min_latency << std::endl;
        std::cout << "    Max: " << std::fixed << std::setprecision(3) << max_latency << std::endl;
        std::cout << "    Avg: " << std::fixed << std::setprecision(3) << avg_latency << std::endl;
        std::cout << "    P50: " << std::fixed << std::setprecision(3) << p50_latency << std::endl;
        std::cout << "    P95: " << std::fixed << std::setprecision(3) << p95_latency << std::endl;
        std::cout << "    P99: " << std::fixed << std::setprecision(3) << p99_latency << std::endl;
        std::cout << "[" << get_timestamp() << "] --- End Benchmark Report ---" << std::endl;

        // --- CSV File Output ---
        const std::string csv_filename = "/home/pi/CoralEdgeTpu/DAILYLOGS/tpu_benchmark.csv";
        std::ofstream csv_file(csv_filename, std::ios::app); // Open in append mode

        // Write header if file is new/empty
        if (csv_file.tellp() == 0) {
            csv_file << "timestamp_utc,module,stage,p50_latency_ms,p95_latency_ms,p99_latency_ms,avg_fps,min_latency_ms,max_latency_ms,avg_latency_ms,frames_processed,total_time_ms\n";
        }

        // Write data row
        csv_file << get_timestamp() << "," // absolute timestamp
                 << "TPU_Inference," // module
                 << "Stage0," // stage
                 << std::fixed << std::setprecision(3) << p50_latency << ","
                 << std::fixed << std::setprecision(3) << p95_latency << ","
                 << std::fixed << std::setprecision(3) << p99_latency << ","
                 << std::fixed << std::setprecision(2) << fps << ","
                 << std::fixed << std::setprecision(3) << min_latency << ","
                 << std::fixed << std::setprecision(3) << max_latency << ","
                 << std::fixed << std::setprecision(3) << avg_latency << ","
                 << frames_processed << ","
                 << total_time_ms << "\n";
        csv_file.close();

        std::cout << "[" << get_timestamp() << "] Benchmark results saved to " << csv_filename << std::endl;
    }


    std::cout << "[" << get_timestamp() << "] [TPU Thread] Stopped." << std::endl;
}

// --- Thread 3: Phone Frame Sender ---
void phone_sender_thread() {
    std::cout << "[" << get_timestamp() << "] [Phone Thread] Started. Waiting for phone connection..." << std::endl;
    
    while(running) { // Outer loop for reconnection
        int sock = -1;
        try {
            sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) throw std::runtime_error("Failed to create TCP socket");

            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_port = htons(PHONE_PORT);
            inet_pton(AF_INET, PHONE_IP, &addr.sin_addr);

            // Set socket to non-blocking for connect
            int flags = fcntl(sock, F_GETFL, 0);
            if (flags == -1) throw std::runtime_error("fcntl F_GETFL failed");
            fcntl(sock, F_SETFL, flags | O_NONBLOCK);

            std::cout << "[" << get_timestamp() << "] [Phone Thread] Attempting to connect to phone..." << std::endl;
            
            int ret = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
            if (ret < 0 && errno != EINPROGRESS) {
                perror("[Phone Thread] connect(tcp_sock)");
            }

            fd_set fdset;
            FD_ZERO(&fdset);
            FD_SET(sock, &fdset);
            struct timeval tv;
            tv.tv_sec = 2; // 2 second timeout for select

            ret = select(sock + 1, NULL, &fdset, NULL, &tv);

            if (ret > 0) {
                int so_error;
                socklen_t len = sizeof(so_error);
                getsockopt(sock, SOL_SOCKET, SO_ERROR, &so_error, &len);
                if (so_error == 0) {
                    std::cout << "[" << get_timestamp() << "] [Phone Thread] Connected to phone." << std::endl;
                } else {
                    // Connection failed
                    if (sock != -1) close(sock);
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    continue; // Go to next iteration of outer loop
                }
            } else {
                // select failed or timed out
                if (sock != -1) close(sock);
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue; // Go to next iteration of outer loop
            }
            
            // Restore blocking mode
            fcntl(sock, F_SETFL, flags);

            // Main sending loop
            JpegFrame frame;
            while (running && phone_queue.pop(frame)) {
                uint32_t current_frame_id = phone_frame_counter.fetch_add(1);
                uint32_t frame_id_net = htonl(current_frame_id);
                uint32_t frame_size_net = htonl(frame.size());
                
                // if (!send_all(sock, &frame_id_net, sizeof(frame_id_net)) ||
                //     !send_all(sock, &frame_size_net, sizeof(frame_size_net)) ||
                //     !send_all(sock, frame.data(), frame.size())) {
                //     perror("[Phone Thread] send_all failed");
                //     break; // Break inner loop to trigger reconnection
                // }
            }
        } catch (const std::exception& e) {
            std::cerr << "[" << get_timestamp() << "] [Phone Thread] Error: " << e.what() << std::endl;
        }

        if (sock != -1) close(sock);

        if (running) {
            std::cout << "[" << get_timestamp() << "] [Phone Thread] Connection lost or client disconnected. Reconnecting in 2 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    std::cout << "[" << get_timestamp() << "] [Phone Thread] Stopped." << std::endl;
}


// Callback function for completed requests
void requestComplete(Request *request) {
    if (request->status() == Request::RequestComplete) {
        if (request->buffers().count(raw_stream) == 0) {
            std::cerr << "[" << get_timestamp() << "] Error: Completed request has no buffer for the raw stream." << std::endl;
        } else {
            FrameBuffer *buffer = request->buffers().at(raw_stream);

            if (!buffer->planes().empty()) {
                const FrameBuffer::Plane &plane = buffer->planes()[0];
                void *data = mmap(NULL, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
                if (data == MAP_FAILED) {
                    std::cerr << "[" << get_timestamp() << "] mmap failed" << std::endl;
                } else {
                    RawFrame raw_frame(plane.length);
                    // Add logging here
                    std::cout << "[" << get_timestamp() << "] [Camera Callback] Captured frame size: " << plane.length << " bytes." << std::endl;
                    inference_queue.push(std::move(raw_frame));
                    munmap(data, plane.length);
                }
            }
        }
    } else {
        std::cerr << "[" << get_timestamp() << "] Request failed with status: " << request->status() << std::endl;
    }

    // Re-queue the request for continuous capture
    if (running) {
        // We must get the buffer *before* calling reuse(), as reuse() clears the buffer map.
        if (request->buffers().count(raw_stream) > 0) {
            FrameBuffer *buffer = request->buffers().at(raw_stream);
            request->reuse();
            if (request->addBuffer(raw_stream, buffer) < 0) {
                std::cerr << "[" << get_timestamp() << "] Error: Failed to re-add buffer to request" << std::endl;
                // Don't queue if we failed to add buffer
                return;
            }
            camera_obj->queueRequest(request);
        }
    }
}

int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create directory for Unix socket (Removed after UDS refactor)
    // if (mkdir(UNIX_SOCK_DIR, 0755) == -1 && errno != EEXIST) {
    //     perror("mkdir /tmp/coral_ipc");
    //     return 1;
    // }

    // Check if another instance is running
    int lock_fd = open(LOCK_FILE, O_CREAT | O_RDWR, 0644);
    if (lock_fd < 0) {
        perror("Failed to create lock file");
        return 1;
    }
    if (flock(lock_fd, LOCK_EX | LOCK_NB) < 0) {
        std::cerr << "[" << get_timestamp() << "] Another instance is already running!" << std::endl;
        close(lock_fd);
        cleanup_resources();
        return 1;
    }

    // 1. Initialize Camera Manager
    camera_manager = std::make_unique<CameraManager>();
    camera_manager->start();

    if (camera_manager->cameras().empty()) {
        std::cerr << "[" << get_timestamp() << "] No cameras found." << std::endl;
        close(lock_fd);
        cleanup_resources();
        return 1;
    }

    // 2. Select a Camera (e.g., the first available camera)
    camera_obj = camera_manager->cameras()[0];
    camera_obj->acquire(); // Acquire the camera for exclusive use

    // 3. Configure Streams for Raw Capture
    std::unique_ptr<CameraConfiguration> config = camera_obj->generateConfiguration({StreamRole::Raw});
    if (!config) {
        std::cerr << "[" << get_timestamp() << "] Failed to generate default configuration." << std::endl;
        camera_obj->release();
        camera_manager->stop();
        close(lock_fd);
        cleanup_resources();
        return 1;
    }

    // Modify the stream configuration for raw capture
    StreamConfiguration &rawStreamConfig = config->at(0);
    rawStreamConfig.pixelFormat = libcamera::formats::RGB888; // Match what libcamera configures
    rawStreamConfig.size.width = RAW_FRAME_WIDTH;
    rawStreamConfig.size.height = RAW_FRAME_HEIGHT;
    rawStreamConfig.bufferCount = 6; // Explicitly set buffer count

    // Validate and apply the configuration
    int ret = config->validate();
    if (ret == -EINVAL) {
        std::cerr << "[" << get_timestamp() << "] Failed to validate camera configuration." << std::endl;
        camera_obj->release();
        camera_manager->stop();
        close(lock_fd);
        cleanup_resources();
        return 1;
    }
    camera_obj->configure(config.get());

    // Get the raw stream
    raw_stream = rawStreamConfig.stream();

    // 4. Allocate Frame Buffers
    allocator = std::make_unique<FrameBufferAllocator>(camera_obj);
    for (StreamConfiguration &cfg : *config) {
        int ret = allocator->allocate(cfg.stream());
        if (ret < 0) {
            std::cerr << "[" << get_timestamp() << "] Can't allocate buffers for stream. Error: " << ret << std::endl;
            camera_obj->release();
            camera_manager->stop();
            close(lock_fd);
            cleanup_resources();
            return 1;
        }
        std::cout << "[" << get_timestamp() << "] Allocated " << allocator->buffers(cfg.stream()).size()
                  << " buffers for stream." << std::endl;
    }

    // 5. Create and Queue Requests
    for (StreamConfiguration &cfg : *config) {
        Stream *stream = cfg.stream();
        const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(stream);

        std::cout << "[" << get_timestamp() << "] Creating requests for stream " << stream << " with " << buffers.size() << " buffers." << std::endl;

        for (const auto &buffer : buffers) {
            std::cout << "[" << get_timestamp() << "] Processing buffer at address: " << buffer.get() << std::endl;
            std::unique_ptr<Request> request = camera_obj->createRequest();
            if (!request) {
                std::cerr << "[" << get_timestamp() << "] Failed to create request." << std::endl;
                camera_obj->release();
                camera_manager->stop();
                close(lock_fd);
                cleanup_resources();
                return 1;
            }
            std::cout << "[" << get_timestamp() << "] Created request " << request.get() << std::endl;

            int add_buffer_ret = request->addBuffer(stream, buffer.get());
            if (add_buffer_ret < 0) {
                std::cerr << "[" << get_timestamp() << "] Failed to add buffer to request: " << add_buffer_ret << std::endl;
                camera_obj->release();
                camera_manager->stop();
                close(lock_fd);
                cleanup_resources();
                return 1; // Or handle error appropriately
            }
            std::cout << "[" << get_timestamp() << "] Added buffer " << buffer.get() << " to request " << request.get() << std::endl;

            libcamera_requests.push_back(std::move(request));
            std::cout << "[" << get_timestamp() << "] Pushed request to vector. Vector size: " << libcamera_requests.size() << std::endl;
        }
    }

    // Set the request completion callback
    camera_obj->requestCompleted.connect(requestComplete);

    // 6. Start Camera
    camera_obj->start();
    std::cout << "[" << get_timestamp() << "] [Main] Camera started." << std::endl;

    // Queue all initial requests
    for (auto &request : libcamera_requests) {
        std::cout << "[" << get_timestamp() << "] Queuing request " << request.get() << std::endl;
        camera_obj->queueRequest(request.get());
    }

    std::cout << "[" << get_timestamp() << "] [Main] Starting threads..." << std::endl;

    const std::string model_path = "/home/pi/CoralEdgeTpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
    const std::string labels_path = "/home/pi/CoralEdgeTpu/coco_labels.txt";

    std::thread t_inference(tpu_inference_thread, model_path, labels_path);
    // std::thread t3(phone_sender_thread); // Disabled for debugging OOM

    // The main thread will now wait for a signal to stop
    // The camera_capture_thread is no longer needed as a separate thread for frame generation.
    // The requestComplete callback handles frame pushing.
    while(running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 7. Stop Camera and Cleanup
    std::cout << "[" << get_timestamp() << "] [Main] Stopping camera..." << std::endl;
    camera_obj->stop();
    inference_queue.stop(); // Renamed from tpu_queue
    phone_queue.stop();
    camera_obj->release();
    camera_manager->stop();

    std::cout << "[" << get_timestamp() << "] [Main] Joining threads..." << std::endl;
    t_inference.join(); // Changed from t2.join()
    // t3.join(); // Disabled for debugging OOM

    // Final cleanup
    close(lock_fd);
    cleanup_resources();

    std::cout << "[" << get_timestamp() << "] [Main] All threads have stopped. Application terminated." << std::endl;
    return 0;
}
