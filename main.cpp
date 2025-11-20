// A C++ application for real-time object detection using a Raspberry Pi Camera
// and Google Coral Edge TPU. This application captures frames from libcamera,
// resizes them to the model's input dimensions, performs inference on the Edge TPU,
// and logs performance metrics.
//
// This project is developed according to a Stage-Gate plan, and this code
// represents the successful completion of Stage 0 for the TPU component.
//
// --- Usage Instructions (README) ---
//
// 1.  **System Requirements:**
//     *   Raspberry Pi 5 with Debian Bookworm (ARM64 architecture).
//     *   Raspberry Pi Camera Module connected and enabled.
//     *   Google Coral Edge TPU (PCIe) connected.
//     *   Kernel 6.6.51+rpt-rpi-v8
//         Assurance of MSI-X (Enable+ Count=128) for the TPU.
//         (Refer to project documentation for DTB modification details if not already applied).
//
// 2.  **Dependencies Installation:**
//     *   **Build Tools & Libraries:**
//         ```bash
//         sudo apt-get update
//         sudo apt-get install -y build-essential cmake git curl lshw python3-pip libusb-1.0-0-dev libcamera-dev
//         ```
//     *   **Google Coral Edge TPU Runtime:**
//         ```bash
//         echo "deb [arch=arm64 signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
//         curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg >/dev/null
//         sudo apt-get update
//         sudo apt-get install -y libedgetpu1-std
//         ```
//
// 3.  **External Source Code (for building TensorFlow Lite and Headers):**
//     *   **TensorFlow Lite (v2.5.0):**
//         ```bash
//         git clone --depth 1 --branch v2.5.0 https://github.com/tensorflow/tensorflow.git ~/tensorflow_src
//         # Note: Replace ~/tensorflow_src with the actual path if cloned elsewhere.
//         ```
//     *   **FlatBuffers (v1.12.0):**
//         ```bash
//         git clone --depth 1 --branch v1.12.0 https://github.com/google/flatbuffers.git ~/flatbuffers_src
//         # Note: Replace ~/flatbuffers_src with the actual path if cloned elsewhere.
//         ```
//     *   **libedgetpu Source (for Headers):**
//         ```bash
//         git clone https://github.com/google-coral/libedgetpu.git ~/libedgetpu_src
//         # Note: Replace ~/libedgetpu_src with the actual path if cloned elsewhere.
//         ```
//
// 4.  **Build and Install FlatBuffers:**
//     ```bash
//     rm -rf /usr/local/include/flatbuffers /usr/local/lib/libflatbuffers.a # Clean old installs
//     mkdir -p ~/flatbuffers_src/build && cd ~/flatbuffers_src/build
//     cmake -DFLATBUFFERS_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release ..
//     sudo make install
//     sudo ldconfig
//     cd - # Return to previous directory
//     ```
//
// 5.  **Build TensorFlow Lite (v2.5.0) Shared Library & Apply Patches:**
//     ```bash
//     cd ~/tensorflow_src
//     # Apply patches for missing includes (crucial for older TensorFlow versions with modern GCC)
//     sed -i '1i#include <limits>' tensorflow/lite/ruy/ruy/block_map.cc
//     sed -i '2i#include <cstddef>' tensorflow/lite/ruy/ruy/block_map.cc
//     sed -i '1i#include <cstddef>' tensorflow/lite/gemmlowp/fixedpoint/fixedpoint.h
//     sed -i '1i#include <cstddef>' tensorflow/lite/gemmlowp/public/gemmlowp.h
//     
//     # Modify Makefile to build shared library
//     sed -i 's/LIB_NAME = libtensorflow-lite.a/LIB_NAME = libtensorflow-lite.so/' tensorflow/lite/tools/make/Makefile
//     sed -i 's/ar rc \$(LIB_NAME)/\$(CXX) -shared -o \$(LIB_NAME)/' tensorflow/lite/tools/make/Makefile
//     sed -i 's/\$(OBJS)/\$(OBJS) \$(LDFLAGS)/' tensorflow/lite/tools/make/Makefile
//     
//     make -f tensorflow/lite/tools/make/Makefile -j$(nproc)
//     cd - # Return to previous directory
//     ```
//
// 6.  **Prepare Project Directories & Copy Artifacts:**
//     ```bash
//     mkdir -p ~/CoralEdgeTpu/include/tensorflow ~/CoralEdgeTpu/include/edgetpu ~/CoralEdgeTpu/include/tflite/public ~/CoralEdgeTpu/lib ~/CoralEdgeTpu/DAILYLOGS
//     
//     # Copy TensorFlow Lite build artifact
//     cp ~/tensorflow_src/tensorflow/lite/libtensorflow-lite.so ~/CoralEdgeTpu/lib/
//     
//     # Copy TensorFlow Lite headers (core TFLite API)
//     rm -rf ~/CoralEdgeTpu/include/tensorflow/lite # Clean previous copies
//     cp -r ~/tensorflow_src/tensorflow/lite ~/CoralEdgeTpu/include/tensorflow/
//     
//     # Copy Edge TPU delegate specific headers (from libedgetpu source)
//     cp ~/libedgetpu_src/tflite/edgetpu_delegate_for_custom_op.h ~/CoralEdgeTpu/include/edgetpu/
//     cp ~/libedgetpu_src/tflite/public/edgetpu.h ~/CoralEdgeTpu/include/tflite/public/
//     
//     # Copy model and labels
//     cp ~/CoralEdgeTpu/modernizeddockertpurunfile/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite ~/CoralEdgeTpu/
//     cp ~/CoralEdgeTpu/modernizeddockertpurunfile/coco_labels.txt ~/CoralEdgeTpu/
//     ```
//
// 7.  **Compile the `detector` Application:**
//     ```bash
//     g++ -O3 -march=native -mtune=native -flto -std=c++17 /home/pi/CoralEdgeTpu/main.cpp /home/pi/CoralEdgeTpu/src/pca9685.cpp $(pkg-config --cflags --libs opencv4) \
//         -I/home/pi/CoralEdgeTpu/include \
//         -I/usr/include/libcamera \
//         /home/pi/CoralEdgeTpu/lib/libtensorflow-lite.so \
//         -L/usr/lib/aarch64-linux-gnu \
//         -ledgetpu \
//         -lcamera \
//         -lcamera-base \
//         -o /home/pi/CoralEdgeTpu/detector \
//         -lpthread -lm -lz -ldl -lusb-1.0 -ljpeg
//     ```
//
// 8.  **Run the `detector` Application:**
//     ```bash
//     /home/pi/CoralEdgeTpu/detector /home/pi/CoralEdgeTpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite /home/pi/CoralEdgeTpu/coco_labels.txt
//     ```
//
// 9.  **Output Files:**
//     *   Console output provides real-time inference logs and a final benchmark summary.
//     *   `/home/pi/CoralEdgeTpu/DAILYLOGS/tpu_benchmark.csv`: Contains the aggregate benchmark report (FPS, P50/P95/P99 latency).
//     *   `/home/pi/CoralEdgeTpu/DAILYLOGS/tpu_frame_latency.csv`: Contains individual frame-by-frame latency data for detailed analysis and graphing.
//
// --- End Usage Instructions ---



#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/op_resolver.h>
#include "edgetpu/edgetpu_delegate_for_custom_op.h"
#include <tensorflow/lite/c/common.h> // For TfLiteDelegate definition
#include <tensorflow/lite/builtin_op_data.h>
#include <fstream> // Required for ReadLabels

#include <fstream> // Required for ReadLabels

#include "pca9685.h"
#include <jpeglib.h>

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

#include <setjmp.h>
#include <memory>
#include <opencv2/opencv.hpp>

// Custom JPEG error management struct
struct JpegErrorManager {
    jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;

    static void error_exit(j_common_ptr cinfo) {
        JpegErrorManager* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
        (*cinfo->err->output_message)(cinfo);
        longjmp(err->setjmp_buffer, 1);
    }

    static void output_message(j_common_ptr cinfo) {
        char buffer[JMSG_LENGTH_MAX];
        (*cinfo->err->format_message)(cinfo, buffer);
        std::cerr << "JPEG error: " << buffer << std::endl;
    }
};

// RAII wrapper for jpeg_compress_struct
class JpegCompressGuard {
public:
    explicit JpegCompressGuard(jpeg_compress_struct* p_cinfo) : cinfo_ptr(p_cinfo) {
        cinfo_ptr->err = jpeg_std_error(&error_mgr.pub);
        error_mgr.pub.error_exit = JpegErrorManager::error_exit;
        error_mgr.pub.output_message = JpegErrorManager::output_message;
        if (setjmp(error_mgr.setjmp_buffer)) {
            // If we get here, a JPEG error occurred
            return;
        }
        jpeg_create_compress(cinfo_ptr);
    }

    ~JpegCompressGuard() {
        if (cinfo_ptr) {
            jpeg_destroy_compress(cinfo_ptr);
        }
    }

private:
    jpeg_compress_struct* cinfo_ptr;
    JpegErrorManager error_mgr;
};

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

const int STREAM_WIDTH = 1536;
const int STREAM_HEIGHT = 864;
const int TPU_INPUT_WIDTH = 300;
const int TPU_INPUT_HEIGHT = 300;
const int RAW_FRAME_SIZE = STREAM_WIDTH * STREAM_HEIGHT * 3; // For BGR888

// Placeholder for JPEG frame size (actual size will vary)
const size_t JPEG_FRAME_MAX_SIZE = 300 * 1024; // 300 KB
const int MAX_QUEUE_SIZE = 120; // Max number of frames to buffer in memory (increased from 10 to 120 as per code review)
const int MJPEG_STREAM_PORT = 8080; // Port for MJPEG streaming server

// --- Globals ---
std::atomic<bool> running(true);

std::unique_ptr<PCA9685> pca9685_controller; // Global PCA9685 controller instance
std::vector<std::string> labels; // Global labels

// --- Thread 2: TPU Inference Thread ---

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

// To be passed from camera to TPU thread
struct FramePacket {
    std::shared_ptr<RawFrame> high_res_frame; // Use shared_ptr to avoid copies
    int high_res_width;
    int high_res_height;
};

// To hold TPU results
struct DetectionResult {
    int class_id;
    float score;
    // BBox coordinates are normalized (0.0 to 1.0)
    float ymin, xmin, ymax, xmax;
};

// To be passed from TPU to MJPEG thread
struct ProcessedPacket {
    std::shared_ptr<RawFrame> high_res_frame;
    int high_res_width;
    int high_res_height;
    std::vector<DetectionResult> detections;
};

ThreadSafeQueue<FramePacket> inference_queue;
ThreadSafeQueue<JpegFrame> phone_queue; // Kept for future use
ThreadSafeQueue<ProcessedPacket> processed_frame_queue;

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
    std::cout << "[INFO] Cleaning up resources..." << std::endl;
    
    // Remove Unix socket file if it exists (Removed after UDS refactor)
    // if (access(UNIX_SOCK_PATH, F_OK) == 0) {
    //     if (unlink(UNIX_SOCK_PATH) == 0) {
    //         std::cout << "[" << get_timestamp() << "] Removed Unix socket file." << std::endl;
    //     }
    // }
    
    // Remove lock file if it exists
    if (access(LOCK_FILE, F_OK) == 0) {
        if (unlink(LOCK_FILE) == 0) {
            std::cout << "[INFO] Removed lock file." << std::endl;
        }
    }
}

// --- Signal Handler ---
void signal_handler(int signum) {
    std::cout << "\n[INFO] Caught signal " << signum << ", shutting down..." << std::endl;
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

// --- Helper function to send all data ---
bool send_all(int sock, const char* data, size_t size) {
    while (size > 0) {
        ssize_t sent = send(sock, data, size, MSG_NOSIGNAL);
        if (sent <= 0) {
            if (errno == EINTR) continue; // Interrupted by signal, retry
            perror("send");
            return false; // Connection closed or error
        }
        data += sent;
        size -= sent;
    }
    return true;
}

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

// Utility function to encode RGB RawFrame to JPEG
// Custom deleter for unsigned char* allocated by jpeg_mem_dest
struct JpegMemDestFree {
    void operator()(unsigned char* p) const {
        if (p) {
            free(p);
        }
    }
};

std::vector<uint8_t> encode_rgb_to_jpeg(const RawFrame& rgb_frame, int width, int height, int quality) {
    std::vector<uint8_t> jpeg_buffer;
    struct jpeg_compress_struct cinfo_raw;
    JpegCompressGuard cinfo_guard(&cinfo_raw); // RAII for cinfo
    jpeg_compress_struct& cinfo = cinfo_raw; // Use reference for convenience

    // Set up destination manager to write to memory
    unsigned char* outbuffer_raw = nullptr;
    unsigned long outsize = 0;
    jpeg_mem_dest(&cinfo, &outbuffer_raw, &outsize);
    std::unique_ptr<unsigned char, JpegMemDestFree> outbuffer(outbuffer_raw); // RAII for outbuffer

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3; // RGB
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = (JSAMPROW)&rgb_frame[cinfo.next_scanline * width * 3];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    
    // Copy data from outbuffer to jpeg_buffer
    if (outsize > 0 && outbuffer.get() != nullptr) {
        jpeg_buffer.assign(outbuffer.get(), outbuffer.get() + outsize);
    }
    
    jpeg_destroy_compress(&cinfo); // This cleans up cinfo, not outbuffer
    // outbuffer is now managed by unique_ptr and will be freed automatically

    return jpeg_buffer;
}

// Simple function to draw a rectangle on an RGB frame
void draw_rectangle(RawFrame& frame, int width, int height, int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b, int thickness) {
    // Clip rectangle to image bounds
    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(width - 1, x + w - 1);
    int y2 = std::min(height - 1, y + h - 1);

    // Ensure thickness is positive and valid
    if (thickness <= 0 || x1 > x2 || y1 > y2) return;

    // Draw top and bottom horizontal bands
    for (int i = 0; i < thickness; ++i) {
        int current_y_top = y1 + i;
        int current_y_bottom = y2 - i;

        if (current_y_top < height && current_y_top >= 0) { // Top line
            for (int cur_x = x1; cur_x <= x2; ++cur_x) {
                size_t index = (current_y_top * width + cur_x) * 3;
                frame[index] = r;
                frame[index + 1] = g;
                frame[index + 2] = b;
            }
        }
        if (current_y_bottom >= 0 && current_y_bottom < height && current_y_bottom > current_y_top) { // Bottom line, avoid overwriting if rectangle is too thin
            for (int cur_x = x1; cur_x <= x2; ++cur_x) {
                size_t index = (current_y_bottom * width + cur_x) * 3;
                frame[index] = r;
                frame[index + 1] = g;
                frame[index + 2] = b;
            }
        }
    }

    // Draw left and right vertical bands
    // Adjusted y-coordinates to avoid overwriting corners already drawn by horizontal bands
    int inner_y1 = y1 + thickness;
    int inner_y2 = y2 - thickness;
    if (inner_y1 > inner_y2) return; // If inner region is invalid, no need to draw vertical lines

    for (int i = 0; i < thickness; ++i) {
        int current_x_left = x1 + i;
        int current_x_right = x2 - i;

        if (current_x_left < width && current_x_left >= 0) { // Left line
            for (int cur_y = inner_y1; cur_y <= inner_y2; ++cur_y) {
                size_t index = (cur_y * width + current_x_left) * 3;
                frame[index] = r;
                frame[index + 1] = g;
                frame[index + 2] = b;
            }
        }
        if (current_x_right >= 0 && current_x_right < width && current_x_right > current_x_left) { // Right line, avoid overwriting
            for (int cur_y = inner_y1; cur_y <= inner_y2; ++cur_y) {
                size_t index = (cur_y * width + current_x_right) * 3;
                frame[index] = r;
                frame[index + 1] = g;
                frame[index + 2] = b;
            }
        }
    }
}
// Simple function to draw text (very basic, for debugging)
// This is extremely basic and will just draw a few pixels for character representation.
// For robust text rendering, a font library like FreeType would be needed, which is out of scope for now.
// For now, let's represent text by drawing a filled rectangle.
void draw_text(RawFrame& frame, int width, int height, int x, int y, const std::string& text, uint8_t r, uint8_t g, uint8_t b) {
    int char_width = 6;  // Approximate pixel width for a character
    int char_height = 8; // Approximate pixel height for a character
    int text_width = text.length() * char_width;

    // Draw a filled rectangle as background for text
    for (int cur_y = y; cur_y < y + char_height && cur_y < height; ++cur_y) {
        for (int cur_x = x; cur_x < x + text_width && cur_x < width; ++cur_x) {
            if (cur_x >= 0 && cur_y >= 0) {
                size_t index = (cur_y * width + cur_x) * 3;
                frame[index] = r;
                frame[index + 1] = g;
                frame[index + 2] = b;
            }
        }
    }
}



extern "C" {
TfLiteDelegate* tflite_plugin_create_delegate(const void* options);
void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate);
}




// --- Thread 2: TPU Inference Thread ---
void tpu_inference_thread(const std::string& model_path, const std::string& labels_path) {
    std::cout << "[STATUS] [TPU Thread] Started. Initializing TensorFlow Lite interpreter..." << std::endl;

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "[ERROR] [TPU Thread] Failed to load model: " << model_path << std::endl;
        running = false;
        return;
    }
    std::cout << "[STATUS] [TPU Thread] TPU model loaded successfully: " << model_path << std::endl;

    // Build the interpreter with the Edge TPU delegate
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "[ERROR] [TPU Thread] Failed to construct interpreter." << std::endl;
        running = false;
        return;
    }
    std::cout << "[STATUS] [TPU Thread] TensorFlow Lite interpreter constructed." << std::endl;

    // Create the Edge TPU delegate using the C API
    TfLiteDelegate* delegate = tflite_plugin_create_delegate(nullptr); // No options for now
    if (!delegate) {
        std::cerr << "[ERROR] [TPU Thread] Failed to create Edge TPU delegate using tflite_plugin_create_delegate. Ensure Edge TPU is connected and drivers are installed." << std::endl;
        running = false;
        return;
    }
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        std::cerr << "[ERROR] [TPU Thread] Failed to apply Edge TPU delegate." << std::endl;
        // The delegate is owned by the interpreter after ModifyGraphWithDelegate if successful.
        // If it fails, we should free it.
        tflite_plugin_destroy_delegate(delegate);
        running = false;
        return;
    }
    std::cout << "[STATUS] [TPU Thread] Edge TPU delegate created and applied." << std::endl;
    // The delegate is owned by the interpreter after ModifyGraphWithDelegate if successful.
    // So, we don't need to explicitly free it here if it succeeds.

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "[ERROR] [TPU Thread] Failed to allocate tensors." << std::endl;
        running = false;
        return;
    }
    std::cout << "[STATUS] [TPU Thread] Tensors allocated." << std::endl;

    // Add logging here
    TfLiteIntArray* input_dims = interpreter->input_tensor(0)->dims;
    int input_height = input_dims->data[1];
    int input_width = input_dims->data[2];
    int input_channels = input_dims->data[3];
    std::cout << "[INFO] [TPU Thread] Model expected input tensor: "
              << input_width << "x" << input_height << "x" << input_channels
              << " (" << interpreter->input_tensor(0)->bytes << " bytes)" << std::endl;

    // Read labels
    auto ReadLabels = [](const std::string& filename) -> std::vector<std::string> {
        std::vector<std::string> labels;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Could not open labels file: " << filename << std::endl;
            return labels;
        }
        std::string line;
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        return labels;
    };
    labels = ReadLabels(labels_path); // Populate global labels vector
    if (labels.empty()) {
        std::cerr << "[ERROR] [TPU Thread] Failed to load labels or labels file is empty." << std::endl;
        running = false;
        return;
    }
    std::cout << "[STATUS] [TPU Thread] Labels loaded: " << labels.size() << " entries." << std::endl;
    std::cout << "[STATUS] [TPU Thread] Entering TPU inference loop." << std::endl;

    FramePacket packet;
    while (running && inference_queue.pop(packet)) {
        // Resize the high-res frame for TPU input
        RawFrame resized_frame = resize_image_rgb(*packet.high_res_frame, packet.high_res_width, packet.high_res_height, input_width, input_height);

        if (resized_frame.size() != interpreter->input_tensor(0)->bytes) {
            std::cerr << "[" << get_timestamp() << "] [TPU Thread] Error: Resized frame size mismatch with tensor size." << std::endl;
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


        // --- NEW OBJECT DETECTION LOGIC ---
        const float* detection_boxes = interpreter->typed_output_tensor<float>(0);
        const float* detection_classes = interpreter->typed_output_tensor<float>(1);
        const float* detection_scores = interpreter->typed_output_tensor<float>(2);
        const float* num_detections_ptr = interpreter->typed_output_tensor<float>(3);
        const int num_detections = static_cast<int>(*num_detections_ptr);
        
        const float score_threshold = 0.5f;

        ProcessedPacket processed_packet;
        processed_packet.high_res_frame = packet.high_res_frame;
        processed_packet.high_res_width = packet.high_res_width;
        processed_packet.high_res_height = packet.high_res_height;

        for (int i = 0; i < num_detections; ++i) {
            if (detection_scores[i] > score_threshold) {
                DetectionResult result;
                result.class_id = static_cast<int>(detection_classes[i]);
                result.score = detection_scores[i];
                result.ymin = detection_boxes[i * 4 + 0];
                result.xmin = detection_boxes[i * 4 + 1];
                result.ymax = detection_boxes[i * 4 + 2];
                result.xmax = detection_boxes[i * 4 + 3];
                processed_packet.detections.push_back(result);
            }
        }
        
        processed_frame_queue.push(std::move(processed_packet));




    }


    std::cout << "[" << get_timestamp() << "] [TPU Thread] Stopped." << std::endl;
}

// --- Thread 3: Phone Frame Sender ---
void phone_sender_thread() {
    std::cout << "[STATUS] [Phone Thread] Started. Waiting for phone connection..." << std::endl;
    
    while(running) { // Outer loop for reconnection
        int sock = -1; // Declared outside try block for wider scope
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

            std::cout << "[INFO] [Phone Thread] Attempting to connect to phone..." << std::endl;
            
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
                    std::cout << "[STATUS] [Phone Thread] Connected to phone." << std::endl;
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
                // Removed phone_frame_counter related code.
            }
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] [Phone Thread] Error: " << e.what() << std::endl;
        }

        if (sock != -1) close(sock);

        if (running) {
            std::cout << "[INFO] [Phone Thread] Connection lost or client disconnected. Reconnecting in 2 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    std::cout << "[STATUS] [Phone Thread] Stopped." << std::endl;
}

// --- MJPEG Streaming Thread ---
void mjpeg_stream_thread(int port) {
    std::cout << "[STATUS] [MJPEG Thread] Started. Listening on port " << port << std::endl;

    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("MJPEG server socket failed");
        return;
    }

    // Forcefully attaching socket to the port
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("MJPEG server setsockopt failed");
        return;
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    // Forcefully attaching socket to the port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("MJPEG server bind failed");
        return;
    }
    if (listen(server_fd, 3) < 0) {
        perror("MJPEG server listen failed");
        return;
    }

    std::cout << "[STATUS] [MJPEG Thread] Server listening on port " << port << std::endl;

    const std::chrono::milliseconds target_frame_time(1000 / 45); // Target 45 FPS

    while (running) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            if (errno == EINTR) continue; // Interrupted by signal, retry
            perror("MJPEG server accept failed");
            continue;
        }

        std::cout << "[STATUS] [MJPEG Thread] Client connected to " << inet_ntoa(address.sin_addr) << ":" << ntohs(address.sin_port) << std::endl;

        // Send MJPEG header
        std::string header = "HTTP/1.0 200 OK\r\n"
                             "Cache-Control: no-cache, private\r\n"
                             "Pragma: no-cache\r\n"
                             "Content-Type: multipart/x-mixed-replace; boundary=BOUNDARY\r\n\r\n";
        if (!send_all(new_socket, header.c_str(), header.length())) {
            close(new_socket);
            continue;
        }

        ProcessedPacket packet;
        while (running && processed_frame_queue.pop(packet)) {
            auto frame_start_time = std::chrono::steady_clock::now(); // Start timer for current frame
            // Directly create a cv::Mat from the RawFrame data (assume RGB for now, will fix if colors are still off)
            cv::Mat frame_mat(packet.high_res_height, packet.high_res_width, CV_8UC3, packet.high_res_frame->data());

            // Draw detections on the high-resolution frame
            for (const auto& det : packet.detections) {
                // De-normalize coordinates
                int box_y = static_cast<int>(det.ymin * packet.high_res_height);
                int box_x = static_cast<int>(det.xmin * packet.high_res_width);
                int box_h = static_cast<int>((det.ymax - det.ymin) * packet.high_res_height);
                int box_w = static_cast<int>((det.xmax - det.xmin) * packet.high_res_width);

                std::string label = "N/A";
                if (det.class_id >= 0 && det.class_id < labels.size()) {
                    label = labels[det.class_id];
                }
                
                draw_rectangle(*packet.high_res_frame, packet.high_res_width, packet.high_res_height, box_x, box_y, box_w, box_h, 255, 0, 0, 2);
                draw_text(*packet.high_res_frame, packet.high_res_width, packet.high_res_height, box_x, box_y - 10, label, 255, 255, 255);
            }

            // Encode to JPEG
            std::vector<uint8_t> jpeg_data = encode_rgb_to_jpeg(*packet.high_res_frame, packet.high_res_width, packet.high_res_height, 75); // JPEG quality to 75 as per code review
            std::string part_header = "--BOUNDARY\r\n";
            part_header += "Content-Type: image/jpeg\r\n";
            part_header += "Content-Length: " + std::to_string(jpeg_data.size()) + "\r\n\r\n";
            
            if (!send_all(new_socket, part_header.c_str(), part_header.length())) {
                break;
            }
            if (!send_all(new_socket, (const char*)jpeg_data.data(), jpeg_data.size())) {
                break;
            }

            auto frame_end_time = std::chrono::steady_clock::now(); // End timer for current frame
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end_time - frame_start_time);

            if (elapsed_time < target_frame_time) {
                std::this_thread::sleep_for(target_frame_time - elapsed_time);
            }
        }
        std::cout << "[STATUS] [MJPEG Thread] Client disconnected." << std::endl;
        close(new_socket);
    }
    close(server_fd);
    std::cout << "[STATUS] [MJPEG Thread] Stopped." << std::endl;
}

// Callback function for completed requests
void requestComplete(Request *request) {
    if (request->status() == Request::RequestComplete) {
        if (request->buffers().count(raw_stream) == 0) {
            std::cerr << "[ERROR] Error: Completed request has no buffer for the raw stream." << std::endl;
        } else {
            FrameBuffer *buffer = request->buffers().at(raw_stream);

            if (!buffer->planes().empty()) {
                const FrameBuffer::Plane &plane = buffer->planes()[0];
                void *data = mmap(NULL, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
                if (data == MAP_FAILED) {
                    std::cerr << "[ERROR] mmap failed" << std::endl;
                } else {
                    auto high_res_frame = std::make_shared<RawFrame>(plane.length);
                    memcpy(high_res_frame->data(), data, plane.length);
                    munmap(data, plane.length);
                    
                    FramePacket packet;
                    packet.high_res_frame = high_res_frame;
                    packet.high_res_width = STREAM_WIDTH;
                    packet.high_res_height = STREAM_HEIGHT;
                    
                    inference_queue.push(std::move(packet));
                    
                    std::cout << "[INFO] [Camera Callback] Captured frame size: " << plane.length << " bytes." << std::endl;
                }
            }
        }
    } else {
        std::cerr << "[ERROR] Request failed with status: " << request->status() << std::endl;
    }

    // Re-queue the request for continuous capture
    if (running.load(std::memory_order_acquire)) {
        // We must get the buffer *before* calling reuse(), as reuse() clears the buffer map.
        if (request->buffers().count(raw_stream) > 0) {
            FrameBuffer *buffer = request->buffers().at(raw_stream);
            request->reuse();
            if (request->addBuffer(raw_stream, buffer) < 0) {
                std::cerr << "[ERROR] Error: Failed to re-add buffer to request" << std::endl;
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
    std::cout << "[STATUS] Signal handlers registered." << std::endl;

    // 1. Initialize Camera Manager
    camera_manager = std::make_unique<CameraManager>();
    camera_manager->start();
    std::cout << "[STATUS] CameraManager initialized and started." << std::endl;

    if (camera_manager->cameras().empty()) {
        std::cerr << "[ERROR] No cameras found." << std::endl;
        cleanup_resources();
        return 1;
    }

    // 2. Select a Camera (e.g., the first available camera)
    camera_obj = camera_manager->cameras()[0];
    camera_obj->acquire(); // Acquire the camera for exclusive use
    std::cout << "[STATUS] Camera acquired: " << camera_obj->id() << std::endl;

    // 3. Configure Streams for Raw Capture
    std::unique_ptr<CameraConfiguration> config = camera_obj->generateConfiguration({StreamRole::Raw});
    if (!config) {
        std::cerr << "[ERROR] Failed to generate default configuration." << std::endl;
        camera_obj->release();
        camera_manager->stop();
        cleanup_resources();
        return 1;
    }

    // Modify the stream configuration for raw capture
    StreamConfiguration &rawStreamConfig = config->at(0);
    rawStreamConfig.pixelFormat = libcamera::formats::RGB888; // Match what libcamera configures
    rawStreamConfig.size.width = STREAM_WIDTH;
    rawStreamConfig.size.height = STREAM_HEIGHT;
    rawStreamConfig.bufferCount = 6; // Explicitly set buffer count

    // Validate and apply the configuration
    int ret = config->validate();
    if (ret == -EINVAL) {
        std::cerr << "[ERROR] Failed to validate camera configuration." << std::endl;
        camera_obj->release();
        camera_manager->stop();
        cleanup_resources();
        return 1;
    }
    std::cout << "[STATUS] Camera configuration validated." << std::endl;
    camera_obj->configure(config.get());
    std::cout << "[STATUS] Camera configured with " << STREAM_WIDTH << "x" << STREAM_HEIGHT << " RGB888." << std::endl;


    // Get the raw stream
    raw_stream = rawStreamConfig.stream();

    // 4. Allocate Frame Buffers
    allocator = std::make_unique<FrameBufferAllocator>(camera_obj);
    for (StreamConfiguration &cfg : *config) {
        int ret = allocator->allocate(cfg.stream());
        if (ret < 0) {
            std::cerr << "[ERROR] Can't allocate buffers for stream. Error: " << ret << std::endl;
            camera_obj->release();
            camera_manager->stop();
            cleanup_resources();
            return 1;
        }
        std::cout << "[STATUS] Allocated " << allocator->buffers(cfg.stream()).size()
                  << " buffers for stream." << std::endl;
    }

    // 5. Create and Queue Requests
    std::cout << "[STATUS] Creating and queuing camera requests..." << std::endl;
    for (StreamConfiguration &cfg : *config) {
        Stream *stream = cfg.stream();
        const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(stream);

        for (const auto &buffer : buffers) {
            std::unique_ptr<Request> request = camera_obj->createRequest();
            if (!request) {
                std::cerr << "[ERROR] Failed to create request." << std::endl;
                camera_obj->release();
                camera_manager->stop();
                cleanup_resources();
                return 1;
            }

            int add_buffer_ret = request->addBuffer(stream, buffer.get());
            if (add_buffer_ret < 0) {
                std::cerr << "[ERROR] Failed to add buffer to request: " << add_buffer_ret << std::endl;
                camera_obj->release();
                camera_manager->stop();
                cleanup_resources();
                return 1; // Or handle error appropriately
            }
            libcamera_requests.push_back(std::move(request));
        }
    }
    std::cout << "[STATUS] All camera requests created." << std::endl;

    // Set the request completion callback
    camera_obj->requestCompleted.connect(requestComplete);
    std::cout << "[STATUS] Camera request completion callback connected." << std::endl;

    // 6. Start Camera
    camera_obj->start();
    std::cout << "[STATUS] Camera capturing started." << std::endl;

    // Queue all initial requests
    for (auto &request : libcamera_requests) {
        camera_obj->queueRequest(request.get());
    }
    std::cout << "[STATUS] Initial camera requests queued." << std::endl;


    // Initialize PCA9685
    pca9685_controller = std::make_unique<PCA9685>(1, PCA9685_I2C_ADDRESS); // Bus 1, default address 0x40
    if (!pca9685_controller->openDevice()) {
        std::cerr << "[ERROR] Failed to open PCA9685 device." << std::endl;
        cleanup_resources();
        return 1;
    }
    std::cout << "[STATUS] PCA9685 controller initialized." << std::endl;

    // Test servo movement
    std::cout << "[STATUS] Testing servo on channel 0..." << std::endl;
    for (int i = 0; i < 3; ++i) { // Move 3 times
        pca9685_controller->setServoAngle(0, 0); // Angle 0 degrees
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        pca9685_controller->setServoAngle(0, 90); // Angle 90 degrees
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        pca9685_controller->setServoAngle(0, 180); // Angle 180 degrees
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    std::cout << "[STATUS] Servo test completed." << std::endl;

    std::cout << "[STATUS] Starting threads..." << std::endl;

    const std::string model_path = "/home/pi/CoralEdgeTpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
    const std::string labels_path = "/home/pi/CoralEdgeTpu/coco_labels.txt";

    std::thread t_inference(tpu_inference_thread, model_path, labels_path);
    std::thread t_mjpeg_stream(mjpeg_stream_thread, MJPEG_STREAM_PORT);
    std::cout << "[STATUS] TPU inference and MJPEG streaming threads started." << std::endl;

    // The main thread will now wait for a signal to stop
    // The camera_capture_thread is no longer needed as a separate thread for frame generation.
    // The requestComplete callback handles frame pushing.
    while(running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 7. Stop Camera and Cleanup
    std::cout << "[STATUS] Shutting down: Stopping camera and joining threads..." << std::endl;
    camera_obj->stop();
    inference_queue.stop(); // Renamed from tpu_queue
    phone_queue.stop();
    camera_obj->release();
    camera_manager->stop();

    std::cout << "[STATUS] Joining threads..." << std::endl;
    t_inference.join(); // Changed from t2.join()
    t_mjpeg_stream.join();
    // t3.join(); // Disabled for debugging OOM

    // Final cleanup
    cleanup_resources();

    std::cout << "[STATUS] Application successfully terminated." << std::endl;
    return 0;

}
