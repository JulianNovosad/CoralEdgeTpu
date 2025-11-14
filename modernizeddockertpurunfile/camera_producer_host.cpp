// camera_producer_host.cpp
// A multi-threaded camera producer that provides two streams:
// 1. 100 FPS raw frames to a Unix socket for a local TPU.
// 2. 45 FPS JPEG-encoded frames to a TCP socket for a remote client (phone).
//
// Compilation:
// g++ -o camera_producer_host camera_producer_host.cpp -O3 -std=c++17 -lcamera -pthread

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
const char* UNIX_SOCK_DIR = "/tmp/coral_ipc";
const char* UNIX_SOCK_PATH = "/tmp/coral_ipc/camera.sock";
const char* LOCK_FILE = "/tmp/coral_ipc/camera_producer.lock"; // Prevent multiple instances
const char* PHONE_IP = "192.168.37.27";
const int PHONE_PORT = 8080; // TCP port for phone frames
const int UDP_PHONE_PORT = 9090; // UDP port for phone detections

// Frame dimensions
const int RAW_FRAME_WIDTH = 1536;
const int RAW_FRAME_HEIGHT = 864;
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

ThreadSafeQueue<RawFrame> tpu_queue;
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
    
    // Remove Unix socket file if it exists
    if (access(UNIX_SOCK_PATH, F_OK) == 0) {
        if (unlink(UNIX_SOCK_PATH) == 0) {
            std::cout << "[" << get_timestamp() << "] Removed Unix socket file." << std::endl;
        }
    }
    
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

// --- Helper function to send all data ---
bool send_all(int sock, const void* data, size_t size) {
    const char* buffer = static_cast<const char*>(data);
    while (size > 0) {
        ssize_t sent = send(sock, buffer, size, MSG_NOSIGNAL);
        if (sent <= 0) {
            if (errno == EINTR) continue; // Interrupted by signal, retry
            perror("send");
            return false; // Connection closed or error
        }
        buffer += sent;
        size -= sent;
    }
    return true;
}


// --- Thread 2: TPU Frame Sender ---


void tpu_sender_thread() {
    std::cout << "[" << get_timestamp() << "] [TPU Thread] Started. Connecting to Unix socket..." << std::endl;
    
    while (running) { // Outer loop for reconnection
        int sock = -1;
        try {
            sock = socket(AF_UNIX, SOCK_STREAM, 0);
            if (sock < 0) throw std::runtime_error("Failed to create Unix socket");

            int buf_size = 32 * 1024 * 1024;  // 32MB send buffer
            if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size)) < 0) {
                perror("setsockopt SO_SNDBUF failed");
                exit(1);
            }
            int actual_sndbuf;
            socklen_t optlen = sizeof(actual_sndbuf);
            getsockopt(sock, SOL_SOCKET, SO_SNDBUF, &actual_sndbuf, &optlen);
            printf("[%s] SO_SNDBUF requested: %d bytes, actual: %d bytes\n", get_timestamp().c_str(), buf_size, actual_sndbuf);

            struct sockaddr_un addr;
            memset(&addr, 0, sizeof(addr));
            addr.sun_family = AF_UNIX;
            strncpy(addr.sun_path, UNIX_SOCK_PATH, sizeof(addr.sun_path) - 1);

            // Connect to the Unix socket with retry and shutdown check
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Initial delay
            int retry_count = 0;
            const int MAX_RETRIES = 10; // Retry for up to 10 seconds

            while (running && retry_count < MAX_RETRIES) {
                if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                    std::cout << "[" << get_timestamp() << "] [TPU Thread] Connected to Unix socket." << std::endl;
                    break;
                }
                if (errno == EINTR) continue;
                if (errno == ENOENT) {
                    perror("[TPU Thread] connect(unix_sock) - No such file or directory. Retrying...");
                } else {
                    perror("[TPU Thread] connect(unix_sock)");
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
                retry_count++;
            }

            if (!running) break; // Exit if shutdown was requested during connect
            if (retry_count >= MAX_RETRIES) {
                 std::cerr << "[" << get_timestamp() << "] [TPU Thread] Failed to connect to Unix socket after multiple retries. Will try again." << std::endl;
                 close(sock);
                 std::this_thread::sleep_for(std::chrono::seconds(2));
                 continue; // Continue to next iteration of the outer while loop
            }

            // Main sending loop
            RawFrame frame;
            while (running && tpu_queue.pop(frame)) {
                uint32_t current_frame_id = tpu_frame_counter.fetch_add(1); // Get and increment frame ID
                uint32_t frame_id_net = htonl(current_frame_id); // Network byte order
                uint32_t frame_size_net = htonl(frame.size()); // Network byte order

                // Send frame_id
                if (!send_all(sock, &frame_id_net, sizeof(frame_id_net))) {
                    perror("[TPU Thread] send_all (frame_id) failed");
                    break; // Break inner loop to trigger reconnection
                }
                // Send frame size
                if (!send_all(sock, &frame_size_net, sizeof(frame_size_net))) {
                    perror("[TPU Thread] send_all (frame_size) failed");
                    break; // Break inner loop to trigger reconnection
                }
                // Send frame data
                if (!send_all(sock, frame.data(), frame.size())) {
                    perror("[TPU Thread] send_all (data) failed");
                    break; // Break inner loop to trigger reconnection
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[" << get_timestamp() << "] [TPU Thread] Error: " << e.what() << std::endl;
        }

        if (sock != -1) close(sock);
        
        if (running) {
            std::cout << "[" << get_timestamp() << "] [TPU Thread] Connection lost or client disconnected. Reconnecting in 2 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
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
                
                if (!send_all(sock, &frame_id_net, sizeof(frame_id_net)) ||
                    !send_all(sock, &frame_size_net, sizeof(frame_size_net)) ||
                    !send_all(sock, frame.data(), frame.size())) {
                    perror("[Phone Thread] send_all failed");
                    break; // Break inner loop to trigger reconnection
                }
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
                    memcpy(raw_frame.data(), data, plane.length);
                    tpu_queue.push(std::move(raw_frame));
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

    // Create directory for Unix socket
    if (mkdir(UNIX_SOCK_DIR, 0755) == -1 && errno != EEXIST) {
        perror("mkdir /tmp/coral_ipc");
        return 1;
    }

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

    std::thread t2(tpu_sender_thread);
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
    tpu_queue.stop();
    phone_queue.stop();
    camera_obj->release();
    camera_manager->stop();

    std::cout << "[" << get_timestamp() << "] [Main] Joining threads..." << std::endl;
    t2.join();
    // t3.join(); // Disabled for debugging OOM

    // Final cleanup
    close(lock_fd);
    cleanup_resources();

    std::cout << "[" << get_timestamp() << "] [Main] All threads have stopped. Application terminated." << std::endl;
    return 0;
}
