#ifndef PROCESS_SUPERVISOR_HPP
#define PROCESS_SUPERVISOR_HPP

#include "util_logging.h" // For LOG_INFO, LOG_WARNING, LOG_ERROR
#include "pipeline_structs.h" // For ImageQueue, MjpegQueue, ImageData, ImageFrame

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <functional> // For std::function
#include <stdexcept> // For std::runtime_error
#include <mutex>    // For std::mutex
#include <numeric>  // For std::iota

#include <unistd.h> // For fork, execv, pipe, close, read
#include <sys/wait.h> // For waitpid
#include <csignal>  // For kill, SIGTERM, SIGKILL
#include <fcntl.h>  // For fcntl, F_SETFL, O_NONBLOCK
#include <cstring>  // For strerror
#include <algorithm> // For std::remove_if
#include <memory>   // For std::unique_ptr

// Forward declaration of the PipeReader. It will be templated on the OutputQueue.
template<typename OutputQueueType, typename DataType>
class PipeReader {
public:
    PipeReader(const std::string& name, int pipe_fd, OutputQueueType& output_queue,
               std::function<bool(std::vector<uint8_t>&, size_t, OutputQueueType&)> frame_parser, // Modified to non-const ref
               std::atomic<bool>& running, std::atomic<std::chrono::steady_clock::time_point>& last_activity_time)
        : name_(name), pipe_fd_(pipe_fd), output_queue_(output_queue), frame_parser_(frame_parser),
          running_(running), last_activity_time_(last_activity_time) {}

    void start() {
        reader_thread_ = std::thread(&PipeReader::run, this);
    }

    void stop() {
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }
    }

private:
    void run() {
        LOG_INFO(name_ + " pipe reader started.");
        std::vector<uint8_t> read_buffer(4096); // Small buffer for non-blocking reads
        std::vector<uint8_t> frame_buffer; // Buffer to accumulate full frames

        // Set pipe to non-blocking
        if (fcntl(pipe_fd_, F_SETFL, O_NONBLOCK) == -1) {
            LOG_ERROR(name_ + " pipe reader: Failed to set pipe to non-blocking: " + strerror(errno));
            running_ = false; // Signal parent to stop
            return;
        }

        while (running_.load()) {
            ssize_t bytes_read = read(pipe_fd_, read_buffer.data(), read_buffer.size());

            if (bytes_read > 0) {
                frame_buffer.insert(frame_buffer.end(), read_buffer.begin(), read_buffer.begin() + bytes_read);
                // The frame_parser function is responsible for consuming data from frame_buffer
                // and returning true if a complete frame was parsed.
                // It should also update last_activity_time_ on successful frame parsing.
                if (frame_parser_(frame_buffer, bytes_read, output_queue_)) {
                     last_activity_time_.store(std::chrono::steady_clock::now());
                }
            } else if (bytes_read == 0) {
                // EOF - child process has closed its end of the pipe or exited
                LOG_WARNING(name_ + " pipe reader reached EOF. Child process likely exited.");
                running_ = false; // Signal parent to stop
                break;
            } else { // bytes_read == -1
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // No data available right now, try again
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } else if (errno == EINTR) {
                    // Interrupted system call, retry
                    continue;
                } else {
                    LOG_ERROR(name_ + " pipe reader error: " + strerror(errno));
                    running_ = false; // Signal parent to stop
                    break;
                }
            }
        }
        LOG_INFO(name_ + " pipe reader stopped.");
    }

    std::string name_;
    int pipe_fd_;
    OutputQueueType& output_queue_;
    std::function<bool(std::vector<uint8_t>&, size_t, OutputQueueType&)> frame_parser_; // Modified to non-const ref
    std::atomic<bool>& running_;
    std::atomic<std::chrono::steady_clock::time_point>& last_activity_time_;
    std::thread reader_thread_;
};


template <typename OutputQueueType, typename DataType>
class ProcessSupervisor {
public:
    // Callback for parsing frames. Takes the raw buffer, bytes read, output queue and returns true if a frame was successfully parsed.
    using FrameParserFn = std::function<bool(std::vector<uint8_t>&, size_t, OutputQueueType&)>;
    // Callback for constructing command arguments.
    using CmdArgsFn = std::function<std::vector<std::string>()>;

    ProcessSupervisor(const std::string& name, CmdArgsFn cmd_args_builder, FrameParserFn frame_parser,
                      OutputQueueType& output_queue, std::chrono::seconds watchdog_timeout)
        : name_(name), cmd_args_builder_(cmd_args_builder), frame_parser_(frame_parser),
          output_queue_(output_queue), watchdog_timeout_(watchdog_timeout) {
        reset_restart_policy();
        last_activity_time_.store(std::chrono::steady_clock::now());
    }

    ~ProcessSupervisor() {
        stop();
    }

    bool start() {
        LOG_INFO(name_ + " manager starting...");
        if (running_.exchange(true)) {
            LOG_ERROR(name_ + " manager is already running.");
            return false;
        }
        output_queue_.set_running(true);
        manager_thread_ = std::thread(&ProcessSupervisor::manager_thread_func, this);
        watchdog_thread_ = std::thread(&ProcessSupervisor::watchdog_thread_func, this);
        return true;
    }

    void stop() {
        if (!running_.exchange(false)) {
            return; // Already stopped
        }
        LOG_INFO(name_ + " manager stopping...");
        terminate_child_process(); // Ensure child is killed during shutdown
        output_queue_.set_running(false);

        if (manager_thread_.joinable()) {
            manager_thread_.join();
        }
        if (watchdog_thread_.joinable()) {
            watchdog_thread_.join();
        }
        LOG_INFO(name_ + " manager stopped.");
    }

    bool is_running() const {
        return running_.load();
    }

private:
    void manager_thread_func() {
        while (running_.load()) {
            apply_restart_policy(); // Apply backoff or cool-down if necessary

            if (!running_.load()) break; // Check if stopped during backoff

            if (launch_child_process()) {
                LOG_INFO(name_ + " child process (PID: " + std::to_string(child_pid_) + ") launched successfully.");
                int status;
                waitpid(child_pid_, &status, 0); // Wait for child to exit
                handle_child_exit(child_pid_, status);
            } else {
                LOG_ERROR(name_ + " failed to launch child process. Will retry based on policy.");
                std::this_thread::sleep_for(std::chrono::seconds(1)); // Small delay before next retry attempt
            }
            cleanup_child_resources();
        }
        LOG_INFO(name_ + " manager thread exited.");
    }
    
    void watchdog_thread_func() {
        LOG_INFO(name_ + " watchdog started.");
        while (running_.load()) {
            std::this_thread::sleep_for(watchdog_timeout_ / 2); // Check more frequently than timeout

            if (!running_.load()) break;

            auto now = std::chrono::steady_clock::now();
            auto last_activity = last_activity_time_.load();
            if (child_pid_ != -1 && std::chrono::duration_cast<std::chrono::seconds>(now - last_activity) > watchdog_timeout_) {
                LOG_WARNING(name_ + " watchdog: No data received for " + std::to_string(watchdog_timeout_.count()) +
                            " seconds. Restarting child process.");
                terminate_child_process(); // Signal manager thread to restart
            }
        }
        LOG_INFO(name_ + " watchdog stopped.");
    }

    bool launch_child_process() {
        pipe_fds_[0] = -1; pipe_fds_[1] = -1; // Reset FDs
        if (pipe(pipe_fds_) == -1) {
            LOG_ERROR(name_ + " failed to create pipe: " + strerror(errno));
            return false;
        }

        child_pid_ = fork();
        if (child_pid_ == -1) {
            LOG_ERROR(name_ + " failed to fork process: " + strerror(errno));
            close(pipe_fds_[0]);
            close(pipe_fds_[1]);
            return false;
        }

        if (child_pid_ == 0) { // Child process
            close(pipe_fds_[0]); // Close read end of pipe
            if (dup2(pipe_fds_[1], STDOUT_FILENO) == -1) {
                perror((name_ + " child: failed to redirect stdout").c_str());
                _exit(EXIT_FAILURE); // Use _exit in child after fork
            }
            close(pipe_fds_[1]); // Close write end of pipe

            auto args_str = cmd_args_builder_();
            std::vector<char*> argv;
            for (const auto& arg : args_str) {
                argv.push_back(const_cast<char*>(arg.c_str()));
            }
            argv.push_back(nullptr); // Null-terminate the argument list

            execv(argv[0], argv.data());
            
            // If execv returns, an error occurred
            perror((name_ + " child: execv failed").c_str());
            _exit(EXIT_FAILURE); // Use _exit in child
        } else { // Parent process
            close(pipe_fds_[1]); // Close write end of pipe in parent
            pipe_reader_ = std::make_unique<PipeReader<OutputQueueType, DataType>>(
                name_, pipe_fds_[0], output_queue_, frame_parser_, running_, last_activity_time_);
            pipe_reader_->start();
            return true;
        }
    }

    void terminate_child_process() {
        if (child_pid_ > 0) {
            LOG_INFO(name_ + " attempting to terminate child process (PID: " + std::to_string(child_pid_) + ")");
            if (kill(child_pid_, SIGTERM) == -1) {
                if (errno != ESRCH) { // ESRCH means process doesn't exist
                    LOG_ERROR(name_ + " failed to send SIGTERM to PID " + std::to_string(child_pid_) + ": " + strerror(errno));
                } else {
                    LOG_WARNING(name_ + " child process (PID: " + std::to_string(child_pid_) + ") already gone before SIGTERM.");
                }
            }
            
            // Give child time to shut down gracefully
            for (int i = 0; i < 5; ++i) {
                int status;
                if (waitpid(child_pid_, &status, WNOHANG) == child_pid_) {
                    LOG_INFO(name_ + " child process (PID: " + std::to_string(child_pid_) + ") terminated gracefully.");
                    child_pid_ = -1; // Mark as terminated
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (child_pid_ > 0) { // If still alive, force kill
                LOG_WARNING(name_ + " child process (PID: " + std::to_string(child_pid_) + ") did not terminate gracefully, sending SIGKILL.");
                if (kill(child_pid_, SIGKILL) == -1) {
                    if (errno != ESRCH) {
                        LOG_ERROR(name_ + " failed to send SIGKILL to PID " + std::to_string(child_pid_) + ": " + strerror(errno));
                    }
                }
                 // Ensure waitpid is called to reap the zombie
                int status;
                waitpid(child_pid_, &status, 0);
                child_pid_ = -1;
            }
        }
    }

    void cleanup_child_resources() {
        if (pipe_reader_) {
            pipe_reader_->stop(); // Ensure reader thread is joined
            pipe_reader_.reset();
        }
        if (pipe_fds_[0] != -1) {
            close(pipe_fds_[0]);
            pipe_fds_[0] = -1;
        }
    }

    void handle_child_exit(pid_t pid, int status) {
        std::string log_msg = name_ + " child process (PID: " + std::to_string(pid) + ") ";
        if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            if (exit_code == 0) {
                LOG_INFO(log_msg + "exited normally with code 0.");
                // Normal exit, reset restart policy
                reset_restart_policy();
            } else {
                LOG_ERROR(log_msg + "exited with code: " + std::to_string(exit_code) + ".");
                // Abnormal exit, increment restart count
                update_restart_policy(true);
            }
        } else if (WIFSIGNALED(status)) {
            int signal_num = WTERMSIG(status);
            LOG_ERROR(log_msg + "terminated by signal: " + std::to_string(signal_num) + " (" + strsignal(signal_num) + ").");
            update_restart_policy(true);
        } else {
            LOG_ERROR(log_msg + "exited with unknown status.");
            update_restart_policy(true);
        }
        child_pid_ = -1; // Reset child PID after handling exit
    }

    void reset_restart_policy() {
        std::lock_guard<std::mutex> lock(restart_mutex_);
        restart_count_ = 0;
        current_backoff_duration_ = INITIAL_BACKOFF_;
        cool_down_end_time_ = std::chrono::steady_clock::time_point::min(); // Not in cool-down
        last_activity_time_.store(std::chrono::steady_clock::now()); // Reset activity time on reset
    }

    void update_restart_policy(bool crash_occurred) {
        std::lock_guard<std::mutex> lock(restart_mutex_);
        if (crash_occurred) {
            restart_count_++;
            if (restart_count_ >= MAX_RAPID_FAILURES) {
                LOG_ERROR(name_ + " experienced " + std::to_string(MAX_RAPID_FAILURES) + " rapid failures. Entering cool-down period.");
                cool_down_end_time_ = std::chrono::steady_clock::now() + COOLDOWN_PERIOD_;
                current_backoff_duration_ = INITIAL_BACKOFF_; // Reset backoff for next attempt after cool-down
                restart_count_ = 0; // Reset rapid failure count
            } else {
                current_backoff_duration_ *= 2; // Exponential backoff
                if (current_backoff_duration_ > MAX_BACKOFF_) {
                    current_backoff_duration_ = MAX_BACKOFF_;
                }
            }
        } else {
            reset_restart_policy(); // If it ran without a crash, reset policy
        }
    }

    void apply_restart_policy() {
        std::lock_guard<std::mutex> lock(restart_mutex_);
        if (cool_down_end_time_ > std::chrono::steady_clock::now()) {
            auto remaining_cooldown = cool_down_end_time_ - std::chrono::steady_clock::now();
            LOG_WARNING(name_ + " in cool-down for another " +
                        std::to_string(std::chrono::duration_cast<std::chrono::seconds>(remaining_cooldown).count()) +
                        " seconds.");
            std::this_thread::sleep_for(remaining_cooldown);
        }

        if (restart_count_ > 0) { // If there were recent crashes, apply backoff
            LOG_INFO(name_ + " applying backoff: waiting for " + std::to_string(current_backoff_duration_.count()) + " seconds.");
            std::this_thread::sleep_for(current_backoff_duration_);
        }
        last_activity_time_.store(std::chrono::steady_clock::now()); // Reset activity time before new launch attempt
    }

    // Member variables
    std::string name_;
    CmdArgsFn cmd_args_builder_;
    FrameParserFn frame_parser_;
    OutputQueueType& output_queue_;
    std::chrono::seconds watchdog_timeout_;

    std::atomic<bool> running_ = false;
    std::thread manager_thread_;
    std::thread watchdog_thread_;

    pid_t child_pid_ = -1;
    int pipe_fds_[2] = {-1, -1};
    std::unique_ptr<PipeReader<OutputQueueType, DataType>> pipe_reader_;
    std::atomic<std::chrono::steady_clock::time_point> last_activity_time_;

    // Restart policy
    std::mutex restart_mutex_;
    int restart_count_ = 0;
    std::chrono::seconds current_backoff_duration_ = INITIAL_BACKOFF_;
    std::chrono::steady_clock::time_point cool_down_end_time_;

    static const std::chrono::seconds INITIAL_BACKOFF_;
    static const std::chrono::seconds MAX_BACKOFF_;
    static const int MAX_RAPID_FAILURES;
    static const std::chrono::seconds COOLDOWN_PERIOD_;
};

// Static member initialization
template <typename OutputQueueType, typename DataType>
const std::chrono::seconds ProcessSupervisor<OutputQueueType, DataType>::INITIAL_BACKOFF_ = std::chrono::seconds(1);
template <typename OutputQueueType, typename DataType>
const std::chrono::seconds ProcessSupervisor<OutputQueueType, DataType>::MAX_BACKOFF_ = std::chrono::seconds(60);
template <typename OutputQueueType, typename DataType>
const int ProcessSupervisor<OutputQueueType, DataType>::MAX_RAPID_FAILURES = 3;
template <typename OutputQueueType, typename DataType>
const std::chrono::seconds ProcessSupervisor<OutputQueueType, DataType>::COOLDOWN_PERIOD_ = std::chrono::seconds(30);

#endif // PROCESS_SUPERVISOR_HPP
