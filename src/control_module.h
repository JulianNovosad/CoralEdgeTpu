// Verified headers: [thread, atomic, string, functional, netinet/in.h]
// Verification timestamp: 2026-01-06 17:08:04
#ifndef CONTROL_MODULE_H
#define CONTROL_MODULE_H

#include <thread>
#include <atomic>
#include <string>
#include <functional>
#include <netinet/in.h>

/**
 * @brief Listens for control commands on TCP port 6000.
 */
class ControlModule {
public:
    using StartCallback = std::function<void(const std::string& phone_ip)>;
    using StopCallback = std::function<void()>;

    ControlModule(int port = 6000);
    ~ControlModule();

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    void on_start(StartCallback cb) { start_cb_ = cb; }
    void on_stop(StopCallback cb) { stop_cb_ = cb; }

private:
    void worker_thread_func();
    void handle_client(int client_fd);

    int port_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    
    StartCallback start_cb_;
    StopCallback stop_cb_;
    
    int server_fd_;
};

#endif // CONTROL_MODULE_H
