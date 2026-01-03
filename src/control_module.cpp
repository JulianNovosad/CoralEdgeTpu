#include "control_module.h"
#include "util_logging.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <vector>
#include <poll.h>
#include <future>
#include <iostream>

ControlModule::ControlModule(int port)
    : port_(port), running_(false), server_fd_(-1) {
}

ControlModule::~ControlModule() {
    stop();
}

bool ControlModule::start() {
    if (running_.load()) return true;

    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        APP_LOG_ERROR("ControlModule: Failed to create socket");
        return false;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        APP_LOG_ERROR("ControlModule: Failed to bind to port " + std::to_string(port_));
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (listen(server_fd_, 3) < 0) {
        APP_LOG_ERROR("ControlModule: Failed to listen");
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_.store(true);
    worker_thread_ = std::thread(&ControlModule::worker_thread_func, this);
    
    APP_LOG_INFO("ControlModule started on port " + std::to_string(port_));
    return true;
}

void ControlModule::stop() {
    if (!running_.load()) return;

    running_.store(false);
    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }

    if (worker_thread_.joinable()) {
        auto shared_promise = std::make_shared<std::promise<void>>();
        std::future<void> future = shared_promise->get_future();
        std::thread joiner_thread([this, shared_promise]() {
            try {
                if (worker_thread_.joinable()) {
                    worker_thread_.join();
                }
                shared_promise->set_value();
            } catch (...) {}
        });
        if (future.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
            std::cerr << "[SHUTDOWN] ControlModule worker thread did not join within 3s, detaching." << std::endl;
            if (worker_thread_.joinable()) worker_thread_.detach();
            joiner_thread.detach();
        } else {
            if (joiner_thread.joinable()) joiner_thread.join();
        }
    }
    
    APP_LOG_INFO("ControlModule stopped.");
}

void ControlModule::worker_thread_func() {
    struct pollfd pfd;
    pfd.fd = server_fd_;
    pfd.events = POLLIN;

    while (running_.load()) {
        int res = poll(&pfd, 1, 500); // 500ms timeout
        if (res > 0 && (pfd.revents & POLLIN)) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &addr_len);
            
            if (client_fd >= 0) {
                handle_client(client_fd);
                close(client_fd);
            }
        }
    }
}

void ControlModule::handle_client(int client_fd) {
    char buffer[1024] = {0};
    ssize_t valread = read(client_fd, buffer, 1024);
    
    if (valread > 0) {
        std::string command(buffer, valread);
        // Remove whitespace
        command.erase(command.find_last_not_of(" \n\r\t") + 1);
        
        APP_LOG_INFO("ControlModule: Received command: " + command);
        
        std::stringstream ss(command);
        std::string action;
        ss >> action;
        
        if (action == "START") {
            std::string phone_ip;
            ss >> phone_ip;
            if (!phone_ip.empty() && start_cb_) {
                start_cb_(phone_ip);
            } else {
                APP_LOG_ERROR("ControlModule: START command missing IP or callback not set");
            }
        } else if (action == "STOP") {
            if (stop_cb_) {
                stop_cb_();
            }
        } else {
            APP_LOG_WARNING("ControlModule: Unknown command: " + action);
        }
    }
}
