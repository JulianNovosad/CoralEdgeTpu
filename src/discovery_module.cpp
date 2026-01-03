#include "discovery_module.h"
#include "util_logging.h"
#include "json.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <future>
#include <iostream>

DiscoveryModule::DiscoveryModule(int beacon_port)
    : beacon_port_(beacon_port), running_(false) {
    zmq_context_ = std::make_unique<zmq::context_t>(1);
}

DiscoveryModule::~DiscoveryModule() {
    stop();
}

bool DiscoveryModule::start() {
    if (running_.load()) return true;
    
    running_.store(true);
    beacon_thread_ = std::thread(&DiscoveryModule::beacon_thread_func, this);
    listener_thread_ = std::thread(&DiscoveryModule::listener_thread_func, this);
    
    APP_LOG_INFO("DiscoveryModule started on port " + std::to_string(beacon_port_));
    return true;
}

void DiscoveryModule::stop() {
    if (!running_.load()) return;
    
    running_.store(false);
    
    if (beacon_thread_.joinable()) {
        auto shared_promise = std::make_shared<std::promise<void>>();
        std::future<void> future = shared_promise->get_future();
        std::thread joiner_thread([this, shared_promise]() {
            try {
                if (beacon_thread_.joinable()) beacon_thread_.join();
                shared_promise->set_value();
            } catch (...) {}
        });
        if (future.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
            std::cerr << "[SHUTDOWN] DiscoveryModule beacon thread did not join within 3s, detaching." << std::endl;
            if (beacon_thread_.joinable()) beacon_thread_.detach();
            joiner_thread.detach();
        } else {
            if (joiner_thread.joinable()) joiner_thread.join();
        }
    }

    if (listener_thread_.joinable()) {
        auto shared_promise = std::make_shared<std::promise<void>>();
        std::future<void> future = shared_promise->get_future();
        std::thread joiner_thread([this, shared_promise]() {
            try {
                if (listener_thread_.joinable()) listener_thread_.join();
                shared_promise->set_value();
            } catch (...) {}
        });
        if (future.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
            std::cerr << "[SHUTDOWN] DiscoveryModule listener thread did not join within 3s, detaching." << std::endl;
            if (listener_thread_.joinable()) listener_thread_.detach();
            joiner_thread.detach();
        } else {
            if (joiner_thread.joinable()) joiner_thread.join();
        }
    }
    
    APP_LOG_INFO("DiscoveryModule stopped.");
}

void DiscoveryModule::beacon_thread_func() {
    while (running_.load()) {
        broadcast_beacon();
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

void DiscoveryModule::broadcast_beacon() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return;

    int broadcastEn = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcastEn, sizeof(broadcastEn));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(beacon_port_);
    addr.sin_addr.s_addr = inet_addr("255.255.255.255");

    nlohmann::json j;
    j["type"] = "CORAL_TPU_PI";
    j["name"] = "RaspberryPi-TPU";
    // We don't necessarily need to send our ports if we are the server for video (UDP sink)
    // but we can send them for info.
    j["video_port"] = 5000;
    
    std::string msg = j.dump();
    sendto(sock, msg.c_str(), msg.size(), 0, (struct sockaddr*)&addr, sizeof(addr));
    
    close(sock);
}

void DiscoveryModule::listener_thread_func() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(beacon_port_);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        APP_LOG_ERROR("DiscoveryModule: Failed to bind listener socket");
        close(sock);
        return;
    }

    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    char buffer[1024];
    while (running_.load()) {
        struct sockaddr_in peer_addr;
        socklen_t peer_addr_len = sizeof(peer_addr);
        int n = recvfrom(sock, buffer, sizeof(buffer) - 1, 0, (struct sockaddr*)&peer_addr, &peer_addr_len);
        
        if (n > 0) {
            buffer[n] = '\0';
            try {
                auto j = nlohmann::json::parse(buffer);
                if (j.contains("type") && j["type"] == "ANDROID_CORAL_CONTROLLER") {
                    PeerInfo peer;
                    peer.ip = inet_ntoa(peer_addr.sin_addr);
                    peer.orientation_port = j.at("orientation_port").get<int>();
                    peer.video_port = j.value("video_port", 5000);
                    peer.name = j.value("name", "Unknown Android");
                    peer.last_seen = std::chrono::steady_clock::now();
                    
                    APP_LOG_INFO("DiscoveryModule: Peer discovered: " + peer.name + " at " + peer.ip);
                    
                    std::lock_guard<std::mutex> lock(callback_mutex_);
                    for (auto& cb : discovery_callbacks_) {
                        cb(peer);
                    }
                }
            } catch (...) {
                // Ignore malformed beacons
            }
        }
    }
    
    close(sock);
}
