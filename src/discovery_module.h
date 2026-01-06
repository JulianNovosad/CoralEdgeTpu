// Verified headers: [string, thread, atomic, functional, vector...]
// Verification timestamp: 2026-01-06 17:08:04
#ifndef DISCOVERY_MODULE_H
#define DISCOVERY_MODULE_H

#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <vector>
#include <mutex>
#include <zmq.hpp>

/**
 * @brief Handles auto-discovery of peers (Android app) using ZeroMQ beacons.
 * 
 * Works by broadcasting a beacon on a specific UDP port and listening for beacons from other peers.
 * When a peer is discovered, it notifies the application to establish streaming sessions.
 */
class DiscoveryModule {
public:
    struct PeerInfo {
        std::string ip;
        int orientation_port;
        int video_port;
        std::string name;
        std::chrono::steady_clock::time_point last_seen;
    };

    using DiscoveryCallback = std::function<void(const PeerInfo&)>;

    DiscoveryModule(int beacon_port = 5678);
    ~DiscoveryModule();

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    void on_peer_discovered(DiscoveryCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        discovery_callbacks_.push_back(callback);
    }

private:
    void beacon_thread_func();
    void listener_thread_func();
    void broadcast_beacon();
    
    int beacon_port_;
    std::atomic<bool> running_;
    
    std::thread beacon_thread_;
    std::thread listener_thread_;
    
    std::mutex peers_mutex_;
    std::vector<PeerInfo> discovered_peers_;
    
    std::mutex callback_mutex_;
    std::vector<DiscoveryCallback> discovery_callbacks_;
    
    std::unique_ptr<zmq::context_t> zmq_context_;
};

#endif // DISCOVERY_MODULE_H
