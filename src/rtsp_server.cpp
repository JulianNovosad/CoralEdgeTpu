#include "rtsp_server.h"
#include "util_logging.h"
#include <chrono>
#include <cstring>
#include <iostream>

RTSPServerWrapper::RTSPServerWrapper(int rtspPort, const std::string& streamName)
    : rtspPort_(rtspPort), streamName_(streamName), 
      scheduler_(nullptr), env_(nullptr), rtspServer_(nullptr), 
      sms_(nullptr), videoSink_(nullptr), rtpGroupsock_(nullptr),
      videoSource_(nullptr), bufferSource_(nullptr),
      running_(false) {
}

RTSPServerWrapper::~RTSPServerWrapper() {
    stop();
}

bool RTSPServerWrapper::start() {
    if (running_) {
        APP_LOG_WARNING("RTSP server is already running");
        return true;
    }
    
    // Create scheduler and environment
    scheduler_ = BasicTaskScheduler::createNew();
    env_ = BasicUsageEnvironment::createNew(*scheduler_);
    
    // Create RTSP server
    rtspServer_ = RTSPServer::createNew(*env_, rtspPort_);
    if (rtspServer_ == nullptr) {
        APP_LOG_ERROR("Failed to create RTSP server: " + std::string(env_->getResultMsg()));
        return false;
    }
    
    // Create server media session
    sms_ = ServerMediaSession::createNew(*env_, streamName_.c_str(), streamName_.c_str(),
                                        "H.264 video stream from Coral Edge TPU");
    if (sms_ == nullptr) {
        APP_LOG_ERROR("Failed to create server media session");
        return false;
    }
    
    // Create groupsock for RTP
    struct sockaddr_storage destinationAddress;
    destinationAddress.ss_family = AF_INET;
    ((struct sockaddr_in&)destinationAddress).sin_addr.s_addr = INADDR_ANY;
    const Port rtpPort(0); // Let system choose port
    const unsigned char ttl = 255;
    
    rtpGroupsock_ = new Groupsock(*env_, destinationAddress, rtpPort, ttl);
    
    // Create RTP sink
    videoSink_ = H264VideoRTPSink::createNew(*env_, rtpGroupsock_, 96);
    if (videoSink_ == nullptr) {
        APP_LOG_ERROR("Failed to create H.264 video RTP sink");
        return false;
    }
    
    // Add subsession to the media session
    sms_->addSubsession(PassiveServerMediaSubsession::createNew(*videoSink_, nullptr));
    rtspServer_->addServerMediaSession(sms_);
    
    // Log the RTSP URL
    char* url = rtspServer_->rtspURL(sms_);
    APP_LOG_INFO("RTSP server started on port " + std::to_string(rtspPort_) + ", stream URL: " + std::string(url));
    delete[] url;
    
    // Start threads
    running_ = true;
    serverThread_ = std::thread(&RTSPServerWrapper::serverThread, this);
    streamThread_ = std::thread(&RTSPServerWrapper::streamThread, this);
    
    APP_LOG_INFO("RTSP server started on port " + std::to_string(rtspPort_));
    return true;
}

void RTSPServerWrapper::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    // Close the RTSP server to interrupt the event loop
    if (rtspServer_ != nullptr) {
        Medium::close(rtspServer_);
        rtspServer_ = nullptr;
    }
    
    if (serverThread_.joinable()) {
        serverThread_.join();
    }
    
    if (streamThread_.joinable()) {
        streamThread_.join();
    }
    
    // Additional cleanup for Live555 objects
    if (sms_ != nullptr) {
        Medium::close(sms_);
        sms_ = nullptr;
    }
    
    if (videoSink_ != nullptr) {
        Medium::close(videoSink_);
        videoSink_ = nullptr;
    }
    
    if (rtpGroupsock_ != nullptr) {
        delete rtpGroupsock_;
        rtpGroupsock_ = nullptr;
    }
    
    videoSource_ = nullptr;
    bufferSource_ = nullptr;
    
    if (env_ != nullptr) {
        env_->reclaim();
        env_ = nullptr;
    }
    
    if (scheduler_ != nullptr) {
        delete scheduler_;
        scheduler_ = nullptr;
    }
    
    APP_LOG_INFO("RTSP server stopped");
}

void RTSPServerWrapper::serverThread() {
    APP_LOG_INFO("RTSP server thread started");
    
    // Run the event loop with proper shutdown handling
    if (env_ && scheduler_) {
        // Live555's doEventLoop() expects an EventLoopWatchVariable* which is std::atomic<char>*
        std::atomic<char> continueEventLoop{1}; // 1 = continue, 0 = stop
        
        while (running_) {
            // Process events - this will block until an event occurs or the flag becomes 0
            env_->taskScheduler().doEventLoop(&continueEventLoop);
            
            // Check if shutdown was requested and update the flag accordingly
            if (!running_) {
                continueEventLoop = 0;
            }
            
            // Brief pause to prevent busy waiting if the event loop returns
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    APP_LOG_INFO("RTSP server thread stopped");
}

void RTSPServerWrapper::streamThread() {
    APP_LOG_INFO("RTSP stream thread started");
    
    std::shared_ptr<H264Buffer> last_frame;
    
    while (running_) {
        std::shared_ptr<H264Buffer> frame;
        
        {
            std::lock_guard<std::mutex> lock(latest_mutex_);
            frame = latest_buffer_;
        }
        
        // Only process if we have a new frame
        if (frame && frame != last_frame) {
            last_frame = frame;
            
            // Here we would send the H.264 data to the clients
            // This is a simplified implementation - in a real implementation,
            // we would need to properly format the data for RTP streaming
            APP_LOG_DEBUG("Processing H.264 buffer with size: " + std::to_string(frame->size));
            
            // In a real implementation, this would be non-blocking send to RTSP clients
            // For now, we just release the frame immediately
        }
        
        // Minimal sleep to prevent busy-waiting, but not holding any buffers
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    APP_LOG_INFO("RTSP stream thread stopped");
}

void RTSPServerWrapper::pushH264Data(std::shared_ptr<H264Buffer> buffer) {
    if (!running_) {
        return;
    }
    
#ifdef DEBUG_MODE
    // Record RTSP push start time
    auto rtsp_push_start_time = std::chrono::high_resolution_clock::now();
#endif
    
    {
        std::lock_guard<std::mutex> lock(latest_mutex_);
        latest_buffer_ = std::move(buffer);
    }
    
#ifdef DEBUG_MODE
    // Record RTSP push end time
    auto rtsp_push_end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(rtsp_push_end_time - rtsp_push_start_time).count();
    APP_LOG_DEBUG("RTSP push time: " + std::to_string(duration_us) + " us");
#endif
}