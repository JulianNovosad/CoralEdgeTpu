#ifndef RTSP_SERVER_H
#define RTSP_SERVER_H

// Include Live555 headers
#include <liveMedia/liveMedia.hh>
#include <BasicUsageEnvironment/BasicUsageEnvironment.hh>
#include <UsageEnvironment/UsageEnvironment.hh>
#include <groupsock/Groupsock.hh>
#include <groupsock/NetCommon.h>

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include "pipeline_structs.h"

class RTSPServerWrapper {
public:
    RTSPServerWrapper(int rtspPort, const std::string& streamName);
    ~RTSPServerWrapper();
    
    bool start();
    void stop();
    bool isRunning() const { return running_; }
    
    // Function to push H.264 NAL units to the stream
    void pushH264Data(std::shared_ptr<H264Buffer> buffer);

private:
    void serverThread();
    void streamThread();
    
    // Server configuration
    int rtspPort_;
    std::string streamName_;
    
    // Live555 components
    TaskScheduler* scheduler_;
    UsageEnvironment* env_;
    RTSPServer* rtspServer_;
    ServerMediaSession* sms_;
    H264VideoRTPSink* videoSink_;
    Groupsock* rtpGroupsock_;  // Add this member variable
    
    // Live555 streaming components
    H264VideoStreamFramer* videoSource_;
    ByteStreamMemoryBufferSource* bufferSource_;
    
    // Threading
    std::atomic<bool> running_;
    std::thread serverThread_;
    std::thread streamThread_;
    
    // Latest frame for RTSP streaming
    std::mutex latest_mutex_;
    std::shared_ptr<H264Buffer> latest_buffer_;
    
    // Accumulated H.264 data for streaming
    std::vector<uint8_t> accumulatedData_;
};

#endif // RTSP_SERVER_H