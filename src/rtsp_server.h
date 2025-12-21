#ifndef RTSP_SERVER_H
#define RTSP_SERVER_H

// Include Live555 headers
#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <UsageEnvironment.hh>
#include <Groupsock.hh>
#include <GroupsockHelper.hh>
#include <H264VideoRTPSink.hh>
#include <H264VideoStreamFramer.hh>
#include <RTSPServer.hh>
#include <ServerMediaSession.hh>
#include <ByteStreamMemoryBufferSource.hh>

#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
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
    
    // Buffer queue for H.264 data
    std::queue<std::shared_ptr<H264Buffer>> bufferQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCondition_;
    
    // Accumulated H.264 data for streaming
    std::vector<uint8_t> accumulatedData_;
};

#endif // RTSP_SERVER_H