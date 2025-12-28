#ifndef UDP_STREAMER_H
#define UDP_STREAMER_H

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include "pipeline_structs.h"

class UDPStreamer {
public:
    UDPStreamer(int width, int height, double fps);
    ~UDPStreamer();

    bool start();
    void stop();

    void pushH264Data(std::shared_ptr<H264Buffer> buffer);

private:
    void setup_pipeline();
    void worker_thread_func();
    
    GstElement *pipeline_, *appsrc_;
    GMainLoop *loop_;
    std::thread server_thread_;
    std::atomic<bool> running_;
    
    int width_, height_;
    double fps_;
    
    mutable std::mutex appsrc_mutex_;
    
    GstClockTime base_time_;
    GstClockTime last_pts_;
};

#endif // UDP_STREAMER_H