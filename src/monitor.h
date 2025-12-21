#ifndef MONITOR_H
#define MONITOR_H

#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

// Forward declaration to avoid circular dependency
class Application;

class Monitor {
public:
    Monitor(Application& app);
    ~Monitor();
    
    void start();
    void stop();
    
private:
    void monitor_thread_func();
    
    Application& app_;
    std::atomic<bool> running_;
    std::thread monitor_thread_;
};

#endif // MONITOR_H