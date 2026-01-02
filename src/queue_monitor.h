#ifndef QUEUE_MONITOR_H
#define QUEUE_MONITOR_H

#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include "pipeline_structs.h"

/**
 * @brief Monitors queue depths and latencies throughout the pipeline.
 */
class QueueMonitor {
public:
    QueueMonitor();
    ~QueueMonitor();

    void start();
    void stop();

    // No-op for now as it's mostly for logging/display
};

#endif // QUEUE_MONITOR_H
