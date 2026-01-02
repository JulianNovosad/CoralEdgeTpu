#include "queue_monitor.h"
#include "util_logging.h"

QueueMonitor::QueueMonitor() {
}

QueueMonitor::~QueueMonitor() {
}

void QueueMonitor::start() {
    APP_LOG_INFO("QueueMonitor: Monitoring started");
}

void QueueMonitor::stop() {
    APP_LOG_INFO("QueueMonitor: Monitoring stopped");
}
