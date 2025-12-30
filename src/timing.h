#ifndef TIMING_H
#define TIMING_H

#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>

/**
 * @brief Authoritative timing source for the entire system.
 * Uses CLOCK_MONOTONIC_RAW to ensure immunity to NTP/system-time adjustments.
 * Implementation is static inline to prevent multiple-definition errors and minimize overhead.
 * @return Current time in milliseconds.
 */
static inline uint64_t get_time_raw_ms() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        // Critical syscall failure. Deterministic timing is impossible.
        std::cerr << "FATAL: clock_gettime(CLOCK_MONOTONIC_RAW) failed!" << std::endl;
        abort();
    }
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

#endif // TIMING_H
