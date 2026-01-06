// Verified headers: [chrono, time.h]
// Verification timestamp: 2026-01-06 17:08:04
#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <time.h>

/**
 * @brief Get high-resolution monotonic time in milliseconds since an unspecified starting point.
 * 
 * Uses CLOCK_MONOTONIC_RAW if available, otherwise falls back to std::chrono steady_clock.
 * @return Monotonic time in milliseconds.
 */
static inline uint64_t get_time_raw_ms() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
    }
    // Fallback
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

/**
 * @brief Get high-resolution monotonic time in nanoseconds since an unspecified starting point.
 * @return Monotonic time in nanoseconds.
 */
static inline uint64_t get_time_raw_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    // Fallback
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

#endif // TIMING_H
