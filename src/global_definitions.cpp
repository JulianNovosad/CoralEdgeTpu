// Verified headers: [pipeline_structs.h, atomic]
// Verification timestamp: 2026-01-06 17:08:04
#include "pipeline_structs.h"
#include <atomic>

// Global running flag for all worker loops
std::atomic<bool> g_running{true};

// Global frame counter definition
std::atomic<int> ImageData::global_frame_counter{0};