#include "pipeline_structs.h"
#include <atomic>

// Global running flag for all worker loops
std::atomic<bool> g_running{true};

// Global frame counter definition
std::atomic<int> ImageData::global_frame_counter{0};