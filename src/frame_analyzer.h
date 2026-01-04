#ifndef FRAME_ANALYZER_H
#define FRAME_ANALYZER_H

#include <vector>
#include <string>
#include <chrono>
#include <deque>
#include <numeric> // For std::accumulate

class FrameAnalyzer {
public:
    FrameAnalyzer(int width, int height);

    void analyze_frame(const uint8_t* bgr_data, uint64_t frame_counter, std::chrono::steady_clock::time_point capture_time);
    
private:
    int width_;
    int height_;
    size_t frame_size_bytes_;
    size_t half_frame_size_bytes_;

    uint32_t top_half_checksum_prev_ = 0;
    uint32_t bottom_half_checksum_prev_ = 0;
    
    std::vector<uint8_t> top_ref_pixels_prev_;
    std::vector<uint8_t> bottom_ref_pixels_prev_;

    // Helper to calculate a simple sum checksum (for quick change detection)
    uint32_t calculate_simple_checksum(const uint8_t* data, size_t length) const;

    // Helper to extract reference pixels
    std::vector<uint8_t> extract_ref_pixels(const uint8_t* data, int region_width, int region_height) const;
};

#endif // FRAME_ANALYZER_H
