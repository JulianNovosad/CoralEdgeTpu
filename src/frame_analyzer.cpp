#include "frame_analyzer.h"
#include "util_logging.h" // For APP_LOG_INFO
#include <iomanip> // For std::hex, std::setw, std::setfill
#include <sstream> // For std::stringstream

using namespace AppLogger;

FrameAnalyzer::FrameAnalyzer(int width, int height)
    : width_(width), height_(height),
      frame_size_bytes_(width * height * 3), // Assuming BGR888
      half_frame_size_bytes_(width * (height / 2) * 3) {
    top_ref_pixels_prev_.resize(9); // 3 pixels * 3 bytes/pixel
    bottom_ref_pixels_prev_.resize(9);
}

uint32_t FrameAnalyzer::calculate_simple_checksum(const uint8_t* data, size_t length) const {
    uint32_t sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += data[i];
    }
    return sum;
}

std::vector<uint8_t> FrameAnalyzer::extract_ref_pixels(const uint8_t* data, int region_width, int region_height) const {
    std::vector<uint8_t> ref_pixels;
    ref_pixels.reserve(9); // 3 pixels (B,G,R) * 3 positions

    // Top-left corner (0,0)
    ref_pixels.push_back(data[0]); // B
    ref_pixels.push_back(data[1]); // G
    ref_pixels.push_back(data[2]); // R

    // Center (region_width/2, region_height/2)
    size_t center_idx = (region_height / 2 * region_width + region_width / 2) * 3;
    if (center_idx + 2 < region_width * region_height * 3) {
        ref_pixels.push_back(data[center_idx]);     // B
        ref_pixels.push_back(data[center_idx + 1]); // G
        ref_pixels.push_back(data[center_idx + 2]); // R
    } else { // Fallback if center_idx is out of bounds due to small region or calculation
        ref_pixels.push_back(0); ref_pixels.push_back(0); ref_pixels.push_back(0);
    }
    

    // Bottom-right corner (region_width-1, region_height-1)
    size_t br_idx = ((region_height - 1) * region_width + (region_width - 1)) * 3;
    if (br_idx + 2 < region_width * region_height * 3) {
        ref_pixels.push_back(data[br_idx]);     // B
        ref_pixels.push_back(data[br_idx + 1]); // G
        ref_pixels.push_back(data[br_idx + 2]); // R
    } else { // Fallback
        ref_pixels.push_back(0); ref_pixels.push_back(0); ref_pixels.push_back(0);
    }

    return ref_pixels;
}


void FrameAnalyzer::analyze_frame(const uint8_t* bgr_data, uint64_t frame_counter, std::chrono::steady_clock::time_point capture_time) {
    // Ensure bgr_data is not null
    if (!bgr_data) {
        APP_LOG_ERROR("FrameAnalyzer: Received null bgr_data.");
        return;
    }

    const uint8_t* top_half_data = bgr_data;
    const uint8_t* bottom_half_data = bgr_data + half_frame_size_bytes_;

    uint32_t current_top_checksum = calculate_simple_checksum(top_half_data, half_frame_size_bytes_);
    uint32_t current_bottom_checksum = calculate_simple_checksum(bottom_half_data, half_frame_size_bytes_);

    std::vector<uint8_t> current_top_ref_pixels = extract_ref_pixels(top_half_data, width_, height_ / 2);
    std::vector<uint8_t> current_bottom_ref_pixels = extract_ref_pixels(bottom_half_data, width_, height_ / 2);

    std::stringstream ss;
    ss << "Frame " << frame_counter << " (ts: " << std::chrono::duration_cast<std::chrono::milliseconds>(capture_time.time_since_epoch()).count() << "ms): ";
    
    // Top half analysis
    ss << "Top: Chk=" << std::hex << std::setw(8) << std::setfill('0') << current_top_checksum;
    if (frame_counter > 0) { // Only compare after the first frame
        bool top_changed = (current_top_checksum != top_half_checksum_prev_ || current_top_ref_pixels != top_ref_pixels_prev_);
        ss << (top_changed ? " UPDATED" : " STALE");
    }
    if (!current_top_ref_pixels.empty()) {
        ss << " Px=[";
        for (size_t i = 0; i < current_top_ref_pixels.size(); ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)current_top_ref_pixels[i] << (i < current_top_ref_pixels.size() - 1 ? " " : "");
        }
        ss << "]";
    }

    // Bottom half analysis
    ss << " | Bottom: Chk=" << std::hex << std::setw(8) << std::setfill('0') << current_bottom_checksum;
    if (frame_counter > 0) { // Only compare after the first frame
        bool bottom_changed = (current_bottom_checksum != bottom_half_checksum_prev_ || current_bottom_ref_pixels != bottom_ref_pixels_prev_);
        ss << (bottom_changed ? " UPDATED" : " STALE");
    }
    if (!current_bottom_ref_pixels.empty()) {
        ss << " Px=[";
        for (size_t i = 0; i < current_bottom_ref_pixels.size(); ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)current_bottom_ref_pixels[i] << (i < current_bottom_ref_pixels.size() - 1 ? " " : "");
        }
        ss << "]";
    }
    APP_LOG_INFO(ss.str());

    // Update previous values for next comparison
    top_half_checksum_prev_ = current_top_checksum;
    bottom_half_checksum_prev_ = current_bottom_checksum;
    top_ref_pixels_prev_ = current_top_ref_pixels;
    bottom_ref_pixels_prev_ = current_bottom_ref_pixels;
}
