#ifndef DRM_DISPLAY_H
#define DRM_DISPLAY_H

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>
#include <cstdint>
#include <memory>

class DrmDisplay {
public:
    DrmDisplay();
    ~DrmDisplay();
    
    bool initialize(uint32_t width, uint32_t height);
    void render_frame(const uint8_t* frame_data, uint32_t frame_width, uint32_t frame_height);
    void cleanup();
    void print_diagnostics();  // Add diagnostic method
    
private:
    int drm_fd_;
    uint32_t crtc_id_;
    uint32_t connector_id_;
    uint32_t fb_id_;
    uint32_t buffer_handle_;
    drmModeModeInfo mode_;
    uint32_t bpp_;
    uint32_t pitch_;
    uint32_t size_;
    uint8_t* map_;
    
    // Timing and diagnostics
    uint64_t last_render_time_;
    uint64_t frame_interval_us_;
    uint64_t total_frames_;
    uint64_t failed_flips_;
    uint64_t skipped_frames_;
    
    // Health monitoring
    uint64_t last_diag_time_;
    uint32_t diag_interval_ms_;
    
    bool find_connector_and_crtc();
    bool create_framebuffer(uint32_t width, uint32_t height);
    bool setup_mode();
    uint64_t get_time_us();
    bool should_render_now();
    void check_system_health();
};

#endif // DRM_DISPLAY_H