#ifndef DRM_DISPLAY_H
#define DRM_DISPLAY_H

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>
#include <cstdint>
#include <memory>  // kept from old header: other files may rely on it transitively

// Minimal correct DRM output: two dumb buffers, vsync-locked flip,
// write only into the buffer that is NOT being scanned out.
class DrmDisplay {
public:
    DrmDisplay();
    ~DrmDisplay();

    bool initialize(uint32_t width, uint32_t height);
    void render_frame(const uint8_t* frame_data, uint32_t frame_width, uint32_t frame_height);
    void cleanup();
    void print_diagnostics();

    // The mode the video pipeline must be sized to. initialize() selects the
    // requested mode if offered, otherwise the display's native (preferred)
    // mode; calling initialize(0, 0) always picks the native mode.
    uint32_t display_width() const { return mode_.hdisplay; }
    uint32_t display_height() const { return mode_.vdisplay; }

private:
    int drm_fd_;
    uint32_t crtc_id_;
    uint32_t connector_id_;
    drmModeModeInfo mode_;

    uint32_t fb_[2];          // framebuffer IDs
    uint32_t handle_[2];      // dumb-buffer handles
    uint8_t* map_[2];         // CPU mappings
    uint32_t cur_;            // index currently displayed
    uint32_t bpp_;
    uint32_t pitch_;          // driver-reported pitch (may exceed width*4)
    uint32_t size_;

    uint64_t total_frames_;
    uint64_t failed_flips_;

    bool create_framebuffer(int index, uint32_t width, uint32_t height);
};

#endif // DRM_DISPLAY_H
