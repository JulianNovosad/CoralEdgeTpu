#include "drm_display.h"
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <sys/sysinfo.h>

DrmDisplay::DrmDisplay() 
    : drm_fd_(-1), crtc_id_(0), connector_id_(0), fb_id_(0), buffer_handle_(0),
      bpp_(32), pitch_(0), size_(0), map_(nullptr),
      last_render_time_(0), frame_interval_us_(16667),  // ~60 FPS target
      total_frames_(0), failed_flips_(0), skipped_frames_(0),
      last_diag_time_(0), diag_interval_ms_(5000) {     // Diagnostics every 5 seconds
    printf("DrmDisplay constructor called\n");
}

DrmDisplay::~DrmDisplay() {
    printf("DrmDisplay destructor called\n");
    print_diagnostics();
    cleanup();
}

void DrmDisplay::print_diagnostics() {
    printf("\n=== DRM DISPLAY DIAGNOSTICS ===\n");
    printf("Total frames rendered: %lu\n", total_frames_);
    printf("Failed page flips: %lu (%.2f%%)\n", failed_flips_, 
           total_frames_ > 0 ? (failed_flips_ * 100.0 / total_frames_) : 0.0);
    printf("Skipped frames: %lu (%.2f%%)\n", skipped_frames_,
           total_frames_ > 0 ? (skipped_frames_ * 100.0 / total_frames_) : 0.0);
    printf("Current framebuffer: ID=%u, Size=%u bytes\n", fb_id_, size_);
    printf("Display mode: %ux%u @ %uHz\n", mode_.hdisplay, mode_.vdisplay, mode_.vrefresh);
    printf("===============================\n\n");
}

uint64_t DrmDisplay::get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

void DrmDisplay::check_system_health() {
    uint64_t now = get_time_us();
    if ((now - last_diag_time_) / 1000 >= diag_interval_ms_) {
        // Print system memory info
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            printf("[HEALTH] Free RAM: %lu MB, Load: %.2f\n", 
                   si.freeram * si.mem_unit / (1024*1024),
                   (double)si.loads[0] / (1 << SI_LOAD_SHIFT));
        }
        
        // Print DRM diagnostics
        print_diagnostics();
        last_diag_time_ = now;
    }
}

bool DrmDisplay::should_render_now() {
    uint64_t now = get_time_us();
    return (now - last_render_time_) >= frame_interval_us_;
}

bool DrmDisplay::initialize(uint32_t width, uint32_t height) {
    printf("🎯 INITIALIZING MINIMAL DRM DISPLAY: %ux%u\n", width, height);
    
    // Open DRM device
    drm_fd_ = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (drm_fd_ < 0) {
        drm_fd_ = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
        if (drm_fd_ < 0) {
            printf("❌ ERROR: Failed to open DRM device\n");
            return false;
        }
    }
    printf("✅ DRM device opened (FD: %d)\n", drm_fd_);
    
    // Find connector and CRTC
    if (!find_connector_and_crtc()) {
        printf("❌ ERROR: Failed to find connector/CRTC\n");
        return false;
    }
    printf("✅ Connector: %u, CRTC: %u\n", connector_id_, crtc_id_);
    
    // Create single framebuffer
    if (!create_framebuffer(width, height)) {
        printf("❌ ERROR: Failed to create framebuffer\n");
        return false;
    }
    printf("✅ Framebuffer created: ID=%u, Size=%u bytes\n", fb_id_, size_);
    
    // Setup display mode
    if (!setup_mode()) {
        printf("❌ ERROR: Failed to setup display mode\n");
        return false;
    }
    
    last_render_time_ = get_time_us();
    printf("🎯 MINIMAL DRM DISPLAY READY\n");
    return true;
}

bool DrmDisplay::find_connector_and_crtc() {
    // Use known working IDs
    connector_id_ = 32;
    crtc_id_ = 91;
    return true;
}

bool DrmDisplay::create_framebuffer(uint32_t width, uint32_t height) {
    pitch_ = width * (bpp_ / 8);
    size_ = pitch_ * height;
    
    // Create dumb buffer
    drm_mode_create_dumb create_request = {};
    create_request.width = width;
    create_request.height = height;
    create_request.bpp = bpp_;
    
    if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_CREATE_DUMB, &create_request) < 0) {
        printf("❌ Create dumb buffer failed: %s\n", strerror(errno));
        return false;
    }
    
    buffer_handle_ = create_request.handle;
    
    // Create framebuffer
    uint32_t offsets[4] = {0};
    uint32_t pitches[4] = {pitch_};
    uint32_t bo_handles[4] = {buffer_handle_};
    
    if (drmModeAddFB2(drm_fd_, width, height, DRM_FORMAT_XRGB8888,
                      bo_handles, pitches, offsets, &fb_id_, 0) != 0) {
        if (drmModeAddFB(drm_fd_, width, height, 24, bpp_, pitch_, buffer_handle_, &fb_id_) != 0) {
            printf("❌ Create framebuffer failed: %s\n", strerror(errno));
            drm_mode_destroy_dumb destroy_request = {};
            destroy_request.handle = buffer_handle_;
            drmIoctl(drm_fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_request);
            return false;
        }
    }
    
    // Map buffer
    drm_mode_map_dumb map_request = {};
    map_request.handle = buffer_handle_;
    
    if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_MAP_DUMB, &map_request) < 0) {
        printf("❌ Map buffer failed: %s\n", strerror(errno));
        drmModeRmFB(drm_fd_, fb_id_);
        drm_mode_destroy_dumb destroy_request = {};
        destroy_request.handle = buffer_handle_;
        drmIoctl(drm_fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_request);
        return false;
    }
    
    map_ = static_cast<uint8_t*>(mmap(0, size_, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd_, map_request.offset));
    if (map_ == MAP_FAILED) {
        printf("❌ mmap failed: %s\n", strerror(errno));
        drmModeRmFB(drm_fd_, fb_id_);
        drm_mode_destroy_dumb destroy_request = {};
        destroy_request.handle = buffer_handle_;
        drmIoctl(drm_fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_request);
        return false;
    }
    
    return true;
}

bool DrmDisplay::setup_mode() {
    drmModeRes* resources = drmModeGetResources(drm_fd_);
    if (!resources) return false;
    
    drmModeConnector* connector = drmModeGetConnector(drm_fd_, connector_id_);
    if (!connector || connector->count_modes == 0) {
        drmModeFreeResources(resources);
        return false;
    }
    
    mode_ = connector->modes[0];
    
    if (drmModeSetCrtc(drm_fd_, crtc_id_, fb_id_, 0, 0, &connector_id_, 1, &mode_) != 0) {
        drmModeFreeConnector(connector);
        drmModeFreeResources(resources);
        return false;
    }
    
    drmModeFreeConnector(connector);
    drmModeFreeResources(resources);
    return true;
}

void DrmDisplay::render_frame(const uint8_t* frame_data, uint32_t frame_width, uint32_t frame_height) {
    if (!map_ || !frame_data) {
        printf("DRM: ❌ Invalid frame data or mapping\n");
        return;
    }
    
    // Check system health periodically
    check_system_health();
    
    // Rate limit to target FPS
    if (!should_render_now()) {
        skipped_frames_++;
        return;
    }
    
    // Validate framebuffer integrity
    if (size_ == 0 || pitch_ == 0) {
        printf("DRM: ❌ Invalid framebuffer parameters\n");
        return;
    }
    
    // Simple nearest-neighbor scaling
    uint32_t disp_width = mode_.hdisplay;
    uint32_t disp_height = mode_.vdisplay;
    
    float scale_x = static_cast<float>(disp_width) / frame_width;
    float scale_y = static_cast<float>(disp_height) / frame_height;
    
    // Direct BGR to ARGB conversion with scaling and bounds checking
    for (uint32_t y = 0; y < disp_height; y++) {
        for (uint32_t x = 0; x < disp_width; x++) {
            uint32_t src_x = static_cast<uint32_t>(x / scale_x);
            uint32_t src_y = static_cast<uint32_t>(y / scale_y);
            
            if (src_x < frame_width && src_y < frame_height) {
                // Bounds check for source data
                size_t src_offset = (src_y * frame_width + src_x) * 3;
                if (src_offset + 2 >= frame_width * frame_height * 3) {
                    continue;
                }
                
                const uint8_t* src = frame_data + src_offset;
                uint32_t* dst = reinterpret_cast<uint32_t*>(map_ + y * pitch_ + x * 4);
                
                // Additional bounds check for destination
                if ((uint8_t*)dst >= map_ + size_) {
                    continue;
                }
                
                *dst = 0xFF000000 |           // Alpha
                       (src[2] << 16) |       // Red
                       (src[1] << 8) |        // Green
                       src[0];                // Blue
            }
        }
    }
    
    // Page flip with error handling
    int ret = drmModePageFlip(drm_fd_, crtc_id_, fb_id_, DRM_MODE_PAGE_FLIP_EVENT, nullptr);
    last_render_time_ = get_time_us();
    total_frames_++;
    
    if (ret != 0) {
        failed_flips_++;
        printf("❌ PAGE FLIP FAILED (%lu): %s (errno: %d)\n", 
               failed_flips_, strerror(-ret), -ret);
        
        // If we're failing consistently, try to recover
        if (failed_flips_ > 10 && (failed_flips_ * 100 / total_frames_) > 50) {
            printf("⚠️  High flip failure rate, attempting recovery...\n");
            // Reset timing to avoid getting stuck
            last_render_time_ = get_time_us();
        }
    } else if (total_frames_ % 300 == 0) {  // Log success every 300 frames
        printf("✅ PAGE FLIP SUCCESSFUL: %lu frames rendered\n", total_frames_);
    }
}

void DrmDisplay::cleanup() {
    printf("🧹 Cleaning up DRM resources...\n");
    
    if (map_) {
        munmap(map_, size_);
        map_ = nullptr;
    }
    
    if (fb_id_) {
        drmModeRmFB(drm_fd_, fb_id_);
        fb_id_ = 0;
    }
    
    if (buffer_handle_) {
        drm_mode_destroy_dumb destroy_request = {};
        destroy_request.handle = buffer_handle_;
        drmIoctl(drm_fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_request);
        buffer_handle_ = 0;
    }
    
    if (drm_fd_ >= 0) {
        close(drm_fd_);
        drm_fd_ = -1;
    }
    
    printf("✅ DRM cleanup complete\n");
}