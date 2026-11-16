// drm_display.cpp — minimal correct DRM output.
//
// What the old version did wrong:
//   1. ONE dumb buffer written in place while the CRTC was scanning it out
//      (torn rows = the RGB banding), and page-flips issued to the already
//      displayed framebuffer (no-ops).
//   2. A per-pixel scalar loop with two float divisions per pixel plus
//      bounds-check `continue`s that left stale pixels behind (latency
//      buildup + more banding).
//   3. DRM_MODE_PAGE_FLIP_EVENT with no event handler — events queued and
//      never drained; a second flip before vsync returned -EBUSY forever.
//
// What this version does:
//   - Two dumb buffers; every frame is rendered into the buffer that is NOT
//     being scanned out.
//   - The pipeline delivers XRGB8888 at the display's native mode, so the
//     hot path is a plain memcpy. If the camera was constrained to a smaller
//     resolution (reduced-res config), one NEON cv::resize is used instead.
//   - vsync-locked: flip with DRM_MODE_PAGE_FLIP_EVENT, then wait for the
//     flip-complete event (bounded by a 100 ms poll timeout).
//   - Assumes 4 bytes/pixel (stride = width*4), which is what the pipeline
//     produces (ISP XRGB8888 output, packed by the visualization processor).

#include "drm_display.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <sys/mman.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

void flip_handler(int, unsigned, unsigned, unsigned, void* user_data) {
    *static_cast<bool*>(user_data) = true;
}

}  // namespace

DrmDisplay::DrmDisplay()
    : drm_fd_(-1), crtc_id_(0), connector_id_(0),
      cur_(0), bpp_(32), pitch_(0), size_(0),
      total_frames_(0), failed_flips_(0) {
    fb_[0] = fb_[1] = 0;
    handle_[0] = handle_[1] = 0;
    map_[0] = map_[1] = nullptr;
    memset(&mode_, 0, sizeof(mode_));
}

DrmDisplay::~DrmDisplay() {
    print_diagnostics();
    cleanup();
}

bool DrmDisplay::initialize(uint32_t width, uint32_t height) {
    // Prefer card1 (Pi 5 HDMI under vc4/v3d), fall back to card0.
    drm_fd_ = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (drm_fd_ < 0) drm_fd_ = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (drm_fd_ < 0) {
        perror("DrmDisplay: open /dev/dri/cardX");
        return false;
    }

    drmModeRes* res = drmModeGetResources(drm_fd_);
    if (!res) {
        fprintf(stderr, "DrmDisplay: drmModeGetResources failed\n");
        cleanup();
        return false;
    }

    // Enumerate; never hardcode connector/CRTC ids (they change per boot).
    drmModeConnector* conn = nullptr;
    for (int i = 0; i < res->count_connectors && !conn; i++) {
        drmModeConnector* c = drmModeGetConnector(drm_fd_, res->connectors[i]);
        if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
            conn = c;
        } else {
            drmModeFreeConnector(c);
        }
    }
    if (!conn) {
        fprintf(stderr, "DrmDisplay: no connected display found\n");
        drmModeFreeResources(res);
        cleanup();
        return false;
    }
    connector_id_ = conn->connector_id;

    // Requested mode if the display offers it, otherwise its preferred mode.
    // width/height == 0 means: always use the native (preferred) mode.
    mode_ = conn->modes[0];
    if (width != 0 && height != 0) {
        for (int i = 0; i < conn->count_modes; i++) {
            if ((uint32_t)conn->modes[i].hdisplay == width &&
                (uint32_t)conn->modes[i].vdisplay == height) {
                mode_ = conn->modes[i];
                break;
            }
        }
    }

    // CRTC: the connector's bound encoder first, else any possible CRTC.
    if (conn->encoder_id) {
        drmModeEncoder* enc = drmModeGetEncoder(drm_fd_, conn->encoder_id);
        if (enc) {
            crtc_id_ = enc->crtc_id;
            drmModeFreeEncoder(enc);
        }
    }
    for (int e = 0; e < conn->count_encoders && !crtc_id_; e++) {
        drmModeEncoder* enc = drmModeGetEncoder(drm_fd_, conn->encoders[e]);
        if (!enc) continue;
        for (int k = 0; k < res->count_crtcs; k++) {
            if (enc->possible_crtcs & (1u << k)) {
                crtc_id_ = res->crtcs[k];
                break;
            }
        }
        drmModeFreeEncoder(enc);
    }
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    if (!crtc_id_) {
        fprintf(stderr, "DrmDisplay: no CRTC available for connector %u\n", connector_id_);
        cleanup();
        return false;
    }

    for (int i = 0; i < 2; i++) {
        if (!create_framebuffer(i, mode_.hdisplay, mode_.vdisplay)) {
            cleanup();
            return false;
        }
    }
    cur_ = 0;

    if (drmModeSetCrtc(drm_fd_, crtc_id_, fb_[0], 0, 0, &connector_id_, 1, &mode_) != 0) {
        perror("DrmDisplay: drmModeSetCrtc");
        cleanup();
        return false;
    }

    printf("DrmDisplay: mode %ux%u, connector %u, crtc %u, pitch %u\n",
           mode_.hdisplay, mode_.vdisplay, connector_id_, crtc_id_, pitch_);
    return true;
}

bool DrmDisplay::create_framebuffer(int index, uint32_t width, uint32_t height) {
    drm_mode_create_dumb create = {};
    create.width = width;
    create.height = height;
    create.bpp = 32;
    if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_CREATE_DUMB, &create) != 0) {
        perror("DrmDisplay: DRM_IOCTL_MODE_CREATE_DUMB");
        return false;
    }
    handle_[index] = create.handle;
    // Driver-reported pitch/size: may be aligned larger than width*4.
    pitch_ = create.pitch;
    size_ = create.size;

    uint32_t handles[1] = { handle_[index] };
    uint32_t pitches[1] = { create.pitch };
    uint32_t offsets[1] = { 0 };
    if (drmModeAddFB2(drm_fd_, width, height, DRM_FORMAT_XRGB8888,
                      handles, pitches, offsets, &fb_[index], 0) != 0) {
        perror("DrmDisplay: drmModeAddFB2");
        return false;
    }

    drm_mode_map_dumb map_req = {};
    map_req.handle = handle_[index];
    if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_MAP_DUMB, &map_req) != 0) {
        perror("DrmDisplay: DRM_IOCTL_MODE_MAP_DUMB");
        return false;
    }
    void* m = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd_, map_req.offset);
    if (m == MAP_FAILED) {
        perror("DrmDisplay: mmap");
        return false;
    }
    map_[index] = static_cast<uint8_t*>(m);
    memset(map_[index], 0, size_);
    return true;
}

void DrmDisplay::render_frame(const uint8_t* frame_data, uint32_t frame_width, uint32_t frame_height) {
    if (drm_fd_ < 0 || !map_[0] || !frame_data || frame_width == 0 || frame_height == 0) return;

    const uint32_t next = 1 - cur_;

    if (frame_width == (uint32_t)mode_.hdisplay && frame_height == (uint32_t)mode_.vdisplay) {
        // Frame already is XRGB8888 at the display's native mode: the whole
        // "render" is a copy into the hidden buffer.
        const size_t row_bytes = (size_t)frame_width * 4;
        if ((size_t)pitch_ == row_bytes) {
            memcpy(map_[next], frame_data, row_bytes * frame_height);
        } else {
            // Driver-aligned pitch (rare): copy row by row.
            uint8_t* dst = map_[next];
            for (uint32_t y = 0; y < frame_height; y++) {
                memcpy(dst + (size_t)y * pitch_, frame_data + (size_t)y * row_bytes, row_bytes);
            }
        }
    } else {
        // Camera constrained to a smaller resolution (reduced-res config):
        // one NEON resize into the hidden buffer.
        const cv::Mat src(frame_height, frame_width, CV_8UC4,
                          const_cast<uint8_t*>(frame_data), frame_width * 4);
        const cv::Mat dst(mode_.vdisplay, mode_.hdisplay, CV_8UC4, map_[next], pitch_);
        cv::resize(src, dst, cv::Size(mode_.hdisplay, mode_.vdisplay), 0, 0, cv::INTER_NEAREST);
    }
    cur_ = next;

    // Flip, then wait for the flip-complete event: the buffer just retired
    // is never rewritten while it is still being scanned out. The poll
    // timeout bounds the wait if events stop arriving.
    bool flipped = false;
    const int ret = drmModePageFlip(drm_fd_, crtc_id_, fb_[cur_],
                                    DRM_MODE_PAGE_FLIP_EVENT, &flipped);
    if (ret == 0) {
        drmEventContext ctx = {};
        ctx.version = DRM_EVENT_CONTEXT_VERSION;
        ctx.page_flip_handler = flip_handler;
        struct pollfd pfd = { drm_fd_, POLLIN, 0 };
        while (!flipped && poll(&pfd, 1, 100) > 0) {
            if (drmHandleEvent(drm_fd_, &ctx) != 0) break;
        }
        total_frames_++;
    } else {
        failed_flips_++;
        // -EBUSY = a flip is still pending from another client; harmless,
        // the frame is simply dropped and the next one wins.
        if (ret != -EBUSY) {
            fprintf(stderr, "DrmDisplay: pageflip failed: %s\n", strerror(-ret));
        }
    }
}

void DrmDisplay::cleanup() {
    for (int i = 0; i < 2; i++) {
        if (map_[i]) {
            munmap(map_[i], size_);
            map_[i] = nullptr;
        }
        if (fb_[i]) {
            drmModeRmFB(drm_fd_, fb_[i]);
            fb_[i] = 0;
        }
        if (handle_[i]) {
            drm_mode_destroy_dumb destroy = {};
            destroy.handle = handle_[i];
            drmIoctl(drm_fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
            handle_[i] = 0;
        }
    }
    if (drm_fd_ >= 0) {
        close(drm_fd_);
        drm_fd_ = -1;
    }
}

void DrmDisplay::print_diagnostics() {
    printf("DrmDisplay: %llu frames rendered, %llu failed flips\n",
           (unsigned long long)total_frames_, (unsigned long long)failed_flips_);
}
