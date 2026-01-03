#pragma once

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <vector>
#include <string>
#include "pipeline_structs.h"

class GpuOverlay {
public:
    GpuOverlay(int width, int height);
    ~GpuOverlay();

    bool initialize();
    void render(uint8_t* bgr_data, const DetectionResults& detections, const OverlayBallisticPoint* ballistic_point, uint64_t frame_counter);

private:
    int width_;
    int height_;
    EGLDisplay egl_display_;
    EGLContext egl_context_;
    EGLSurface egl_surface_;

    GLuint program_;
    GLuint texture_id_;
    GLuint vbo_;
    GLuint font_texture_id_;

    void setup_shaders();
    void setup_textures();
    void setup_geometry();
    void draw_rect(float x1, float y1, float x2, float y2, float r, float g, float b, float thickness);
    void draw_line(float x1, float y1, float x2, float y2, float r, float g, float b, float thickness);
    void draw_text(const std::string& text, float x, float y, float size, float r, float g, float b);
    void draw_circle(float cx, float cy, float radius, float r, float g, float b, float thickness);
    void draw_marker(float x, float y, float r, float g, float b, float thickness);
};
