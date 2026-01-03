#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <gbm.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include "gpu_overlay.h"
#include "util_logging.h"
#include <iostream>
#include <vector>
#include <cmath>

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA     0x31DD
#endif

#ifndef EGL_PLATFORM_GBM_MESA
#define EGL_PLATFORM_GBM_MESA             0x31D7
#endif

// Simple 8x8 bitmap font (subset)
static const uint8_t font8x8_basic[128][8] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 0x00
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // Space
    {0x18,0x3c,0x3c,0x18,0x18,0x00,0x18,0x00}, // !
    {0x6c,0x6c,0x6c,0x00,0x00,0x00,0x00,0x00}, // "
    {0x6c,0x6c,0xfe,0x6c,0xfe,0x6c,0x6c,0x00}, // #
    {0x18,0x3e,0x60,0x3c,0x06,0x7c,0x18,0x00}, // $
    {0x00,0xc6,0xcc,0x18,0x30,0x66,0xc6,0x00}, // %
    {0x38,0x6c,0x38,0x76,0xdc,0xcc,0x76,0x00}, // &
    {0x30,0x30,0x60,0x00,0x00,0x00,0x00,0x00}, // '
    {0x0c,0x18,0x30,0x30,0x30,0x18,0x0c,0x00}, // (
    {0x30,0x18,0x0c,0x0c,0x0c,0x18,0x30,0x00}, // )
    {0x00,0x66,0x3c,0xff,0x3c,0x66,0x00,0x00}, // *
    {0x00,0x18,0x18,0x7e,0x18,0x18,0x00,0x00}, // +
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x30}, // ,
    {0x00,0x00,0x00,0x7e,0x00,0x00,0x00,0x00}, // -
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00}, // .
    {0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x00}, // /
    {0x3c,0x66,0x6e,0x7e,0x76,0x66,0x3c,0x00}, // 0
    {0x18,0x38,0x18,0x18,0x18,0x18,0x7e,0x00}, // 1
    {0x3c,0x66,0x06,0x0c,0x18,0x30,0x7e,0x00}, // 2
    {0x3c,0x66,0x06,0x1c,0x06,0x66,0x3c,0x00}, // 3
    {0x1c,0x3c,0x6c,0xcc,0xfe,0x0c,0x1e,0x00}, // 4
    {0x7e,0x60,0x7c,0x06,0x06,0x66,0x3c,0x00}, // 5
    {0x1c,0x30,0x60,0x7c,0x66,0x66,0x3c,0x00}, // 6
    {0x7e,0x66,0x06,0x0c,0x18,0x18,0x18,0x00}, // 7
    {0x3c,0x66,0x66,0x3c,0x66,0x66,0x3c,0x00}, // 8
    {0x3c,0x66,0x66,0x3e,0x06,0x0c,0x38,0x00}, // 9
    {0x00,0x18,0x18,0x00,0x18,0x18,0x00,0x00}, // :
    {0x00,0x18,0x18,0x00,0x18,0x18,0x30,0x00}, // ;
    {0x0c,0x18,0x30,0x60,0x30,0x18,0x0c,0x00}, // <
    {0x00,0x00,0x7e,0x00,0x7e,0x00,0x00,0x00}, // =
    {0x30,0x18,0x0c,0x06,0x0c,0x18,0x30,0x00}, // >
    {0x3c,0x66,0x06,0x0c,0x18,0x00,0x18,0x00}, // ?
    {0x3c,0x66,0x6e,0x6e,0x60,0x3e,0x00,0x00}, // @
    {0x18,0x3c,0x66,0x66,0x7e,0x66,0x66,0x00}, // A
    {0x7c,0x66,0x66,0x7c,0x66,0x66,0x7c,0x00}, // B
    {0x3c,0x66,0x60,0x60,0x60,0x66,0x3c,0x00}, // C
    {0x78,0x6c,0x66,0x66,0x66,0x6c,0x78,0x00}, // D
    {0x7e,0x60,0x60,0x78,0x60,0x60,0x7e,0x00}, // E
    {0x7e,0x60,0x60,0x78,0x60,0x60,0x60,0x00}, // F
    {0x3c,0x66,0x60,0x6e,0x66,0x66,0x3e,0x00}, // G
    {0x66,0x66,0x66,0x7e,0x66,0x66,0x66,0x00}, // H
    {0x3c,0x18,0x18,0x18,0x18,0x18,0x3c,0x00}, // I
    {0x1e,0x0c,0x0c,0x0c,0x0c,0xcc,0x78,0x00}, // J
    {0x66,0x6c,0x78,0x70,0x78,0x6c,0x66,0x00}, // K
    {0x60,0x60,0x60,0x60,0x60,0x60,0x7e,0x00}, // L
    {0xc6,0xee,0xfe,0xfe,0xd6,0xc6,0xc6,0x00}, // M
    {0xc6,0xe6,0xf6,0xde,0xce,0xc6,0xc6,0x00}, // N
    {0x3c,0x66,0x66,0x66,0x66,0x66,0x3c,0x00}, // O
    {0x7c,0x66,0x66,0x7c,0x60,0x60,0x60,0x00}, // P
    {0x3c,0x66,0x66,0x66,0x66,0x3c,0x0e,0x00}, // Q
    {0x7c,0x66,0x66,0x7c,0x78,0x6c,0x66,0x00}, // R
    {0x3c,0x66,0x30,0x18,0x0c,0x66,0x3c,0x00}, // S
    {0x7e,0x5a,0x18,0x18,0x18,0x18,0x18,0x00}, // T
    {0x66,0x66,0x66,0x66,0x66,0x66,0x3c,0x00}, // U
    {0x66,0x66,0x66,0x66,0x66,0x3c,0x18,0x00}, // V
    {0xc6,0xc6,0xc6,0xd6,0xfe,0xee,0xc6,0x00}, // W
    {0x66,0x66,0x3c,0x18,0x3c,0x66,0x66,0x00}, // X
    {0x66,0x66,0x66,0x3c,0x18,0x18,0x18,0x00}, // Y
    {0x7e,0x06,0x0c,0x18,0x30,0x60,0x7e,0x00}, // Z
    {0x3c,0x30,0x30,0x30,0x30,0x30,0x3c,0x00}, // [
    {0x80,0x40,0x20,0x10,0x08,0x04,0x02,0x00}, // \
    {0x3c,0x0c,0x0c,0x0c,0x0c,0x0c,0x3c,0x00}, // ]
    {0x10,0x38,0x6c,0xc6,0x00,0x00,0x00,0x00}, // ^
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff}, // _
    {0x30,0x30,0x18,0x00,0x00,0x00,0x00,0x00}, // `
    {0x00,0x00,0x3c,0x06,0x3e,0x66,0x3e,0x00}, // a
    {0x60,0x60,0x7c,0x66,0x66,0x66,0x7c,0x00}, // b
    {0x00,0x00,0x3c,0x60,0x60,0x66,0x3c,0x00}, // c
    {0x06,0x06,0x3e,0x66,0x66,0x66,0x3e,0x00}, // d
    {0x00,0x00,0x3c,0x66,0x7e,0x60,0x3c,0x00}, // e
    {0x0e,0x18,0x3e,0x18,0x18,0x18,0x18,0x00}, // f
    {0x00,0x00,0x3e,0x66,0x66,0x3e,0x06,0x3c}, // g
    {0x60,0x60,0x7c,0x66,0x66,0x66,0x66,0x00}, // h
    {0x18,0x00,0x38,0x18,0x18,0x18,0x3c,0x00}, // i
    {0x06,0x00,0x06,0x06,0x06,0x66,0x66,0x3c}, // j
    {0x60,0x60,0x66,0x6c,0x78,0x6c,0x66,0x00}, // k
    {0x38,0x18,0x18,0x18,0x18,0x18,0x3c,0x00}, // l
    {0x00,0x00,0xfe,0xd6,0xd6,0xd6,0xd6,0x00}, // m
    {0x00,0x00,0x7c,0x66,0x66,0x66,0x66,0x00}, // n
    {0x00,0x00,0x3c,0x66,0x66,0x66,0x3c,0x00}, // o
    {0x00,0x00,0x7c,0x66,0x66,0x7c,0x60,0x60}, // p
    {0x00,0x00,0x3e,0x66,0x66,0x3e,0x06,0x06}, // q
    {0x00,0x00,0x7c,0x66,0x60,0x60,0x60,0x00}, // r
    {0x00,0x00,0x3e,0x60,0x3c,0x06,0x7c,0x00}, // s
    {0x18,0x18,0x7e,0x18,0x18,0x18,0x0e,0x00}, // t
    {0x00,0x00,0x66,0x66,0x66,0x66,0x3e,0x00}, // u
    {0x00,0x00,0x66,0x66,0x66,0x3c,0x18,0x00}, // v
    {0x00,0x00,0xc6,0xd6,0xd6,0xfe,0x6c,0x00}, // w
    {0x00,0x00,0x66,0x3c,0x18,0x3c,0x66,0x00}, // x
    {0x00,0x00,0x66,0x66,0x66,0x3e,0x06,0x3c}, // y
    {0x00,0x00,0x7e,0x0c,0x18,0x30,0x7e,0x00}, // z
    {0x0c,0x18,0x18,0x70,0x18,0x18,0x0c,0x00}, // {
    {0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00}, // |
    {0x30,0x18,0x18,0x0e,0x18,0x18,0x30,0x00}, // }
    {0x00,0x00,0x00,0x3c,0x5a,0x00,0x00,0x00}, // ~
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}
};

static const char* vertex_shader_source = R"(#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec4 aColor;
layout(location = 3) in int aType; // 0: background, 1: primitive, 2: text

out vec2 vTexCoord;
out vec4 vColor;
flat out int vType;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
    vColor = aColor;
    vType = aType;
}
)";

static const char* fragment_shader_source = R"(#version 300 es
precision mediump float;
in vec2 vTexCoord;
in vec4 vColor;
flat in int vType;

uniform sampler2D uTexture;
uniform sampler2D uFontTexture;

out vec4 fragColor;

void main() {
    if (vType == 0) {
        // Background texture (BGR to RGB swap)
        vec4 tex = texture(uTexture, vTexCoord);
        fragColor = vec4(tex.b, tex.g, tex.r, 1.0);
    } else if (vType == 1) {
        // Primitive
        fragColor = vColor;
    } else if (vType == 2) {
        // Text
        float tex = texture(uFontTexture, vTexCoord).r;
        if (tex < 0.5) discard;
        fragColor = vColor;
    }
}
)";

GpuOverlay::GpuOverlay(int width, int height) : width_(width), height_(height), egl_display_(EGL_NO_DISPLAY), egl_context_(EGL_NO_CONTEXT), egl_surface_(EGL_NO_SURFACE) {
}

GpuOverlay::~GpuOverlay() {
    if (egl_display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (egl_context_ != EGL_NO_CONTEXT) eglDestroyContext(egl_display_, egl_context_);
        if (egl_surface_ != EGL_NO_SURFACE) eglDestroySurface(egl_display_, egl_surface_);
        eglTerminate(egl_display_);
    }
}

bool GpuOverlay::initialize() {
    APP_LOG_INFO("GpuOverlay: Initializing EGL (Headless)...");

    typedef EGLDisplay (EGLAPIENTRYP PFNEGLGETPLATFORMDISPLAYPROC) (EGLenum platform, void *native_display, const EGLAttrib *attrib_list);
    PFNEGLGETPLATFORMDISPLAYPROC eglGetPlatformDisplay_ptr = (PFNEGLGETPLATFORMDISPLAYPROC)eglGetProcAddress("eglGetPlatformDisplay");

    if (eglGetPlatformDisplay_ptr) {
        // SURFACELESS is generally best for headless
        APP_LOG_INFO("GpuOverlay: Trying EGL_PLATFORM_SURFACELESS_MESA.");
        egl_display_ = eglGetPlatformDisplay_ptr(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    }

    if (egl_display_ == EGL_NO_DISPLAY) {
        APP_LOG_WARNING("GpuOverlay: eglGetPlatformDisplay(SURFACELESS) failed, trying GBM.");
        int drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
        if (drm_fd >= 0) {
            struct gbm_device* gbm = gbm_create_device(drm_fd);
            if (gbm && eglGetPlatformDisplay_ptr) {
                egl_display_ = eglGetPlatformDisplay_ptr(EGL_PLATFORM_GBM_MESA, gbm, nullptr);
            } else if (drm_fd >= 0) {
                close(drm_fd);
            }
        }
    }

    if (egl_display_ == EGL_NO_DISPLAY) {
        APP_LOG_WARNING("GpuOverlay: All platform attempts failed, trying eglGetDisplay(DEFAULT).");
        egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    }

    if (egl_display_ == EGL_NO_DISPLAY) {
        APP_LOG_ERROR("GpuOverlay: Failed to get EGL display, error: " + std::to_string(eglGetError()));
        return false;
    }

    EGLint major, minor;
    if (!eglInitialize(egl_display_, &major, &minor)) {
        APP_LOG_ERROR("GpuOverlay: eglInitialize failed, error: " + std::to_string(eglGetError()));
        return false;
    }
    APP_LOG_INFO("GpuOverlay: EGL version " + std::to_string(major) + "." + std::to_string(minor));

    EGLint num_configs_total = 0;
    if (!eglGetConfigs(egl_display_, nullptr, 0, &num_configs_total) || num_configs_total == 0) {
        APP_LOG_ERROR("GpuOverlay: eglGetConfigs failed, error: " + std::to_string(eglGetError()));
        return false;
    }
    
    std::vector<EGLConfig> all_configs(num_configs_total);
    eglGetConfigs(egl_display_, all_configs.data(), num_configs_total, &num_configs_total);

    EGLConfig config = nullptr;
    for (const auto& c : all_configs) {
        EGLint surface_type, renderable_type;
        eglGetConfigAttrib(egl_display_, c, EGL_SURFACE_TYPE, &surface_type);
        eglGetConfigAttrib(egl_display_, c, EGL_RENDERABLE_TYPE, &renderable_type);
        
        if ((surface_type & EGL_PBUFFER_BIT) && (renderable_type & (EGL_OPENGL_ES2_BIT | EGL_OPENGL_ES3_BIT))) {
            config = c;
            APP_LOG_INFO("GpuOverlay: Found suitable config with PBUFFER and GLES2/3 support.");
            break;
        }
    }

    if (!config) {
        APP_LOG_WARNING("GpuOverlay: No config with PBUFFER found, trying any GLES2/3 config.");
        for (const auto& c : all_configs) {
            EGLint renderable_type;
            eglGetConfigAttrib(egl_display_, c, EGL_RENDERABLE_TYPE, &renderable_type);
            if (renderable_type & (EGL_OPENGL_ES2_BIT | EGL_OPENGL_ES3_BIT)) {
                config = c;
                break;
            }
        }
    }

    if (!config) {
        APP_LOG_ERROR("GpuOverlay: No suitable EGL config found.");
        return false;
    }

    EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };

    egl_context_ = eglCreateContext(egl_display_, config, EGL_NO_CONTEXT, context_attribs);
    if (egl_context_ == EGL_NO_CONTEXT) {
        APP_LOG_ERROR("GpuOverlay: eglCreateContext failed, error: " + std::to_string(eglGetError()));
        return false;
    }

    // With SURFACELESS, we might not need a surface at all if EGL_KHR_surfaceless_context is supported.
    // Try to make current with NO_SURFACE first if using surfaceless platform.
    if (eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, egl_context_)) {
        APP_LOG_INFO("GpuOverlay: Made current with EGL_NO_SURFACE (Surfaceless).");
    } else {
        APP_LOG_WARNING("GpuOverlay: eglMakeCurrent(NO_SURFACE) failed, creating Pbuffer surface.");
        EGLint pbuffer_attribs[] = {
            EGL_WIDTH, width_,
            EGL_HEIGHT, height_,
            EGL_NONE
        };
        egl_surface_ = eglCreatePbufferSurface(egl_display_, config, pbuffer_attribs);
        if (egl_surface_ == EGL_NO_SURFACE) {
            APP_LOG_ERROR("GpuOverlay: eglCreatePbufferSurface failed, error: " + std::to_string(eglGetError()));
            return false;
        }
        if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
            APP_LOG_ERROR("GpuOverlay: eglMakeCurrent(Pbuffer) failed, error: " + std::to_string(eglGetError()));
            return false;
        }
    }


    APP_LOG_INFO("GpuOverlay: EGL/GLES3 context created and made current.");
    setup_shaders();
    setup_textures();
    setup_geometry();

    return true;
}

void GpuOverlay::setup_shaders() {
    auto compile_shader = [](GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char info[512];
            glGetShaderInfoLog(shader, 512, nullptr, info);
            std::cerr << "Shader compile error: " << info << std::endl;
        }
        return shader;
    };

    GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source);

    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);

    glDeleteShader(vs);
    glDeleteShader(fs);

    glUseProgram(program_);
    glUniform1i(glGetUniformLocation(program_, "uTexture"), 0);
    glUniform1i(glGetUniformLocation(program_, "uFontTexture"), 1);
}

void GpuOverlay::setup_textures() {
    glGenTextures(1, &texture_id_);
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_, height_, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &font_texture_id_);
    glBindTexture(GL_TEXTURE_2D, font_texture_id_);
    
    // Create 128x8 font texture
    std::vector<uint8_t> font_data(128 * 8 * 8, 0);
    for (int c = 0; c < 128; ++c) {
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                if (font8x8_basic[c][y] & (1 << x)) {
                    font_data[(y * 128 * 8) + (c * 8) + x] = 255;
                }
            }
        }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 128 * 8, 8, 0, GL_RED, GL_UNSIGNED_BYTE, font_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

void GpuOverlay::setup_geometry() {
    glGenBuffers(1, &vbo_);
}

struct Vertex {
    float x, y;
    float u, v;
    float r, g, b, a;
    int type;
};

void GpuOverlay::render(uint8_t* bgr_data, const DetectionResults& detections, const OverlayBallisticPoint* ballistic_point, uint64_t frame_counter) {
    glViewport(0, 0, width_, height_);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program_);

    // Update background texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, bgr_data);

    // Draw background quad
    std::vector<Vertex> vertices = {
        {-1, 1, 0, 1, 1, 1, 1, 1, 0},
        {-1, -1, 0, 0, 1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 0},
        {1, -1, 1, 0, 1, 1, 1, 1, 0}
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STREAM_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, x));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, u));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, r));
    glEnableVertexAttribArray(3);
    glVertexAttribIPointer(3, 1, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, type));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Render Overlays
    // ASPECT RATIO AUTHORITY: TPU is center-crop of 16:9
    const float sensor_ar = 16.0f / 9.0f;
    const float tpu_ar = 1.0f / 1.0f;
    const float tpu_width_in_sensor = tpu_ar / sensor_ar;
    const float tpu_x_offset_in_sensor = (1.0f - tpu_width_in_sensor) / 2.0f;

    for (const auto& d : detections) {
        float x_min_norm = tpu_x_offset_in_sensor + d.xmin * tpu_width_in_sensor;
        float x_max_norm = tpu_x_offset_in_sensor + d.xmax * tpu_width_in_sensor;
        float y_min_norm = d.ymin;
        float y_max_norm = d.ymax;

        // Convert normalized [0,1] to GL [-1,1]
        float gl_x1 = x_min_norm * 2.0f - 1.0f;
        float gl_y1 = 1.0f - y_min_norm * 2.0f;
        float gl_x2 = x_max_norm * 2.0f - 1.0f;
        float gl_y2 = 1.0f - y_max_norm * 2.0f;

        draw_rect(gl_x1, gl_y1, gl_x2, gl_y2, 1, 0, 0, 2.0f);
        
        std::string label = "ID:" + std::to_string(d.class_id) + " " + std::to_string((int)(d.score * 100)) + "%" ;
        draw_text(label, gl_x1, gl_y1 + 0.05f, 0.04f, 1, 0, 0);
    }

    // Crosshair
    draw_line(-0.05f, 0, 0.05f, 0, 1, 1, 1, 2.0f);
    draw_line(0, -0.05f, 0, 0.05f, 1, 1, 1, 2.0f);

    draw_text("OVERLAY PATH EXECUTED (GPU)", -0.95f, 0.9f, 0.05f, 1, 1, 1);
    draw_text("Frame: " + std::to_string(frame_counter), -0.95f, 0.8f, 0.04f, 1, 1, 1);

    if (ballistic_point && ballistic_point->is_valid && ballistic_point->frame_id == (int)frame_counter) {
        float b_x = (ballistic_point->impact_px_x / width_) * 2.0f - 1.0f;
        float b_y = 1.0f - (ballistic_point->impact_px_y / height_) * 2.0f;
        float r_x = (ballistic_point->safety_cone_radius_px / width_) * 2.0f;
        
        float ix1 = ballistic_point->inner_xmin * 2.0f - 1.0f;
        float iy1 = 1.0f - ballistic_point->inner_ymin * 2.0f;
        float ix2 = ballistic_point->inner_xmax * 2.0f - 1.0f;
        float iy2 = 1.0f - ballistic_point->inner_ymax * 2.0f;

        draw_rect(ix1, iy1, ix2, iy2, 1, 1, 0, 1.0f);
        
        float cone_r = ballistic_point->safety_cone_violation ? 1.0f : 0.0f;
        float cone_g = ballistic_point->safety_cone_violation ? 0.0f : 1.0f;
        draw_circle(b_x, b_y, r_x, cone_r, cone_g, 0, 2.0f);
        draw_marker(b_x, b_y, 1, 0, 0, 2.0f);

        char tel[128];
        snprintf(tel, sizeof(tel), "CONF: %.1f%% STREAK: %d", ballistic_point->confidence * 100.0f, ballistic_point->hit_streak);
        draw_text(tel, b_x + 0.02f, b_y + 0.02f, 0.035f, 1, 1, 1);
    }

    // Read back to CPU
    glReadPixels(0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, bgr_data);
    
    // NOTE: glReadPixels returns image with bottom row first. We might need to flip it or handle it in encoder.
    // Actually, cv::Mat uses top-left origin. So we need a vertical flip if we want it correct.
    // Optimization: Since we draw it flipped in GL, it might come back correctly?
    // Let's check: Background quad uses (-1,1) for (0,0) texcoord. 
    // In GLES (0,0) is bottom-left. So (0,0) texcoord in background quad is bottom-left of texture.
    // If our BGR data is top-left origin, and we upload it normally, GL (0,0) is bottom-left.
    // So the background quad should map (0,1) texcoord to (-1,1) GL pos.
    // Let's fix vertices to be top-down.
}

void GpuOverlay::draw_rect(float x1, float y1, float x2, float y2, float r, float g, float b, float thickness) {
    draw_line(x1, y1, x2, y1, r, g, b, thickness);
    draw_line(x2, y1, x2, y2, r, g, b, thickness);
    draw_line(x2, y2, x1, y2, r, g, b, thickness);
    draw_line(x1, y2, x1, y1, r, g, b, thickness);
}

void GpuOverlay::draw_line(float x1, float y1, float x2, float y2, float r, float g, float b, float) {
    std::vector<Vertex> vertices = {
        {x1, y1, 0, 0, r, g, b, 1, 1},
        {x2, y2, 0, 0, r, g, b, 1, 1}
    };
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_LINES, 0, 2);
}

void GpuOverlay::draw_text(const std::string& text, float x, float y, float size, float r, float g, float b) {
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, font_texture_id_);
    
    float aspect = (float)width_ / height_;
    float char_w = size;
    float char_h = size * aspect;

    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        float u_start = (float)c / 128.0f;
        float u_end = (float)(c + 1) / 128.0f;
        
        float cx = x + i * char_w;
        
        std::vector<Vertex> vertices = {
            {cx, y, u_start, 0, r, g, b, 1, 2},
            {cx, y - char_h, u_start, 1, r, g, b, 1, 2},
            {cx + char_w, y, u_end, 0, r, g, b, 1, 2},
            {cx + char_w, y - char_h, u_end, 1, r, g, b, 1, 2}
        };
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
}

void GpuOverlay::draw_circle(float cx, float cy, float radius, float r, float g, float b, float) {
    const int segments = 32;
    std::vector<Vertex> vertices;
    float aspect = (float)width_ / height_;
    for (int i = 0; i <= segments; ++i) {
        float theta = 2.0f * 3.14159f * (float)i / segments;
        vertices.push_back({cx + radius * std::cos(theta), cy + radius * std::sin(theta) * aspect, 0, 0, r, g, b, 1, 1});
    }
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_LINE_STRIP, 0, vertices.size());
}

void GpuOverlay::draw_marker(float x, float y, float r, float g, float b, float thickness) {
    float s = 0.03f;
    draw_line(x - s, y - s, x + s, y + s, r, g, b, thickness);
    draw_line(x - s, y + s, x + s, y - s, r, g, b, thickness);
}
