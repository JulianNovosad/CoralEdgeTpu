#ifndef JPEG_WRAPPER_H
#define JPEG_WRAPPER_H

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdio> // For FILE*

#include <setjmp.h> // For jmp_buf, setjmp, longjmp
#include <jpeglib.h> // Include libjpeg header

// Custom destination manager for libjpeg
struct JpegMemoryDestination {
    struct jpeg_destination_mgr pub; // public fields
    std::vector<uint8_t>* buffer;   // target buffer
    size_t buffer_size;             // current buffer size
    JOCTET* temp_buffer;            // temp buffer for libjpeg
};

// Libjpeg error handling (simplified)
struct CustomErrorMgr {
    struct jpeg_error_mgr pub; // public fields
    jmp_buf setjmp_buffer;     // for return to caller
};

class JpegCompressGuard {
public:
    JpegCompressGuard();
    ~JpegCompressGuard();

    std::vector<uint8_t> compress_image(
        const uint8_t* pixel_data,
        int width,
        int height,
        int quality,
        J_COLOR_SPACE color_space // e.g., JCS_RGB, JCS_GRAYSCALE
    );

private:
    jpeg_compress_struct cinfo_;
    CustomErrorMgr jerr_; // Use CustomErrorMgr here
};

#endif // JPEG_WRAPPER_H
