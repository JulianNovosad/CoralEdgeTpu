/**
 * @file jpeg_wrapper.h
 * @brief Provides declarations for a C++ wrapper around libjpeg-turbo for JPEG compression.
 *
 * This header defines the necessary structures and a class to facilitate
 * robust JPEG compression from raw pixel data into memory, integrating custom
 * error handling and a custom destination manager for `std::vector<uint8_t>` output.
 */

#ifndef JPEG_WRAPPER_H
#define JPEG_WRAPPER_H

#include <vector>
#include <memory>   // For std::unique_ptr
#include <string>
#include <stdexcept>
#include <cstdio> // For FILE* (used by libjpeg internally)

#include <setjmp.h>  // Required for jmp_buf, setjmp, longjmp for libjpeg error handling
#include <jpeglib.h> // Include libjpeg header for core types and functions

// --- Custom Destination Manager for libjpeg ---

/**
 * @brief Custom structure for managing libjpeg output to a `std::vector<uint8_t>`.
 *
 * This structure extends libjpeg's `jpeg_destination_mgr` to redirect compressed
 * JPEG data into a dynamically growing `std::vector<uint8_t>`, using an
 * intermediate temporary buffer.
 */
struct JpegMemoryDestination {
    struct jpeg_destination_mgr pub; ///< Public fields of the standard destination manager.
    std::vector<uint8_t>* buffer;   ///< Pointer to the `std::vector<uint8_t>` where compressed data will be stored.
    size_t buffer_size;             ///< Size of the temporary buffer.
    JOCTET* temp_buffer;            ///< Temporary buffer used by libjpeg for output.
};

// --- Custom Error Handling for libjpeg ---

/**
 * @brief Custom error manager structure for libjpeg.
 *
 * This structure embeds a `jmp_buf` to allow libjpeg's error handler to
 * perform a non-local jump back to a safe point in the calling C++ code,
 * effectively converting a C-style error into a C++ exception-like behavior.
 */
struct CustomErrorMgr {
    struct jpeg_error_mgr pub; ///< Public fields of the standard error manager.
    jmp_buf setjmp_buffer;     ///< Jump buffer for non-local jumps back to the caller.
};

/**
 * @brief A RAII (Resource Acquisition Is Initialization) wrapper for libjpeg-turbo compression.
 *
 * This class handles the initialization and cleanup of libjpeg's compression
 * objects (`jpeg_compress_struct`) and provides a convenient method for
 * compressing raw pixel data into JPEG format with custom error handling.
 */
class JpegCompressGuard {
public:
    /**
     * @brief Constructor for JpegCompressGuard.
     *
     * Initializes the libjpeg compression object and sets up a custom
     * error handler using `setjmp` for robust error recovery.
     *
     * @throws std::runtime_error if libjpeg initialization fails.
     */
    JpegCompressGuard();

    /**
     * @brief Destructor for JpegCompressGuard.
     *
     * Releases all resources associated with the libjpeg compression object
     * by calling `jpeg_destroy_compress`.
     */
    ~JpegCompressGuard();

    /**
     * @brief Compresses raw pixel data into JPEG format.
     *
     * Takes raw image data, its dimensions, a quality setting, and the input
     * color space, then compresses it into a JPEG byte stream. The compressed
     * data is returned as a `std::vector<uint8_t>`.
     *
     * @param pixel_data Pointer to the raw pixel data buffer.
     * @param width The width of the image in pixels.
     * @param height The height of the image in pixels.
     * @param quality The JPEG compression quality (0-100), where 100 is best quality.
     * @param color_space The color space of the input pixel data (e.g., JCS_RGB, JCS_GRAYSCALE).
     * @return A `std::vector<uint8_t>` containing the compressed JPEG data.
     * @throws std::runtime_error if a JPEG compression error occurs during the process.
     */
    std::vector<uint8_t> compress_image(
        const uint8_t* pixel_data,
        int width,
        int height,
        int quality,
        J_COLOR_SPACE color_space // e.g., JCS_RGB, JCS_GRAYSCALE
    );

private:
    struct jpeg_compress_struct cinfo_; ///< The libjpeg compression object.
    CustomErrorMgr jerr_;               ///< Custom error manager for libjpeg.
};

#endif // JPEG_WRAPPER_H