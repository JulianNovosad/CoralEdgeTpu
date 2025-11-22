/**
 * @file jpeg_wrapper.cpp
 * @brief Implements a C++ wrapper for libjpeg-turbo for JPEG compression.
 *
 * This file provides a utility class `JpegCompressGuard` to simplify the
 * process of compressing raw pixel data into JPEG format using the libjpeg-turbo
 * library. It includes custom error handling via `setjmp`/`longjmp` and
 * a custom destination manager for writing compressed JPEG data directly into
 * a `std::vector<uint8_t>`.
 */

#include "jpeg_wrapper.h"
#include "util_logging.h" // For LOG_ERROR
#include <iostream>
#include <cstring> // For memcpy
#include <csetjmp> // For setjmp/longjmp

// --- Custom Error Handling for libjpeg ---

/**
 * @brief Custom error exit handler for libjpeg.
 *
 * This function replaces libjpeg's default error handler. Instead of exiting
 * the program, it uses `longjmp` to return control to a predefined point
 * (`setjmp_buffer`) in the calling function, allowing for C++ exception-like
 * error recovery within the libjpeg context.
 *
 * @param cinfo Pointer to the common JPEG object, which contains error manager info.
 */
METHODDEF(void)
my_error_exit(j_common_ptr cinfo) {
    // cinfo->err really points to a CustomErrorMgr struct, so coerce pointer
    CustomErrorMgr* myerr = (CustomErrorMgr*)cinfo->err;

    // Always display long message for troubleshooting.
    (*cinfo->err->output_message)(cinfo);

    // Return control to the setjmp point
    longjmp(myerr->setjmp_buffer, 1);
}

// --- Custom Destination Manager for libjpeg ---

/**
 * @brief Initializes the custom JPEG destination manager.
 *
 * Called by libjpeg to prepare the output buffer for compressed data.
 * Clears the destination `std::vector<uint8_t>` and sets up the temporary buffer.
 *
 * @param cinfo Pointer to the JPEG compression object.
 */
METHODDEF(void)
init_destination(j_compress_ptr cinfo) {
    JpegMemoryDestination* dest = (JpegMemoryDestination*)cinfo->dest;
    dest->buffer->clear(); // Clear the output vector at the start of compression
    dest->pub.next_output_byte = dest->temp_buffer; // Set pointer to temporary buffer
    dest->pub.free_in_buffer = dest->buffer_size; // Set available space in temporary buffer
}

/**
 * @brief Handles a full output buffer in the custom JPEG destination manager.
 *
 * Called by libjpeg when the temporary output buffer is full. It appends
 * the contents of the temporary buffer to the main `std::vector<uint8_t>`
 * and resets the temporary buffer for further writing.
 *
 * @param cinfo Pointer to the JPEG compression object.
 * @return Always TRUE to indicate success.
 */
METHODDEF(boolean)
empty_output_buffer(j_compress_ptr cinfo) {
    JpegMemoryDestination* dest = (JpegMemoryDestination*)cinfo->dest;
    // When the buffer is full, append its content to the main vector
    dest->buffer->insert(dest->buffer->end(), dest->temp_buffer, dest->temp_buffer + dest->buffer_size);
    dest->pub.next_output_byte = dest->temp_buffer; // Reset pointer to temporary buffer
    dest->pub.free_in_buffer = dest->buffer_size; // Reset available space
    return TRUE;
}

/**
 * @brief Terminates the custom JPEG destination manager.
 *
 * Called by libjpeg at the end of compression. Copies any remaining data
 * from the temporary buffer to the main `std::vector<uint8_t>`.
 *
 * @param cinfo Pointer to the JPEG compression object.
 */
METHODDEF(void)
term_destination(j_compress_ptr cinfo) {
    JpegMemoryDestination* dest = (JpegMemoryDestination*)cinfo->dest;
    // Copy any remaining data from the temporary buffer to the vector
    size_t datacount = dest->buffer_size - dest->pub.free_in_buffer;
    if (datacount > 0) {
        dest->buffer->insert(dest->buffer->end(), dest->temp_buffer, dest->temp_buffer + datacount);
    }
}

/**
 * @brief Sets up a custom memory destination manager for libjpeg.
 *
 * This function initializes libjpeg's destination manager to write compressed
 * JPEG data into a `std::vector<uint8_t>` using a temporary buffer.
 *
 * @param cinfo Pointer to the JPEG compression object.
 * @param buffer Pointer to the `std::vector<uint8_t>` where the compressed data will be stored.
 * @param temp_buffer_size The size of the temporary buffer to use for output.
 */
GLOBAL(void)
jpeg_mem_dest(j_compress_ptr cinfo, std::vector<uint8_t>* buffer, size_t temp_buffer_size) {
    // Allocate JpegMemoryDestination structure if not already allocated
    if (cinfo->dest == nullptr) {
        cinfo->dest = (struct jpeg_destination_mgr *)(*cinfo->mem->alloc_small)((j_common_ptr)cinfo, JPOOL_PERMANENT, sizeof(JpegMemoryDestination));
    }

    JpegMemoryDestination* dest = (JpegMemoryDestination*)cinfo->dest;
    dest->pub.init_destination = init_destination;     // Set initialization function
    dest->pub.empty_output_buffer = empty_output_buffer; // Set function for full buffer handling
    dest->pub.term_destination = term_destination;     // Set termination function
    dest->buffer = buffer;                             // Store pointer to the user's vector
    dest->buffer_size = temp_buffer_size;              // Store temporary buffer size
    // Allocate the temporary buffer used by libjpeg
    dest->temp_buffer = (JOCTET *)(*cinfo->mem->alloc_small)((j_common_ptr)cinfo, JPOOL_PERMANENT, temp_buffer_size);
}


/**
 * @brief Constructor for JpegCompressGuard.
 *
 * Initializes the libjpeg compression object and sets up custom error handling.
 * Throws a `std::runtime_error` if an error occurs during libjpeg initialization.
 */
JpegCompressGuard::JpegCompressGuard() {
    // Set up the normal JPEG error routines, then override error_exit.
    cinfo_.err = jpeg_std_error(&jerr_.pub);
    jerr_.pub.error_exit = my_error_exit;

    // Establish the setjmp return point for error recovery.
    if (setjmp(jerr_.setjmp_buffer)) {
        // If we get here, the JPEG code has signaled an error.
        LOG_ERROR("JPEG compression error during initialization.");
        throw std::runtime_error("JPEG compression error during initialization.");
    }

    // Initialize the JPEG compression object.
    jpeg_create_compress(&cinfo_);
}

/**
 * @brief Destructor for JpegCompressGuard.
 *
 * Releases resources associated with the libjpeg compression object.
 */
JpegCompressGuard::~JpegCompressGuard() {
    jpeg_destroy_compress(&cinfo_);
}

/**
 * @brief Compresses raw pixel data into JPEG format.
 *
 * Takes raw pixel data, dimensions, quality, and color space, and compresses
 * it into a JPEG byte stream stored in a `std::vector<uint8_t>`.
 * Uses custom error handling to catch libjpeg errors.
 *
 * @param pixel_data Pointer to the raw pixel data (e.g., RGB or Grayscale).
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @param quality The JPEG compression quality (0-100).
 * @param color_space The color space of the input pixel data (e.g., JCS_RGB, JCS_GRAYSCALE).
 * @return A `std::vector<uint8_t>` containing the compressed JPEG data.
 * @throws std::runtime_error if a JPEG compression error occurs.
 */
std::vector<uint8_t> JpegCompressGuard::compress_image(
    const uint8_t* pixel_data,
    int width,
    int height,
    int quality,
    J_COLOR_SPACE color_space
) {
    std::vector<uint8_t> compressed_data;

    // Establish the setjmp return point for error recovery during compression.
    if (setjmp(jerr_.setjmp_buffer)) {
        // If we get here, the JPEG code has signaled an error.
        LOG_ERROR("JPEG compression error during image compression.");
        throw std::runtime_error("JPEG compression error.");
    }

    // Set up the custom memory destination manager to write into `compressed_data`.
    // A 4KB temporary buffer is used internally by libjpeg.
    jpeg_mem_dest(&cinfo_, &compressed_data, 4096);

    // Configure image parameters for libjpeg.
    cinfo_.image_width = width;
    cinfo_.image_height = height;
    cinfo_.input_components = (color_space == JCS_RGB || color_space == JCS_YCbCr) ? 3 : 1; // 3 for color, 1 for grayscale
    cinfo_.in_color_space = color_space;

    // Set default compression parameters and quality.
    jpeg_set_defaults(&cinfo_);
    jpeg_set_quality(&cinfo_, quality, TRUE); // TRUE = limit to baseline-JPEG values

    // Start the compression process.
    jpeg_start_compress(&cinfo_, TRUE);

    JSAMPROW row_pointer[1]; // Pointer to JSAMPLE row[s]
    // Write scanlines one by one.
    while (cinfo_.next_scanline < cinfo_.image_height) {
        row_pointer[0] = (JSAMPROW)(pixel_data + cinfo_.next_scanline * width * cinfo_.input_components);
        jpeg_write_scanlines(&cinfo_, row_pointer, 1);
    }

    // Finish the compression process.
    jpeg_finish_compress(&cinfo_);

    return compressed_data;
}