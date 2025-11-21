#include "jpeg_wrapper.h"
#include <iostream>
#include <cstring> // For memcpy
#include <csetjmp> // For setjmp/longjmp



METHODDEF(void)
my_error_exit(j_common_ptr cinfo) {
    // cinfo->err really points to a CustomErrorMgr struct, so coerce pointer
    CustomErrorMgr* myerr = (CustomErrorMgr*)cinfo->err;

    // Always display long message for troubleshooting.
    (*cinfo->err->output_message)(cinfo);

    // Return control to the setjmp point
    longjmp(myerr->setjmp_buffer, 1);
}

// Custom destination manager routines
METHODDEF(void)
init_destination(j_compress_ptr cinfo) {
    JpegMemoryDestination* dest = (JpegMemoryDestination*)cinfo->dest;
    dest->buffer->clear(); // Clear the buffer at start
    dest->pub.next_output_byte = dest->temp_buffer;
    dest->pub.free_in_buffer = dest->buffer_size;
}

METHODDEF(boolean)
empty_output_buffer(j_compress_ptr cinfo) {
    JpegMemoryDestination* dest = (JpegMemoryDestination*)cinfo->dest;
    // When the buffer is full, append it to the vector
    dest->buffer->insert(dest->buffer->end(), dest->temp_buffer, dest->temp_buffer + dest->buffer_size);
    dest->pub.next_output_byte = dest->temp_buffer;
    dest->pub.free_in_buffer = dest->buffer_size;
    return TRUE;
}

METHODDEF(void)
term_destination(j_compress_ptr cinfo) {
    JpegMemoryDestination* dest = (JpegMemoryDestination*)cinfo->dest;
    // Copy any remaining data from the temp buffer to the vector
    size_t datacount = dest->buffer_size - dest->pub.free_in_buffer;
    dest->buffer->insert(dest->buffer->end(), dest->temp_buffer, dest->temp_buffer + datacount);
}

GLOBAL(void)
jpeg_mem_dest(j_compress_ptr cinfo, std::vector<uint8_t>* buffer, size_t temp_buffer_size) {
    if (cinfo->dest == nullptr) {
        cinfo->dest = (struct jpeg_destination_mgr *)(*cinfo->mem->alloc_small)((j_common_ptr)cinfo, JPOOL_PERMANENT, sizeof(JpegMemoryDestination));
    }

    JpegMemoryDestination* dest = (JpegMemoryDestination*)cinfo->dest;
    dest->pub.init_destination = init_destination;
    dest->pub.empty_output_buffer = empty_output_buffer;
    dest->pub.term_destination = term_destination;
    dest->buffer = buffer;
    dest->buffer_size = temp_buffer_size;
    dest->temp_buffer = (JOCTET *)(*cinfo->mem->alloc_small)((j_common_ptr)cinfo, JPOOL_PERMANENT, temp_buffer_size);
}


JpegCompressGuard::JpegCompressGuard() {
    cinfo_.err = jpeg_std_error(&jerr_.pub);
    jerr_.pub.error_exit = my_error_exit;

    if (setjmp(jerr_.setjmp_buffer)) {
        // If we get here, the JPEG code has signaled an error.
        throw std::runtime_error("JPEG compression error during initialization.");
    }

    jpeg_create_compress(&cinfo_);
}

JpegCompressGuard::~JpegCompressGuard() {
    jpeg_destroy_compress(&cinfo_);
}

std::vector<uint8_t> JpegCompressGuard::compress_image(
    const uint8_t* pixel_data,
    int width,
    int height,
    int quality,
    J_COLOR_SPACE color_space
) {
    std::vector<uint8_t> compressed_data;

    if (setjmp(jerr_.setjmp_buffer)) {
        // If we get here, the JPEG code has signaled an error.
        throw std::runtime_error("JPEG compression error.");
    }

    jpeg_mem_dest(&cinfo_, &compressed_data, 4096); // 4KB temp buffer

    cinfo_.image_width = width;
    cinfo_.image_height = height;
    cinfo_.input_components = (color_space == JCS_RGB || color_space == JCS_YCbCr) ? 3 : 1;
    cinfo_.in_color_space = color_space;

    jpeg_set_defaults(&cinfo_);
    jpeg_set_quality(&cinfo_, quality, TRUE); // TRUE = limit to baseline-JPEG values

    jpeg_start_compress(&cinfo_, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo_.next_scanline < cinfo_.image_height) {
        row_pointer[0] = (JSAMPROW)(pixel_data + cinfo_.next_scanline * width * cinfo_.input_components);
        jpeg_write_scanlines(&cinfo_, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo_);

    return compressed_data;
}
