cmake_minimum_required(VERSION 3.16)
project(CoralEdgeTpuDetector CXX)

find_package(PkgConfig REQUIRED)

# --- Local Dependencies ---

# FlatBuffers
set(FLATBUFFERS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/include)
set(FLATBUFFERS_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/lib)
message(STATUS "Flatbuffers include dir: ${FLATBUFFERS_INCLUDE_DIR}")
message(STATUS "Flatbuffers lib dir: ${FLATBUFFERS_LIB_DIR}")

# TensorFlow Lite
set(TFLITE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TFLITE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
message(STATUS "TFLite include dir: ${TFLITE_INCLUDE_DIR}")
message(STATUS "TFLite lib dir: ${TFLITE_LIB_DIR}")

# --- C++ Standard and Compiler Flags ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release build optimization flags are typically handled by default CMake behavior.
# Avoid manually setting CMAKE_CXX_FLAGS to prevent parsing issues.

# Set RPATH for shared libraries to be found at runtime
set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# --- Application Target ---

# Find libcamera using pkg-config
pkg_check_modules(LIBCAMERA REQUIRED libcamera)
if (NOT LIBCAMERA_FOUND)
    message(FATAL_ERROR "libcamera not found!")
endif()

# Find libusb-1.0 using pkg-config (often a dependency for libcamera)
pkg_check_modules(LIBUSB1 REQUIRED libusb-1.0)
if (NOT LIBUSB1_FOUND)
    message(FATAL_ERROR "libusb-1.0 not found!")
endif()

add_executable(detector src/main.cpp src/camera_capture.cpp src/inference.cpp src/mjpeg_server.cpp src/udp_sender.cpp src/jpeg_wrapper.cpp src/util_logging.cpp src/output_processor.cpp)

# Apply compiler flags directly to the target to ensure they are used
target_compile_options(detector PRIVATE -O3 -std=c++17)

# Link FlatBuffers manually
target_include_directories(detector PRIVATE ${FLATBUFFERS_INCLUDE_DIR} ${LIBCAMERA_INCLUDE_DIRS})
target_link_directories(detector PRIVATE ${FLATBUFFERS_LIB_DIR})
# Link against libflatbuffers.a. CMake will infer the library name flatbuffers from libflatbuffers.a
target_link_libraries(detector PRIVATE flatbuffers ${LIBCAMERA_LIBRARIES} ${LIBUSB1_LIBRARIES})

# Link TensorFlow Lite
target_include_directories(detector PRIVATE
    ${TFLITE_INCLUDE_DIR}
    ${TFLITE_INCLUDE_DIR}/tensorflow/lite
    ${TFLITE_INCLUDE_DIR}/tensorflow/lite/c
    ${TFLITE_INCLUDE_DIR}/tensorflow/lite/core # For core headers if needed directly
)
target_link_directories(detector PRIVATE ${TFLITE_LIB_DIR})
# Link against libtensorflowlite.so. CMake will infer the library name tensorflowlite from libtensorflowlite.so
target_link_libraries(detector PRIVATE tensorflowlite edgetpu jpeg)

# Set RPATH for the executable itself. The  variable is resolved at runtime
# to the directory containing the executable. We want to link to ../lib.
target_link_options(detector PRIVATE
    "-Wl,-rpath,'$ORIGIN/../lib'"
)

message(STATUS "Detector executable will be built.")

