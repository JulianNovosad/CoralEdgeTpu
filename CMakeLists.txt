cmake_minimum_required(VERSION 3.16)
project(CoralEdgeTpuDetector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(LIBCAMERA REQUIRED libcamera)
message(STATUS "Found libcamera: ${LIBCAMERA_LIBRARIES}")

# --- Find System Dependencies ---
pkg_check_modules(JPEG REQUIRED libjpeg)
find_package(OpenCV REQUIRED)
message(STATUS "Found libjpeg: ${JPEG_LIBRARIES}")
message(STATUS "Found OpenCV: ${OpenCV_LIBS}")

# --- Find Local/Custom Dependencies ---
set(TFLITE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TFLITE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(FLATBUFFERS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/include)

if(NOT EXISTS ${TFLITE_INCLUDE_DIR})
    message(FATAL_ERROR "TensorFlow Lite include directory not found at ${TFLITE_INCLUDE_DIR}")
endif()
if(NOT EXISTS ${FLATBUFFERS_INCLUDE_DIR})
    message(FATAL_ERROR "Flatbuffers include directory not found at ${FLATBUFFERS_INCLUDE_DIR}")
endif()

# --- Set up Executable Target ---
add_executable(detector
    src/main.cpp
    src/camera_capture.cpp
    src/inference.cpp
    src/mjpeg_server.cpp
    src/udp_sender.cpp
    src/udp_video_sender.cpp
    src/video_overlay_processor.cpp
    src/jpeg_wrapper.cpp
    src/util_logging.cpp
    src/config_loader.cpp # Added ConfigLoader source file
)

target_link_directories(detector PRIVATE ${TFLITE_LIB_DIR})

target_link_libraries(detector PRIVATE
    ${OpenCV_LIBS}
    # Explicitly link required libraries
    jpeg # For libjpeg-turbo functions (from JPEG_LIBRARIES)
    tensorflowlite # For TensorFlow Lite functions
    edgetpu # For Edge TPU delegate functions
    camera # For libcamera core functions (from LIBCAMERA_LIBRARIES)
    camera-base # For libcamera base functions (from LIBCAMERA_LIBRARIES)
    Threads::Threads # For pthreads
)

target_link_options(detector PRIVATE "-Wl,-rpath,'$ORIGIN/../lib'")

message(STATUS "Target 'detector' configured. Ready to build.")
target_include_directories(detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TFLITE_INCLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIRS}
    /usr/include/libcamera # Explicitly add /usr/include/libcamera for libcamera headers
)