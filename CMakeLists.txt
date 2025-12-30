cmake_minimum_required(VERSION 3.16)
project(CoralEdgeTpuDetector CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_BUILD_TYPE Release)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(ZEROMQ REQUIRED libzmq)
message(STATUS "Found ZeroMQ: ${ZEROMQ_LIBRARIES}")

# --- Find System Dependencies ---
# Use pkg-config to find the local libcamera installation (which should be in /usr/local)
find_program(PKG_CONFIG_EXECUTABLE NAMES pkg-config REQUIRED)
set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/aarch64-linux-gnu/pkgconfig:$ENV{PKG_CONFIG_PATH}")

pkg_check_modules(LIBCAMERA REQUIRED libcamera)
message(STATUS "Found libcamera: ${LIBCAMERA_LIBRARIES}")

pkg_check_modules(JPEG REQUIRED libjpeg)
find_package(OpenCV REQUIRED)
message(STATUS "Found libjpeg: ${JPEG_LIBRARIES}")
message(STATUS "Found OpenCV: ${OpenCV_LIBS}")

# Attempt to find x264
pkg_check_modules(X264 QUIET x264)
if (X264_FOUND)
    message(STATUS "Found x264: ${X264_LIBRARIES}")
    include_directories(${X264_INCLUDE_DIRS})
else()
    message(WARNING "x264 not found. H.264 encoding via libx264 may not work.")
endif()

# Attempt to find GStreamer RTSP Server
pkg_check_modules(GSTREAMER_RTSP_SERVER QUIET gstreamer-rtsp-server-1.0 gstreamer-rtsp-1.0 gstreamer-1.0)
if (GSTREAMER_RTSP_SERVER_FOUND)
    message(STATUS "Found GStreamer RTSP Server: ${GSTREAMER_RTSP_SERVER_LIBRARIES}")
    include_directories(${GSTREAMER_RTSP_SERVER_INCLUDE_DIRS})
else()
    message(WARNING "GStreamer RTSP Server not found. RTSP streaming may not work.")
endif()

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
    src/application.cpp
    src/camera_capture.cpp
    src/inference.cpp
    src/image_processor.cpp
    src/util_logging.cpp
    src/config_loader.cpp
    src/logic.cpp
    src/pca9685_controller.cpp
    src/monitor.cpp
    src/global_definitions.cpp
    src/h264_encoder.cpp
    src/orientation_sensor.cpp
    src/application_supervisor.cpp
    src/system_monitor.cpp
    src/keyboard_monitor.cpp
    src/udp_streamer.cpp
)

target_link_directories(detector PRIVATE ${TFLITE_LIB_DIR})

add_definitions(-DOPENSSL_API_3_0)
option(DEBUG_MODE "Enable debug logging" OFF)
if(DEBUG_MODE)
    add_definitions(-DDEBUG_MODE)
endif()

target_link_libraries(detector PRIVATE
    ${OpenCV_LIBS}
    jpeg
    tensorflowlite
    edgetpu
    /usr/local/lib/aarch64-linux-gnu/libcamera.so
    /usr/local/lib/aarch64-linux-gnu/libcamera-base.so
    Threads::Threads
    ${ZEROMQ_LIBRARIES}
    /lib/aarch64-linux-gnu/libssl.so.3
    /lib/aarch64-linux-gnu/libcrypto.so.3
    m
)

if (X264_FOUND)
    target_link_libraries(detector PRIVATE ${X264_LIBRARIES})
endif()

if (GSTREAMER_RTSP_SERVER_FOUND)
    target_link_libraries(detector PRIVATE ${GSTREAMER_RTSP_SERVER_LIBRARIES})
endif()

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
if (GSTREAMER_FOUND)
    target_link_libraries(detector PRIVATE ${GSTREAMER_LIBRARIES})
    target_include_directories(detector PRIVATE ${GSTREAMER_INCLUDE_DIRS})
endif()

pkg_check_modules(GSTREAMER_APP QUIET gstreamer-app-1.0)
if (GSTREAMER_APP_FOUND)
    target_link_libraries(detector PRIVATE ${GSTREAMER_APP_LIBRARIES})
    target_include_directories(detector PRIVATE ${GSTREAMER_APP_INCLUDE_DIRS})
endif()

pkg_check_modules(GSTREAMER_VIDEO QUIET gstreamer-video-1.0)
if (GSTREAMER_VIDEO_FOUND)
    target_link_libraries(detector PRIVATE ${GSTREAMER_VIDEO_LIBRARIES})
    target_include_directories(detector PRIVATE ${GSTREAMER_VIDEO_INCLUDE_DIRS})
endif()

pkg_check_modules(GSTREAMER_BASE QUIET gstreamer-base-1.0)
if (GSTREAMER_BASE_FOUND)
    target_link_libraries(detector PRIVATE ${GSTREAMER_BASE_LIBRARIES})
    target_include_directories(detector PRIVATE ${GSTREAMER_BASE_INCLUDE_DIRS})
endif()

target_include_directories(detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${JPEG_INCLUDE_DIRS}
    ${ZEROMQ_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}
    ${GSTREAMER_RTSP_SERVER_INCLUDE_DIRS}
    civetweb/include
    src
    SYSTEM ${TFLITE_INCLUDE_DIR}
    SYSTEM ${FLATBUFFERS_INCLUDE_DIR}
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_2.5.0
)

target_compile_options(detector PRIVATE
    -Wno-error
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-parameter>
)

message(STATUS "Target 'detector' configured. Ready to build.")