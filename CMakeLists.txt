cmake_minimum_required(VERSION 3.10)
project(CoralEdgeTpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0 gstreamer-rtsp-server-1.0)

# libcamera
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# ZeroMQ
pkg_check_modules(ZMQ REQUIRED libzmq cppzmq)

# X264
pkg_check_modules(X264 REQUIRED x264)

# GLESv2 and EGL
pkg_check_modules(GLESV2 REQUIRED glesv2)
pkg_check_modules(EGL REQUIRED egl)
pkg_check_modules(GBM REQUIRED gbm)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/include
    ${OpenCV_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIRS}
    ${X264_INCLUDE_DIRS}
    ${GLESV2_INCLUDE_DIRS}
    ${EGL_INCLUDE_DIRS}
    ${GBM_INCLUDE_DIRS}
)

# Link directories
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Source files
set(SOURCE_FILES
    src/main.cpp
    src/application.cpp
    src/application_supervisor.cpp
    src/camera_capture.cpp
    src/image_processor.cpp
    src/gpu_overlay.cpp
    src/inference.cpp
    src/logic.cpp
    src/monitor.cpp
    src/system_monitor.cpp
    src/util_logging.cpp
    src/config_loader.cpp
    src/h264_encoder.cpp
    src/mpegts_server.cpp
    src/orientation_sensor.cpp
    src/pca9685_controller.cpp
    src/rtsp_server.cpp
    src/keyboard_monitor.cpp
    src/discovery_module.cpp
    src/global_definitions.cpp
)

add_executable(detector ${SOURCE_FILES})

add_executable(control_listener src/control_listener.cpp)
target_link_libraries(control_listener Threads::Threads)

target_link_libraries(detector
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    ${LIBCAMERA_LIBRARIES}
    ${ZMQ_LIBRARIES}
    ${X264_LIBRARIES}
    ${GLESV2_LIBRARIES}
    ${EGL_LIBRARIES}
    ${GBM_LIBRARIES}
    Threads::Threads
    atomic
)

# TFLite and Edge TPU
target_link_libraries(detector
    tensorflowlite
    edgetpu
)

install(TARGETS detector DESTINATION bin)
