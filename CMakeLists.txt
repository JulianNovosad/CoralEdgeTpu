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
find_package(GTest REQUIRED)

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
pkg_check_modules(GSTREAMER_RTSP_SERVER QUIET gstreamer-rtsp-server-1.0 gstreamer-1.0)
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
    src/image_processor.cpp # New source file
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
    src/http_streamer.cpp
    src/rtsp_server.cpp
    civetweb/src/civetweb.c
    civetweb_temp/src/CivetServer.cpp
    src/civetweb_wrapper.cpp
)
add_executable(camera_isolation_test
    src/camera_isolation_test.cpp
)

add_executable(inference_test_no_logging
    src/inference_test_no_logging.cpp
    src/inference.cpp
    src/config_loader.cpp
    src/util_logging.cpp
)

add_executable(raw_tpu_test
    src/raw_tpu_test.cpp
    src/inference.cpp
    src/config_loader.cpp
    src/util_logging.cpp
)

add_executable(tpu_diagnostic
    tests/tpu_diagnostic.cpp
)

add_executable(list_tpu_devices
    tests/list_tpu_devices.cpp
)

add_executable(check_edgetpu_version
    tests/check_edgetpu_version.cpp
)
# Compile civetweb.c as C with proper defines for C++ wrapper
set_source_files_properties(civetweb/src/civetweb.c PROPERTIES LANGUAGE C COMPILE_DEFINITIONS "CIVETWEB_CGI_EXTENSIONS=\"\";USE_WEBSOCKET;NO_CACHING")

# Add C++ wrapper files to ensure C++ wrapper functions are compiled
set(CIVETWEB_CPP_WRAPPER 
    civetweb_temp/src/CivetServer.cpp
    src/civetweb_wrapper.cpp
)
add_library(civetweb_cpp_wrapper ${CIVETWEB_CPP_WRAPPER})
target_include_directories(civetweb_cpp_wrapper PRIVATE civetweb/include civetweb/src)
set_source_files_properties(${CIVETWEB_CPP_WRAPPER} PROPERTIES COMPILE_DEFINITIONS "USE_WEBSOCKET;NO_CACHING")
target_compile_options(civetweb_cpp_wrapper PRIVATE -Wno-error=missing-field-initializers)

target_link_directories(detector PRIVATE ${TFLITE_LIB_DIR})

add_definitions(-DOPENSSL_API_3_0)
option(DEBUG_MODE "Enable debug logging" OFF)
if(DEBUG_MODE)
    add_definitions(-DDEBUG_MODE)
endif()
target_link_libraries(detector PRIVATE
    ${OpenCV_LIBS}
    # Explicitly link required libraries
    jpeg # For libjpeg-turbo functions (from JPEG_LIBRARIES)
    tensorflowlite # For TensorFlow Lite functions
    edgetpu # For Edge TPU delegate functions
    /usr/local/lib/aarch64-linux-gnu/libcamera.so
    /usr/local/lib/aarch64-linux-gnu/libcamera-base.so
    Threads::Threads # For pthreads
    ${ZEROMQ_LIBRARIES} # For ZeroMQ library
    /lib/aarch64-linux-gnu/libssl.so.3
    /lib/aarch64-linux-gnu/libcrypto.so.3
    m # Math library
    civetweb_cpp_wrapper
)

if (X264_FOUND)
    target_link_libraries(detector PRIVATE ${X264_LIBRARIES})
endif()

if (GSTREAMER_RTSP_SERVER_FOUND)
    target_link_libraries(detector PRIVATE ${GSTREAMER_RTSP_SERVER_LIBRARIES})
endif()

# Check for and link GStreamer app library (needed for gst_app_src_push_buffer)
pkg_check_modules(GSTREAMER_APP QUIET gstreamer-app-1.0)
if (GSTREAMER_APP_FOUND)
    message(STATUS "Found GStreamer App: ${GSTREAMER_APP_LIBRARIES}")
    target_link_libraries(detector PRIVATE ${GSTREAMER_APP_LIBRARIES})
    target_include_directories(detector PRIVATE ${GSTREAMER_APP_INCLUDE_DIRS})
endif()

target_link_libraries(camera_isolation_test PRIVATE
    /usr/local/lib/aarch64-linux-gnu/libcamera.so
    /usr/local/lib/aarch64-linux-gnu/libcamera-base.so
    Threads::Threads # For pthreads
)

# Add dashboard executable
add_executable(dashboard
    tests/dashboard_fixed.cpp
)

target_link_libraries(dashboard PRIVATE
    Threads::Threads
)

# Add integrated entry point
add_executable(integrated_system
    src/integrated_main.cpp
)

target_link_libraries(integrated_system PRIVATE
    Threads::Threads
)

target_link_libraries(inference_test_no_logging PRIVATE
    ${OpenCV_LIBS}
    ${TFLITE_LIB_DIR}/libtensorflowlite.so # For TensorFlow Lite functions
    edgetpu # For Edge TPU delegate functions
    /usr/local/lib/aarch64-linux-gnu/libcamera.so
    /usr/local/lib/aarch64-linux-gnu/libcamera-base.so
    Threads::Threads # For pthreads
    ${ZEROMQ_LIBRARIES} # For ZeroMQ library
)

target_link_libraries(raw_tpu_test PRIVATE
    ${OpenCV_LIBS}
    ${TFLITE_LIB_DIR}/libtensorflowlite.so # For TensorFlow Lite functions
    edgetpu # For Edge TPU delegate functions
    /usr/local/lib/aarch64-linux-gnu/libcamera.so
    /usr/local/lib/aarch64-linux-gnu/libcamera-base.so
    Threads::Threads # For pthreads
    ${ZEROMQ_LIBRARIES} # For ZeroMQ library
)

target_link_libraries(tpu_diagnostic PRIVATE
    edgetpu # For Edge TPU delegate functions
)

target_link_libraries(list_tpu_devices PRIVATE
    edgetpu # For Edge TPU delegate functions
)

target_include_directories(list_tpu_devices PRIVATE
    SYSTEM ${TFLITE_INCLUDE_DIR}
)

target_link_libraries(check_edgetpu_version PRIVATE
    edgetpu # For Edge TPU delegate functions
)

target_include_directories(check_edgetpu_version PRIVATE
    SYSTEM ${TFLITE_INCLUDE_DIR}
)

target_include_directories(tpu_diagnostic PRIVATE
    SYSTEM ${TFLITE_INCLUDE_DIR}
)

target_include_directories(camera_isolation_test PRIVATE
    ${LIBCAMERA_INCLUDE_DIRS} # Use pkg-config defined include directories for libcamera
)

target_include_directories(inference_test_no_logging PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${TFLITE_INCLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIRS}
    ${ZEROMQ_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS} # Use pkg-config defined include directories for libcamera
    /usr/include/liveMedia
    /usr/include/BasicUsageEnvironment
    /usr/include/UsageEnvironment
    /usr/include/groupsock
    civetweb/include
    src # Include the src directory for h264_encoder.h, etc.
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_2.5.0
)

target_compile_options(inference_test_no_logging PRIVATE
    -Wno-error
)

target_include_directories(raw_tpu_test PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${TFLITE_INCLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIRS}
    ${ZEROMQ_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS} # Use pkg-config defined include directories for libcamera
    /usr/include/liveMedia
    /usr/include/BasicUsageEnvironment
    /usr/include/UsageEnvironment
    /usr/include/groupsock
    civetweb/include
    src # Include the src directory for h264_encoder.h, etc.
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_2.5.0
)

target_compile_options(raw_tpu_test PRIVATE
    -Wno-error
)

message(STATUS "Target 'detector' configured. Ready to build.")
target_include_directories(detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${JPEG_INCLUDE_DIRS}
    ${ZEROMQ_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS} # Use pkg-config defined include directories for libcamera
    ${GSTREAMER_RTSP_SERVER_INCLUDE_DIRS}
    civetweb/include
    src # Include the src directory for h264_encoder.h, etc.
    SYSTEM ${TFLITE_INCLUDE_DIR}
    SYSTEM ${FLATBUFFERS_INCLUDE_DIR}
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_2.5.0
)

target_compile_options(detector PRIVATE
    -Wno-error
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-parameter>
)


add_executable(config_loader_test tests/config_loader_test.cpp src/config_loader.cpp src/util_logging.cpp)
target_link_libraries(config_loader_test PRIVATE
    GTest::gtest_main
    GTest::gtest
)
target_include_directories(config_loader_test PRIVATE
    src
    civetweb/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBCAMERA_INCLUDE_DIRS} # Use pkg-config defined include directories for libcamera
)

add_executable(servo_test
    src/servo_test.cpp
    src/pca9685_controller.cpp
    src/util_logging.cpp
)
target_link_libraries(servo_test PRIVATE
    Threads::Threads
)
target_include_directories(servo_test PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(dlopen_test tests/dlopen_test.cpp)
target_link_libraries(dlopen_test PRIVATE dl)

# Add TPU performance test executable
add_executable(tpu_performance_test
    tests/tpu_performance_test.cpp
)
target_link_libraries(tpu_performance_test PRIVATE
    ${TFLITE_LIB_DIR}/libtensorflowlite.so
    edgetpu
    Threads::Threads
)
target_include_directories(tpu_performance_test PRIVATE
    SYSTEM ${TFLITE_INCLUDE_DIR}
    SYSTEM ${FLATBUFFERS_INCLUDE_DIR}
)
# Disable strict warnings for this utility
target_compile_options(tpu_performance_test PRIVATE
    -Wno-error
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
)

# Test executables (commented out as source files are missing)
# add_executable(test_config test_config.cpp src/config_loader.cpp src/util_logging.cpp)
# target_link_libraries(test_config PRIVATE)
# target_include_directories(test_config PRIVATE
#     src
#     include
#     /usr/include/libcamera
# )

# add_executable(minimal_model_test minimal_model_test.cpp)
# target_link_libraries(minimal_model_test PRIVATE
#     ${TFLITE_LIB_DIR}/libtensorflowlite.so
#     edgetpu
#     Threads::Threads
# )
# target_include_directories(minimal_model_test PRIVATE
#     include
#     ${TFLITE_INCLUDE_DIR}
#     ${FLATBUFFERS_INCLUDE_DIR}
#     /usr/include
# )
# Disable strict warnings for this utility
# target_compile_options(minimal_model_test PRIVATE
#     -Wno-error
# )

# add_executable(inspect_model_tensors inspect_model_tensors.cpp)
# target_link_libraries(inspect_model_tensors PRIVATE
#     ${TFLITE_LIB_DIR}/libtensorflowlite.so
# )
# target_include_directories(inspect_model_tensors PRIVATE
#     ${TFLITE_INCLUDE_DIR}
#     ${FLATBUFFERS_INCLUDE_DIR}
# )
# Disable strict warnings for this utility
# target_compile_options(inspect_model_tensors PRIVATE
#     -Wno-error
# )

# add_executable(test_tensor_access test_tensor_access.cpp)
# target_link_libraries(test_tensor_access PRIVATE
#     ${TFLITE_LIB_DIR}/libtensorflowlite.so
#     edgetpu
#     Threads::Threads
#     ${OpenCV_LIBS}
# )
# target_include_directories(test_tensor_access PRIVATE
#     include
#     ${TFLITE_INCLUDE_DIR}
#     ${FLATBUFFERS_INCLUDE_DIR}
#     /usr/include
#     ${OpenCV_INCLUDE_DIRS}
# )
# Disable strict warnings for this utility
# target_compile_options(test_tensor_access PRIVATE
#     -Wno-error
# )

# add_executable(test_ballistics_fix test_ballistics_fix.cpp src/logic.cpp src/config_loader.cpp src/util_logging.cpp)
# target_link_libraries(test_ballistics_fix PRIVATE
#     Threads::Threads
# )
# target_include_directories(test_ballistics_fix PRIVATE
#     src
#     include
#     ${TFLITE_INCLUDE_DIR}
#     ${FLATBUFFERS_INCLUDE_DIR}
#     /usr/include/libcamera
# )
# Disable strict warnings for this utility
# target_compile_options(test_ballistics_fix PRIVATE
#     -Wno-error
# )

# add_executable(test_ballistics_simple test_ballistics_simple.cpp)
# target_link_libraries(test_ballistics_simple PRIVATE
#     Threads::Threads
# )
# Disable strict warnings for this utility
# target_compile_options(test_ballistics_simple PRIVATE
#     -Wno-error
# )

# add_executable(test_ballistics_actual test_ballistics_actual.cpp src/logic.cpp src/config_loader.cpp src/util_logging.cpp)
# target_link_libraries(test_ballistics_actual PRIVATE
#     Threads::Threads
# )
# target_include_directories(test_ballistics_actual PRIVATE
#     src
#     include
#     ${TFLITE_INCLUDE_DIR}
#     ${FLATBUFFERS_INCLUDE_DIR}
#     /usr/include/libcamera
# )
# Disable strict warnings for this utility
# target_compile_options(test_ballistics_actual PRIVATE
#     -Wno-error
# )

# add_executable(test_ballistics_fixed test_ballistics_fixed.cpp)
# target_link_libraries(test_ballistics_fixed PRIVATE
#     Threads::Threads
# )
# Disable strict warnings for this utility
# target_compile_options(test_ballistics_fixed PRIVATE
#     -Wno-error
# )