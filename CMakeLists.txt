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

# Attempt to find Live555
pkg_check_modules(LIVE555 QUIET live555)
if (LIVE555_FOUND)
    message(STATUS "Found Live555: ${LIVE555_LIBRARIES}")
    include_directories(${LIVE555_INCLUDE_DIRS})
    # Add specific Live555 include directory
    include_directories(/usr/include/liveMedia /usr/include/BasicUsageEnvironment /usr/include/UsageEnvironment /usr/include/groupsock)
else()
    message(WARNING "Live555 not found. RTSP streaming may not work.")
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

    src/h264_encoder.cpp
    src/orientation_sensor.cpp
    src/application_supervisor.cpp
    src/system_monitor.cpp
    src/keyboard_monitor.cpp
    src/rtsp_server.cpp
    civetweb/src/civetweb.c
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
    tpu_diagnostic.cpp
)

add_executable(list_tpu_devices
    list_tpu_devices.cpp
)

add_executable(check_edgetpu_version
    check_edgetpu_version.cpp
)
set_source_files_properties(civetweb/src/civetweb.c PROPERTIES LANGUAGE C COMPILE_DEFINITIONS USE_WEBSOCKET)

target_link_directories(detector PRIVATE ${TFLITE_LIB_DIR})

add_definitions(-DOPENSSL_API_3_0)
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
)

if (X264_FOUND)
    target_link_libraries(detector PRIVATE ${X264_LIBRARIES})
endif()

if (LIVE555_FOUND)
    target_link_libraries(detector PRIVATE ${LIVE555_LIBRARIES})
endif()

target_link_libraries(camera_isolation_test PRIVATE
    /usr/local/lib/aarch64-linux-gnu/libcamera.so
    /usr/local/lib/aarch64-linux-gnu/libcamera-base.so
    Threads::Threads # For pthreads
)

# Add dashboard executable
add_executable(dashboard
    dashboard_fixed.cpp
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
    /usr/include/liveMedia
    /usr/include/BasicUsageEnvironment
    /usr/include/UsageEnvironment
    /usr/include/groupsock
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
    tpu_performance_test.cpp
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