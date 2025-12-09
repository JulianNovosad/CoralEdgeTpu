cmake_minimum_required(VERSION 3.16)
project(CoralEdgeTpuDetector CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_BUILD_TYPE Debug)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(ZEROMQ REQUIRED libzmq)
message(STATUS "Found ZeroMQ: ${ZEROMQ_LIBRARIES}")
find_package(GTest REQUIRED)

pkg_check_modules(LIBCAMERA REQUIRED libcamera)
message(STATUS "Found libcamera: ${LIBCAMERA_LIBRARIES}")

# --- Find System Dependencies ---
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
    src/util_logging.cpp
    src/config_loader.cpp # Added ConfigLoader source file
    src/logic.cpp

    src/h264_encoder.cpp
    src/orientation_sensor.cpp
    src/application_supervisor.cpp
    src/system_monitor.cpp
    civetweb/src/civetweb.c
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
    camera # For libcamera core functions (from LIBCAMERA_LIBRARIES)
    camera-base # For libcamera base functions (from LIBCAMERA_LIBRARIES)
    Threads::Threads # For pthreads
    ${ZEROMQ_LIBRARIES} # For ZeroMQ library
    /lib/aarch64-linux-gnu/libssl.so.3
    /lib/aarch64-linux-gnu/libcrypto.so.3
)

if (X264_FOUND)
    target_link_libraries(detector PRIVATE ${X264_LIBRARIES})
endif()

target_link_options(detector PRIVATE "-Wl,-rpath,'$ORIGIN/../lib'")

message(STATUS "Target 'detector' configured. Ready to build.")
target_include_directories(detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${TFLITE_INCLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIRS}
    ${ZEROMQ_INCLUDE_DIRS}
    /usr/include/libcamera # Explicitly add /usr/include/libcamera for libcamera headers
    civetweb/include
    src # Include the src directory for h264_encoder.h, etc.
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_2.5.0
)

target_compile_options(detector PRIVATE
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
)

add_executable(dlopen_test tests/dlopen_test.cpp)
target_link_libraries(dlopen_test PRIVATE dl)

