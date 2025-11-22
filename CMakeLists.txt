cmake_minimum_required(VERSION 3.16)
project(CoralEdgeTpuDetector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# --- Find System Dependencies ---
pkg_check_modules(JPEG REQUIRED libjpeg)
message(STATUS "Found libjpeg: ${JPEG_LIBRARIES}")

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
    src/mjpeg_capture.cpp # Added new module
    src/inference.cpp
    src/mjpeg_server.cpp
    src/udp_sender.cpp
    src/jpeg_wrapper.cpp
    src/util_logging.cpp
)

# --- Link Libraries and Include Directories ---
target_include_directories(detector PRIVATE
    ${TFLITE_INCLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIRS}
)

target_link_directories(detector PRIVATE ${TFLITE_LIB_DIR})

target_link_libraries(detector PRIVATE
    ${JPEG_LIBRARIES}
    tensorflowlite
    edgetpu
    Threads::Threads
)

target_link_options(detector PRIVATE "-Wl,-rpath,'$ORIGIN/../lib'")

message(STATUS "Target 'detector' configured. Ready to build.")
