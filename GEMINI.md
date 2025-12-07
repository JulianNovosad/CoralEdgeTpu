# GEMINI Project: CoralEdgeTpu

## Project Overview

This repository contains a complete, modern, and stable C++-based inference stack for the Google Coral M.2 TPU, specifically designed for the Raspberry Pi 5. The project aims to provide a reproducible and high-performance solution for real-time object detection, replacing the often complex and poorly maintained default toolchain.

The core of the project is a multi-threaded C++ application that forms a complete video processing pipeline:

1.  **Camera Capture**: Captures video from a libcamera-compatible camera.
2.  **Inference**: Performs object detection using a TensorFlow Lite model accelerated by the Edge TPU.
3.  **Video Overlay**: Draws bounding boxes and labels on the video frames.
4.  **H.264 Encoding**: Encodes the processed video into H.264 format.
5.  **HTTP Streaming**: Streams the encoded video over HTTP for remote viewing.

The project is self-contained, including all necessary source code, patched libraries, headers, and build scripts to ensure a deterministic build process.

## Building and Running

The project uses CMake for building and provides a comprehensive shell script to automate the entire process, including dependency installation and library compilation.

### Prerequisites

*   Raspberry Pi 5
*   Google Coral M.2 TPU (PCIe)
*   `libedgetpu1-std`
*   CMake ≥ 3.16
*   g++ ≥ 10

### Build Command

To build the entire project from scratch, including all dependencies:

```bash
./build.sh
```

This script will:
1.  Install system dependencies (`libcamera-dev`, `libedgetpu1-std`).
2.  Download and build the correct versions of `flatbuffers` and `tensorflow`.
3.  Build the main `detector` executable.

### Running the Application

After a successful build, the main executable will be located at `build/detector`.

```bash
./build/detector
```

The application is configured via `config.json`.

## Development Conventions

*   **C++17**: The project is written in modern C++.
*   **Multi-threading**: The application is heavily multi-threaded, using queues (`ImageQueue`, `DetectionResultsQueue`, etc.) to pass data between different processing stages (camera, inference, overlay, encoding, streaming).
*   **Buffer Pooling**: The application uses buffer pools (`BufferPool`) to efficiently manage memory for large data like image frames.
*   **Logging**: A custom logger (`Logger`) is used for structured logging.
*   **Staged Development**: The `README.md` file contains a detailed "Stage-Gate Plan" which outlines a structured development and validation process. New development should follow this plan.
*   **Configuration**: Application settings are managed through a `config.json` file, loaded by the `ConfigLoader` class.

## Key Files and Directories

*   `src/main.cpp`: The main entry point of the application, responsible for initializing and orchestrating the pipeline.
*   `src/inference.cpp`: Contains the `InferenceEngine` class, which manages the Edge TPU and runs the TFLite model.
*   `src/camera_capture.cpp`: Manages video capture using `libcamera`.
*   `src/video_overlay_processor.cpp`: Handles drawing on video frames.
*   `src/http_server.cpp`: Implements the web server for streaming video.
*   `CMakeLists.txt`: Defines the project structure and build targets for CMake.
*   `build.sh`: The main build script that automates the entire build process.
*   `README.md`: Contains detailed documentation about the project, including hardware setup and development plans.
*   `config.json`: (Not present in the repo, but expected by the application) Configuration file for the application.

---

## Debugging Session Knowledge (December 8, 2025)

This session focused on implementing all Stage 0 and Stage 1 requirements from the `README.md` and debugging the resulting application.

### Stage-Gate Compliance Implementation:

*   **Stage 0: Ballistics Theory**:
    *   **Action**: Updated `docs/ballistics_theory.md` from a placeholder to include detailed theoretical frameworks for projectile motion, uncertainty sources (sensor noise, model inaccuracy), and safety criteria, providing a solid foundation for future implementation.

*   **Stage 0/1: Performance Metrics & Logging**:
    *   **Action**: Implemented performance metric collection (p50/p95/p99 latency, throughput) in `CameraCapture`, `InferenceEngine`, and `LogicModule`.
    *   **Action**: Modified the `Logger` to use `CLOCK_MONOTONIC_RAW` for all CSV log timestamps, fulfilling a strict real-time requirement.
    *   **Action**: Connected the `get_performance_metrics()` methods in all modules to the CSV logger.
    *   **Action**: Configured the logger to use the absolute path `/home/pi/CoralEdgeTpu/logs/`.

*   **Stage 1: Modularization and Refactoring**:
    *   **`LogicModule` Refactoring**: Refactored `LogicModule` to run in its own dedicated worker thread, consuming detection results from a dedicated lock-free queue (`boost::lockfree::spsc_queue`). This decouples it from the main thread and improves performance.
    *   **`IMUSensor` Module**: Created a new `IMUSensor` module (`src/imu_sensor.h/cpp`) to encapsulate IMU data reading (currently provides mock data, but is ready for hardware integration).
    *   **`ApplicationSupervisor` Module**: Created a global `ApplicationSupervisor` (`src/application_supervisor.h/cpp`) to handle graceful shutdown. It registers all pipeline modules and stops them in the correct order upon receiving a `SIGINT` or `SIGTERM` signal.
    *   **`SystemMonitor` Module**: Created a new `SystemMonitor` module (`src/system_monitor.h/cpp`) to periodically read CPU temperature and memory usage and log them to CSV.

*   **Stage 1: Baseline Algorithm Implementation**:
    *   **Object Tracking**: Replaced the simple centroid-based tracking in `LogicModule` with a more robust Intersection over Union (IoU) based association logic.
    *   **Ballistics Prediction**: Implemented a baseline 2D projectile motion model (considering gravity) in `predict_impact_point`.
    *   **Safety Checks**: Enhanced `perform_safety_and_uncertainty_checks` to use configurable thresholds for track stability (`hit_streak`) and uncertainty.

*   **Stage 1: Fallback Modes**:
    *   **Action**: Implemented a state machine in `LogicModule` to handle different `FallbackMode`s (`NORMAL_OPERATION`, `FALLBACK_A_REDUCED_PERFORMANCE`, `FALLBACK_B_WARNING_STATE`). The system now transitions between these states based on the output of the safety checks.

*   **Build System (`CMakeLists.txt`)**:
    *   **Action**: Updated `CMakeLists.txt` to include all newly created source files (`imu_sensor.cpp`, `application_supervisor.cpp`, `system_monitor.cpp`) and correctly link them into the final `detector` executable.

### Resolved Issues:

*   **Numerous Compilation Errors**: Fixed a wide range of compilation errors stemming from the introduction of new modules, queue changes (`boost::lockfree::spsc_queue`), and refactoring. This included fixing missing headers, incorrect function signatures, access specifier issues, and type mismatches.
*   **`std::map` Insertion of Non-Copyable Types**: Resolved a complex C++ issue in the `Logger` where `std::map::emplace` was failing with non-copyable `CsvLogger` objects (due to `std::ofstream` and `std::mutex` members). The fix involved switching to `std::map::try_emplace` (C++17), which constructs the object in-place.
*   **`x264` `memcpy` Crash (SPS/PPS Headers)**: Temporarily disabled the copying of SPS/PPS headers to isolate the recurring `SIGSEGV`.

### Unresolved/Current Issue (Segmentation Fault during Camera Startup):

*   **Persistent `Segmentation fault`**: Despite extensive debugging and fixes, the application still crashes with `SIGSEGV` (Exit Code 139) in `CameraCapture::request_complete_callback` at the `memcpy` line for the BGR image conversion.
*   **Diagnosis**: `gdb` backtrace points to `__memcpy_generic` called from line 516 of `src/camera_capture.cpp`. The debug logs added before the crash show that the `cv::cvtColor` call, which converts YUV to BGR, is likely producing an invalid `cv::Mat` (`bgr_image`) with a corrupted `dataend` pointer. This makes the subsequent `memcpy` to the pooled buffer unsafe. The root cause is likely an issue with the `cv::cvtColor` function itself or the data being passed to it, rather than the `memcpy` destination buffer.

### Next Steps:

*   **Focus on `cv::cvtColor`**: Investigate why `cv::cvtColor(yuv_image, bgr_image, cv::COLOR_YUV2BGR_I420)` is producing an invalid `cv::Mat`. This could be due to:
    *   An issue with the OpenCV build or version.
    *   A memory alignment problem.
    *   An underlying issue with the YUV data coming from `libcamera`.
*   **Isolate and Test**: Create a minimal test case outside of the main application that takes a saved YUV frame from `libcamera` and attempts the `cv::cvtColor` operation to see if the issue can be reproduced in isolation.

---
## Previous Debugging Session Knowledge (December 7, 2025)

This session focused on resolving a persistent `Segmentation fault` during application startup/runtime.

### Resolved Issues:

1.  **Memory Management (Double Free / `std::bad_weak_ptr`)**:
    *   **Problem**: Initial implementation of `BufferPool` and `PooledPtr` led to `double free or corruption` and `std::bad_weak_ptr` errors on shutdown.
    *   **Resolution**:
        *   Refactored `BufferPool` to use `std::shared_ptr` internally with `std::enable_shared_from_this` for robust lifetime management of pooled buffers.
        *   Updated `src/camera_capture.h/cpp`, `src/inference.h/cpp`, and `src/h264_encoder.h/cpp` to accept and store `std::shared_ptr<BufferPool<...>>` instead of references.
        *   Modified `src/main.cpp` to create all `BufferPool`s using `std::make_shared` and pass them as `std::shared_ptr` to component constructors.
        *   Implemented and called `clear()` methods on `ThreadSafeQueue`s during shutdown in `main.cpp` to ensure all `PooledPtr`s (which are `std::shared_ptr`s) are released before `BufferPool` destruction.

2.  **Camera Codec (`bgr888` / `YUV420`)**:
    *   **Problem**: `rpicam-vid` subprocess (initially used) exited with code 255 due to "unrecognised codec bgr888".
    *   **Resolution**: Changed `CameraCapture` to request `YUV420` from `libcamera` and perform an explicit `YUV2BGR_I420` conversion using OpenCV before passing to downstream components.

3.  **H264Encoder `memcpy` Crash (SPS/PPS Headers)**:
    *   **Problem**: `Segmentation fault` in `H264Encoder::worker_thread_func` during `memcpy` of SPS/PPS NAL units from `x264_encoder_headers`. Initial diagnosis suspected buffer overflow or `x264` multi-threading issues.
    *   **Resolution**:
        *   Increased `h264_pool` buffer size from 256KB to 1MB in `src/main.cpp`. (This did not resolve the issue, but ensured buffer capacity).
        *   Fixed incorrect `memcpy` loop logic in `H264Encoder::stop()` (flush loop) to correctly iterate through all NAL units (`nal[i].p_payload`, `nal[i].i_payload`). (This resolved a separate potential `memcpy` crash in shutdown, but not the current issue).
        *   Temporarily disabled `param.i_threads = X264_SYNC_LOOKAHEAD_AUTO` to `1`. (This did not resolve the issue, reverted).

### Unresolved/Current Issue (Compilation Errors in `src/camera_capture.cpp`):

The `AGRD-V2` branch's `src/camera_capture.cpp` file has repeatedly shown compilation errors related to bracing, variable declarations (`bgr_pooled_buffer`), and `buffer_pool.acquire()` syntax (`.acquire()` vs `->acquire()`). This indicates an inconsistent state in the file, making it difficult to apply targeted fixes. The current approach of `git checkout AGRD-V2 src/camera_capture.cpp src/camera_capture.h` is reliably pulling a version of the file that does not compile.

### Next Steps:
*   **Prioritize getting `src/camera_capture.cpp` to compile cleanly.** This is the immediate blocker.
*   **Strategy**: Overwrite the *entire* `src/camera_capture.cpp` with a version that has been manually reconstructed to include all known `std::shared_ptr<BufferPool>` fixes, correct `->acquire()` calls, proper bracing, and correct logic for dual streams and YUV420 processing. This ensures a consistent and compilable baseline.

---

### Previous Unresolved/Current Issue (Segmentation Fault during Camera Startup):

1.  **Persistent `Segmentation fault` during Camera Startup**:
    *   **Problem**: The application consistently crashes with a `Segmentation fault` (Exit Code: 139) immediately after `CameraCapture: Initial requests queued.` and before any application code within `CameraCapture::request_complete_callback` is executed.
    *   **Symptoms**:
        *   Last `CameraCapture` log printed: `[INFO] CameraCapture: Initial requests queued.`
        *   Often preceded by `[libpisp warning] PushEndDown: (output1) Unable to achieve mandatory alignment 32`. This warning suggests a low-level memory alignment issue in the `libpisp` driver (Raspberry Pi Image Signal Processor) when `libcamera` tries to acquire or manage buffers.
    *   **Diagnosis**: The crash occurs within `libcamera`'s internal handling of the `requestCompleted` signal, likely when it attempts to invoke our callback with a buffer that does not meet the underlying hardware/driver's mandatory alignment requirements. Our application code is not the direct cause of this specific crash, as all processing logic within the callback was successively commented out, down to a minimal logging/requeuing stub, and the crash persisted before its entry log. The `memmove` change was an attempt to mitigate potential unaligned memory access during frame processing, but the crash occurs earlier.
    *   **Current State**:
        *   The compilation errors in `src/camera_capture.cpp` are currently blocking further runtime debugging of this issue.
        *   The `libpisp` alignment warning remains the strongest lead, indicating a hardware/driver/`libcamera` configuration mismatch for buffer alignment.

### Next Steps (once compilation is fixed):
*   Investigate `libcamera` documentation/community for known solutions or workarounds regarding `libpisp` alignment issues on Raspberry Pi 5.
*   Experiment with other `libcamera` `StreamConfiguration` options or `FrameBufferAllocator` flags if available.
*   Consider differences in `libcamera` versions or `libpisp` driver updates.
*   If no direct `libcamera` solution is found, a potential path could be to try an older/different `libcamera` version or to report the bug to the `libcamera` project.