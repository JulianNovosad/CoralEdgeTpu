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

### Resolved Issues:

*   **Numerous Compilation Errors**: Fixed a wide range of compilation errors stemming from the introduction of new modules, queue changes (`boost::lockfree::spsc_queue`), and refactoring. This included fixing missing headers, incorrect function signatures, access specifier issues, and type mismatches.
*   **`std::map` Insertion of Non-Copyable Types**: Resolved a complex C++ issue in the `Logger` where `std::map::emplace` was failing with non-copyable `CsvLogger` objects (due to `std::ofstream` and `std::mutex` members). The fix involved switching to `std::map::try_emplace` (C++17), which constructs the object in-place.
*   **`x264` `memcpy` Crash (SPS/PPS Headers)**: Temporarily disabled the copying of SPS/PPS headers to isolate the recurring `SIGSEGV`.
*   **Persistent `Segmentation fault` during Camera Startup (YUV Multi-Plane Handling)**:
    *   **Problem**: The application was crashing with `SIGSEGV` (Exit Code 139) within `CameraCapture::process_frame_buffer` during the `memcpy` of YUV data from `libcamera` buffers. `libcamera` provides YUV data in multiple planes, and the initial `mmap` and `memcpy` logic was not correctly handling these separate planes or their respective sizes.
    *   **Resolution**: Refactored `CameraCapture::process_frame_buffer` to correctly `mmap` each `libcamera` frame buffer plane individually (using its specific `fd` and `offset`) and copy its data into the appropriate location within a single `PooledBuffer`. This ensures that `memcpy` operations access valid memory regions provided by `libcamera` for each plane.
*   **`std::bad_weak_ptr` Exception (BufferPool Lifetime Management)**:
    *   **Problem**: The application was terminating with a `std::bad_weak_ptr` exception, indicating incorrect usage of `std::shared_ptr` and `std::weak_ptr` within the `BufferPool` class, specifically when `this->shared_from_this()` was called in the constructor.
    *   **Resolution**: Refactored the `BufferPool` class to properly manage `std::shared_ptr` lifetimes. This involved ensuring that `shared_from_this()` is not called in the constructor and redesigning the `PooledPtr`'s custom deleter and the pool's `return_buffer_to_pool` method to correctly re-wrap raw buffer pointers into `PooledPtr`s, preventing double-free or invalid `shared_ptr` aliasing.
*   **`H264Encoder` `x264_encoder_open` Crash**:
    *   **Problem**: The application was crashing after `x264_encoder_open` in `H264Encoder::worker_thread_func`, even after `x264_picture_alloc`. This was due to a misunderstanding of `x264_picture_alloc`'s role; it *does* allocate the internal plane buffers for `x264_picture_t`. Manual `malloc` calls after `x264_picture_alloc` were redundant and causing memory corruption when `memcpy` attempted to write to these invalidly managed pointers.
    *   **Resolution**: Removed the redundant manual `malloc` calls for `picture_in_`'s planes. Relied solely on `x264_picture_alloc` to manage the allocation of these internal buffers, ensuring that the `memcpy` operations correctly target memory managed by x264.
*   **`config.json` Loading Failure**:
    *   **Problem**: The application failed to load configuration, reporting "Model file not found: ./build/../model.tflite". This was because `ConfigLoader` expected configuration values under an "application" object, but `config.json` had a flat structure.
    *   **Resolution**: Updated `config.json` to include all configuration keys nested under an "application" object, matching `ConfigLoader`'s expected structure.
*   **Logger Not Printing Output**:
    *   **Problem**: Log messages were not appearing on stdout or in log files at startup, making debugging difficult.
    *   **Resolution**: Added an explicit call to `logger.start_writer_thread()` after initializing the `Logger` instance in `main.cpp`. The logger's asynchronous writer thread needs to be explicitly started to process and output log messages.
*   **`LogicModule` Helper Function Definitions Missing/Duplicated**:
    *   **Problem**: Compilation errors due to missing definitions for `LogicModule`'s helper methods (`perform_sensor_fusion`, `update_object_tracks`, `calculate_ballistics_for_tracks`, `perform_safety_and_actuation`) or duplicate definitions causing conflicts.
    *   **Resolution**: Refactored `src/logic.cpp` to correctly define these helper methods by extracting their logic from the monolithic `process` function. Ensured each method had a single, correct definition, and replaced `IMUData` with `OrientationData` where appropriate.
*   **`IMUSensor` to `OrientationSensor` Refactoring**:
    *   **Problem**: The user requested removing all "IMU" mentions and focusing on "orientation" data.
    *   **Resolution**: Renamed `src/imu_sensor.h` to `src/orientation_sensor.h` and `src/imu_sensor.cpp` to `src/orientation_sensor.cpp`. Updated all include directives, class names, variable names, method calls, and log messages across the project (including `src/main.cpp`, `src/logic.h`, `src/logic.cpp`) to consistently use "OrientationSensor" and "OrientationData".

### Current Status:

The application now builds and runs without encountering critical startup crashes. All core modules are initialized and their worker threads are started successfully.

### Next Steps:

*   **Continuous Operation Testing**: Verify that the application continues to run stably over an extended period.
*   **Functional Verification**: Confirm that all pipeline stages (camera capture, inference, video overlay, H.264 encoding, HTTP streaming) are performing as expected.
*   **Performance Monitoring**: Utilize the implemented CSV logging to analyze performance metrics (latency, throughput, CPU/memory usage) and identify potential bottlenecks.
*   **Edge TPU Delegate Integration**: Although not a current crash, ensure the Edge TPU delegate is being correctly utilized for inference and not falling back to CPU.
*   **Camera Configuration**: Investigate and address the `[libpisp warning] PushEndDown: (output1) Unable to achieve mandatory alignment 32` if it leads to performance issues or unexpected behavior.
*   **Review `x264` SPS/PPS headers**: Re-enable and properly handle the SPS/PPS headers in `H264Encoder` if client streaming requires them, ensuring they are copied to the H.264 buffer correctly.