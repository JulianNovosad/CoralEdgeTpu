# GEMINI Project Memory (CoralEdgeTpu)

## Project Overview

This project is a C++ Edge TPU inference stack designed for the Google Coral M.2 Accelerator, focusing on native C++ inferencing. It represents a modern, stable, and reproducible re-implementation of an older Coral/TensorFlow Lite toolchain, aimed at eliminating dependency issues.

The application features a multi-threaded architecture handling camera capture, real-time inference using TensorFlow Lite with Edge TPU acceleration, H.264 video encoding, and streaming via UDP/RTP or RTSP. It also supports bounding box detection, reticle coordinates, and telemetry through ZeroMQ PUB/SUB. Configuration is managed via `config.json`, with robust logging, microbenchmarks, and a `system_monitor` for overall health. The core real-time decision-making, including 3D ballistics, hit-scan, servo-actuation, and safety propagation, is handled by the `logic` module.

## Building and Running

### Requirements

*   **Hardware:** Raspberry Pi 5, Google Coral M.2 TPU (PCIe)
*   **System Dependencies:** `libcamera-dev`, `libedgetpu1-std`, `libzmq3-dev`
*   **Build Tools:** CMake ≥ 3.16, g++ ≥ 10

### Build Instructions

The project uses an all-in-one script (`build.sh`) to download, compile dependencies, and build the main application.

```bash
chmod +x build.sh
./build.sh
```

### Running the Application

After a successful build, the executable `detector` will be located in the `build/` directory.

```bash
./build/detector
```

The application expects a `config.json` file to be present in the root of the repository for configuration.

## Development Principles (from original GEMINI.md and README.md)

*   Never hardcode paths; always locate files using `find` or `grep -r`.
*   After 2-3 code changes, run `/home/pi/CoralEdgeTpu/build.sh` to validate build and integration.
*   Prefer `gdb` or `valgrind` for debugging concurrency issues.
*   Track changes with Git and include patch files in `/home/pi/CoralEdgeTpu/patches`.
*   Emphasis on reproducibility, offline builds, and eliminating dependency chaos.
*   Extensive logging with microbenchmarks and CSV output.
*   Central supervision via `system_monitor`.
*   All core real-time action and decisions via `logic` module (3D ballistics, hit-scan, servo-actuation, safety/uncertainty propagation).
*   Zero-copy pipeline optimization where feasible.

## Safety-Critical Constraints (IEC 61508 / SIL2) (from original GEMINI.md)

*   **Threads**:
    *   RT threads: IMU, Camera, Ballistics
    *   Control thread: orchestrates predictive fire & safety gate
*   **Memory**: pre-allocate in init phase; zero heap allocation in RT loops.
*   **Synchronization**: use lock-free `boost::lockfree::spsc_queue` and `std::atomic<uint64_t> frame_seq` with `memory_order_seq_cst`.
*   **Timestamps**: use `CLOCK_MONOTONIC_RAW` for all timestamps.
*   Inspect device-specific API first, then targeted web searches for kernel/TPU/ARM to avoid generic Stack Overflow copy-paste.

## Documentation & API Contracts (from original GEMINI.md and README.md)

*   **Structs & unions**: document in README.md with size, alignment, thread ownership, invariants (e.g., `IMU::ts_us` monotonic)
*   **Latency budgets**: version-controlled in `config/latency_budgets.csv` (Note: `config/latency_budgets.csv` not explicitly found in folder structure, but referenced here. Will assume it's an intended file.)
*   **Fault injection tests**: `/home/pi/CoralEdgeTpu/tests/FAULT_INJECTION.md`
*   **Debug mode**: `#define DEBUG_SAFETY_GATE_BYPASS 0` → bypass only for testing, never in prod.
*   **Pinout & I2C**: `hardware/pinout.md` with PCA9685 address (0x40), GPIO numbers, interrupts. (Note: `hardware/pinout.md` not explicitly found in folder structure, but referenced here. Will assume it's an intended file.)
*   Logging per Subsystem: Each core subsystem (Camera, TPU, Encoder, Logic, System Monitor) must have its own CSV log files in a subdirectory under the configured `log_path` (e.g., `/logs/camera`). Filename convention: `module_YYYY_MM_DD_HH:MM.csv`. Max 3 log rotations per subsystem.
*   Look at the `CMakeLists.txt` for build- and compilerconfiguratie.

## Stage-Gate Plan

The project follows a rigorous Stage-Gate Plan to ensure systematic development and validation:

### Stage 0: Technical feasibility & performance limits

*   **Goal:** Measure basal throughput and latency of core subsystems without full integration.
*   **Subsystems:** Logic, Camera (`src/camera_capture.*`, `src/buffer_pool.h`, `src/pipeline_structs.h`), TPU (`src/inference.*` + `.tflite model`), Encoder (`src/h264_encoder.*`).
*   **Gating criteria:** FPS, calculations/s, and latency per subsystem measured; Kernel adaptations documented; Logging per Subsystem (CSV log files, specific naming conventions, rotation).

### Stage 1: System-wide C++ implementation & bottleneck analysis

*   **Goal:** All core real-time actions via `logic` module, using mutex-based `ThreadSafeQueue`.
*   **Core Subsystems:** Logic (3D ballistics, hit-scan, servo-actuation, uncertainty propagation), Camera & DMA (`src/camera_capture.*`, `src/buffer_pool.h`, `src/pipeline_structs.h`), TPU Inference (`src/inference.*` + `.tflite` model), System Monitor (`src/system_monitor.*`).
*   **Gating criteria:** All core real-time functions running independently.

### Stage 2: Full integration & zero-copy optimization

*   **Goal:** End-to-end pipeline with DMA-sharing buffers and validation over 100,000 frames.
*   **Features:** Zero-copy pipeline (`src/camera_capture.*` → `logic` → `src/inference.*` → `src/video_overlay_processor.*`), video stream (UDP/RTP or RTSP), bounding boxes/telemetry (ZeroMQ PUB/SUB), fallback switching tests.
*   **Performance Requirements:** E2E latency < 100 ms with <5% jitter; TPU throughput ≥ 90 FPS per 100 FPS capture; stable temperature, no throttling.
*   **Gating criteria:** Pipeline operational, fallback modes tested, video & telemetry streams working.

### Stage 3: Validation & verification

*   **Goal:** 4-hour stress test and firing range validation of all critical systems.
*   **Tests:** Continuous logging of thermals and jitter (CSV + PNG graphs), uncertainty propagation verification, `logic` module verification, `system_monitor` supervision and logging tests.
*   **Stability Requirement:** E2E latency within 5% of nominal value over full test duration.
*   **Gating criteria:** Safety margins confirmed, thermal stability proven, uncertainty model validated.

## Core Data Structures and Threading Model

*   `ImageData` (`pipeline_structs.h`): Producer `CameraCapture`, Consumer `logic`, `VideoOverlayProcessor`.
*   `OrientationData` (`pipeline_structs.h`): Producer `OrientationSensor`, Consumer `logic`.
*   `DetectionResult` (`pipeline_structs.h`): Producer `logic`/`InferenceEngine`, Consumer `VideoOverlayProcessor`, telemetry via ZeroMQ.
*   `TrackedObject` (`src/logic.h`): Represents an object tracked over time, managed by `logic`.

## Repository Structure

```
CoralEdgeTpu/
├── build/                   # Build output (generated)
├── civetweb/                # Submodule for the webserver (optional)
├── docs/                    # Documentation
├── flatbuffers/             # Submodule for Flatbuffers
├── include/                 # TFLite headers
├── lib/                     # Compiled libraries (.so files)
├── model/                   # .tflite AI models
├── src/                     # C++ source code of the application
├── tests/                   # Standalone test-utilities
├── tensorflow-src/          # TensorFlow source code (submodule)
├── build.sh                 # Main build script
├── CMakeLists.txt           # CMake build configuration
├── config.json              # Application configuration
└── README.md                # This file
```
