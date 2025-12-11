# GEMINI Project Memory: CoralEdgeTpu

This document provides a high-level overview of the CoralEdgeTpu project, its architecture, and development conventions. It is intended to be used as a quick reference for developers and for context when interacting with the Gemini CLI.

## Project Overview

The `CoralEdgeTpu` project is a high-performance, real-time video processing and machine learning inference application designed to run on a Raspberry Pi 5 with a Google Coral M.2 TPU accelerator. The project is written in C++ and aims for a stable, reproducible, and efficient pipeline, avoiding Python dependencies for the core application.

**Core Technologies:**
- **C++17:** The primary programming language.
- **TensorFlow Lite & Edge TPU:** For hardware-accelerated machine learning inference.
- **libcamera:** Modern C++ API for camera control on the Raspberry Pi.
- **CMake:** Used for building the application.
- **ZeroMQ:** For message passing (telemetry, bounding boxes).
- **CivetWeb:** For providing a web server interface.
- **OpenCV:** For image processing tasks.

**Architecture:**
The application uses a multi-threaded, pipeline-based architecture. Different modules run on separate threads and communicate through thread-safe queues and buffer pools. This design allows for parallel processing of camera capture, inference, video encoding, and core logic.

The main modules are:
- `Application`: The central class that initializes and manages the lifecycle of all other modules.
- `CameraCapture`: Handles capturing video frames from the camera using `libcamera`.
- `InferenceEngine`: Performs object detection on video frames using the TensorFlow Lite model and the Edge TPU delegate.
- `LogicModule`: Contains the core application logic, including 3D ballistics, tracking, and safety systems.
- `H264Encoder`: Encodes video frames into the H.264 format.
- `SystemMonitor`: Monitors system resources like CPU, RAM, and temperature.
- `ApplicationSupervisor`: Manages graceful startup and shutdown of all modules.

## Building and Running

**Prerequisites:**
- Raspberry Pi 5 with a Google Coral M.2 TPU.
- System dependencies as listed in `build.sh` (e.g., `libcamera-dev`, `libzmq3-dev`, `cmake`, `g++`).

**Building:**
The project includes a comprehensive build script that handles all steps, including installing dependencies, cloning submodules, and compiling the code.

```bash
# Make the script executable
chmod +x build.sh

# Run the build script
./build.sh
```
The final executable will be located at `build/detector`.

**Running:**
The application requires a `config.json` file in the root directory.

```bash
./build/detector
```

## Development Conventions

- **Submodules:** Dependencies like TensorFlow, Flatbuffers, and CivetWeb are included as Git submodules. The `build.sh` script handles their initialization.
- **Dependency Patching:** The `patches/` directory contains patches for third-party libraries. This indicates a convention of applying targeted fixes rather than maintaining full forks.
- **Strict Compiler Flags:** The project is compiled with `-Wall -Wextra -Werror`, enforcing a high level of code quality and preventing common errors.
- **Stage-Gate Plan:** The `README.md` file outlines a formal "Stage-Gate Plan" for development, breaking the project into logical stages with clear goals and gating criteria. This is the primary roadmap for the project.
- **Logging:** A structured CSV-based logging format is defined in the `README.md`. All modules are expected to adhere to this format, which includes timestamps for latency calculation (`produced_ts_epoch_ms` - `call_ts_epoch_ms`).
- **Configuration:** All major parameters are expected to be configurable through the `config.json` file, which is loaded at startup by the `ConfigLoader` module.
