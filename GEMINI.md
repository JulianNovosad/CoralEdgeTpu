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
