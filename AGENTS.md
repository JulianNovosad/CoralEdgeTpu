# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

This is a C++ computer vision application that runs on Raspberry Pi with Google Coral Edge TPU hardware. The application implements a real-time object detection pipeline with the following stages:

1. Camera capture using libcamera
2. Image preprocessing
3. Object detection using TensorFlow Lite and Edge TPU
4. Logic processing for ballistics calculations
5. Video encoding and streaming
6. Telemetry publishing via ZeroMQ

The application follows a stage-gate development plan with 4 stages focusing on performance, integration, and validation.

## Build Commands

To build the project:
```bash
./build.sh
```

This script will:
1. Install required packages (libcamera-dev, libzmq3-dev, libedgetpu-dev)
2. Build FlatBuffers v1.12.0
3. Clone and set up CivetWeb
4. Checkout and build TensorFlow Lite v2.5.0
5. Build the main C++ application with CMake

To manually build with CMake:
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-Werror" ../
make -j$(nproc)
```

## Test Commands

To run unit tests:
```bash
cd build && ./config_loader_test
```

To build and run tests:
```bash
cd build && make && ./config_loader_test
```

## Code Architecture

The application follows a modular architecture with these key components:

### Core Modules
- `Application` - Main orchestrator that initializes and manages all subsystems
- `CameraCapture` - Handles camera initialization and frame capture using libcamera
- `InferenceEngine` - Manages TensorFlow Lite models and Edge TPU delegation
- `LogicModule` - Implements ballistics calculations and hit prediction
- `ImageProcessor` - Preprocesses images for inference
- `H264Encoder` - Encodes video streams
- `SystemMonitor` - Monitors system health (CPU, temperature, memory)

### Support Modules
- `ConfigLoader` - Loads and parses JSON configuration files
- `BufferPool` - Manages memory pools for efficient buffer allocation
- `UtilLogging` - Centralized logging system with file output
- `KeyboardMonitor` - Handles keyboard input for application control

### Data Flow
1. Raw frames are captured by `CameraCapture` and placed in queues
2. `ImageProcessor` preprocesses frames for inference
3. `InferenceEngine` performs object detection on the Edge TPU
4. Results are processed by `LogicModule` for ballistics calculations
5. Processed video is encoded by `H264Encoder`
6. Telemetry is published via ZeroMQ

### Key Data Structures
- `DetectionResult` - Contains bounding box, class, and confidence information
- `ImageBuffer` - Holds raw image data with metadata
- `BallisticsData` - Contains trajectory and targeting information
- Various queue types for inter-module communication

## Configuration

The application uses JSON configuration files with the following structure:
```json
{
  "application": {
    "model_path": "path/to/model.tflite",
    "labels_path": "path/to/labels.txt",
    "camera_fps": 60.0,
    "detection_score_threshold": 0.7,
    "ballistics": {
      "muzzle_velocity_mps": 850.0,
      "bullet_mass_kg": 0.008
    }
  }
}
```

## Logging

All modules log to timestamped CSV files in `/logs/<module>/` directories. Logs follow a universal CSV header format and contain chronological, consistent data with no missing values.

## Development Guidelines

1. All code must compile without warnings (-Werror flag enabled)
2. Memory management uses smart pointers and buffer pools
3. Thread safety is implemented with mutexes and atomic operations
4. All modules must gracefully handle shutdown signals
5. Performance is critical - maintain ≥120 FPS for camera and TPU inference
6. All inter-module communication uses lock-free queues where possible
- Do not work aroud build.sh issues by building directly with `cmake` shell commands
