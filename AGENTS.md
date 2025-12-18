# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

This is the Avant-garde M1-Delta Mk II weapon safety system, a real-time computer platform that combines AI object recognition with ballistic modeling to enhance weapon safety and accuracy. The system acts as a "smart safety" mechanism that physically blocks the trigger and only releases it when aiming at a validated target with >90% hit probability.

## Key Features

- Real-time object detection using MobileNetSSD model (INT8 quantized)
- Safety gating with servo motor that blocks trigger
- 3D ballistic calculations for impact prediction
- Augmented reality video streaming with bounding boxes
- Ultra-low latency (<100ms end-to-end) with custom kernel optimizations

## Architecture Overview

The system follows a modular pipeline architecture with these core components:

1. **CameraCapture** - Acquires video frames using libcamera with zero-copy DMA buffers
2. **InferenceEngine** - Runs AI object detection on Google Coral Edge TPU
3. **LogicModule** - Performs 3D ballistics, hit scanning, and servo actuation
4. **ImageProcessor** - Handles image preprocessing and postprocessing
5. **SystemMonitor** - Tracks system health metrics (CPU temp, memory usage)
6. **H264Encoder** - Encodes video stream for network transmission
7. **Application** - Central coordinator that manages all modules

## Build Commands

```bash
# Configure the project with CMake
cmake .

# Build the main application
make detector

# Build tests
make config_loader_test
```

## Run Commands

```bash
# Run the main application
./detector

# Run tests
./config_loader_test
```

## Key Files and Directories

- `src/` - Main source code directory
- `src/application.cpp/h` - Main application orchestrator
- `src/camera_capture.cpp/h` - Camera frame acquisition
- `src/inference.cpp/h` - AI inference engine
- `src/logic.cpp/h` - Core logic, ballistics, and actuation
- `src/system_monitor.cpp/h` - System health monitoring
- `CMakeLists.txt` - Build configuration
- `config.json` - Application configuration
- `tests/` - Unit tests

## Development Guidelines

- All code is written in C++17 with strict compiler flags (-Wall -Wextra -Werror)
- Uses lock-free SPSC queues for inter-module communication
- Implements buffer pooling to minimize memory allocations
- Follows strict latency budgets (total <100ms)
- Extensive structured logging with CSV format for performance analysis
- All modules must shut down gracefully within 100ms of stop signal

## Testing

Unit tests use Google Test framework:
```bash
# Build and run specific tests
make config_loader_test && ./config_loader_test
```

## Dependencies

- TensorFlow Lite with Edge TPU support
- libcamera for camera capture
- OpenCV for image processing
- ZeroMQ for telemetry
- Boost lockfree queues
- libjpeg for image encoding
- x264 for video encoding (optional)