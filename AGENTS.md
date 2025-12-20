# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

This is the **Avant-garde M1-Delta Mk II Wapenveiligheidssysteem**, an advanced firearm safety system that combines AI object recognition with ballistic modeling to increase user safety and accuracy. The system acts as a "smart safety" that physically blocks the trigger and only releases it when aiming at a validated target with >90% hit probability.

## Core Architecture

The system is designed with an extremely tight **latency budget**:
- Total end-to-end latency requirement: <100 ms
- Mechanical servo actuation delay: ~70 ms
- Software pipeline budget: <30 ms

Key architectural components:
1. **Doeldetectie**: Object detection using a custom-trained MobileNetSSD model (INT8 quantized)
2. **Safety Gating**: Servo motor that blocks the trigger, only releasing when ballistic hit point is within bullseye (13x13cm)
3. **Ballistic Calculations**: Calculates impact point based on distance, bullet trajectory, and sensor data
4. **Feedback**: Streams video with augmented reality overlay to an Android app

## Technology Stack

- **Language**: C++ (no Python/containers in production)
- **Hardware**: Raspberry Pi 5, Google Coral M.2 Edge TPU (PCIe), MG995 Servo via PCA9685 PWM driver
- **Libraries**: TensorFlow Lite, libcamera, OpenCV, ZeroMQ, Boost
- **Optimizations**: Zero-copy pipelines with DMA buffers, custom kernel patches, CPU isolation

## Key Modules

- **Application**: Main orchestrator (src/application.h, src/application.cpp)
- **CameraCapture**: Video capture pipeline using libcamera (src/camera_capture.h, src/camera_capture.cpp)
- **InferenceEngine**: TensorFlow Lite inference with Edge TPU acceleration (src/inference.h, src/inference.cpp)
- **LogicModule**: Core ballistic calculations and safety logic (src/logic.h, src/logic.cpp)
- **ImageProcessor**: Image preprocessing (src/image_processor.h, src/image_processor.cpp)
- **SystemMonitor**: System health monitoring (src/system_monitor.h, src/system_monitor.cpp)

## Build Commands

```bash
# Full build (installs dependencies, builds FlatBuffers, sets up TensorFlow)
./build.sh

# Alternative manual build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-Werror" ../
make -j$(nproc)

# Build specific targets
make detector                    # Main application
make config_loader_test         # Configuration loader tests
make servo_test                 # Servo controller tests
```

## Test Commands

```bash
# Run specific tests
./config_loader_test            # Configuration loader unit tests
make config_loader_test && ./config_loader_test  # Build and run config tests

# List all available test targets
make help | grep test
```

## Configuration

Main configuration file: `config.json`

Key sections:
- `application`: Model paths, resolution settings, thresholds
- `ballistics`: Weapon and ammunition properties
- `tracking`: Object tracking parameters
- `safety`: Safety thresholds for trigger release
- `network_ports`: Network port mappings for communication

## Code Structure

```
src/
├── application.h/cpp          # Main application orchestrator
├── camera_capture.h/cpp       # Libcamera-based video capture
├── inference.h/cpp            # TFLite inference engine
├── logic.h/cpp                # Ballistic calculations and safety logic
├── image_processor.h/cpp      # Image preprocessing
├── config_loader.h/cpp        # Configuration management
├── pca9685_controller.h/cpp   # Servo motor control
├── system_monitor.h/cpp       # System health monitoring
├── pipeline_structs.h         # Data structures for pipeline
└── buffer_pool.h             # Memory management utilities

tests/
├── config_loader_test.cpp    # Unit tests for config loader
└── other test files
```

## Performance Requirements

Stage Gate Plan:
1. **Stage 0**: Technical feasibility & performance limits (≥120 FPS camera/TPU)
2. **Stage 1**: System-wide C++ implementation & bottleneck analysis
3. **Stage 2**: Full integration & zero-copy optimization (<100ms end-to-end latency)
4. **Stage 3**: Validation & verification (4-hour stress test)

## Key Files

- `detector`: Main executable
- `config.json`: Main configuration
- `detect_int8_edgetpu.tflite`: AI model
- `coco_labels.txt`: Object detection labels
- `logs/`: Log directory for all subsystems