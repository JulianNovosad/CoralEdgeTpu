# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

The CoralEdgeTpu project is an advanced computer vision system built for real-time object detection and tracking using Google Coral Edge TPU hardware. It implements a "smart safety" system that performs AI-based object recognition with ballistics calculations for precision targeting applications.

## Architecture

The system follows a multi-threaded pipeline architecture with these main components:

- **Camera Capture**: Uses libcamera for high-performance video capture with dual streams (main display and TPU inference)
- **Inference Engine**: TensorFlow Lite with Edge TPU acceleration for object detection
- **Logic Module**: Core safety and ballistics calculations, including 3D trajectory prediction
- **Image Processor**: Handles image preprocessing and post-processing operations
- **H264 Encoder**: Video encoding for streaming
- **RTSP Server**: Real-time streaming protocol server
- **System Monitor**: Performance and health monitoring
- **Orientation Sensor**: 3D orientation tracking
- **PCA9685 Controller**: Servo motor control for actuation

## Build System

The project uses CMake for building with the following structure:
- Primary executable: `detector` - Main application binary
- Supporting executables: `camera_isolation_test`, `inference_test_no_logging`, `raw_tpu_test`, `tpu_diagnostic`, `tpu_performance_test`
- Dependencies: libcamera, OpenCV, TensorFlow Lite, Edge TPU runtime, ZeroMQ, Live555, x264

## Development Commands

### Building
```bash
# Full build process (includes dependency setup)
./build.sh

# Standard CMake build (if dependencies already set up)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Alternative debug build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Running
```bash
# Run main detector application
cd build && ./detector

# Run integrated system with dashboard
./start_integrated.sh

# Run tests
cd build
./config_loader_test
./camera_isolation_test
./inference_test_no_logging
```

### Testing and Analysis
```bash
# Run stress test
./run_stress_test.sh

# Run 4-hour GDB test
./run_detector_gdb_4h.sh

# Monitor stress test
./stress_test_monitor.sh

# RTSP frame capture and analysis
python3 rtsp_frame_capture.py

# Various analysis scripts
python3 calculate_fps.py
python3 comprehensive_analysis.py
python3 enhanced_analysis.py
```

### Cleanup
```bash
# Cleanup script
./cleanup.sh

# Clean build directory
rm -rf build/
```

## Key Configuration

- Configuration file: `config.json` - Contains model paths, camera settings, ballistics parameters, and network ports
- Model: `detect_int8_edgetpu.tflite` - INT8 quantized MobileNetSSD model
- Labels: `labelmap.pbtxt` - Object class labels
- Target latency: <100ms end-to-end (with 30ms software budget after 70ms servo actuation)

## Logging and Monitoring

- Log files: `/logs/` directory with CSV-formatted logs
- Real-time monitoring: `Monitor` class with performance metrics
- Telemetry: ZeroMQ-based data streaming on port 11002
- RTSP streaming: Available on port 8554 with mount point `/live`