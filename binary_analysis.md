# Binary Analysis

## detector
- **Source file**: `src/main.cpp`
- **Purpose**: Main application that runs the full detection pipeline
- **Components**:
  - Camera capture (libcamera)
  - Image processing
  - TFLite inference with EdgeTPU acceleration
  - Logic module with ballistics calculations
  - Servo control
  - System monitoring
  - H.264 encoding and RTSP streaming
  - Orientation sensing
- **Entry point**: `main()` in `src/main.cpp` which creates and runs an `Application` object

## dashboard
- **Source file**: `dashboard_fixed.cpp`
- **Purpose**: Monitoring and visualization tool that parses log output
- **Components**:
  - Log parsing and analysis
  - Real-time dashboard display
  - System metrics tracking
  - Detection visualization
- **Entry point**: `main()` in `dashboard_fixed.cpp`

## integrated_system
- **Source file**: `src/integrated_main.cpp`
- **Purpose**: Wrapper that runs both detector and dashboard together
- **Components**:
  - Process management (forks detector as child process)
  - FIFO pipe communication between detector and dashboard
  - Signal handling for graceful shutdown
- **Entry point**: `main()` in `src/integrated_main.cpp`

## Other test binaries
- **camera_isolation_test**: Tests camera isolation functionality
- **inference_test_no_logging**: Tests inference without logging overhead
- **raw_tpu_test**: Tests raw TPU functionality
- **config_loader_test**: Tests configuration loading
- **servo_test**: Tests servo control
- **tpu_performance_test**: Tests TPU performance characteristics

## Summary
- **detector** is the main application that performs all core functionality
- **dashboard** is a monitoring tool that visualizes system performance
- **integrated_system** is a convenience wrapper that runs both detector and dashboard together