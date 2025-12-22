# CoralEdgeTpu Cleanup Mechanisms

## Overview

This document describes the comprehensive cleanup mechanisms implemented for the CoralEdgeTpu detector system. These mechanisms ensure proper resource deallocation, temporary file removal, and system stability during both normal operation and error conditions.

## Application-Level Cleanup

### Pre-Launch Cleanup

Before starting the application, the system performs aggressive cleanup to ensure a clean state:

1. **Process Termination**: Any existing detector instances are terminated
2. **Resource Release**: Edge TPU and camera resources are released
3. **Socket Cleanup**: Telemetry sockets are cleared
4. **Aggressive Resource Cleanup**: Additional cleanup of temporary files, IPC resources, and zombie processes
5. **TPU Availability Check**: Verifies TPU is available and waits if needed, or force releases if necessary

### Post-Shutdown Cleanup

After the application shuts down, comprehensive cleanup is performed:

1. **Thread Management**: All threads are properly stopped and joined
2. **Resource Release**: Camera and Edge TPU delegates are released
3. **Socket Cleanup**: Telemetry sockets are closed and temporary files removed
4. **Aggressive Resource Cleanup**: Extended cleanup procedures
5. **TPU Status Verification**: Ensures TPU is properly released for next use

### Aggressive Resource Cleanup Components

The aggressive cleanup includes several specialized functions:

#### Memory Leak Detection
- Performs checks for memory leaks (integration point for memory profiling tools)

#### Temporary File Cleanup
- Removes detector pipe files
- Cleans up `.tmp` files in the current directory
- Tracks number of files removed

#### IPC Resource Cleanup
- Cleans up ZeroMQ sockets and other IPC resources

#### Shared Memory Cleanup
- Identifies and removes shared memory segments

#### Zombie Process Cleanup
- Reaps any zombie child processes
- Tracks number of zombies reaped

#### Cleanup Reporting
- Generates detailed reports of cleanup activities

## TPU Occupancy Management

### TPU Availability Check
- Checks if Edge TPU devices are present
- Attempts to create a test delegate to verify availability
- Logs warnings if TPU is not available

### Wait for TPU Release
- Waits for a specified time (default 10 seconds) for TPU to become available
- Periodically checks TPU availability
- Returns success or failure based on whether TPU became available

### Force Release TPU Resources
- Kills detector processes that might be using the TPU
- Provides a fallback mechanism when normal waiting fails

### TPU Status Verification
- Verifies TPU device exists at `/dev/apex_0`
- Confirms TPU availability through delegate creation
- Reports status for diagnostic purposes

## System-Wide Cleanup Script

The `cleanup.sh` script provides system-wide cleanup capabilities:

1. **Process Termination**: Kills all detector-related processes
2. **Temporary File Removal**: Removes log files and temporary files
3. **Build Artifact Cleanup**: Removes build directories
4. **Shared Memory Cleanup**: Removes shared memory segments
5. **Semaphore Cleanup**: Removes semaphore arrays
6. **Resource Verification**: Checks for dangling file descriptors

## Usage

### Application-Level Cleanup
The cleanup mechanisms are automatically invoked during application startup and shutdown.

### System-Wide Cleanup
To perform a system-wide cleanup, run:
```bash
./cleanup.sh
```

## Benefits

1. **Resource Efficiency**: Prevents resource leaks and accumulation
2. **System Stability**: Ensures clean state for subsequent runs
3. **Error Recovery**: Helps recover from abnormal termination
4. **Performance**: Maintains optimal system performance by preventing resource exhaustion
5. **TPU Management**: Prevents "TPU busy" issues by explicitly managing TPU occupancy