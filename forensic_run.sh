#!/bin/bash

# Build and run with forensic dump capability
# Uses make -j2 to keep memory pressure low as specified
# Performs make clean first to wipe any corrupted artifacts

echo "Starting forensic run of detector..."

# Clean previous build artifacts
echo "Cleaning previous build artifacts..."
make -C build clean

# Build with low memory pressure
echo "Building detector with make -j2..."
make -j2 -C build

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully."

# Run with the specified hard kill sequence
echo "Starting detector with 15-second timeout..."
./build/detector & 
DETECTOR_PID=$!
sleep 15
kill -9 $DETECTOR_PID 2>/dev/null || true
echo "Detector (PID: $DETECTOR_PID) has been terminated."

echo "Forensic run completed."