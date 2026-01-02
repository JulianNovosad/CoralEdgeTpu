#!/bin/bash
set -e

# Cleanup potential zombie processes and release resources
echo "Performing pre-test resource cleanup..."
killall -9 detector 2>/dev/null || true
killall -9 libcamera-vid 2>/dev/null || true
killall -9 libcamera-hello 2>/dev/null || true
# Give time for hardware to reset
sleep 3

# Navigate to project directory
cd /home/pi/CoralEdgeTpu

# Create and enter build directory
mkdir -p build
cd build

# Configure and build
echo "Configuring with CMake..."
cmake ..

echo "Building with make..."
make -j2

if [ $? -eq 0 ]; then
    echo "Build successful."
    echo "Starting detector for 30s..."
    cp ../config.json .
    timeout 30s ./detector || true
else
    echo "Build failed."
    exit 1
fi
