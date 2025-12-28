#!/bin/bash

# Test script to verify RTSP stream functionality

# Configuration
RTSP_URL="rtsp://127.0.0.1:8554/live"
DETECTOR_PATH="./build/detector"

# Function to cleanup
cleanup() {
    echo "Cleaning up..."
    pkill -9 detector 2>/dev/null || true
    exit
}

# Set up signal trap for cleanup
trap cleanup EXIT INT TERM

echo "Starting detector process..."
$DETECTOR_PATH > /dev/null 2>&1 &
DETECTOR_PID=$!

echo "Waiting 5 seconds for TPU and RTSP server to initialize..."
sleep 5

echo "Testing RTSP stream with VLC..."
VLC_OUTPUT=$(timeout 10s cvlc --no-video --play-and-exit --run-time=5 "$RTSP_URL" 2>&1)

echo "VLC test completed. Analyzing output..."

# Check for success indicators
if echo "$VLC_OUTPUT" | grep -E -i "successfully opened|track.*info" | grep -v "buffer too late" > /dev/null; then
    echo "✅ SUCCESS: Stream appears to be working correctly!"
    echo "VLC output contained success indicators and no 'buffer too late' errors."
else
    echo "❌ FAILURE: Stream may not be working properly."
    echo "VLC output did not contain expected success indicators or contained errors."
fi

echo ""
echo "=== VLC Output ==="
echo "$VLC_OUTPUT"
echo "=================="

# Kill detector process
pkill -9 detector 2>/dev/null || true

exit 0