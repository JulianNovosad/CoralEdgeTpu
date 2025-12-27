#!/bin/bash
# Test script to verify RTSP server functionality

echo "Testing RTSP server functionality..."

# First, clean up any existing processes
echo "Cleaning up existing processes..."
./kill_existing_processes.sh

# Start the detector in the background
echo "Starting detector application..."
timeout 30s ./build/detector &
DETECTOR_PID=$!

# Wait for the server to start
sleep 5

# Check if the detector is still running
if kill -0 $DETECTOR_PID 2>/dev/null; then
    echo "Detector is running, checking for RTSP server on port 8554..."
    
    # Check if port 8554 is in use
    if lsof -i :8554 >/dev/null 2>&1; then
        echo "Port 8554 is in use by RTSP server"
        
        # Test with netcat to see if we can connect to the RTSP server
        echo "Testing basic RTSP connection..."
        echo -e "OPTIONS rtsp://127.0.0.1:8554/live RTSP/1.0\r\nCSeq: 1\r\n\r\n" | nc -w 5 127.0.0.1 8554
        
        echo "Attempting to connect with VLC for 10 seconds..."
        timeout 10s cvlc rtsp://127.0.0.1:8554/live --sout dummy --quiet --intf dummy --play-and-exit
    else
        echo "Port 8554 is not in use"
    fi
else
    echo "Detector failed to start or crashed"
fi

# Kill the detector if still running
if kill -0 $DETECTOR_PID 2>/dev/null; then
    kill $DETECTOR_PID
fi

echo "RTSP server test completed."