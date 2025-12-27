#!/bin/bash

# Test script for RTSP server connection issues

# Start the RTSP server in the background and redirect output to a log file
echo "Starting RTSP server..."
./build/detector > server_output.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait a moment for the server to initialize
sleep 3

# Create a log file for the test results
TEST_LOG="test_rtsp_fix.log"
echo "$(date): Starting 10-second RTSP connectivity test" > $TEST_LOG

# Test client connections every second for 10 seconds
for i in {1..10}; do
    echo "$(date): Attempt $i - Testing client connection..."
    echo "$(date): Attempt $i - Testing client connection..." >> $TEST_LOG
    
    # Try with gst-launch-1.0 first
    timeout 5 gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/live latency=0 ! fakesink 2>&1 | tee -a $TEST_LOG
    
    # If gst-launch-1.0 is not available, try cvlc
    if ! command -v gst-launch-1.0 &> /dev/null; then
        timeout 5 cvlc rtsp://127.0.0.1:8554/live --intf dummy --run-time=1 2>&1 | tee -a $TEST_LOG
    fi
    
    sleep 1
done

echo "$(date): RTSP connectivity test completed" >> $TEST_LOG

# Check for h264parse warnings in server output
echo "$(date): Checking server output for h264parse warnings..." >> $TEST_LOG
grep -i "h264parse\|refused\|caps\|SPS\|PPS" server_output.log >> $TEST_LOG || echo "No h264parse warnings found in server output" >> $TEST_LOG

# Kill the server process
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Server process $SERVER_PID terminated"

echo "Test completed. Results logged to $TEST_LOG"