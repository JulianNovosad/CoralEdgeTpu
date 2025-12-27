#!/bin/bash

# Create a temporary log file
LOG_FILE="/tmp/rtsp_test_$(date +%s).log"
CLIENT_LOG="/tmp/rtsp_client_$(date +%s).log"
SERVER_LOG="/tmp/rtsp_server_$(date +%s).log"

# Function to cleanup
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "detector.*--rtsp" 2>/dev/null || true
    pkill -f "gst-launch-1.0" 2>/dev/null || true
    sleep 2
    pkill -9 -f "detector.*--rtsp" 2>/dev/null || true
    pkill -9 -f "gst-launch-1.0" 2>/dev/null || true
}

# Set up signal trap
trap cleanup EXIT

echo "Starting RTSP server test..." > $LOG_FILE
echo "Server log: $SERVER_LOG" >> $LOG_FILE
echo "Client log: $CLIENT_LOG" >> $LOG_FILE

# Start the RTSP server in the background
echo "Starting RTSP server..." >> $LOG_FILE
timeout 20s /home/pi/CoralEdgeTpu/build/detector --rtsp > $SERVER_LOG 2>&1 &
SERVER_PID=$!

# Wait a bit for server to initialize
sleep 3

# Check if server started successfully
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "RTSP server started successfully (PID: $SERVER_PID)" >> $LOG_FILE
else
    echo "RTSP server failed to start" >> $LOG_FILE
    cat $SERVER_LOG >> $LOG_FILE
    exit 1
fi

# Wait 10 seconds before starting the client (total: 13 seconds have passed since start)
echo "Waiting 10 seconds before starting client..." >> $LOG_FILE
sleep 10

# Start the test client after 10 seconds
echo "Starting RTSP client..." >> $LOG_FILE
timeout 10s gst-launch-1.0 -v rtspsrc location=rtsp://127.0.0.1:8554/live protocols=tcp ! rtph264depay ! avdec_h264 ! fakesink silent=false sync=false > $CLIENT_LOG 2>&1 &
CLIENT_PID=$!

# Wait for client to finish (should be ~10 seconds)
sleep 12

# Check if both processes are still running and kill if needed
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Stopping server (PID: $SERVER_PID)" >> $LOG_FILE
    kill $SERVER_PID 2>/dev/null || true
fi

if kill -0 $CLIENT_PID 2>/dev/null; then
    echo "Stopping client (PID: $CLIENT_PID)" >> $LOG_FILE
    kill $CLIENT_PID 2>/dev/null || true
fi

# Wait a bit for graceful shutdown
sleep 2

# Collect results
echo "=== RTSP Test Results ===" >> $LOG_FILE
echo "Server log contents:" >> $LOG_FILE
cat $SERVER_LOG >> $LOG_FILE
echo "" >> $LOG_FILE
echo "Client log contents:" >> $LOG_FILE
cat $CLIENT_LOG >> $LOG_FILE

# Analyze results
echo "" >> $LOG_FILE
echo "=== ANALYSIS ===" >> $LOG_FILE

# Check for client connection success
if grep -q "connected" $SERVER_LOG || grep -q "PLAY" $SERVER_LOG || grep -q "SET" $SERVER_LOG; then
    echo "Client connected successfully" >> $LOG_FILE
else
    echo "Client connection failed" >> $LOG_FILE
fi

# Check for frame streaming
if grep -q "push" $SERVER_LOG || grep -q "buffer" $SERVER_LOG; then
    echo "Frames were successfully streamed" >> $LOG_FILE
else
    echo "No frames detected in stream" >> $LOG_FILE
fi

# Check for errors
if grep -i "error\|warning\|failed\|disconnect\|reconnect" $SERVER_LOG | grep -v "unused" | grep -v "deprecated" > /tmp/errors.txt; then
    echo "Errors/Warnings found:" >> $LOG_FILE
    cat /tmp/errors.txt >> $LOG_FILE
else
    echo "No significant errors found" >> $LOG_FILE
fi

# Final status
if grep -q "connected" $SERVER_LOG && grep -q "push" $SERVER_LOG; then
    echo "STATUS: SUCCESS - Client connected and frames streamed" >> $LOG_FILE
else
    echo "STATUS: PARTIAL or FAILED" >> $LOG_FILE
fi

# Output final log
cat $LOG_FILE

# Clean up temp files
rm -f $SERVER_LOG $CLIENT_LOG /tmp/errors.txt