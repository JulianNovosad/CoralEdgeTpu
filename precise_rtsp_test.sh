#!/bin/bash

# Script to test RTSP server with precise timing
# Server starts immediately, client connects after 10 seconds, everything stops after 20 seconds

# Create log files with timestamps
TIMESTAMP=$(date +%s)
SERVER_LOG="/tmp/server_${TIMESTAMP}.log"
CLIENT_LOG="/tmp/client_${TIMESTAMP}.log"
COMBINED_LOG="/tmp/combined_${TIMESTAMP}.log"

echo "$(date): Starting RTSP server test with precise timing" > $COMBINED_LOG
echo "$(date): Server log: $SERVER_LOG" >> $COMBINED_LOG
echo "$(date): Client log: $CLIENT_LOG" >> $COMBINED_LOG

# Function to cleanup processes
cleanup() {
    echo "$(date): Cleaning up processes..." >> $COMBINED_LOG
    pkill -f "detector.*--rtsp" 2>/dev/null || true
    pkill -f "gst-launch-1.0" 2>/dev/null || true
    sleep 2
    pkill -9 -f "detector.*--rtsp" 2>/dev/null || true
    pkill -9 -f "gst-launch-1.0" 2>/dev/null || true
    echo "$(date): Cleanup completed" >> $COMBINED_LOG
}

# Set up signal trap
trap cleanup EXIT

# Start the RTSP server in the background
echo "$(date): Starting RTSP server..." >> $COMBINED_LOG
/home/pi/CoralEdgeTpu/build/detector --rtsp > $SERVER_LOG 2>&1 &
SERVER_PID=$!

# Wait for server to initialize
sleep 5

# Verify server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "$(date): RTSP server started successfully (PID: $SERVER_PID)" >> $COMBINED_LOG
else
    echo "$(date): ERROR: RTSP server failed to start" >> $COMBINED_LOG
    exit 1
fi

# Wait exactly 10 seconds before starting client (total ~15 seconds since start)
echo "$(date): Waiting 10 seconds before starting client..." >> $COMBINED_LOG
sleep 10

echo "$(date): Starting RTSP client to connect to rtsp://127.0.0.1:8554/live" >> $COMBINED_LOG
timeout 5s gst-launch-1.0 -v rtspsrc location=rtsp://127.0.0.1:8554/live protocols=tcp ! rtph264depay ! avdec_h264 ! fakesink silent=false sync=false > $CLIENT_LOG 2>&1 &
CLIENT_PID=$!

# Wait for the remainder of the 20-second total runtime (about 5 more seconds)
sleep 5

# Stop both processes
echo "$(date): Stopping client (PID: $CLIENT_PID) and server (PID: $SERVER_PID)" >> $COMBINED_LOG
kill $CLIENT_PID 2>/dev/null || true
kill $SERVER_PID 2>/dev/null || true

sleep 2

# Generate summary
echo "" >> $COMBINED_LOG
echo "=== TEST SUMMARY ===" >> $COMBINED_LOG
echo "Test Duration: 20 seconds total" >> $COMBINED_LOG
echo "Client Start Time: ~10 seconds after server start" >> $COMBINED_LOG
echo "" >> $COMBINED_LOG

# Check if client connected successfully
if grep -q "connected\|PLAY\|SET" $SERVER_LOG; then
    echo "✅ Client connected successfully" >> $COMBINED_LOG
    CONNECTED="YES"
else
    echo "❌ Client connection failed" >> $COMBINED_LOG
    CONNECTED="NO"
fi

# Check if frames were received by client
if grep -q "push\|buffer\|frame" $CLIENT_LOG; then
    echo "✅ Client received frames" >> $COMBINED_LOG
    FRAMES_RECEIVED="YES"
else
    echo "❌ Client did not receive frames" >> $COMBINED_LOG
    FRAMES_RECEIVED="NO"
fi

# Check for errors
SERVER_ERRORS=$(grep -i "error\|warning\|disconnect\|failed" $SERVER_LOG | grep -v "unused\|deprecated\|stream-format" | wc -l)
CLIENT_ERRORS=$(grep -i "error\|warning\|disconnect\|failed" $CLIENT_LOG | wc -l)

if [ $SERVER_ERRORS -gt 0 ] || [ $CLIENT_ERRORS -gt 0 ]; then
    echo "⚠️  Errors or warnings detected:" >> $COMBINED_LOG
    if [ $SERVER_ERRORS -gt 0 ]; then
        echo "   Server errors: $SERVER_ERRORS" >> $COMBINED_LOG
    fi
    if [ $CLIENT_ERRORS -gt 0 ]; then
        echo "   Client errors: $CLIENT_ERRORS" >> $COMBINED_LOG
    fi
else
    echo "✅ No significant errors detected" >> $COMBINED_LOG
fi

echo "" >> $COMBINED_LOG
echo "=== SERVER LOG (Last 50 lines) ===" >> $COMBINED_LOG
tail -n 50 $SERVER_LOG >> $COMBINED_LOG

echo "" >> $COMBINED_LOG
echo "=== CLIENT LOG (Last 50 lines) ===" >> $COMBINED_LOG
tail -n 50 $CLIENT_LOG >> $COMBINED_LOG

# Output final results
cat $COMBINED_LOG