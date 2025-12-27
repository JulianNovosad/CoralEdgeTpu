#!/bin/bash

# RTSP Server Test Script
# Starts the RTSP server, waits 10 seconds, then connects a client
# Runs for a total of 20 seconds, then gracefully stops both

# Set up log files
SERVER_LOG="server_test.log"
CLIENT_LOG="client_test.log"
COMBINED_LOG="test_summary.log"

echo "Starting RTSP server test..." | tee "$COMBINED_LOG"

# Function to cleanup processes
cleanup() {
    echo "Stopping processes..." | tee -a "$COMBINED_LOG"
    pkill -f "detector" 2>/dev/null || true
    pkill -f "gst-launch-1.0" 2>/dev/null || true
    pkill -f "vlc" 2>/dev/null || true
    sleep 2
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# First, kill any existing detector processes as per AGENTS.md instructions
echo "Checking for existing detector processes..." | tee -a "$COMBINED_LOG"
if pgrep detector > /dev/null; then
    echo "Found existing detector process, killing it..." | tee -a "$COMBINED_LOG"
    pkill -9 detector
    sleep 3
fi

# Verify port 8554 is available
echo "Checking if port 8554 is available..." | tee -a "$COMBINED_LOG"
if lsof -i :8554 > /dev/null 2>&1; then
    echo "Port 8554 is still in use, killing associated processes..." | tee -a "$COMBINED_LOG"
    lsof -i :8554 | grep LISTEN | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    sleep 3
fi

# Start the RTSP server (detector) in the background and log its output
echo "Starting RTSP server (detector)..." | tee -a "$COMBINED_LOG"
timeout 25s ./detector > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID" | tee -a "$COMBINED_LOG"

# Wait 10 seconds before starting the client
echo "Waiting 10 seconds before starting client..." | tee -a "$COMBINED_LOG"
sleep 10

# Start the RTSP client to connect to the server
echo "Starting RTSP client to connect to rtsp://127.0.0.1:8554/live" | tee -a "$COMBINED_LOG"
gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/live latency=100 ! decodebin ! videoconvert ! fakesink silent=false sync=false > "$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!
echo "Client started with PID: $CLIENT_PID" | tee -a "$COMBINED_LOG"

# Wait for remaining 10 seconds (total 20 seconds from server start)
echo "Waiting for remaining 10 seconds (total runtime will be 20 seconds)..." | tee -a "$COMBINED_LOG"
sleep 10

# After 20 seconds total, stop both processes
echo "20 seconds elapsed, stopping client and server..." | tee -a "$COMBINED_LOG"
kill $CLIENT_PID 2>/dev/null || true
sleep 2
kill $SERVER_PID 2>/dev/null || true

# Wait a bit for processes to stop
sleep 2

# Generate summary
echo "" | tee -a "$COMBINED_LOG"
echo "==================== TEST SUMMARY ====================" | tee -a "$COMBINED_LOG"
echo "Test duration: 20 seconds total (server started immediately, client connected after 10 seconds)" | tee -a "$COMBINED_LOG"

# Check if server was running properly
if [ -f "$SERVER_LOG" ]; then
    echo "" | tee -a "$COMBINED_LOG"
    echo "SERVER LOG (last 50 lines):" | tee -a "$COMBINED_LOG"
    tail -50 "$SERVER_LOG" | tee -a "$COMBINED_LOG"
fi

if [ -f "$CLIENT_LOG" ]; then
    echo "" | tee -a "$COMBINED_LOG"
    echo "CLIENT LOG (last 50 lines):" | tee -a "$COMBINED_LOG"
    tail -50 "$CLIENT_LOG" | tee -a "$COMBINED_LOG"
fi

# Analyze results
echo "" | tee -a "$COMBINED_LOG"
echo "==================== ANALYSIS ====================" | tee -a "$COMBINED_LOG"

SERVER_SUCCESS=$(grep -c "RTSP server started on port 8554" "$SERVER_LOG" 2>/dev/null || echo 0)
CLIENT_CONNECTED=$(grep -c "PLAY response" "$CLIENT_LOG" 2>/dev/null || echo 0)
FRAMES_RECEIVED=$(grep -c -i "frame" "$CLIENT_LOG" 2>/dev/null || echo 0)
ERRORS_IN_SERVER=$(grep -c -i "error\|failed\|exception" "$SERVER_LOG" 2>/dev/null || echo 0)
ERRORS_IN_CLIENT=$(grep -c -i "error\|failed\|exception\|warning" "$CLIENT_LOG" 2>/dev/null || echo 0)

echo "Server started successfully: $([ $SERVER_SUCCESS -gt 0 ] && echo 'YES' || echo 'NO')" | tee -a "$COMBINED_LOG"
echo "Client connected successfully: $([ $CLIENT_CONNECTED -gt 0 ] && echo 'YES' || echo 'NO')" | tee -a "$COMBINED_LOG"
echo "Frames received by client: $FRAMES_RECEIVED" | tee -a "$COMBINED_LOG"
echo "Errors in server log: $ERRORS_IN_SERVER" | tee -a "$COMBINED_LOG"
echo "Errors in client log: $ERRORS_IN_CLIENT" | tee -a "$COMBINED_LOG"

# Final assessment
echo "" | tee -a "$COMBINED_LOG"
echo "==================== FINAL ASSESSMENT ====================" | tee -a "$COMBINED_LOG"

if [ $SERVER_SUCCESS -gt 0 ] && [ $CLIENT_CONNECTED -gt 0 ]; then
    echo "✅ CLIENT CONNECTION: SUCCESS - Client successfully connected to server" | tee -a "$COMBINED_LOG"
else
    echo "❌ CLIENT CONNECTION: FAILED - Client did not connect properly" | tee -a "$COMBINED_LOG"
fi

if [ $FRAMES_RECEIVED -gt 0 ]; then
    echo "✅ FRAME STREAMING: SUCCESS - Frames were successfully streamed" | tee -a "$COMBINED_LOG"
else
    echo "⚠️  FRAME STREAMING: NO FRAMES DETECTED - May need to check client log for details" | tee -a "$COMBINED_LOG"
fi

if [ $ERRORS_IN_SERVER -eq 0 ] && [ $ERRORS_IN_CLIENT -eq 0 ]; then
    echo "✅ STABILITY: GOOD - No errors detected in server or client" | tee -a "$COMBINED_LOG"
else
    echo "⚠️  STABILITY: ISSUES DETECTED - Errors found in server or client logs" | tee -a "$COMBINED_LOG"
fi

echo "Test completed. Logs saved to $SERVER_LOG, $CLIENT_LOG, and $COMBINED_LOG" | tee -a "$COMBINED_LOG"