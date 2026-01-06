#!/bin/bash
# HIL Stage 2 Execution Script
# Target: 120 seconds of telemetry capture

echo "✦ Initializing Stage 2 HIL Run..."

# Start detector
./build/detector > /tmp/detector_hil.log 2>&1 &
DETECTOR_PID=$!

echo "✦ Detector started with PID: $DETECTOR_PID. Waiting for stabilization (5s)..."
sleep 5

# Send START signal
echo "✦ Sending START signal to fire-control loop..."
echo "START 127.0.0.1" | nc -w 1 localhost 6005 || echo "⚠ Failed to send START signal."

echo "✦ Run in progress (120s)..."
sleep 120

echo "✦ Terminating run..."
kill -15 $DETECTOR_PID
sleep 3
kill -9 $DETECTOR_PID 2>/dev/null || true

echo "✦ HIL Run complete."
LATEST_SESSION=$(ls -td logs/session_* | head -1)
echo "✦ Latest Session Directory: $LATEST_SESSION"
