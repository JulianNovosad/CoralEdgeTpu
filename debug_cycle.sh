#!/bin/bash
LOG_FILE="detector.log"
RTSP_URL="rtsp://localhost:8554/live"

echo "[1/4] Cleaning old logs and killing stale processes..."
rm -f "$LOG_FILE"
pkill -f "./build/detector" || true

echo "[2/4] Starting detector in background..."
# Using stdbuf to prevent the OS from buffering logs, ensuring we see real-time issues
stdbuf -oL -eL ./build/detector > "$LOG_FILE" 2>&1 &
DETECTOR_PID=$!

echo "[3/4] Waiting 4 seconds for TPU and RTSP init..."
sleep 4

echo "[4/4] Attempting VLC connection..."
# cvlc is used to avoid GUI popups; --verbose 2 helps if you want to see VLC's side too
timeout 10s cvlc --no-gui --play-and-exit "$RTSP_URL" vlc://quit > vlc_debug.log 2>&1

echo "------------------------------------------------"
echo "DETECTOR LOG SNIPPET (Tail):"
tail -n 20 "$LOG_FILE"
echo "------------------------------------------------"

# Check if the port is even listening
if ! ss -lptn | grep -q ":8554"; then
    echo "CRITICAL: Nothing is listening on port 8554. Check for 'bind' errors in $LOG_FILE."
fi

# Cleanup
kill $DETECTOR_PID 2>/dev/null || true
