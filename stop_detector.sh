#!/bin/bash

# This script is executed by the control_listener to stop the main detector application.

LOG_FILE="/home/pi/detector.log"
PID_FILE="/var/tmp/detector.pid"
LOCK_FILE="/var/tmp/detector.lock"

# Redirect all output from this script to the main log file.
exec >> "$LOG_FILE" 2>&1

echo "[stop_detector.sh] Received STOP command."

if [ ! -f "$PID_FILE" ]; then
    echo "[stop_detector.sh] PID file not found. Cannot stop detector. It might already be stopped."
    # As a precaution, remove the lock file if it exists, in case of a stale lock.
    rm -f "$LOCK_FILE"
    exit 1
fi

TARGET_PID=$(cat "$PID_FILE")

if [ -z "$TARGET_PID" ]; then
    echo "[stop_detector.sh] PID file is empty. Nothing to do."
    rm -f "$PID_FILE"
    rm -f "$LOCK_FILE"
    exit 0
fi

if ! ps -p "$TARGET_PID" > /dev/null; then
    echo "[stop_detector.sh] Process with PID $TARGET_PID is not running. Cleaning up stale PID and lock files."
    rm -f "$PID_FILE"
    rm -f "$LOCK_FILE"
    exit 0
fi

echo "[stop_detector.sh] Sending SIGINT (graceful shutdown) to detector process $TARGET_PID..."
kill -SIGINT "$TARGET_PID"

# Wait for a moment to allow for graceful shutdown.
sleep 1

# Check if the process is still running.
if ps -p "$TARGET_PID" > /dev/null; then
    echo "[stop_detector.sh] Process $TARGET_PID did not terminate after SIGINT. Sending SIGKILL (force kill)..."
    kill -SIGKILL "$TARGET_PID"
    sleep 0.5
fi

# Final check
if ps -p "$TARGET_PID" > /dev/null; then
    echo "[stop_detector.sh] ERROR: Failed to kill process $TARGET_PID."
else
    echo "[stop_detector.sh] Detector process $TARGET_PID terminated successfully."
fi

# Clean up PID and lock files
rm -f "$PID_FILE"
rm -f "$LOCK_FILE"

echo "[stop_detector.sh] Cleanup complete."
exit 0
