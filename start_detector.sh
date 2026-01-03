#!/bin/bash

# This script is executed by the control_listener to start the main detector application.

LOG_FILE="/home/pi/detector.log"
LOCK_FILE="/var/tmp/detector.lock"
PID_FILE="/var/tmp/detector.pid"
DETECTOR_BINARY="/home/pi/CoralEdgeTpu/build/detector"

# Redirect all output from this script to the main log file.
exec >> "$LOG_FILE" 2>&1

echo "[start_detector.sh] Received START command."

# Ensure lock and pid file directories exist
mkdir -p /var/tmp

# Use flock to ensure only one instance of the detector is started.
# The file descriptor 200 is arbitrarily chosen.
exec 200>"$LOCK_FILE"
flock -n 200 || {
    echo "[start_detector.sh] Lock is already held. Detector process is likely already running. Ignoring START command."
    exit 1
}

# If we get here, the lock was acquired.
echo "[start_detector.sh] Acquired lock. Starting new detector instance."

# Start the detector binary in the background.
nohup "$DETECTOR_BINARY" "$@" &
DETECTOR_PID=$!

# Store the PID in a file so the stop script knows which process to kill.
echo $DETECTOR_PID > "$PID_FILE"

echo "[start_detector.sh] Detector process started with PID $DETECTOR_PID."

# The lock (file descriptor 200) will be automatically released when this script exits.
# The detector process will continue running in the background.
exit 0
