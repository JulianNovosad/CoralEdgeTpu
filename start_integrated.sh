#!/bin/bash

# Integrated startup script for CoralEdgeTpu detector with dashboard
# This script creates a FIFO pipe, starts the detector writing to the pipe,
# and starts the dashboard reading from the pipe.

PIPE_PATH="/tmp/detector_pipe"
PID_FILE="/tmp/integrated_system.pid"

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    
    # Kill detector process if running
    if [ -f "/tmp/detector.pid" ]; then
        DETECTOR_PID=$(cat /tmp/detector.pid)
        if kill -0 $DETECTOR_PID 2>/dev/null; then
            echo "Killing detector process (PID: $DETECTOR_PID)"
            kill $DETECTOR_PID
            sleep 1
            # Force kill if still running
            if kill -0 $DETECTOR_PID 2>/dev/null; then
                kill -9 $DETECTOR_PID
            fi
        fi
        rm -f /tmp/detector.pid
    fi
    
    # Kill dashboard process if running
    if [ -f "/tmp/dashboard.pid" ]; then
        DASHBOARD_PID=$(cat /tmp/dashboard.pid)
        if kill -0 $DASHBOARD_PID 2>/dev/null; then
            echo "Killing dashboard process (PID: $DASHBOARD_PID)"
            kill $DASHBOARD_PID
            sleep 1
            # Force kill if still running
            if kill -0 $DASHBOARD_PID 2>/dev/null; then
                kill -9 $DASHBOARD_PID
            fi
        fi
        rm -f /tmp/dashboard.pid
    fi
    
    # Remove FIFO pipe
    if [ -p "$PIPE_PATH" ]; then
        rm -f "$PIPE_PATH"
    fi
    
    # Remove PID file
    rm -f $PID_FILE
    
    echo "Cleanup complete."
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGINT SIGTERM

# Check if already running
if [ -f $PID_FILE ]; then
    OLD_PID=$(cat $PID_FILE)
    if kill -0 $OLD_PID 2>/dev/null; then
        echo "Integrated system already running (PID: $OLD_PID). Use 'kill $OLD_PID' to stop it."
        exit 1
    else
        # Stale PID file, remove it
        rm -f $PID_FILE
    fi
fi

# Save our PID
echo $$ > $PID_FILE

# Create FIFO pipe
echo "Creating FIFO pipe at $PIPE_PATH"
rm -f "$PIPE_PATH"
mkfifo "$PIPE_PATH"

# Start integrated system in background
echo "Starting integrated system..."
cd /home/pi/CoralEdgeTpu/build
./integrated_system > "$PIPE_PATH" 2>&1 &
DETECTOR_PID=$!
echo $DETECTOR_PID > /tmp/detector.pid
echo "Integrated system started with PID: $DETECTOR_PID"

# Wait a moment for detector to initialize
sleep 2

# Start dashboard in background
echo "Starting dashboard..."
cd /home/pi/CoralEdgeTpu/build
./dashboard < "$PIPE_PATH" &
DASHBOARD_PID=$!
echo $DASHBOARD_PID > /tmp/dashboard.pid
echo "Dashboard started with PID: $DASHBOARD_PID"

echo "Integrated system started. PID: $$"
echo "Integrated system PID: $DETECTOR_PID"
echo "Dashboard PID: $DASHBOARD_PID"
echo ""
echo "Press Ctrl+C to stop."

# Wait indefinitely
wait $DETECTOR_PID