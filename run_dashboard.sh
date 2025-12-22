#!/bin/bash

# Create a named pipe
PIPE_NAME="/tmp/detector_output_pipe"
rm -f $PIPE_NAME
mkfifo $PIPE_NAME

# Start the detector in the background
timeout 30s ./build/detector > $PIPE_NAME 2>&1 &
DETECTOR_PID=$!

# Give the detector a moment to start
sleep 2

# Start the dashboard
./dashboard < $PIPE_NAME

# Clean up
rm -f $PIPE_NAME