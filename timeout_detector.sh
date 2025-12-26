#!/bin/bash

# Run the detector process in the background
./build/detector &
DETECTOR_PID=$!

# Start a background process to kill the detector after 10 seconds
(
    sleep 10
    if kill -0 $DETECTOR_PID 2>/dev/null; then
        echo "10 seconds elapsed, killing detector process with PID $DETECTOR_PID"
        pkill -9 detector
        # Also kill the specific process in case pkill didn't work
        kill -9 $DETECTOR_PID 2>/dev/null
    fi
) &

# Wait for the detector process to finish
wait $DETECTOR_PID