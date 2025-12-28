#!/bin/bash

# RTSP Server Stress Test Script
# Rapidly connects and disconnects from the RTSP server to test stability

RTSP_URL=${1:-"rtsp://localhost:8554/stream"}
TEST_DURATION=${2:-30}  # seconds
CONCURRENT_CLIENTS=${3:-5}
LOG_FILE="rtsp_stress_test_$(date +%Y%m%d_%H%M%S).log"

echo "RTSP Stress Test Starting"
echo "URL: $RTSP_URL"
echo "Duration: ${TEST_DURATION}s"
echo "Concurrent clients: $CONCURRENT_CLIENTS"
echo "Log file: $LOG_FILE"
echo "----------------------------------------"

# Function to run a single client test
run_client() {
    local client_id=$1
    local start_time=$(date +%s)
    local end_time=$((start_time + $TEST_DURATION))
    
    while [ $(date +%s) -lt $end_time ]; do
        echo "$(date): Client $client_id - Connecting to $RTSP_URL" | tee -a $LOG_FILE
        
        # Use gst-launch-1.0 to connect to the RTSP stream briefly
        timeout 3 gst-launch-1.0 -q rtspsrc location="$RTSP_URL" latency=100 ! decodebin ! videoconvert ! fakesink 2>/dev/null
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "$(date): Client $client_id - Successfully connected/disconnected" | tee -a $LOG_FILE
        elif [ $exit_code -eq 124 ]; then  # timeout exit code
            echo "$(date): Client $client_id - Connection timed out (expected)" | tee -a $LOG_FILE
        else
            echo "$(date): Client $client_id - Connection failed with exit code $exit_code" | tee -a $LOG_FILE
        fi
        
        # Brief pause before next connection
        sleep 0.5
    done
    
    echo "$(date): Client $client_id - Test completed" | tee -a $LOG_FILE
}

# Start concurrent clients
echo "Starting $CONCURRENT_CLIENTS concurrent clients..."
for i in $(seq 1 $CONCURRENT_CLIENTS); do
    run_client $i &
done

# Wait for all background processes to complete
wait

echo "----------------------------------------"
echo "RTSP Stress Test Completed"
echo "Check $LOG_FILE for detailed results"