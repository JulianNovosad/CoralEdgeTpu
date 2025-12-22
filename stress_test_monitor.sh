#!/bin/bash

# Stress test monitoring script for CoralEdgeTpu
# This script monitors the stress test and provides periodic status updates

LOG_DIR="/home/pi/CoralEdgeTpu/logs"
TEST_DURATION=14400  # 4 hours in seconds
START_TIME=$(date +%s)
END_TIME=$((START_TIME + TEST_DURATION))

echo "=== CoralEdgeTpu 4-Hour Stress Test Monitoring Started ==="
echo "Start time: $(date)"
echo "End time: $(date -d @$END_TIME)"
echo "PID of detector process: $(pgrep detector)"

# Function to get FPS from camera logs
get_fps() {
    local latest_log=$(ls -t $LOG_DIR/camera/CameraCapture_*.csv | head -n 1)
    if [ -f "$latest_log" ]; then
        # Count frames in the last 5 seconds
        local end_time=$(tail -n 1 "$latest_log" | cut -d',' -f1)
        local start_time=$((end_time - 5000))  # 5 seconds ago
        local frame_count=$(awk -F',' -v start="$start_time" -v end="$end_time" '$1 >= start && $1 <= end' "$latest_log" | wc -l)
        echo $((frame_count / 5))
    else
        echo "0"
    fi
}

# Function to get IPS from TPU logs
get_ips() {
    local latest_log=$(ls -t $LOG_DIR/tpu/InferenceEngine_*.csv | head -n 1)
    if [ -f "$latest_log" ]; then
        # Count inferences in the last 5 seconds
        local end_time=$(tail -n 1 "$latest_log" | cut -d',' -f1)
        local start_time=$((end_time - 5000))  # 5 seconds ago
        local inference_count=$(awk -F',' -v start="$start_time" -v end="$end_time" '$1 >= start && $1 <= end && $3=="inference_done"' "$latest_log" | wc -l)
        echo $((inference_count / 5))
    else
        echo "0"
    fi
}

# Function to get CPU temperature
get_cpu_temp() {
    if [ -f "/sys/class/thermal/thermal_zone0/temp" ]; then
        local temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        echo $((temp / 1000))
    else
        echo "0"
    fi
}

# Function to check if detector is still running
is_detector_running() {
    pgrep detector > /dev/null
    return $?
}

# Main monitoring loop
while true; do
    CURRENT_TIME=$(date +%s)
    
    # Check if test duration has elapsed
    if [ $CURRENT_TIME -ge $END_TIME ]; then
        echo "=== 4-Hour Stress Test Completed ==="
        echo "End time: $(date)"
        break
    fi
    
    # Check if detector is still running
    if ! is_detector_running; then
        echo "=== ERROR: Detector process has stopped unexpectedly ==="
        echo "Time: $(date)"
        # Try to restart
        echo "Attempting to restart detector..."
        cd /home/pi/CoralEdgeTpu/build && ./detector --duration=14400 &
        sleep 10
        if is_detector_running; then
            echo "Detector restarted successfully"
        else
            echo "Failed to restart detector"
        fi
    fi
    
    # Get current metrics
    FPS=$(get_fps)
    IPS=$(get_ips)
    TEMP=$(get_cpu_temp)
    
    echo "=== Status Update ($(date)) ==="
    echo "Elapsed time: $(((CURRENT_TIME - START_TIME) / 60)) minutes"
    echo "Camera FPS: $FPS"
    echo "TPU IPS: $IPS"
    echo "CPU Temperature: ${TEMP}°C"
    echo "Detector running: $(is_detector_running && echo 'Yes' || echo 'No')"
    echo ""
    
    # Wait 30 minutes for next update
    sleep 1800
done

echo "Stress test monitoring completed"