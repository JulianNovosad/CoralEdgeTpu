#!/bin/bash
# HIL Stage 3 Execution Script
# Includes pre-run cleanup, hardened stabilization sequence, and thermal monitoring

# REQ-S3-008: Purge residual processes
echo "✦ Purging residual processes..."
# REQ-S3-008: Purge residual processes and free Edge TPU device
echo "✦ Purging residual processes..."
sudo pkill -9 detector || true
sleep 1 # Give a moment for processes to terminate

# Identify and kill any process holding /dev/apex_0
echo "✦ Checking for processes holding /dev/apex_0..."
APEX_HOLDERS=$(sudo lsof /dev/apex_0 | awk 'NR>1 {print $2}' | sort -u)
if [ -n "$APEX_HOLDERS" ]; then
    echo "⚠ Processes holding /dev/apex_0 found: $APEX_HOLDERS. Attempting graceful termination..."
    for PID in $APEX_HOLDERS; do
        if kill -0 $PID 2>/dev/null; then # Check if process still exists
            echo "  - Sending SIGTERM to PID $PID..."
            sudo kill -15 $PID
            sleep 2 # Give process time to terminate gracefully
            if kill -0 $PID 2>/dev/null; then # Check again if process still exists
                echo "  - PID $PID still active. Sending SIGKILL..."
                sudo kill -9 $PID
                sleep 1
            fi
        fi
    done
    APEX_HOLDERS_AFTER_KILL=$(sudo lsof /dev/apex_0 | awk 'NR>1 {print $2}' | sort -u)
    if [ -n "$APEX_HOLDERS_AFTER_KILL" ]; then
        echo "⛔ WARNING: Some processes still holding /dev/apex_0: $APEX_HOLDERS_AFTER_KILL"
    else
        echo "✅ All processes holding /dev/apex_0 terminated."
    fi
else
    echo "✅ No processes found holding /dev/apex_0."
fi

# Add a sync to ensure filesystem buffers are flushed
sync
sleep 1
# End of Edge TPU cleanup block

# Ensure the detector executable is built
if [ ! -f "./build/detector" ]; then
    echo "ERROR: Detector executable not found. Please run ./build.sh first."
    exit 1
fi

echo "✦ Initializing Stage 3 HIL Run (Hardened)..."

# Start detector in foreground to use its internal logging
./build/detector > /home/pi/.gemini/tmp/cb83514526cc8aaa463003a0036731827a1f9b1c9622a23c0d70841bfddddc06/detector_output.log 2>&1 &
DETECTOR_PID=$!

echo "✦ Detector started with PID: $DETECTOR_PID. Waiting for stabilization (10s)..."
sleep 10

# Verify PID is still alive
if ! kill -0 $DETECTOR_PID 2>/dev/null; then
    echo "⚠ ERROR: Detector process died during initialization. Check application logs."
    exit 1
fi

# Send START signal
echo "✦ Sending START signal to fire-control loop..."
# Attempt to use the control listener if active, or send directly to logic port if implemented
# For now, we assume direct START trigger is needed
echo "START 127.0.0.1" | nc -w 1 localhost 6005 || echo "⚠ WARNING: Failed to send START signal (Listener may be inactive)."

echo "✦ Run in progress (5 minutes)..."
THERMAL_LOG="thermal_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Timestamp,CPU_Temp_C" > "$THERMAL_LOG"

# REQ-S3-011: Run Stage 3 HIL test (1-5 minutes)
# Periodically check health and log CPU temperature
# The loop runs for 5 minutes (300 seconds)
for (( i=0; i<30; i++ )); do
    sleep 10 # Check every 10 seconds
    
    # REQ-S3-008: Verify Detector is still running
    if ! kill -0 $DETECTOR_PID 2>/dev/null; then
        echo "⚠ ERROR: Detector process terminated prematurely after $((i*10))s"
        break
    fi
    
    # REQ-S3-010: Log CPU Temperature
    CPU_TEMP=$(vcgencmd measure_temp | grep -oP '\d+\.\d+' || echo "N/A")
    CURRENT_TIME=$(date +%Y-%m-%dT%H:%M:%S)
    echo "$CURRENT_TIME,$CPU_TEMP" >> "$THERMAL_LOG"
    echo "✦ Progress: $((i*10))s, CPU Temp: $CPU_TEMP°C"
done

echo "✦ Terminating run..."
kill -15 $DETECTOR_PID
sleep 5
kill -9 $DETECTOR_PID 2>/dev/null || true # Force kill if still running

echo "✦ HIL Run complete."
LATEST_SESSION=$(ls -td logs/session_* | head -1)
echo "✦ Latest Session Directory: $LATEST_SESSION"
echo "✦ Thermal log saved to: $THERMAL_LOG"

# Move thermal log into the latest session directory for archival
if [ -d "$LATEST_SESSION" ]; then
    mv "$THERMAL_LOG" "$LATEST_SESSION/"
    echo "✦ Thermal log moved to $LATEST_SESSION/$THERMAL_LOG"
else
    echo "⚠ WARNING: Could not find latest session directory to archive thermal log."
fi