#!/bin/bash

# CoralEdgeTpu System Cleanup Script
# Aggressively cleans up all processes, resources, and temporary files

set -e

echo "=== CoralEdgeTpu System Cleanup ==="

# Function to safely kill processes
kill_process() {
    local process_name=$1
    local signal=${2:-TERM}
    
    echo "Looking for $process_name processes..."
    pids=$(pgrep -f "$process_name" 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        echo "Found processes: $pids"
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Sending SIG$signal to process $pid..."
                kill -$signal "$pid" 2>/dev/null || true
                
                # Wait a bit for graceful termination
                if [ "$signal" = "TERM" ]; then
                    sleep 3
                    if kill -0 "$pid" 2>/dev/null; then
                        echo "Process $pid still running, sending SIGKILL..."
                        kill -KILL "$pid" 2>/dev/null || true
                    fi
                fi
            else
                echo "Process $pid not running"
            fi
        done
    else
        echo "No $process_name processes found"
    fi
}

# 1. Kill detector processes
echo "1. Terminating detector processes..."
kill_process "detector"
kill_process "integrated_system"
kill_process "dashboard"

# 2. Kill test processes
echo "2. Terminating test processes..."
kill_process "camera_isolation_test"
kill_process "inference_test_no_logging"
kill_process "raw_tpu_test"
kill_process "tpu_diagnostic"
kill_process "list_tpu_devices"
kill_process "check_edgetpu_version"
kill_process "config_loader_test"
kill_process "servo_test"

# 3. Kill any remaining related processes
echo "3. Terminating other related processes..."
kill_process "CoralEdgeTpu"
kill_process "EdgeTpu"

# 4. Clean up temporary files and directories
echo "4. Cleaning up temporary files..."
rm -f ./detector_pipe 2>/dev/null || true
rm -f /tmp/detector* 2>/dev/null || true
rm -f *.log 2>/dev/null || true

# Clean up build artifacts
echo "5. Cleaning up build artifacts..."
rm -rf ./build 2>/dev/null || true

# Clean up temporary directories
rm -rf ./tmp 2>/dev/null || true

# 6. Clean up shared memory segments
echo "6. Cleaning up shared memory..."
ipcs -m | grep -E "0x[0-9a-f]+" | awk '{print $2}' | while read shmid; do
    ipcrm -m "$shmid" 2>/dev/null || true
done

# 7. Clean up semaphore arrays
echo "7. Cleaning up semaphores..."
ipcs -s | grep -E "0x[0-9a-f]+" | awk '{print $2}' | while read semid; do
    ipcrm -s "$semid" 2>/dev/null || true
done

# 8. Unmount any mounted filesystems related to the project
echo "8. Checking for mounted filesystems..."
mount | grep -i "coral\|edgetpu" | while read line; do
    echo "Found mount: $line"
    # We won't automatically unmount for safety, but we'll report it
done

# 9. Reset Edge TPU if needed (unload and reload the driver)
echo "9. Checking Edge TPU status..."
if lsmod | grep -q "pcie_edgetpu"; then
    echo "Edge TPU driver loaded"
    # We won't automatically unload for safety, but we'll report it
else
    echo "Edge TPU driver not loaded or not present"
fi

# 10. Clean up any dangling file descriptors
echo "10. Checking for dangling file descriptors..."
lsof | grep -E "(coral|edgetpu|detector)" | while read line; do
    echo "Potential dangling descriptor: $line"
done

# 11. Clean up ZeroMQ endpoints
echo "11. Cleaning up ZeroMQ endpoints..."
rm -f /tmp/zmq_* 2>/dev/null || true

# 12. Final system check
echo "12. Performing final system check..."

# Check for any remaining processes
echo "Checking for remaining detector processes:"
pgrep -fa "detector|integrated_system|CoralEdgeTpu" || echo "No detector processes found"

echo ""
echo "=== Cleanup Complete ==="
echo "All known CoralEdgeTpu processes and resources have been terminated."
echo "Please verify that no critical system processes were affected."

exit 0