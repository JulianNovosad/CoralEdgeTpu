#!/bin/bash
# Script to kill existing detector processes and free up port 8554

echo "Checking for existing detector processes..."
pids=$(pgrep -f "detector")
if [ ! -z "$pids" ]; then
    echo "Killing existing detector processes: $pids"
    kill -9 $pids
    sleep 2
fi

echo "Checking for processes using port 8554..."
port_pids=$(sudo lsof -i :8554 -t 2>/dev/null)
if [ ! -z "$port_pids" ]; then
    echo "Killing processes using port 8554: $port_pids"
    sudo kill -9 $port_pids
    sleep 2
fi

# Also check for any GStreamer processes that might be holding the port
gst_pids=$(pgrep -f "gst" | xargs -r ps -p {} -o pid= -o cmd= 2>/dev/null | grep -i rtsp | awk '{print $1}' 2>/dev/null)
if [ ! -z "$gst_pids" ]; then
    echo "Killing RTSP/GStreamer processes: $gst_pids"
    kill -9 $gst_pids 2>/dev/null
    sleep 2
fi

echo "Port 8554 should now be available."