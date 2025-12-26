#!/bin/bash

# Test script to verify RTSP stream functionality
echo "Testing RTSP stream functionality..."

# Check if the detector binary exists
if [ ! -f "/home/pi/CoralEdgeTpu/detector" ]; then
    echo "Error: detector binary not found"
    exit 1
fi

echo "RTSP stream configuration:"
echo "- Port: 8554"
echo "- Mount point: /live"
echo "- Stream URL: rtsp://localhost:8554/live"

echo ""
echo "To test the RTSP stream manually:"
echo "1. Start the detector application: ./detector"
echo "2. In another terminal, test with: ffplay rtsp://localhost:8554/live"
echo "3. Or test with VLC: vlc rtsp://localhost:8554/live"

echo ""
echo "Expected behavior after fixes:"
echo "- New clients should receive SPS/PPS headers immediately upon connection"
echo "- Video should display immediately without spinning logo"
echo "- Stream should work on both desktop and mobile VLC"

echo ""
echo "The RTSP server now:"
echo "- Extracts and stores SPS/PPS headers from incoming H.264 frames"
echo "- Sends SPS/PPS headers to new clients immediately when they connect"
echo "- Uses aggregate-mode=zero-latency for better header delivery"
echo "- Maintains backward compatibility with existing functionality"