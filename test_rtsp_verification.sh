#!/bin/bash

# Test script to verify RTSP stream functionality with external clients

echo "=== RTSP Stream Verification Test ==="
echo ""

# Check if the detector binary exists
if [ ! -f "/home/pi/CoralEdgeTpu/detector" ]; then
    echo "Error: detector binary not found"
    exit 1
fi

echo "RTSP Configuration:"
echo "- Port: 8554"
echo "- Mount point: /live"
echo "- Stream URL: rtsp://localhost:8554/live"
echo ""

# Function to test with GStreamer
test_with_gstreamer() {
    echo "Testing RTSP stream with GStreamer..."
    echo "Command: gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/live ! decodebin ! videoconvert ! xvimagesink sync=false"
    echo "This will attempt to play the RTSP stream using GStreamer."
    echo "Press Ctrl+C to stop the test."
    echo ""
    
    # Use timeout to prevent hanging
    timeout 10 gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/live ! decodebin ! fakesink silent=true 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ GStreamer test successful - stream is accessible"
    else
        echo "✗ GStreamer test failed - stream may not be accessible"
    fi
    echo ""
}

# Function to test with ffplay
test_with_ffplay() {
    echo "Testing RTSP stream with ffplay..."
    echo "Command: ffplay -v debug rtsp://localhost:8554/live"
    echo "This will attempt to play the RTSP stream using ffplay."
    echo "Press 'q' to stop the test."
    echo ""
    
    # Test with a short timeout to check if connection is possible
    timeout 5 ffplay -v quiet -loglevel error -t 3 -i rtsp://localhost:8554/live 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ ffplay test successful - stream is accessible"
    else
        echo "✗ ffplay test failed - stream may not be accessible"
    fi
    echo ""
}

# Function to test with VLC (if available)
test_with_vlc() {
    if command -v vlc &> /dev/null; then
        echo "Testing RTSP stream with VLC..."
        echo "Command: vlc rtsp://localhost:8554/live --intf dummy --play-and-exit"
        echo ""
        
        timeout 5 vlc rtsp://localhost:8554/live --intf dummy --play-and-exit --quiet 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✓ VLC test successful - stream is accessible"
        else
            echo "✗ VLC test failed - stream may not be accessible"
        fi
        echo ""
    else
        echo "VLC not found, skipping VLC test"
        echo ""
    fi
}

# Function to check if port is open
check_port() {
    echo "Checking if RTSP port 8554 is open..."
    if nc -z localhost 8554; then
        echo "✓ Port 8554 is open"
    else
        echo "✗ Port 8554 is not accessible"
    fi
    echo ""
}

# Main test execution
echo "=== Starting RTSP Stream Tests ==="
check_port
test_with_gstreamer
test_with_ffplay
test_with_vlc

echo "=== Test Summary ==="
echo "After running these tests, check the detector application logs for:"
echo "- SPS/PPS header extraction and storage messages"
echo "- SPS/PPS header delivery to new clients"
echo "- NAL unit type logging (IDR, SPS, PPS)"
echo ""
echo "To run the detector application with the RTSP server:"
echo "1. Start the detector: ./detector"
echo "2. Run this test script in another terminal"
echo "3. Check logs in /home/pi/CoralEdgeTpu/logs/"
echo ""
echo "Expected successful log messages:"
echo "- 'Stored SPS header of size X'"
echo "- 'Stored PPS header of size X'"
echo "- 'Successfully pushed SPS header of size X to new client'"
echo "- 'Successfully pushed PPS header of size X to new client'"
echo "- 'H264 Consumer: Frame X, NAL type: SPS/PPS/IDR-Slice'"