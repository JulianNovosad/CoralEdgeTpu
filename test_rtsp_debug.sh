#!/bin/bash

# Test script to debug RTSP stream with VLC and ffplay

echo "=== RTSP Stream Debugging Test ==="
echo ""

RTSP_URL="rtsp://localhost:8554/live"

echo "RTSP Stream URL: $RTSP_URL"
echo ""

# Function to test with ffplay using TCP transport
test_ffplay_tcp() {
    echo "Testing with ffplay using TCP transport..."
    echo "Command: ffplay -rtsp_transport tcp $RTSP_URL"
    echo "Press 'q' to quit ffplay"
    echo ""
    
    timeout 10 ffplay -rtsp_transport tcp -autoexit -nodisp "$RTSP_URL" 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ ffplay TCP test completed successfully"
    else
        echo "✗ ffplay TCP test failed"
    fi
    echo ""
}

# Function to test with ffplay using UDP transport
test_ffplay_udp() {
    echo "Testing with ffplay using UDP transport..."
    echo "Command: ffplay -rtsp_transport udp $RTSP_URL"
    echo "Press 'q' to quit ffplay"
    echo ""
    
    timeout 10 ffplay -rtsp_transport udp -autoexit -nodisp "$RTSP_URL" 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ ffplay UDP test completed successfully"
    else
        echo "✗ ffplay UDP test failed"
    fi
    echo ""
}

# Function to test with VLC using TCP transport
test_vlc_tcp() {
    if command -v vlc &> /dev/null; then
        echo "Testing with VLC using TCP transport..."
        echo "Command: vlc $RTSP_URL --demux=rtsp --rtsp-caching=100 --network-caching=100 --sout '#rtp{transport=tcp}'"
        echo "This will run VLC in verbose mode to capture debug output"
        echo ""
        
        timeout 10 vlc "$RTSP_URL" --demux=rtsp --rtsp-caching=100 --network-caching=100 --verbose=2 --intf dummy --play-and-exit 2>&1 | head -50
        if [ $? -eq 0 ]; then
            echo "✓ VLC TCP test completed"
        else
            echo "✗ VLC TCP test failed"
        fi
        echo ""
    else
        echo "VLC not found, skipping VLC test"
        echo ""
    fi
}

# Function to test with VLC using UDP transport
test_vlc_udp() {
    if command -v vlc &> /dev/null; then
        echo "Testing with VLC using UDP transport..."
        echo "Command: vlc $RTSP_URL --demux=rtsp --rtsp-caching=100 --network-caching=100"
        echo ""
        
        timeout 10 vlc "$RTSP_URL" --demux=rtsp --rtsp-caching=100 --network-caching=100 --verbose=2 --intf dummy --play-and-exit 2>&1 | head -50
        if [ $? -eq 0 ]; then
            echo "✓ VLC UDP test completed"
        else
            echo "✗ VLC UDP test failed"
        fi
        echo ""
    else
        echo "VLC not found, skipping VLC test"
        echo ""
    fi
}

# Function to check if RTSP server is running
check_rtsp_server() {
    echo "Checking if RTSP server is running on port 8554..."
    if nc -z localhost 8554; then
        echo "✓ RTSP server is running on port 8554"
        return 0
    else
        echo "✗ RTSP server is not accessible on port 8554"
        echo "Make sure the detector application is running"
        return 1
    fi
    echo ""
}

# Main test execution
if check_rtsp_server; then
    echo "Starting RTSP stream tests..."
    echo ""
    
    test_ffplay_tcp
    test_ffplay_udp
    test_vlc_tcp
    test_vlc_udp
    
    echo "=== Test Summary ==="
    echo "Check the logs from detector application for:"
    echo "- 'CLIENT CONNECTION' messages"
    echo "- 'RTP PACKET' sequences (SPS->PPS->IDR)"
    echo "- Complete SPS/PPS/IDR headers with start codes"
    echo "- NAL unit type verification"
    echo ""
    echo "Expected logs from detector:"
    echo "- 'RTSP media configuration callback triggered - new client connecting'"
    echo "- 'RTP PACKET #0: Successfully pushed SPS header...'"
    echo "- 'RTP PACKET #1: Successfully pushed PPS header...'"
    echo "- 'RTP PACKET #2: Successfully pushed latest keyframe...'"
    echo "- 'SPS/PPS header has valid start code: YES'"
    echo "- 'Keyframe is IDR-Slice...IDR: YES'"
    echo ""
    echo "If VLC still shows spinning logo, check for issues with:"
    echo "1. GStreamer RTP packetization"
    echo "2. Missing SEI (Sequence End Indicator) frames"
    echo "3. Incorrect profile level in SPS header"
    echo "4. Timing issues between headers and keyframes"
else
    echo "RTSP server not accessible. Please start the detector application first."
fi