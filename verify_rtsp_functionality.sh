#!/bin/bash

# Comprehensive RTSP Stream Verification Script
# This script tests the complete RTSP streaming functionality

echo "=== Comprehensive RTSP Stream Verification ==="
echo ""

# Function to check if detector is running
is_detector_running() {
    pgrep -f "detector" > /dev/null
    return $?
}

# Function to start detector if not running
start_detector() {
    if ! is_detector_running; then
        echo "Starting detector application..."
        ./detector &
        DETECTOR_PID=$!
        echo "Detector started with PID: $DETECTOR_PID"
        
        # Wait a few seconds for initialization
        sleep 5
    else
        echo "Detector is already running"
    fi
}

# Function to stop detector
stop_detector() {
    if is_detector_running; then
        echo "Stopping detector application..."
        pkill -f "detector"
        sleep 2
        if is_detector_running; then
            pkill -9 -f "detector"
        fi
        echo "Detector stopped"
    fi
}

# Function to check RTSP server availability
check_rtsp_server() {
    echo "Checking RTSP server availability..."
    if nc -z localhost 8554; then
        echo "✓ RTSP server is listening on port 8554"
        return 0
    else
        echo "✗ RTSP server is not accessible on port 8554"
        return 1
    fi
}

# Function to test basic connectivity
test_connectivity() {
    echo ""
    echo "Testing basic RTSP connectivity..."
    
    # Try to connect with a simple RTSP client command
    timeout 3 bash -c "echo -e 'OPTIONS rtsp://localhost:8554/live RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: test-client\r\n\r\n' | nc localhost 8554 2>/dev/null | head -n 10" > /tmp/rtsp_test.txt
    
    if grep -q "RTSP/1.0" /tmp/rtsp_test.txt; then
        echo "✓ RTSP server responded to OPTIONS request"
    else
        echo "✗ RTSP server did not respond properly"
    fi
}

# Function to test with GStreamer
test_gstreamer() {
    echo ""
    echo "Testing RTSP stream with GStreamer..."
    
    # Test if the stream can be accessed
    timeout 5 gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/live ! decodebin ! fakesink silent=true 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ GStreamer can access and decode the RTSP stream"
    else
        echo "✗ GStreamer failed to access the RTSP stream"
    fi
}

# Function to analyze logs for expected messages
analyze_logs() {
    echo ""
    echo "Analyzing detector logs for expected behavior..."
    
    LOG_DIR="/home/pi/CoralEdgeTpu/logs"
    if [ -d "$LOG_DIR" ]; then
        echo "Checking logs in: $LOG_DIR"
        
        # Look for SPS/PPS header extraction
        SPS_LOGS=$(find "$LOG_DIR" -name "*.log" -exec grep -l "Stored SPS header" {} \; 2>/dev/null | head -n 1)
        if [ -n "$SPS_LOGS" ]; then
            echo "✓ Found SPS header extraction logs"
            grep "Stored SPS header" "$SPS_LOGS" | tail -n 3
        else
            echo "✗ No SPS header extraction logs found"
        fi
        
        # Look for PPS header extraction
        PPS_LOGS=$(find "$LOG_DIR" -name "*.log" -exec grep -l "Stored PPS header" {} \; 2>/dev/null | head -n 1)
        if [ -n "$PPS_LOGS" ]; then
            echo "✓ Found PPS header extraction logs"
            grep "Stored PPS header" "$PPS_LOGS" | tail -n 3
        else
            echo "✗ No PPS header extraction logs found"
        fi
        
        # Look for header delivery to new clients
        DELIVERY_LOGS=$(find "$LOG_DIR" -name "*.log" -exec grep -l "Successfully pushed.*header.*to new client" {} \; 2>/dev/null | head -n 1)
        if [ -n "$DELIVERY_LOGS" ]; then
            echo "✓ Found header delivery logs to new clients"
            grep "Successfully pushed.*header.*to new client" "$DELIVERY_LOGS" | tail -n 3
        else
            echo "✗ No header delivery logs to new clients found"
        fi
        
        # Look for frame type logging
        FRAME_LOGS=$(find "$LOG_DIR" -name "*.log" -exec grep -l "H264 Consumer: Frame.*NAL type" {} \; 2>/dev/null | head -n 1)
        if [ -n "$FRAME_LOGS" ]; then
            echo "✓ Found frame type logging"
            grep "H264 Consumer: Frame.*NAL type" "$FRAME_LOGS" | tail -n 5
        else
            echo "✗ No frame type logging found"
        fi
    else
        echo "✗ Log directory does not exist: $LOG_DIR"
    fi
}

# Function to check keyframe interval
check_keyframe_interval() {
    echo ""
    echo "Checking keyframe generation interval..."
    
    LOG_DIR="/home/pi/CoralEdgeTpu/logs"
    if [ -d "$LOG_DIR" ]; then
        # Look for IDR frame logs (keyframes)
        IDR_LOGS=$(find "$LOG_DIR" -name "*.log" -exec grep -l "NAL type: IDR-Slice" {} \; 2>/dev/null | head -n 1)
        if [ -n "$IDR_LOGS" ]; then
            echo "✓ Found IDR (keyframe) logs"
            # Show recent keyframes
            grep "NAL type: IDR-Slice" "$IDR_LOGS" | tail -n 5
        else
            echo "✗ No IDR (keyframe) logs found"
        fi
    fi
}

# Main execution
echo "This script will:"
echo "1. Check if detector is running"
echo "2. Test RTSP server connectivity"
echo "3. Test stream with GStreamer"
echo "4. Analyze logs for expected behavior"
echo "5. Check keyframe generation"
echo ""

echo "Starting verification tests..."
echo ""

# Start detector if needed
start_detector

# Wait for system to initialize
sleep 10

# Run all tests
check_rtsp_server
test_connectivity
test_gstreamer
analyze_logs
check_keyframe_interval

echo ""
echo "=== Verification Summary ==="
echo "After running these tests, verify that:"
echo "1. RTSP server is listening on port 8554"
echo "2. SPS/PPS headers are extracted and stored"
echo "3. Headers are delivered to new clients immediately"
echo "4. Keyframes (IDR slices) are generated at appropriate intervals"
echo "5. Frame types are logged correctly"
echo ""
echo "To test with VLC or other players:"
echo "vlc rtsp://localhost:8554/live"
echo ""
echo "Expected in logs:"
echo "- 'Stored SPS header of size X'"
echo "- 'Stored PPS header of size X'"
echo "- 'Successfully pushed SPS header of size X to new client'"
echo "- 'Successfully pushed PPS header of size X to new client'"
echo "- 'H264 Consumer: Frame X, NAL type: SPS/PPS/IDR-Slice'"

# Don't stop detector automatically as user might want to continue testing
echo ""
echo "Note: Detector is still running. Stop it manually with: pkill -f detector"