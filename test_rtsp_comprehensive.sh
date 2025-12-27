#!/bin/bash

echo "Starting comprehensive RTSP server test..."
echo "Timestamp: $(date)"

SUCCESS=0
ATTEMPT=0
MAX_ATTEMPTS=3

while [ $ATTEMPT -lt $MAX_ATTEMPTS ] && [ $SUCCESS -eq 0 ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "Attempt $ATTEMPT of $MAX_ATTEMPTS"
    
    # Start detector in background
    DETECTOR_LOG="/tmp/detector_log_$$.txt"
    timeout 45s ./build/detector > "$DETECTOR_LOG" 2>&1 &
    DETECTOR_PID=$!
    
    echo "Detector started with PID $DETECTOR_PID"
    
    # Wait a bit for log to start being written
    sleep 3
    
    # Check for RTSP server startup and first keyframe
    START_TIME=$(date +%s)
    TIMEOUT=35  # 35 second timeout for initialization
    
    RTSP_READY=0
    KEYFRAME_READY=0
    
    while [ $(( $(date +%s) - START_TIME )) -lt $TIMEOUT ]; do
        # Check if RTSP server is listening on port 8554
        if netstat -tuln | grep -q ":8554 "; then
            RTSP_READY=1
            echo "RTSP server is listening on port 8554"
        fi
        
        # Check if first keyframe (IDR) has been produced in the log
        if [ -f "$DETECTOR_LOG" ] && grep -q "NAL Type=IDR-Slice" "$DETECTOR_LOG"; then
            KEYFRAME_READY=1
            echo "First keyframe (IDR) detected in logs"
            break  # Both conditions met
        fi
        
        sleep 1
        
        # Check if detector process is still running
        if ! kill -0 $DETECTOR_PID 2>/dev/null; then
            echo "Detector process died during initialization"
            break
        fi
    done
    
    if [ $RTSP_READY -eq 1 ] && [ $KEYFRAME_READY -eq 1 ]; then
        echo "Both RTSP server and keyframe are ready, attempting client connection..."
        
        # Test RTSP connection
        CONNECTION_RESULT=$(timeout 10 gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/live latency=0 ! decodebin ! fakesink num-buffers=2 2>&1)
        CONNECTION_EXIT_CODE=$?
        
        if [ $CONNECTION_EXIT_CODE -eq 0 ]; then
            echo "SUCCESS: RTSP client connected and received frames"
            echo "Connection result: $CONNECTION_RESULT"
            SUCCESS=1
        else
            echo "FAILURE: RTSP client connection failed"
            echo "Connection result: $CONNECTION_RESULT"
        fi
    else
        echo "FAILURE: RTSP server or keyframe not ready in time"
        echo "RTSP ready: $RTSP_READY, Keyframe ready: $KEYFRAME_READY"
    fi
    
    # Clean up detector process
    if kill -0 $DETECTOR_PID 2>/dev/null; then
        kill $DETECTOR_PID 2>/dev/null
        sleep 2
        if kill -0 $DETECTOR_PID 2>/dev/null; then
            kill -9 $DETECTOR_PID 2>/dev/null
        fi
    fi
    
    # Show logs if available
    if [ -f "$DETECTOR_LOG" ]; then
        echo "Detector logs (last 15 lines):"
        tail -15 "$DETECTOR_LOG"
        rm "$DETECTOR_LOG"
    fi
    
    if [ $SUCCESS -eq 0 ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
        echo "Retrying in 1 second..."
        sleep 1
    fi
done

if [ $SUCCESS -eq 1 ]; then
    echo "Test PASSED: RTSP server is working correctly with client connections"
    echo "Timestamp: $(date)"
    exit 0
else
    echo "Test FAILED: Could not establish successful RTSP connection after $MAX_ATTEMPTS attempts"
    echo "Timestamp: $(date)"
    exit 1
fi
