#!/bin/bash

echo "Testing GStreamer pipeline configuration with the fixed parameters..."

# Test the pipeline configuration that matches our RTSP server
echo "Testing pipeline with fixed configuration:"
echo "appsrc name=video_source is-live=true block=false format=GST_FORMAT_TIME caps=video/x-h264,stream-format=avc,alignment=au,profile=constrained-baseline ! h264parse config-interval=1 ! rtph264pay config-interval=1 aggregate-mode=zero-latency pt=96 mtu=1400 name=pay0"

# Test if the pipeline can be constructed without errors
gst-launch-1.0 -v appsrc name=video_source is-live=true block=false format=GST_FORMAT_TIME caps=video/x-h264,stream-format=avc,alignment=au,profile=constrained-baseline ! h264parse config-interval=1 ! rtph264pay config-interval=1 aggregate-mode=zero-latency pt=96 mtu=1400 name=pay0 ! fakesink silent=true 2>&1 | head -n 20

echo ""
echo "Pipeline test completed. If no major errors were shown, the configuration is valid."

echo ""
echo "Key changes made to fix RTSP stream:"
echo "1. Changed rtph264pay config-interval from 3 to 1 to ensure SPS/PPS headers are sent with keyframes"
echo "2. Added dummy keyframe mechanism to ensure pipeline can preroll without live data"
echo "3. Ensured static caps are set properly for consistent format negotiation"
echo "4. Improved the send_latest_keyframe function to handle cases where no keyframe exists"