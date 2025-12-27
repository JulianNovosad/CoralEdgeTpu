#!/bin/bash

echo "Testing the fixed GStreamer pipeline configuration..."
echo ""

# Test the pipeline configuration that matches our RTSP server
echo "Testing pipeline with fixed configuration:"
echo "appsrc name=video_source is-live=false block=false format=GST_FORMAT_TIME caps=video/x-h264,stream-format=byte-stream,alignment=au,profile=constrained-baseline ! h264parse config-interval=1 ! rtph264pay config-interval=1 aggregate-mode=zero-latency pt=96 mtu=1400 name=pay0"
echo ""

# Test if the pipeline can be constructed without errors
gst-launch-1.0 -v appsrc name=video_source is-live=false block=false format=GST_FORMAT_TIME caps=video/x-h264,stream-format=byte-stream,alignment=au,profile=constrained-baseline ! queue ! h264parse config-interval=1 ! rtph264pay config-interval=1 aggregate-mode=zero-latency pt=96 mtu=1400 name=pay0 ! fakesink silent=true 2>&1 | head -n 30

echo ""
echo "Pipeline test completed. If no major errors were shown, the configuration is valid."
echo ""
echo "Key changes made to fix RTSP stream:"
echo "1. Changed stream-format from 'avc' to 'byte-stream' to avoid 'no codec_data' error"
echo "2. Removed deprecated session-timeout and max-connections properties"
echo "3. Kept config-interval=1 to ensure SPS/PPS headers are sent with keyframes"
echo "4. Maintained reuse-socket to prevent 'Address already in use' errors"