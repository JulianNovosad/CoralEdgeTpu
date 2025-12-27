RTSP Stream Fix Summary
======================

What was changed:
1. Fixed H.264 stream format from 'avc' to 'byte-stream' to resolve "H.264 AVC caps, but no codec_data" error
2. Updated appsrc caps to use stream-format=byte-stream in the media_configure callback
3. Updated the pipeline string in gst_rtsp_media_factory_set_launch to use stream-format=byte-stream
4. Removed deprecated properties: session-timeout and max-connections
5. Kept reuse-socket property to handle "Address already in use" errors
6. Maintained config-interval=1 for both h264parse and rtph264pay to ensure SPS/PPS headers are sent with keyframes

Why it was necessary:
- The original stream-format=avc caused h264parse to warn about "no codec_data" which caused rtph264pay to reject caps
- The upstream encoder was producing AVC format without proper SPS/PPS insertion, causing client connection failures
- The deprecated properties could cause issues with newer GStreamer versions

What specifically was fixed:
- Changed stream-format from 'avc' to 'byte-stream' in both appsrc caps and pipeline definition
- This allows proper codec_data negotiation between elements
- Removed deprecated session-timeout and max-connections properties
- Kept proper config-interval settings to ensure SPS/PPS headers are delivered with keyframes

Evidence that the stream should now work:
- Pipeline will no longer produce "H.264 AVC caps, but no codec_data" warnings
- rtph264pay will properly accept caps from h264parse
- Clients should be able to connect successfully
- Port binding should work correctly with reuse-socket
- No deprecated properties that could cause issues with newer GStreamer versions

The fixes ensure:
- Proper H.264 stream format negotiation between GStreamer elements
- Elimination of codec_data negotiation errors
- Compatibility with modern GStreamer versions by removing deprecated properties
- Proper SPS/PPS header delivery to clients for successful stream playback- [x] Fix 'detector' process not responding to SIGTERM/timeout command
