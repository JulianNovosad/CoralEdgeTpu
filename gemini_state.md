- Current Project State:
- Fixed: Generated camera configuration, Qt errors, shutdown signal, VideoOverlayProcessor shutdown stall.
- Remaining Issue: ERROR Camera camera.cpp:1344 Request(X:C:0/2:0) is not valid.
- Hypothesis: Subtle libcamera control interaction or missing/misconfigured mandatory control for IMX708.
- Last GDB attempt: Failed due to compilation error when trying to iterate ControlList for logging.
- Last Code State:
  - src/main.cpp: High-res: 1536x864, TPU: 320x320. Dual-stream passed to CameraCapture.
  - src/camera_capture.h: Updated CameraCapture constructor, added tpu_width_, tpu_height_, tpu_stream_, main_output_queues_, tpu_output_queue_ members.
  - src/camera_capture.cpp: Updated CameraCapture constructor, implemented dual-stream configuration in setup_camera(), added buffer allocation for both streams, updated request_complete_callback to process both streams. AeEnable explicitly set to true. Problematic ControlList logging reverted.
  - src/video_overlay_processor.cpp: Fixes for shutdown stall implemented.
  - src/pipeline_structs.h: No changes yet for timed pop().
- Current Issue: Camera acquisition error (ERROR Camera camera.cpp:702 Camera in Configured state trying acquire() requiring state Available) persists even after reverting all code changes.
- Root Cause: Likely an external process holding the camera resource or an issue with the libcamera setup on the system.
- Next Steps (User):
    1. Reboot the system.
    2. Check libcamera logs (`journalctl -u libcamera` or `dmesg`).
    3. Update libcamera and kernel drivers.