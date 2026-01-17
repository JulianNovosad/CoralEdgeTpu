# CoralEdgeTpu DRM Display Issues - Expert Audit Preparation

## Current Problem Statement
Heavy artifacting and buffering issues persist in HDMI output:
- RGB stripes and color banding
- Severe latency buildup over time
- Degraded video quality despite multiple optimization attempts
- System responsiveness degrades quickly

## Timeline of Attempts Made

### Initial State (Working)
- Had functional DRM display with camera feed visible on HDMI
- Acceptable performance with minor dark image and latency issues

### Issue Onset
- DRM regression occurred, functionality lost
- Black screen output initially

### Fix Attempts and Results

#### Attempt 1: Basic DRM Restoration
- Fixed connector/CRTC hard-coded IDs (32/91)
- Restored framebuffer creation and page flipping
- Result: Got video output but with latency/artifacting

#### Attempt 2: Format Conversion Fixes
- Identified pipeline: YUV420 → BGR → ARGB32
- Fixed format handling in DRM renderer
- Result: Same issues persisted

#### Attempt 3: Memory Management
- Implemented double buffering
- Added proper resource cleanup
- Fixed memory leaks
- Result: Still had latency and artifacting

#### Attempt 4: Minimal Single Buffer Approach
- Simplified to single framebuffer
- Direct BGR to ARGB conversion
- Added timing controls
- Result: Issues remained

#### Attempt 5: Diagnostic Implementation
- Added comprehensive logging
- Memory and system health monitoring
- Queue depth tracking
- Result: Identified queue backpressure but fixes ineffective

#### Attempt 6: Queue Size Reduction
- Dramatically reduced queue sizes (8/16/8 elements)
- Reduced buffer pools (50/50/8 buffers)
- Result: Made issues worse - increased frame drops and stalls

#### Attempt 7: Intelligent Frame Dropping
- Implemented adaptive frame dropping at 75% queue capacity
- Added pressure-based flow control
- Restored reasonable queue sizes (64/32/16)
- Result: Some improvement but artifacting persists

## Current Code State

### Key Components Modified:
1. `src/drm_display.cpp` - Multiple iterations of rendering logic
2. `src/image_processor.cpp` - Added frame dropping logic
3. `src/application.cpp` - Extended timeout, modified buffer sizes
4. `src/pipeline_structs.h` - Adjusted queue capacities
5. `src/lockfree_queue.h` - Added capacity monitoring

### Pipeline Architecture:
```
Camera (YUV420) 
→ Image Processor (YUV420→BGR) 
→ Visualization Processor 
→ DRM Renderer (BGR→ARGB32) 
→ HDMI Display (1280x720)
```

## Observed Symptoms
- RGB color stripes/banding in output
- Progressive latency increase during runtime
- Heavy artifacting that worsens over time
- System becomes unresponsive after ~5-10 minutes
- Memory pressure indicators in logs

## Diagnostic Data Available
- Full application logs with timestamps
- Queue depth monitoring output
- Frame processing timing information
- System resource usage metrics
- Page flip success/failure rates

## Recommended Expert Audit Areas

### 1. Memory Management
- GPU memory allocation patterns
- Framebuffer lifecycle management
- Buffer pooling efficiency

### 2. Timing and Synchronization
- Page flip timing vs display refresh rate
- Pipeline stage synchronization
- Buffer availability coordination

### 3. Color Space Conversion
- YUV420 to BGR conversion accuracy
- BGR to ARGB32 mapping correctness
- Bit depth and color range handling

### 4. Pipeline Flow Control
- Queue management strategies
- Backpressure handling effectiveness
- Frame dropping algorithms

### 5. Hardware Integration
- DRM/KMS driver interaction
- Display controller configuration
- Memory-mapped I/O performance

## Next Steps for Expert
1. Review current implementation in `src/drm_display.cpp`
2. Analyze pipeline flow in `src/image_processor.cpp` 
3. Examine queue implementations in `src/lockfree_queue.h`
4. Check system logs in `/home/pi/CoralEdgeTpu/logs/`
5. Profile memory usage and allocation patterns
6. Test with different timing configurations
7. Verify color space conversion mathematics

The system shows all the right behaviors (page flips succeeding, frames being processed) but the output quality is severely degraded. An expert familiar with DRM/KMS, embedded graphics, and real-time pipeline optimization would be best positioned to identify the root cause.