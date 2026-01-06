# HEADER VERIFICATION PLAN

**Status:** DOCTRINE-MANDATED ARTIFACT
**Date:** 2026-01-06

## 1. Objective
To ensure no "hallucinated" APIs are used and that the build environment matches the code assumptions.

## 2. Native Headers to Verify
*   `<libcamera/camera.h>`: Version and API compatibility.
*   `"edgetpu_c.h"`: Verify `edgetpu_create_delegate` signature.
*   `"tensorflow/lite/interpreter.h"`: Verify `tflite::Interpreter` API.
*   `<sys/mman.h>`: For zero-copy buffers.
*   `<linux/dma-buf.h>`: For DMA file descriptors.

## 3. Verification Script (Pre-Build)
A script `tools/verify_headers.py` will be created to:
1.  Scan source code for `#include`.
2.  Attempt to locate these headers in the system include paths.
3.  Check for critical symbols (e.g., `libcamera::CameraManager`).
4.  Generate a timestamped `verified_headers.log`.

## 4. Source Annotation
All `.cpp` files must contain:
```cpp
// Verified headers: <list of critical headers>
// Verification timestamp: 2026-01-06 ...
```
(This will be added manually or via script during implementation phase).
