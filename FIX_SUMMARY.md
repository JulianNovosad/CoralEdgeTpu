# Fix Summary: Segmentation Fault in CameraCapture

## Problem
The application was crashing with a segmentation fault when trying to start the CameraCapture module. The issue was in the recovery mechanism where the recovery thread was started immediately when the Application object was constructed, but the modules weren't initialized until later in the `run()` method.

The recovery thread would check if modules were running and attempt to recover them before they were even started, leading to a situation where:
1. CameraCapture object was created but not yet started
2. Recovery thread detected it wasn't running and tried to recover it
3. Recovery process reset the CameraCapture object but failed to recreate it properly
4. When `start_modules()` was later called, it tried to call `start()` on a null pointer, causing the segmentation fault

## Solution
Modified the Application class to delay starting the recovery thread until after modules are initialized and started:

### Changes Made:
1. Added a new flag `recovery_enabled_` in `application.h` to control when recovery is active
2. Modified the Application constructor to not start the recovery thread immediately
3. Added logic in the `run()` method to start the recovery thread after modules are initialized and started
4. Modified the recovery thread function to check if recovery is enabled before attempting recovery

### Files Modified:
- `src/application.h` - Added `recovery_enabled_` flag
- `src/application.cpp` - Modified constructor and added recovery thread start after module initialization

## Result
The segmentation fault is resolved. The application now runs successfully and processes camera frames without crashing.