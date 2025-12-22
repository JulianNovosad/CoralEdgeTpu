# End-to-End Test Plan for Enhanced Visualization and Logging

## Overview
This test plan validates the enhanced visualization and logging features implemented in the CoralEdgeTpu system. The enhancements include:
1. Visual overlay for detection bounding boxes and inner fraction centers
2. Enhanced distance smoothing verification logging
3. Servo actuation timing verification
4. Enhanced angular error verification logging

## Prerequisites
- Raspberry Pi 5 with Google Coral M.2 Edge TPU
- Camera properly connected and configured
- Servo motor connected to PCA9685 controller
- All dependencies installed as per build instructions

## Test Procedures

### 1. Visual Overlay Functionality Test
**Objective:** Verify that the visualization system correctly displays detection bounding boxes and inner fraction centers.

**Steps:**
1. Start the detector application
2. Position objects of various sizes within the camera's field of view
3. Observe the visualization output on the display or network stream
4. Verify that:
   - Detection bounding boxes appear in orange around detected objects
   - Inner fraction bounding boxes appear in blue within the detection boxes
   - Blue dots appear at the center of inner fraction boxes
   - White lines connect the center crosshair to inner fraction centers
   - Grid lines are visible for coordinate reference
   - Visualization info text shows correct dimensions and detection count

**Expected Results:**
- All visual elements appear correctly positioned and colored
- Inner fraction boxes are proportionally smaller than detection boxes based on the inner_fraction parameter
- Visualizations update in real-time as objects move

### 2. Distance Smoothing Logging Test
**Objective:** Verify that distance smoothing logging provides detailed information about the smoothing process.

**Steps:**
1. Start the detector application with logging enabled
2. Position objects at various known distances from the camera
3. Monitor the application logs for DISTANCE_SMOOTHING entries
4. Verify that:
   - Logs contain class ID, raw distance, corrected distance, and smoothed distance
   - Window count information is accurate
   - Window values show the history of distance estimates
   - Smoothed distances are reasonable compared to raw measurements

**Expected Results:**
- DISTANCE_SMOOTHING log entries appear for each detection
- Window values show a rolling history of 10 distance estimates
- Smoothed distances are more stable than raw measurements

### 3. Servo Actuation Timing Test
**Objective:** Verify that servo actuation timing follows the required sequence and is properly logged.

**Steps:**
1. Start the detector application with servo connected
2. Present objects that trigger servo actuation
3. Monitor the application logs for SERVO_TIMING entries
4. Verify that:
   - Queue to execution times are reasonable
   - Cooldown periods are properly enforced (minimum 300ms between actuations)
   - Execution times are consistent
   - Servo commands follow the 50ms → 30ms → 300ms timing sequence

**Expected Results:**
- SERVO_TIMING log entries show proper timing measurements
- Cooldown periods prevent excessive servo actuation
- Execution times are within expected ranges

### 4. Angular Error Verification Test
**Objective:** Verify that angular error calculations and logging provide detailed verification information.

**Steps:**
1. Start the detector application
2. Position objects at various locations within the camera's field of view
3. Monitor the application logs for ANGULAR_ERROR_VALIDATION and ANGULAR_ERROR_DEBUG entries
4. Verify that:
   - Logs contain track ID, radial pixel distance, focal length, and angular degrees
   - Calculated angular error matches the function result
   - Debug information shows correct pixel coordinates and displacements
   - Angular errors are reasonable for object positions

**Expected Results:**
- ANGULAR_ERROR_VALIDATION log entries show detailed calculation information
- Calculated and function results match
- Angular errors increase as objects move further from center
- Objects outside the angular threshold are properly rejected

## Validation Criteria
All tests pass if:
1. Visual overlays accurately represent detection data
2. Distance smoothing logs show detailed window information
3. Servo timing logs demonstrate proper sequencing
4. Angular error logs provide comprehensive verification data
5. System maintains real-time performance

## Troubleshooting
If issues are encountered:
1. Check camera connectivity and configuration
2. Verify servo motor connections and power
3. Confirm all configuration parameters are set correctly
4. Review system logs for error messages
5. Validate that all dependencies are properly installed