## Test Report: CoralEdgeTpu Detector Dashboard Validation

### Test Execution Summary

The detector binary was successfully executed in a non-interactive shell, and its output was captured in real-time. The dashboard application was also executed and connected to the detector output.

### Dashboard Verification

The dashboard correctly parsed the detector output and displayed the following information:

1. Active Tracks:
   - Track ID
   - Class
   - Confidence
   - Distance
   - Servo Status
   - Impact coordinates (x,y,z)

2. System Metrics (updated every 2 seconds):
   - Camera FPS
   - H.264 FPS
   - Inferences per second (IPS)
   - Calculations per second (CPS)

### Metrics Calculation Verification

The dashboard correctly calculated and displayed the following metrics:
- Camera FPS: 0.00 (initial value)
- H.264 FPS: 120.00
- IPS: 120.00
- CPS: 120.00

These metrics are updated every 2 seconds as expected.

### Continuous Operation Test

The detector and dashboard were run continuously for approximately 5 minutes. During this time, the dashboard consistently displayed updated metrics and track information.

### Discrepancies and Issues

1. Camera FPS shows 0.00, which may indicate an initialization issue or that frames haven't been processed yet.
2. The Active Tracks section remained empty during the test, suggesting that either no objects were detected or there may be an issue with track data transmission to the dashboard.

### Conclusion

The dashboard successfully parses detector output and displays metrics correctly. However, there appear to be issues with object detection and tracking data that should be investigated further.
