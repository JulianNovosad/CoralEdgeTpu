# HAZARD ANALYSIS & SAFETY CASE

**Status:** DOCTRINE-MANDATED ARTIFACT
**Date:** 2026-01-06

## 1. System Safety Scope
The system controls a servo motor capable of actuating a trigger mechanism.
**Assumed Hazard:** Unintended discharge / actuation outside of valid target parameters.

## 2. Fault Tree Analysis (Simplified)

**Top Event:** Unintended Actuation (Servo moves to FIRE position incorrectly)

*   **OR Gate 1: Logic Failure**
    *   False Positive Detection (Object classified as target with high confidence).
        *   *Mitigation:* High confidence threshold (>90%), Consistency check (N consecutive frames), Ballistic validity check.
    *   Stale Data (Logic actuates on old frame).
        *   *Mitigation:* Timestamp freshness check (<50ms age).
    *   Bug in State Machine.
        *   *Mitigation:* Code audit, Unit tests, Default state = IDLE.

*   **OR Gate 2: Hardware/Driver Failure**
    *   I2C Corruption (Bit flip causes PWM spike).
        *   *Mitigation:* Checksum (if avail), Periodic "Heartbeat" reset to SAFE.
    *   Servo Stuck.
        *   *Mitigation:* None (Open loop). **Accepted Risk**.
    *   Processor Freeze (Output held high).
        *   *Mitigation:* External Watchdog (if avail) or `SystemMonitor` soft-watchdog killing process (GPIO failsafe).

## 3. Safety Controls (Fail-Closed)

### SC-1: The "Dead Man" Switch (Software)
The servo requires a continuous stream of "ENGAGE" commands to stay active.
*   **Mechanism:** `LogicModule` sends pulses. `PCA9685Controller` does *not* auto-reset, so `LogicModule` must explicitly send "RETRACT" on any uncertainty.
*   **Implementation:** If `Inference` queue is empty for >100ms, `LogicModule` forces `RETRACT`.

### SC-2: The Ballistic Gate
Actuation is only permitted if:
1.  Target Class is Valid.
2.  Target Confidence > Threshold.
3.  Target Range is within Effective Range.
4.  Projectile Path is Clear (simplified check).

### SC-3: Thermal Interlock
If `SystemMonitor` detects TPU/CPU temp > 80°C:
1.  Signal `LogicModule` to Enter `FALLBACK_C_CRITICAL_STATE`.
2.  `LogicModule` forces Servo to SAFE.
3.  System shuts down.

## 4. Verification Plan
*   **Unit Test:** Inject "False Positive" detections -> Verify Servo remains SAFE.
*   **Unit Test:** Inject "Stale" timestamps -> Verify Servo remains SAFE.
*   **Integration Test:** Disconnect Camera -> Verify Servo returns to SAFE within 200ms.
