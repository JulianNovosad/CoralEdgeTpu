# INTENT DECOMPOSITION

**Status:** DOCTRINE-MANDATED ARTIFACT
**Date:** 2026-01-06

## Operational Goal
To design and implement a **deterministically safe fire-control system** ("Avant-garde Mk V") for a robotics platform. The system uses computer vision (Edge TPU) and ballistic logic to authorize or inhibit physical actuation (servo/trigger) based on a strict "fail-closed" safety policy. The system must operate with an end-to-end latency of ≤ 100ms.

## Non-Goals
*   **General-purpose object detection:** The system is specialized for specific targets only.
*   **Cloud connectivity:** The system must operate entirely offline/locally.
*   **Aesthetic UI:** The UI is for telemetry/debugging only, not consumer appeal.
*   **Maximizing FPS beyond requirements:** 120 FPS is the target; higher is not a priority if it compromises determinism.

## Hard Constraints (Inviolable)
1.  **Fail-Closed Actuation:** If *any* sub-system fails, hangs, or reports low confidence, the servo must return to (or remain in) the SAFE state.
2.  **Latency Budget:** Photon-to-actuation latency must not exceed 100ms.
3.  **Deterministic Timing:** All internal timing must be relative to a monotonic `t_zero` epoch.
4.  **No Dynamic Allocation in Hot Paths:** After initialization, no heap memory may be allocated or freed during the runtime loop.

## Soft Constraints (Preferences)
1.  **Zero-Copy:** Prefer DMA-buf / shared memory over `memcpy`.
2.  **Thermal Stability:** System should throttle processing before thermal shutdown, defaulting to safe state.

## Implicit Assumptions Detected
*   **Sensor Trust:** The camera sensor data is assumed to be a "true" representation of reality (no adversarial attacks).
*   **Actuator Feedback:** The servo is assumed to respond within 70ms; there is no hardware feedback sensor to confirm its position (open-loop control). **RISK: High.**
*   **Time Synchronization:** It is assumed that the TPU and CPU share a stable enough clock reference for correlated timestamps.
