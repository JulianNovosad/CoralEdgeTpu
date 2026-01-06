# TIMING & DETERMINISM AUDIT PLAN

**Status:** DOCTRINE-MANDATED ARTIFACT
**Date:** 2026-01-06

## 1. Clock Source
*   **Authoritative Clock:** `CLOCK_MONOTONIC_RAW` (via `clock_gettime`).
*   **Reason:** Unaffected by NTP steps or slew. Essential for accurate interval measurement.
*   **Implementation:** Wrapper class `MonotonicClock` in `src/timing.h`.

## 2. Time Zero (t_zero)
*   **Definition:** The value of `CLOCK_MONOTONIC_RAW` at the first line of `main()`.
*   **Usage:** All logs and internal logic timestamps will be `current_time - t_zero`.
*   **Goal:** Human-readable millisecond/microsecond logs starting from 0.000.

## 3. Latency Budgeting (REQ-001)

| Stage | Budget (ms) | Check |
| :--- | :--- | :--- |
| **Camera Capture** | 15.0 | `t_captured` marked. |
| **Pre-process** | 5.0 | Resize/Format. |
| **Inference (TPU)** | 20.0 | Hardware dependent. |
| **Logic/Ballistics**| 5.0 | Math heavy. |
| **OS/Scheduling** | 5.0 | Jitter margin. |
| **Servo Response** | 50.0 | Mechanical lag (Est). |
| **TOTAL** | **100.0** | **Pass/Fail** |

## 4. Concurrency & Scheduling
*   **Main Thread:** Supervisor / Cleanup.
*   **Camera Thread:** `SCHED_FIFO` (Priority High). Producer.
*   **Inference Thread:** `SCHED_FIFO` (Priority Medium). Consumer/Producer.
*   **Logic Thread:** `SCHED_FIFO` (Priority High). Consumer/Controller.
*   **Logging Thread:** `SCHED_OTHER` (Low).

## 5. Hot-Path Audit
*   **Forbidden:** `new`, `malloc`, `cout` (blocking I/O), `sleep` (in logic loops).
*   **Required:** Pre-allocated `std::vector` (via `reserve` or `BufferPool`), `lock_free_queue` or mutex-guarded pre-allocated buffers.
