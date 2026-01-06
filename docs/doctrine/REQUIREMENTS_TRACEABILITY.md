# REQUIREMENTS TRACEABILITY MATRIX

**Status:** DOCTRINE-MANDATED ARTIFACT
**Date:** 2026-01-06

| REQ-ID | Description | Verification Method | Failure Behavior | Module |
| :--- | :--- | :--- | :--- | :--- |
| **REQ-001** | End-to-end latency (photon-to-servo) ≤ 100ms. | Telemetry histogram (timestamp diffs). | Inhibit actuation if >100ms avg. | `Application` |
| **REQ-002** | System must default to SAFE state on startup. | Physical inspection + Log audit. | N/A (Initial state). | `LogicModule` |
| **REQ-003** | System must revert to SAFE state if target confidence < 90%. | Injection test (low conf data). | Immediate Servo Retract. | `LogicModule` |
| **REQ-004** | System must revert to SAFE state if inference stops > 200ms. | Watchdog timer test. | Immediate Servo Retract. | `LogicModule` |
| **REQ-005** | Detection accuracy for target class ≥ 90% mAP. | Static dataset validation. | Log warning. | `Inference` |
| **REQ-006** | No heap allocation in hot-path loop. | Static analysis / Heap profiler. | Build failure / Audit fail. | `All` |
| **REQ-007** | Camera capture rate ≥ 100 FPS. | Runtime FPS counter. | Log warning. | `CameraCapture` |
| **REQ-008** | Actuation command must adhere to 300ms cooldown. | Logic unit test. | Command ignored. | `LogicModule` |
| **REQ-009** | All logs must be timestamped relative to `t_zero`. | Log inspection. | Audit fail. | `Logger` |
| **REQ-010** | Thermal throttling at CPU/TPU > 80°C. | Thermal stress test. | Enter SAFE mode. | `SystemMonitor` |
