# Aurore Mk V – Project Context & Ground‑Truth Contract

## I. Project Overview

**Aurore Mk V** is a safety‑critical, real‑time fire‑control and weapon‑safety system. It must be treated as *live* weapon‑safety software in all builds (Debug and Release). **Actuation is real and physical.** The trigger interlock releases **ONLY** when hit‑probability > 90% **and** all safety gates pass.

### Hardware & Environmental Assumptions

* **Hardware:** Raspberry Pi 5 + Coral M.2 (PCIe). Hardware is unreliable by default: drivers may drop frames/interrupts; TPU may throttle.
* **Actuator Stack:** PCA9685 PWM over I2C. PWM output ceases if the userspace process stops.
* **Safe State (Fail‑Closed):**

  * Servo at Neutral (0°)
  * PWM signal active but commanding the interlock‑closed position
  * Trigger solenoid de‑energized
* **Interlock Policy:** Fail‑Closed. Any ambiguity, timeout, or invariant violation transitions immediately to Safe State.

---

## II. Actuation Semantics (Authoritative)

* **Actuation Definition:** One actuation equals a single **forward→back→forward** servo flick executed as fast as possible, followed by a **mandatory 300 ms cooldown**.
* **Reassertion:** Cooldown is enforced by a **software interlock** that continuously reasserts the 300 ms minimum interval between actuations.
* **Authority:** Exactly **one thread** (the LogicModule worker) may issue servo commands.
* **Ordering:** **No PWM pulse may be generated** until all safety gates pass **and** accounting equilibrium (P=C+D) is established.
* **No Simulation Mode:** There is **no dry‑run / non‑actuating path** in deployed builds.

---

## III. Development & Operational Invariants

### 1. Clock Authority & Timing

* **Authoritative Clock:** `CLOCK_MONOTONIC_RAW` (preferred) or `std::chrono::steady_clock` bound to MONOTONIC_RAW semantics.
* **Observational Clock:** `system_clock` is for telemetry stamps only; never for logic.
* **Per‑Frame Accounting:** All timing decisions are **per frame**.
* **Budgets:**

  * Nominal: 120 FPS (8.33 ms)
  * **Temporal Gate:** Frames exceeding **8.95 ms** are dropped (do not actuate).
  * **E2E Constraint:** 100 ms total; ~70 ms physical servo actuation leaves ~30 ms for compute.
* **Capture Timestamp (`tcapture`):** Defined at **sensor exposure start** (photon arrival).

### 2. Determinism & State Boundaries

* **Determinism Scope:** Required for **Logical State** only (ballistics, gating, counters).
* **Excluded from State:** Temperature, CPU jitter, absolute wall‑time, and queue depth **must not** influence branching.
* **Dropped Frames:** Must **not** influence future gating. Treat drops as if time stood still for that module.
* **Numerics:**

  * Floating point **forbidden in hot paths** unless fixed‑point or compiled with deterministic flags (e.g., `-ffloat-store`, `-fno-fast-math`) ensuring bit‑identical results across RPi 5 builds.
  * SIMD/auto‑vectorization allowed **only if deterministic**.
* **Memory Discipline:**

  * Hot paths: **no heap allocation** (`new/malloc`) and no `std::vector::push_back`.
  * Stack allocations must be **bounded and documented**; no `alloca()` or VLAs.
  * `std::optional` and `std::array` are **approved**.
  * `new` is allowed **only during `init()`**. Any allocation after RUNNING is a **fatal invariant violation**.

---

## IV. Failure Handling (No Latching HALT)

* **Continuous Operation:** The system **must run continuously** and attempt self‑recovery. There is **no latched HALT state**.
* **Operator Stop:** Only explicit operator intent (Ctrl+C / SIGINT) stops the system.
* **Failure Precedence:**

  1. **Timing Violation (> 8.95 ms):** Drop frame; increment Dropped; **do not actuate**.
  2. **State Inconsistency (P ≠ C + D):** Immediate transition to **Safe State**, emit CRITICAL telemetry, **recover and continue**.
  3. **Gate Violations (Spatial/Confidence):** Reject actuation for the current frame.

---

## V. The P=C+D Accounting Law

* **Invariant:** `Produced = Consumed + Dropped` at all times.
* **Heartbeat:** If the TPU returns zero detections, the InferenceEngine **must still produce** an empty result buffer and increment Consumed.

---

## VI. Concurrency & Scheduling

* **Scheduling:** LogicModule and InferenceEngine run at **SCHED_FIFO**.
* **Queues:** **Boost.Lockfree** queues are mandatory. Blocking with timeouts is allowed **only** in the ApplicationSupervisor.
* **I/O:** Servo I/O must be **asynchronous**; LogicModule must not block on I2C completion.
* **Signals:** Signal handlers (SIGINT) may **only** set `std::atomic<bool> running_`.

---

## VII. Mandatory Pre‑Flight Audit (Auto‑Audit)

The Agent **MUST** output this report before any diff. If any answer is NO, discard the diff.

### 1. Complexity & Performance

* All per‑frame work is **O(1)** with **compile‑time bounded loops** only.
* No `std::endl`, `fflush`, or logging frameworks in hot paths. Use the pre‑allocated **APP_LOG ring buffer** only.

### 2. Verification Hooks

* **Counters:** Use `app_ref_->inference_produced_` and `app_ref_->inference_consumed_`.

### 3. Forbidden Actions (Hard No)

* No helper abstractions that hide allocations.
* No refactoring timing logic for readability.
* No replacing fixed arrays with `std::vector` / `std::deque`.
* No exception‑based control flow.

### 4. Forensic Report Template (REQUIRED)

**FORENSIC AUDIT REPORT**

* Module:
* P=C+D Path:
* Timing Source (MONOTONIC_RAW):
* Actuation Cooldown Enforcement (300 ms):
* Result: READY / REJECTED

---

## VIII. Scope & Governance

* Applies to **all builds**.
* Assertions in Release are **mandatory** when invariants are violated.
* This document **supersedes all inline comments**.
* Automatic recovery after failures is **required**; latched HALT is **forbidden**.
* Remote actuation and model hot‑swapping are **out of scope**.
* Assume **MIL‑STD‑882E** review.

---

## IX. Troubleshooting

* **TPU Stall:** `pkill -9 detector`; verify MSI‑X interrupts.
* **Bayer Unpacking:** 10‑bit Raw (5 bytes / 4 pixels); explicit MSB shifting required.

**Reference Baseline:** Commit `a4990fd3a8a286acd2ab7953f6f538de893809a5` (known‑good visualization and VLC feed).
- remember to never, ever use ./detector. always use ./build/detector
- do not ever run ./build.sh
