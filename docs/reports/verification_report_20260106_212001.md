INTENT DECOMPOSITION

Operational Goal: Eliminate E2E latency spikes, recover full camera capture rate (120 FPS), implement thread and scheduling discipline, perform HIL pre-run hardening, and generate a Stage 3 Verification Report. (Original Goal). The current sub-goal is to diagnose and resolve the 0 IPS issue in the Inference Module and enable proper TPU usage.

Non-Goals: Debugging unrelated modules, optimizing non-bottleneck areas prematurely.

Hard Constraints:
- Strict adherence to Edge TPU usage for inference (no CPU fallback unless explicitly requested for diagnostic purposes).
- Achieve 120 FPS camera capture rate.
- Implement robust thread and scheduling discipline.
- Generate Stage 3 Verification Report.
- Ensure system survivability; application must not crash silently.

Soft Constraints:
- Minimize overall latency.
- Maintain thermal stability.
- Maintain code clarity and adherence to project conventions.

Implicit Assumptions Detected:
- The Edge TPU hardware is physically present and functional.
- Edge TPU drivers and libraries are correctly installed and compatible.
- The `libedgetpu` library functions as expected without returning corrupted pointers on failure.
- The system environment (OS, kernel) is stable enough to support Edge TPU operations.
- `std::unique_ptr` with custom deleter for `edgetpu_device` is sufficient to prevent resource leaks/corruption in `edgetpu_list_devices`.
- Standard C++ exception handling for `std::runtime_error` will successfully catch errors thrown by `InferenceEngine` constructor.

REQUIREMENT EXTRACTION

REQ-INF-001: The Inference Module shall report a non-zero Inference Per Second (IPS) rate during active operation when the Edge TPU is available.
Verification: Observe `Inference Rate` in System Monitor output / `unified_test.csv`.
Failure Mode: 0 IPS, indicating failed inference.

REQ-TPU-001: The system shall strictly utilize the Edge TPU for inference if a model path is configured for the TPU.
Verification: Verify `EdgeTPU delegate applied successfully` logs, and no CPU fallback warnings.
Failure Mode: Inference proceeds on CPU when TPU is configured, or application crashes due to non-TPU delegate.

REQ-TPU-002: The InferenceEngine constructor shall throw a `std::runtime_error` if the Edge TPU device is not found or its delegate cannot be successfully created.
Verification: Application terminates immediately with a `std::runtime_error` message from the constructor if TPU is unavailable.
Failure Mode: Application attempts to run without TPU when not permitted, or crashes with an unclear error.

REQ-SYS-001: The application shall not experience memory corruption (e.g., `free(): corrupted unsorted chunks`) leading to `SIGABRT`.
Verification: Application runs without crashing due to memory corruption.
Failure Mode: Application crashes prematurely with memory corruption errors.

REQ-DRV-001: The Edge TPU kernel driver shall successfully enable RAM within expected timeouts and open the device without returning error code -110.
Verification: `dmesg | grep -i apex` shows no "RAM did not enable within timeout" or "Error in device open cb: -110" messages.
Failure Mode: Kernel driver reports hardware-level issues with the Edge TPU.

ARCHITECTURE SYNTHESIS

The current issue is a low-level problem related to the Edge TPU, likely within the `InferenceEngine`'s interaction with the `libedgetpu` library or the underlying kernel driver.

The relevant components of the architecture are:

*   **Application (Supervisor):** Orchestrates module startup and shutdown.
*   **InferenceEngine:** Responsible for loading the ML model, interacting with the Edge TPU delegate, and performing inference. This is where `libedgetpu` is directly used.
*   **libedgetpu Library:** Userspace library for interacting with the Edge TPU.
*   **Edge TPU Kernel Driver:** Low-level kernel module that manages the Edge TPU hardware (`/dev/apex_0`).
*   **Edge TPU Hardware:** The physical Coral Edge TPU accelerator.

ASCII Block Diagram (Conceptual Focus on TPU interaction):

```
+---------------------+
|     Application     |
|   (orchestration)   |
+----------+----------+
           |
           | Calls `InferenceEngine` constructor
           v
+---------------------+
|   InferenceEngine   |
|  (model loading,    |
|   delegate mgmt)    |
+----------+----------+
           |
           | Calls `edgetpu_list_devices`, `edgetpu_create_delegate`
           v
+---------------------+
|   libedgetpu        |
|     Library         |
| (userspace driver   |
|   interface)        |
+----------+----------+
           |
           | Accesses `/dev/apex_0`
           v
+---------------------+
| Edge TPU Kernel     |
|      Driver         |
| (RAM init, device   |
|    management)      |
+----------+----------+
           |
           | Communicates with hardware
           v
+---------------------+
| Edge TPU Hardware   |
|  (accelerator)      |
+---------------------+
```

TRADE-SPACE ENUMERATION

Given the persistent low-level crash (`free(): corrupted unsorted chunks`, `SIGABRT`) and the `dmesg` errors (`RAM did not enable within timeout`, `Error in device open cb: -110`), the primary trade-off is between:

*   **Option A: Attempting further software-only debugging/workarounds.**
    *   **Rationale:** Continue trying to find a software bug or configuration issue that might be triggering the `libedgetpu` crash. This avoids needing physical access or system-level changes.
    *   **Risks/Failure Modes:** High likelihood of continued crashes and 0 IPS if the underlying issue is truly hardware or driver-level. Significant time sink with low probability of success if the hardware/driver is indeed faulty.

*   **Option B: Assume hardware/driver fault and recommend external intervention.**
    *   **Rationale:** The `dmesg` errors are strong indicators of a problem outside the application's direct control. Recommending a physical reset (reseating the Edge TPU, rebooting the system) or driver reinstallation directly addresses the most probable root cause.
    *   **Risks/Failure Modes:** Requires user intervention and potentially system downtime. If the issue turns out to be a subtle software bug, this might be an unnecessary step. However, given the nature of the errors, this is the most direct path to resolution.

Chosen Option Rationale:
I will choose **Option B**. The evidence from `dmesg` (kernel reporting RAM initialization failures and device open errors) combined with the `free(): corrupted unsorted chunks` occurring extremely early in the application's startup, even before explicit application-level logs can fully initialize, strongly suggests a fundamental issue with the Edge TPU device or its core drivers. My cleanup scripts effectively eliminated process contention, yet the problem persists at a lower level. This indicates a state beyond what application-level code can reasonably recover from without external intervention.

CROSS-DOMAIN IMPACT CHECK

The current issue, a `SIGABRT` due to memory corruption (`free(): corrupted unsorted chunks`) originating from the Edge TPU initialization, has broad implications across several domains:

Vision: The entire vision pipeline is halted as inference (and thus detection) cannot occur. This directly impacts scene understanding.
Inference / ML: The core ML inference capability, specifically on the Edge TPU, is non-functional. The intended accelerated inference is entirely unavailable.
Control / Actuation: Since the Logic Module relies on inference results for decision-making, the control and actuation systems would be starved of input, leading to a "fail-closed" or unresponsive state.
Real-Time Scheduling: Threads related to inference and downstream processing (Logic, Overlay) are either not starting or immediately crashing, disrupting the intended real-time schedule and potentially leaving CPU cores idle that should be performing inference.
Thermal / Power: While the current issue is not directly thermal, a malfunctioning Edge TPU could draw abnormal power or contribute to system instability. The `dmesg` logs already indicate the TPU's RAM isn't initializing, which might be a power-related issue at a very low level.
Safety / Hazard: A system relying on real-time vision for safety functions would be severely compromised. Lack of inference means lack of hazard detection, potentially leading to unsafe conditions if physical actuation were involved.
Telemetry / Forensics: The ability to collect meaningful telemetry beyond the initial crash is severely limited. The application exits prematurely, preventing long-term data collection. Existing forensic data (like `dmesg`) points to a critical underlying issue.

EXECUTION AUTHORIZATION
Status: BLOCKED
Blocking Issues:
- REQ-SYS-001 (Memory Safety): Application crashes with SIGABRT due to memory corruption during Edge TPU initialization.
- REQ-DRV-001 (Edge TPU Kernel Driver): `dmesg` reports "RAM did not enable within timeout" and "Error in device open cb: -110" messages, indicating a low-level hardware or driver fault with the Edge TPU.
- REQ-TPU-002 (TPU Initialization): The InferenceEngine constructor is unable to successfully initialize the Edge TPU, leading to application termination (or memory corruption before it can throw a clean exception).

Justification for BLOCKED status: The core issue is outside the application's software control. The underlying Edge TPU hardware/driver is in a dysfunctional state, preventing any progress on inference-related tasks. Attempting further software changes without resolving this foundational problem is unproductive and risks further obscuring the root cause.


