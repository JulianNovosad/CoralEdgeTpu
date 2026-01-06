# SYSTEM ARCHITECTURE SYNTHESIS

**Status:** DOCTRINE-MANDATED ARTIFACT
**Date:** 2026-01-06

## Architectural Style
**Pipeline-based, Event-driven, Soft Real-Time.**
Data flows unidirectionally from Sensor to Actuator, with parallel side-branches for Inference and Telemetry.

## ASCII Block Diagram

```ascii
      [ REALITY ]
           | (Photons)
           v
    +------+------+
    | CameraCapture |  <-- REQ-007 (120 FPS)
    +------+------+
           | (DMA Buffer - Zero Copy)
           +-------------------------+-------------------------+
           |                         |                         |
           v                         v                         v
    +------+------+           +------+------+           +------+------+
    | Inference   |           | VideoEnc    |           | Display/Viz |
    | (Edge TPU)  |           | (H.264)     |           | (Overlay)   |
    +------+------+           +------+------+           +------+------+
           | (Detections)            |                         |
           v                         v                         v
    +------+------+           +------+------+           +------+------+
    | LogicModule |           | StreamSvr   |           | App/Phone   |
    | (Ballistics)|---------->| (RTSP/UDP)  |           | (WiFi)      |
    +------+------+           +-------------+           +-------------+
           |
           +---------------------+
           | (Command)           | (Feedback/Telemetry)
           v                     v
    +------+------+       +------+------+
    | PCA9685     |       | SystemMonitor|
    | (Servo Driver)|     | (Watchdog)   |
    +------+------+       +------+------+
           |                     |
           v                     v
      [ ACTUATOR ]          [ LOGS / SAFETY CUTOFF ]
```

## Module Responsibilities

1.  **CameraCapture:** Owns the `libcamera` interface. Producer of `ImageData`.
    *   *Authority:* Camera settings, Buffer allocation.
2.  **Inference:** Owns the `tflite::Interpreter` and Edge TPU delegate. Consumer of `ImageData`. Producer of `DetectionResult`.
    *   *Authority:* Inference timing, Model integrity.
3.  **LogicModule:** The "Brain". Consumer of `DetectionResult`. Owns the State Machine.
    *   *Authority:* **SAFETY INTERLOCK**, Ballistic calculations, Actuation commands.
4.  **PCA9685Controller:** Dumb driver. Translates logical commands to I2C PWM.
    *   *Authority:* I2C bus writing.
5.  **SystemMonitor:** Independent watchdog.
    *   *Authority:* Thermal limits, Process termination.

## Data Flow & Ownership
*   **Images:** Owned by `BufferPool`. Borrowed by Camera, Inference, Encoder. Returned to Pool.
*   **Detections:** Owned by `LogicModule` (value semantics) after copying from Inference output.
*   **Commands:** Pure value types (`struct ServoCommand`).

## Timing Boundaries
*   **t_capture:** Stamped at `CameraCapture` (Start of budget).
*   **t_inference:** Stamped at `Inference` completion.
*   **t_actuate:** Stamped at `LogicModule` decision.
*   **Total Latency:** `t_actuate - t_capture` (Must be ≤ REQ-001).
