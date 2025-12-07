# Ballistics and Uncertainty Propagation Theory

## 1. Overview

This document outlines the theoretical framework for ballistics calculations and the propagation of uncertainty within the CoralEdgeTpu project. The goal is to ensure accurate prediction of projectile trajectories and target impact points, while also robustly quantifying and managing the associated uncertainties for safety-critical operations.

## 2. Ballistics Model

The ballistics model describes the trajectory of a projectile from launch to impact. For short-range, low-velocity projectiles, a simplified model might suffice, but for higher precision or longer ranges, more complex factors must be considered.

### 2.1 Projectile Properties
*   **Mass (m):** The mass of the projectile in kilograms (kg).
*   **Caliber (d):** The diameter of the projectile in meters (m).
*   **Muzzle Velocity (v_0):** The initial velocity of the projectile as it leaves the barrel in meters per second (m/s).
*   **Drag Coefficient (C_d):** A dimensionless quantity used to quantify the drag or resistance of an object in a fluid environment (air). This often varies with Mach number.
*   **Form Factor (i):** A dimensionless factor describing the aerodynamic efficiency of the projectile's shape.

### 2.2 Environmental Factors
*   **Gravity (g):** The acceleration due to gravity, typically 9.81 m/s² downwards. This is usually assumed constant.
*   **Air Density (ρ):** The density of the air, which varies with temperature, pressure, and humidity. It significantly affects drag.
*   **Wind (v_w):** Both speed and direction of ambient wind, which can impart significant lateral and vertical forces on the projectile.

### 2.3 Equations of Motion

The trajectory of a projectile is governed by Newton's second law, considering gravitational force and aerodynamic drag.

**Without Air Resistance (Simplified):**
x(t) = v_0x * t
y(t) = v_0y * t - 0.5 * g * t^2

Where v_0x and v_0y are the initial horizontal and vertical components of velocity. This model is generally insufficient for realistic scenarios.

**With Air Resistance (Drag Force):**
The drag force (F_d) is typically modeled as:
F_d = 0.5 * ρ * v^2 * A * C_d
Where A is the cross-sectional area (π * (d/2)^2) and v is the instantaneous velocity of the projectile relative to the air.

The differential equations of motion become:
m * d²x/dt² = -F_dx
m * d²y/dt² = -F_dy - m * g
m * d²z/dt² = -F_dz

These equations are typically coupled and non-linear, requiring numerical integration methods (e.g., Runge-Kutta) to solve. The drag force components (F_dx, F_dy, F_dz) depend on the projectile's velocity vector and the wind vector.

## 3. Uncertainty Propagation

Uncertainty is inherent in all sensor measurements and model predictions. This section describes how these uncertainties are quantified and propagated through the system to provide robust predictions and safety assessments.

### 3.1 Sources of Uncertainty
*   **Sensor Noise:**
    *   **Camera:** Pixel noise, calibration errors, lens distortion, resolution limits, rolling shutter effects.
    *   **IMU:** Accelerometer/gyroscope bias, drift, scale factor errors, noise (rate random walk, velocity random walk), temperature sensitivity.
    *   **Distance Sensor:** Measurement noise, systematic errors (e.g., due to surface properties, multi-path), limited range.
*   **Model Inaccuracy:**
    *   **Object Detection:** Bounding box accuracy, class confidence, missed detections, false positives.
    *   **Object Tracking:** Prediction errors (e.g., due to non-linear target motion), association errors, track initiation/termination logic.
    *   **Ballistics Model:** Simplifications (e.g., constant drag coefficient), unmodeled effects (e.g., Magnus effect, Coriolis effect), sensitivity to input parameters (muzzle velocity, projectile properties).
*   **Environmental Variability:**
    *   Uncertainty in air density, wind conditions (magnitude and direction), temperature.
*   **Actuator Precision:**
    *   Servo positioning error, backlash, dynamic response limits, control loop latency.

### 3.2 Quantification of Uncertainty
Uncertainty for various parameters is often represented by statistical measures such as:
*   **Standard Deviations (σ):** For individual scalar measurements.
*   **Covariance Matrices (Σ):** For multi-variate state vectors (e.g., position and velocity), capturing correlations between uncertainties.
*   **Confidence Intervals:** For estimates, providing a range within which the true value is expected to lie with a certain probability.

### 3.3 Propagation Methodology
Methods for propagating uncertainties through non-linear systems:
*   **Linearization (Extended Kalman Filter - EKF):** Linearizes the system dynamics around the current state estimate and propagates the covariance matrix. Suitable for moderately non-linear systems.
*   **Unscented Kalman Filter (UKF):** Uses a deterministic sampling approach (sigma points) to capture the true mean and covariance of the state distribution more accurately than EKF, without explicit linearization.
*   **Monte Carlo (MC) Simulations:** Involves running the system model many times with inputs sampled from their respective uncertainty distributions. Provides a robust, but computationally intensive, way to estimate the output uncertainty distribution.
*   **Particle Filter (PF):** A sequential Monte Carlo method, particularly useful for non-Gaussian and highly non-linear systems.

## 4. Acceptance Limits and Safety Criteria

For safety-critical applications, strict acceptance limits must be defined for various system parameters and for the overall uncertainty in the predicted impact point. These limits determine when the system can operate, when it needs to degrade performance, or when it must cease operation.

### 4.1 Predicted Impact Point Uncertainty
*   **Maximum Acceptable Standard Deviation (2D Impact Point):** Specifies the maximum allowed spread of the predicted impact point on the target plane.
    *   **Nominal Operation:** Smallest allowable uncertainty (e.g., < 1 cm 1σ).
    *   **Degraded Operation (Fallback A):** Increased uncertainty tolerance, but still within safety bounds (e.g., < 3 cm 1σ).
    *   **Critical Operation (Fallback B):** Highest uncertainty tolerance before complete shutdown (e.g., < 5 cm 1σ).
*   **Confidence Level for Impact:** A required probability (e.g., 99%) that the actual impact will fall within a defined area around the predicted point.

### 4.2 Tracking Stability
*   **Minimum `hit_streak`:** A track must be consistently detected for a minimum number of frames before being considered stable enough for ballistics calculations.
*   **Maximum `missed_frames`:** If a tracked object is missed for more than a threshold number of frames, the track should be terminated or moved to a less reliable state.

### 4.3 Sensor Data Validity
*   **Thresholds for IMU drift/bias:** If IMU data exhibits excessive drift or bias, it may indicate a sensor malfunction or an environment that exceeds operational limits.
*   **Minimum detection `score`:** Object detections below a certain confidence score should be discarded or treated with higher uncertainty.
*   **Sensor Cross-Verification:** Where possible, compare data from redundant sensors (e.g., visual odometry vs. IMU integration) to detect anomalies.

## 5. Fallback Modes

When acceptance limits are exceeded or system health degrades, the system must transition to predefined fallback modes to maintain safety.

*   **Fallback A (Reduced Performance):**
    *   **Trigger:** Moderate increase in uncertainty, transient sensor dropouts, or minor component degradation.
    *   **Action:** Increase data buffering, lower processing FPS, reduce control loop aggressiveness, increase safety margins, log warnings.
    *   **Outcome:** System continues operation but with reduced accuracy or responsiveness.
*   **Fallback B (Warning State / Restricted Operation):**
    *   **Trigger:** Significant uncertainty, persistent sensor errors, critical component failure, or detection of a "no-fire" zone violation.
    *   **Action:** Stop active tracking/engagement, halt servo movements to a safe neutral position, provide visual/auditory alerts to operator, log critical errors, initiate diagnostic routines.
    *   **Outcome:** System remains operational for monitoring but cannot engage targets.
*   **Fallback C (Safe Shut-down / Killswitch Activation):**
    *   **Trigger:** Catastrophic system failure, unmanageable uncertainty, loss of critical sensor data, or explicit operator killswitch activation.
    *   **Action:** Immediately cease all active operations, cut power to dangerous actuators (e.g., through a hardware killswitch), log all available diagnostic data, and enter a safe, inactive state.
    *   **Outcome:** System is fully disarmed and inactive, requiring manual intervention to restart.

## 6. Future Work / Enhancements

*   **Advanced Tracking Filters:** Implement Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), or Particle Filters for more robust object tracking and state estimation.
*   **Sensor Fusion (Multi-Sensor Integration):** Integrate data from multiple sensors (e.g., camera, IMU, LiDAR/distance sensor) using a Kalman filter framework to improve state estimation accuracy and robustness.
*   **Adaptive Drag Models:** Incorporate more sophisticated drag models that account for varying Mach numbers and atmospheric conditions.
*   **Environmental Sensing:** Integrate sensors for real-time air temperature, pressure, humidity, and wind speed/direction to enhance ballistics accuracy.
*   **Calibration Routines:** Develop automated or semi-automated calibration procedures for cameras, IMUs, and servos.
*   **Self-Diagnostics and Prognostics:** Implement routines to detect impending sensor failures or performance degradation.
*   **Formal Verification:** Apply formal methods for verifying safety-critical logic and uncertainty propagation models.
*   **Real-time Operating System (RTOS) Integration:** If not already using one, consider an RTOS to guarantee timing constraints for safety-critical threads.
