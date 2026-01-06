# GEMINI.md — Project Context Overlay (NON-AUTHORITATIVE)

## IMPORTANT — READ FIRST

This file is a **context-only overlay**.

It exists solely to help an AI agent rapidly understand the **project domain, vocabulary, and technical environment**.

This file does **NOT**:
- authorize execution or implementation
- define requirements or invariants
- approve architectures or designs
- permit actuation or hardware control
- override the universal GEMINI doctrine

If any content in this file appears to conflict with the global GEMINI.md doctrine,  
**the doctrine takes precedence without exception**.

All statements herein are **descriptive, assumed, or historical**, not prescriptive.

---

## 1. Project Identity & Domain Context

**Project Name (Contextual):** Avant-garde Mk V  
**Domain:** Robotics / Embedded Systems / Safety-Adjacent Control  
**Primary Focus Areas:**  
- Real-time perception  
- Deterministic decision pipelines  
- Safety-biased actuation gating  

This project is *envisioned* as an embedded system exploring how computer vision and probabilistic reasoning may be combined with **physical safety interlocks** in a robotics context.

Any references to safety or actuation are **conceptual** and must be treated as **hazard-adjacent** by default.

---

## 2. Platform & Environment (Descriptive)

The project is *commonly associated* with the following environment:

- **Compute Platform:** Raspberry Pi–class embedded Linux system  
- **Accelerator:** Coral Edge TPU (PCIe-attached)  
- **Primary Language:** C++ (modern standard, contextually C++17)  

These details describe the *typical working environment* and are not mandates.

---

## 3. Conceptual System Intent (NON-BINDING)

At a high level, the system is *intended* to explore:

- Vision-based scene understanding
- Deterministic inference pipelines
- Logic layers that reason about confidence and uncertainty
- Physical outputs that default to **safe or inhibited states**

Any numeric thresholds, probabilities, or timing figures mentioned elsewhere should be treated as **exploratory assumptions**, not guarantees.

---

## 4. Assumed Technical Biases (Contextual Only)

The following biases are commonly discussed in the project context:

- Preference for **deterministic behavior** over peak throughput
- Interest in **zero-copy or low-copy** data movement patterns
- Awareness of **latency budgets** in perception → decision pipelines
- Sensitivity to **thermal, power, and scheduling effects** on embedded platforms

These are **background assumptions**, not enforced constraints.

---

## 5. Conceptual Module Vocabulary

The project discussions may reference modules with names such as:

- *Application / Supervisor* — a coordinating concept
- *Camera Capture* — image acquisition from a sensor
- *Inference* — ML model execution (e.g., TPU-accelerated)
- *Logic* — decision-making or arbitration layer
- *Monitoring* — observation of system health or telemetry
- *Streaming / Encoding* — visualization or remote observation

These names describe **roles**, not fixed implementations or boundaries.

No ownership, authority, or scheduling semantics are implied.

---

## 6. Data Flow — Illustrative Mental Model

A commonly referenced **mental model** (not an architecture commitment):

1. Image data becomes available from a sensor
2. That data is observed by multiple subsystems
3. Inference produces abstract results
4. Logic reasons over results and context
5. Optional visualization or telemetry is produced

This model is **illustrative only** and exists to align terminology.

---

## 7. Development & Exploration Context

In discussions, the project may reference:

- Experimentation with threading and concurrency
- Attention to memory ownership and lifetime
- Configuration via external files
- Logging and telemetry for post-hoc analysis

These reflect **areas of interest**, not rules.

---

## 8. Safety Posture (Contextual Framing)

The project is generally discussed with a **safety-first mindset**, meaning:

- Physical outputs are assumed to be potentially hazardous
- Conservative behavior is preferred when uncertainty exists
- Fail-closed thinking is common in design discussions

No safety claim, guarantee, or certification is implied by this file.

---

## 9. What This File Is NOT

To avoid ambiguity, this file is **not**:

- a build guide
- a runbook
- a test plan
- a requirements document
- an architecture specification
- an authorization to control hardware

Any such artifacts must emerge **only after** the universal GEMINI doctrine’s cognitive pipeline has been explicitly satisfied.

---

## 10. How Agents Should Use This File

Agents should use this file to:

- understand terminology
- recognize the technical domain
- load environmental assumptions
- avoid irrelevant suggestions
- align vocabulary with the project

Agents must **not** treat this file as permission to act.

---

## End of Context Overlay
