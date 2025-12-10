
````markdown
# CoralEdgeTpu

**C++ Edge TPU inference stack voor de Google Coral M.2 Accelerator**

Deze repository is een volledige heropbouw van de oude, slecht onderhouden Coral/TensorFlow Lite toolchain — modern, stabiel, reproduceerbaar en volledig gericht op **native C++ inferencing op de Google Coral M.2 TPU**.

Het project bevat álle bronbestanden, patches, TensorFlow-Lite headers, Flatbuffers, build-scripts en dependency-versies die nodig zijn om de Edge TPU 100% deterministisch en offline te kunnen builden. Dit elimineert versie-hell, Python-dependency chaos en ontbrekende upstream files.

---

## ✨ Functionaliteit

* Volledige C++ applicatie die de gehele pijplijn beheert.
* Multi-threaded architectuur voor camera capture, inferentie, encoding en streaming.
* Gebruik van `libcamera` voor moderne en efficiënte camera-aansturing op de Raspberry Pi.
* Edge TPU-versnelde inferentie met TensorFlow Lite.
* H.264 video encoding en realtime streaming via **UDP/RTP of RTSP**.
* Bounding boxes, reticle coordinates, en telemetrie via **ZeroMQ PUB/SUB**.
* Uitgebreide configuratie via `config.json`.
* Robuuste logging met microbenchmarks en CSV output.
* Centrale supervisie via `system_monitor` (CPU, RAM, temperatuur, thread health, etc.)
* Alle core realtime actie en beslissingen via `logic` module (3D ballistiek, hit-scan, servo-actuatie, veiligheids-/onzekerheidspropagatie).

---

## 🚀 Bouwen en Draaien

### Vereisten

* Raspberry Pi 5
* Google Coral M.2 TPU (PCIe)
* Systeem-dependencies: `libcamera-dev`, `libedgetpu1-std`, `libzmq3-dev`
* CMake ≥ 3.16
* g++ ≥ 10

### Bouwen

Het project wordt gebouwd met een alles-in-één script dat alle dependencies downloadt, compileert en de hoofdapplicatie bouwt.

```bash
chmod +x build.sh
./build.sh
````

### Draaien

Na een succesvolle build staat de applicatie in `build/detector`.

```bash
./build/detector
```

De applicatie verwacht een `config.json` bestand in de root van de repository.

---

## 📂 Repository Structuur

```
CoralEdgeTpu/
├── build/                   # Build output (gegenereerd)
├── civetweb/                # Submodule voor de webserver (optioneel)
├── docs/                    # Documentatie
├── flatbuffers/             # Submodule voor Flatbuffers
├── include/                 # TFLite headers
├── lib/                     # Gecompileerde libraries (.so files)
├── model/                   # .tflite AI modellen
├── src/                     # C++ broncode van de applicatie
├── tests/                   # Losstaande test-utilities
├── tensorflow-src/          # TensorFlow broncode (submodule)
├── build.sh                 # Hoofd build script
├── CMakeLists.txt           # CMake build configuratie
├── config.json              # Applicatieconfiguratie
└── README.md                # Dit bestand
```

---


## 🔧 Stage-Gate Plan

### Stage 0: Technische haalbaarheid & prestatiegrenzen

**Doel:** Basale throughput en latentie meten van kernsubsystemen zonder volledige integratie.

**Subsystemen:**

* Logic: `logic.*` : actuation, 3D ballistiek, hit-scan, veiligheids-/onzekerheidspropagatie 
* Camera: `src/camera_capture.*`, `src/buffer_pool.h`, `src/pipeline_structs.h`
* TPU: `src/inference.*` + `.tflite model`
* Encoder: `src/h264_encoder.*`

**Gating criteria:**

* FPS, berekeningen/s en latentie per subsystem gemeten
* Kernel-aanpassingen (PCIe, IRQ-affiniteiten, MSI-X) gedocumenteerd met lspci -vvv , sudo dmesg | grep -i apex , etc.
* **Logging per Subsystem:**
    *   Elk kernsubsystem (Camera, TPU, Encoder, Logic, System Monitor) moet zijn eigen CSV-logbestanden hebben.
    *   Deze logbestanden moeten worden opgeslagen in een subdirectory onder de geconfigureerde `log_path`, met de naam van het subsystem (bijv. `/logs/camera/`, `/logs/tpu/`).
    *   De bestandsnaamconventie moet `module_YYYY_MM_DD_HH:MM.csv` zijn.
    *   Er mogen maximaal 3 rotaties van logbestanden per subsystem worden bewaard.

#### Logging format (Stage 0)

Stage-0 logging moet een uniforme prefix van kolommen gebruiken (dezelfde eerste velden voor elke subsystem CSV), gevolgd door module-specifieke metrics.

**Unified prefix columns (epoch UTC in milliseconden):**
   - `produced_ts_epoch_ms`  — timestamp wanneer deze logregel werd geproduceerd (epoch ms, UTC)
   - `module`                — module naam: `camera`|`tpu`|`encoder`|`logic`|`sysmon`
   - `thread_id`             — numerieke OS thread id (TID)
   - `event`                 — korte label (bijv. `frame_captured`, `inference_done`, `encode_done`)
   - `call_ts_epoch_ms`      — timestamp wanneer de module *gevraagd/geïnitieerd* werd om te beginnen met werken (epoch ms, UTC)

**Latentie (per-module):** `produced_ts_epoch_ms` − `call_ts_epoch_ms`.
Dit is de duur tussen het moment dat een subsystem werd gevraagd om te beginnen met werken (call) en het moment dat het een resultaat produceerde. Gebruik **geen** inter-frame intervallen of tijd tussen logs als latency.
Modules moeten `call_ts_epoch_ms` emitteren op het moment dat werk in de wachtrij wordt geplaatst/aangevraagd (bijv. camera: wanneer de opnameaanvraag wordt gepost; inferentie: wanneer een frame de inferentie-wachtrij binnenkomt).

**Voorbeeld CSV header & sample lines:**

Common prefix: `produced_ts_epoch_ms,module,thread_id,event,call_ts_epoch_ms,<module-specifieke-kolommen...>`

Camera voorbeeld:
```csv
produced_ts_epoch_ms,module,thread_id,event,call_ts_epoch_ms,frame_id,width,height,exposure_ms,copy_time_ms
1712681235123,camera,2234,frame_available,1712681235120,12345,1536,864,25,0.12
```

TPU voorbeeld:
```csv
produced_ts_epoch_ms,module,thread_id,event,call_ts_epoch_ms,inference_ms,input_w,input_h,tpu_temp_c
1712681235789,tpu,2299,inference_done,1712681235787,2.1,300,300,45.3
```

**Plaatsing & bestandsnaam regels:**
Logbestanden moeten worden opgeslagen in een subdirectory onder de geconfigureerde `log_path`, met de naam van het subsystem (bijv. `/logs/camera/`, `/logs/tpu/`). De bestandsnaamconventie moet `module_YYYY_MM_DD_HH:MM.csv` zijn. Er mogen maximaal 3 rotaties van logbestanden per subsystem worden bewaard.

**Opmerking:** Alle modules moeten een consistente tijdbron gebruiken (epoch ms UTC), of de monotonic-klok + offset methode documenteren indien gebruikt.


**Resultaat:** Subsystemale prestatiegrenzen vastgesteld → goedkeuring Stage 1

---

### Stage 1: Systeembrede C++ implementatie & bottleneckanalyse

**Doel:** Alle core realtime acties via **`logic` module**, mutex-gebaseerde `ThreadSafeQueue`.

**Core Subsystems:**

* **Logic (`logic`)**

  * 3D ballistiek
  * Hit-scan
  * Servo-actuatie
  * Veiligheids-/onzekerheidspropagatie (RK4 import uit `system_monitor`)
  * Logging & microbenchmarks

* **Camera & DMA:** `src/camera_capture.*`, `src/buffer_pool.h`, `src/pipeline_structs.h`

* **TPU Inferentie:** `src/inference.*` + `.tflite` model

* **System Monitor:** `src/system_monitor.*` — CPU, RAM, temperatuur, thread health, supervisie

* **Overige non-RT modules:** `src/config_loader.*`, `src/h264_encoder.*`

**Logging:** Configureerbaar via `config.json`.

**Gating criteria:** Alle core realtime functies draaien zelfstandig → goedkeuring Stage 2

---

### Stage 2: Volledige integratie & zero-copy optimalisatie

**Doel:** End-to-end pipeline met DMA-delende buffers en validatie over 100.000 frames.

* Zero-copy pipeline: `src/camera_capture.*` → `logic` → `src/inference.*` → `src/video_overlay_processor.*`
* Video stream via UDP/RTP of RTSP
* Bounding boxes en telemetrie via ZeroMQ PUB/SUB
* Fallback switching getest met correcte logging

**Prestatie eisen:**

* E2E latency < 100 ms met <5% jitter
* TPU throughput ≥ 90 FPS per 100 FPS capture
* Temperatuur stabiel, geen throttling

**Gating criteria:** Pipeline operationeel, fallback modes getest, video & telemetry streams werkend → goedkeuring Stage 3

---

### Stage 3: Validatie & verificatie

**Doel:** 4-uur stress test en schietzaal-validatie van alle kritieke systemen.

* Continue logging van thermiek en jitter naar CSV + PNG grafieken
* Onzekerheidspropagatie verificatie in schietzaal
* `logic` module verificatie van 3D ballistiek, hit-scan en actuatie
* `system_monitor` supervisie en logging testen

**Stabiliteitseis:** E2E latency binnen 5% van nominale waarde over volledige testduur

**Gating criteria:** Veiligheidsmarges bevestigd, thermische stabiliteit bewezen, onzekerheidsmodel gevalideerd → goedkeuring Stage 4

---

## 🧩 Core Data Structures and Threading Model

### `ImageData` (`pipeline_structs.h`)

* Producer: `CameraCapture`
* Consumer: `logic`, `VideoOverlayProcessor`

### `OrientationData` (`pipeline_structs.h`)

* Producer: `OrientationSensor`
* Consumer: `logic`

### `DetectionResult` (`pipeline_structs.h`)

* Producer: `logic`/`InferenceEngine`
* Consumer: `VideoOverlayProcessor`, telemetry via ZeroMQ

### `TrackedObject` (`src/logic.h`)

* Represents een object dat over tijd wordt gevolgd
* Eigentijdelijk beheerd door `logic`

```

