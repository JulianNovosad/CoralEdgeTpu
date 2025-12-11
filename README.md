Hier is je README volledig opnieuw, nu zonder enige emoji’s:

````markdown
# CoralEdgeTpu

**C++ Edge TPU inference stack voor de Google Coral M.2 Accelerator**

Deze repository is een volledige heropbouw van de oude, slecht onderhouden Coral/TensorFlow Lite toolchain — modern, stabiel, reproduceerbaar en volledig gericht op **native C++ inferencing op de Google Coral M.2 TPU**.

Het project bevat álle bronbestanden, patches, TensorFlow-Lite headers, Flatbuffers, build-scripts en dependency-versies die nodig zijn om de Edge TPU 100% deterministisch en offline te kunnen builden. Dit elimineert versie-hell, Python-dependency chaos en ontbrekende upstream files.

---

## Functionaliteit

* Volledige C++ applicatie die de gehele pijplijn beheert.
* Multi-threaded architectuur voor camera capture, inferentie, encoding en streaming.
* Gebruik van `libcamera` voor moderne en efficiënte camera-aansturing op de Raspberry Pi.
* Edge TPU-versnelde inferentie met TensorFlow Lite.
* H.264 video encoding en realtime streaming via UDP/RTP of RTSP.
* Bounding boxes, reticle coordinates, en telemetrie via ZeroMQ PUB/SUB.
* Uitgebreide configuratie via `config.json`.
* Robuuste logging met microbenchmarks en CSV output.
* Centrale supervisie via `system_monitor` (CPU, RAM, temperatuur, thread health, etc.)
* Alle core realtime actie en beslissingen via `logic` module (3D ballistiek, hit-scan, servo-actuatie, veiligheids-/onzekerheidspropagatie).

---

## Bouwen en Draaien

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

## Repository Structuur

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

## Stage-Gate Plan

### Stage 0: Technische haalbaarheid & prestatiegrenzen

**Doel:** Basale throughput en latentie meten van kernsubsystemen zonder volledige integratie.

**Subsystemen:**

* Logic: `logic.*` : actuation, 3D ballistiek, hit-scan, veiligheids-/onzekerheidspropagatie
* Camera: `src/camera_capture.*`, `src/buffer_pool.h`, `src/pipeline_structs.h`
* TPU: `src/inference.*` + `.tflite model`
* Encoder: `src/h264_encoder.*`

**Gating Requirements:**

* Alle kernsubsystemen draaien zonder **segfaults**.
* Camera frames gemiddeld **≥ 120 FPS**.
* TPU inferentie gemiddeld **≥ 120 FPS**.
* Logging per subsystem correct gegenereerd (`/logs/<module>/`), bestandsnaamconventie gevolgd.
* Kernel-aanpassingen (PCIe, IRQ-affiniteiten, MSI-X) gedocumenteerd (`lspci -vvv`, `sudo dmesg | grep -i apex`).

#### Logging format (Stage 0)

Stage-0 logging gebruikt een **universele header** over alle modules heen. Dit betekent dat elk logbestand dezelfde set kolommen bevat, ongeacht de module die de log produceert. Niet-relevante metrische kolommen worden opgevuld met een standaardwaarde (bijv. `-1`).

De header bevat een uniforme prefix van kolommen, gevolgd door alle mogelijke module-specifieke metrics van het hele systeem.

**Unified prefix columns (epoch UTC in milliseconden):**

* `produced_ts_epoch_ms`  — timestamp wanneer deze logregel werd geproduceerd (epoch ms, UTC)
* `module`                — module naam: `CameraCapture`|`InferenceEngine`|`H264Encoder`|`LogicModule`|`SystemMonitor`
* `thread_id`             — numerieke OS thread id (TID)
* `event`                 — korte label (bijv. `frame_captured`, `inference_done`, `encode_done`)
* `call_ts_epoch_ms`      — timestamp wanneer de module *gevraagd/geïnitieerd* werd om te beginnen met werken (epoch ms, UTC)

**Latentie:** `produced_ts_epoch_ms` − `call_ts_epoch_ms`.

**Voorbeeld Universele CSV header & sample lines:**

```csv
produced_ts_epoch_ms,module,thread_id,event,call_ts_epoch_ms,camera_frame_id,camera_width,camera_height,camera_exposure_ms,camera_copy_time_ms,tpu_inference_ms,tpu_input_w,tpu_input_h,tpu_temp_c,encoder_encode_ms,encoder_total_encoded_frames,encoder_average_fps,logic_metric_ballistics,logic_metric_hit_scan,logic_metric_servo_actuation,sysmon_cpu_temp_c,sysmon_cpu_usage_percent,sysmon_mem_usage_percent,p50_latency_ms,p95_latency_ms,p99_latency_ms,average_fps,total_frames_processed_or_inferences,average_latency_ms,details
1765476550098,CameraCapture,520472000761713998,main_frame_processed,1765476549812,0,1536,864,0,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,""
1765476550131,InferenceEngine,-2932907159835259584,inference_done,1765476549812,-1,-1,-1,-1,-1,25,300,300,32.05,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,""
```

**Plaatsing & bestandsnaam regels:**
Logbestanden moeten worden opgeslagen in een subdirectory onder de geconfigureerde `log_path`, met de naam van het subsystem (bijv. `/logs/camera/`, `/logs/tpu/`). De bestandsnaamconventie moet `module_YYYY_MM_DD_HH:MM.csv` zijn. Er mogen maximaal 3 rotaties van logbestanden per subsystem worden bewaard.

**Opmerking:** Alle modules moeten een consistente tijdbron gebruiken (epoch ms UTC), of de monotonic-klok + offset methode documenteren indien gebruikt.

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

**Gating Requirements:**

* Alle core realtime functies draaien zelfstandig zonder **segfaults**.
* `ThreadSafeQueue` en mutex-gebaseerde synchronisatie veroorzaken geen deadlocks.
* Camera frames en TPU inferentie consistent met Stage 0 performance (≈120 FPS).
* Logging consistent en correct geformatteerd.

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

**Gating Requirements:**

* Pipeline draait operationeel zonder crashes of segfaults over ≥ 100.000 frames.
* Zero-copy buffers correct gedeeld.
* Video stream via UDP/RTP of RTSP stabiel.
* Bounding boxes en telemetrie correct.
* Geen logregel bevat “error” of “segfault”.
* Latentie ≤ 100 ms met <5% jitter.
* TPU throughput ≥ 90 FPS per 100 FPS capture.
* Temperatuur stabiel, geen throttling.

---

### Stage 3: Validatie & verificatie

**Doel:** 4-uur stress test en schietzaal-validatie van alle kritieke systemen.

* Continue logging van thermiek en jitter naar CSV + PNG grafieken
* Onzekerheidspropagatie verificatie in schietzaal
* `logic` module verificatie van 3D ballistiek, hit-scan en actuatie
* `system_monitor` supervisie en logging testen

**Stabiliteitseis:** E2E latency binnen 5% van nominale waarde over volledige testduur

**Gating Requirements:**

* Stress test draait zonder crashes of segfaults.
* Thermische en timing logs consistent en binnen tolerantie.
* Logic module correct uitgevoerd: 3D ballistiek, hit-scan, servo-actuatie.
* System monitor supervisie en logging operationeel.
* E2E latency binnen 5% van nominale waarde over gehele testduur.

---

## Core Data Structures and Threading Model

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
```
