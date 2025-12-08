# CoralEdgeTpu

**C++ Edge TPU inference stack voor de Google Coral M.2 Accelerator**

Deze repository is een volledige heropbouw van de oude, slecht onderhouden Coral/TensorFlow Lite toolchain — maar dan modern, stabiel, reproduceerbaar en volledig gericht op **native C++ inferencing op de Google Coral M.2 TPU**.

Het project bevat álle bronbestanden, patches, TensorFlow-Lite headers, Flatbuffers, build-scripts en dependency-versies die nodig zijn om de Edge TPU 100% deterministisch en offline te kunnen builden.
Dit elimineert versie-hell, Python-dependency chaos en ontbrekende upstream files.

---

## ✨ Functionaliteit

* Volledige C++ inference pipeline:

  * Model laden
  * Preprocessing
  * Edge TPU delegatie
  * Postprocessing (bounding boxes, scores, klassen)
* Handmatig gebouwde TensorFlow Lite 2.5.0 voor ARM64
* Gepatchte en gestabiliseerde Flatbuffers-versie
* Volledige dependency-graph **meegeleverd in deze repo**
* Ondersteuning voor `*_edgetpu.tflite` AI modellen.

---

## 🌐 Netwerkpoorten (TCP/UDP)

Dit project maakt gebruik van de volgende netwerkpoorten voor communicatie tussen de Raspberry Pi en verbonden client-applicaties (telefoon/workstation). De configuratie van deze poorten is te vinden in `config.json` onder de `application.network_ports` sectie.

### Pi → Telefoon (Output streams)

*   **1001 (TCP): Livestream Video**
    *   **Doel:** Realtime H.264 videostream van de Pi naar de client (1536x864 resolutie).
    *   **Module:** `HttpServer` (`src/http_server.cpp`) - fungeert als een MJPEG-stream of WebSocket-server voor de H.264-data.
*   **1002 (TCP): Bounding Box Stream**
    *   **Doel:** Stream van detectieresultaten (bounding boxes, klassen, scores) van de Pi naar de client.
    *   **Module:** `HttpServer` (`src/http_server.cpp`) - via een WebSocket-verbinding.
*   **1003 (TCP): Reticle Coördinaat (eenmalig)**
    *   **Doel:** Eenmalige verzending van het berekende richtpunt (reticle-coördinaat) voor kalibratie of feedback.
    *   **Module:** `HttpServer` (`src/http_server.cpp`) - via een specifiek HTTP-endpoint of WebSocket-bericht.
*   **1004 (TCP): Status/Telemetrie Stream**
    *   **Doel:** Continue stream van systeemstatus en telemetrie ('locked', 'distance', 'speed') van de Pi naar de client.
    *   **Module:** `HttpServer` (`src/http_server.cpp`) - via een WebSocket-verbinding.

### Telefoon → Pi (Input streams)

*   **2001 (UDP): Oriëntatie (Yaw)**
    *   **Doel:** Ontvangst van 'yaw' oriëntatiedata van de telefoon naar de Pi.
    *   **Module:** `OrientationSensor` (`src/orientation_sensor.cpp`) - luistert naar UDP-pakketten.
*   **2002 (UDP): Oriëntatie (Pitch)**
    *   **Doel:** Ontvangst van 'pitch' oriëntatiedata van de telefoon naar de Pi.
    *   **Module:** `OrientationSensor` (`src/orientation_sensor.cpp`) - luistert naar UDP-pakketten.
*   **2003 (UDP): Oriëntatie (Roll)**
    *   **Doel:** Ontvangst van 'roll' oriëntatiedata van de telefoon naar de Pi.
    *   **Module:** `OrientationSensor` (`src/orientation_sensor.cpp`) - luistert naar UDP-pakketten.

---

## 📂 Repository structuur

```
CoralEdgeTpu/
 ├── build/                   # Build output
 ├── include/tensorflow/lite/ # TFLite headers (gepatcht & compleet)
 ├── lib/                     # libtensorflow-lite.so + libedgetpu.so
 ├── model/                   # Voorbeeld EdgeTPU modellen
 ├── src/                     # C++ inference engine
 ├── patches/                 # TFLite / docs / profiling patches
 ├── detector/                # Object-detection utilities
 ├── logs/                    # Build/runtime logs
 ├── build_tflite.sh          # Rebuild script
 ├── CMakeLists.txt           
 └── Makefile
```

---

## 🚀 Bouwen van het project

### Vereisten

* Raspberry Pi 5
* Google Coral M.2 TPU (PCIe)
* `libedgetpu1-std`
* CMake ≥ 3.16
* g++ ≥ 10
* Bazelisk (meegeleverd)

### TFLite bouwen (alleen nodig bij wijzigingen)

```
chmod +x build_tflite.sh
./build_tflite.sh
```

### Project bouwen

```
mkdir build
cd build
cmake ..
make -j4
```

---

# 🔧 Hardware & Kernel Vereisten (BELANGRIJK!)

De Coral M.2 TPU werkt alleen stabiel op de Raspberry Pi 5 met zeer specifieke kernel/PCIe-instellingen.
De accelerator faalt zonder deze instellingen (geen MSI-interrupts, enumeratie mislukt, gasket error 43, enz.).

---

## 1. Juiste Kernel

Je moet exact deze configuratie gebruiken:

* **Kernel:** `6.6.51`
* **Architectuur:** `v8` (AArch64)
* **Page size:** **4096 bytes**

Controleer met:

```
getconf PAGE_SIZE
uname -a
```

---

## 2. Juiste PCIe-instellingen

In `/boot/firmware/config.txt`:

```
dtparam=pciex1
dtparam=pciex1_gen=2
kernel=kernel8.img
```

In `/boot/firmware/config.txt`:

```
pcie_aspm=off
```

Reboot hierna.

---

## 3. APEX/Gasket driver (TPU kernel driver)

De TPU vereist Google's officiële kernelmodules:

Repo:
[https://github.com/google/gasket-driver](https://github.com/google/gasket-driver)

Je kan ze native builden, maar we leveren de .deb package dat al voor de Pi gebouwd is. 

Als je het toch zelf build: 

```
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
sudo make -C /lib/modules/$(uname -r)/build M=$(pwd) modules_install
sudo depmod -a
```

Controleer:

```
lsmod | grep gasket
lsmod | grep apex
```

---

## 4. Verplichte Raspberry Pi 5 DTB-patch

De RPi5 gebruikt een foute `msi-parent` voor de PCIe root complex.
Hierdoor werkt de Coral TPU **niet**, omdat MSI-interrupts nooit aankomen.

### Patchprocedure

Back-up:

```
sudo cp /boot/firmware/bcm2712-rpi-5-b.dtb /boot/firmware/bcm2712-rpi-5-b.dtb.bak
```

DTB → DTS:

```
dtc -I dtb -O dts /boot/firmware/bcm2712-rpi-5-b.dtb -o ~/test.dts
nano ~/test.dts
```

Zoek node:

```
pcie@110000
```

Wijzig:

```
msi-parent = <0x2f>;
```

Naar:

```
msi-parent = <0x68>;
```

Compileer terug:

```
dtc -I dts -O dtb ~/test.dts -o ~/test.dtb
sudo mv ~/test.dtb /boot/firmware/bcm2712-rpi-5-b.dtb
```

Reboot.

Daarna verschijnt de TPU correct in:

```
lspci -v
dmesg | grep gasket
dmesg | grep apex
```

---

# Geoptimaliseerd Stage‑Gate Plan (Herzien v2)

Dit document bevat het bijgewerkte, beknopte en actionabele Stage‑Gate plan voor het CoralEdgeTpu‑project. Het vervangt de vorige versies en reflecteert de huidige architectuur‑ en meetbeslissingen.

---

## Stage 0: Technische haalbaarheid & prestatiegrenzen

**Doel:** Basale throughput en latentie meten van kernsubsystemen zonder volledige integratie.

**Te meten subsystemen (kort):**

* **Logica module (RT‑thread single source):** alle ballistiek‑gerelateerde code (sensorfusie/IMU, object tracking, hit‑scan / voorspeld inslagpunt, veiligheids‑ & onzekerheidspropagatie, servo‑actuatie API). Implementatiebestand: `src/logic.*` (inclusief `src/object_tracker.*` functionality). *Opmerking:* servo reactietijd wordt **niet** runtime gemeten (zie servo_latency.csv).
* **Camera (RT‑thread):** `src/camera_capture.*` — FPS en frame latency (gescheiden meting).
* **TPU (inference):** `src/inference.*` + `.tflite` model — inferentie latentie (gescheiden meting).

> **Belangrijk:** De "logica module" is een enkele bronfile (of set tightly‑coupled bestanden) die sensorfusie, objecttracking, hit‑scan, veiligheidslogica en de servo/actuator interface **centraliseert**. Camera en TPU blijven als aparte, meetbare subsystemen.

**Gating criteria (Stage 0):**

* Meetrapport per subsystem met minimaal: p50/p95/p99 latency, throughput (FPS of_ops/s), en korte beschrijving van meetmethode.
* Logica module: time‑to‑prediction (end‑to‑end door sensorfusie → hit‑scan → predictie) gemeten en gedocumenteerd (servo latency **niet** inbegrepen).
* Kernel‑aanpassingen (PCIe, IRQ‑affiniteiten, MSI‑X) gedocumenteerd indien toegepast.
* Theorie‑document voor ballistiek + onzekerheidspropagatie met acceptatiegrenzen.

**Resultaat:** Subsystemale prestatiegrenzen vastgesteld → Goedkeuring voor Stage 1

---

## Stage 1: Systeembrede C++ implementatie & bottleneckanalyse

**Doel:** Alle core subsystems functioneel systeem‑breed met expliciete code‑mappings, correcte synchronisatie en volledige documentatie.

**Core Subsystems (directe code‑mappings):**

* **AI + Logica (Orchestrator):** `src/logic.*` — bevat nu ook object tracking (`object_tracker`), orientatiesensoren, hit‑scan (voorspeld inslagpunt tegenover de draadkruis), veiligheids/onzekerheidspropagatie en de servo actuatie API (PCA9685 interface / safety controller behaviour).
* **Camera & DMA (RT‑thread):** `src/camera_capture.*`, `src/buffer_pool.h`, `src/pipeline_structs.h`.
* **TPU Inferentie:** `src/inference.*` + `.tflite` model.
* **Overige non‑RT modules:** `src/video_overlay_processor.*`, `src/config_loader.*`, `src/h264_encoder.*`, `src/http_server.*`.

**Nieuwe/Vernieuwde componenten (Stage 1):**

* **Hardware/Software Killswitch:** `src/process_supervisor.*`.
* **Fallback modes b/c/d:** in `src/logic.*` met expliciete logging en activatie‑criteria.
* **Software monitoring:** temperatuur & resource‑gebruik via `src/process_supervisor.*` (CSV logs per module).
* **Logging:** `src/util_logging.*` levert gestructureerde CSV logs naar `/home/pi/CoralEdgeTpu/logs/` met rotatie (3 bestanden per module).

  * **CSV Formaat:** `monotonic_time_ns,module,stage,p50,p95,p99,temp,fps`.
  * **Timestamp bron:** `CLOCK_MONOTONIC_RAW`.

**Gating criteria (Stage 1):**

* Alle subsystems draaien onafhankelijk en kunnen start/stop en gezondheidschecks doorstaan.
* `src/logic.*` moet een bewezen veiligheids‑ en onzekerheidspropagatiemodule bevatten (alle inputs behalve servo latency worden hierin gebruikt).
* Communicatie tussen RT‑modules gebruikt lock‑free of RT‑vriendelijke queues (bij voorkeur `boost::lockfree::spsc_queue` of equivalent) met duidelijk gedefinieerde ownership/invariants in `README.md`.
* Documentatie: kern‑`structs` en `unions` zijn beschreven met grootte, alignment en thread ownership.
* Logging conform CSV‑format en rotatie zoals hierboven. → Goedkeuring voor Stage 2

---

## Stage 2: Volledige integratie & performance engineering

**Doel:** End‑to‑end pipeline met minimale copies en real‑time karakter voor productie‑achtige workloads (doel: 100k frames validatie). We streven naar zero‑copy waar mogelijk zonder kernel/library rebuilds; als directe DMA buffer‑deling niet haalbaar is, implementeer de snelst mogelijke RT‑vriendelijke fallback (geen rebuilding van `libedgetpu` of `tflite`).

**Integratie focus:**

* **Zero‑copy pipeline:** probeer `memfd`/dmabuf deling tussen `src/camera_capture.*` → `src/inference.*` → `src/video_overlay_processor.*`. Indien dit niet haalbaar zonder grote rebuilds, ontwerp en implementeer de snelst mogelijke, veilige RT‑fallback (gedeelde preallocated buffers + `mmap`/synchronisatie).
* **Volledige integratie:** alle subsystems werken samen zonder onnodige shims.
* **Fallback‑switching:** gedocumenteerde tests en log‑meldingen via `src/util_logging.*` en UI‑meldingen via `src/http_server.*`.

**Frontend (extern UI):**

* De client UI draait **niet** op de Pi zelf maar op een extern workstation of mobiele app.
* Vereisten client: ontvangt live video, ontvangt bounding boxes én het **voorspelde ballistische inslagpunt**; deze drie elementen (video + bounding boxes + voorspeld inslagpunt) moeten zichtbaar en gesynchroniseerd zijn op de client. Het statische crosshair is alleen een referentie en hoeft niet te worden gematcht op tijd‑basis.
* Synchronisatie: client/stream moet timestamps of sequentienummers gebruiken om overlay data correct op frames te plakken.

**Prestatie eisen (100.000 frames):**

* **E2E latentie:** ≤ 100 ms gemiddeld, jitter < 5% (p95/p99 within budget).
* **TPU throughput:** ≥ 90 FPS op 100 FPS capture (90% realtime verwerking target).
* **Thermisch:** geen throttling tijdens lange tests.
* **Jitter monitoring:** continue logging naar CSV.

**Build deliverable:** foutloos systeemimage voor Raspberry Pi 5B (target kernel en distro documenteren in release notes).

**Gating criteria (Stage 2):**

* Zero‑copy of snelle RT fallback geïmplementeerd en getest.
* Alle fallback modes en UI‑integratie getest.
* Prestatie‑doelen gehaald of gedocumenteerde mitigaties beschikbaar. → Goedkeuring voor Stage 3

---

## Stage 3: Validatie & verificatie

**Doel:** Lange duurstress‑tests en veldvalidatie van kritieke systemen.

**Test procedures:**

* **4‑uur stress test:** continue logging van thermiek en jitter naar CSV + geautomatiseerde PNG grafieken.
* **Vuurleiding verificatie:** tests in gecontroleerde omgeving/schietzaal: effect op vuurbeslissingen meten en validatie van onzekerheidsmodel.
* **Killswitch verificatie:** hardware/software killswitch testen tijdens volle belasting.
* **Experimentatie:** meerdere resoluties en configuraties via `src/config_loader.*`.

**Stabiliteitseis:** E2E latency blijft binnen 5% van nominale waarde over de testduur.

**Gating criteria (Stage 3):**

* Veiligheidsmarges bevestigd.
* Thermische stabiliteit aangetoond.
* Vuurleidingsmodel gevalideerd. → Goedkeuring voor Stage 4

---

## Opslag van voorkennis en artefacten

* **Servo latency:** `servo_latency.csv` (pre‑measured), wordt gebruikt als input voor analyses maar **niet** voor runtime onzekerheidspropagatie.
* **Meetdata & logs:** `/home/pi/CoralEdgeTpu/logs/` met per‑module subdirectory en rotatie (3 bestandshistorie).
* **Release snapshots:** bewaar git‑tags of commit‑hashes die gebruikt zijn voor reproduceerbaarheid.

---

## Notities & richtlijnen

* Nooit harde paden coderen; gebruik `find`/`grep` voor discovery in scripts en CI.
* Na 2–3 codewijzigingen: run `./build.sh` en valideer build logs.
* Debugging: gebruik `gdb`/`valgrind`/`perf` voor concurrency/latency analysis.
* Structureer `src/logic.*` zodat alle veiligheidskritieke routines (uncertainty propagation, safety checks) duidelijk gemarkeerd en testbaar zijn.

---

## Opslag van voorkennis en artefacten

*   **Servo latency:** `servo_latency.csv` (pre‑measured), wordt gebruikt als input voor analyses maar **niet** voor runtime onzekerheidspropagatie.
*   **Meetdata & logs:** `/home/pi/CoralEdgeTpu/logs/` met per‑module subdirectory en rotatie (3 bestandshistorie).
*   **Release snapshots:** bewaar git‑tags of commit‑hashes die gebruikt zijn voor reproduceerbaarheid.

---

## Notities & richtlijnen

* Nooit harde paden coderen; gebruik `find`/`grep` voor discovery in scripts en CI.
* Na 2–3 codewijzigingen: run `./build.sh` en valideer build logs.
* Debugging: gebruik `gdb`/`valgrind`/`perf` voor concurrency/latency analysis.
* Structureer `src/logic.*` zodat alle veiligheidskritieke routines (uncertainty propagation, safety checks) duidelijk gemarkeerd en testbaar zijn.

---

## Core Data Structures and Threading Model

This section documents critical data structures (`struct`s and `union`s) used in the pipeline, detailing their estimated size, memory alignment, and typical thread ownership/access patterns. This information is vital for understanding memory layout, potential cache efficiency, and thread-safety considerations.

### `ImageData` (`pipeline_structs.h`)
*   **Purpose:** Represents a raw image frame, typically consumed by `InferenceEngine` and `VideoOverlayProcessor`.
*   **Estimated Size (64-bit ARM64):** ~32 bytes
    *   `BufferPool<uint8_t>::PooledPtr buffer` (std::shared_ptr): 8 bytes
    *   `size_t width, height`: 8 bytes each (16 bytes total)
    *   `std::chrono::steady_clock::time_point timestamp`: 8 bytes
*   **Estimated Alignment:** 8 bytes (due to `std::shared_ptr` and `std::chrono::steady_clock::time_point`)
*   **Thread Ownership/Access:**
    *   **Producer:** `CameraCapture` (writes all members, owns `buffer`'s `PooledPtr` temporarily).
    *   **Consumer:** `InferenceEngine`, `VideoOverlayProcessor` (reads all members).
    *   **Shared:** The underlying raw `uint8_t` buffer pointed to by `PooledPtr` is shared and managed by `BufferPool`. Access to the buffer itself is assumed to be read-only by consumers once pushed into a queue.

### `IMUData` (`pipeline_structs.h`)
*   **Purpose:** Stores Inertial Measurement Unit readings (accelerometer, gyroscope, magnetometer) and a timestamp.
*   **Estimated Size (64-bit ARM64):** ~44 bytes
    *   `float accel_x, accel_y, accel_z`: 4 bytes each (12 bytes total)
    *   `float gyro_x, gyro_y, gyro_z`: 4 bytes each (12 bytes total)
    *   `float mag_x, mag_y, mag_z`: 4 bytes each (12 bytes total)
    *   `std::chrono::steady_clock::time_point timestamp`: 8 bytes
*   **Estimated Alignment:** 8 bytes (due to `std::chrono::steady_clock::time_point`)
*   **Thread Ownership/Access:**
    *   **Producer:** Dedicated IMU reader thread (writes all members).
    *   **Consumer:** `LogicModule` (reads all members).

### `DetectionResult` (`pipeline_structs.h`)
*   **Purpose:** Represents a single object detection, including class, score, bounding box, and detection timestamp.
*   **Estimated Size (64-bit ARM64):** ~32 bytes
    *   `int class_id`: 4 bytes
    *   `float score`: 4 bytes
    *   `float xmin, ymin, xmax, ymax`: 4 bytes each (16 bytes total)
    *   `std::chrono::high_resolution_clock::time_point timestamp`: 8 bytes
*   **Estimated Alignment:** 8 bytes (due to `std::chrono::high_resolution_clock::time_point`)
*   **Thread Ownership/Access:**
    *   **Producer:** `InferenceEngine` (writes all members).
    *   **Consumer:** `VideoOverlayProcessor`, `LogicModule` (reads all members).

### `TargetStateForBallistics` (`src/logic.h`)
*   **Purpose:** Encapsulates the state of a detected target relevant for ballistics, including detection details, estimated distance, and IMU data from the device.
*   **Estimated Size (64-bit ARM64):** ~84 bytes
    *   `DetectionResult detection`: ~32 bytes
    *   `double distance`: 8 bytes
    *   `IMUData imu_data`: ~44 bytes
*   **Estimated Alignment:** 8 bytes
*   **Thread Ownership/Access:**
    *   **Producer:** `main` loop (constructs from `DetectionResult` and mock `IMUData` for now). Will eventually be an orchestrator combining different data sources.
    *   **Consumer:** `LogicModule` (reads all members in `process` method).

### `TrackedObject` (`src/logic.h`)
*   **Purpose:** Represents a single object being tracked over time, maintaining its estimated 3D position, velocity, and tracking metadata.
*   **Estimated Size (64-bit ARM64):** ~80 bytes
    *   `long id`: 8 bytes
    *   `DetectionResult last_detection`: ~32 bytes
    *   `float pos_x, pos_y, pos_z`: 4 bytes each (12 bytes total)
    *   `float vel_x, vel_y, vel_z`: 4 bytes each (12 bytes total)
    *   `std::chrono::high_resolution_clock::time_point last_update_time`: 8 bytes
    *   `int hit_streak, missed_frames`: 4 bytes each (8 bytes total)
*   **Estimated Alignment:** 8 bytes
*   **Thread Ownership/Access:**
    *   **Producer:** `LogicModule` (creates and updates `TrackedObject` instances based on new detections).
    *   **Consumer:** `LogicModule` (reads for tracking, ballistics, safety checks). `active_tracks_` vector is owned by `LogicModule`.
