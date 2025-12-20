---
# Avant-garde Mk V Wapenveiligheidssysteem

Dit project bevat de broncode en documentatie voor het **Avant-garde M1-Delta Mk II**, een geavanceerd vuurleidings- en veiligheidssysteem ontwikkeld als profielwerkstuk (PWS).

## Wat is het?

Het Avant-garde systeem is een compact, *real-time* computerplatform dat op een wapen (of replica/airsoft) gemonteerd kan worden. Het koppelt AI-objectherkenning aan ballistische modellering om de veiligheid en nauwkeurigheid van de gebruiker te verhogen.

Het systeem fungeert als een "slimme beveiliging": het blokkeert de trekker fysiek en geeft deze alleen vrij wanneer de gebruiker op een gevalideerd doelwit mikt en de raakkans >90% is.

## Wat doet het?

Het systeem voert continu de volgende taken uit in een *high-performance* loop:

* 
**Doeldetectie:** Herkent specifieke schietschijven met behulp van een zelf-getraind *MobileNetSSD* model (INT8 quantized).


* **Safety Gating:** Een servomotor blokkeert de trekker. Het systeem deblokkeert alleen als het ballistische raakpunt binnen de 'bullseye' (13x13cm) valt.


* 
**Ballistische Berekeningen:** Berekent het inslagpunt op basis van afstand, kogelbaan en sensordata.


* 
**Feedback:** Streamt video met *augmented reality* overlay (bounding boxes, status) naar een Android-app.



## Hoe doet hij het?

Het systeem is ontworpen met een extreem strak **latency-budget**.

### Latency Budget & Prestaties

De harde eis voor de totale *end-to-end* latentie (van foton tot servobeweging) is **<100 ms**.

* 
**Actuatie-vertraging:** De mechanische servomotor heeft een gemeten gemiddelde reactietijd van **70 ms**.


* **Software-budget:** Dit laat slechts **30 ms** over voor de volledige software-pipeline (camera capture, AI-inferentie, ballistiek en logica).

Om dit budget te halen, zijn de volgende optimalisaties toegepast:

### Software Architectuur (Plan Delta)

De software is geschreven in **C++** zonder runtime-overhead (geen Python/Containers in productie).

* 
**Zero-Copy Pipelines:** Gebruik van DMA-buffers en `memfd` om data tussen Camera, TPU en Display te delen zonder CPU-kopieerslagen.


* 
**Custom Kernel:** Een handmatig gepatched Linux Kernel (6.6.51+) om 128 MSI-X interrupts te forceren voor de Coral TPU.


* 
**CPU Isolatie:** Kritieke processen draaien op geisoleerde CPU-cores om *jitter* te minimaliseren.



### Hardware

* **Compute:** Raspberry Pi 5.
* **AI:** Google Coral M.2 Edge TPU (PCIe).
* 
**Actuatie:** MG995 Servo via PCA9685 PWM-driver.



---

### STAGE GATE PLAN

### Stage 0: Technische haalbaarheid & prestatiegrenzen

**Doel:** Basale throughput en latentie meten van kernsubsystemen zonder verplichte volledige integratie.

**Subsystemen:**

* Logic: `logic.*` : actuation, 3D ballistiek, hit-scan, veiligheids-/onzekerheidspropagatie
* Camera: `src/camera_capture.*`, `src/buffer_pool.h`, `src/pipeline_structs.h`
* TPU: `src/inference.*` + `.tflite model`
* Encoder: `src/h264_encoder.*`

**Gating Requirements:**

* Kernsubsystemen draaien zonder **segfaults** of crashes.
* Camera frames gemiddeld **≥ 120 FPS**.
* TPU inferentie gemiddeld **≥ 120 FPS**.
* Logging per subsystem correct gegenereerd (`/logs/<module>/`), bestandsnaamconventie gevolgd.
* **Logbestanden volledig, chronologisch, consistent met universele CSV header.**
* Kernel-aanpassingen (PCIe, IRQ-affiniteiten, MSI-X) gedocumenteerd (`lspci -vvv`, `sudo dmesg | grep -i apex`).
* Applicatie stopt netjes na gespecificeerde `RUN_DURATION` of `stop_after`.

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

**Gating Requirements:**

* Alle core realtime functies draaien zelfstandig zonder **segfaults**.
* `ThreadSafeQueue` en mutex-gebaseerde synchronisatie veroorzaken geen deadlocks.
* Camera frames en TPU inferentie consistent met Stage 0 performance (≈120 FPS).
* Logging consistent, correct geformatteerd, en chronologisch.
* **Applicatie reageert op stop-signalen en runtime parameters; gegarandeerd beëindigen van alle threads en modules binnen ±100 ms van stop-aanvraag.**
* Debug logs aanwezig bij fout of waarschuwing (bijv. `InferenceEngine: No detections`) met voldoende context om oorzaak te reproduceren.

**Logging format:**

* Universele CSV header zoals Stage 0, met extra kolommen voor Stage 1 metrics: thread latency, queue sizes, TPU inference count, encoder fps, logic metrics.
* **Logs mogen geen ontbrekende of negatieve waarden bevatten.**
* Alle subsystemen gebruiken dezelfde tijdbron (epoch ms UTC) of gedocumenteerde offset.

---

### Stage 2: Volledige integratie & zero-copy optimalisatie

**Doel:** End-to-end pipeline met DMA-delende buffers en validatie over 100.000 frames.

**Pipeline:**

* `src/camera_capture.*` → `logic` → `src/inference.*` → `src/video_overlay_processor.*`
* Video stream via UDP/RTP of RTSP
* Bounding boxes en telemetrie via ZeroMQ PUB/SUB
* Fallback switching getest met correcte logging

**Prestatie eisen:**

* End-to-end latency < 100 ms met <5% jitter
* TPU throughput ≥ 90 FPS per 100 FPS capture
* Temperatuur stabiel, geen throttling

**Gating Requirements:**

* Pipeline draait operationeel zonder crashes of segfaults over ≥ 100.000 frames.
* Zero-copy buffers correct gedeeld.
* Video stream via UDP/RTP of RTSP stabiel.
* Bounding boxes en telemetrie correct en chronologisch gelogd.
* **Geen logregel bevat “error”, “segfault”, of ontbrekende tijdstempels.**
* Latentie ≤ 100 ms met <5% jitter.
* TPU throughput ≥ 90 FPS per 100 FPS capture.
* Temperatuur stabiel, geen throttling.
* **Applicatie stopt volledig en deterministisch bij stop-signaal of na voltooiing van frame-count test.**

---

### Stage 3: Validatie & verificatie

**Doel:** 4-uur stress test en schietzaal-validatie van alle kritieke systemen.

**Taken:**

* Continue logging van thermiek, FPS, en jitter naar CSV + PNG grafieken
* Onzekerheidspropagatie verificatie in schietzaal
* `logic` module verificatie van 3D ballistiek, hit-scan en actuatie
* `system_monitor` supervisie en logging testen

**Stabiliteitseis:** E2E latency binnen 5% van nominale waarde over volledige testduur.

**Gating Requirements:**

* Stress test draait zonder crashes of segfaults.
* Thermische en timing logs consistent, volledig en binnen tolerantie.
* Logic module correct uitgevoerd: 3D ballistiek, hit-scan, servo-actuatie.
* System monitor supervisie en logging operationeel.
* E2E latency binnen 5% van nominale waarde over gehele testduur.
* **Applicatie en alle threads stoppen deterministisch bij test-einde of onvoorziene fout, inclusief forced PID kill fallback als watchdog nodig is.**
* **Log verificatie: alle subsystemen bevatten volledige headers, geen ontbrekende frames, monotone tijdstempels en plausibele waarde ranges.**
```

---
