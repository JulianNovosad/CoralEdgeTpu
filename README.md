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
* Handmatig gebouwde TensorFlow Lite 2.11.1 voor ARM64
* Gepatchte en gestabiliseerde Flatbuffers-versie
* Volledige dependency-graph **meegeleverd in deze repo**
* Werkende example app (`src/main.cpp`)
* Ondersteuning voor `ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite`

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
* `libedgetpu1-std` of `libedgetpu1-max`
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

## ✉️ Contact

Project door: **Julian Novosad**
