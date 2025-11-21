#!/bin/bash
set -e

echo "Starting build process..."

# Clean any lingering CMakeCache.txt files to ensure a fresh build
find . -name "CMakeCache.txt" -delete

# 1. Package installs and verification
echo "1. Installing required packages..."
sudo apt-get update -y
sudo apt-get install -y build-essential cmake git libjpeg-dev libjpeg62-turbo-dev libusb-1.0-0-dev libcamera-dev pkg-config

# 2. EdgeTPU runtime install and check
echo "2. Installing EdgeTPU runtime and checking device..."
sudo apt-get install -y libedgetpu1-std || true
ls -l /usr/lib | grep edgetpu || true
ls -l /dev | grep apex || true

# Create symlinks for libedgetpu if not exists
sudo rm -f lib/libedgetpu.so lib/libedgetpu.so.1 # Remove existing symlinks
echo "Creating symlinks for libedgetpu.so and libedgetpu.so.1..."
sudo ln -s /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0 lib/libedgetpu.so.1
sudo ln -s libedgetpu.so.1 lib/libedgetpu.so


# 3. FlatBuffers v1.12.0 (clone, build, local install)
echo "3. Building and installing FlatBuffers v1.12.0..."
if [ ! -d "flatbuffers-src" ]; then
    git clone --branch v1.12.0 https://github.com/google/flatbuffers.git flatbuffers-src
fi
cd flatbuffers-src
cmake -DFLATBUFFERS_BUILD_TESTS=OFF -B build -S .
cmake --build build -j$(nproc)
mkdir -p ../flatbuffers
cp -r ../flatbuffers-src/include ../flatbuffers/
cp -r build/lib* ../flatbuffers/lib || true
cd ..


# 4. TensorFlow v2.5.0 checkout, patch, and build TFLite shared lib
echo "4. Building TensorFlow Lite v2.5.0 shared library..."
if [ ! -f "lib/libtensorflowlite.so" ]; then
    if [ ! -d "tensorflow_2.5.0" ]; then
        git clone https://github.com/tensorflow/tensorflow.git tensorflow_2.5.0
        cd tensorflow_2.5.0
        git checkout v2.5.0
        cd ..
    fi
    echo "Building libtensorflow-lite.so using build_rpi_lib.sh..."
    (cd tensorflow_2.5.0/tensorflow/lite/tools/make && ./build_rpi_lib.sh)
    # The build_rpi_lib.sh produces a static library (.a).
    # To get a shared library (.so), we'd typically need to modify the TFLite Makefile
    # or link the static library into a shared library.
    # For now, we'll try to convert the static library to a shared one if it exists.
    if [ -f "tensorflow_2.5.0/tensorflow/lite/tools/make/gen/lib/libtensorflow-lite.a" ]; then
        echo "Converting static libtensorflow-lite.a to shared libtensorflow-lite.so..."
        g++ -shared -o lib/libtensorflowlite.so -Wl,--whole-archive tensorflow_2.5.0/tensorflow/lite/tools/make/gen/lib/libtensorflow-lite.a -Wl,--no-whole-archive -lm -latomic -pthread -ldl
    else
        echo "WARNING: libtensorflow-lite.a not found. libtensorflowlite.so cannot be built."
    fi
fi


# 5. Build final C++ app with CMake
echo "5. Building final C++ application..."
rm -rf build # Clean previous build artifacts
mkdir -p build
cd build
cmake ../
make -j$(nproc)
cd ..
cd ..

echo "Build process completed."
