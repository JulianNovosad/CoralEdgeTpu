#!/bin/bash
set -e

echo "Starting build process..."

# Clean any lingering CMakeCache.txt files to ensure a fresh build
find . -name "CMakeCache.txt" -delete

# 1. Package installs and verification
echo "1. Installing required packages..."
sudo apt-get update -y
sudo apt-get install -y libcamera-dev
sudo apt-get install -y libzmq3-dev

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

# 3.5. CivetWeb (clone and copy)
echo "3.5. Cloning and copying CivetWeb..."
if [ ! -d "civetweb_temp" ]; then
    echo "Cloning CivetWeb v1.16 for the first time..."
    git clone https://github.com/civetweb/civetweb.git --branch v1.16 --depth 1 civetweb_temp
else
    echo "CivetWeb already cloned."
fi
# Ensure the target civetweb directory is clean before copying
rm -rf civetweb/*
mkdir -p civetweb/src
mkdir -p civetweb/include # Ensure include directory exists
cp civetweb_temp/src/civetweb.c civetweb/src/
cp civetweb_temp/src/*.inl civetweb/src/ # Copy all .inl files
cp -r civetweb_temp/include civetweb/


# 4. TensorFlow v2.5.0 checkout, patch, and build TFLite shared lib
echo "4. Building TensorFlow Lite v2.5.0 shared library..."
if [ ! -d "tensorflow_2.5.0" ]; then
    echo "Cloning TensorFlow v2.5.0 for the first time..."
    git clone https://github.com/tensorflow/tensorflow.git tensorflow_2.5.0
    cd tensorflow_2.5.0
    git checkout v2.5.0
    cd ..
else
    echo "TensorFlow v2.5.0 already cloned. Ensuring correct version."
    cd tensorflow_2.5.0
    git fetch origin
    git checkout v2.5.0
    cd ..
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
