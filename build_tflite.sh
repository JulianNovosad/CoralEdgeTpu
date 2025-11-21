#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

TENSORFLOW_REPO_DIR="$HOME/CoralEdgeTpu/tensorflow_2.5.0"
INSTALL_DIR="$HOME/CoralEdgeTpu"
FLATBUFFERS_INCLUDE_DIR="$HOME/CoralEdgeTpu/flatbuffers/include"
FLATBUFFERS_LIB_DIR="$HOME/CoralEdgeTpu/flatbuffers/lib"

# Navigate to the TensorFlow Lite source directory for CMake configuration
cd "$TENSORFLOW_REPO_DIR/tensorflow/lite"

# Ensure build directories are clean or use a dedicated build folder
rm -rf build_tflite
mkdir build_tflite
cd build_tflite

# Configure CMake for TFLite build
# Use C++14 standard for TFLite build.
# Specify build options to ensure a shared library is produced
# and to point to our local FlatBuffers installation.
cmake ..     -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR     -DCMAKE_PREFIX_PATH="$HOME/CoralEdgeTpu/flatbuffers"     -DTFLITE_BUILD_SHARED_LIBS=ON     -DCMAKE_POSITION_INDEPENDENT_CODE=ON     -DCMAKE_CXX_STANDARD=14     -DCMAKE_CXX_STANDARD_REQUIRED=ON     -DCMAKE_BUILD_TYPE=Release     -G "Unix Makefiles" # Explicitly use Makefiles generator

# Build TFLite
cmake --build . -j4

# Install TFLite (headers and library) to the local install directory
cmake --install . --prefix=$INSTALL_DIR

echo "TensorFlow Lite build process initiated."

# Verify installation
echo "Verifying installation..."
ls -l $INSTALL_DIR/lib/libtensorflow-lite.so
ls -l $INSTALL_DIR/include/tensorflow/lite/
