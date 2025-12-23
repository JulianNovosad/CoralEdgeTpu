#!/bin/bash
set -e

echo "Starting build process..."

# Clean any lingering CMakeCache.txt files to ensure a fresh build
find . -name "CMakeCache.txt" -delete 2>/dev/null || true

# 1. Package installs and verification
echo "1. Installing required packages..."
# Fix any dpkg issues first
sudo dpkg --configure -a || true

# Update package lists with error handling
echo "Updating package lists..."
MAX_RETRIES=3
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if sudo apt-get update -y; then
        echo "Package lists updated successfully"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Attempt $RETRY_COUNT of $MAX_RETRIES failed to update package lists, retrying in 5 seconds..."
        sleep 5
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "WARNING: Failed to update package lists after $MAX_RETRIES attempts, continuing with existing lists"
fi

# Install packages with retry logic
install_package() {
    local package=$1
    local description=$2
    local retry_count=0
    
    echo "Installing $description ($package)..."
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if sudo apt-get install -y "$package"; then
            echo "Successfully installed $description"
            return 0
        else
            retry_count=$((retry_count + 1))
            echo "Attempt $retry_count of $MAX_RETRIES failed to install $package, retrying in 5 seconds..."
            sleep 5
        fi
    done
    
    echo "WARNING: Failed to install $package after $MAX_RETRIES attempts"
    return 1
}

# Install core build dependencies
install_package "build-essential" "build essentials"
install_package "cmake" "CMake"
install_package "pkg-config" "pkg-config"

# Install project-specific dependencies (excluding libcamera-dev since we use local build)
install_package "libzmq3-dev" "ZeroMQ development files"
install_package "libjpeg-dev" "JPEG development files"
install_package "libopencv-dev" "OpenCV development files"
install_package "libx264-dev" "x264 development files"

# Try to install EdgeTPU runtime (optional)
echo "Installing EdgeTPU runtime (optional)..."
sudo apt-get install -y libedgetpu-dev || echo "NOTE: EdgeTPU development files not available, continuing without them"

# Check for EdgeTPU devices and libraries
echo "Checking for EdgeTPU devices and libraries..."
ls -l /usr/lib | grep edgetpu 2>/dev/null || echo "NOTE: libedgetpu not found in /usr/lib"
ls -l /dev | grep apex 2>/dev/null || echo "NOTE: Edge TPU device not found in /dev"

# Create symlinks for libedgetpu if it exists
echo "Setting up EdgeTPU library symlinks..."
if [ -f "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0" ]; then
    sudo rm -f lib/libedgetpu.so lib/libedgetpu.so.1 2>/dev/null || true
    mkdir -p lib
    sudo ln -sf /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0 lib/libedgetpu.so.1
    sudo ln -sf libedgetpu.so.1 lib/libedgetpu.so
    echo "Successfully created Edge TPU library symlinks"
elif [ -f "/usr/lib/libedgetpu.so.1" ]; then
    sudo rm -f lib/libedgetpu.so lib/libedgetpu.so.1 2>/dev/null || true
    mkdir -p lib
    sudo ln -sf /usr/lib/libedgetpu.so.1 lib/libedgetpu.so.1
    sudo ln -sf libedgetpu.so.1 lib/libedgetpu.so
    echo "Successfully created Edge TPU library symlinks (alternative location)"
else
    echo "NOTE: Edge TPU library not found, symlink creation skipped"
fi

# 2. FlatBuffers v1.12.0 (clone, build, local install)
echo "2. Building and installing FlatBuffers v1.12.0..."
# Remove existing directories to ensure clean build
rm -rf flatbuffers-src flatbuffers

echo "Cloning FlatBuffers v1.12.0..."
git clone --branch v1.12.0 https://github.com/google/flatbuffers.git flatbuffers-src || {
    echo "ERROR: Failed to clone FlatBuffers, trying alternative method..."
    # Try shallow clone
    rm -rf flatbuffers-src
    git clone --branch v1.12.0 --depth 1 https://github.com/google/flatbuffers.git flatbuffers-src || {
        echo "ERROR: Failed to clone FlatBuffers"
        exit 1
    }
}

cd flatbuffers-src
echo "Configuring FlatBuffers..."
cmake -DFLATBUFFERS_BUILD_TESTS=OFF -B build -S . || {
    echo "ERROR: Failed to configure FlatBuffers"
    exit 1
}
echo "Building FlatBuffers..."
cmake --build build -j$(nproc) || {
    echo "ERROR: Failed to build FlatBuffers"
    exit 1
}
cd ..

mkdir -p flatbuffers/lib
cp -r flatbuffers-src/include flatbuffers/
cp flatbuffers-src/build/libflatbuffers.a flatbuffers/lib/ || {
    echo "ERROR: Failed to copy FlatBuffers library"
    exit 1
}

# 3. CivetWeb (clone and copy)
echo "3. Cloning and copying CivetWeb..."
# Remove existing directories to ensure clean setup
rm -rf civetweb_temp civetweb

echo "Cloning CivetWeb v1.16..."
git clone https://github.com/civetweb/civetweb.git --branch v1.16 --depth 1 civetweb_temp || {
    echo "ERROR: Failed to clone CivetWeb"
    exit 1
}

# Ensure the target civetweb directory is clean before copying
mkdir -p civetweb/src
mkdir -p civetweb/include

if [ -f "civetweb_temp/src/civetweb.c" ]; then
    cp civetweb_temp/src/civetweb.c civetweb/src/
    cp civetweb_temp/src/*.inl civetweb/src/ 2>/dev/null || true
    cp -r civetweb_temp/include civetweb/ 2>/dev/null || true
    echo "Successfully copied CivetWeb files"
else
    echo "ERROR: CivetWeb source files not found"
    exit 1
fi

# 4. TensorFlow v2.5.0 checkout, patch, and build TFLite shared lib
echo "4. Setting up TensorFlow Lite v2.5.0..."
TF_VALID=false

# Check if TensorFlow is already cloned and has the correct version
if [ -d "tensorflow_2.5.0" ]; then
    echo "TensorFlow directory exists, checking version and integrity..."
    if [ -d "tensorflow_2.5.0/.git" ]; then
        cd tensorflow_2.5.0
        # Check if it's a git repository and has the correct tag
        if git rev-parse --git-dir > /dev/null 2>&1; then
            CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "none")
            if [ "$CURRENT_TAG" = "v2.5.0" ]; then
                # Additional check: verify that key files exist and the repository is not corrupted
                if [ -f "tensorflow/lite/BUILD" ] && [ -f "WORKSPACE" ]; then
                    echo "TensorFlow v2.5.0 already cloned and verified."
                    TF_VALID=true
                else
                    echo "TensorFlow directory exists but appears to be corrupted (missing key files)"
                fi
            else
                echo "TensorFlow directory exists but is not v2.5.0 (found tag: $CURRENT_TAG)"
            fi
        else
            echo "TensorFlow directory exists but is not a valid git repository"
        fi
        cd ..
    else
        echo "TensorFlow directory exists but is not a git repository"
    fi
    
    # If TF is not valid, remove it
    if [ "$TF_VALID" = "false" ]; then
        echo "Removing existing TensorFlow directory..."
        rm -rf tensorflow_2.5.0
    fi
else
    echo "TensorFlow directory does not exist"
fi

# Clone TensorFlow v2.5.0 if needed
if [ "$TF_VALID" = "false" ]; then
    echo "Cloning TensorFlow v2.5.0..."
    git clone https://github.com/tensorflow/tensorflow.git tensorflow_2.5.0 || {
        echo "ERROR: Failed to clone TensorFlow"
        exit 1
    }
    cd tensorflow_2.5.0
    git checkout v2.5.0 || {
        echo "ERROR: Failed to checkout TensorFlow v2.5.0"
        exit 1
    }
    cd ..
else
    echo "Skipping TensorFlow download - valid version already exists"
fi

# 4a. Build and install local libcamera (if not already done)
echo "4a. Checking libcamera installation..."
if [ ! -f "/usr/local/lib/aarch64-linux-gnu/libcamera.so" ]; then
    echo "Building and installing local libcamera from /home/pi/libcamera..."
    if [ -d "/home/pi/libcamera" ]; then
        cd /home/pi/libcamera
        # Ensure meson build directory exists
        if [ ! -d "build" ]; then
            meson setup build --prefix=/usr/local
        fi
        # Build and install
        ninja -C build
        sudo meson install -C build
        sudo ldconfig
        echo "Local libcamera built and installed successfully"
        cd /home/pi/CoralEdgeTpu
    else
        echo "WARNING: /home/pi/libcamera directory not found, using system libcamera"
    fi
else
    echo "Local libcamera already installed"
fi

# 5. Build final C++ app with CMake
echo "5. Building final C++ application..."
# Clean build directory
rm -rf build
mkdir -p build
cd build
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-I/usr/include/liveMedia -I/usr/include/groupsock -I/usr/include/BasicUsageEnvironment -I/usr/include/UsageEnvironment -Werror" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/lib/aarch64-linux-gnu -lliveMedia -lgroupsock -lBasicUsageEnvironment -lUsageEnvironment" || {
    echo "ERROR: CMake configuration failed"
    exit 1
}
echo "Compiling with make..."
make -j$(nproc) || {
    echo "ERROR: Compilation failed"
    exit 1
}
cd ..

echo "Build process completed successfully!"