# Dockerfile for Coral PCIe TPU Inference Server
# Base image
FROM debian:10

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update APT sources for the Debian 10 archive and install prerequisites
RUN echo "deb http://archive.debian.org/debian/  buster main" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security/  buster/updates main" >> /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian/  buster-updates main" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libusb-1.0-0-dev \
    libatlas-base-dev \
    libhdf5-dev \
    ca-certificates \
    curl \
    gnupg \
    procps \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Add the Coral package repository
RUN echo "deb https://packages.cloud.google.com/apt  coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install Coral TPU libraries (libedgetpu1-std only, as TFLite is built from source)
RUN apt-get update && apt-get install -y \
    libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow Lite from source
WORKDIR /
RUN git clone https://github.com/tensorflow/tensorflow.git
WORKDIR /tensorflow
RUN ./tensorflow/lite/tools/make/download_dependencies.sh
RUN make -f tensorflow/lite/tools/make/Makefile -j$(nproc)

# Set up environment variables
ENV LD_LIBRARY_PATH=/tensorflow/tensorflow/lite/tools/make/gen/lib:$LD_LIBRARY_PATH

WORKDIR /workspace