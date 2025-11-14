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
    ca-certificates \
    curl \
    gnupg \
    python3-numpy \
    python3-pil \
    python3-posix-ipc \
    procps \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Add the Coral package repository
RUN echo "deb https://packages.cloud.google.com/apt  coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install Coral TPU libraries
# libedgetpu1-std supports PCIe TPUs.
# python3-pycoral provides the Python API.
# python3-tflite-runtime provides the TensorFlow Lite runtime.
RUN apt-get update && apt-get install -y \
    libedgetpu1-std \
    python3-pycoral \
    python3-tflite-runtime \
    && rm -rf /var/lib/apt/lists/*

# Set up the application directory
WORKDIR /app

# Create socket directory with proper permissions (defensive programming)
RUN mkdir -p /app/socket && chmod 777 /app/socket

# Copy the inference server script and model files
COPY tpu_server.py /app/tpu_server.py
COPY coco_labels.txt /app/coco_labels.txt
COPY ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite /app/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite

# Set the entrypoint for the container, ensuring unbuffered output
ENTRYPOINT ["python3", "-u", "/app/tpu_server.py"]