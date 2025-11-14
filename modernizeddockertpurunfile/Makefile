# Makefile for Coral TPU Inference Application

# Compiler and flags
CXX := g++
# Use pkg-config for libcamera flags
LIBCAMERA_CFLAGS := $(shell pkg-config --cflags libcamera)
LIBCAMERA_LIBS := $(shell pkg-config --libs libcamera)

CXXFLAGS := -O3 -std=c++17 -pthread $(LIBCAMERA_CFLAGS)
LDFLAGS := $(LIBCAMERA_LIBS)

# Source and output files
PRODUCER_SRC := camera_producer_host.cpp
PRODUCER_BIN := camera_producer_host

# Docker settings
DOCKER_IMAGE_NAME := coral-tpu
DOCKER_IMAGE_TAG := latest
DOCKER_CONTAINER_NAME := coral-tpu-container

# Environment variables for the container
PHONE_IP ?= 192.168.37.27
PHONE_PORT ?= 9090

.PHONY: all build run stop shell clean producer kill-tpu-procs

all: build producer

# Target to build the Docker image
build:
	@echo "Building Docker image $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)..."
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) .

# Target to kill any processes holding the TPU (PREVENTS DEVICE LOCKS)
kill-tpu-procs:
	@echo "Killing any processes holding /dev/apex_0..."
	-sudo fuser -k /dev/apex_0 2>/dev/null || true
	-sudo killall -9 python3 2>/dev/null || true  # Kill zombie Python processes
	-sleep 5  # Wait for kernel to release the device

# Target to run the Docker container
run: kill-tpu-procs
	@echo "Stopping and removing any pre-existing container..."
	-docker stop $(DOCKER_CONTAINER_NAME)
	-docker rm $(DOCKER_CONTAINER_NAME)
	@echo "Running Docker container $(DOCKER_CONTAINER_NAME)..."
	@echo "Using PHONE_IP=$(PHONE_IP) and PHONE_PORT=$(PHONE_PORT)"
	docker run --rm --name $(DOCKER_CONTAINER_NAME) \
		--device /dev/apex_0:/dev/apex_0 \
		-v /tmp/coral_ipc:/app/socket \
		-e PHONE_IP=$(PHONE_IP) \
		-e PHONE_PORT=$(PHONE_PORT) \
		$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

# Target to stop the Docker container
stop:
	@echo "Stopping and removing container $(DOCKER_CONTAINER_NAME)..."
	docker stop $(DOCKER_CONTAINER_NAME) || true
	docker rm $(DOCKER_CONTAINER_NAME) || true

# Target to get an interactive shell in the container for debugging
shell:
	@echo "Starting interactive shell in $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)..."
	docker run -it --rm --name $(DOCKER_CONTAINER_NAME)-shell \
		--device /dev/apex_0:/dev/apex_0 \
		-v /tmp/coral_ipc:/app/socket \
		-e PHONE_IP=$(PHONE_IP) \
		-e PHONE_PORT=$(PHONE_PORT) \
		--entrypoint /bin/bash \
		$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

# Target to compile the C++ camera producer
producer: $(PRODUCER_BIN)

$(PRODUCER_BIN): $(PRODUCER_SRC)
	@echo "Compiling C++ camera producer..."
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Target to clean up build artifacts
# DEPENDS ON 'stop' to ensure containers are removed before image deletion
clean: stop
	@echo "Cleaning up..."
	rm -f $(PRODUCER_BIN)
	-docker image rm $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)