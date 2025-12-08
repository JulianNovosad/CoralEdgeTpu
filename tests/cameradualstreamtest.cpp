#include <iostream>
#include <memory>
#include <vector>
#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/camera.h>
#include <libcamera/stream.h>

int main() {
    libcamera::CameraManager cm;
    if (cm.start()) {
        std::cerr << "Failed to start CameraManager\n";
        return 1;
    }

    auto cameras = cm.cameras();
    if (cameras.empty()) {
        std::cerr << "No cameras found\n";
        return 1;
    }

    auto camera = cameras[0];
    std::cout << "Using camera: " << camera->id() << "\n";

    if (camera->acquire()) {
        std::cerr << "Failed to acquire camera\n";
        return 1;
    }

    // Generate configuration for dual streams
    std::unique_ptr<libcamera::CameraConfiguration> config;
    config = camera->generateConfiguration({
        libcamera::StreamRole::Viewfinder,   // high-res stream
        libcamera::StreamRole::VideoRecording // TPU stream
    });

    if (!config) {
        std::cerr << "Failed to generate configuration\n";
        camera->release();
        return 1;
    }

    // High-resolution stream
    config->at(0).pixelFormat = libcamera::formats::RGB888;
    config->at(0).size.width = 1536;
    config->at(0).size.height = 864;

    // TPU-resolution stream
    config->at(1).pixelFormat = libcamera::formats::RGB888;
    config->at(1).size.width = 300;
    config->at(1).size.height = 300;

    // Validate configuration
    libcamera::CameraConfiguration::Status status = config->validate();
    if (status != libcamera::CameraConfiguration::Valid) {
        std::cerr << "Camera configuration invalid: " << int(status) << "\n";
        camera->release();
        return 1;
    }

    // Configure camera
    if (camera->configure(config.get())) {
        std::cerr << "Failed to configure camera\n";
        camera->release();
        return 1;
    }

    std::cout << "Dual stream configuration successful!\n";

    camera->release();
    cm.stop();
    return 0;
}
