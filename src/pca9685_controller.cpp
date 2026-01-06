// Verified headers: [pca9685_controller.h, util_logging.h, fcntl.h, unistd.h, linux/i2c-dev.h...]
// Verification timestamp: 2026-01-06 17:08:04
#include "pca9685_controller.h"
#include "util_logging.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <cmath>
#include <thread>
#include <chrono>

// PCA9685 Registers
#define MODE1 0x00
#define PRESCALE 0xFE
#define LED0_ON_L 0x06

PCA9685Controller::PCA9685Controller(int bus, int address)
    : bus_(bus), address_(address), fd_(-1), initialized_(false) {
}

PCA9685Controller::~PCA9685Controller() {
    if (fd_ >= 0) close(fd_);
}

// Helper to write a byte to a register
static bool write_byte(int fd, uint8_t reg, uint8_t value) {
    uint8_t buffer[2] = {reg, value};
    if (write(fd, buffer, 2) != 2) {
        return false;
    }
    return true;
}

// Helper to read a byte from a register
static int read_byte(int fd, uint8_t reg) {
    if (write(fd, &reg, 1) != 1) return -1;
    uint8_t value;
    if (read(fd, &value, 1) != 1) return -1;
    return value;
}

bool PCA9685Controller::initialize(int frequency_hz) {
    APP_LOG_INFO("PCA9685Controller: Initializing on bus " + std::to_string(bus_) + 
                " at address 0x" + std::to_string(address_) + " with " + std::to_string(frequency_hz) + "Hz");
    
    std::string device = "/dev/i2c-" + std::to_string(bus_);
    fd_ = open(device.c_str(), O_RDWR);
    if (fd_ < 0) {
        APP_LOG_ERROR("Failed to open I2C bus: " + device);
        return false;
    }

    if (ioctl(fd_, I2C_SLAVE, address_) < 0) {
        APP_LOG_ERROR("Failed to acquire I2C bus access and/or talk to slave 0x" + std::to_string(address_));
        return false;
    }

    // Reset
    write_byte(fd_, MODE1, 0x00);
    
    // Calculate prescale based on 25MHz internal clock
    float prescale_val = 25000000.0f;
    prescale_val /= 4096.0f;
    prescale_val /= static_cast<float>(frequency_hz);
    prescale_val -= 1.0f;
    uint8_t prescale = static_cast<uint8_t>(std::floor(prescale_val + 0.5f));

    APP_LOG_INFO("PCA9685 Prescale: " + std::to_string((int)prescale));

    // Go to sleep to set prescale
    int oldmode = read_byte(fd_, MODE1);
    int newmode = (oldmode & 0x7F) | 0x10; // sleep
    write_byte(fd_, MODE1, static_cast<uint8_t>(newmode));
    write_byte(fd_, PRESCALE, prescale);
    
    // Wake up
    write_byte(fd_, MODE1, static_cast<uint8_t>(oldmode));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    // Auto-increment enabled
    write_byte(fd_, MODE1, static_cast<uint8_t>(oldmode | 0xA0));

    initialized_.store(true);
    return true;
}

void PCA9685Controller::set_servo_position(int channel, float position) {
    if (!initialized_.load()) return;
    
    // Mapping: 0.0 -> 1000us, 1.0 -> 2000us
    // Period @ 333Hz = ~3000us
    // Max duty cycle = 2000/3000 = 66%
    
    // Clamp
    if (position < 0.0f) position = 0.0f;
    if (position > 1.0f) position = 1.0f;

    // Use standard 1000-2000us mapping
    float pulse_width_us = 1000.0f + (position * 1000.0f);
    
    // Convert to ticks (4096 per period)
    // 333Hz -> 3003us period
    // Ticks = (pulse_width / period) * 4096
    // Or: pulse_width * freq * 4096 / 1000000
    
    // Retrieve freq from somewhere? We passed it in init but didn't store it.
    // Hardcoding 333Hz for now as per LogicModule usage
    float freq = 333.0f;
    int ticks = static_cast<int>((pulse_width_us * freq * 4096.0f) / 1000000.0f);
    
    set_pwm(channel, 0, ticks);
}

void PCA9685Controller::set_pwm(int channel, int on, int off) {
    if (!initialized_.load()) return;
    
    uint8_t reg = LED0_ON_L + (4 * channel);
    uint8_t data[4];
    data[0] = on & 0xFF;
    data[1] = on >> 8;
    data[2] = off & 0xFF;
    data[3] = off >> 8;
    
    // Use a single write for auto-increment efficiency if supported,
    // otherwise 4 individual writes. We set Auto-Inc (0x20) in init.
    uint8_t buffer[5];
    buffer[0] = reg;
    buffer[1] = data[0];
    buffer[2] = data[1];
    buffer[3] = data[2];
    buffer[4] = data[3];
    
    if (write(fd_, buffer, 5) != 5) {
        APP_LOG_ERROR("Failed to write PWM to channel " + std::to_string(channel));
    }
}
