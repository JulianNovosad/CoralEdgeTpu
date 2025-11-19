#include "../include/pca9685.h"
#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <unistd.h>
#include <cstring>
#include <cmath>

#define SUBADR1 0x02
#define SUBADR2 0x03
#define SUBADR3 0x04
#define ALLCALLADR 0x05
#define US_PER_TICK_FACTOR 0.244140625 // 1,000,000 us / (4096 ticks * 16 steps)

PCA9685::PCA9685(uint8_t busNum, uint8_t address)
    : i2c_fd(-1), i2c_bus_num(busNum), i2c_address(address), current_pwm_freq(0.0f), us_per_tick(0) {
    // Constructor initializes member variables. Actual device opening in openDevice().
}

PCA9685::~PCA9685() {
    closeDevice();
}

bool PCA9685::openDevice() {
    std::string filename = "/dev/i2c-" + std::to_string(i2c_bus_num);
    i2c_fd = open(filename.c_str(), O_RDWR);

    if (i2c_fd < 0) {
        std::cerr << "PCA9685: Failed to open I2C bus " << (int)i2c_bus_num << ": " << strerror(errno) << std::endl;
        return false;
    }

    if (ioctl(i2c_fd, I2C_SLAVE, i2c_address) < 0) {
        std::cerr << "PCA9685: Failed to acquire I2C access to address " << std::hex << (int)i2c_address << ": " << strerror(errno) << std::endl;
        close(i2c_fd);
        i2c_fd = -1;
        return false;
    }
    
    // Reset the PCA9685
    if (!reset()) {
        std::cerr << "PCA9685: Failed to reset." << std::endl;
        return false;
    }
    
    // Set a default frequency for servos
    if (!setPWMFreq(50)) {
        std::cerr << "PCA9685: Failed to set default PWM frequency." << std::endl;
        return false;
    }

    std::cout << "PCA9685: Successfully opened I2C device on bus " << (int)i2c_bus_num 
              << " at address 0x" << std::hex << (int)i2c_address << std::dec << std::endl;
    return true;
}

void PCA9685::closeDevice() {
    if (i2c_fd >= 0) {
        close(i2c_fd);
        i2c_fd = -1;
        std::cout << "PCA9685: I2C device closed." << std::endl;
    }
}

bool PCA9685::writeByte(uint8_t reg, uint8_t value) {
    if (i2c_fd < 0) {
        std::cerr << "PCA9685: Device not open. Cannot write byte." << std::endl;
        return false;
    }
    uint8_t buffer[2] = {reg, value};
    if (write(i2c_fd, buffer, 2) != 2) {
        std::cerr << "PCA9685: Failed to write byte to register 0x" << std::hex << (int)reg 
                  << ": " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

uint8_t PCA9685::readByte(uint8_t reg) {
    if (i2c_fd < 0) {
        std::cerr << "PCA9685: Device not open. Cannot read byte." << std::endl;
        return 0;
    }
    if (write(i2c_fd, &reg, 1) != 1) {
        std::cerr << "PCA9685: Failed to write register address for read 0x" << std::hex << (int)reg 
                  << ": " << strerror(errno) << std::endl;
        return 0;
    }
    uint8_t value;
    if (read(i2c_fd, &value, 1) != 1) {
        std::cerr << "PCA9685: Failed to read byte from register 0x" << std::hex << (int)reg 
                  << ": " << strerror(errno) << std::endl;
        return 0;
    }
    return value;
}

bool PCA9685::reset() {
    writeByte(PCA9685_MODE1, MODE1_RESTART);
    usleep(10000); // Wait for restart
    return true;
}

bool PCA9685::setPWMFreq(float freq) {
    current_pwm_freq = freq;
    // Calculate prescale value
    // prescale_value = round(osc_clock / (4096 * update_rate)) - 1
    // osc_clock = 25MHz = 25000000
    // update_rate is 'freq'
    float prescaleval = 25000000.0f;
    prescaleval /= 4096.0f;
    prescaleval /= freq;
    prescaleval -= 1.0f;
    uint8_t prescale = static_cast<uint8_t>(floor(prescaleval + 0.5f));

    uint8_t oldmode = readByte(PCA9685_MODE1);
    uint8_t newmode = (oldmode & 0x7F) | MODE1_SLEEP; // go to sleep
    if (!writeByte(PCA9685_MODE1, newmode)) return false; // Write sleep mode
    if (!writeByte(PCA9685_PRESCALE, prescale)) return false; // Write prescale
    if (!writeByte(PCA9685_MODE1, oldmode)) return false; // Wake up
    usleep(5000); // Wait for oscillator to stabilize
    if (!writeByte(PCA9685_MODE1, oldmode | MODE1_RESTART)) return false; // Enable auto-increment
    
    us_per_tick = round(1000000.0f / (freq * 4096.0f)); // Microseconds per tick

    std::cout << "PCA9685: PWM frequency set to " << freq << " Hz. Prescale: " << (int)prescale << std::endl;
    return true;
}

bool PCA9685::setPWM(uint8_t channel, uint16_t on, uint16_t off) {
    if (channel > 15) return false;
    std::cout << "PCA9685: Setting Channel " << (int)channel 
              << " | ON tick: " << on << " | OFF tick: " << off << std::endl;
    if (!writeByte(PCA9685_LED0_ON_L + 4 * channel, on & 0xFF)) return false;
    if (!writeByte(PCA9685_LED0_ON_H + 4 * channel, on >> 8)) return false;
    if (!writeByte(PCA9685_LED0_OFF_L + 4 * channel, off & 0xFF)) return false;
    if (!writeByte(PCA9685_LED0_OFF_H + 4 * channel, off >> 8)) return false;
    return true;
}

bool PCA9685::setServoPulse(uint8_t channel, uint16_t pulse_us) {
    if (us_per_tick == 0) {
        std::cerr << "PCA9685: PWM frequency not set, cannot calculate servo pulse." << std::endl;
        return false;
    }
    // Calculate the 'off' tick for the given pulse_us
    // pulse_us / (1,000,000 us / (freq * 4096 ticks)) = pulse_us * freq * 4096 / 1,000,000
    // This is simplified to pulse_us / us_per_tick
    uint16_t off_tick = static_cast<uint16_t>(pulse_us / us_per_tick);
    
    // Ensure the off_tick is within valid range (0-4095)
    if (off_tick > 4095) off_tick = 4095;

    return setPWM(channel, 0, off_tick);
}

bool PCA9685::setServoAngle(uint8_t channel, float angle) {
    if (angle < 0.0f) angle = 0.0f;
    if (angle > SERVO_MAX_ANGLE) angle = SERVO_MAX_ANGLE;

    // Map angle (0-180) to pulse width (SERVO_MIN_PULSE_US - SERVO_MAX_PULSE_US)
    float pulse_width_range = SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US;
    uint16_t pulse_us = static_cast<uint16_t>(SERVO_MIN_PULSE_US + (angle / SERVO_MAX_ANGLE) * pulse_width_range);

    return setServoPulse(channel, pulse_us);
}
