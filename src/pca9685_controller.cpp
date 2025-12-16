#include "pca9685_controller.h"
#include "util_logging.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <cstring>
#include <sstream>
#include <cmath>

// PCA9685 Register addresses
#define PCA9685_MODE1           0x00
#define PCA9685_MODE2           0x01
#define PCA9685_LED0_ON_L       0x06
#define PCA9685_LED0_ON_H       0x07
#define PCA9685_LED0_OFF_L      0x08
#define PCA9685_LED0_OFF_H      0x09
#define PCA9685_ALL_LED_ON_L    0xFA
#define PCA9685_ALL_LED_ON_H    0xFB
#define PCA9685_ALL_LED_OFF_L   0xFC
#define PCA9685_ALL_LED_OFF_H   0xFD
#define PCA9685_PRESCALE        0xFE

// MODE1 bits
#define PCA9685_MODE1_RESTART   0x80
#define PCA9685_MODE1_EXTCLK    0x40
#define PCA9685_MODE1_AI        0x20
#define PCA9685_MODE1_SLEEP     0x10
#define PCA9685_MODE1_SUB1      0x08
#define PCA9685_MODE1_SUB2      0x04
#define PCA9685_MODE1_SUB3      0x02
#define PCA9685_MODE1_ALLCALL   0x01

// Default I2C frequency: 100kHz
// Default PWM frequency: 50Hz (20ms period)

PCA9685Controller::PCA9685Controller(int i2c_bus, uint8_t i2c_address)
    : i2c_bus_(i2c_bus), i2c_address_(i2c_address), i2c_fd_(-1), initialized_(false) {
}

PCA9685Controller::~PCA9685Controller() {
    if (i2c_fd_ >= 0) {
        close(i2c_fd_);
    }
}

bool PCA9685Controller::initialize() {
    return initialize(50); // Default to 50Hz for backward compatibility
}

bool PCA9685Controller::initialize(uint16_t pwm_frequency) {
    if (initialized_) {
        APP_LOG_WARNING("PCA9685Controller already initialized");
        return true;
    }
    
    // Create I2C device path
    std::stringstream ss;
    ss << "/dev/i2c-" << i2c_bus_;
    std::string device_path = ss.str();
    
    // Open I2C bus
    i2c_fd_ = open(device_path.c_str(), O_RDWR);
    if (i2c_fd_ < 0) {
        APP_LOG_ERROR("Failed to open I2C bus " + device_path);
        return false;
    }
    
    // Set I2C slave address
    if (ioctl(i2c_fd_, I2C_SLAVE, i2c_address_) < 0) {
        APP_LOG_ERROR("Failed to set I2C slave address 0x" + 
                     std::to_string(i2c_address_));
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }
    
    // Reset the PCA9685
    if (!write_register(PCA9685_MODE1, 0x00)) {
        APP_LOG_ERROR("Failed to reset PCA9685");
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }
    
    // Calculate prescale value for desired PWM frequency
    // Formula: prescale = round((25000000 / (4096 * frequency)) - 1)
    uint8_t prescale = static_cast<uint8_t>(round(25000000.0 / (4096.0 * pwm_frequency) - 1));
    
    // Put oscillator in sleep mode
    if (!write_register(PCA9685_MODE1, PCA9685_MODE1_SLEEP)) {
        APP_LOG_ERROR("Failed to put PCA9685 to sleep");
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }
    
    // Set prescale value
    if (!write_register(PCA9685_PRESCALE, prescale)) {
        APP_LOG_ERROR("Failed to set prescale value");
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }
    
    // Wake up oscillator
    if (!write_register(PCA9685_MODE1, PCA9685_MODE1_AI)) {
        APP_LOG_ERROR("Failed to wake up PCA9685");
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }
    
    // Wait for oscillator to stabilize
    usleep(5000); // 5ms delay
    
    // Enable auto-increment and restart
    if (!write_register(PCA9685_MODE1, PCA9685_MODE1_RESTART | PCA9685_MODE1_AI)) {
        APP_LOG_ERROR("Failed to enable auto-increment and restart");
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }
    
    initialized_ = true;
    APP_LOG_INFO("PCA9685Controller initialized successfully on bus " + 
                 std::to_string(i2c_bus_) + " at address 0x" + 
                 std::to_string(i2c_address_) + " with " +
                 std::to_string(pwm_frequency) + "Hz PWM frequency");
    return true;
}

bool PCA9685Controller::set_led_brightness(uint8_t channel, uint16_t brightness) {
    if (!initialized_) {
        APP_LOG_ERROR("PCA9685Controller not initialized");
        return false;
    }
    
    if (channel > 15) {
        APP_LOG_ERROR("Invalid LED channel: " + std::to_string(channel));
        return false;
    }
    
    if (brightness > 4095) {
        brightness = 4095;
    }
    
    // Calculate register address for this channel
    uint8_t reg = PCA9685_LED0_ON_L + (channel * 4);
    
    // Set PWM values: turn on at 0, turn off at brightness value
    return set_pwm(reg, 0, brightness);
}

bool PCA9685Controller::set_servo_position(uint8_t channel, float position) {
    if (!initialized_) {
        APP_LOG_ERROR("PCA9685Controller not initialized");
        return false;
    }
    
    if (channel > 15) {
        APP_LOG_ERROR("Invalid servo channel: " + std::to_string(channel));
        return false;
    }
    
    // Clamp position to 0.0-1.0 range
    if (position < 0.0f) position = 0.0f;
    if (position > 1.0f) position = 1.0f;
    
    // For 333Hz frequency (3ms period):
    // 1ms pulse = 0°, 2ms pulse = 180°
    // 1ms = 1/3ms * 4096 = 1365.3 ticks
    // 2ms = 2/3ms * 4096 = 2730.7 ticks
    
    uint16_t min_pulse = 1365;  // 1ms at 333Hz
    uint16_t max_pulse = 2731;  // 2ms at 333Hz
    uint16_t pulse_range = max_pulse - min_pulse;
    uint16_t off_value = min_pulse + static_cast<uint16_t>(position * pulse_range);
    
    // Calculate register address for this channel
    uint8_t reg = PCA9685_LED0_ON_L + (channel * 4);
    
    // Set PWM values: turn on at 0, turn off at pulse value
    return set_pwm(reg, 0, off_value);
}

bool PCA9685Controller::turn_off_led(uint8_t channel) {
    return set_led_brightness(channel, 0);
}

bool PCA9685Controller::turn_on_led(uint8_t channel) {
    return set_led_brightness(channel, 4095);
}

bool PCA9685Controller::write_register(uint8_t reg, uint8_t value) {
    uint8_t buffer[2] = {reg, value};
    
    if (write(i2c_fd_, buffer, 2) != 2) {
        APP_LOG_ERROR("Failed to write to PCA9685 register 0x" + 
                     std::to_string(reg));
        return false;
    }
    
    return true;
}

bool PCA9685Controller::set_pwm(uint8_t reg, uint16_t on_value, uint16_t off_value) {
    uint8_t buffer[5] = {
        reg,
        static_cast<uint8_t>(on_value & 0xFF),
        static_cast<uint8_t>((on_value >> 8) & 0xFF),
        static_cast<uint8_t>(off_value & 0xFF),
        static_cast<uint8_t>((off_value >> 8) & 0xFF)
    };
    
    if (write(i2c_fd_, buffer, 5) != 5) {
        APP_LOG_ERROR("Failed to set PWM values");
        return false;
    }
    
    return true;
}