#ifndef PCA9685_CONTROLLER_H
#define PCA9685_CONTROLLER_H

#include <cstdint>
#include <string>
#include <memory>

/**
 * @brief Controller class for PCA9685 PWM LED driver
 * 
 * This class provides an interface to control LEDs connected to a PCA9685
 * PWM controller over I2C. It supports setting LED brightness levels.
 */
class PCA9685Controller {
public:
    /**
     * @brief Construct a new PCA9685Controller object
     * 
     * @param i2c_bus The I2C bus number (e.g., 1 for /dev/i2c-1)
     * @param i2c_address The I2C address of the PCA9685 (default 0x40)
     */
    PCA9685Controller(int i2c_bus, uint8_t i2c_address = 0x40);
    
    /**
     * @brief Destroy the PCA9685Controller object
     */
    ~PCA9685Controller();
    
    /**
     * @brief Initialize the PCA9685 controller with default 50Hz frequency
     * 
     * @return true if initialization was successful
     * @return false if initialization failed
     */
    bool initialize();
    
    /**
     * @brief Initialize the PCA9685 controller with custom PWM frequency
     * 
     * @param pwm_frequency Desired PWM frequency in Hz (typically 50-1000Hz)
     * @return true if initialization was successful
     * @return false if initialization failed
     */
    bool initialize(uint16_t pwm_frequency);
    
    /**
     * @brief Set the brightness of an LED channel
     * 
     * @param channel LED channel number (0-15)
     * @param brightness Brightness level (0-4095, where 0 is off and 4095 is full brightness)
     * @return true if successful
     * @return false if failed
     */
    bool set_led_brightness(uint8_t channel, uint16_t brightness);
    
    /**
     * @brief Set the position of a servo motor
     * 
     * @param channel Servo channel number (0-15)
     * @param position Position value (0.0-1.0, where 0.0 is minimum angle and 1.0 is maximum angle)
     * @return true if successful
     * @return false if failed
     */
    bool set_servo_position(uint8_t channel, float position);
    
    /**
     * @brief Turn off an LED channel
     * 
     * @param channel LED channel number (0-15)
     * @return true if successful
     * @return false if failed
     */
    bool turn_off_led(uint8_t channel);
    
    /**
     * @brief Turn on an LED channel at full brightness
     * 
     * @param channel LED channel number (0-15)
     * @return true if successful
     * @return false if failed
     */
    bool turn_on_led(uint8_t channel);
    
    /**
     * @brief Check if the controller is initialized
     * 
     * @return true if initialized
     * @return false if not initialized
     */
    bool is_initialized() const { return initialized_; }

private:
    /**
     * @brief Write a byte to a register
     * 
     * @param reg Register address
     * @param value Value to write
     * @return true if successful
     * @return false if failed
     */
    bool write_register(uint8_t reg, uint8_t value);
    
    /**
     * @brief Write two bytes to set PWM values
     * 
     * @param reg Register address
     * @param on_value ON value (0-4095)
     * @param off_value OFF value (0-4095)
     * @return true if successful
     * @return false if failed
     */
    bool set_pwm(uint8_t reg, uint16_t on_value, uint16_t off_value);
    
    int i2c_bus_;
    uint8_t i2c_address_;
    int i2c_fd_;
    bool initialized_;
};

#endif // PCA9685_CONTROLLER_H