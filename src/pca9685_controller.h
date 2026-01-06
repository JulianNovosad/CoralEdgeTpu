// Verified headers: [string, atomic]
// Verification timestamp: 2026-01-06 17:08:04
#ifndef PCA9685_CONTROLLER_H
#define PCA9685_CONTROLLER_H

#include <string>
#include <atomic>

/**
 * @brief Controls a PCA9685 PWM controller over I2C.
 * Used for servo actuation in the fire-control system.
 */
class PCA9685Controller {
public:
    PCA9685Controller(int bus = 1, int address = 0x40);
    ~PCA9685Controller();

    bool initialize(int frequency_hz);
    bool is_initialized() const { return initialized_.load(); }
    
    void set_servo_position(int channel, float position_normalized);
    void set_pwm(int channel, int on, int off);

private:
    int bus_;
    int address_;
    int fd_;
    std::atomic<bool> initialized_;
};

#endif // PCA9685_CONTROLLER_H
