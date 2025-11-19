#ifndef PCA9685_H
#define PCA9685_H

#include <string>
#include <cstdint>

// PCA9685 Register Addresses
#define PCA9685_MODE1 0x00
#define PCA9685_MODE2 0x01
#define PCA9685_SUBADR1 0x02
#define PCA9685_SUBADR2 0x03
#define PCA9685_SUBADR3 0x04
#define PCA9685_ALLCALLADR 0x05
#define PCA9685_LED0_ON_L 0x06
#define PCA9685_LED0_ON_H 0x07
#define PCA9685_LED0_OFF_L 0x08
#define PCA9685_LED0_OFF_H 0x09
#define PCA9685_ALLLED_ON_L 0xFA
#define PCA9685_ALLLED_ON_H 0xFB
#define PCA9685_ALLLED_OFF_L 0xFC
#define PCA9685_ALLLED_OFF_H 0xFD
#define PCA9685_PRESCALE 0xFE
#define PCA9685_TESTMODE 0xFF

// MODE1 bits
#define MODE1_ALLCALL 0x01
#define MODE1_SUB3 0x02
#define MODE1_SUB2 0x04
#define MODE1_SUB1 0x08
#define MODE1_SLEEP 0x10
#define MODE1_AI 0x20
#define MODE1_EXTCLK 0x40
#define MODE1_RESTART 0x80

// MODE2 bits
#define MODE2_OUTNE_0 0x01 // Output logic state not inverted
#define MODE2_OUTNE_1 0x02 // Output logic state inverted
#define MODE2_OUTDRV 0x04  // Totem pole structure
#define MODE2_OCH 0x08     // Outputs change on stop command
#define MODE2_INVRT 0x10   // Outputs inverted

#define PCA9685_I2C_ADDRESS 0x40 // Default PCA9685 address

// Standard Servo ranges
#define SERVO_MIN_PULSE_US 500  // Minimum pulse width in microseconds (often 500us)
#define SERVO_MAX_PULSE_US 2500 // Maximum pulse width in microseconds (often 2500us)
#define SERVO_MAX_ANGLE 180     // Max angle in degrees for a standard servo

class PCA9685 {
public:
    PCA9685(uint8_t busNum, uint8_t address = PCA9685_I2C_ADDRESS);
    ~PCA9685();

    bool openDevice();
    void closeDevice();

    bool reset();
    bool setPWMFreq(float freq);
    bool setPWM(uint8_t channel, uint16_t on, uint16_t off);
    bool setServoPulse(uint8_t channel, uint16_t pulse_us);
    bool setServoAngle(uint8_t channel, float angle);

private:
    int i2c_fd;
    uint8_t i2c_bus_num;
    uint8_t i2c_address;
    float current_pwm_freq;
    uint16_t us_per_tick; // Microseconds per PWM tick
    
    bool writeByte(uint8_t reg, uint8_t value);
    uint8_t readByte(uint8_t reg);
};

#endif // PCA9685_H
