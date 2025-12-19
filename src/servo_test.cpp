#include "pca9685_controller.h"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "Servo Test Program" << std::endl;
    
    // Create PCA9685 controller instance
    // Bus 1, Address 0x40, 333Hz PWM frequency
    PCA9685Controller servoController(1, 0x40);
    
    // Initialize the controller with 333Hz frequency
    if (!servoController.initialize(333)) {
        std::cerr << "Failed to initialize PCA9685 controller" << std::endl;
        return 1;
    }
    
    std::cout << "PCA9685 controller initialized successfully" << std::endl;
    
    // Test servo on channel 0
    std::cout << "Testing servo on channel 0" << std::endl;
    
    // Move servo to 0% position (1ms pulse)
    std::cout << "Moving servo to 0% position" << std::endl;
    servoController.set_servo_position(0, 0.0);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Move servo to 50% position (1.5ms pulse)
    std::cout << "Moving servo to 50% position" << std::endl;
    servoController.set_servo_position(0, 0.5);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Move servo to 100% position (2ms pulse)
    std::cout << "Moving servo to 100% position" << std::endl;
    servoController.set_servo_position(0, 1.0);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Sweep servo from 0% to 100% in 10% increments
    std::cout << "Sweeping servo from 0% to 100%" << std::endl;
    for (int i = 0; i <= 10; i++) {
        float position = i * 0.1;
        std::cout << "Setting servo to " << (position * 100) << "% position" << std::endl;
        servoController.set_servo_position(0, position);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    // Turn off servo
    std::cout << "Turning off servo" << std::endl;
    servoController.turn_off_led(0);
    
    std::cout << "Servo test completed" << std::endl;
    
    return 0;
}