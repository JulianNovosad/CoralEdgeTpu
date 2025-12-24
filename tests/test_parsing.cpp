#include <iostream>
#include <string>

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::cout << "Line: " << line << std::endl;
        
        if (line.find("DETECTION_INVARIANT") != std::string::npos) {
            std::cout << "  -> Contains DETECTION_INVARIANT" << std::endl;
        }
        
        if (line.find("DETECTION_INVARIANT: Detection") != std::string::npos) {
            std::cout << "  -> Matches DETECTION_INVARIANT: Detection pattern" << std::endl;
        }
        
        if (line.find("DETECTION_DISTANCE:") != std::string::npos) {
            std::cout << "  -> Matches DETECTION_DISTANCE: pattern" << std::endl;
        }
        
        if (line.find("CAUSALITY_VALIDATION:") != std::string::npos) {
            std::cout << "  -> Matches CAUSALITY_VALIDATION: pattern" << std::endl;
        }
        
        if (line.find("CameraCapture: Total time to process request") != std::string::npos) {
            std::cout << "  -> Matches CameraCapture process time pattern" << std::endl;
        }
    }
    return 0;
}