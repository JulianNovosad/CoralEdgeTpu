#include <iostream>
#include <string>

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        // Skip lines that don't contain our target log types
        if (line.find("DETECTION_INVARIANT") == std::string::npos &&
            line.find("DETECTION_DISTANCE") == std::string::npos &&
            line.find("CAUSALITY_VALIDATION") == std::string::npos &&
            line.find("INTERNAL DISTANCE ESTIMATE") == std::string::npos &&
            line.find("CameraCapture: Total time to process request") == std::string::npos) {
            continue;
        }
        
        std::cout << "Processing line: " << line << std::endl;
        
        if (line.find("DETECTION_INVARIANT: Detection") != std::string::npos) {
            std::cout << "  -> Matched DETECTION_INVARIANT pattern" << std::endl;
        } else if (line.find("DETECTION_DISTANCE:") != std::string::npos) {
            std::cout << "  -> Matched DETECTION_DISTANCE pattern" << std::endl;
        } else if (line.find("CAUSALITY_VALIDATION:") != std::string::npos) {
            std::cout << "  -> Matched CAUSALITY_VALIDATION pattern" << std::endl;
        } else if (line.find("INTERNAL DISTANCE ESTIMATE:") != std::string::npos) {
            std::cout << "  -> Matched INTERNAL_DISTANCE_ESTIMATE pattern" << std::endl;
        } else if (line.find("CameraCapture: Total time to process request") != std::string::npos) {
            std::cout << "  -> Matched CAMERA_PROCESS_TIME pattern" << std::endl;
        } else {
            std::cout << "  -> No pattern matched" << std::endl;
        }
    }
    return 0;
}