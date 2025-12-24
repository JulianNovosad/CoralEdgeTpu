#include <iostream>
#include <fstream>
#include "include/json.hpp"

using json = nlohmann::json;

int main() {
    std::cout << "Verifying class scale factors...\n";
    
    // Read the scale factors JSON file
    std::ifstream file("class_scale_factors.json");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open class_scale_factors.json\n";
        return 1;
    }
    
    json scale_factors;
    try {
        file >> scale_factors;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse JSON file: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "\nLoaded scale factors:\n";
    std::cout << "Class\tScale Factor\n";
    std::cout << "-----\t-----------\n";
    
    for (auto& [key, value] : scale_factors.items()) {
        std::cout << key << "\t" << value.get<float>() << "\n";
    }
    
    std::cout << "\nVerification completed successfully.\n";
    
    return 0;
}