#include <iostream>
#include <map>
#include <fstream>
#include "include/json.hpp"

using json = nlohmann::json;

// Simple function to apply scale factor correction (similar to what's in LogicModule)
float apply_scale_factor_correction(int class_id, float raw_distance, const std::map<int, float>& scale_factors) {
    auto it = scale_factors.find(class_id);
    if (it != scale_factors.end()) {
        return raw_distance * it->second;
    }
    // Fallback for unmapped classes
    return raw_distance;
}

int main() {
    std::cout << "Testing distance correction with scale factors...\n";
    
    // Load scale factors from JSON file
    std::ifstream file("class_scale_factors.json");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open class_scale_factors.json\n";
        return 1;
    }
    
    json scale_factors_json;
    try {
        file >> scale_factors_json;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse JSON file: " << e.what() << "\n";
        return 1;
    }
    
    // Convert JSON to map
    std::map<int, float> scale_factors;
    for (auto& [key, value] : scale_factors_json.items()) {
        try {
            int class_id = std::stoi(key);
            scale_factors[class_id] = value.get<float>();
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse entry for key: " << key << "\n";
        }
    }
    
    // Test cases with expected results
    std::vector<std::tuple<int, float, float>> test_cases = {
        {5, 2.5f, 2.15f},   // Class 5: 2.5 * 0.86 = 2.15
        {6, 1.6f, 1.2f},    // Class 6: 1.6 * 0.75 = 1.2
        {8, 1.0f, 0.865f},  // Class 8: 1.0 * 0.865 = 0.865
        {10, 0.5f, 0.414f}, // Class 10: 0.5 * 0.828 ≈ 0.414
        {7, 1.1f, 0.93f},   // Class 7: 1.1 * 0.845455 ≈ 0.93
        {99, 2.0f, 2.0f}    // Unknown class: should remain unchanged
    };
    
    std::cout << "\nTesting distance corrections:\n";
    std::cout << "Class\tRaw Distance\tExpected\tActual\t\tDifference\n";
    std::cout << "-----\t------------\t--------\t------\t\t----------\n";
    
    bool all_tests_passed = true;
    
    for (const auto& test_case : test_cases) {
        int class_id = std::get<0>(test_case);
        float raw_distance = std::get<1>(test_case);
        float expected_distance = std::get<2>(test_case);
        
        float actual_distance = apply_scale_factor_correction(class_id, raw_distance, scale_factors);
        float difference = std::abs(actual_distance - expected_distance);
        
        // Check if the result is within acceptable tolerance (0.01m)
        bool passed = difference < 0.01f;
        if (!passed) {
            all_tests_passed = false;
        }
        
        std::cout << class_id << "\t" << raw_distance << "m\t\t" 
                  << expected_distance << "m\t\t" << actual_distance << "m\t\t" 
                  << (passed ? "PASS" : "FAIL") << "\n";
    }
    
    std::cout << "\nTest result: " << (all_tests_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    
    return all_tests_passed ? 0 : 1;
}