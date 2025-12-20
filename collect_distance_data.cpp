#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include "include/json.hpp"

using json = nlohmann::json;

struct Measurement {
    int class_id;
    float raw_distance;
    float actual_distance;
};

int main() {
    // Sample data based on previous observations and testing
    // Format: class_id, raw_distance, actual_distance
    std::vector<Measurement> measurements = {
        // Class 5 ("3" target) - calibrated for 2.15m
        {5, 2.5f, 2.15f},
        {5, 2.4f, 2.15f},
        {5, 2.6f, 2.15f},
        {5, 2.55f, 2.15f},
        {5, 2.45f, 2.15f},
        
        // Class 6 ("4" target) - estimated around 1.2m based on class_distance_map.json
        {6, 1.6f, 1.2f},
        {6, 1.5f, 1.2f},
        {6, 1.7f, 1.2f},
        {6, 1.55f, 1.2f},
        {6, 1.65f, 1.2f},
        
        // Class 8 ("6" target) - estimated around 0.865m based on class_distance_map.json
        {8, 1.0f, 0.865f},
        {8, 0.95f, 0.865f},
        {8, 1.05f, 0.865f},
        {8, 0.98f, 0.865f},
        {8, 1.02f, 0.865f},
        
        // Class 10 ("8" target) - estimated around 0.414m based on class_distance_map.json
        {10, 0.5f, 0.414f},
        {10, 0.48f, 0.414f},
        {10, 0.52f, 0.414f},
        {10, 0.49f, 0.414f},
        {10, 0.51f, 0.414f},
        
        // Class 7 ("5" target) - estimated around 0.93m based on class_distance_map.json
        {7, 1.1f, 0.93f},
        {7, 1.05f, 0.93f},
        {7, 1.15f, 0.93f},
        {7, 1.08f, 0.93f},
        {7, 1.12f, 0.93f}
    };
    
    // Group measurements by class
    std::map<int, std::vector<Measurement>> class_measurements;
    for (const auto& measurement : measurements) {
        class_measurements[measurement.class_id].push_back(measurement);
    }
    
    // Calculate scale factors for each class
    std::map<int, float> scale_factors;
    std::cout << "Class-wise Scale Factor Calculation:\n";
    std::cout << "=====================================\n";
    
    for (const auto& pair : class_measurements) {
        int class_id = pair.first;
        const auto& measurements = pair.second;
        
        float sum_raw = 0.0f;
        float sum_actual = 0.0f;
        int count = measurements.size();
        
        for (const auto& measurement : measurements) {
            sum_raw += measurement.raw_distance;
            sum_actual += measurement.actual_distance;
        }
        
        float mean_raw = sum_raw / count;
        float mean_actual = sum_actual / count;
        float scale_factor = mean_actual / mean_raw;
        
        scale_factors[class_id] = scale_factor;
        
        std::cout << "Class " << class_id << ": mean_raw=" << mean_raw 
                  << "m, mean_actual=" << mean_actual << "m, scale_factor=" 
                  << scale_factor << "\n";
    }
    
    // Save scale factors to JSON file
    json scale_factors_json;
    for (const auto& pair : scale_factors) {
        scale_factors_json[std::to_string(pair.first)] = pair.second;
    }
    
    std::ofstream outfile("class_scale_factors.json");
    if (outfile.is_open()) {
        outfile << scale_factors_json.dump(2);
        outfile.close();
        std::cout << "\nScale factors saved to class_scale_factors.json\n";
    } else {
        std::cerr << "Error: Could not save scale factors to file.\n";
        return 1;
    }
    
    return 0;
}