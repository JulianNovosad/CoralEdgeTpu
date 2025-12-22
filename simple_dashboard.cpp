#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <queue>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <atomic>
#include <mutex>

// Structure to hold parsed log entries
struct LogEntry {
    std::string type;
    std::map<std::string, std::string> fields;
    long long timestamp;
};

void parse_detection_invariant(const std::string& line, LogEntry& entry) {
    // Parse line like:
    // [INFO] DETECTION_INVARIANT: Detection 0: class=27, score=11.011765, area=5075.929688, box=[134.521408,59.189419,203.127319,133.176193], timestamp=1766327902791
    // Or:
    // [INFO] DETECTION_INVARIANT: Detections received by logic module: 100
    
    // Check if this is the "Detections received by logic module" line
    if (line.find("Detections received by logic module") != std::string::npos) {
        // We don't need to process this line for track information
        return;
    }
    
    // Check if this is the "Active tracks for ballistics" line
    if (line.find("Active tracks for ballistics") != std::string::npos) {
        // We don't need to process this line for track information
        return;
    }
    
    // Check if this is a zeroed detection line (class=0, score=0.000000)
    if (line.find("class=0, score=0.000000") != std::string::npos) {
        entry.fields["ignored"] = "true";
        return;
    }
    
    std::istringstream iss(line);
    std::string token;
    
    // Skip to "DETECTION_INVARIANT:"
    while (iss >> token && token != "DETECTION_INVARIANT:");
    
    // Skip "Detection"
    iss >> token;
    
    if (iss >> token) { // Track ID with colon
        entry.fields["track_id"] = token.substr(0, token.length()-1); // Remove trailing ':'
    }
    
    // Parse class, score, area, box, timestamp
    std::string field;
    while (iss >> field) {
        if (field.back() == ',') field.pop_back();
        
        size_t eq_pos = field.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = field.substr(0, eq_pos);
            std::string value = field.substr(eq_pos + 1);
            entry.fields[key] = value;
        }
    }
    
    // Convert score to confidence (0-1)
    if (entry.fields.count("score")) {
        float score = std::stof(entry.fields["score"]);
        entry.fields["confidence"] = std::to_string(score / 100.0f);
    }
}

void parse_detection_distance(const std::string& line, LogEntry& entry) {
    // Parse line like:
    // [INFO] DETECTION_DISTANCE: class=27, score=11.011765, box=[134.521408,59.189419,203.127319,133.176193], distance=1.589222m
    std::istringstream iss(line);
    std::string token;
    
    // Skip to "DETECTION_DISTANCE:"
    while (iss >> token && token != "DETECTION_DISTANCE:");
    
    // Parse class, score, box, distance
    std::string field;
    while (iss >> field) {
        if (field.back() == ',') field.pop_back();
        
        size_t eq_pos = field.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = field.substr(0, eq_pos);
            std::string value = field.substr(eq_pos + 1);
            
            // Remove 'm' from distance
            if (key == "distance" && value.back() == 'm') {
                // Check if value is "too_small"
                if (value == "too_small") {
                    entry.fields[key] = "0.0";
                } else {
                    value.pop_back();
                    entry.fields[key] = value;
                }
            } else {
                entry.fields[key] = value;
            }
        }
    }
}

void parse_causality_validation(const std::string& line, LogEntry& entry) {
    // Parse line like:
    // [INFO] CAUSALITY_VALIDATION: Ballistics solved for Track ID 1: impact=(984.38, -1417.99, 6.73) conf=99.28% detection=[class=40,score=8.0000,box=(102.2,149.3,114.3,162.8)]
    std::istringstream iss(line);
    std::string token;
    
    // Skip to "CAUSALITY_VALIDATION:"
    while (iss >> token && token != "CAUSALITY_VALIDATION:");
    
    // Skip "Ballistics" "solved" "for" "Track"
    iss >> token; // Ballistics
    iss >> token; // solved
    iss >> token; // for
    iss >> token; // Track
    
    if (iss >> token) { // ID
        entry.fields["track_id"] = token.substr(0, token.length()-1); // Remove trailing ':'
    }
    
    // Skip to "impact="
    while (iss >> token && token != "impact=");
    
    if (iss >> token) { // (x,
        token = token.substr(1); // Remove leading '('
        entry.fields["impact_x"] = token.substr(0, token.length()-1); // Remove trailing ','
    }
    
    if (iss >> token) { // y,
        entry.fields["impact_y"] = token.substr(0, token.length()-1); // Remove trailing ','
    }
    
    if (iss >> token) { // z)
        entry.fields["impact_z"] = token.substr(0, token.length()-1); // Remove trailing ')'
    }
    
    // Skip to "conf="
    while (iss >> token && token != "conf=");
    
    if (iss >> token) { // 99.28%
        entry.fields["confidence"] = token.substr(0, token.length()-1); // Remove trailing '%'
    }
}

void parse_camera_process_time(const std::string& line, LogEntry& entry) {
    // Parse line like:
    // [INFO] CameraCapture: Total time to process request (frame_id=0): 1187 us
    std::istringstream iss(line);
    std::string token;
    
    // Skip to "CameraCapture:"
    while (iss >> token && token != "CameraCapture:");
    
    // Skip "Total" "time" "to" "process" "request"
    iss >> token; // Total
    iss >> token; // time
    iss >> token; // to
    iss >> token; // process
    iss >> token; // request
    
    if (iss >> token) { // (frame_id=0):
        // Extract frame_id
        size_t start = token.find('(');
        size_t end = token.find(')');
        if (start != std::string::npos && end != std::string::npos) {
            std::string frame_info = token.substr(start+1, end-start-1);
            size_t eq_pos = frame_info.find('=');
            if (eq_pos != std::string::npos) {
                entry.fields["frame_id"] = frame_info.substr(eq_pos+1);
            }
        }
    }
    
    if (iss >> token) { // 1187
        entry.fields["process_time_us"] = token;
    }
}

void parse_log_line(const std::string& line, std::queue<LogEntry>& log_queue_) {
    // Skip lines that don't contain our target log types
    if (line.find("DETECTION_INVARIANT") == std::string::npos &&
        line.find("DETECTION_DISTANCE") == std::string::npos &&
        line.find("CAUSALITY_VALIDATION") == std::string::npos &&
        line.find("INTERNAL DISTANCE ESTIMATE") == std::string::npos &&
        line.find("CameraCapture: Total time to process request") == std::string::npos) {
        return;
    }
    
    std::cerr << "DEBUG: Parsing line: " << line << std::endl;
    
    LogEntry entry;
    entry.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    
    if (line.find("DETECTION_INVARIANT: Detection") != std::string::npos) {
        std::cerr << "DEBUG: Matched DETECTION_INVARIANT pattern" << std::endl;
        entry.type = "DETECTION_INVARIANT";
        parse_detection_invariant(line, entry);
    } else if (line.find("DETECTION_DISTANCE:") != std::string::npos) {
        std::cerr << "DEBUG: Matched DETECTION_DISTANCE pattern" << std::endl;
        entry.type = "DETECTION_DISTANCE";
        parse_detection_distance(line, entry);
    } else if (line.find("CAUSALITY_VALIDATION:") != std::string::npos) {
        std::cerr << "DEBUG: Matched CAUSALITY_VALIDATION pattern" << std::endl;
        entry.type = "CAUSALITY_VALIDATION";
        parse_causality_validation(line, entry);
    } else if (line.find("CameraCapture: Total time to process request") != std::string::npos) {
        std::cerr << "DEBUG: Matched CAMERA_PROCESS_TIME pattern" << std::endl;
        entry.type = "CAMERA_PROCESS_TIME";
        parse_camera_process_time(line, entry);
    } else {
        std::cerr << "DEBUG: No pattern matched for line" << std::endl;
        return;
    }
    
    std::cerr << "DEBUG: Adding entry of type " << entry.type << " to queue" << std::endl;
    log_queue_.push(entry);
    std::cerr << "DEBUG: Queue size after adding: " << log_queue_.size() << std::endl;
}

int main() {
    std::queue<LogEntry> log_queue_;
    std::string line;
    
    while (std::getline(std::cin, line)) {
        std::cerr << "DEBUG: Received line: " << line << std::endl;
        parse_log_line(line, log_queue_);
    }
    
    std::cerr << "DEBUG: Finished reading input. Queue size: " << log_queue_.size() << std::endl;
    
    // Process all entries in the queue
    while (!log_queue_.empty()) {
        LogEntry entry = log_queue_.front();
        log_queue_.pop();
        std::cerr << "DEBUG: Processing entry of type: " << entry.type << std::endl;
        if (entry.fields.count("track_id")) {
            std::cerr << "  Track ID: " << entry.fields["track_id"] << std::endl;
        }
        if (entry.fields.count("class")) {
            std::cerr << "  Class: " << entry.fields["class"] << std::endl;
        }
        if (entry.fields.count("confidence")) {
            std::cerr << "  Confidence: " << entry.fields["confidence"] << std::endl;
        }
    }
    
    return 0;
}