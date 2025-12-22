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
#include <set>
#include <unistd.h>

// Structure to hold detection information
struct Detection {
    int track_id;
    int class_id;
    float confidence;
    float distance;
    std::string servo_status;
    float impact_x;
    float impact_y;
    float impact_z;
    bool ballistics_solved;
    long long timestamp;
};

// Structure to hold system metrics
struct SystemMetrics {
    float camera_capture_rate_fps;     // True camera capture rate (frames per second)
    float camera_processing_latency_us; // Average processing latency per frame (microseconds)
    float ips;                        // Inferences per second
    float h264_fps;                   // H.264 encoding rate
    float cps;                        // Logic module calculations per second
    long long timestamp;
};

// Structure to hold parsed log entries
struct LogEntry {
    std::string type;
    std::map<std::string, std::string> fields;
    long long timestamp;
};

// Structure to hold frame timing information
struct FrameTiming {
    int frame_id;
    long long process_time_us;
    long long capture_timestamp;  // When the frame was captured
};

class Dashboard {
public:
    Dashboard() : last_metrics_update_(get_current_timestamp()) {}
    
    void run() {
        std::string line;
        last_metrics_update_ = get_current_timestamp();
        
        // Use non-blocking input with timeout
        std::cin.setstate(std::ios::goodbit); // Clear any error flags
        
        while (true) {
            // Check for input with a timeout mechanism
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(STDIN_FILENO, &read_fds);
            
            struct timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 100000; // 100ms timeout
            
            int result = select(STDIN_FILENO + 1, &read_fds, NULL, NULL, &timeout);
            
            if (result > 0 && FD_ISSET(STDIN_FILENO, &read_fds)) {
                if (std::getline(std::cin, line)) {
                    // Parse the line
                    parse_log_line(line);
                } else {
                    // EOF or error
                    break;
                }
            } else if (result == -1) {
                // Error in select
                break;
            }
            // If result == 0, timeout occurred, continue to update display
            
            // Update display every 2 seconds
            long long now = get_current_timestamp();
            if (now - last_metrics_update_ >= 2000) { // 2 seconds
                display_dashboard();
                last_metrics_update_ = now;
            }
        }
    }
    
private:
    void parse_log_line(const std::string& line) {
        // Skip lines that don't contain our target log types
        if (line.find("DETECTION_INVARIANT") == std::string::npos &&
            line.find("DETECTION_DISTANCE") == std::string::npos &&
            line.find("CAUSALITY_VALIDATION") == std::string::npos &&
            line.find("CameraCapture: Total time to process request") == std::string::npos &&
            line.find("LogicModule: Pop successful") == std::string::npos) {
            return;
        }
        
        LogEntry entry;
        entry.timestamp = get_current_timestamp();
        
        if (line.find("DETECTION_INVARIANT: Detection") != std::string::npos) {
            entry.type = "DETECTION_INVARIANT";
            parse_detection_invariant(line, entry);
        } else if (line.find("DETECTION_DISTANCE:") != std::string::npos) {
            entry.type = "DETECTION_DISTANCE";
            parse_detection_distance(line, entry);
        } else if (line.find("CAUSALITY_VALIDATION:") != std::string::npos) {
            entry.type = "CAUSALITY_VALIDATION";
            parse_causality_validation(line, entry);
        } else if (line.find("CameraCapture: Total time to process request") != std::string::npos) {
            entry.type = "CAMERA_PROCESS_TIME";
            parse_camera_process_time(line, entry);
        } else if (line.find("LogicModule: Pop successful") != std::string::npos) {
            entry.type = "LOGIC_MODULE_PROCESS";
            parse_logic_module_process(line, entry);
        } else {
            return;
        }
        
        // Process the entry immediately
        process_log_entry(entry);
    }
    
    void parse_detection_invariant(const std::string& line, LogEntry& entry) {
        // Parse line like:
        // [INFO] DETECTION_INVARIANT: Detection 0: class=27, score=11.011765, area=5075.929688, box=[134.521408,59.189419,203.127319,133.176193], timestamp=1766327902791
        // Or:
        // [INFO] DETECTION_INVARIANT: Detections received by logic module: 100
        
        // Check if this is the "Detections received by logic module" line
        if (line.find("Detections received by logic module") != std::string::npos) {
            return;
        }
        
        // Check if this is the "Active tracks for ballistics" line
        if (line.find("Active tracks for ballistics") != std::string::npos) {
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
            try {
                float score = std::stof(entry.fields["score"]);
                entry.fields["confidence"] = std::to_string(score / 100.0f);
            } catch (...) {
                entry.fields["confidence"] = "0.0";
            }
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
    
    void parse_logic_module_process(const std::string& line, LogEntry& entry) {
        // Parse line like:
        // [INFO] LogicModule: Pop successful. Detections buffer received.
        // This indicates one inference processing cycle
        (void)line; // Suppress unused parameter warning
        entry.fields["inference_cycle"] = "1";
    }
    
    void process_log_entry(const LogEntry& entry) {
        if (entry.type == "DETECTION_INVARIANT") {
            // Check if this is a "Detections received by logic module" message
            if (entry.fields.count("Detections received by logic module")) {
                return;
            }
            
            // Check if this is an "Active tracks for ballistics" message
            if (entry.fields.count("Active tracks for ballistics")) {
                return;
            }
            
            if (entry.fields.count("ignored") && entry.fields.at("ignored") == "true") {
                return; // Skip zeroed detections
            }
            
            if (!entry.fields.count("track_id")) {
                return;
            }
            
            try {
                int track_id = std::stoi(entry.fields.at("track_id"));
                auto& detection = detections_[track_id];
                detection.track_id = track_id;
                
                if (entry.fields.count("class")) {
                    detection.class_id = std::stoi(entry.fields.at("class"));
                }
                
                if (entry.fields.count("confidence")) {
                    detection.confidence = std::stof(entry.fields.at("confidence"));
                }
                
                detection.timestamp = entry.timestamp;
            } catch (const std::exception& e) {
                // Ignore parsing errors
                return;
            }
        } else if (entry.type == "DETECTION_DISTANCE") {
            // Find the most recent detection with the same class and update its distance
            for (auto& pair : detections_) {
                auto& detection = pair.second;
                if (entry.fields.count("class")) {
                    try {
                        if (detection.class_id == std::stoi(entry.fields.at("class"))) {
                            if (entry.fields.count("distance")) {
                                detection.distance = std::stof(entry.fields.at("distance"));
                            }
                            break;
                        }
                    } catch (const std::exception& e) {
                        // Ignore parsing errors
                        continue;
                    }
                }
            }
        } else if (entry.type == "CAUSALITY_VALIDATION") {
            if (!entry.fields.count("track_id")) {
                return;
            }
            
            try {
                int track_id = std::stoi(entry.fields.at("track_id"));
                auto& detection = detections_[track_id];
                detection.track_id = track_id;
                
                if (entry.fields.count("impact_x")) {
                    detection.impact_x = std::stof(entry.fields.at("impact_x"));
                }
                
                if (entry.fields.count("impact_y")) {
                    detection.impact_y = std::stof(entry.fields.at("impact_y"));
                }
                
                if (entry.fields.count("impact_z")) {
                    detection.impact_z = std::stof(entry.fields.at("impact_z"));
                }
                
                detection.ballistics_solved = true;
                detection.timestamp = entry.timestamp;
            } catch (const std::exception& e) {
                // Ignore parsing errors
                return;
            }
        } else if (entry.type == "CAMERA_PROCESS_TIME") {
            if (entry.fields.count("process_time_us") && entry.fields.count("frame_id")) {
                try {
                    long long process_time_us = std::stoll(entry.fields.at("process_time_us"));
                    int frame_id = std::stoi(entry.fields.at("frame_id"));
                    
                    // Store frame timing information
                    FrameTiming timing;
                    timing.frame_id = frame_id;
                    timing.process_time_us = process_time_us;
                    timing.capture_timestamp = entry.timestamp;
                    
                    camera_frame_timings_.push_back(timing);
                    
                    // Keep only the last 500 entries for memory efficiency
                    const size_t max_window_size = 500;
                    if (camera_frame_timings_.size() > max_window_size) {
                        camera_frame_timings_.erase(camera_frame_timings_.begin(), 
                                                   camera_frame_timings_.begin() + (camera_frame_timings_.size() - max_window_size));
                    }
                } catch (const std::exception& e) {
                    // Ignore parsing errors
                    return;
                }
            }
        } else if (entry.type == "LOGIC_MODULE_PROCESS") {
            // Count inference processing cycles
            inference_events_.push_back(entry.timestamp);
            
            // Keep only events from the last 10 seconds for memory efficiency
            const long long max_window_ms = 10000;
            long long window_start = entry.timestamp - max_window_ms;
            
            // Remove old events
            while (!inference_events_.empty() && inference_events_.front() < window_start) {
                inference_events_.pop_front();
            }
        }
    }
    
    void update_metrics() {
        long long now = get_current_timestamp();
        
        // Calculate true camera capture rate (frames per second)
        if (!camera_frame_timings_.empty()) {
            // Count unique frames in the last 2 seconds
            long long window_start = now - 2000; // 2 seconds ago
            int frames_in_window = 0;
            
            // Count frames in the time window
            for (const auto& timing : camera_frame_timings_) {
                if (timing.capture_timestamp >= window_start) {
                    frames_in_window++;
                }
            }
            
            // Calculate FPS based on actual frame arrival rate over the last 2 seconds
            current_metrics_.camera_capture_rate_fps = static_cast<float>(frames_in_window) / 2.0f;
            
            // Calculate average processing latency
            long long total_process_time = 0;
            int process_count = 0;
            
            // Calculate average over last 100 frames for latency
            size_t start_idx = camera_frame_timings_.size() > 100 ? 
                              camera_frame_timings_.size() - 100 : 0;
            
            for (size_t i = start_idx; i < camera_frame_timings_.size(); ++i) {
                total_process_time += camera_frame_timings_[i].process_time_us;
                process_count++;
            }
            
            if (process_count > 0) {
                current_metrics_.camera_processing_latency_us = 
                    static_cast<float>(total_process_time) / process_count;
            } else {
                current_metrics_.camera_processing_latency_us = 0.0f;
            }
        } else {
            current_metrics_.camera_capture_rate_fps = 0.0f;
            current_metrics_.camera_processing_latency_us = 0.0f;
        }
        
        // Calculate IPS (Inferences Per Second)
        if (!inference_events_.empty()) {
            // Count events in the last 2 seconds
            long long window_start = now - 2000; // 2 seconds ago
            int events_in_window = 0;
            
            for (const auto& timestamp : inference_events_) {
                if (timestamp >= window_start) {
                    events_in_window++;
                }
            }
            
            // Calculate IPS based on events in the last 2 seconds
            current_metrics_.ips = static_cast<float>(events_in_window) / 2.0f;
        } else {
            current_metrics_.ips = 0.0f;
        }
        
        // Set other metrics to N/A values
        current_metrics_.h264_fps = 0.0f; // Will implement if H.264 messages are found
        current_metrics_.cps = 0.0f; // Will implement if logic module cycle messages are found
        current_metrics_.timestamp = now;
    }
    
    void display_dashboard() {
        // Clear screen (ANSI escape code)
        std::cout << "\033[2J\033[1;1H";
        
        // Print header
        std::cout << "==========================================\n";
        std::cout << "     CoralEdgeTpu Detector Dashboard      \n";
        std::cout << "==========================================\n\n";
        
        // Print active tracks
        std::cout << "Active Tracks:\n";
        std::cout << "------------------------------------------\n";
        std::cout << std::setw(8) << "Track ID" 
                  << std::setw(8) << "Class" 
                  << std::setw(12) << "Confidence" 
                  << std::setw(10) << "Distance" 
                  << std::setw(15) << "Servo Status" 
                  << std::setw(20) << "Impact (x,y,z)" << "\n";
        std::cout << "------------------------------------------\n";
        
        for (const auto& pair : detections_) {
            const Detection& detection = pair.second;
            std::cout << std::setw(8) << detection.track_id
                      << std::setw(8) << detection.class_id
                      << std::setw(12) << std::fixed << std::setprecision(2) << detection.confidence
                      << std::setw(10) << std::fixed << std::setprecision(2) << detection.distance << "m"
                      << std::setw(15) << detection.servo_status
                      << std::setw(20);
            
            if (detection.ballistics_solved) {
                std::cout << "(" << std::fixed << std::setprecision(2) << detection.impact_x << ","
                          << std::fixed << std::setprecision(2) << detection.impact_y << ","
                          << std::fixed << std::setprecision(2) << detection.impact_z << ")";
            } else {
                std::cout << "N/A";
            }
            std::cout << "\n";
        }
        
        std::cout << "\n";
        
        // Update and print system metrics
        update_metrics();
        
        std::cout << "System Metrics (2-second averages):\n";
        std::cout << "------------------------------------------\n";
        std::cout << std::setw(30) << "Metric" 
                  << std::setw(15) << "Value" << "\n";
        std::cout << "------------------------------------------\n";
        std::cout << std::setw(30) << "Camera Capture Rate (FPS)" 
                  << std::setw(15) << std::fixed << std::setprecision(2) << current_metrics_.camera_capture_rate_fps << "\n";
        std::cout << std::setw(30) << "Camera Processing Latency (µs)" 
                  << std::setw(15) << std::fixed << std::setprecision(2) << current_metrics_.camera_processing_latency_us << "\n";
        std::cout << std::setw(30) << "Inferences Per Second (IPS)" 
                  << std::setw(15) << std::fixed << std::setprecision(2) << current_metrics_.ips << "\n";
        std::cout << std::setw(30) << "H.264 FPS" 
                  << std::setw(15) << "N/A" << "\n";
        std::cout << std::setw(30) << "Calculations Per Second (CPS)" 
                  << std::setw(15) << "N/A" << "\n";
        std::cout << "------------------------------------------\n\n";
        
        // Print timestamp
        auto now_time = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now_time);
        std::cout << "Last Updated: " << std::ctime(&time_t);
        
        std::cout.flush();
    }
    
    long long get_current_timestamp() {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    }
    
    std::map<int, Detection> detections_;
    std::vector<FrameTiming> camera_frame_timings_;
    std::deque<long long> inference_events_;  // Timestamps of inference events
    long long last_metrics_update_;
    SystemMetrics current_metrics_;
};

int main() {
    Dashboard dashboard;
    dashboard.run();
    return 0;
}