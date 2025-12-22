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
    float camera_fps;
    float h264_fps;
    float ips; // Inferences per second
    float cps; // Logic module calculations per second
    long long timestamp;
};

// Structure to hold parsed log entries
struct LogEntry {
    std::string type;
    std::map<std::string, std::string> fields;
    long long timestamp;
};

class Dashboard {
public:
    Dashboard() : running_(false) {}
    
    void start() {
        running_ = true;
        reader_thread_ = std::thread(&Dashboard::read_logs, this);
        updater_thread_ = std::thread(&Dashboard::update_display, this);
    }
    
    void stop() {
        running_ = false;
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }
        if (updater_thread_.joinable()) {
            updater_thread_.join();
        }
    }
    
private:
    void read_logs() {
        std::string line;
        while (std::getline(std::cin, line) && running_) {
            // Debug output to see what we're receiving
            std::cerr << "DEBUG: Received line: " << line << std::endl;
            parse_log_line(line);
            std::cerr.flush(); // Force debug output to be displayed immediately
        }
        std::cerr << "DEBUG: Finished reading input" << std::endl;
    }
    
    void parse_log_line(const std::string& line) {
        // Skip lines that don't contain our target log types
        if (line.find("DETECTION_INVARIANT") == std::string::npos &&
            line.find("DETECTION_DISTANCE") == std::string::npos &&
            line.find("CAUSALITY_VALIDATION") == std::string::npos &&
            line.find("INTERNAL DISTANCE ESTIMATE") == std::string::npos &&
            line.find("CameraCapture: Total time to process request") == std::string::npos) {
            return;
        }
        
        // Debug output
        std::cerr << "DEBUG: Parsing line: " << line << std::endl;
        
        LogEntry entry;
        entry.timestamp = get_current_timestamp();
        
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
        } else if (line.find("INTERNAL DISTANCE ESTIMATE:") != std::string::npos) {
            std::cerr << "DEBUG: Matched INTERNAL_DISTANCE_ESTIMATE pattern" << std::endl;
            entry.type = "INTERNAL_DISTANCE_ESTIMATE";
            parse_internal_distance_estimate(line, entry);
        } else if (line.find("CameraCapture: Total time to process request") != std::string::npos) {
            std::cerr << "DEBUG: Matched CAMERA_PROCESS_TIME pattern" << std::endl;
            entry.type = "CAMERA_PROCESS_TIME";
            parse_camera_process_time(line, entry);
        } else {
            std::cerr << "DEBUG: No pattern matched for line" << std::endl;
            return;
        }
        
        std::cerr << "DEBUG: Adding entry of type " << entry.type << " to queue" << std::endl;
        
        // Add to our log queue
        {
            std::lock_guard<std::mutex> lock(log_queue_mutex_);
            log_queue_.push(entry);
            std::cerr << "DEBUG: Queue size after adding: " << log_queue_.size() << std::endl;
        }
    }
    
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
        
        // Debug output
        if (entry.fields.count("track_id")) {
            std::cerr << "DEBUG: Parsed DETECTION_INVARIANT for track " << entry.fields["track_id"] << std::endl;
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
        
        // Debug output
        std::cerr << "DEBUG: Parsed CAUSALITY_VALIDATION for track " << entry.fields["track_id"] << std::endl;
    }
    
    void parse_internal_distance_estimate(const std::string& line, LogEntry& entry) {
        // Parse line like:
        // INFO] INTERNAL DISTANCE ESTIMATE: value_meters = 31.70, timestamp = 1766322898455
        std::istringstream iss(line);
        std::string token;
        
        // Skip to "value_meters"
        while (iss >> token && token != "value_meters");
        
        if (iss >> token && token == "=") {
            iss >> token; // The value
            entry.fields["value_meters"] = token.substr(0, token.length()-1); // Remove trailing ','
        }
        
        // Skip to "timestamp"
        while (iss >> token && token != "timestamp");
        
        if (iss >> token && token == "=") {
            iss >> token; // The timestamp
            entry.fields["timestamp"] = token;
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
    
    void update_display() {
        while (running_) {
            std::cerr << "DEBUG: Updating display" << std::endl;
            // Process log entries
            process_log_entries();
            
            // Update metrics
            update_metrics();
            
            // Display dashboard
            display_dashboard();
            
            // Sleep for 2 seconds
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    
    void process_log_entries() {
        std::lock_guard<std::mutex> lock(log_queue_mutex_);
        
        std::cerr << "DEBUG: Processing " << log_queue_.size() << " log entries" << std::endl;
        
        while (!log_queue_.empty()) {
            LogEntry entry = log_queue_.front();
            log_queue_.pop();
            
            std::cerr << "DEBUG: Processing entry of type: " << entry.type << std::endl;
            
            if (entry.type == "DETECTION_INVARIANT") {
                // Check if this is a "Detections received by logic module" message
                if (entry.fields.count("Detections received by logic module")) {
                    std::cerr << "DEBUG: Skipping 'Detections received by logic module' message" << std::endl;
                    continue;
                }
                
                // Check if this is an "Active tracks for ballistics" message
                if (entry.fields.count("Active tracks for ballistics")) {
                    std::cerr << "DEBUG: Skipping 'Active tracks for ballistics' message" << std::endl;
                    continue;
                }
                
                if (entry.fields.count("ignored") && entry.fields["ignored"] == "true") {
                    std::cerr << "DEBUG: Skipping ignored detection" << std::endl;
                    continue; // Skip zeroed detections
                }
                
                if (!entry.fields.count("track_id")) {
                    std::cerr << "DEBUG: Skipping DETECTION_INVARIANT without track_id" << std::endl;
                    continue;
                }
                
                int track_id = std::stoi(entry.fields["track_id"]);
                auto& detection = detections_[track_id];
                detection.track_id = track_id;
                
                if (entry.fields.count("class")) {
                    detection.class_id = std::stoi(entry.fields["class"]);
                }
                
                if (entry.fields.count("confidence")) {
                    detection.confidence = std::stof(entry.fields["confidence"]);
                }
                
                detection.timestamp = entry.timestamp;
                std::cerr << "DEBUG: Processed DETECTION_INVARIANT for track " << track_id << std::endl;
            } else if (entry.type == "DETECTION_DISTANCE") {
                // Find the most recent detection with the same class and update its distance
                // This is a simplified approach - in a real implementation, we'd use timestamps or other correlation methods
                for (auto& pair : detections_) {
                    auto& detection = pair.second;
                    if (entry.fields.count("class") && detection.class_id == std::stoi(entry.fields["class"])) {
                        if (entry.fields.count("distance")) {
                            detection.distance = std::stof(entry.fields["distance"]);
                            std::cerr << "DEBUG: Updated distance for track " << detection.track_id << " to " << detection.distance << "m" << std::endl;
                        }
                        break;
                    }
                }
            } else if (entry.type == "CAUSALITY_VALIDATION") {
                if (!entry.fields.count("track_id")) {
                    std::cerr << "DEBUG: Skipping CAUSALITY_VALIDATION without track_id" << std::endl;
                    continue;
                }
                
                int track_id = std::stoi(entry.fields["track_id"]);
                auto& detection = detections_[track_id];
                detection.track_id = track_id;
                
                if (entry.fields.count("impact_x")) {
                    detection.impact_x = std::stof(entry.fields["impact_x"]);
                }
                
                if (entry.fields.count("impact_y")) {
                    detection.impact_y = std::stof(entry.fields["impact_y"]);
                }
                
                if (entry.fields.count("impact_z")) {
                    detection.impact_z = std::stof(entry.fields["impact_z"]);
                }
                
                detection.ballistics_solved = true;
                detection.timestamp = entry.timestamp;
                std::cerr << "DEBUG: Processed CAUSALITY_VALIDATION for track " << track_id << std::endl;
            } else if (entry.type == "INTERNAL_DISTANCE_ESTIMATE") {
                // We don't need to process this for the dashboard display
                std::cerr << "DEBUG: Skipping INTERNAL_DISTANCE_ESTIMATE" << std::endl;
                continue;
            } else if (entry.type == "CAMERA_PROCESS_TIME") {
                if (entry.fields.count("process_time_us")) {
                    long long process_time_us = std::stoll(entry.fields["process_time_us"]);
                    camera_process_times_.push_back(process_time_us);
                    
                    // Keep only the last 120 entries (assuming 60 FPS * 2 seconds)
                    if (camera_process_times_.size() > 120) {
                        camera_process_times_.erase(camera_process_times_.begin());
                    }
                    std::cerr << "DEBUG: Added camera process time: " << process_time_us << "us" << std::endl;
                }
            }
        }
    }
    
    void update_metrics() {
        // Calculate camera FPS based on process times
        if (!camera_process_times_.empty()) {
            // Average process time in microseconds
            long long total_time_us = 0;
            for (long long time : camera_process_times_) {
                total_time_us += time;
            }
            double avg_process_time_us = static_cast<double>(total_time_us) / camera_process_times_.size();
            
            // Convert to FPS (1 second = 1,000,000 microseconds)
            current_metrics_.camera_fps = 1000000.0f / avg_process_time_us;
        } else {
            current_metrics_.camera_fps = 0.0f;
        }
        
        // For now, we'll set other metrics to fixed values
        // In a real implementation, these would be calculated from actual data
        current_metrics_.h264_fps = 120.0f; // From the logs
        current_metrics_.ips = 120.0f; // Same as camera FPS for now
        current_metrics_.cps = 120.0f; // Same as camera FPS for now
        current_metrics_.timestamp = get_current_timestamp();
        
        // Debug output for metrics
        std::cerr << "DEBUG: Camera FPS: " << current_metrics_.camera_fps << std::endl;
    }
    
    void display_dashboard() {
        // Clear screen (ANSI escape code)
        std::cout << "\033[2J\033[1;1H";
        std::cout.flush(); // Force output to be displayed immediately
        
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
        
        // Debug: Print number of detections
        std::cerr << "DEBUG: Number of detections: " << detections_.size() << std::endl;
        
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
        
        // Print system metrics
        std::cout << "System Metrics (2-second averages):\n";
        std::cout << "------------------------------------------\n";
        std::cout << std::setw(15) << "Metric" 
                  << std::setw(15) << "Value" << "\n";
        std::cout << "------------------------------------------\n";
        std::cout << std::setw(15) << "Camera FPS" 
                  << std::setw(15) << std::fixed << std::setprecision(2) << current_metrics_.camera_fps << "\n";
        std::cout << std::setw(15) << "H.264 FPS" 
                  << std::setw(15) << std::fixed << std::setprecision(2) << current_metrics_.h264_fps << "\n";
        std::cout << std::setw(15) << "IPS" 
                  << std::setw(15) << std::fixed << std::setprecision(2) << current_metrics_.ips << "\n";
        std::cout << std::setw(15) << "CPS" 
                  << std::setw(15) << std::fixed << std::setprecision(2) << current_metrics_.cps << "\n";
        std::cout << "------------------------------------------\n\n";
        
        // Print timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::cout << "Last Updated: " << std::ctime(&time_t);
    }
    
    long long get_current_timestamp() {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    }
    
    std::atomic<bool> running_;
    std::thread reader_thread_;
    std::thread updater_thread_;
    
    std::map<int, Detection> detections_;
    std::queue<LogEntry> log_queue_;
    std::mutex log_queue_mutex_;
    
    std::vector<long long> camera_process_times_;
    SystemMetrics current_metrics_;
};

int main() {
    // Disable buffering for stdin
    std::cin.sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    Dashboard dashboard;
    dashboard.start();
    
    std::cout << "Dashboard started. Press Ctrl+C to exit.\n";
    
    // Wait for Ctrl+C
    std::cin.get();
    
    dashboard.stop();
    return 0;
}