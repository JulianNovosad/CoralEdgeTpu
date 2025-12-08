#include "application.h"
#include <fstream>
#include <vector>
#include <string>
#include "util_logging.h"

// This is a global flag checked by the main loop and set by the ApplicationSupervisor.
std::atomic<bool> shutdown_requested(false);

std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open labels file: " + path);
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

int main(int argc, char** argv) {
    Application app(argc, argv);
    return app.run();
}