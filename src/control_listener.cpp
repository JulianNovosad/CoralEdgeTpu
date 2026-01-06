// Verified headers: [iostream, string, vector, netinet/in.h, unistd.h...]
// Verification timestamp: 2026-01-06 17:08:04
#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <poll.h>
#include <cstdlib>
#include <fstream>
#include <signal.h>

#define PORT 6005
#define LOG_FILE "/home/pi/detector.log"

void log_message(const std::string& message) {
    std::ofstream log_stream(LOG_FILE, std::ios_base::app);
    if (log_stream.is_open()) {
        // Get current time
        time_t now = time(0);
        char* dt = ctime(&now);
        std::string time_str(dt);
        time_str.pop_back(); // Remove trailing newline

        log_stream << "[" << time_str << "] [control_listener] " << message << std::endl;
    }
    std::cout << "[control_listener] " << message << std::endl;
}

void handle_client(int client_fd) {
    char buffer[1024] = {0};
    ssize_t valread = read(client_fd, buffer, 1024);

    if (valread > 0) {
        std::string command(buffer, valread);
        command.erase(command.find_last_not_of(" \n\r\t") + 1);

        log_message("Received command: '" + command + "'");

        std::stringstream ss(command);
        std::string action;
        ss >> action;

        if (action == "START") {
            std::string phone_ip;
            ss >> phone_ip;
            if (!phone_ip.empty()) {
                log_message("Executing START detector script.");
                std::string start_command = "/home/pi/CoralEdgeTpu/start_detector.sh " + phone_ip + " > /dev/null 2>&1 &";
                system(start_command.c_str());
            } else {
                log_message("START command missing IP address.");
            }
        } else if (action == "STOP") {
            log_message("Executing STOP detector script.");
            system("/home/pi/CoralEdgeTpu/stop_detector.sh > /dev/null 2>&1 &");
        } else {
            log_message("Unknown command received: " + action);
        }
    }
    close(client_fd);
}

int main() {
    int server_fd;
    struct sockaddr_in address;
    int opt = 1;

    // Ignore SIGPIPE
    signal(SIGPIPE, SIG_IGN);

    log_message("Starting control_listener...");

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        log_message("Socket creation failed");
        return -1;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        log_message("setsockopt failed");
        return -1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        log_message("Bind failed on port " + std::to_string(PORT));
        return -1;
    }

    if (listen(server_fd, 3) < 0) {
        log_message("Listen failed");
        return -1;
    }

    log_message("Listener started on port " + std::to_string(PORT));

    struct pollfd pfd;
    pfd.fd = server_fd;
    pfd.events = POLLIN;

    while (true) {
        int res = poll(&pfd, 1, -1); // Wait indefinitely
        if (res > 0 && (pfd.revents & POLLIN)) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
            
            if (client_fd >= 0) {
                handle_client(client_fd);
            }
        }
    }

    close(server_fd);
    return 0;
}
