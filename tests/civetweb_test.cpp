#include "CivetServer.h"
#include <iostream>

int main() {
    std::vector<std::string> options = {"listening_ports", "8080"};
    CivetServer server(options);
    std::cout << "Civetweb server started on port 8080." << std::endl;
    while (true) {
        // Keep the server running
    }
    return 0;
}