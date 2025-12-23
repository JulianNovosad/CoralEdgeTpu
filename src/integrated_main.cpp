#include <iostream>
#include <thread>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>

// Global variables for process management
static volatile bool shutdown_requested = false;
static pid_t detector_pid = -1;
static const char* pipe_path = "/tmp/detector_pipe";

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "Received signal " << signal << ", requesting shutdown..." << std::endl;
    shutdown_requested = true;
    
    // Forward the signal to the detector process
    if (detector_pid > 0) {
        std::cout << "Forwarding signal to detector process (PID: " << detector_pid << ")" << std::endl;
        // First try graceful termination
        kill(detector_pid, SIGTERM);
        
        // Wait briefly for graceful shutdown
        usleep(500000); // 500ms
        
        // Check if process is still alive and kill if necessary
        if (kill(detector_pid, 0) == 0) {  // Process still exists
            std::cout << "Detector process still alive, sending SIGKILL..." << std::endl;
            kill(detector_pid, SIGKILL);
        }
    }
}

// Function to create FIFO pipe
bool create_fifo_pipe() {
    // Remove existing pipe if it exists
    unlink(pipe_path);
    
    // Create FIFO pipe
    if (mkfifo(pipe_path, 0666) == -1) {
        perror("mkfifo");
        return false;
    }
    
    return true;
}

// Function to remove FIFO pipe
void remove_fifo_pipe() {
    unlink(pipe_path);
}

// Function to start detector process
bool start_detector() {
    // Fork to create child process for detector
    detector_pid = fork();
    
    if (detector_pid == -1) {
        perror("fork");
        return false;
    }
    
    if (detector_pid == 0) {
        // Child process - run detector
        // Redirect stdout to FIFO pipe
        int pipe_fd = open(pipe_path, O_WRONLY);
        if (pipe_fd == -1) {
            perror("open pipe for writing");
            exit(1);
        }
        
        if (dup2(pipe_fd, STDOUT_FILENO) == -1) {
            perror("dup2 stdout");
            close(pipe_fd);
            exit(1);
        }
        
        close(pipe_fd);
        
        // Change working directory to build directory
        if (chdir("/home/pi/CoralEdgeTpu/build") == -1) {
            perror("chdir to build directory");
            exit(1);
        }
        
        // Execute detector binary
        execl("./detector", "./detector", (char*)NULL);
        
        // If execl returns, it failed
        perror("execl detector");
        exit(1);
    } else {
        // Parent process continues
        std::cout << "Started detector process with PID " << detector_pid << std::endl;
        return true;
    }
}

// Function to stop detector process
void stop_detector() {
    if (detector_pid > 0) {
        std::cout << "Terminating detector process..." << std::endl;
        
        // Try graceful termination first
        kill(detector_pid, SIGTERM);
        
        // Wait briefly for graceful shutdown (non-blocking)
        int status;
        pid_t result = waitpid(detector_pid, &status, WNOHANG);
        if (result == 0) {
            // Process didn't exit immediately, wait a bit more
            usleep(500000); // 500ms
            
            // Check again
            result = waitpid(detector_pid, &status, WNOHANG);
            if (result == 0) {
                // Process still hasn't exited, force kill
                std::cout << "Detector process not responding to SIGTERM, sending SIGKILL..." << std::endl;
                kill(detector_pid, SIGKILL);
                
                // Wait for the process to be killed
                waitpid(detector_pid, &status, 0);
            }
        }
        detector_pid = -1;
    }
}

int main(int /*argc*/, char** /*argv*/) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create FIFO pipe
    if (!create_fifo_pipe()) {
        std::cerr << "Failed to create FIFO pipe at " << pipe_path << std::endl;
        return 1;
    }
    
    std::cout << "Created FIFO pipe at " << pipe_path << std::endl;
    
    // Start detector process
    if (!start_detector()) {
        std::cerr << "Failed to start detector process" << std::endl;
        remove_fifo_pipe();
        return 1;
    }
    
    // Open FIFO pipe for reading
    int pipe_fd = open(pipe_path, O_RDONLY);
    if (pipe_fd == -1) {
        perror("open pipe for reading");
        stop_detector();
        remove_fifo_pipe();
        return 1;
    }
    
    // Redirect stdin to read from FIFO pipe
    if (dup2(pipe_fd, STDIN_FILENO) == -1) {
        perror("dup2 stdin");
        close(pipe_fd);
        stop_detector();
        remove_fifo_pipe();
        return 1;
    }
    
    close(pipe_fd);
    
    // Main loop - wait for detector process or shutdown signal
    while (!shutdown_requested) {
        // Wait for a short time to check if shutdown was requested
        sleep(1);
        
        // Check if detector process is still running
        int status;
        pid_t result = waitpid(detector_pid, &status, WNOHANG);
        if (result == detector_pid) {
            // Detector process has exited
            std::cout << "Detector process has exited." << std::endl;
            break;
        } else if (result == -1) {
            perror("waitpid");
            break;
        }
    }
    
    // If shutdown was requested, stop the detector
    if (shutdown_requested) {
        stop_detector();
    }
    
    remove_fifo_pipe();
    
    // Create and run dashboard by executing the dashboard binary
    std::cout << "Starting integrated dashboard..." << std::endl;
    
    // Execute dashboard binary with stdin redirected from the pipe
    execl("/home/pi/CoralEdgeTpu/build/dashboard", "/home/pi/CoralEdgeTpu/build/dashboard", (char*)NULL);
    
    // If execl returns, it failed
    perror("execl dashboard");
    stop_detector();
    remove_fifo_pipe();
    return 1;
}