#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Function to run a single test iteration
run_test() {
    set -e # Exit immediately if a command exits with a non-zero status
    local run_number=$1
    echo "Starting Build for Run $run_number..."
    
    # Clean and build aggressively
    make clean || true # Attempt make clean, ignore errors if Makefile doesn't exist
    rm -rf build
    cmake -Bbuild -H. --trace-expand --debug-output
    if [ $? -ne 0 ]; then
        echo "CMake configure failed for Run $run_number." >&2
        exit 1
    fi
    cmake --build build -j2 -v
    if [ $? -ne 0 ]; then
        echo "CMake build failed for Run $run_number." >&2
        exit 1
    fi
    echo "Build complete for Run $run_number."
    if [ $? -ne 0 ]; then
        echo "Build failed for Run $run_number. Printing build_run${run_number}_stderr.log:" >&2
        cat build_run${run_number}_stderr.log >&2
        exit 1
    fi
    echo "Build complete for Run $run_number."

    echo "Starting stability test (60 seconds) for Run $run_number..."
    ./build/detector > detector_run${run_number}_stdout.log 2> detector_run${run_number}_stderr.log &
    DETECTOR_PID=$!

    echo "Detector running for Run $run_number with PID $DETECTOR_PID"
    echo "Monitoring logs for 2s before sending START signal..."
    # Give detector time to initialize, but tail its output
    timeout 2 tail -f detector_run${run_number}_stdout.log detector_run${run_number}_stderr.log || true

    echo "Sending START command for Run $run_number..."
    echo "START 127.0.0.1" | nc -w 1 localhost 6005 || echo "Failed to send START command (nc not found or port closed)"

    echo "Test in progress (60s). Tail-ing logs for Run $run_number..."
    for i in {1..6}; do
        echo "Progress for Run $run_number: $((i*10))s elapsed..."
        timeout 10 tail -f detector_run${run_number}_stdout.log detector_run${run_number}_stderr.log || true
    done

    echo "60 seconds elapsed. Killing detector for Run $run_number..."
    kill -9 $DETECTOR_PID
    wait $DETECTOR_PID 2>/dev/null || true

    echo "Stability test complete for Run $run_number."
    
    echo "--- Logs for Run $run_number ---"
    echo "STDOUT:"
    tail -n 50 detector_run${run_number}_stdout.log
    echo "STDERR:"
    tail -n 50 detector_run${run_number}_stderr.log
    echo "---------------------------"
}

# Run the tests twice
run_test 1
run_test 2

echo "All stability tests complete."
