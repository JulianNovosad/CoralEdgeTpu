#!/bin/bash

# Script to run CoralEdgeTpu detector under gdb with 4-hour timeout
# Automatically captures backtrace on segfault

LOG_DIR="/home/pi/CoralEdgeTpu/logs"
OUTPUT_LOG="${LOG_DIR}/gdb_session_output.log"
CRASH_LOG="${LOG_DIR}/crash_info.log"
BACKTRACE_FILE="${LOG_DIR}/gdb_backtrace_4h.txt"

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

echo "=== CoralEdgeTpu Detector 4-Hour Stress Test ===" > "${OUTPUT_LOG}"
echo "Started at: $(date)" >> "${OUTPUT_LOG}"
echo "Log file: ${OUTPUT_LOG}" >> "${OUTPUT_LOG}"
echo "Crash log: ${CRASH_LOG}" >> "${OUTPUT_LOG}"
echo "Backtrace file: ${BACKTRACE_FILE}" >> "${OUTPUT_LOG}"
echo "=================================================" >> "${OUTPUT_LOG}"

# Create a temporary gdb command file
GDB_CMD_FILE="/tmp/gdb_commands_4h.txt"
cat > "${GDB_CMD_FILE}" << 'EOF'
# GDB initialization commands
set confirm off
set pagination off
set logging file /home/pi/CoralEdgeTpu/logs/gdb_debug.log
set logging on

# Run the program
run

# Commands executed only if program crashes
echo \n=== SEGMENTATION FAULT DETECTED ===\n
echo Crash time: 
shell date
echo \n

# Save backtrace to file
set logging file /home/pi/CoralEdgeTpu/logs/gdb_backtrace_4h.txt
set logging on
echo === BACKTRACE AT TIME OF CRASH ===
thread apply all bt
set logging off

# Additional crash information
echo \n=== ADDITIONAL CRASH INFORMATION ===\n
info registers
echo \n
info threads
echo \n

# Exit gdb
quit
EOF

echo "GDB command file created at: ${GDB_CMD_FILE}" >> "${OUTPUT_LOG}"

# Run detector under gdb with 4-hour timeout
echo "Starting detector under gdb with 4-hour timeout..." >> "${OUTPUT_LOG}"
timeout 4h gdb -x "${GDB_CMD_FILE}" ./detector 2>&1 | tee -a "${OUTPUT_LOG}"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
echo "GDB session completed at: $(date)" >> "${OUTPUT_LOG}"
echo "Exit code: ${EXIT_CODE}" >> "${OUTPUT_LOG}"

if [ ${EXIT_CODE} -eq 124 ]; then
    echo "SUCCESS: 4-hour timeout reached without segmentation fault" >> "${OUTPUT_LOG}"
    echo "Test completed normally - no crashes detected" >> "${CRASH_LOG}"
elif [ ${EXIT_CODE} -eq 0 ]; then
    echo "Program exited normally" >> "${OUTPUT_LOG}"
    echo "Test completed normally - no crashes detected" >> "${CRASH_LOG}"
else
    echo "WARNING: Program terminated with exit code ${EXIT_CODE}" >> "${OUTPUT_LOG}"
    echo "Check ${OUTPUT_LOG} for details" >> "${CRASH_LOG}"
fi

# Clean up temporary file
rm -f "${GDB_CMD_FILE}"

echo "Session ended at: $(date)" >> "${OUTPUT_LOG}"
echo "=== End of Session ===" >> "${OUTPUT_LOG}"