#!/bin/bash

# Script to run detector under gdb for 4 hours stress test
# Will automatically capture backtrace if segfault occurs

echo "Starting 4-hour stress test of CoralEdgeTpu detector..."
echo "Test started at: $(date)" > /home/pi/CoralEdgeTpu/stress_test_log.txt

# Run detector under gdb with automatic backtrace capture on crash
timeout 14400s gdb -batch \
  -ex "set confirm off" \
  -ex "set pagination off" \
  -ex "run" \
  -ex "echo \\n=== SEGFAULT DETECTED ===\\n" \
  -ex "echo Crash occurred at: $(date)\\n" >> /home/pi/CoralEdgeTpu/stress_test_log.txt \
  -ex "set logging file /home/pi/CoralEdgeTpu/gdb_backtrace_4h.txt" \
  -ex "set logging on" \
  -ex "thread apply all bt" \
  -ex "set logging off" \
  -ex "echo Backtrace saved to gdb_backtrace_4h.txt\\n" >> /home/pi/CoralEdgeTpu/stress_test_log.txt \
  -ex "quit" \
  ./detector

# Check exit status
EXIT_CODE=$?
echo "Test completed at: $(date)" >> /home/pi/CoralEdgeTpu/stress_test_log.txt
echo "Exit code: $EXIT_CODE" >> /home/pi/CoralEdgeTpu/stress_test_log.txt

if [ $EXIT_CODE -eq 124 ]; then
  echo "Test completed successfully - 4 hour timeout reached without crash" >> /home/pi/CoralEdgeTpu/stress_test_log.txt
elif [ $EXIT_CODE -ne 0 ]; then
  echo "Test terminated with error code: $EXIT_CODE" >> /home/pi/CoralEdgeTpu/stress_test_log.txt
else
  echo "Program exited normally" >> /home/pi/CoralEdgeTpu/stress_test_log.txt
fi

echo "Stress test finished"