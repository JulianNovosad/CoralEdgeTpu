#!/usr/bin/env python3

import csv

def analyze_wallclock_timing(log_file):
    """Analyze actual wall-clock timing from log entries"""
    
    timestamps = []
    events = []
    
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 5 and row[3] == 'inference_done':
                try:
                    produced_ts = int(row[0])  # produced_ts_epoch_ms (milliseconds)
                    call_ts = int(row[4])      # call_ts_epoch_ms (milliseconds)
                    timestamps.append(produced_ts)
                    events.append({'produced': produced_ts, 'call': call_ts})
                except (ValueError, IndexError):
                    continue
    
    print("Wall-Clock Timing Analysis:")
    print("=" * 50)
    print(f"Total inference events: {len(timestamps)}")
    print(f"First timestamp: {timestamps[0]} ms")
    print(f"Last timestamp: {timestamps[-1]} ms")
    print(f"Total duration: {timestamps[-1] - timestamps[0]} ms")
    
    if len(timestamps) > 1:
        # Calculate intervals between consecutive events
        intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            intervals.append(interval)
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            min_interval = min(intervals)
            max_interval = max(intervals)
            fps = 1000.0 / avg_interval if avg_interval > 0 else 0
            
            print(f"\nInterval Analysis:")
            print(f"  Average interval: {avg_interval:.2f} ms")
            print(f"  Min/Max interval: {min_interval} ms / {max_interval} ms")
            print(f"  Equivalent FPS: {fps:.2f}")
            
            # Show distribution of intervals
            intervals_sorted = sorted(intervals)
            print(f"\nInterval Distribution:")
            print(f"  10th percentile: {intervals_sorted[len(intervals_sorted)//10]} ms")
            print(f"  50th percentile: {intervals_sorted[len(intervals_sorted)//2]} ms")
            print(f"  90th percentile: {intervals_sorted[int(len(intervals_sorted)*0.9)]} ms")
            
            # Look for patterns in large intervals
            large_intervals = [i for i in intervals if i > avg_interval * 2]
            if large_intervals:
                print(f"\nLarge Intervals (> 2x average): {len(large_intervals)} events")
                print(f"  Average large interval: {sum(large_intervals)/len(large_intervals):.2f} ms")

if __name__ == "__main__":
    log_file = "/home/pi/CoralEdgeTpu/logs/tpu/InferenceEngine_2025_12_21_00_51.csv"
    analyze_wallclock_timing(log_file)