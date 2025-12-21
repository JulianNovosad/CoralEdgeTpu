#!/usr/bin/env python3

import csv
import os
import glob
from collections import defaultdict

def analyze_multiple_log_files(base_path="/home/pi/CoralEdgeTpu/logs"):
    """Analyze timing across multiple log files"""
    
    # Collect data from all TPU log files
    all_timestamps = []
    
    tpu_files = glob.glob(os.path.join(base_path, "tpu", "*.csv"))
    
    for log_file in tpu_files:
        timestamps = []
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                if len(row) >= 5 and row[3] == 'inference_done':
                    try:
                        produced_ts = int(row[0])  # produced_ts_epoch_ms (milliseconds)
                        timestamps.append(produced_ts)
                    except (ValueError, IndexError):
                        continue
        
        all_timestamps.extend(timestamps)
    
    # Sort all timestamps
    all_timestamps.sort()
    
    print("Comprehensive TPU Performance Analysis:")
    print("=" * 60)
    print(f"Total inference events across all files: {len(all_timestamps)}")
    
    if len(all_timestamps) > 1:
        # Calculate intervals between consecutive events
        intervals = []
        for i in range(1, len(all_timestamps)):
            interval = all_timestamps[i] - all_timestamps[i-1]
            intervals.append(interval)
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            min_interval = min(intervals)
            max_interval = max(intervals)
            fps = 1000.0 / avg_interval if avg_interval > 0 else 0
            
            print(f"\nOverall Performance:")
            print(f"  First timestamp: {all_timestamps[0]} ms")
            print(f"  Last timestamp: {all_timestamps[-1]} ms")
            print(f"  Total duration: {(all_timestamps[-1] - all_timestamps[0])/1000:.2f} seconds")
            print(f"  Average interval: {avg_interval:.2f} ms")
            print(f"  Min/Max interval: {min_interval} ms / {max_interval} ms")
            print(f"  Equivalent FPS: {fps:.2f}")
            
            # Flag discrepancies
            target_fps = 120.0
            deviation = abs(fps - target_fps) / target_fps * 100
            print(f"\nPerformance Gap Analysis:")
            print(f"  Configured target: {target_fps} FPS")
            print(f"  Actual achieved: {fps:.2f} FPS")
            print(f"  Deviation: {deviation:.1f}%")
            
            if deviation > 10:
                print("  ⚠️  SEVERE PERFORMANCE GAP DETECTED!")
            else:
                print("  ✅ Performance within acceptable range")
            
            # Analyze interval distribution
            intervals_sorted = sorted(intervals)
            print(f"\nInterval Distribution:")
            print(f"  5th percentile: {intervals_sorted[int(len(intervals_sorted)*0.05)]} ms")
            print(f"  25th percentile: {intervals_sorted[int(len(intervals_sorted)*0.25)]} ms")
            print(f"  Median: {intervals_sorted[len(intervals_sorted)//2]} ms")
            print(f"  75th percentile: {intervals_sorted[int(len(intervals_sorted)*0.75)]} ms")
            print(f"  95th percentile: {intervals_sorted[int(len(intervals_sorted)*0.95)]} ms")
            
            # Look for patterns in large intervals
            large_intervals = [i for i in intervals if i > avg_interval * 3]
            if large_intervals:
                print(f"\nPerformance Anomalies:")
                print(f"  Large intervals (> 3x average): {len(large_intervals)} events")
                print(f"  Percentage of total: {len(large_intervals)/len(intervals)*100:.1f}%")
                if large_intervals:
                    print(f"  Average large interval: {sum(large_intervals)/len(large_intervals):.2f} ms")

if __name__ == "__main__":
    analyze_multiple_log_files()