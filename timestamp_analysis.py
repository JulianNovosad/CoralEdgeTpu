#!/usr/bin/env python3

import csv

def analyze_timestamp_differences(log_file):
    """Analyze timestamp differences to determine true units"""
    
    produced_timestamps = []
    call_timestamps = []
    
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 5:
                try:
                    produced_ts = int(row[0])  # produced_ts_epoch_ms
                    call_ts = int(row[4])      # call_ts_epoch_ms
                    produced_timestamps.append(produced_ts)
                    call_timestamps.append(call_ts)
                except (ValueError, IndexError):
                    continue
    
    print("Timestamp Analysis:")
    print("=" * 50)
    
    if produced_timestamps:
        print(f"Produced timestamps sample: {produced_timestamps[:5]}")
        print(f"Average produced timestamp: {sum(produced_timestamps[:10])/len(produced_timestamps[:10]):.0f}")
        
        # Calculate differences between consecutive timestamps
        if len(produced_timestamps) > 1:
            diffs = [produced_timestamps[i] - produced_timestamps[i-1] for i in range(1, min(11, len(produced_timestamps)))]
            print(f"Consecutive differences (first 10): {diffs}")
            print(f"Average difference: {sum(diffs)/len(diffs):.2f}")
            
            # Determine unit based on magnitude
            avg_diff = sum(diffs)/len(diffs)
            if avg_diff < 100:  # Less than 100ms
                print("Timestamps appear to be in MILLISECONDS")
            elif avg_diff < 100000:  # Less than 100s
                print("Timestamps appear to be in MICROSECONDS")
            else:
                print("Timestamps appear to be in NANOSECONDS")
    
    if call_timestamps:
        print(f"\nCall timestamps sample: {call_timestamps[:5]}")
        print(f"Average call timestamp: {sum(call_timestamps[:10])/len(call_timestamps[:10]):.0f}")
        
        # Calculate differences between consecutive call timestamps
        if len(call_timestamps) > 1:
            call_diffs = [call_timestamps[i] - call_timestamps[i-1] for i in range(1, min(11, len(call_timestamps)))]
            print(f"Call consecutive differences (first 10): {call_diffs}")
            print(f"Average call difference: {sum(call_diffs)/len(call_diffs):.2f}")

if __name__ == "__main__":
    log_file = "/home/pi/CoralEdgeTpu/logs/tpu/InferenceEngine_2025_12_21_00_51.csv"
    analyze_timestamp_differences(log_file)