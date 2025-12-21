#!/usr/bin/env python3

import csv
from collections import defaultdict

def detailed_timestamp_analysis(log_file):
    """Detailed analysis of timestamp patterns"""
    
    # Group by thread_id to analyze per-thread behavior
    thread_data = defaultdict(list)
    
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 5:
                try:
                    produced_ts = int(row[0])  # produced_ts_epoch_ms
                    thread_id = row[2]         # thread_id
                    event_type = row[3]        # event
                    call_ts = int(row[4])      # call_ts_epoch_ms
                    
                    thread_data[thread_id].append({
                        'produced_ts': produced_ts,
                        'call_ts': call_ts,
                        'event_type': event_type
                    })
                except (ValueError, IndexError):
                    continue
    
    print("Detailed Timestamp Analysis by Thread:")
    print("=" * 60)
    
    for thread_id, events in thread_data.items():
        if len(events) < 2:
            continue
            
        print(f"\nThread ID: {thread_id} (Events: {len(events)})")
        
        # Sort by produced timestamp
        events.sort(key=lambda x: x['produced_ts'])
        
        # Calculate intervals for inference_done events only
        inference_events = [e for e in events if e['event_type'] == 'inference_done']
        if len(inference_events) > 1:
            intervals = []
            for i in range(1, len(inference_events)):
                interval = inference_events[i]['produced_ts'] - inference_events[i-1]['produced_ts']
                intervals.append(interval)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                min_interval = min(intervals)
                max_interval = max(intervals)
                
                # Convert to proper units for display
                avg_ms = avg_interval / 1000.0
                min_ms = min_interval / 1000.0
                max_ms = max_interval / 1000.0
                
                fps = 1000.0 / avg_ms if avg_ms > 0 else 0
                
                print(f"  Inference events: {len(inference_events)}")
                print(f"  Avg interval: {avg_interval:.0f} μs ({avg_ms:.2f} ms)")
                print(f"  Min/Max interval: {min_interval} μs / {max_interval} μs")
                print(f"  Equivalent FPS: {fps:.2f}")
                
                # Show first few intervals
                print(f"  First intervals (μs): {intervals[:5]}")

if __name__ == "__main__":
    log_file = "/home/pi/CoralEdgeTpu/logs/tpu/InferenceEngine_2025_12_21_00_51.csv"
    detailed_timestamp_analysis(log_file)