#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict

def calculate_average_intervals(log_file):
    """Calculate average intervals between events in milliseconds"""
    
    # Read all timestamps for different event types
    events = defaultdict(list)
    
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        # Find the indices for the columns we need
        produced_ts_idx = header.index('produced_ts_epoch_ms')
        event_idx = header.index('event')
        
        for row in reader:
            if len(row) > max(produced_ts_idx, event_idx):
                try:
                    timestamp = int(row[produced_ts_idx])
                    event_type = row[event_idx]
                    events[event_type].append(timestamp)
                except (ValueError, IndexError):
                    continue
    
    # Calculate intervals for each event type
    results = {}
    for event_type, timestamps in events.items():
        if len(timestamps) > 1:
            # Sort timestamps
            timestamps.sort()
            
            # Calculate intervals in milliseconds
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            
            # Calculate average interval
            if intervals:
                avg_interval_ms = sum(intervals) / len(intervals)
                fps = 1000.0 / avg_interval_ms if avg_interval_ms > 0 else 0
                
                results[event_type] = {
                    'avg_interval_ms': avg_interval_ms,
                    'fps': fps,
                    'count': len(timestamps),
                    'min_interval_ms': min(intervals) if intervals else 0,
                    'max_interval_ms': max(intervals) if intervals else 0
                }
    
    return results

def print_results(results):
    """Print formatted results"""
    print("Event Timing Analysis")
    print("=" * 50)
    
    for event_type, stats in results.items():
        print(f"\n{event_type}:")
        print(f"  Count: {stats['count']} events")
        print(f"  Average interval: {stats['avg_interval_ms']:.2f} ms")
        print(f"  Min interval: {stats['min_interval_ms']:.2f} ms")
        print(f"  Max interval: {stats['max_interval_ms']:.2f} ms")
        if stats['fps'] > 0:
            print(f"  Equivalent FPS: {stats['fps']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 calculate_fps.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    try:
        results = calculate_average_intervals(log_file)
        print_results(results)
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)