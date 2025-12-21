#!/usr/bin/env python3

import csv
import sys
import os
import glob
from collections import defaultdict

def analyze_all_logs(base_path="/home/pi/CoralEdgeTpu/logs"):
    """Analyze timing data from all log modules"""
    
    results = {}
    
    # Define log directories and their primary events
    log_modules = {
        'camera': ('frame_captured_tpu', 'Camera'),
        'tpu': ('inference_done', 'TPU'),
        'logic': ('logic_cycle_done', 'Logic'),
        'system_monitor': ('sysmon_metrics', 'System Monitor')
    }
    
    for module_dir, (primary_event, module_name) in log_modules.items():
        module_path = os.path.join(base_path, module_dir)
        if not os.path.exists(module_path):
            continue
            
        # Find the most recent CSV file for this module
        csv_files = glob.glob(os.path.join(module_path, "*.csv"))
        if not csv_files:
            continue
            
        # Get the most recent file
        latest_file = max(csv_files, key=os.path.getmtime)
        
        # Analyze the file
        module_results = calculate_intervals(latest_file, primary_event)
        if module_results:
            results[module_name] = module_results
    
    return results

def calculate_intervals(log_file, primary_event_filter=None):
    """Calculate intervals for events in a log file"""
    
    events = defaultdict(list)
    
    try:
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
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None
    
    # Calculate intervals for each event type
    results = {}
    for event_type, timestamps in events.items():
        # If filtering, only process the primary event
        if primary_event_filter and event_type != primary_event_filter:
            continue
            
        if len(timestamps) > 1:
            # Sort timestamps
            timestamps.sort()
            
            # Calculate intervals in milliseconds
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            
            # Calculate statistics
            if intervals:
                avg_interval_ms = sum(intervals) / len(intervals)
                fps = 1000.0 / avg_interval_ms if avg_interval_ms > 0 else 0
                
                results[event_type] = {
                    'avg_interval_ms': avg_interval_ms,
                    'fps': fps,
                    'count': len(timestamps),
                    'min_interval_ms': min(intervals) if intervals else 0,
                    'max_interval_ms': max(intervals) if intervals else 0,
                    'total_duration_ms': timestamps[-1] - timestamps[0] if timestamps else 0
                }
    
    return results

def print_comprehensive_results(results):
    """Print formatted results for all modules"""
    print("Comprehensive System Performance Analysis")
    print("=" * 60)
    
    # Overall system FPS
    print("\nMODULE PERFORMANCE:")
    print("-" * 40)
    
    for module_name, module_results in results.items():
        for event_type, stats in module_results.items():
            print(f"\n{module_name} ({event_type}):")
            print(f"  Events processed: {stats['count']}")
            print(f"  Average interval: {stats['avg_interval_ms']:.2f} ms")
            print(f"  Min interval: {stats['min_interval_ms']:.2f} ms")
            print(f"  Max interval: {stats['max_interval_ms']:.2f} ms")
            print(f"  Equivalent rate: {stats['fps']:.2f} per second")
            print(f"  Total duration: {stats['total_duration_ms']/1000:.2f} seconds")

if __name__ == "__main__":
    try:
        results = analyze_all_logs()
        if results:
            print_comprehensive_results(results)
        else:
            print("No log data found or processed successfully")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)