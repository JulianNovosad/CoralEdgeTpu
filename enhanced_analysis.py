#!/usr/bin/env python3

import csv
import sys
import os
import glob
import math
from collections import defaultdict

def detect_timestamp_unit(timestamps):
    """Detect the unit of timestamps based on magnitude"""
    if not timestamps:
        return "ms"  # default
    
    # Take a sample of timestamps to determine unit
    sample = timestamps[:min(10, len(timestamps))]
    avg_timestamp = sum(sample) / len(sample)
    
    # For epoch timestamps around 2025:
    # - Milliseconds: ~1.7 * 10^12
    # - Microseconds: ~1.7 * 10^15
    # - Nanoseconds: ~1.7 * 10^18
    
    # Check if it looks like milliseconds since epoch (typical for our logs)
    if 1e12 <= avg_timestamp <= 1e13:
        return "ms"
    # If timestamp is extremely large (> 1e15), it might be nanoseconds
    elif avg_timestamp > 1e15:
        return "ns"
    # If timestamp is very large (> 1e12), it's likely microseconds
    elif avg_timestamp > 1e12:
        return "μs"
    # Otherwise, it's likely milliseconds
    else:
        return "ms"

def convert_to_milliseconds(timestamps, unit):
    """Convert timestamps to milliseconds based on detected unit"""
    if unit == "ns":
        return [ts / 1000000.0 for ts in timestamps]
    elif unit == "μs":
        return [ts / 1000.0 for ts in timestamps]
    else:  # ms
        return timestamps

def calculate_statistics(values):
    """Calculate mean, min, max, and standard deviation"""
    if not values:
        return None
    
    mean = sum(values) / len(values)
    minimum = min(values)
    maximum = max(values)
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = math.sqrt(variance)
    
    return {
        'mean': mean,
        'min': minimum,
        'max': maximum,
        'std_dev': std_dev
    }

def analyze_log_file(log_file):
    """Analyze a single log file and return event timing statistics"""
    
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
    
    # Process each event type
    results = {}
    for event_type, timestamps in events.items():
        if len(timestamps) < 2:
            continue
            
        # Detect timestamp unit
        unit = detect_timestamp_unit(timestamps)
        
        # Convert to milliseconds
        timestamps_ms = convert_to_milliseconds(timestamps, unit)
        
        # Sort timestamps
        timestamps_ms.sort()
        
        # Calculate intervals in milliseconds
        intervals = [timestamps_ms[i] - timestamps_ms[i-1] for i in range(1, len(timestamps_ms))]
        
        # Calculate statistics
        stats = calculate_statistics(intervals)
        if stats:
            # Calculate rate (per second)
            rate_per_second = 1000.0 / stats['mean'] if stats['mean'] > 0 else 0
            
            results[event_type] = {
                'intervals': stats,
                'rate_per_second': rate_per_second,
                'count': len(timestamps),
                'unit_detected': unit,
                'total_duration_ms': timestamps_ms[-1] - timestamps_ms[0] if timestamps_ms else 0
            }
    
    return results

def get_configured_targets():
    """Return configured targets for comparison"""
    return {
        'frame_captured_tpu': {'target_rate': 120.0, 'type': 'FPS'},
        'frame_captured_main': {'target_rate': 45.0, 'type': 'FPS'},
        'inference_done': {'target_rate': 120.0, 'type': 'IPS'},
        'logic_cycle_done': {'target_rate': 120.0, 'type': 'CPS'},
        'sysmon_metrics': {'target_rate': 0.2, 'type': 'updates/sec'}  # Every 5 seconds
    }

def analyze_all_modules(base_path="/home/pi/CoralEdgeTpu/logs"):
    """Analyze all log modules and return comprehensive results"""
    
    results = {}
    targets = get_configured_targets()
    
    # Define log directories
    log_dirs = ['camera', 'tpu', 'logic', 'system_monitor']
    
    for module_dir in log_dirs:
        module_path = os.path.join(base_path, module_dir)
        if not os.path.exists(module_path):
            continue
            
        # Find all CSV files for this module
        csv_files = glob.glob(os.path.join(module_path, "*.csv"))
        if not csv_files:
            continue
        
        # Process each file and accumulate results
        module_results = {}
        for csv_file in csv_files:
            file_results = analyze_log_file(csv_file)
            if file_results:
                # Merge results
                for event_type, stats in file_results.items():
                    if event_type not in module_results:
                        module_results[event_type] = []
                    module_results[event_type].append(stats)
        
        # Average results across files
        if module_results:
            averaged_results = {}
            for event_type, file_stats_list in module_results.items():
                # Average the statistics
                avg_intervals = {
                    'mean': sum(s['intervals']['mean'] for s in file_stats_list) / len(file_stats_list),
                    'min': min(s['intervals']['min'] for s in file_stats_list),
                    'max': max(s['intervals']['max'] for s in file_stats_list),
                    'std_dev': sum(s['intervals']['std_dev'] for s in file_stats_list) / len(file_stats_list)
                }
                
                avg_rate = sum(s['rate_per_second'] for s in file_stats_list) / len(file_stats_list)
                avg_count = sum(s['count'] for s in file_stats_list) / len(file_stats_list)
                
                averaged_results[event_type] = {
                    'intervals': avg_intervals,
                    'rate_per_second': avg_rate,
                    'count': round(avg_count),
                    'unit_detected': file_stats_list[0]['unit_detected'],
                    'total_duration_ms': sum(s['total_duration_ms'] for s in file_stats_list) / len(file_stats_list)
                }
            
            results[module_dir] = averaged_results
    
    return results, targets

def flag_discrepancies(measured_rate, target_rate, threshold=0.10):
    """Flag discrepancies exceeding threshold"""
    if target_rate <= 0:
        return False, 0
    
    deviation = abs(measured_rate - target_rate) / target_rate
    return deviation > threshold, deviation * 100

def print_tabular_summary(results, targets):
    """Print results in tabular format"""
    print("SYSTEM PERFORMANCE ANALYSIS")
    print("=" * 120)
    print(f"{'Module':<15} {'Event Type':<25} {'Avg Interval (ms)':<20} {'Min/Max (ms)':<20} {'Rate':<12} {'Target':<10} {'Deviation':<12} {'Status':<10}")
    print("-" * 120)
    
    module_names = {
        'camera': 'Camera',
        'tpu': 'TPU',
        'logic': 'Logic',
        'system_monitor': 'SysMon'
    }
    
    for module_dir, module_results in results.items():
        module_display_name = module_names.get(module_dir, module_dir.capitalize())
        
        for event_type, stats in module_results.items():
            avg_interval = stats['intervals']['mean']
            min_interval = stats['intervals']['min']
            max_interval = stats['intervals']['max']
            rate = stats['rate_per_second']
            
            # Get target rate
            target_info = targets.get(event_type, {})
            target_rate = target_info.get('target_rate', 0)
            target_type = target_info.get('type', '')
            
            # Format rate display
            rate_display = f"{rate:.2f} {target_type}"
            target_display = f"{target_rate:.1f}" if target_rate > 0 else "N/A"
            
            # Check for discrepancies
            is_flagged, deviation_pct = flag_discrepancies(rate, target_rate)
            deviation_display = f"{deviation_pct:.1f}%" if target_rate > 0 else "N/A"
            status = "FLAGGED" if is_flagged else "OK"
            
            print(f"{module_display_name:<15} {event_type:<25} {avg_interval:<20.2f} {f'{min_interval:.1f}/{max_interval:.1f}':<20} {rate_display:<12} {target_display:<10} {deviation_display:<12} {status:<10}")
    
    print("\nANALYSIS SUMMARY:")
    print("-" * 50)
    
    # Check TPU stream specifically
    tpu_results = results.get('camera', {}).get('frame_captured_tpu', None)
    if tpu_results:
        tpu_rate = tpu_results['rate_per_second']
        print(f"TPU Stream Achieved Rate: {tpu_rate:.2f} FPS")
        print(f"TPU Stream Configured Target: 120.0 FPS")
        _, deviation_pct = flag_discrepancies(tpu_rate, 120.0)
        print(f"TPU Stream Deviation: {deviation_pct:.1f}%")
        
        if deviation_pct > 10:
            print("⚠️  TPU STREAM IS SIGNIFICANTLY BELOW TARGET!")
        else:
            print("✅ TPU stream is within acceptable range")
    
    # Timestamp unit verification
    print("\nTIMESTAMP UNIT DETECTION:")
    for module_dir, module_results in results.items():
        for event_type, stats in module_results.items():
            unit = stats['unit_detected']
            print(f"  {module_dir}/{event_type}: {unit}")

if __name__ == "__main__":
    try:
        results, targets = analyze_all_modules()
        if results:
            print_tabular_summary(results, targets)
        else:
            print("No log data found or processed successfully")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)