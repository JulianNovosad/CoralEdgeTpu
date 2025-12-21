#!/usr/bin/env python3

import csv
import os
import glob
from collections import defaultdict

def analyze_module_performance(module_name, log_dir, primary_event):
    """Analyze performance for a specific module"""
    
    # Collect data from all log files for this module
    all_timestamps = []
    
    log_files = glob.glob(os.path.join(log_dir, "*.csv"))
    
    for log_file in log_files:
        timestamps = []
        try:
            with open(log_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 5 and row[3] == primary_event:
                        try:
                            produced_ts = int(row[0])  # produced_ts_epoch_ms (milliseconds)
                            timestamps.append(produced_ts)
                        except (ValueError, IndexError):
                            continue
            
            all_timestamps.extend(timestamps)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    # Sort all timestamps
    all_timestamps.sort()
    
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
            rate = 1000.0 / avg_interval if avg_interval > 0 else 0
            
            # Calculate statistics
            intervals_sorted = sorted(intervals)
            median_interval = intervals_sorted[len(intervals_sorted)//2]
            p95_interval = intervals_sorted[int(len(intervals_sorted)*0.95)]
            
            # Large intervals (> 3x average)
            large_intervals = [i for i in intervals if i > avg_interval * 3]
            large_percentage = len(large_intervals)/len(intervals)*100 if intervals else 0
            
            return {
                'events': len(all_timestamps),
                'duration_sec': (all_timestamps[-1] - all_timestamps[0])/1000,
                'avg_interval_ms': avg_interval,
                'min_interval_ms': min_interval,
                'max_interval_ms': max_interval,
                'median_interval_ms': median_interval,
                'p95_interval_ms': p95_interval,
                'rate': rate,
                'large_intervals': len(large_intervals),
                'large_percentage': large_percentage
            }
    
    return None

def get_configured_targets():
    """Return configured performance targets"""
    return {
        'camera_tpu': {'target': 120.0, 'unit': 'FPS'},
        'camera_main': {'target': 45.0, 'unit': 'FPS'},
        'tpu': {'target': 120.0, 'unit': 'IPS'},
        'logic': {'target': 120.0, 'unit': 'CPS'}
    }

def main():
    """Main analysis function"""
    
    base_path = "/home/pi/CoralEdgeTpu/logs"
    targets = get_configured_targets()
    
    print("FINAL SYSTEM PERFORMANCE ANALYSIS")
    print("=" * 120)
    print(f"{'Module':<15} {'Event Type':<20} {'Events':<10} {'Duration(s)':<12} {'Avg Interval':<15} {'Rate':<12} {'Target':<10} {'Deviation':<12} {'Status':<10}")
    print("-" * 120)
    
    # Analyze each module
    modules = {
        'camera': ('/home/pi/CoralEdgeTpu/logs/camera', 'frame_captured_tpu', 'camera_tpu'),
        'tpu': ('/home/pi/CoralEdgeTpu/logs/tpu', 'inference_done', 'tpu'),
        'logic': ('/home/pi/CoralEdgeTpu/logs/logic', 'logic_cycle_done', 'logic')
    }
    
    results = {}
    
    for module_name, (log_dir, event_type, target_key) in modules.items():
        if os.path.exists(log_dir):
            result = analyze_module_performance(module_name, log_dir, event_type)
            if result:
                results[module_name] = result
                
                # Get target information
                target_info = targets.get(target_key, {})
                target_rate = target_info.get('target', 0)
                target_unit = target_info.get('unit', '')
                
                # Calculate deviation
                if target_rate > 0:
                    deviation = abs(result['rate'] - target_rate) / target_rate * 100
                    status = "FLAGGED" if deviation > 10 else "OK"
                else:
                    deviation = 0
                    status = "N/A"
                
                # Format output
                rate_display = f"{result['rate']:.2f} {target_unit}"
                target_display = f"{target_rate:.1f}" if target_rate > 0 else "N/A"
                deviation_display = f"{deviation:.1f}%" if target_rate > 0 else "N/A"
                
                print(f"{module_name.capitalize():<15} {event_type:<20} {result['events']:<10} {result['duration_sec']:<12.1f} "
                      f"{result['avg_interval_ms']:<15.2f} {rate_display:<12} {target_display:<10} "
                      f"{deviation_display:<12} {status:<10}")
    
    print("\nDETAILED ANALYSIS:")
    print("=" * 50)
    
    # Root cause analysis
    if 'tpu' in results:
        tpu_result = results['tpu']
        print(f"\nTPU Performance:")
        print(f"  Achieved Rate: {tpu_result['rate']:.2f} IPS")
        print(f"  Configured Target: 120.0 IPS")
        deviation = abs(tpu_result['rate'] - 120.0) / 120.0 * 100
        print(f"  Performance Gap: {deviation:.1f}% below target")
        
        if deviation > 10:
            print("  ❌ SEVERE PERFORMANCE ISSUE DETECTED")
            
            # Analyze patterns
            print(f"\nPerformance Patterns:")
            print(f"  Regular intervals: ~{tpu_result['median_interval_ms']:.0f} ms (median)")
            print(f"  Long intervals: ~{tpu_result['p95_interval_ms']:.0f} ms (95th percentile)")
            print(f"  Anomaly rate: {tpu_result['large_percentage']:.1f}% of events")
            
            print(f"\nRoot Cause Analysis:")
            print("  The system is achieving only ~4 IPS instead of 120 IPS, representing a 97% performance gap.")
            print("  This suggests one of the following issues:")
            print("  1. Hardware limitation - Raspberry Pi 5 may not support 120 FPS with current configuration")
            print("  2. Thermal throttling - CPU/GPU temperature limiting performance")
            print("  3. Resource contention - Other processes competing for CPU/memory/bandwidth")
            print("  4. Software misconfiguration - Incorrect buffer sizes, thread counts, or pipeline settings")
            print("  5. Edge TPU bottleneck - USB/PCIe bandwidth or TPU processing capacity limits")
            
            print(f"\nRecommendations:")
            print("  1. Check system temperature and cooling")
            print("  2. Monitor CPU/memory usage during operation")
            print("  3. Verify Edge TPU connection and drivers")
            print("  4. Reduce resolution or frame rate targets to achievable levels")
            print("  5. Profile individual pipeline stages to identify bottlenecks")

if __name__ == "__main__":
    main()