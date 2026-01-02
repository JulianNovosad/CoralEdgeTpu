import pandas as pd
import sys
import os

def validate_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    required_columns = [
        "produced_ts_epoch_ms", "call_ts_epoch_ms", "module", "event", "thread_id",
        "cam_frame_id", "cam_exposure_ms", "cam_isp_latency_ms", "cam_buffer_usage_percent",
        "tpu_inference_ms", "tpu_temp_c", "tpu_model_score", "tpu_class_id",
        "logic_target_dist_m", "logic_ballistic_drop_m", "logic_windage_m", 
        "logic_servo_x_cmd", "logic_servo_y_cmd", "logic_solution_time_ms",
        "enc_process_ms", "enc_bitrate_mbps", "enc_queue_depth",
        "sys_cpu_temp_c", "sys_cpu_usage_pct", "sys_ram_usage_pct", "sys_voltage_v"
    ]

    # 1. Check columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"FAILED: Missing columns: {missing_cols}")
        return False
    else:
        print("PASSED: All 26 columns present.")

    # 2. Check for empty fields / placeholder values
    # We expect some fields might be -1.0 in early development, but let's check for NaN
    if df.isnull().values.any():
        print("WARNING: Found NaN values in CSV.")
        # print(df.isnull().sum())
    else:
        print("PASSED: No NaN values found.")

    # 3. Check chronological order
    if not df['produced_ts_epoch_ms'].is_monotonic_increasing:
        print("FAILED: produced_ts_epoch_ms is NOT monotonic increasing.")
        # Find where it's not monotonic
        diff = df['produced_ts_epoch_ms'].diff()
        violations = diff[diff < 0]
        print(f"Found {len(violations)} synchronization violations.")
        return False
    else:
        print("PASSED: Row order is strictly chronological.")

    # 4. Check frame ID monotonicity (for the same module)
    # Since Logic logs per frame, cam_frame_id should increase
    if not df['cam_frame_id'].is_monotonic_increasing:
         print("WARNING: cam_frame_id is NOT strictly monotonic increasing (dropped frames?)")
    
    # 5. Check populated fields (not just placeholders)
    # Check if we have actual camera telemetry
    if (df['cam_exposure_ms'] != -1.0).any():
        print("PASSED: Found active camera exposure data.")
    else:
        print("WARNING: All camera exposure values are -1.0.")

    if (df['tpu_temp_c'] != -1.0).any():
        print("PASSED: Found active TPU temperature data.")
    else:
        print("WARNING: All TPU temperature values are -1.0.")

    if (df['sys_cpu_temp_c'] != 0.0).any(): # SystemMonitor defaults to 0.0 initially
        print("PASSED: Found active System CPU temp data.")
    else:
        print("WARNING: All System CPU temp values are 0.0 or default.")

    print("\nSummary: Telemetry integrity looks GOOD." if not missing_cols else "Summary: Telemetry integrity FAILED.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_telemetry.py <path_to_unified.csv>")
        sys.exit(1)
    
    validate_csv(sys.argv[1])
