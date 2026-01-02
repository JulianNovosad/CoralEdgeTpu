import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/home/pi/CoralEdgeTpu/logs/session_20260101_215747/unified.csv')

# Display basic info about the dataset
print('Total rows:', len(df))
print('Columns:', list(df.columns))
print('\nUnique modules:', df['module'].unique())

# Define modules and their latency columns
modules_latency = {
    'CameraCapture': 'cam_isp_latency_ms',
    'ImageProcessor_TPU': 'image_proc_ms',
    'ImageProcessor_Viz': 'image_proc_ms', 
    'InferenceEngine': 'tpu_inference_ms',
    'LogicModule': 'logic_solution_time_ms',
    'Encoder': 'enc_process_ms'
}

# Analyze each module
results = []
for module, latency_col in modules_latency.items():
    module_data = df[df['module'] == module]
    if len(module_data) > 0:
        total_frames = len(module_data)
        valid_latency_count = module_data[latency_col].notna().sum()
        latency_percentage = (valid_latency_count / total_frames) * 100
        
        # Get frame IDs with missing latency if any
        missing_frames = module_data[module_data[latency_col].isna()]
        if len(missing_frames) > 0:
            missing_frame_ids = missing_frames['cam_frame_id'].dropna().unique()
            if len(missing_frame_ids) > 0:
                missing_range = f'{int(min(missing_frame_ids))}-{int(max(missing_frame_ids))}'
            else:
                missing_range = 'None'
        else:
            missing_range = 'None'
        
        results.append({
            'Module': module,
            'Total Frames': total_frames,
            'Latency Recorded': valid_latency_count,
            '% Coverage': round(latency_percentage, 2),
            'Missing Frame IDs Range': missing_range
        })

# Print the results table
print('\nLatency Coverage Analysis:')
print('Module | Total Frames | Latency Recorded | % Coverage | Missing Frame IDs Range')
print('-' * 85)
for result in results:
    print(f"{result['Module']} | {result['Total Frames']} | {result['Latency Recorded']} | {result['% Coverage']}% | {result['Missing Frame IDs Range']}")
    
# Identify modules needing attention
print('\nModules needing attention (< 100% coverage):')
for result in results:
    if result['% Coverage'] < 100:
        print(f"- {result['Module']}: {result['Missing Frame IDs Range']}")
