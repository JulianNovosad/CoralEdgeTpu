import csv
import sys
from datetime import datetime

def analyze_camera_log(file_path):
    anomalies = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        last_timestamp = None
        frame_ids = set()
        
        for i, row in enumerate(reader):
            line_number = i + 2 # 1-based, plus header
            
            # Create a dictionary from the row
            try:
                row_dict = dict(zip(header, row))
            except IndexError:
                anomalies.append({
                    'anomaly_type': 'malformed_row',
                    'line': line_number,
                    'value': row
                })
                continue

            # Check timestamp order
            try:
                current_timestamp = int(row_dict['produced_ts_epoch_ms'])
                if last_timestamp and current_timestamp < last_timestamp:
                    anomalies.append({
                        'anomaly_type': 'out_of_order_timestamp',
                        'line': line_number,
                        'value': f"current: {current_timestamp}, previous: {last_timestamp}"
                    })
                last_timestamp = current_timestamp
            except (ValueError, KeyError):
                anomalies.append({
                    'anomaly_type': 'invalid_timestamp',
                    'line': line_number,
                    'value': row_dict.get('produced_ts_epoch_ms', 'N/A')
                })


            # Check for negative values in numeric columns
            for col in ['camera_exposure_ms', 'camera_copy_time_ms']:
                try:
                    value = float(row_dict[col])
                    if value < 0 and value != -1: # -1 is used as a placeholder
                        anomalies.append({
                            'anomaly_type': f'negative_value_in_{col}',
                            'line': line_number,
                            'value': value
                        })
                except (ValueError, KeyError):
                    # Ignore if the column is not a valid number
                    pass

            # Check for missing or duplicated frame IDs
            try:
                frame_id = int(row_dict['camera_frame_id'])
                if (frame_id, row_dict['event']) in frame_ids:
                    anomalies.append({
                        'anomaly_type': 'duplicate_frame_id_and_event',
                        'line': line_number,
                        'value': (frame_id, row_dict['event'])
                    })
                else:
                    frame_ids.add((frame_id, row_dict['event']))
            except (ValueError, KeyError):
                anomalies.append({
                    'anomaly_type': 'invalid_frame_id',
                    'line': line_number,
                    'value': row_dict.get('camera_frame_id', 'N/A')
                })


    if anomalies:
        for anomaly in anomalies:
            print(f"Anomaly: {anomaly['anomaly_type']} at line {anomaly['line']}: {anomaly['value']}")
    else:
        print("No anomalies found in the file.")
        
    # Summary Statistics
    if frame_ids:
        # get unique frame ids
        unique_frame_ids = {x[0] for x in frame_ids}
        max_frame_id = max(unique_frame_ids)
        if len(unique_frame_ids) != max_frame_id + 1:
            print(f"Warning: Missing frame IDs. Expected {max_frame_id + 1} frames, but found {len(unique_frame_ids)}.")
        else:
            print("Frame IDs are sequential.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_camera_log(sys.argv[1])
    else:
        print("Please provide a file path.")
