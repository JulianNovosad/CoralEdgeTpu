import csv
import sys

def analyze_encoder_log(file_path):
    anomalies = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        last_timestamp = None
        last_encoded_frames = -1
        
        for i, row in enumerate(reader):
            line_number = i + 2 # 1-based, plus header
            
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
            try:
                value = float(row_dict['encoder_encode_ms'])
                if value < 0:
                    anomalies.append({
                        'anomaly_type': 'negative_value_in_encoder_encode_ms',
                        'line': line_number,
                        'value': value
                    })
            except (ValueError, KeyError):
                # Ignore if the column is not a valid number
                pass

            # Check for missing or duplicated total_encoded_frames
            try:
                total_frames = int(row_dict['encoder_total_encoded_frames'])
                if total_frames != -1: # Ignore placeholder value
                    if last_encoded_frames != -1 and total_frames <= last_encoded_frames:
                         anomalies.append({
                            'anomaly_type': 'non_increasing_total_encoded_frames',
                            'line': line_number,
                            'value': f"current: {total_frames}, previous: {last_encoded_frames}"
                        })
                    last_encoded_frames = total_frames
            except (ValueError, KeyError):
                pass


    if anomalies:
        for anomaly in anomalies:
            print(f"Anomaly: {anomaly['anomaly_type']} at line {anomaly['line']}: {anomaly['value']}")
    else:
        print("No anomalies found in the file.")
        
    # Summary Statistics
    # placeholder for summary stats

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_encoder_log(sys.argv[1])
    else:
        print("Please provide a file path.")
