import csv
import sys

def analyze_sysmon_log(file_path):
    anomalies = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        last_timestamp = None
        
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
            for col in ['sysmon_cpu_temp_c', 'sysmon_cpu_usage_percent', 'sysmon_mem_usage_percent']:
                try:
                    value = float(row_dict[col])
                    if value < 0:
                        anomalies.append({
                            'anomaly_type': f'negative_value_in_{col}',
                            'line': line_number,
                            'value': value
                        })
                except (ValueError, KeyError):
                    # Ignore if the column is not a valid number
                    pass


    if anomalies:
        for anomaly in anomalies:
            print(f"Anomaly: {anomaly['anomaly_type']} at line {anomaly['line']}: {anomaly['value']}")
    else:
        print("No anomalies found in the file.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_sysmon_log(sys.argv[1])
    else:
        print("Please provide a file path.")
