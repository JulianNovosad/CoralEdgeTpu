import json
from datetime import datetime

def analyze_log_file(file_path):
    anomalies = []
    with open(file_path, 'r') as f:
        last_timestamp = None
        lines = f.readlines()
        for line_number, line in enumerate(lines, 1):
            try:
                data = json.loads(line)
                # Check timestamp order
                current_timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                if last_timestamp and current_timestamp < last_timestamp:
                    anomalies.append({
                        'anomaly_type': 'out_of_order_timestamp',
                        'line': line_number,
                        'value': f"current: {current_timestamp}, previous: {last_timestamp}"
                    })
                last_timestamp = current_timestamp

                # Check for negative durations
                if 'message' in data:
                    message = data['message'].lower()
                    if 'time' in message or 'duration' in message:
                        for word in message.split():
                            try:
                                if word.startswith('-') and word[1:].replace('.', '', 1).isdigit():
                                    value = float(word)
                                    if value < 0:
                                        anomalies.append({
                                            'anomaly_type': 'negative_duration',
                                            'line': line_number,
                                            'value': data['message']
                                        })
                            except (ValueError, IndexError):
                                continue

                # Check for ERROR or CRITICAL messages
                if 'level' in data and data['level'] in ['ERROR', 'CRITICAL']:
                    anomalies.append({
                        'anomaly_type': 'error_message',
                        'line': line_number,
                        'value': data['message']
                    })

            except json.JSONDecodeError:
                anomalies.append({
                    'anomaly_type': 'json_decode_error',
                    'line': line_number,
                    'value': line.strip()
                })
            except Exception as e:
                anomalies.append({
                    'anomaly_type': 'parsing_error',
                    'line': line_number,
                    'value': str(e)
                })

        # Check for duplicate lines
        for i in range(len(lines) - 1):
            if lines[i] == lines[i+1]:
                anomalies.append({
                    'anomaly_type': 'duplicate_line',
                    'line': i + 2,
                    'value': lines[i].strip()
                })

    if anomalies:
        for anomaly in anomalies:
            print(f"Anomaly: {anomaly['anomaly_type']} at line {anomaly['line']}: {anomaly['value']}")
    else:
        print("No anomalies found in the file.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        analyze_log_file(sys.argv[1])
    else:
        print("Please provide a file path.")
