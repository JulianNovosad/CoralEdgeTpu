import sys
import re

def fix_json_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path + '.fixed', 'w') as f:
        for line in lines:
            if 'Telemetry (simulated): Sending impact point data' in line:
                # This is a bit of a hack, but we know the structure of the broken lines.
                # The "message" field contains a json-like string that is not properly escaped.
                # We will use regex to capture the parts of the line and reconstruct it.
                match = re.match(r'(.*"message":")(.*)("\})', line)
                if match:
                    try:
                        pre, message, post = match.groups()
                        # The message itself contains an unescaped json.
                        # let's escape the json part of the message
                        message = message.replace('"', '\\"')
                        f.write(f'{pre}{message}{post}\n')
                    except Exception:
                        f.write(line)

                else:
                    f.write(line)
            else:
                f.write(line)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fix_json_log(sys.argv[1])
    else:
        print("Please provide a file path.")