#!/usr/bin/env python3

# Read the actual labelmap.pbtxt file
with open('labelmap.pbtxt', 'r') as f:
    content = f.read()

# Parse the label mappings
import re
labels = {}

# Extract item blocks
items = re.findall(r'item\s*{([^}]+)}', content, re.DOTALL)
for item in items:
    # Extract id and display_name
    id_match = re.search(r'id:\s*(\d+)', item)
    name_match = re.search(r'display_name:\s*"([^"]+)"', item)
    
    if id_match and name_match:
        class_id = int(id_match.group(1))
        display_name = name_match.group(1)
        labels[class_id] = display_name

print("Actual Label Mapping (from labelmap.pbtxt):")
print("Class ID\tDisplay Name")
print("--------\t------------")

# Show labels for the most relevant classes from our analysis
relevant_classes = [5, 6, 7, 8, 10, 23, 31, 35, 40, 65, 73, 81, 90]
for class_id in relevant_classes:
    if class_id in labels:
        print(f"{class_id}\t\t{labels[class_id]}")
    else:
        print(f"{class_id}\t\t<not in labelmap>")

# Also show the classes we actually detected
detected_classes = [5, 6, 7, 8, 10, 23, 31, 35, 40, 73, 81, 90]
print("\nDetected Classes:")
print("Class ID\tDisplay Name")
print("--------\t------------")
for class_id in detected_classes:
    if class_id in labels:
        print(f"{class_id}\t\t{labels[class_id]}")
    else:
        print(f"{class_id}\t\t<not in labelmap>")