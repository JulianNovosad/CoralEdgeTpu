#!/usr/bin/env python3

# Read COCO labels
with open('coco_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

print("COCO Labels Mapping (for detected classes):")
print("Class ID\tLabel")
print("--------\t-----")

# Show labels for the most relevant classes from our analysis
relevant_classes = [5, 6, 7, 8, 10, 23, 31, 35, 40, 65, 73, 81, 90]
for class_id in relevant_classes:
    if class_id <= len(labels):
        print(f"{class_id}\t\t{labels[class_id-1]}")  # COCO classes are 1-indexed in the file
    else:
        print(f"{class_id}\t\t<unknown>")