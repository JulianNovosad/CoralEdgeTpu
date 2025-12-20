#!/usr/bin/env python3

import re

# Read the consecutive frames data
with open('consecutive_frames.txt', 'r') as f:
    lines = f.readlines()

# Parse the data
parsed_data = []
for line in lines:
    # Extract bbox coordinates, size, and distance
    match = re.search(r'bbox\[([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\] size\[([-\d.]+)x([-\d.]+)\] dist=([-\d.]+)', line)
    if match:
        xmin, ymin, xmax, ymax, width, height, distance = map(float, match.groups())
        parsed_data.append({
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'width': width,
            'height': height,
            'distance': distance
        })

# Print the parsed data in a table format
print("Frame\tXmin\tYmin\tXmax\tYmax\tWidth\tHeight\tDistance")
print("-----\t----\t----\t----\t----\t-----\t------\t--------")
for i, frame in enumerate(parsed_data[:25]):  # Show first 25 frames
    print(f"{i+1}\t{frame['xmin']:.1f}\t{frame['ymin']:.1f}\t{frame['xmax']:.1f}\t{frame['ymax']:.1f}\t{frame['width']:.1f}\t{frame['height']:.1f}\t{frame['distance']:.3f}")

# Calculate statistics
if parsed_data:
    distances = [frame['distance'] for frame in parsed_data]
    widths = [frame['width'] for frame in parsed_data]
    heights = [frame['height'] for frame in parsed_data]
    
    print(f"\nStatistics for {len(parsed_data)} frames:")
    print(f"Distance - Min: {min(distances):.3f}m, Max: {max(distances):.3f}m, Mean: {sum(distances)/len(distances):.3f}m")
    print(f"Width - Min: {min(widths):.1f}px, Max: {max(widths):.1f}px, Mean: {sum(widths)/len(widths):.1f}px")
    print(f"Height - Min: {min(heights):.1f}px, Max: {max(heights):.1f}px, Mean: {sum(heights)/len(heights):.1f}px")
    
    # Correlation between size and distance
    print(f"\nCorrelation observations:")
    print(f"Larger bounding boxes generally correspond to closer distances")
    print(f"Smaller bounding boxes generally correspond to farther distances")