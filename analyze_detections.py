#!/usr/bin/env python3

import re
from collections import defaultdict

# Read the valid detections
with open('valid_detections.txt', 'r') as f:
    lines = f.readlines()

# Parse the data
parsed_data = []
class_distances = defaultdict(list)

for line in lines:
    # Extract class ID, score, bounding box, and distance
    match = re.search(r'class=(\d+), score=([0-9.]+), box=$$(.*)$$, distance=([0-9.]+)m', line)
    if match:
        class_id, score, box_coords, distance = match.groups()
        class_id = int(class_id)
        score = float(score)
        distance = float(distance)
        
        parsed_data.append({
            'class_id': class_id,
            'score': score,
            'box_coords': box_coords,
            'distance': distance
        })
        
        # Store distance for each class
        class_distances[class_id].append(distance)

# Analyze the data
print("Detection Analysis Report")
print("========================")
print(f"Total valid detections: {len(parsed_data)}")
print()

# Calculate statistics for each class
class_stats = {}
for class_id, distances in class_distances.items():
    distances.sort()
    median_dist = distances[len(distances)//2]
    mean_dist = sum(distances) / len(distances)
    
    # Count how many times this class matched ~2.8m (within ±0.05m)
    matches_2_8m = sum(1 for d in distances if abs(d - 2.8) <= 0.05)
    
    class_stats[class_id] = {
        'count': len(distances),
        'median': median_dist,
        'mean': mean_dist,
        'matches_2_8m': matches_2_8m,
        'distances': distances
    }

# Sort classes by number of detections
sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True)

print("Class Statistics (top 15 by detection count):")
print("Class ID\tDetections\tMedian Dist\tMean Dist\tMatches 2.8m")
print("--------\t----------\t-----------\t---------\t------------")

for class_id, stats in sorted_classes[:15]:
    print(f"{class_id}\t\t{stats['count']}\t\t{stats['median']:.3f}m\t\t{stats['mean']:.3f}m\t\t{stats['matches_2_8m']}")

print()
print("Classes that best represent the actual target at 2.8m:")
print("(Sorted by number of matches to 2.8m ±0.05m)")

# Sort by number of matches to 2.8m
classes_by_matches = sorted(class_stats.items(), key=lambda x: x[1]['matches_2_8m'], reverse=True)

best_classes = []
for class_id, stats in classes_by_matches:
    if stats['matches_2_8m'] > 0:
        best_classes.append((class_id, stats['matches_2_8m'], stats['median']))
        print(f"Class {class_id}: {stats['matches_2_8m']} matches, median distance: {stats['median']:.3f}m")

print()
if best_classes:
    print("Summary:")
    print(f"The best class representing the 2.8m target is Class {best_classes[0][0]}")
    print(f"with {best_classes[0][1]} matches and median distance of {best_classes[0][2]:.3f}m")
else:
    print("No classes had detections that closely matched the 2.8m reference distance.")

# Additional analysis: Show some examples of detections near 2.8m
print()
print("Sample detections near 2.8m (±0.1m):")
print("Class\tDistance\tBounding Box")
print("-----\t--------\t------------")

count = 0
for detection in parsed_data:
    if abs(detection['distance'] - 2.8) <= 0.1:
        print(f"{detection['class_id']}\t{detection['distance']:.3f}m\t{detection['box_coords']}")
        count += 1
        if count >= 10:  # Show first 10 examples
            break

if count == 0:
    print("No detections found within 0.1m of 2.8m")