#!/usr/bin/env python3

# Simple manual parsing
with open('valid_detections.txt', 'r') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Manual parsing
parsed_data = []
for line in lines:
    if 'DETECTION_DISTANCE:' in line and 'class=' in line:
        try:
            # Extract class
            class_start = line.find('class=') + 6
            class_end = line.find(',', class_start)
            class_id = int(line[class_start:class_end])
            
            # Extract score
            score_start = line.find('score=') + 6
            score_end = line.find(',', score_start)
            score = float(line[score_start:score_end])
            
            # Extract box coordinates
            box_start = line.find('box=[') + 5
            box_end = line.find(']', box_start)
            box_coords = line[box_start:box_end]
            
            # Extract distance
            dist_start = line.find('distance=') + 9
            dist_end = line.find('m', dist_start)
            distance_str = line[dist_start:dist_end]
            
            # Skip if distance is "too_small"
            if distance_str == "too_small":
                continue
                
            distance = float(distance_str)
            
            parsed_data.append({
                'class_id': class_id,
                'score': score,
                'box_coords': box_coords,
                'distance': distance
            })
        except Exception as e:
            continue

print(f"Parsed {len(parsed_data)} detections")

if not parsed_data:
    print("No valid detections parsed. Exiting.")
    exit(0)

# Basic statistics
from collections import defaultdict

class_distances = defaultdict(list)
for entry in parsed_data:
    class_distances[entry['class_id']].append(entry['distance'])

print(f"\nFound {len(class_distances)} different classes")

# Detailed analysis for the report
print("\n" + "="*60)
print("DETECTION ANALYSIS REPORT")
print("="*60)

print(f"\nTotal valid detections analyzed: {len(parsed_data)}")

# Show statistics for each class
print("\nClass Statistics (all classes):")
print("Class ID\tDetections\tMedian Dist\tMean Dist\tStd Dev\t\tMin Dist\tMax Dist")
print("--------\t----------\t-----------\t---------\t-------\t\t--------\t--------")

class_stats = {}
for class_id, distances in sorted(class_distances.items()):
    distances_sorted = sorted(distances)
    median_dist = distances_sorted[len(distances_sorted)//2]
    mean_dist = sum(distances_sorted) / len(distances_sorted)
    std_dev = (sum((d - mean_dist)**2 for d in distances_sorted) / len(distances_sorted))**0.5
    min_dist = min(distances_sorted)
    max_dist = max(distances_sorted)
    
    class_stats[class_id] = {
        'count': len(distances),
        'median': median_dist,
        'mean': mean_dist,
        'std_dev': std_dev,
        'min': min_dist,
        'max': max_dist,
        'distances': distances_sorted
    }
    
    print(f"{class_id}\t\t{len(distances)}\t\t{median_dist:.3f}m\t\t{mean_dist:.3f}m\t\t{std_dev:.3f}m\t\t{min_dist:.3f}m\t\t{max_dist:.3f}m")

# Identify classes that match 2.8m reference distance
reference_distance = 2.8
tolerance = 0.05  # ±0.05m

print(f"\nClasses matching reference distance of {reference_distance}m (±{tolerance}m):")
print("Class ID\tMatches\t\tPercentage\tMedian Dist")
print("--------\t-------\t\t----------\t-----------")

matching_classes = []
for class_id, stats in class_stats.items():
    matches = sum(1 for d in stats['distances'] if abs(d - reference_distance) <= tolerance)
    if matches > 0:
        percentage = (matches / stats['count']) * 100
        matching_classes.append((class_id, matches, percentage, stats['median']))
        print(f"{class_id}\t\t{matches}\t\t{percentage:.1f}%\t\t{stats['median']:.3f}m")

# Sort matching classes by number of matches
matching_classes.sort(key=lambda x: x[1], reverse=True)

print(f"\nBest classes representing the actual target at {reference_distance}m:")
if matching_classes:
    print("Class ID\tMatches\t\tPercentage\tMedian Distance")
    print("--------\t-------\t\t----------\t--------------")
    for class_id, matches, percentage, median_dist in matching_classes:
        print(f"Class {class_id}\t\t{matches}\t\t{percentage:.1f}%\t\t{median_dist:.3f}m")
    
    best_class = matching_classes[0]
    print(f"\nSUMMARY:")
    print(f"The best class representing the {reference_distance}m target is Class {best_class[0]}")
    print(f"with {best_class[1]} matches ({best_class[2]:.1f}% of its detections)")
    print(f"and a median distance of {best_class[3]:.3f}m")
else:
    print(f"No classes had detections that closely matched the {reference_distance}m reference distance.")

# Additional analysis: Show some examples of detections near 2.8m
print(f"\nSample detections near {reference_distance}m (±0.1m):")
print("Class\tScore\t\tDistance\tBounding Box")
print("-----\t-----\t\t--------\t------------")

count = 0
for entry in parsed_data:
    if abs(entry['distance'] - reference_distance) <= 0.1:
        print(f"{entry['class_id']}\t{entry['score']:.3f}\t\t{entry['distance']:.3f}m\t\t[{entry['box_coords']}]")
        count += 1
        if count >= 10:  # Show first 10 examples
            break

if count == 0:
    print(f"No detections found within 0.1m of {reference_distance}m")

# Show top 5 classes by detection count
print(f"\nTop 5 classes by detection count:")
print("Class ID\tDetections\tMedian Dist")
print("--------\t----------\t-----------")
sorted_by_count = sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True)
for class_id, stats in sorted_by_count[:5]:
    print(f"{class_id}\t\t{stats['count']}\t\t{stats['median']:.3f}m")