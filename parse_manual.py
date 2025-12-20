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
            print(f"Error parsing line: {line.strip()}")
            print(f"Error: {e}")
            continue

print(f"Parsed {len(parsed_data)} detections")

# Display first few parsed entries
print("\nFirst 5 parsed entries:")
for i, entry in enumerate(parsed_data[:5]):
    print(f"{i+1}. Class {entry['class_id']}, Score {entry['score']}, Box [{entry['box_coords']}], Distance {entry['distance']:.3f}m")

# Basic statistics
if parsed_data:
    from collections import defaultdict
    
    class_distances = defaultdict(list)
    for entry in parsed_data:
        class_distances[entry['class_id']].append(entry['distance'])
    
    print(f"\nFound {len(class_distances)} different classes")
    
    # Show statistics for each class
    print("\nClass Statistics:")
    print("Class\tCount\tMin Dist\tMax Dist\tMedian Dist")
    print("-----\t-----\t--------\t--------\t-----------")
    
    for class_id, distances in sorted(class_distances.items()):
        distances_sorted = sorted(distances)
        median_dist = distances_sorted[len(distances_sorted)//2]
        min_dist = min(distances_sorted)
        max_dist = max(distances_sorted)
        print(f"{class_id}\t{len(distances)}\t{min_dist:.3f}m\t\t{max_dist:.3f}m\t\t{median_dist:.3f}m")
    
    # Find classes closest to 2.8m
    print("\nClasses with median distance closest to 2.8m:")
    class_medians = []
    for class_id, distances in class_distances.items():
        distances_sorted = sorted(distances)
        median_dist = distances_sorted[len(distances_sorted)//2]
        class_medians.append((class_id, median_dist, len(distances)))
    
    # Sort by distance to 2.8m
    class_medians.sort(key=lambda x: abs(x[1] - 2.8))
    
    for class_id, median_dist, count in class_medians[:10]:
        print(f"Class {class_id}: {count} detections, median distance {median_dist:.3f}m (diff: {abs(median_dist - 2.8):.3f}m)")
    
    # Count matches to 2.8m ±0.05m
    print("\nMatches to 2.8m ±0.05m:")
    for class_id, distances in sorted(class_distances.items()):
        matches = sum(1 for d in distances if abs(d - 2.8) <= 0.05)
        if matches > 0:
            print(f"Class {class_id}: {matches} matches out of {len(distances)} detections")