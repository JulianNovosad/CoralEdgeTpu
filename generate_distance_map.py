#!/usr/bin/env python3
"""
Automatically generate a per-class distance map from detector outputs.
"""

import json
import csv
from collections import defaultdict

def parse_labelmap(labelmap_file):
    """Parse the labelmap.pbtxt file to get class ID to display name mapping."""
    with open(labelmap_file, 'r') as f:
        content = f.read()
    
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
    
    return labels

def parse_detections(detections_file):
    """Parse detection data from the valid detections file."""
    with open(detections_file, 'r') as f:
        lines = f.readlines()
    
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
                    'distance': distance
                })
            except Exception as e:
                continue
    
    return parsed_data

def aggregate_statistics(parsed_data):
    """Aggregate statistics per class."""
    class_distances = defaultdict(list)
    
    # Group distances by class
    for entry in parsed_data:
        class_distances[entry['class_id']].append(entry['distance'])
    
    # Calculate statistics for each class
    class_stats = {}
    for class_id, distances in class_distances.items():
        if not distances:
            continue
            
        distances_sorted = sorted(distances)
        median_dist = distances_sorted[len(distances_sorted)//2]
        mean_dist = sum(distances_sorted) / len(distances_sorted)
        min_dist = min(distances_sorted)
        max_dist = max(distances_sorted)
        
        class_stats[class_id] = {
            'count': len(distances),
            'median': median_dist,
            'mean': mean_dist,
            'min': min_dist,
            'max': max_dist,
            'distances': distances_sorted
        }
    
    return class_stats

def find_best_matches(class_stats, reference_distance=2.8, tolerance=0.05):
    """Find classes that best match the reference distance."""
    matching_classes = []
    
    for class_id, stats in class_stats.items():
        matches = sum(1 for d in stats['distances'] if abs(d - reference_distance) <= tolerance)
        if matches > 0:
            percentage = (matches / stats['count']) * 100
            matching_classes.append((class_id, matches, percentage, stats['median'], stats['count']))
    
    # Sort by number of matches (descending)
    matching_classes.sort(key=lambda x: x[1], reverse=True)
    
    return matching_classes

def generate_distance_map(class_stats, labels=None, matching_classes=None):
    """Generate per-class distance map."""
    distance_map = {}
    
    # Create a lookup dictionary for matching classes
    matching_lookup = {}
    if matching_classes:
        for match in matching_classes:
            class_id, matches, percentage, median_dist, count = match
            matching_lookup[class_id] = {
                'matches': matches,
                'percentage': round(percentage, 1),
                'match_count': count
            }
    
    for class_id, stats in class_stats.items():
        display_name = labels.get(class_id, f"<not mapped>") if labels else f"Class {class_id}"
        
        distance_map[class_id] = {
            'display_name': display_name,
            'median_distance': round(stats['median'], 3),
            'detection_count': stats['count'],
            'mean_distance': round(stats['mean'], 3),
            'min_distance': round(stats['min'], 3),
            'max_distance': round(stats['max'], 3)
        }
        
        # Add matching information if available
        if class_id in matching_lookup:
            distance_map[class_id]['reference_matches'] = matching_lookup[class_id]
    
    return distance_map

def save_to_json(data, filename):
    """Save data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def save_to_csv(data, filename, reference_distance=2.8, tolerance=0.05):
    """Save data to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class ID', 'Display Name', 'Detection Count', 'Median Distance (m)', 
                        'Mean Distance (m)', 'Min Distance (m)', 'Max Distance (m)'])
        
        # Sort by median distance
        sorted_data = sorted(data.items(), key=lambda x: x[1]['median_distance'])
        
        for class_id, stats in sorted_data:
            writer.writerow([
                class_id,
                stats['display_name'],
                stats['detection_count'],
                stats['median_distance'],
                stats['mean_distance'],
                stats['min_distance'],
                stats['max_distance']
            ])

def main():
    """Main function to generate per-class distance map."""
    print("Generating per-class distance map...")
    
    # Step 1: Parse labelmap
    try:
        labels = parse_labelmap('labelmap.pbtxt')
        print(f"Parsed {len(labels)} labels from labelmap.pbtxt")
    except FileNotFoundError:
        print("labelmap.pbtxt not found, using class IDs only")
        labels = {}
    
    # Step 2: Parse detections
    try:
        parsed_data = parse_detections('valid_detections.txt')
        print(f"Parsed {len(parsed_data)} detections from valid_detections.txt")
    except FileNotFoundError:
        print("valid_detections.txt not found")
        return
    
    if not parsed_data:
        print("No valid detections found")
        return
    
    # Step 3: Aggregate statistics
    class_stats = aggregate_statistics(parsed_data)
    print(f"Aggregated statistics for {len(class_stats)} classes")
    
    # Step 4: Find best matches for reference distance
    reference_distance = 2.8
    tolerance = 0.05
    matching_classes = find_best_matches(class_stats, reference_distance, tolerance)
    
    # Step 5: Generate distance map
    distance_map = generate_distance_map(class_stats, labels, matching_classes)
    
    # Step 6: Output results
    print("\n" + "="*80)
    print("PER-CLASS DISTANCE MAP")
    print("="*80)
    
    print(f"\nReference distance: {reference_distance}m (tolerance: ±{tolerance}m)")
    
    if matching_classes:
        print(f"\nClasses matching reference distance:")
        print("Class ID\tDisplay Name\t\tMatches\tPercentage\tMedian Distance")
        print("--------\t------------\t\t-------\t----------\t--------------")
        for class_id, matches, percentage, median_dist, count in matching_classes:
            display_name = labels.get(class_id, f"<not mapped>")
            print(f"{class_id}\t\t{display_name}\t\t{matches}\t{percentage:.1f}%\t\t{median_dist:.3f}m")
        
        best_match = matching_classes[0]
        print(f"\nPRIMARY CANDIDATE:")
        print(f"Class {best_match[0]} ({labels.get(best_match[0], '<not mapped>')}) "
              f"with {best_match[1]} matches ({best_match[2]:.1f}% of its {best_match[4]} detections)")
    else:
        print(f"\nNo classes found matching the reference distance of {reference_distance}m")
    
    print(f"\nComplete Per-Class Distance Map:")
    print("Class ID\tDisplay Name\t\tCount\tMedian Dist\tMean Dist\tMin Dist\tMax Dist")
    print("--------\t------------\t\t-----\t-----------\t---------\t--------\t--------")
    
    # Sort by median distance
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['median'])
    
    for class_id, stats in sorted_classes:
        display_name = labels.get(class_id, f"<not mapped>")
        print(f"{class_id}\t\t{display_name}\t\t{stats['count']}\t{stats['median']:.3f}m\t\t"
              f"{stats['mean']:.3f}m\t\t{stats['min']:.3f}m\t\t{stats['max']:.3f}m")
    
    # Step 7: Save to files
    try:
        save_to_json(distance_map, 'class_distance_map.json')
        print(f"\nSaved distance map to class_distance_map.json")
    except Exception as e:
        print(f"\nError saving JSON file: {e}")
    
    try:
        save_to_csv(distance_map, 'class_distance_map.csv')
        print(f"Saved distance map to class_distance_map.csv")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
    
    print(f"\nGeneration complete!")

if __name__ == "__main__":
    main()