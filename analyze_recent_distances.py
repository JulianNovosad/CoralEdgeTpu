#!/usr/bin/env python3

# Read recent distance values from file
with open('recent_distances.txt', 'r') as f:
    distances = [float(line.strip()) for line in f.readlines()]

# Calculate statistics for recent values
if distances:
    min_dist = min(distances)
    max_dist = max(distances)
    median_dist = sorted(distances)[len(distances)//2]
    
    # Calculate mean
    mean_dist = sum(distances) / len(distances)
    
    # Calculate variance and standard deviation
    variance = sum((d - mean_dist)**2 for d in distances) / len(distances)
    std_dev = variance**0.5
    
    print(f"Recent distance estimates (last 100): {len(distances)}")
    print(f"Minimum distance: {min_dist:.3f} meters")
    print(f"Maximum distance: {max_dist:.3f} meters")
    print(f"Median distance: {median_dist:.3f} meters")
    print(f"Mean distance: {mean_dist:.3f} meters")
    print(f"Standard deviation: {std_dev:.3f} meters")
    print(f"Variance: {variance:.3f}")
    
    # Calculate jitter as the difference between max and min
    jitter = max_dist - min_dist
    print(f"Jitter (max-min): {jitter:.3f} meters")
    
    # Find most frequent values (modes)
    from collections import Counter
    counter = Counter(distances)
    most_common = counter.most_common(5)
    print("\nMost frequent recent distance values:")
    for value, count in most_common:
        print(f"  {value:.3f} meters ({count} times)")
        
    # Calculate how many values are close to the most common value
    most_common_value = most_common[0][0] if most_common else 0
    close_values = [d for d in distances if abs(d - most_common_value) < 1.0]  # Within 1 meter
    print(f"\nValues within 1 meter of most common ({most_common_value:.3f}): {len(close_values)}/{len(distances)} ({100*len(close_values)/len(distances):.1f}%)")
    
else:
    print("No recent distance values found.")