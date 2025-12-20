#!/usr/bin/env python3

# Read distance values from file
with open('distance_values.txt', 'r') as f:
    distances = [float(line.strip()) for line in f.readlines()]

# Calculate statistics
if distances:
    min_dist = min(distances)
    max_dist = max(distances)
    median_dist = sorted(distances)[len(distances)//2]
    
    # Calculate mean
    mean_dist = sum(distances) / len(distances)
    
    # Calculate variance and standard deviation
    variance = sum((d - mean_dist)**2 for d in distances) / len(distances)
    std_dev = variance**0.5
    
    print(f"Number of distance estimates: {len(distances)}")
    print(f"Minimum distance: {min_dist:.3f} meters")
    print(f"Maximum distance: {max_dist:.3f} meters")
    print(f"Median distance: {median_dist:.3f} meters")
    print(f"Mean distance: {mean_dist:.3f} meters")
    print(f"Standard deviation: {std_dev:.3f} meters")
    print(f"Variance: {variance:.3f}")
    
    # Find most frequent values (modes)
    from collections import Counter
    counter = Counter(distances)
    most_common = counter.most_common(5)
    print("\nMost frequent distance values:")
    for value, count in most_common:
        print(f"  {value:.3f} meters ({count} times)")
else:
    print("No distance values found.")