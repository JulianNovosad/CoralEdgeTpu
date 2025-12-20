#!/usr/bin/env python3

# Read stable window distance values from file
with open('stable_window.txt', 'r') as f:
    distances = [float(line.strip()) for line in f.readlines()]

# Calculate statistics for stable window
if distances:
    min_dist = min(distances)
    max_dist = max(distances)
    median_dist = sorted(distances)[len(distances)//2]
    
    # Calculate mean
    mean_dist = sum(distances) / len(distances)
    
    # Calculate variance and standard deviation
    variance = sum((d - mean_dist)**2 for d in distances) / len(distances)
    std_dev = variance**0.5
    
    print(f"Stable window distance estimates: {len(distances)}")
    print(f"Minimum distance: {min_dist:.3f} meters")
    print(f"Maximum distance: {max_dist:.3f} meters")
    print(f"Median distance: {median_dist:.3f} meters")
    print(f"Mean distance: {mean_dist:.3f} meters")
    print(f"Standard deviation: {std_dev:.3f} meters")
    print(f"Variance: {variance:.3f}")
    
    # Calculate jitter as the difference between max and min
    jitter = max_dist - min_dist
    print(f"Jitter (max-min): {jitter:.3f} meters")
    
    # Count how many values are exactly 22.187 meters
    count_22_187 = sum(1 for d in distances if abs(d - 22.187172) < 0.001)
    print(f"Values at 22.187 meters: {count_22_187}/{len(distances)} ({100*count_22_187/len(distances):.1f}%)")
    
else:
    print("No stable window distance values found.")