#!/usr/bin/env python3

# Camera parameters from the code
CAMERA_FOCAL_LENGTH_MM = 4.74
SENSOR_WIDTH_MM = 6.45
IMAGE_WIDTH_PIXELS = 320
TARGET_WIDTH_METERS = 0.5  # 50 cm

# Calculate focal length in pixels
focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * IMAGE_WIDTH_PIXELS) / SENSOR_WIDTH_MM
print(f"Focal length in pixels: {focal_length_pixels:.2f}")

# Calculate distance for different bounding box widths
# Using the formula: distance = (real_world_size * focal_length_pixels) / pixel_size

# Example: If the bounding box width is 100 pixels
pixel_size = 100
distance = (TARGET_WIDTH_METERS * focal_length_pixels) / pixel_size
print(f"For a 100-pixel wide bounding box: {distance:.2f} meters")

# Example: If the bounding box width is 50 pixels
pixel_size = 50
distance = (TARGET_WIDTH_METERS * focal_length_pixels) / pixel_size
print(f"For a 50-pixel wide bounding box: {distance:.2f} meters")

# Example: If the bounding box width is 25 pixels
pixel_size = 25
distance = (TARGET_WIDTH_METERS * focal_length_pixels) / pixel_size
print(f"For a 25-pixel wide bounding box: {distance:.2f} meters")

# Calculate what pixel size would correspond to 1.875 meters
target_distance = 1.875
pixel_size = (TARGET_WIDTH_METERS * focal_length_pixels) / target_distance
print(f"\nFor a target distance of 1.875 meters: {pixel_size:.2f} pixels wide bounding box")

# Calculate what pixel size would correspond to 22.187 meters (the most common value)
target_distance = 22.187
pixel_size = (TARGET_WIDTH_METERS * focal_length_pixels) / target_distance
print(f"For a target distance of 22.187 meters: {pixel_size:.2f} pixels wide bounding box")