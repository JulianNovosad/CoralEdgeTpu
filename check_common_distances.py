#!/usr/bin/env python3

# Camera parameters from the code
CAMERA_FOCAL_LENGTH_MM = 4.74
SENSOR_WIDTH_MM = 6.45
IMAGE_WIDTH_PIXELS = 320
TARGET_WIDTH_METERS = 0.5  # 50 cm

# Calculate focal length in pixels
focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * IMAGE_WIDTH_PIXELS) / SENSOR_WIDTH_MM
print(f"Focal length in pixels: {focal_length_pixels:.2f}")

# Calculate what pixel size would correspond to 22.187 meters (the most common value)
target_distance = 22.187
pixel_size = (TARGET_WIDTH_METERS * focal_length_pixels) / target_distance
print(f"For a target distance of 22.187 meters: {pixel_size:.2f} pixels wide bounding box")

# Calculate what pixel size would correspond to 24.036 meters (second most common value)
target_distance = 24.036
pixel_size = (TARGET_WIDTH_METERS * focal_length_pixels) / target_distance
print(f"For a target distance of 24.036 meters: {pixel_size:.2f} pixels wide bounding box")

# Calculate what pixel size would correspond to 20.800 meters (third most common value)
target_distance = 20.800
pixel_size = (TARGET_WIDTH_METERS * focal_length_pixels) / target_distance
print(f"For a target distance of 20.800 meters: {pixel_size:.2f} pixels wide bounding box")

# Calculate what distance would correspond to a 10-pixel wide bounding box
pixel_size = 10
distance = (TARGET_WIDTH_METERS * focal_length_pixels) / pixel_size
print(f"\nFor a 10-pixel wide bounding box: {distance:.2f} meters")

# Calculate what distance would correspond to a 5-pixel wide bounding box
pixel_size = 5
distance = (TARGET_WIDTH_METERS * focal_length_pixels) / pixel_size
print(f"For a 5-pixel wide bounding box: {distance:.2f} meters")