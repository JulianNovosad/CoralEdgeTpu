#!/usr/bin/env python3

# Camera parameters from the code
CAMERA_FOCAL_LENGTH_MM = 4.74
SENSOR_WIDTH_MM = 6.45
IMAGE_WIDTH_PIXELS = 320
TARGET_WIDTH_METERS = 0.5  # 50 cm

# Calculate focal length in pixels
focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM * IMAGE_WIDTH_PIXELS) / SENSOR_WIDTH_MM
print(f"Focal length in pixels: {focal_length_pixels:.2f}")

# Verify a few calculations from our data
test_cases = [
    {"width_px": 125.1, "reported_dist": 1.877},
    {"width_px": 26.9, "reported_dist": 3.104},
    {"width_px": 8.1, "reported_dist": 4.678},
    {"width_px": 18.8, "reported_dist": 6.357},
]

print("\nVerification of distance calculations:")
print("Bounding Box Width\tReported Distance\tCalculated Distance\tDifference")
print("-----------------\t-----------------\t-------------------\t----------")

for case in test_cases:
    # Calculate distance using the pinhole camera model
    calculated_dist = (TARGET_WIDTH_METERS * focal_length_pixels) / case["width_px"]
    difference = abs(case["reported_dist"] - calculated_dist)
    print(f"{case['width_px']:>17.1f}\t{case['reported_dist']:>17.3f}\t{calculated_dist:>19.3f}\t{difference:>10.3f}")