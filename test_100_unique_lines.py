"""
Test script for traversing circle with 100 unique lines using line avoidance.

This script demonstrates the improved algorithm that traces 100 different lines
by avoiding already-traced lines.
"""

import numpy as np
from app import ImageProcessor

print("="*70)
print("Testing Circle Traversal with 100 Unique Lines")
print("="*70)
print()

# Load image and calculate sinogram
print("Loading image and calculating sinogram...")
processor = ImageProcessor()
processor.load("images/timik.jpg")
processor.calculate_sinogram(1000)
print(f"Image shape: {processor.processed_image.shape}")
print(f"Sinogram shape: {processor.sinogram.shape}")
print()

# Traverse circle with 100 steps and line avoidance
print("Traversing circle with 100 steps (avoiding traced lines)...")
path = processor.traverse_circle(100, avoid_traced=True)
print(f"Path generated with {len(path)} points")
print()

# Analyze path
phis = [p[0] for p in path]
thetas = [p[1] for p in path]
values = [p[2] for p in path]

# Verify uniqueness
unique_lines = set()
for phi, theta in zip(phis, thetas):
    unique_lines.add((round(phi, 1), round(theta, 1)))

print("Path statistics:")
print(f"  Total steps: {len(path)}")
print(f"  Unique lines: {len(unique_lines)}")
print(f"  Efficiency: {len(unique_lines)}/{len(path)} = {len(unique_lines)/len(path)*100:.1f}%")
print()
print(f"  Phi range: [{min(phis):.1f}deg, {max(phis):.1f}deg]")
print(f"  Theta range: [{min(thetas):.1f}deg, {max(thetas):.1f}deg]")
print()
print(f"  Starting value: {values[0]:.4f}")
print(f"  Ending value: {values[-1]:.4f}")
print(f"  Value drop: {values[0] - values[-1]:.4f} ({(values[0]-values[-1])/values[0]*100:.2f}%)")
print(f"  Min value: {min(values):.4f}")
print(f"  Max value: {max(values):.4f}")
print(f"  Mean value: {np.mean(values):.4f}")
print()

# Distribution analysis
print("Theta distribution:")
theta_bins = {}
for theta in thetas:
    bin_key = int(theta / 10) * 10
    theta_bins[bin_key] = theta_bins.get(bin_key, 0) + 1

for bin_start in sorted(theta_bins.keys()):
    count = theta_bins[bin_start]
    bar = '#' * (count // 2)
    print(f"  {bin_start:3d}-{bin_start+10:3d}deg: {count:3d} {bar}")
print()

# Show sample of steps
print("Sample of traced lines:")
print(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}")
print("-" * 50)
sample_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
for i in sample_indices:
    if i < len(path):
        phi, theta, value = path[i]
        print(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}")
print()

# Draw the traversal
print("Drawing traversal lines...")
output_path = "traversal_100_unique_lines.png"
result_img = processor.draw_traversal(path, output_path=output_path)
print(f"Saved visualization to: {output_path}")
print(f"Image size: {result_img.size}")
print()

# Quality analysis
print("Quality metrics:")
high_quality = sum(1 for v in values if v >= values[0] * 0.99)
medium_quality = sum(1 for v in values if values[0] * 0.95 <= v < values[0] * 0.99)
lower_quality = sum(1 for v in values if v < values[0] * 0.95)

print(f"  High quality (>99% of max):   {high_quality:3d} lines ({high_quality/len(values)*100:5.1f}%)")
print(f"  Medium quality (95-99% of max): {medium_quality:3d} lines ({medium_quality/len(values)*100:5.1f}%)")
print(f"  Lower quality (<95% of max):   {lower_quality:3d} lines ({lower_quality/len(values)*100:5.1f}%)")
print()

# Check for cycles
print("Cycle detection:")
phi_distances = []
for i in range(1, len(path)):
    phi1, phi2 = phis[i-1], phis[i]
    dist = min(abs(phi2 - phi1), 360 - abs(phi2 - phi1))
    phi_distances.append(dist)

print(f"  Mean phi jump: {np.mean(phi_distances):.2f}deg")
print(f"  Median phi jump: {np.median(phi_distances):.2f}deg")
print(f"  Max phi jump: {max(phi_distances):.2f}deg")
print(f"  Min phi jump: {min(phi_distances):.2f}deg")
print()

# Large jumps (likely switching between different line clusters)
large_jumps = sum(1 for d in phi_distances if d > 90)
print(f"  Large jumps (>90deg): {large_jumps}")
print(f"  Average lines per cluster: {len(path) / (large_jumps + 1):.1f}")
print()

print("="*70)
print("Test completed successfully!")
print(f"Successfully traced {len(unique_lines)} unique lines!")
print(f"Visualization saved to: {output_path}")
print("="*70)
