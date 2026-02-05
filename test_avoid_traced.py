"""
Test script comparing traversal with and without line avoidance.

This demonstrates the difference between:
1. Retracing the same line repeatedly (avoid_traced=False)
2. Finding new lines each time (avoid_traced=True)
"""

import numpy as np
from app import ImageProcessor

print("="*70)
print("Testing Line Avoidance Feature")
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

# Test 1: Without line avoidance (old behavior)
print("="*70)
print("Test 1: WITHOUT line avoidance (avoid_traced=False)")
print("="*70)
path_no_avoid = processor.traverse_circle(20, avoid_traced=False)
print(f"Generated {len(path_no_avoid)} steps")
print()

# Analyze unique lines
phis_no_avoid = [p[0] for p in path_no_avoid]
thetas_no_avoid = [p[1] for p in path_no_avoid]
values_no_avoid = [p[2] for p in path_no_avoid]

unique_lines_no_avoid = len(set([
    (round(p, 1), round(t, 1))
    for p, t in zip(phis_no_avoid, thetas_no_avoid)
]))

print(f"Unique lines traced: {unique_lines_no_avoid}")
print(f"Value range: [{min(values_no_avoid):.2f}, {max(values_no_avoid):.2f}]")
print()

print("First 10 steps:")
print(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}")
print("-" * 50)
for i in range(min(10, len(path_no_avoid))):
    phi, theta, value = path_no_avoid[i]
    print(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}")
print()

# Test 2: With line avoidance (new behavior)
print("="*70)
print("Test 2: WITH line avoidance (avoid_traced=True)")
print("="*70)
path_with_avoid = processor.traverse_circle(100, avoid_traced=True)
print(f"Generated {len(path_with_avoid)} steps")
print()

# Analyze unique lines
phis_with_avoid = [p[0] for p in path_with_avoid]
thetas_with_avoid = [p[1] for p in path_with_avoid]
values_with_avoid = [p[2] for p in path_with_avoid]

unique_lines_with_avoid = len(set([
    (round(p, 1), round(t, 1))
    for p, t in zip(phis_with_avoid, thetas_with_avoid)
]))

print(f"Unique lines traced: {unique_lines_with_avoid}")
print(f"Value range: [{min(values_with_avoid):.2f}, {max(values_with_avoid):.2f}]")
print(f"Value drop: {values_with_avoid[0]:.2f} -> {values_with_avoid[-1]:.2f} ({values_with_avoid[0] - values_with_avoid[-1]:.2f})")
print()

print("First 10 steps:")
print(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}")
print("-" * 50)
for i in range(min(10, len(path_with_avoid))):
    phi, theta, value = path_with_avoid[i]
    print(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}")
print()

print("Last 10 steps:")
print(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}")
print("-" * 50)
for i in range(max(0, len(path_with_avoid)-10), len(path_with_avoid)):
    phi, theta, value = path_with_avoid[i]
    print(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}")
print()

# Value degradation analysis
print("Value degradation analysis:")
step_size = len(path_with_avoid) // 10
for i in range(0, len(path_with_avoid), step_size):
    if i < len(path_with_avoid):
        value = values_with_avoid[i]
        pct_of_max = (value / values_with_avoid[0]) * 100
        print(f"  Step {i:3d}: value={value:7.2f} ({pct_of_max:5.1f}% of max)")
print()

# Visualize both paths
print("="*70)
print("Generating visualizations...")
print("="*70)

print("Drawing path WITHOUT line avoidance...")
img_no_avoid = processor.draw_traversal(path_no_avoid, "traversal_no_avoid.png")
print(f"  Saved: traversal_no_avoid.png")

print("Drawing path WITH line avoidance...")
img_with_avoid = processor.draw_traversal(path_with_avoid, "traversal_with_avoid.png")
print(f"  Saved: traversal_with_avoid.png")
print()

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
print(f"WITHOUT avoidance:")
print(f"  Steps: {len(path_no_avoid)}")
print(f"  Unique lines: {unique_lines_no_avoid}")
print(f"  Efficiency: {unique_lines_no_avoid}/{len(path_no_avoid)} = {unique_lines_no_avoid/len(path_no_avoid)*100:.1f}%")
print()
print(f"WITH avoidance:")
print(f"  Steps: {len(path_with_avoid)}")
print(f"  Unique lines: {unique_lines_with_avoid}")
print(f"  Efficiency: {unique_lines_with_avoid}/{len(path_with_avoid)} = {unique_lines_with_avoid/len(path_with_avoid)*100:.1f}%")
print()
print(f"Improvement: {unique_lines_with_avoid - unique_lines_no_avoid} more unique lines")
print("="*70)
