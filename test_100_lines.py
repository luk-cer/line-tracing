"""
Test script for traversing circle with 100 lines and visualizing the result.

This script:
1. Loads an image
2. Calculates the sinogram
3. Traverses the circle for 100 steps
4. Draws the lines on a white background
5. Saves the result
"""

import numpy as np
from app import ImageProcessor

print("="*70)
print("Testing Circle Traversal with 100 Lines")
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

# Find global maximum
print("Finding global maximum...")
phi_max, theta_max, value_max = processor.find_global_max()
print(f"Global max at phi={phi_max:.2f}deg, theta={theta_max:.2f}deg")
print(f"Value: {value_max:.4f}")
print()

# Traverse circle with 100 steps
print("Traversing circle with 100 steps...")
path = processor.traverse_circle(100)
print(f"Path generated with {len(path)} points")
print()

# Analyze path
phis = [p[0] for p in path]
thetas = [p[1] for p in path]
values = [p[2] for p in path]

print("Path statistics:")
print(f"  Phi range: [{min(phis):.1f}deg, {max(phis):.1f}deg]")
print(f"  Theta range: [{min(thetas):.1f}deg, {max(thetas):.1f}deg]")
print(f"  Value range: [{min(values):.2f}, {max(values):.2f}]")
print(f"  Starting value: {values[0]:.4f}")
print(f"  Ending value: {values[-1]:.4f}")
print(f"  Mean value: {np.mean(values):.4f}")
print()

# Count unique points
unique_phis = len(set([round(p, 1) for p in phis]))
unique_thetas = len(set([round(t, 1) for t in thetas]))
print(f"Unique phi positions (rounded to 0.1deg): {unique_phis}")
print(f"Unique theta directions (rounded to 0.1deg): {unique_thetas}")
print()

# Show first 10 and last 10 steps
print("First 10 steps:")
print(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}")
print("-" * 50)
for i in range(min(10, len(path))):
    phi, theta, value = path[i]
    print(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}")
print()

print("Last 10 steps:")
print(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}")
print("-" * 50)
for i in range(max(0, len(path)-10), len(path)):
    phi, theta, value = path[i]
    print(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}")
print()

# Draw the traversal
print("Drawing traversal lines...")
output_path = "traversal_100_lines.png"
result_img = processor.draw_traversal(path, output_path=output_path)
print(f"Saved visualization to: {output_path}")
print(f"Image size: {result_img.size}")
print()

# Additional analysis
print("Cycle analysis:")
# Check if path forms cycles by looking at phi distances
phi_distances = []
for i in range(1, len(path)):
    phi1, phi2 = phis[i-1], phis[i]
    # Calculate shortest angular distance
    dist = min(abs(phi2 - phi1), 360 - abs(phi2 - phi1))
    phi_distances.append(dist)

print(f"  Mean phi jump: {np.mean(phi_distances):.2f}deg")
print(f"  Max phi jump: {max(phi_distances):.2f}deg")
print(f"  Min phi jump: {min(phi_distances):.2f}deg")

# Check if we're oscillating between two points
if len(path) >= 3:
    is_oscillating = True
    for i in range(2, min(20, len(path))):
        if not (np.isclose(phis[i], phis[i-2], atol=1.0)):
            is_oscillating = False
            break
    print(f"  Oscillating between 2 points: {is_oscillating}")
print()

print("="*70)
print("Test completed successfully!")
print(f"Visualization saved to: {output_path}")
print("="*70)
