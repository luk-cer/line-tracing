"""
Run line tracing algorithm with 200 lines and save to output folder.

This script:
1. Loads the image and calculates sinogram
2. Traverses the circle for 200 steps with line avoidance
3. Saves the black-on-white visualization to output folder
4. Prints statistics about the traced lines
"""

import numpy as np
from app import ImageProcessor
import os

print("="*70)
print("Line Tracing Algorithm - 200 Lines")
print("="*70)
print()

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load image and calculate sinogram
print("Loading image and calculating sinogram...")
processor = ImageProcessor()
processor.load("images/timik.jpg")
processor.calculate_sinogram(1000)
print(f"Image shape: {processor.processed_image.shape}")
print(f"Sinogram shape: {processor.sinogram.shape}")
print()

# Traverse circle with 200 steps
print("Tracing 200 lines (avoiding already-traced lines)...")
path = processor.traverse_circle(200, avoid_traced=True)
print(f"Successfully traced {len(path)} lines")
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

print("Value statistics:")
print(f"  Starting value: {values[0]:.4f}")
print(f"  Ending value: {values[-1]:.4f}")
print(f"  Value drop: {values[0] - values[-1]:.4f} ({(values[0]-values[-1])/values[0]*100:.2f}%)")
print(f"  Min value: {min(values):.4f}")
print(f"  Max value: {max(values):.4f}")
print(f"  Mean value: {np.mean(values):.4f}")
print()

# Quality metrics
high_quality = sum(1 for v in values if v >= values[0] * 0.99)
medium_quality = sum(1 for v in values if values[0] * 0.95 <= v < values[0] * 0.99)
lower_quality = sum(1 for v in values if v < values[0] * 0.95)

print("Quality distribution:")
print(f"  High quality (>99% of max):   {high_quality:3d} lines ({high_quality/len(values)*100:5.1f}%)")
print(f"  Medium quality (95-99% of max): {medium_quality:3d} lines ({medium_quality/len(values)*100:5.1f}%)")
print(f"  Lower quality (<95% of max):   {lower_quality:3d} lines ({lower_quality/len(values)*100:5.1f}%)")
print()

# Angle distribution
print("Angle coverage:")
print(f"  Phi range: [{min(phis):.1f}deg, {max(phis):.1f}deg]")
print(f"  Theta range: [{min(thetas):.1f}deg, {max(thetas):.1f}deg]")
print()

# Render visualization
print("="*70)
print("Rendering visualization...")
print("="*70)

output_path = os.path.join("output", "traced_200_lines.png")
result_img = processor.draw_traversal(path, output_path=output_path)

print(f"Saved visualization to: {output_path}")
print(f"Image size: {result_img.size}")
print()

# Save statistics to text file
stats_path = os.path.join("output", "traced_200_lines_stats.txt")
with open(stats_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("Line Tracing Algorithm - Statistics for 200 Lines\n")
    f.write("="*70 + "\n\n")

    f.write("Path Statistics:\n")
    f.write(f"  Total steps: {len(path)}\n")
    f.write(f"  Unique lines: {len(unique_lines)}\n")
    f.write(f"  Efficiency: {len(unique_lines)}/{len(path)} = {len(unique_lines)/len(path)*100:.1f}%\n\n")

    f.write("Value Statistics:\n")
    f.write(f"  Starting value: {values[0]:.4f}\n")
    f.write(f"  Ending value: {values[-1]:.4f}\n")
    f.write(f"  Value drop: {values[0] - values[-1]:.4f} ({(values[0]-values[-1])/values[0]*100:.2f}%)\n")
    f.write(f"  Min value: {min(values):.4f}\n")
    f.write(f"  Max value: {max(values):.4f}\n")
    f.write(f"  Mean value: {np.mean(values):.4f}\n\n")

    f.write("Quality Distribution:\n")
    f.write(f"  High quality (>99% of max):   {high_quality:3d} lines ({high_quality/len(values)*100:5.1f}%)\n")
    f.write(f"  Medium quality (95-99% of max): {medium_quality:3d} lines ({medium_quality/len(values)*100:5.1f}%)\n")
    f.write(f"  Lower quality (<95% of max):   {lower_quality:3d} lines ({lower_quality/len(values)*100:5.1f}%)\n\n")

    f.write("Angle Coverage:\n")
    f.write(f"  Phi range: [{min(phis):.1f}deg, {max(phis):.1f}deg]\n")
    f.write(f"  Theta range: [{min(thetas):.1f}deg, {max(thetas):.1f}deg]\n\n")

    f.write("Detailed Line Data:\n")
    f.write(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}\n")
    f.write("-" * 50 + "\n")
    for i, (phi, theta, value) in enumerate(path):
        f.write(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}\n")

print(f"Saved statistics to: {stats_path}")
print()

print("="*70)
print("COMPLETED SUCCESSFULLY")
print("="*70)
print(f"Output files:")
print(f"  - {output_path}")
print(f"  - {stats_path}")
print()
print(f"Traced {len(unique_lines)} unique lines with {len(unique_lines)/len(path)*100:.1f}% efficiency")
print("="*70)
