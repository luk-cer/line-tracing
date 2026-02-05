"""
Run line tracing algorithm with 3000 lines and save to output folder.

This script:
1. Loads the image and calculates sinogram
2. Traverses the circle for 3000 steps with line avoidance
3. Saves the black-on-white visualization to output folder
4. Prints statistics about the traced lines
"""

import numpy as np
from app import ImageProcessor
import os
import time

print("="*70)
print("Line Tracing Algorithm - 3000 Lines")
print("="*70)
print()

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load image and calculate sinogram
print("Loading image and calculating sinogram...")
start_time = time.time()
processor = ImageProcessor()
processor.load("images/timik.jpg")
processor.calculate_sinogram(1000)
load_time = time.time() - start_time
print(f"Image shape: {processor.processed_image.shape}")
print(f"Sinogram shape: {processor.sinogram.shape}")
print(f"Load time: {load_time:.2f} seconds")
print()

# Traverse circle with 3000 steps
print("Tracing 3000 lines (avoiding already-traced lines)...")
print("This may take a few minutes...")
trace_start = time.time()
path = processor.traverse_circle(3000, avoid_traced=True)
trace_time = time.time() - trace_start
print(f"Successfully traced {len(path)} lines in {trace_time:.2f} seconds")
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
print(f"  Median value: {np.median(values):.4f}")
print()

# Quality metrics
high_quality = sum(1 for v in values if v >= values[0] * 0.99)
medium_quality = sum(1 for v in values if values[0] * 0.95 <= v < values[0] * 0.99)
lower_quality = sum(1 for v in values if v < values[0] * 0.95)

print("Quality distribution:")
print(f"  High quality (>99% of max):   {high_quality:4d} lines ({high_quality/len(values)*100:5.1f}%)")
print(f"  Medium quality (95-99% of max): {medium_quality:4d} lines ({medium_quality/len(values)*100:5.1f}%)")
print(f"  Lower quality (<95% of max):   {lower_quality:4d} lines ({lower_quality/len(values)*100:5.1f}%)")
print()

# Angle distribution
print("Angle coverage:")
print(f"  Phi range: [{min(phis):.1f}deg, {max(phis):.1f}deg]")
print(f"  Theta range: [{min(thetas):.1f}deg, {max(thetas):.1f}deg]")

# Unique phi and theta values
unique_phis = len(set([round(p, 1) for p in phis]))
unique_thetas = len(set([round(t, 1) for t in thetas]))
print(f"  Unique phi positions: {unique_phis}")
print(f"  Unique theta angles: {unique_thetas}")
print()

# Render visualization
print("="*70)
print("Rendering visualization...")
print("="*70)

render_start = time.time()
output_path = os.path.join("output", "traced_3000_lines.png")
result_img = processor.draw_traversal(path, output_path=output_path)
render_time = time.time() - render_start

print(f"Saved visualization to: {output_path}")
print(f"Image size: {result_img.size}")
print(f"Render time: {render_time:.2f} seconds")
print()

# Save statistics to text file
print("Saving statistics...")
stats_path = os.path.join("output", "traced_3000_lines_stats.txt")
with open(stats_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("Line Tracing Algorithm - Statistics for 3000 Lines\n")
    f.write("="*70 + "\n\n")

    f.write("Performance:\n")
    f.write(f"  Load time: {load_time:.2f} seconds\n")
    f.write(f"  Trace time: {trace_time:.2f} seconds\n")
    f.write(f"  Render time: {render_time:.2f} seconds\n")
    f.write(f"  Total time: {load_time + trace_time + render_time:.2f} seconds\n\n")

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
    f.write(f"  Mean value: {np.mean(values):.4f}\n")
    f.write(f"  Median value: {np.median(values):.4f}\n\n")

    f.write("Quality Distribution:\n")
    f.write(f"  High quality (>99% of max):   {high_quality:4d} lines ({high_quality/len(values)*100:5.1f}%)\n")
    f.write(f"  Medium quality (95-99% of max): {medium_quality:4d} lines ({medium_quality/len(values)*100:5.1f}%)\n")
    f.write(f"  Lower quality (<95% of max):   {lower_quality:4d} lines ({lower_quality/len(values)*100:5.1f}%)\n\n")

    f.write("Angle Coverage:\n")
    f.write(f"  Phi range: [{min(phis):.1f}deg, {max(phis):.1f}deg]\n")
    f.write(f"  Theta range: [{min(thetas):.1f}deg, {max(thetas):.1f}deg]\n")
    f.write(f"  Unique phi positions: {unique_phis}\n")
    f.write(f"  Unique theta angles: {unique_thetas}\n\n")

    f.write("Value Distribution by Percentile:\n")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(values, p)
        f.write(f"  {p:2d}th percentile: {val:.4f}\n")
    f.write("\n")

    f.write("Detailed Line Data (first 100, last 100, and every 100th):\n")
    f.write(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}\n")
    f.write("-" * 50 + "\n")

    # First 100
    for i in range(min(100, len(path))):
        phi, theta, value = path[i]
        f.write(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}\n")

    f.write("...\n")

    # Every 100th from 100 to len-100
    for i in range(100, max(0, len(path)-100), 100):
        phi, theta, value = path[i]
        f.write(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}\n")

    f.write("...\n")

    # Last 100
    for i in range(max(0, len(path)-100), len(path)):
        phi, theta, value = path[i]
        f.write(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}\n")

print(f"Saved statistics to: {stats_path}")
print()

total_time = load_time + trace_time + render_time

print("="*70)
print("COMPLETED SUCCESSFULLY")
print("="*70)
print(f"Output files:")
print(f"  - {output_path}")
print(f"  - {stats_path}")
print()
print(f"Performance summary:")
print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"  Lines per second: {len(path)/trace_time:.1f}")
print()
print(f"Traced {len(unique_lines)} unique lines with {len(unique_lines)/len(path)*100:.1f}% efficiency")
print("="*70)
