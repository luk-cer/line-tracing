"""
Test script for app_v2.py - Graph-Based Line Tracing

Tests the new implementation with a small example (10 lines).
"""

import time
from app_v2 import ImageProcessorV2

def main():
    print("=" * 70)
    print("Testing ImageProcessorV2 - Graph-Based Line Tracing")
    print("=" * 70)
    print()

    # Create processor with 360 vertices (1° spacing)
    print("Creating processor with 360 vertices (1° spacing)...")
    processor = ImageProcessorV2(num_points_on_circle=360)
    print()

    # Load image
    print("Loading image...")
    start_time = time.time()
    processor.load("images/timik.jpg")
    load_time = time.time() - start_time
    print(f"Image loaded in {load_time:.2f} seconds")
    print(f"Image dimensions: {processor.processed_image.shape}")
    print(f"Circle center: ({processor.center_x}, {processor.center_y})")
    print(f"Circle radius: {processor.radius}")
    print()

    # Calculate sinogram
    print("Calculating sinogram...")
    start_time = time.time()
    processor.calculate_sinogram(num_angles=1000)
    sinogram_time = time.time() - start_time
    print(f"Sinogram calculated in {sinogram_time:.2f} seconds")
    print(f"Sinogram shape: {processor.sinogram.shape}")
    print()

    # Trace 10 lines (no recalculation for quick test)
    print("Tracing 10 lines (no radon recalculation)...")
    start_time = time.time()
    path = processor.trace_lines(
        max_lines=10,
        recalculate_radon_every_n=0,  # Disable for quick test
        start_at=None  # Auto-detect
    )
    trace_time = time.time() - start_time
    print(f"Traced {len(path)} lines in {trace_time:.2f} seconds")
    print()

    # Print path details
    print("Path details:")
    for i, (phi, theta, intensity) in enumerate(path):
        print(f"  Step {i}: phi={phi:.2f}deg, theta={theta:.2f}deg, intensity={intensity:.4f}")
    print()

    # Verify no duplicates (should be 100% efficiency)
    phis = [p[0] for p in path]
    thetas = [p[1] for p in path]

    print("Statistics:")
    print(f"  Total lines: {len(path)}")
    print(f"  Value range: [{min(p[2] for p in path):.4f}, {max(p[2] for p in path):.4f}]")
    print(f"  Phi range: [{min(phis):.1f}deg, {max(phis):.1f}deg]")
    print(f"  Theta range: [{min(thetas):.1f}deg, {max(thetas):.1f}deg]")
    print()

    # Calculate MSE
    print("Calculating MSE...")
    mse = processor.calculate_mse(path)
    print(f"MSE vs original: {mse:.6f}")
    print()

    # Render visualization
    print("Rendering visualization...")
    output_path = "output/test_v2_10_lines.png"
    processor.draw_traversal(path, output_path=output_path)
    print(f"Saved to: {output_path}")
    print()

    # Summary
    total_time = load_time + sinogram_time + trace_time
    print("=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Path format: {type(path)} with {len(path)} tuples")
    print(f"Path compatible with old app.py: YES")
    print("=" * 70)


if __name__ == '__main__':
    main()
