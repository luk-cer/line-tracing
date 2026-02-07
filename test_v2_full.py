"""
Test script for app_v2.py - Full test with 5000 lines

Tests the new graph-based implementation with:
- 360 vertices (1° spacing)
- 5000 lines to trace
- Radon recalculation every 10 lines
"""

import time
from app_v2 import ImageProcessorV2

def main():
    print("=" * 70)
    print("Testing ImageProcessorV2 - Full Test (5000 lines, r=10)")
    print("=" * 70)
    print()

    # Create processor with 360 vertices
    print("Creating processor with 360 vertices...")
    processor = ImageProcessorV2(num_points_on_circle=360)
    print()

    # Load image
    print("Loading image...")
    start_time = time.time()
    processor.load("images/timik.jpg")
    load_time = time.time() - start_time
    print(f"Load time: {load_time:.2f} seconds")
    print()

    # Calculate sinogram
    print("Calculating sinogram...")
    start_time = time.time()
    processor.calculate_sinogram(num_angles=1000)
    sinogram_time = time.time() - start_time
    print(f"Sinogram calculation time: {sinogram_time:.2f} seconds")
    print()

    # Trace 5000 lines with recalculation every 10
    print("Tracing 5000 lines (recalculate radon every 10)...")
    start_time = time.time()
    path = processor.trace_lines(
        max_lines=5000,
        recalculate_radon_every_n=10,
        start_at=None
    )
    trace_time = time.time() - start_time
    print(f"Traced {len(path)} lines in {trace_time:.2f} seconds ({trace_time/60:.2f} minutes)")
    print()

    # Calculate efficiency
    print("Calculating efficiency...")
    unique_lines = set()
    for phi, theta, _ in path:
        # Use same rounding as old implementation
        phi_exit = processor.get_exit_point(phi, theta)
        line_id = frozenset({round(phi, 1), round(phi_exit, 1)})
        unique_lines.add(line_id)

    efficiency = len(unique_lines) / len(path) * 100
    print(f"Total steps: {len(path)}")
    print(f"Unique lines: {len(unique_lines)}")
    print(f"Efficiency: {len(unique_lines)}/{len(path)} = {efficiency:.1f}%")
    print()

    # Value statistics
    values = [p[2] for p in path]
    print("Value statistics:")
    print(f"  Starting value: {values[0]:.4f}")
    print(f"  Ending value: {values[-1]:.4f}")
    print(f"  Value drop: {values[0] - values[-1]:.4f} ({(values[0]-values[-1])/values[0]*100:.2f}%)")
    print(f"  Min value: {min(values):.4f}")
    print(f"  Max value: {max(values):.4f}")
    print(f"  Mean value: {sum(values)/len(values):.4f}")
    print()

    # Quality distribution
    high_quality = sum(1 for v in values if v >= values[0] * 0.99)
    medium_quality = sum(1 for v in values if values[0] * 0.95 <= v < values[0] * 0.99)
    lower_quality = sum(1 for v in values if v < values[0] * 0.95)

    print("Quality distribution:")
    print(f"  High quality (>99% of max):   {high_quality:4d} lines ({high_quality/len(values)*100:5.1f}%)")
    print(f"  Medium quality (95-99% of max): {medium_quality:4d} lines ({medium_quality/len(values)*100:5.1f}%)")
    print(f"  Lower quality (<95% of max):   {lower_quality:4d} lines ({lower_quality/len(values)*100:5.1f}%)")
    print()

    # Calculate MSE
    print("Calculating MSE...")
    mse = processor.calculate_mse(path)
    print(f"MSE vs original: {mse:.6f}")
    print()

    # Render visualization
    print("Rendering visualization...")
    render_start = time.time()
    output_path = "output/traced_timik_v2_5000_lines.png"
    processor.draw_traversal(path, output_path=output_path)
    render_time = time.time() - render_start
    print(f"Saved to: {output_path}")
    print(f"Render time: {render_time:.2f} seconds")
    print()

    # Final summary
    total_time = load_time + sinogram_time + trace_time + render_time
    print("=" * 70)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Traced {len(unique_lines)} unique lines with {efficiency:.1f}% efficiency")
    print(f"Lines per second: {len(path)/trace_time:.2f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
