"""
Test script for app_v2.py - 60 nodes, 100 lines

Tests with:
- 60 vertices (6° spacing)
- 100 lines to trace
- Radon recalculation every 10 lines
"""

import sys
import time
from app_v2 import ImageProcessorV2

def main():
    print("=" * 70, flush=True)
    print("Testing ImageProcessorV2 - 60 Nodes, 100 Lines (r=10)", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    # Create processor with 60 vertices (6° spacing)
    print("Creating processor with 60 vertices (6° spacing)...", flush=True)
    processor = ImageProcessorV2(num_points_on_circle=60)
    print(flush=True)

    # Load image
    print("Loading image...", flush=True)
    start_time = time.time()
    processor.load("images/timik.jpg")
    load_time = time.time() - start_time
    print(f"Load time: {load_time:.2f} seconds", flush=True)
    print(flush=True)

    # Calculate sinogram
    print("Calculating sinogram...", flush=True)
    sys.stdout.flush()
    start_time = time.time()
    processor.calculate_sinogram(num_angles=1000)
    sinogram_time = time.time() - start_time
    print(f"Sinogram calculation time: {sinogram_time:.2f} seconds", flush=True)
    print(f"Graph edges: {processor.num_vertices * (processor.num_vertices - 1) // 2} unique", flush=True)
    print(flush=True)

    # Trace 100 lines
    print("Tracing 100 lines (recalculate radon every 10)...", flush=True)
    sys.stdout.flush()
    start_time = time.time()
    path = processor.trace_lines(
        max_lines=100,
        recalculate_radon_every_n=10,
        start_at=None
    )
    trace_time = time.time() - start_time
    print(f"Traced {len(path)} lines in {trace_time:.2f} seconds", flush=True)
    print(flush=True)

    # Calculate efficiency
    print("Analyzing results...", flush=True)
    unique_lines = set()
    for phi, theta, _ in path:
        try:
            phi_exit = processor.get_exit_point(phi, theta)
            line_id = frozenset({round(phi, 1), round(phi_exit, 1)})
            unique_lines.add(line_id)
        except ValueError:
            pass

    efficiency = len(unique_lines) / len(path) * 100
    print(f"Total steps: {len(path)}", flush=True)
    print(f"Unique lines: {len(unique_lines)}", flush=True)
    print(f"Efficiency: {efficiency:.1f}%", flush=True)
    print(flush=True)

    # Value statistics
    values = [p[2] for p in path]
    print("Value statistics:", flush=True)
    print(f"  Starting value: {values[0]:.4f}", flush=True)
    print(f"  Ending value: {values[-1]:.4f}", flush=True)
    print(f"  Value drop: {values[0] - values[-1]:.4f} ({(values[0]-values[-1])/values[0]*100:.2f}%)", flush=True)
    print(flush=True)

    # Quality distribution
    high_quality = sum(1 for v in values if v >= values[0] * 0.99)
    medium_quality = sum(1 for v in values if values[0] * 0.95 <= v < values[0] * 0.99)
    lower_quality = sum(1 for v in values if v < values[0] * 0.95)

    print("Quality distribution:", flush=True)
    print(f"  High quality (>99% of max):   {high_quality:4d} lines ({high_quality/len(values)*100:5.1f}%)", flush=True)
    print(f"  Medium quality (95-99% of max): {medium_quality:4d} lines ({medium_quality/len(values)*100:5.1f}%)", flush=True)
    print(f"  Lower quality (<95% of max):   {lower_quality:4d} lines ({lower_quality/len(values)*100:5.1f}%)", flush=True)
    print(flush=True)

    # Calculate MSE
    print("Calculating MSE...", flush=True)
    mse = processor.calculate_mse(path)
    print(f"MSE: {mse:.6f}", flush=True)
    print(flush=True)

    # Render visualization
    print("Rendering visualization...", flush=True)
    output_path = "output/traced_timik_v2_60nodes_100lines.png"
    processor.draw_traversal(path, output_path=output_path)
    print(f"Saved to: {output_path}", flush=True)
    print(flush=True)

    # Summary
    total_time = load_time + sinogram_time + trace_time
    print("=" * 70, flush=True)
    print("COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 70, flush=True)
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)", flush=True)
    print(f"Efficiency: {efficiency:.1f}%", flush=True)
    print(f"Lines per second: {len(path)/trace_time:.2f}", flush=True)
    print(f"Graph size: 60 vertices, {60 * 59 // 2} edges", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
