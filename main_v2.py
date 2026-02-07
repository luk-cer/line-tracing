"""
Line Tracing Algorithm V2 - Main CLI Application (Graph-Based)

Usage:
    python main_v2.py --num-lines 200 --num-nodes 360 --image images/timik.jpg
    python main_v2.py -n 1000 -N 60 -i images/myimage.png -o results/
    python main_v2.py --help
"""

import click
import numpy as np
import os
import time
from app_v2 import ImageProcessorV2


@click.command()
@click.option('--num-lines', '-n', default=100, type=int, help='Number of lines to trace')
@click.option('--num-nodes', '-N', default=360, type=int, help='Number of vertices on circle (nodes in graph)')
@click.option('--image', '-i', default='images/timik.jpg', type=click.Path(exists=True), help='Path to input image')
@click.option('--output-dir', '-o', default='output', type=click.Path(), help='Output directory for results')
@click.option('--sinogram-angles', '-s', default=1000, type=int, help='Number of angles for sinogram calculation')
@click.option('--recalc-radon-every', '-r', default=10, type=int, help='Recalculate Radon transform every N lines (0 to disable)')
@click.option('--min-intensity', '-m', default=0.0, type=float, help='Minimum intensity threshold (stop when below)')
@click.option('--local-search', '-l', is_flag=True, help='Use local traversal instead of global best edge search')
@click.option('--direct-sampling', '-d', is_flag=True, help='Use direct pixel sampling instead of Radon transform')
@click.option('--line-darkness', '-D', default=0.05, type=float, help='Line darkness for direct sampling (0.05 = 5%)')
@click.option('--verbose', '-v', is_flag=True, help='Print detailed progress information')
def main(num_lines, num_nodes, image, output_dir, sinogram_angles, recalc_radon_every, min_intensity, local_search, direct_sampling, line_darkness, verbose):
    """Line tracing algorithm V2 using graph-based architecture with Radon transform."""

    # Print header
    click.echo("="*70)
    click.echo(f"Line Tracing Algorithm V2 - {num_lines} Lines, {num_nodes} Nodes")
    click.echo("="*70)
    click.echo()

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Load image and calculate sinogram
    click.echo(f"Creating processor with {num_nodes} vertices (spacing: {360.0/num_nodes:.2f}° apart)...")
    start_time = time.time()

    processor = ImageProcessorV2(num_points_on_circle=num_nodes)
    processor.load(image)

    if verbose:
        click.echo(f"Image shape: {processor.processed_image.shape}")
        click.echo(f"Circle center: ({processor.center_x}, {processor.center_y})")
        click.echo(f"Circle radius: {processor.radius}")

    processor.calculate_sinogram(sinogram_angles)

    load_time = time.time() - start_time

    if verbose:
        click.echo(f"Sinogram shape: {processor.sinogram.shape}")
        num_edges = num_nodes * (num_nodes - 1) // 2
        click.echo(f"Graph size: {num_nodes} vertices, {num_edges} edges")
    click.echo(f"Load time: {load_time:.2f} seconds")
    click.echo()

    # Trace lines
    trace_start = time.time()

    if direct_sampling:
        click.echo(f"Tracing {num_lines} lines (DIRECT PIXEL SAMPLING)...")
        click.echo(f"Line darkness: {line_darkness}, Min intensity: {min_intensity}")
        path = processor.trace_lines_direct(
            max_lines=num_lines,
            min_intensity=min_intensity,
            update_every_n=1,
            line_darkness=line_darkness
        )
    else:
        avoid_text = "with" if recalc_radon_every > 0 else "without"
        search_text = "local" if local_search else "global"
        click.echo(f"Tracing {num_lines} lines ({avoid_text} radon recalculation, {search_text} search)...")
        click.echo(f"Minimum intensity threshold: {min_intensity}")
        path = processor.trace_lines(
            max_lines=num_lines,
            recalculate_radon_every_n=recalc_radon_every,
            start_at=None,
            min_intensity=min_intensity,
            use_global_search=not local_search
        )

    trace_time = time.time() - trace_start

    click.echo(f"Successfully traced {len(path)} lines in {trace_time:.2f} seconds")
    click.echo()

    # Analyze path
    phis = [p[0] for p in path]
    thetas = [p[1] for p in path]
    values = [p[2] for p in path]

    # Verify uniqueness
    unique_lines = set()
    for phi, theta in zip(phis, thetas):
        try:
            phi_exit = processor.get_exit_point(phi, theta)
            line_id = frozenset({round(phi, 1), round(phi_exit, 1)})
            unique_lines.add(line_id)
        except ValueError:
            pass

    # Calculate MSE
    click.echo("Calculating MSE...")
    mse = processor.calculate_mse(path)

    # Print statistics
    click.echo("Path statistics:")
    click.echo(f"  Total steps: {len(path)}")
    click.echo(f"  Unique lines: {len(unique_lines)}")
    click.echo(f"  Efficiency: {len(unique_lines)}/{len(path)} = {len(unique_lines)/len(path)*100:.1f}%")
    click.echo(f"  MSE vs original: {mse:.6f}")
    click.echo()

    if verbose:
        click.echo("Value statistics:")
        click.echo(f"  Starting value: {values[0]:.4f}")
        click.echo(f"  Ending value: {values[-1]:.4f}")
        click.echo(f"  Value drop: {values[0] - values[-1]:.4f} ({(values[0]-values[-1])/values[0]*100:.2f}%)")
        click.echo(f"  Min value: {min(values):.4f}")
        click.echo(f"  Max value: {max(values):.4f}")
        click.echo(f"  Mean value: {np.mean(values):.4f}")
        click.echo()

        # Quality metrics
        high_quality = sum(1 for v in values if v >= values[0] * 0.99)
        medium_quality = sum(1 for v in values if values[0] * 0.95 <= v < values[0] * 0.99)
        lower_quality = sum(1 for v in values if v < values[0] * 0.95)

        click.echo("Quality distribution:")
        click.echo(f"  High quality (>99% of max):   {high_quality:4d} lines ({high_quality/len(values)*100:5.1f}%)")
        click.echo(f"  Medium quality (95-99% of max): {medium_quality:4d} lines ({medium_quality/len(values)*100:5.1f}%)")
        click.echo(f"  Lower quality (<95% of max):   {lower_quality:4d} lines ({lower_quality/len(values)*100:5.1f}%)")
        click.echo()

    # Render visualization
    click.echo("Rendering visualization...")
    render_start = time.time()

    # Generate output filename
    image_basename = os.path.splitext(os.path.basename(image))[0]
    output_filename = f"traced_v2_{image_basename}_{num_nodes}nodes_{len(path)}lines.png"
    output_path = os.path.join(output_dir, output_filename)

    result_img = processor.draw_traversal(path, output_path=output_path)
    render_time = time.time() - render_start

    click.echo(f"Saved visualization to: {output_path}")
    if verbose:
        click.echo(f"Image size: {result_img.size}")
        click.echo(f"Render time: {render_time:.2f} seconds")
    click.echo()

    # Save statistics to text file
    if verbose:
        click.echo("Saving statistics...")

    stats_filename = f"traced_v2_{image_basename}_{num_nodes}nodes_{len(path)}lines_stats.txt"
    stats_path = os.path.join(output_dir, stats_filename)

    with open(stats_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"Line Tracing Algorithm V2 - Statistics for {len(path)} Lines\n")
        f.write("="*70 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Input image: {image}\n")
        f.write(f"  Number of nodes: {num_nodes} (spacing: {360.0/num_nodes:.2f}° apart)\n")
        f.write(f"  Graph size: {num_nodes} vertices, {num_nodes * (num_nodes - 1) // 2} edges\n")
        f.write(f"  Sinogram angles: {sinogram_angles}\n")
        f.write(f"  Recalculate Radon every: {recalc_radon_every if recalc_radon_every > 0 else 'Disabled'}\n\n")

        f.write("Performance:\n")
        f.write(f"  Load time: {load_time:.2f} seconds\n")
        f.write(f"  Trace time: {trace_time:.2f} seconds\n")
        f.write(f"  Render time: {render_time:.2f} seconds\n")
        f.write(f"  Total time: {load_time + trace_time + render_time:.2f} seconds\n")
        f.write(f"  Lines per second: {len(path)/trace_time:.2f}\n\n")

        f.write("Path Statistics:\n")
        f.write(f"  Total steps: {len(path)}\n")
        f.write(f"  Unique lines: {len(unique_lines)}\n")
        f.write(f"  Efficiency: {len(unique_lines)}/{len(path)} = {len(unique_lines)/len(path)*100:.1f}%\n")
        f.write(f"  MSE vs original: {mse:.6f}\n\n")

        f.write("Value Statistics:\n")
        f.write(f"  Starting value: {values[0]:.4f}\n")
        f.write(f"  Ending value: {values[-1]:.4f}\n")
        f.write(f"  Value drop: {values[0] - values[-1]:.4f} ({(values[0]-values[-1])/values[0]*100:.2f}%)\n")
        f.write(f"  Min value: {min(values):.4f}\n")
        f.write(f"  Max value: {max(values):.4f}\n")
        f.write(f"  Mean value: {np.mean(values):.4f}\n")
        f.write(f"  Median value: {np.median(values):.4f}\n\n")

        # Quality metrics
        high_quality = sum(1 for v in values if v >= values[0] * 0.99)
        medium_quality = sum(1 for v in values if values[0] * 0.95 <= v < values[0] * 0.99)
        lower_quality = sum(1 for v in values if v < values[0] * 0.95)

        f.write("Quality Distribution:\n")
        f.write(f"  High quality (>99% of max):   {high_quality:4d} lines ({high_quality/len(values)*100:5.1f}%)\n")
        f.write(f"  Medium quality (95-99% of max): {medium_quality:4d} lines ({medium_quality/len(values)*100:5.1f}%)\n")
        f.write(f"  Lower quality (<95% of max):   {lower_quality:4d} lines ({lower_quality/len(values)*100:5.1f}%)\n\n")

        unique_phis = len(set([round(p, 1) for p in phis]))
        unique_thetas = len(set([round(t, 1) for t in thetas]))

        f.write("Angle Coverage:\n")
        f.write(f"  Phi range: [{min(phis):.1f}deg, {max(phis):.1f}deg]\n")
        f.write(f"  Theta range: [{min(thetas):.1f}deg, {max(thetas):.1f}deg]\n")
        f.write(f"  Unique phi positions: {unique_phis}\n")
        f.write(f"  Unique theta angles: {unique_thetas}\n\n")

        # Write sample of lines
        f.write("Sample of traced lines (first 50 and last 50):\n")
        f.write(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}\n")
        f.write("-" * 50 + "\n")

        # First 50
        for i in range(min(50, len(path))):
            phi, theta, value = path[i]
            f.write(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}\n")

        if len(path) > 100:
            f.write("...\n")

            # Last 50
            for i in range(max(50, len(path)-50), len(path)):
                phi, theta, value = path[i]
                f.write(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}\n")

    click.echo(f"Saved statistics to: {stats_path}")
    click.echo()

    # Final summary
    total_time = load_time + trace_time + render_time
    click.echo("="*70)
    click.secho("COMPLETED SUCCESSFULLY", fg='green', bold=True)
    click.echo("="*70)
    click.echo(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    click.echo(f"Traced {len(unique_lines)} unique lines with {len(unique_lines)/len(path)*100:.1f}% efficiency")
    click.echo(f"Graph: {num_nodes} nodes (spacing: {360.0/num_nodes:.2f}°)")
    click.echo("="*70)


if __name__ == '__main__':
    main()
