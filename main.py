"""
Line Tracing Algorithm - Main CLI Application

Usage:
    python main.py --num-lines 200 --image images/timik.jpg
    python main.py -n 1000 -i images/myimage.png -o results/
    python main.py --help
"""

import click
import numpy as np
import os
import time
from app import ImageProcessor


@click.command()
@click.option('--num-lines', '-n', default=100, type=int, help='Number of lines to trace')
@click.option('--image', '-i', default='images/timik.jpg', type=click.Path(exists=True), help='Path to input image')
@click.option('--output-dir', '-o', default='output', type=click.Path(), help='Output directory for results')
@click.option('--sinogram-angles', '-s', default=1000, type=int, help='Number of angles for sinogram calculation')
@click.option('--avoid-traced/--no-avoid-traced', default=True, help='Avoid retracing lines')
@click.option('--recalc-radon-every', '-r', default=50, type=int, help='Recalculate Radon transform every N lines (0 to disable)')
@click.option('--verbose', '-v', is_flag=True, help='Print detailed progress information')
def main(num_lines, image, output_dir, sinogram_angles, avoid_traced, recalc_radon_every, verbose):
    """Line tracing algorithm using Radon transform and sinogram optimization."""

    # Print header
    click.echo("="*70)
    click.echo(f"Line Tracing Algorithm - {num_lines} Lines")
    click.echo("="*70)
    click.echo()

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Load image and calculate sinogram
    click.echo("Loading image and calculating sinogram...")
    start_time = time.time()

    processor = ImageProcessor()
    processor.load(image)
    processor.calculate_sinogram(sinogram_angles)

    load_time = time.time() - start_time

    if verbose:
        click.echo(f"Image shape: {processor.processed_image.shape}")
        click.echo(f"Sinogram shape: {processor.sinogram.shape}")
    click.echo(f"Load time: {load_time:.2f} seconds")
    click.echo()

    # Traverse circle
    avoid_text = "with" if avoid_traced else "without"
    click.echo(f"Tracing {num_lines} lines ({avoid_text} line avoidance)...")

    trace_start = time.time()
    path = processor.traverse_circle(num_lines, avoid_traced=avoid_traced, recalc_radon_every=recalc_radon_every)
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
        unique_lines.add((round(phi, 1), round(theta, 1)))

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
    output_filename = f"traced_{image_basename}_{len(path)}_lines.png"
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

    stats_filename = f"traced_{image_basename}_{len(path)}_lines_stats.txt"
    stats_path = os.path.join(output_dir, stats_filename)

    with open(stats_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"Line Tracing Algorithm - Statistics for {len(path)} Lines\n")
        f.write("="*70 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Input image: {image}\n")
        f.write(f"  Sinogram angles: {sinogram_angles}\n")
        f.write(f"  Avoid traced lines: {avoid_traced}\n")
        f.write(f"  Recalculate Radon every: {recalc_radon_every if recalc_radon_every > 0 else 'Disabled'}\n\n")

        f.write("Performance:\n")
        f.write(f"  Load time: {load_time:.2f} seconds\n")
        f.write(f"  Trace time: {trace_time:.2f} seconds\n")
        f.write(f"  Render time: {render_time:.2f} seconds\n")
        f.write(f"  Total time: {load_time + trace_time + render_time:.2f} seconds\n")
        f.write(f"  Lines per second: {len(path)/trace_time:.1f}\n\n")

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
    click.echo("="*70)


if __name__ == '__main__':
    main()
