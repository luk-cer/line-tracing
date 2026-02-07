#!/usr/bin/env python3
"""
String Art Generator
====================

Approximates a grayscale image by stretching dark threads between pins
arranged around a circle. Uses a greedy algorithm that iteratively picks
the chord reducing the most residual error.

Outputs:
    - PNG render of the string art
    - SVG vector file
    - CSV of pin connections

Usage:
    python string_art.py image.jpg [--pins 200] [--lines 3000] [--darkness 25]

Dependencies:
    numpy, Pillow, matplotlib, scikit-image
    Optional: numba (for JIT-accelerated inner loop)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.draw import line as bresenham_line
import matplotlib

matplotlib.use("Agg")  # default non-interactive; switched in preview mode
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Numba acceleration (optional)
# ---------------------------------------------------------------------------
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op decorator when Numba is not installed."""
        def wrapper(fn):
            return fn
        return wrapper


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def compute_pin_positions(num_pins, radius, center):
    """Compute (y, x) positions of pins equally spaced around a circle.

    Args:
        num_pins: Number of pins to place.
        radius: Radius of the circle in pixels.
        center: (cy, cx) center of the circle.

    Returns:
        np.ndarray of shape (num_pins, 2) with integer (row, col) coords.
    """
    angles = np.linspace(0, 2 * np.pi, num_pins, endpoint=False)
    cy, cx = center
    ys = (cy + radius * np.sin(angles)).astype(np.int32)
    xs = (cx + radius * np.cos(angles)).astype(np.int32)
    return np.column_stack([ys, xs])


def precompute_lines(pin_positions, img_shape):
    """Precompute pixel indices for every unique pin pair.

    Uses Bresenham rasterisation. Stores results in flat arrays suitable
    for fast NumPy (or Numba) advanced indexing.

    Args:
        pin_positions: (num_pins, 2) array of (row, col) pin coords.
        img_shape: (height, width) of the working image.

    Returns:
        line_rows: 1-D int32 array of all row indices, concatenated.
        line_cols: 1-D int32 array of all col indices, concatenated.
        offsets: 2-D int64 array of shape (num_pins, num_pins) where
                 offsets[a, b] is the start index into line_rows/line_cols
                 for the chord from pin a to pin b.
        lengths: 2-D int32 array of shape (num_pins, num_pins) giving
                 the number of pixels in each chord.
    """
    n = len(pin_positions)
    h, w = img_shape

    # First pass: collect all lines and measure total pixels
    all_rows = []
    all_cols = []
    lengths = np.zeros((n, n), dtype=np.int32)

    for a in range(n):
        for b in range(a + 1, n):
            rr, cc = bresenham_line(
                pin_positions[a, 0], pin_positions[a, 1],
                pin_positions[b, 0], pin_positions[b, 1],
            )
            # Clip to image bounds
            mask = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            rr, cc = rr[mask], cc[mask]
            all_rows.append(rr)
            all_cols.append(cc)
            lengths[a, b] = len(rr)
            lengths[b, a] = len(rr)

    line_rows = np.concatenate(all_rows).astype(np.int32)
    line_cols = np.concatenate(all_cols).astype(np.int32)

    # Build offset lookup — offsets[a][b] points into flat arrays
    offsets = np.zeros((n, n), dtype=np.int64)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            offsets[a, b] = idx
            offsets[b, a] = idx  # symmetric
            idx += lengths[a, b]

    return line_rows, line_cols, offsets, lengths


# ---------------------------------------------------------------------------
# Core greedy algorithm
# ---------------------------------------------------------------------------

def _greedy_loop_numpy(residual, line_rows, line_cols, offsets, lengths,
                       num_pins, num_lines, line_darkness, min_distance,
                       callback=None, callback_interval=100):
    """Greedy string art loop — pure NumPy implementation.

    At each step, evaluates all chords from the current pin and picks
    the one crossing the darkest residual area. Then subtracts thread
    darkness along that chord.

    Args:
        residual: 2-D float64 array (0 = white, 255 = black).
        line_rows: Flat array of precomputed row indices.
        line_cols: Flat array of precomputed col indices.
        offsets: (N, N) start-offset array into line_rows/cols.
        lengths: (N, N) pixel-count array.
        num_pins: Number of pins.
        num_lines: Maximum number of threads to place.
        line_darkness: Intensity to subtract per thread (0–255 scale).
        min_distance: Minimum pin index distance to skip short chords.
        callback: Optional callable(step, connections, residual) for preview.
        callback_interval: Call callback every this many steps.

    Returns:
        connections: list of (from_pin, to_pin) tuples.
    """
    connections = []
    current_pin = 0

    for step in range(num_lines):
        best_score = -1.0
        best_pin = -1

        for candidate in range(num_pins):
            # Skip self and nearby pins
            dist = min(abs(candidate - current_pin),
                       num_pins - abs(candidate - current_pin))
            if dist < min_distance:
                continue

            length = lengths[current_pin, candidate]
            if length == 0:
                continue

            off = offsets[current_pin, candidate]
            rr = line_rows[off: off + length]
            cc = line_cols[off: off + length]

            # Score = mean residual intensity along the chord
            score = np.mean(residual[rr, cc])
            if score > best_score:
                best_score = score
                best_pin = candidate

        if best_pin < 0 or best_score <= 0:
            print(f"  Stopping early at step {step}: no beneficial chord found.")
            break

        # Subtract darkness along chosen chord
        off = offsets[current_pin, best_pin]
        length = lengths[current_pin, best_pin]
        rr = line_rows[off: off + length]
        cc = line_cols[off: off + length]
        residual[rr, cc] = np.maximum(residual[rr, cc] - line_darkness, 0.0)

        connections.append((current_pin, best_pin))
        current_pin = best_pin

        # Progress / callback
        if callback and (step + 1) % callback_interval == 0:
            callback(step + 1, connections, residual)

    return connections


# Optional Numba-accelerated version (used automatically if available)
if HAS_NUMBA:
    @njit(cache=True)
    def _greedy_loop_numba(residual, line_rows, line_cols, offsets, lengths,
                           num_pins, num_lines, line_darkness, min_distance):
        """JIT-compiled greedy loop for maximum speed.

        Same logic as _greedy_loop_numpy but runs ~10-50x faster under Numba.

        Returns:
            connections_from: int32 array of source pins.
            connections_to: int32 array of destination pins.
            count: number of connections actually made.
        """
        conn_from = np.empty(num_lines, dtype=np.int32)
        conn_to = np.empty(num_lines, dtype=np.int32)
        current_pin = 0
        count = 0

        for step in range(num_lines):
            best_score = -1.0
            best_pin = -1

            for candidate in range(num_pins):
                dist = min(abs(candidate - current_pin),
                           num_pins - abs(candidate - current_pin))
                if dist < min_distance:
                    continue

                length = lengths[current_pin, candidate]
                if length == 0:
                    continue

                off = offsets[current_pin, candidate]
                total = 0.0
                for i in range(length):
                    total += residual[line_rows[off + i], line_cols[off + i]]
                score = total / length

                if score > best_score:
                    best_score = score
                    best_pin = candidate

            if best_pin < 0 or best_score <= 0:
                break

            off = offsets[current_pin, best_pin]
            length = lengths[current_pin, best_pin]
            for i in range(length):
                r = line_rows[off + i]
                c = line_cols[off + i]
                val = residual[r, c] - line_darkness
                residual[r, c] = val if val > 0.0 else 0.0

            conn_from[count] = current_pin
            conn_to[count] = best_pin
            count += 1
            current_pin = best_pin

        return conn_from, conn_to, count


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_string_art(pin_positions, connections, img_size, thread_alpha=0.05):
    """Render the string art result as a PIL Image with antialiased lines.

    Args:
        pin_positions: (N, 2) array of (row, col) pin positions.
        connections: List of (from_pin, to_pin) tuples.
        img_size: (height, width) of the output image.
        thread_alpha: Opacity of each thread (0.0–1.0).

    Returns:
        PIL.Image in RGB mode.
    """
    h, w = img_size
    canvas = np.ones((h, w), dtype=np.float64)  # 1.0 = white

    # Accumulate darkness
    darkness_per_line = thread_alpha
    for from_pin, to_pin in connections:
        rr, cc = bresenham_line(
            pin_positions[from_pin, 0], pin_positions[from_pin, 1],
            pin_positions[to_pin, 0], pin_positions[to_pin, 1],
        )
        mask = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr, cc = rr[mask], cc[mask]
        canvas[rr, cc] = np.maximum(canvas[rr, cc] - darkness_per_line, 0.0)

    img_array = (canvas * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode="L").convert("RGB")


def render_antialiased(pin_positions, connections, img_size, line_width=1,
                       thread_color=(0, 0, 0, 15)):
    """Render with Pillow's antialiased line drawing for smoother output.

    Args:
        pin_positions: (N, 2) array of (row, col) pin positions.
        connections: List of (from_pin, to_pin) tuples.
        img_size: (height, width).
        line_width: Width of each thread line in pixels.
        thread_color: RGBA color tuple for each thread.

    Returns:
        PIL.Image in RGB mode.
    """
    h, w = img_size
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for from_pin, to_pin in connections:
        y0, x0 = pin_positions[from_pin]
        y1, x1 = pin_positions[to_pin]
        draw.line([(int(x0), int(y0)), (int(x1), int(y1))],
                  fill=thread_color, width=line_width)

    canvas = Image.alpha_composite(canvas, overlay)
    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# SVG Export
# ---------------------------------------------------------------------------

def export_svg(pin_positions, connections, img_size, output_path,
               stroke_opacity=0.05, stroke_width=0.5):
    """Export the string art as an SVG file.

    Args:
        pin_positions: (N, 2) array of (row, col) pin positions.
        connections: List of (from_pin, to_pin) tuples.
        img_size: (height, width).
        output_path: Path to write the SVG file.
        stroke_opacity: Opacity of each thread line.
        stroke_width: Width of each thread line.
    """
    h, w = img_size
    cx, cy = w / 2, h / 2
    radius = min(w, h) / 2 - 2

    lines_svg = []
    for from_pin, to_pin in connections:
        y0, x0 = pin_positions[from_pin]
        y1, x1 = pin_positions[to_pin]
        lines_svg.append(
            f'  <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
            f'stroke="black" stroke-width="{stroke_width}" '
            f'stroke-opacity="{stroke_opacity}"/>'
        )

    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="white"/>
  <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#ccc" stroke-width="1"/>
{chr(10).join(lines_svg)}
</svg>"""

    Path(output_path).write_text(svg_content)


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_connections_csv(connections, output_path):
    """Save pin connections as a CSV file.

    Args:
        connections: List of (from_pin, to_pin) tuples.
        output_path: Path to write the CSV file.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "from_pin", "to_pin"])
        for i, (a, b) in enumerate(connections):
            writer.writerow([i + 1, a, b])


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def prepare_image(image_path, size=500):
    """Load an image, convert to grayscale, resize, and crop to a circle.

    The image is resized so the shorter side equals `size`, then
    center-cropped to a square, and finally masked to a circle.

    Args:
        image_path: Path to the input image.
        size: Output dimension (square, size x size pixels).

    Returns:
        np.ndarray of shape (size, size) with float64 values 0–255.
            Pixels outside the circle are set to 0 (white in residual space).
    """
    img = Image.open(image_path).convert("L")

    # Resize preserving aspect ratio, then center crop
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    # Invert: dark areas in image → high values in residual
    arr = 255.0 - np.array(img, dtype=np.float64)

    # Mask outside the circle to zero
    cy, cx = size / 2, size / 2
    radius = size / 2 - 1
    yy, xx = np.mgrid[:size, :size]
    circle_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    arr[~circle_mask] = 0.0

    return arr


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_string_art(image_path, num_pins=200, num_lines=3000,
                        line_darkness=25, min_distance=20, img_size=500,
                        preview=False, output_dir=None):
    """Run the full string art generation pipeline.

    Args:
        image_path: Path to source image.
        num_pins: Number of pins around the circle.
        num_lines: Maximum number of threads to place.
        line_darkness: Intensity subtracted per thread (0–255).
        min_distance: Minimum pin-index gap (skips short chords).
        img_size: Working resolution (square, in pixels).
        preview: If True, show a live matplotlib preview.
        output_dir: Directory for output files. Defaults to image directory.

    Returns:
        connections: List of (from_pin, to_pin) tuples.
        result_image: PIL.Image of the rendered string art.
    """
    if output_dir is None:
        output_dir = Path(image_path).parent
    output_dir = Path(output_dir)
    stem = Path(image_path).stem

    # --- Prepare image ---
    print(f"Loading and preparing image: {image_path}")
    residual = prepare_image(image_path, img_size)
    h, w = residual.shape

    # --- Compute pin positions ---
    center = (h // 2, w // 2)
    radius = h // 2 - 2
    pin_positions = compute_pin_positions(num_pins, radius, center)
    print(f"Placed {num_pins} pins on circle (radius={radius}px)")

    # --- Precompute line pixels ---
    print("Precomputing line pixel indices...")
    t0 = time.time()
    line_rows, line_cols, offsets, lengths = precompute_lines(
        pin_positions, (h, w)
    )
    total_pixels = len(line_rows)
    num_pairs = num_pins * (num_pins - 1) // 2
    print(f"  {num_pairs} pin pairs, {total_pixels:,} total pixels "
          f"({time.time() - t0:.1f}s)")

    # --- Setup preview ---
    callback = None
    if preview:
        matplotlib.use("TkAgg")
        plt.ion()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("String Art — Live Preview")

        # Show original (inverted back for display)
        original_display = 255 - residual
        axes[0].imshow(original_display, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original")
        axes[0].axis("off")

        im_residual = axes[1].imshow(residual, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Residual")
        axes[1].axis("off")

        # Start with blank white canvas for string art preview
        preview_canvas = np.ones((h, w), dtype=np.float64) * 255
        im_result = axes[2].imshow(preview_canvas, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("String Art (0 lines)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.pause(0.01)

        def _preview_callback(step, connections, residual_state):
            """Update the live preview display."""
            im_residual.set_data(residual_state)
            axes[1].set_title(f"Residual (step {step})")

            # Re-render string art for display
            canvas = np.ones((h, w), dtype=np.float64) * 255
            for fa, fb in connections:
                rr, cc = bresenham_line(
                    pin_positions[fa, 0], pin_positions[fa, 1],
                    pin_positions[fb, 0], pin_positions[fb, 1],
                )
                valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rr, cc = rr[valid], cc[valid]
                canvas[rr, cc] = np.maximum(canvas[rr, cc] - 25, 0)

            im_result.set_data(canvas)
            axes[2].set_title(f"String Art ({step} lines)")
            fig.canvas.draw_idle()
            plt.pause(0.001)

        callback = _preview_callback

    # --- Run greedy algorithm ---
    print(f"Running greedy algorithm ({num_lines} lines, darkness={line_darkness})...")
    t0 = time.time()

    if HAS_NUMBA and not preview:
        # Use Numba-accelerated version (no callback support)
        print("  Using Numba JIT acceleration")
        conn_from, conn_to, count = _greedy_loop_numba(
            residual, line_rows, line_cols, offsets, lengths,
            num_pins, num_lines, float(line_darkness), min_distance
        )
        connections = list(zip(conn_from[:count].tolist(),
                               conn_to[:count].tolist()))
    else:
        if preview:
            print("  Using NumPy with live preview")
        else:
            print("  Using NumPy (install numba for ~10-50x speedup)")
        connections = _greedy_loop_numpy(
            residual, line_rows, line_cols, offsets, lengths,
            num_pins, num_lines, line_darkness, min_distance,
            callback=callback, callback_interval=50
        )

    elapsed = time.time() - t0
    print(f"  Placed {len(connections)} threads in {elapsed:.1f}s "
          f"({len(connections) / max(elapsed, 0.001):.0f} lines/sec)")

    # --- Render final result ---
    print("Rendering final image...")
    # Thread alpha must match what the algorithm subtracted per thread
    # so that the rendered image matches the residual the algorithm optimised for.
    thread_alpha = line_darkness / 255.0
    print(f"  thread_alpha={thread_alpha:.4f} (matching algorithm darkness={line_darkness}/255)")
    result_image = render_string_art(
        pin_positions, connections, (h, w),
        thread_alpha=thread_alpha
    )

    # --- Save outputs ---
    png_path = output_dir / f"{stem}_string_art.png"
    svg_path = output_dir / f"{stem}_string_art.svg"
    csv_path = output_dir / f"{stem}_connections.csv"

    result_image.save(png_path)
    print(f"  PNG saved: {png_path}")

    export_svg(pin_positions, connections, (h, w), svg_path)
    print(f"  SVG saved: {svg_path}")

    export_connections_csv(connections, csv_path)
    print(f"  CSV saved: {csv_path}")

    if preview:
        plt.ioff()
        plt.show()

    return connections, result_image


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for the string art generator."""
    parser = argparse.ArgumentParser(
        description="Generate string art from an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--pins", type=int, default=200,
                        help="Number of pins around the circle")
    parser.add_argument("--lines", type=int, default=3000,
                        help="Maximum number of threads")
    parser.add_argument("--darkness", type=int, default=25,
                        help="Darkness per thread (0-255)")
    parser.add_argument("--min-distance", type=int, default=20,
                        help="Minimum pin index distance")
    parser.add_argument("--size", type=int, default=500,
                        help="Working image resolution (square)")
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview during generation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as input)")

    args = parser.parse_args()

    generate_string_art(
        image_path=args.image,
        num_pins=args.pins,
        num_lines=args.lines,
        line_darkness=args.darkness,
        min_distance=args.min_distance,
        img_size=args.size,
        preview=args.preview,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
