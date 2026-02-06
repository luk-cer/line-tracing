import numpy as np
from PIL import Image, ImageDraw
from skimage.transform import radon
from scipy.ndimage import map_coordinates


class ImageProcessor:
    def __init__(self):
        """Initialize the ImageProcessor with no parameters."""
        self.image = None
        self.processed_image = None
        self.original_processed_image = None  # Backup for Radon recalculation
        self.original_grayscale = None  # For MSE calculation
        self.sinogram = None
        self.angles = None
        self.coord_cache = {}  # Cache for (phi, theta) -> (row, col) mappings
        self.sinogram_value_cache = {}  # Cache for (phi, theta) -> value

    def load(self, image_path):
        """
        Load and process an image:
        1. Convert to grayscale/intensity
        2. Invert intensities (black=1, white=0)
        3. Apply circular mask

        Args:
            image_path: Path to the image file

        Returns:
            Processed image as numpy array
        """
        # Load image
        img = Image.open(image_path)

        # Convert to grayscale
        gray_img = img.convert('L')

        # Convert to numpy array and normalize to [0, 1]
        intensity = np.array(gray_img, dtype=np.float32) / 255.0

        # Store original grayscale for MSE calculation
        self.original_grayscale = intensity.copy()

        # Invert intensities (black=1, white=0)
        inverted = 1.0 - intensity

        # Apply circular mask
        height, width = inverted.shape
        center_y, center_x = height // 2, width // 2
        radius = min(center_x, center_y)

        # Create coordinate grids
        y, x = np.ogrid[:height, :width]

        # Calculate distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Create circular mask (1 inside circle, 0 outside)
        mask = distance <= radius

        # Apply mask to inverted image
        self.processed_image = inverted * mask

        # Store backup for Radon recalculation
        self.original_processed_image = self.processed_image.copy()

        self.image = img

        return self.processed_image

    def calculate_sinogram(self, num_angles):
        """
        Calculate and store the Radon transform sinogram of the processed image.

        Args:
            num_angles: Number of angles to use (evenly spaced between 0 and 180 degrees)

        Returns:
            Sinogram as numpy array
        """
        if self.processed_image is None:
            raise ValueError("No processed image available. Call load() first.")

        # Generate angles between 0 and 180 degrees (exclusive of 180)
        self.angles = np.linspace(0, 180, num_angles, endpoint=False)

        # Compute Radon transform
        self.sinogram = radon(self.processed_image, theta=self.angles, circle=True)

        # Build coordinate cache ONCE (geometric mapping, doesn't change)
        if not self.coord_cache:
            self._build_coord_cache()

        # Build/rebuild value cache (depends on sinogram values)
        self._build_value_cache()

        return self.sinogram

    def _build_coord_cache(self):
        """
        Pre-calculate sinogram coordinates for all phi/theta combinations.
        This is geometric mapping and only needs to be done ONCE.
        """
        self.coord_cache.clear()

        print("  Building coordinate cache...")

        num_rows, num_cols = self.sinogram.shape

        # Vectorized coordinate calculation
        all_phis = np.arange(360, dtype=np.float64)
        all_thetas = self.angles.astype(np.float64)

        phi_grid, theta_grid = np.meshgrid(all_phis, all_thetas, indexing='ij')

        phi_norm = phi_grid % 360
        theta_norm = theta_grid % 180
        r = np.sin((phi_norm - theta_norm) * np.pi / 180.0)

        rows = (r + 1) * num_rows / 2.0
        cols = theta_norm * num_cols / 180.0

        # Store coordinates in cache
        for phi_idx in range(360):
            for theta_idx, theta in enumerate(all_thetas):
                key = (round(all_phis[phi_idx], 1), round(theta, 3))
                self.coord_cache[key] = (rows[phi_idx, theta_idx], cols[phi_idx, theta_idx])

    def _build_value_cache(self):
        """
        Pre-calculate interpolated sinogram values using cached coordinates.
        This needs to be rebuilt whenever sinogram values change (Radon recalc).
        """
        self.sinogram_value_cache.clear()

        print("  Building value cache...")

        # Collect all coordinates from coord_cache for vectorized interpolation
        all_phis = np.arange(360, dtype=np.float64)
        all_thetas = self.angles.astype(np.float64)

        # Build arrays of row/col coordinates for ALL (phi, theta) pairs
        rows_list = []
        cols_list = []

        for phi_idx in range(360):
            for theta in all_thetas:
                key = (round(all_phis[phi_idx], 1), round(theta, 3))
                row, col = self.coord_cache[key]
                rows_list.append(row)
                cols_list.append(col)

        # Single vectorized interpolation call (replaces 360k individual calls!)
        coords = np.array([rows_list, cols_list])
        all_values = map_coordinates(self.sinogram, coords, order=1, mode='nearest')

        # Reshape and store in cache
        all_values = all_values.reshape(360, len(all_thetas))

        for phi_idx in range(360):
            phi_key = round(all_phis[phi_idx], 1)
            self.sinogram_value_cache[phi_key] = {
                'values': all_values[phi_idx, :],
                'thetas': all_thetas
            }

    def map_polar_to_sinogram(self, phi, theta):
        """
        Map polar coordinates (phi, theta) to sinogram coordinates.
        Uses pre-calculated cache for common values for performance.

        Args:
            phi: Angle of point on unit circle (degrees, 0-360)
            theta: Angle of line direction (degrees, 0-180)

        Returns:
            (row, col): Float coordinates in sinogram space
                - row: vertical coordinate (distance from center)
                - col: horizontal coordinate (angle)

        Raises:
            ValueError: If sinogram not computed or coordinates out of bounds
        """
        if self.sinogram is None:
            raise ValueError("No sinogram available. Call calculate_sinogram() first.")

        # Normalize angles
        phi = phi % 360
        theta = theta % 180

        # Try cache lookup first (for integer phi and theta in angles array)
        cache_key = (round(phi, 1), round(theta, 3))
        if cache_key in self.coord_cache:
            return self.coord_cache[cache_key]

        # Calculate if not in cache
        # Calculate perpendicular distance from origin to line
        # r = sin(phi - theta) for unit circle
        r = np.sin((phi - theta) * np.pi / 180.0)

        # Map theta to column index
        num_cols = self.sinogram.shape[1]
        col = theta * num_cols / 180.0

        # Map r ∈ [-1, 1] to row index
        # Center row corresponds to r=0, top row to r=-1, bottom row to r=1
        num_rows = self.sinogram.shape[0]
        row = (r + 1) * num_rows / 2.0

        # Verify coordinates are within bounds
        if not (0 <= row <= num_rows) or not (0 <= col < num_cols):
            raise ValueError(
                f"Coordinates out of bounds: row={row:.2f} (valid: [0, {num_rows})), "
                f"col={col:.2f} (valid: [0, {num_cols}))"
            )

        return (row, col)

    def get_exit_point(self, phi, theta):
        """
        Get the polar angle of the exit point where a line crosses the circle.

        Given a point at angle phi on the unit circle and a line direction theta,
        this calculates where the line exits the circle on the other side.

        Args:
            phi: Angle of entry point on unit circle (degrees, 0-360)
            theta: Angle of line direction (degrees, 0-180)

        Returns:
            float: Angle of exit point in degrees [0, 360)

        Raises:
            ValueError: If the line is tangent to the circle (doesn't cross through)
        """
        # Normalize angles
        phi = phi % 360
        theta = theta % 180

        # Convert to radians for calculation
        phi_rad = phi * np.pi / 180.0
        theta_rad = theta * np.pi / 180.0

        # Check if line is tangent (cos(phi - theta) ≈ 0)
        cos_diff = np.cos(phi_rad - theta_rad)
        if np.abs(cos_diff) < 1e-10:
            raise ValueError(
                f"Line is tangent to circle at phi={phi:.2f}deg, theta={theta:.2f}deg. "
                "No exit point exists."
            )

        # Entry point on circle
        entry_x = np.cos(phi_rad)
        entry_y = np.sin(phi_rad)

        # Direction vector of line
        dir_x = np.cos(theta_rad)
        dir_y = np.sin(theta_rad)

        # Parameter for exit point: t = -2*cos(phi - theta)
        t = -2 * cos_diff

        # Exit point coordinates
        exit_x = entry_x + t * dir_x
        exit_y = entry_y + t * dir_y

        # Calculate angle of exit point
        exit_angle_rad = np.arctan2(exit_y, exit_x)

        # Convert to degrees and normalize to [0, 360)
        exit_angle = (exit_angle_rad * 180.0 / np.pi) % 360

        return exit_angle

    def get_sinogram_value(self, phi, theta):
        """
        Get interpolated sinogram value at polar coordinates (phi, theta).

        Args:
            phi: Angle of point on unit circle (degrees, 0-360)
            theta: Angle of line direction (degrees, 0-180)

        Returns:
            float: Interpolated sinogram value at (phi, theta)

        Raises:
            ValueError: If sinogram not computed
        """
        if self.sinogram is None:
            raise ValueError("No sinogram available. Call calculate_sinogram() first.")

        # Get sinogram coordinates
        row, col = self.map_polar_to_sinogram(phi, theta)

        # Use bilinear interpolation to get value at float coordinates
        # map_coordinates expects coordinates as [[rows], [cols]]
        coords = np.array([[row], [col]])
        value = map_coordinates(self.sinogram, coords, order=1, mode='nearest')[0]

        return float(value)

    def find_max_theta_for_phi(self, phi, traced_lines=None):
        """
        Find theta that maximizes sinogram value at given phi.

        Args:
            phi: Angle of point on unit circle (degrees, 0-360)
            traced_lines: Optional set of (row, col) tuples representing
                         already traced lines to exclude from search

        Returns:
            tuple: (theta_max, value_max) where theta_max is the angle
                   that maximizes the sinogram value and value_max is
                   the maximum value found

        Raises:
            ValueError: If sinogram not computed or no valid theta found
        """
        if self.sinogram is None:
            raise ValueError("No sinogram available. Call calculate_sinogram() first.")

        if traced_lines is None:
            traced_lines = set()

        # Use pre-computed cache for fast argmax lookup
        phi_key = round(phi % 360, 1)
        cached_data = self.sinogram_value_cache.get(phi_key)

        if cached_data is not None:
            values = cached_data['values'].copy()  # Copy to avoid modifying cache
            thetas = cached_data['thetas']

            # Mask out already traced lines
            if traced_lines:
                for i, theta in enumerate(thetas):
                    row, col = self.map_polar_to_sinogram(phi, theta)
                    row_key = round(row, 1)
                    col_key = round(col, 1)
                    if (row_key, col_key) in traced_lines:
                        values[i] = -np.inf

            # Use numpy argmax to find best theta in O(1)
            max_idx = np.argmax(values)
            max_value = values[max_idx]

            if max_value == -np.inf:
                raise ValueError(f"No valid theta found for phi={phi:.2f}deg (all lines already traced)")

            return (float(thetas[max_idx]), float(max_value))

        # Fallback to loop-based method if cache not available
        max_value = -np.inf
        max_theta = 0

        for theta in self.angles:
            row, col = self.map_polar_to_sinogram(phi, theta)
            row_key = round(row, 1)
            col_key = round(col, 1)

            if (row_key, col_key) in traced_lines:
                continue

            value = self.get_sinogram_value(phi, theta)
            if value > max_value:
                max_value = value
                max_theta = theta

        if max_value == -np.inf:
            raise ValueError(f"No valid theta found for phi={phi:.2f}deg (all lines already traced)")

        return (float(max_theta), float(max_value))

    def find_global_max(self, traced_lines=None):
        """
        Find (phi, theta) with maximum sinogram value globally.

        Searches over phi values [0, 360) and finds the theta that
        maximizes the sinogram value at each phi, then returns the
        global maximum.

        Args:
            traced_lines: Optional set of (row, col) tuples representing
                         already traced lines to exclude from search

        Returns:
            tuple: (phi_max, theta_max, value_max) where phi_max and
                   theta_max are the coordinates of the maximum and
                   value_max is the maximum sinogram value

        Raises:
            ValueError: If sinogram not computed or no valid lines found
        """
        if self.sinogram is None:
            raise ValueError("No sinogram available. Call calculate_sinogram() first.")

        if traced_lines is None:
            traced_lines = set()

        # Search over phi values (sample every degree for good coverage)
        global_max_value = -np.inf
        global_max_phi = 0
        global_max_theta = 0

        for phi in range(360):
            try:
                theta_max, value_max = self.find_max_theta_for_phi(phi, traced_lines)
                if value_max > global_max_value:
                    global_max_value = value_max
                    global_max_phi = phi
                    global_max_theta = theta_max
            except ValueError:
                # No valid theta for this phi (all traced)
                continue

        # Check if we found any valid line
        if global_max_value == -np.inf:
            raise ValueError("No valid lines found (all lines already traced)")

        return (float(global_max_phi), float(global_max_theta), float(global_max_value))

    def traverse_circle(self, num_steps=10, avoid_traced=True, recalc_radon_every=50):
        """
        Traverse circle by following maximum sinogram values.

        Starts from global maximum and traverses the circle by:
        1. Crossing to exit point
        2. Finding best theta at exit point (excluding already traced lines)
        3. Every N steps, erases last N lines and recalculates Radon transform
        4. Repeating for num_steps

        Args:
            num_steps: Number of steps to traverse (default: 10)
            avoid_traced: If True, avoids retracing lines (default: True)
            recalc_radon_every: Recalculate Radon transform every N lines (default: 50)

        Returns:
            List of (phi, theta, value) tuples representing the path

        Raises:
            ValueError: If sinogram not computed, tangent line encountered,
                       or no more lines available to trace
        """
        if self.sinogram is None:
            raise ValueError("No sinogram available. Call calculate_sinogram() first.")

        if self.original_processed_image is None:
            raise ValueError("No backup image available. Call load() first.")

        # Track traced lines if avoiding them
        traced_lines = set() if avoid_traced else None

        # Find global maximum as starting point
        phi, theta, value = self.find_global_max(traced_lines)

        # Mark this line as traced
        if avoid_traced:
            row, col = self.map_polar_to_sinogram(phi, theta)
            traced_lines.add((round(row, 1), round(col, 1)))

        # Initialize path with starting point
        path = [(phi, theta, value)]

        # Traverse for num_steps
        for i in range(num_steps - 1):
            # Get exit point
            phi_exit = self.get_exit_point(phi, theta)

            # Find best theta at exit point (excluding traced lines)
            try:
                theta_new, value_new = self.find_max_theta_for_phi(phi_exit, traced_lines)
            except ValueError as e:
                # No more valid lines to trace
                print(f"Warning: Stopped after {len(path)} steps - {str(e)}")
                break

            # Mark this line as traced
            if avoid_traced:
                row, col = self.map_polar_to_sinogram(phi_exit, theta_new)
                traced_lines.add((round(row, 1), round(col, 1)))

            # Add to path
            path.append((phi_exit, theta_new, value_new))

            # Recalculate Radon transform every N lines
            if recalc_radon_every > 0 and len(path) % recalc_radon_every == 0:
                print(f"  Recalculating Radon transform at step {len(path)}...")

                # Reset processed image from backup
                self.processed_image = self.original_processed_image.copy()

                # Erase only the last N lines
                start_idx = max(0, len(path) - recalc_radon_every)
                lines_to_erase = path[start_idx:]
                self.erase_lines_from_image(lines_to_erase)

                # Recalculate sinogram with erased lines
                self.sinogram = radon(self.processed_image, theta=self.angles, circle=True)

                # Rebuild only VALUE cache (coordinates don't change)
                self._build_value_cache()

            # Update current position
            phi = phi_exit
            theta = theta_new

        return path

    def draw_traversal(self, path, output_path=None):
        """
        Draw the traversal path as black lines on white background.

        Args:
            path: List of (phi, theta, value) tuples from traverse_circle
            output_path: Optional path to save the image (if None, displays image)

        Returns:
            PIL Image object with the drawn lines
        """
        if self.processed_image is None:
            raise ValueError("No processed image available. Call load() first.")

        # Get image dimensions
        height, width = self.processed_image.shape
        center_y, center_x = height // 2, width // 2
        radius = min(center_x, center_y)

        # Create white background
        result_img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(result_img)

        # Draw each line in the path
        for phi, theta, value in path:
            # Get entry point on circle
            phi_rad = phi * np.pi / 180.0
            entry_x = center_x + radius * np.cos(phi_rad)
            entry_y = center_y + radius * np.sin(phi_rad)

            # Get exit point on circle
            try:
                phi_exit = self.get_exit_point(phi, theta)
                phi_exit_rad = phi_exit * np.pi / 180.0
                exit_x = center_x + radius * np.cos(phi_exit_rad)
                exit_y = center_y + radius * np.sin(phi_exit_rad)

                # Draw line from entry to exit (black line)
                draw.line([(entry_x, entry_y), (exit_x, exit_y)], fill='black', width=1)
            except ValueError:
                # Skip tangent lines
                pass

        # Save or show
        if output_path:
            result_img.save(output_path)
        else:
            result_img.show()

        return result_img

    def calculate_mse(self, path):
        """
        Calculate Mean Squared Error between rendered lines and original grayscale image.

        Args:
            path: List of (phi, theta, value) tuples from traverse_circle

        Returns:
            float: MSE value
        """
        if self.original_grayscale is None:
            raise ValueError("No original grayscale image available. Call load() first.")

        # Render lines on white background (same size as original)
        rendered = self.draw_traversal(path, output_path=None)

        # Convert PIL image to numpy array and normalize
        rendered_array = np.array(rendered.convert('L'), dtype=np.float32) / 255.0

        # Calculate MSE (both images are [0, 1] range)
        mse = np.mean((rendered_array - self.original_grayscale) ** 2)

        return float(mse)

    def erase_lines_from_image(self, path):
        """
        Erase traced lines from processed image by setting them to 0 (white).

        Args:
            path: List of (phi, theta, value) tuples representing lines to erase
        """
        if self.processed_image is None:
            raise ValueError("No processed image available.")

        height, width = self.processed_image.shape
        center_y, center_x = height // 2, width // 2
        radius = min(center_x, center_y)

        # For each line, draw it as white (0) on the processed image
        for phi, theta, value in path:
            try:
                # Get entry point
                phi_rad = phi * np.pi / 180.0
                entry_x = int(center_x + radius * np.cos(phi_rad))
                entry_y = int(center_y + radius * np.sin(phi_rad))

                # Get exit point
                phi_exit = self.get_exit_point(phi, theta)
                phi_exit_rad = phi_exit * np.pi / 180.0
                exit_x = int(center_x + radius * np.cos(phi_exit_rad))
                exit_y = int(center_y + radius * np.sin(phi_exit_rad))

                # Draw line by setting pixels to 0 (white) using Bresenham algorithm
                self._draw_line_on_array(entry_x, entry_y, exit_x, exit_y)
            except ValueError:
                # Skip tangent lines
                pass

    def _draw_line_on_array(self, x0, y0, x1, y1, line_width=2):
        """
        Draw a line on processed_image array by setting pixels to 0.
        Uses Bresenham's line algorithm.

        Args:
            x0, y0: Start coordinates
            x1, y1: End coordinates
            line_width: Width of line in pixels
        """
        height, width = self.processed_image.shape

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Set pixel and surrounding pixels to 0 (white) for line width
            for dx_offset in range(-line_width, line_width + 1):
                for dy_offset in range(-line_width, line_width + 1):
                    px = x0 + dx_offset
                    py = y0 + dy_offset
                    if 0 <= px < width and 0 <= py < height:
                        self.processed_image[py, px] = 0.0

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def save(self, output_path):
        """Save the processed image to a file."""
        if self.processed_image is not None:
            # Convert back to 0-255 range for saving
            output = (self.processed_image * 255).astype(np.uint8)
            Image.fromarray(output).save(output_path)
        else:
            raise ValueError("No processed image to save. Call load() first.")

    def show(self):
        """Display the processed image."""
        if self.processed_image is not None:
            # Convert back to 0-255 range for display
            output = (self.processed_image * 255).astype(np.uint8)
            Image.fromarray(output).show()
        else:
            raise ValueError("No processed image to show. Call load() first.")


# Example usage
if __name__ == "__main__":
    processor = ImageProcessor()
    processor.load("images/timik.jpg")
    processor.show()
    # processor.save("output_image.jpg")
