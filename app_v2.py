"""
Line Tracing Algorithm - Version 2 (Graph-Based Architecture)

This module implements a graph-based line tracing algorithm where:
- Vertices represent points on the circle perimeter
- Edges represent potential lines between vertices
- Edge attributes store radon intensity values
- Edge masks track which lines have been traced
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from skimage.transform import radon
from graph import FullyConnectedGraph


class ImageProcessorV2:
    """
    Graph-based image processor for line tracing using Radon transform.

    Attributes:
        num_vertices: Number of points on circle perimeter
        graph: FullyConnectedGraph with vertices and edges
        processed_image: Inverted, masked image for radon calculation
        original_grayscale: Backup for MSE calculation
        original_processed_image: Backup for radon recalculation
        sinogram: Radon transform result
        angles: Theta angles used in radon transform
        edge_to_theta: Pre-computed theta directions for edges
        lines_traced: Counter for number of traced lines
    """

    def __init__(self, num_points_on_circle=360):
        """
        Initialize processor with vertex count.

        Args:
            num_points_on_circle: Number of vertices on circle perimeter
                                  (360 = 1° apart, 720 = 0.5° apart, etc.)
        """
        # Image attributes
        self.image = None
        self.processed_image = None
        self.original_processed_image = None
        self.original_grayscale = None

        # Dimensions
        self.radius = None
        self.center_x = None
        self.center_y = None

        # Graph configuration
        self.num_vertices = num_points_on_circle
        self.graph = None  # Created after load()

        # Radon data
        self.sinogram = None
        self.angles = None

        # Mapping structures
        self.edge_to_theta = None  # Pre-computed theta for each edge

        # Tracking
        self.lines_traced = 0

        # Working image for direct pixel sampling (modified as lines are drawn)
        self.working_image = None

    def load(self, image_path):
        """
        Load and process an image (PRESERVED FROM app.py).

        Steps:
        1. Convert to grayscale
        2. Invert intensities (black=1, white=0)
        3. Apply circular mask
        4. Store backups for MSE and radon recalculation

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

        # Store dimensions
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

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

        # Working image for direct pixel sampling (will be modified as lines are drawn)
        self.working_image = self.processed_image.copy()

        self.image = img

        return self.processed_image

    def _initialize_graph(self):
        """
        Create graph with vertices on circle perimeter.
        Each vertex stores x, y, phi, and degree (number of connections).
        """
        # Create graph with specified attributes
        self.graph = FullyConnectedGraph(
            num_nodes=self.num_vertices,
            edge_attr_names=['intensity'],
            vert_attr_names=['x', 'y', 'phi', 'degree']
        )

        print(f"  Initializing graph with {self.num_vertices} vertices...")

        # Calculate vertex positions on circle
        for i in range(self.num_vertices):
            phi = i * 360.0 / self.num_vertices
            phi_rad = phi * np.pi / 180.0

            x = self.center_x + self.radius * np.cos(phi_rad)
            y = self.center_y + self.radius * np.sin(phi_rad)

            # Set vertex attributes
            self.graph.nodes.x[i] = x
            self.graph.nodes.y[i] = y
            self.graph.nodes.phi[i] = phi
            # Initial degree = num_vertices - 1 (connected to all others)
            self.graph.nodes.degree[i] = self.num_vertices - 1

        # Build edge-to-theta mapping
        self._build_edge_theta_mapping()

    def _build_edge_theta_mapping(self):
        """
        Pre-calculate theta direction for each edge.
        This is a geometric property that never changes.

        theta[i,j] = direction from vertex i to vertex j
        """
        print("  Building edge-to-theta mapping...")
        self.edge_to_theta = np.zeros((self.num_vertices, self.num_vertices))

        for i in range(self.num_vertices):
            x_i = self.graph.nodes.x[i]
            y_i = self.graph.nodes.y[i]

            for j in range(self.num_vertices):
                if i == j:
                    continue  # No self-loops

                x_j = self.graph.nodes.x[j]
                y_j = self.graph.nodes.y[j]

                # Calculate line direction
                dx = x_j - x_i
                dy = y_j - y_i
                theta_rad = np.arctan2(dy, dx)
                theta = (theta_rad * 180.0 / np.pi) % 180

                self.edge_to_theta[i, j] = theta

    def calculate_sinogram(self, num_angles):
        """
        Calculate Radon transform and map values to graph edges.

        Args:
            num_angles: Number of angles to use (evenly spaced between 0 and 180)

        Returns:
            Sinogram as numpy array
        """
        if self.processed_image is None:
            raise ValueError("No image. Call load() first.")

        print("Calculating sinogram...")

        # Generate angles
        self.angles = np.linspace(0, 180, num_angles, endpoint=False)

        # Compute Radon transform
        self.sinogram = radon(self.processed_image, theta=self.angles, circle=True)

        # Initialize graph if not already done
        if self.graph is None:
            self._initialize_graph()

        # Map radon values to edges
        self._map_radon_to_graph()

        return self.sinogram

    def _map_radon_to_graph(self):
        """
        Map sinogram values to graph edge intensities.
        DEPRECATED: Use _map_pixels_to_graph() for better accuracy.
        """
        print("  Mapping radon values to graph edges...")

        num_rows, num_cols = self.sinogram.shape
        phi_array = self.graph.nodes.phi

        for i in range(self.num_vertices):
            phi_i = phi_array[i]
            theta_line = self.edge_to_theta[i, :]
            theta_proj = (theta_line + 90.0) % 180.0
            phi_rad = phi_i * np.pi / 180.0
            theta_line_rad = theta_line * np.pi / 180.0
            s_normalized = np.sin(phi_rad - theta_line_rad)
            rows = (s_normalized + 1.0) * (num_rows - 1) / 2.0
            cols = theta_proj * (num_cols - 1) / 180.0
            rows = np.clip(rows, 0, num_rows - 1)
            cols = np.clip(cols, 0, num_cols - 1)
            coords = np.array([rows, cols])
            values = map_coordinates(self.sinogram, coords, order=1, mode='nearest')
            self.graph.edges.intensity[i, :] = values
            self.graph.edges.intensity[:, i] = values

    def _map_pixels_to_graph(self):
        """
        Map pixel values directly to graph edge intensities by sampling along each line.
        This is more accurate than Radon transform because:
        - Direct measurement of what the line covers
        - No coordinate conversion issues
        - Subtraction matches exactly what we measure
        """
        print("  Mapping pixel values to graph edges (direct sampling)...")

        height, width = self.working_image.shape

        # For each unique edge (i < j to avoid duplicates)
        for i in range(self.num_vertices):
            x_i = int(self.graph.nodes.x[i])
            y_i = int(self.graph.nodes.y[i])

            for j in range(i + 1, self.num_vertices):
                x_j = int(self.graph.nodes.x[j])
                y_j = int(self.graph.nodes.y[j])

                # Sample pixels along the line
                intensity = self._sample_line_intensity(x_i, y_i, x_j, y_j)

                # Store symmetrically
                self.graph.edges.intensity[i, j] = intensity
                self.graph.edges.intensity[j, i] = intensity

    def _sample_line_intensity(self, x0, y0, x1, y1):
        """
        Sample pixel values along a line and return the sum.
        Uses Bresenham's algorithm to get all pixels on the line.

        Returns:
            float: Sum of pixel intensities along the line
        """
        height, width = self.working_image.shape
        total = 0.0
        count = 0

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            if 0 <= x < width and 0 <= y < height:
                total += self.working_image[y, x]
                count += 1

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return total

    def _update_edge_intensities_after_line(self, src_vertex, dst_vertex):
        """
        After drawing a line, update the intensities of all edges that might be affected.
        For efficiency, only update edges that share a vertex with the drawn line.
        """
        # Get line endpoints
        x0 = int(self.graph.nodes.x[src_vertex])
        y0 = int(self.graph.nodes.y[src_vertex])
        x1 = int(self.graph.nodes.x[dst_vertex])
        y1 = int(self.graph.nodes.y[dst_vertex])

        # Subtract line from working image
        self._draw_line_on_working_image(x0, y0, x1, y1)

        # Update intensities for edges from src_vertex
        for j in range(self.num_vertices):
            if j != src_vertex and self.graph.e_mask[src_vertex, j]:
                x_j = int(self.graph.nodes.x[j])
                y_j = int(self.graph.nodes.y[j])
                intensity = self._sample_line_intensity(x0, y0, x_j, y_j)
                self.graph.edges.intensity[src_vertex, j] = intensity
                self.graph.edges.intensity[j, src_vertex] = intensity

        # Update intensities for edges from dst_vertex
        for j in range(self.num_vertices):
            if j != dst_vertex and self.graph.e_mask[dst_vertex, j]:
                x_j = int(self.graph.nodes.x[j])
                y_j = int(self.graph.nodes.y[j])
                intensity = self._sample_line_intensity(x1, y1, x_j, y_j)
                self.graph.edges.intensity[dst_vertex, j] = intensity
                self.graph.edges.intensity[j, dst_vertex] = intensity

    def _draw_line_on_working_image(self, x0, y0, x1, y1, subtract_value=0.05):
        """
        Subtract a line from the working image.
        Each pixel on the line has subtract_value removed.
        """
        height, width = self.working_image.shape

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            if 0 <= x < width and 0 <= y < height:
                self.working_image[y, x] = max(0, self.working_image[y, x] - subtract_value)

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def trace_lines(self, max_lines=5000, recalculate_radon_every_n=10, start_at=None,
                    min_intensity=0.0, use_global_search=True):
        """
        Main tracing algorithm - finds best lines globally (not local traversal).

        Args:
            max_lines: Maximum number of lines to trace
            recalculate_radon_every_n: Recalculate radon every N lines (0 to disable)
            start_at: Starting vertex index (ignored if use_global_search=True)
            min_intensity: Minimum intensity threshold - stop when best edge is below this
            use_global_search: If True, find globally best edge. If False, walk locally.

        Returns:
            List of (phi, theta, intensity) tuples (compatible with draw_traversal)
        """
        if self.graph is None:
            raise ValueError("Graph not initialized. Call calculate_sinogram() first.")

        path = []
        current_vertex = start_at  # Only used if use_global_search=False

        for step in range(max_lines):
            if use_global_search:
                # GLOBAL SEARCH: Find the best edge anywhere in the graph
                src_vertex, dst_vertex, intensity = self._find_global_best_edge()
            else:
                # LOCAL SEARCH: Find best edge from current vertex only
                if current_vertex is None:
                    current_vertex = self._find_max_intensity_vertex()
                dst_vertex, intensity = self._find_best_edge(current_vertex)
                src_vertex = current_vertex

            # Check if we found a valid edge
            if dst_vertex is None or intensity < min_intensity:
                if dst_vertex is None:
                    print(f"No more available edges at step {step}")
                else:
                    print(f"Stopping at step {step}: intensity {intensity:.2f} below threshold {min_intensity}")
                break

            # Get edge properties
            phi = self.graph.nodes.phi[src_vertex]
            theta = self.edge_to_theta[src_vertex, dst_vertex]

            # Record line
            path.append((phi, theta, intensity))

            # Deactivate edge (bidirectional)
            self.graph.set_edge_mask([src_vertex], [dst_vertex], False)

            # Update degrees
            self.graph.nodes.degree[src_vertex] -= 1
            self.graph.nodes.degree[dst_vertex] -= 1

            self.lines_traced += 1

            # Recalculate radon if needed
            if recalculate_radon_every_n > 0 and (step + 1) % recalculate_radon_every_n == 0:
                print(f"  Recalculating Radon at step {step + 1}...")
                self._recalculate_radon(path[-recalculate_radon_every_n:])

            # For local search, move to next vertex
            if not use_global_search:
                current_vertex = dst_vertex

            # Progress indicator every 100 lines
            if (step + 1) % 100 == 0:
                print(f"  Traced {step + 1} lines, current intensity: {intensity:.2f}")

        return path

    def trace_lines_direct(self, max_lines=5000, min_intensity=1.0, update_every_n=1,
                           line_darkness=0.05):
        """
        Trace lines using direct pixel sampling (no Radon transform).

        This approach:
        1. Samples pixels directly along each potential line
        2. Picks the line with highest total darkness
        3. Subtracts the line from working image
        4. Repeats

        Args:
            max_lines: Maximum number of lines to trace
            min_intensity: Stop when best line intensity falls below this
            update_every_n: Update edge intensities every N lines (1 = always, higher = faster but less accurate)
            line_darkness: How much to subtract from each pixel when drawing (0.05 = 5%)

        Returns:
            List of (phi, theta, intensity) tuples
        """
        if self.graph is None:
            self._initialize_graph()

        # Reset working image
        self.working_image = self.processed_image.copy()

        # Initial mapping of all edge intensities
        print("  Initial pixel sampling for all edges...")
        self._map_pixels_to_graph()

        path = []
        print(f"Starting direct pixel tracing (line_darkness={line_darkness})...")

        for step in range(max_lines):
            # Find globally best edge
            src_vertex, dst_vertex, intensity = self._find_global_best_edge()

            if dst_vertex is None or intensity < min_intensity:
                if dst_vertex is None:
                    print(f"No more available edges at step {step}")
                else:
                    print(f"Stopping at step {step}: intensity {intensity:.2f} below threshold {min_intensity}")
                break

            # Get edge properties
            phi = self.graph.nodes.phi[src_vertex]
            theta = self.edge_to_theta[src_vertex, dst_vertex]

            # Record line
            path.append((phi, theta, intensity))

            # Deactivate edge
            self.graph.set_edge_mask([src_vertex], [dst_vertex], False)

            # Draw line on working image and update affected edge intensities
            x0 = int(self.graph.nodes.x[src_vertex])
            y0 = int(self.graph.nodes.y[src_vertex])
            x1 = int(self.graph.nodes.x[dst_vertex])
            y1 = int(self.graph.nodes.y[dst_vertex])

            # Subtract line from working image
            self._draw_line_on_working_image(x0, y0, x1, y1, subtract_value=line_darkness)

            # Update edge intensities (full or partial update based on update_every_n)
            if update_every_n == 1 or (step + 1) % update_every_n == 0:
                # Full update of all active edges
                self._map_pixels_to_graph()

            self.lines_traced += 1

            # Progress indicator
            if (step + 1) % 100 == 0:
                print(f"  Traced {step + 1} lines, current intensity: {intensity:.2f}")

        return path

    def _find_max_intensity_vertex(self):
        """
        Find vertex with outgoing edge having maximum intensity.
        Only considers active edges.

        Returns:
            int: Vertex index with max intensity edge
        """
        # Apply mask to get only active edges
        active_intensities = self.graph.edges.intensity * self.graph.e_mask

        # Find maximum
        max_idx = np.unravel_index(
            np.argmax(active_intensities),
            active_intensities.shape
        )

        return max_idx[0]  # Return source vertex

    def _find_global_best_edge(self):
        """
        Find the globally best active edge anywhere in the graph.
        This is the key improvement over local traversal.

        Returns:
            tuple: (src_vertex, dst_vertex, intensity) or (None, None, None)
        """
        # Get all edge intensities with mask applied
        masked_intensities = np.where(
            self.graph.e_mask,
            self.graph.edges.intensity,
            -np.inf
        )

        # Find global maximum
        max_idx = np.unravel_index(
            np.argmax(masked_intensities),
            masked_intensities.shape
        )

        src_vertex, dst_vertex = max_idx
        intensity = masked_intensities[src_vertex, dst_vertex]

        if intensity == -np.inf:
            return None, None, None

        return src_vertex, dst_vertex, intensity

    def _find_best_edge(self, vertex):
        """
        Find active edge from vertex with maximum intensity.

        Args:
            vertex: Source vertex index

        Returns:
            tuple: (next_vertex, intensity) or (None, None) if no edges available
        """
        # Get edges from this vertex
        edge_intensities = self.graph.edges.intensity[vertex, :]
        edge_mask = self.graph.e_mask[vertex, :]

        # Mask inactive edges
        masked_intensities = edge_intensities.copy() #coppy is not needed , we can just use mask when we do argmax , lets get rid of this
        masked_intensities[~edge_mask] = -np.inf #

        # Find best edge
        best_idx = np.argmax(masked_intensities)
        best_intensity = masked_intensities[best_idx]

        if best_intensity == -np.inf:
            return None, None

        return best_idx, best_intensity

    def _recalculate_radon(self, recent_lines):
        """
        Recalculate Radon transform by subtracting recently traced lines.

        Args:
            recent_lines: List of (phi, theta, intensity) from recent steps
        """
        # Create image with only recent lines
        lines_only = np.zeros_like(self.processed_image)

        for phi, theta, _ in recent_lines:
            # Get entry point
            phi_rad = phi * np.pi / 180.0
            entry_x = int(self.center_x + self.radius * np.cos(phi_rad))
            entry_y = int(self.center_y + self.radius * np.sin(phi_rad))

            # Get exit point
            try:
                exit_phi = self.get_exit_point(phi, theta)
                exit_rad = exit_phi * np.pi / 180.0
                exit_x = int(self.center_x + self.radius * np.cos(exit_rad))
                exit_y = int(self.center_y + self.radius * np.sin(exit_rad))

                # Draw line with width=1 to match rendering (PIL uses width=1)
                self._draw_line_value(lines_only, entry_x, entry_y, exit_x, exit_y, 1.0, line_width=1)
            except ValueError:
                # Skip tangent lines
                pass

        # Calculate radon of lines and subtract
        lines_sinogram = radon(lines_only, theta=self.angles, circle=True)
        self.sinogram -= lines_sinogram

        # Remap to graph
        self._map_radon_to_graph()

    def get_exit_point(self, phi, theta):
        """
        Get the polar angle of exit point where a line crosses the circle.
        PRESERVED FROM app.py

        Args:
            phi: Angle of entry point on unit circle (degrees, 0-360)
            theta: Angle of line direction (degrees, 0-180)

        Returns:
            float: Angle of exit point in degrees [0, 360)

        Raises:
            ValueError: If line is tangent to circle (doesn't cross through)
        """
        # Normalize angles
        phi = phi % 360
        theta = theta % 180

        # Convert to radians
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

    def draw_traversal(self, path, output_path=None):
        """
        Draw traversal path as black lines on white background.
        PRESERVED FROM app.py

        Args:
            path: List of (phi, theta, value) tuples from trace_lines
            output_path: Optional path to save image (if None, displays image)

        Returns:
            PIL Image object with drawn lines
        """
        if self.processed_image is None:
            raise ValueError("No processed image available. Call load() first.")

        # Get image dimensions
        height, width = self.processed_image.shape

        # Create white background
        result_img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(result_img)

        # Draw each line in the path
        for phi, theta, value in path:
            # Get entry point on circle
            phi_rad = phi * np.pi / 180.0
            entry_x = self.center_x + self.radius * np.cos(phi_rad)
            entry_y = self.center_y + self.radius * np.sin(phi_rad)

            # Get exit point on circle
            try:
                phi_exit = self.get_exit_point(phi, theta)
                phi_exit_rad = phi_exit * np.pi / 180.0
                exit_x = self.center_x + self.radius * np.cos(phi_exit_rad)
                exit_y = self.center_y + self.radius * np.sin(phi_exit_rad)

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
        PRESERVED FROM app.py

        Args:
            path: List of (phi, theta, value) tuples from trace_lines

        Returns:
            float: MSE value
        """
        if self.original_grayscale is None:
            raise ValueError("No original grayscale image available. Call load() first.")

        # Render lines on white background
        rendered = self.draw_traversal(path, output_path=None)

        # Convert PIL image to numpy array and normalize
        rendered_array = np.array(rendered.convert('L'), dtype=np.float32) / 255.0

        # Calculate MSE (both images are [0, 1] range)
        mse = np.mean((rendered_array - self.original_grayscale) ** 2)

        return float(mse)

    def _draw_line_value(self, array, x0, y0, x1, y1, value, line_width=2):
        """
        Draw a line on given array with specified value.
        Uses Bresenham's line algorithm.
        PRESERVED FROM app.py

        Args:
            array: Numpy array to draw on
            x0, y0: Start coordinates
            x1, y1: End coordinates
            value: Value to set pixels to
            line_width: Width of line in pixels
        """
        height, width = array.shape

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Set pixel and surrounding pixels to specified value for line width
            for dx_offset in range(-line_width, line_width + 1):
                for dy_offset in range(-line_width, line_width + 1):
                    px = x0 + dx_offset
                    py = y0 + dy_offset
                    if 0 <= px < width and 0 <= py < height:
                        array[py, px] = value

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
