import numpy as np
from PIL import Image
from skimage.transform import radon


class ImageProcessor:
    def __init__(self):
        """Initialize the ImageProcessor with no parameters."""
        self.image = None
        self.processed_image = None
        self.sinogram = None
        self.angles = None

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

        return self.sinogram

    def map_polar_to_sinogram(self, phi, theta):
        """
        Map polar coordinates (phi, theta) to sinogram coordinates.

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
