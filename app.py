import numpy as np
from PIL import Image


class ImageProcessor:
    def __init__(self):
        """Initialize the ImageProcessor with no parameters."""
        self.image = None
        self.processed_image = None

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
    # processor.load("input_image.jpg")
    # processor.show()
    # processor.save("output_image.jpg")
