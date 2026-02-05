"""
Unit tests for ImageProcessor polar coordinate functions.

Tests cover:
- map_polar_to_sinogram: Mapping (phi, theta) to sinogram coordinates
- get_exit_point: Finding exit angle when a line crosses the circle
"""

import unittest
import numpy as np
from app import ImageProcessor


class TestMapPolarToSinogram(unittest.TestCase):
    """Tests for map_polar_to_sinogram method."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with dummy image and sinogram once for all tests."""
        cls.processor = ImageProcessor()
        # Create a simple test image
        cls.processor.processed_image = np.ones((100, 100), dtype=np.float32)
        # Calculate sinogram with 1000 angles
        cls.processor.calculate_sinogram(1000)

    def test_center_crossing_zero(self):
        """Test line through center at phi=0, theta=0."""
        row, col = self.processor.map_polar_to_sinogram(0, 0)
        expected_row = self.processor.sinogram.shape[0] / 2
        self.assertAlmostEqual(row, expected_row, places=1)
        self.assertAlmostEqual(col, 0.0, places=1)

    def test_center_crossing_90(self):
        """Test line through center at phi=90, theta=90."""
        row, col = self.processor.map_polar_to_sinogram(90, 90)
        expected_row = self.processor.sinogram.shape[0] / 2
        expected_col = self.processor.sinogram.shape[1] / 2
        self.assertAlmostEqual(row, expected_row, places=1)
        self.assertAlmostEqual(col, expected_col, places=1)

    def test_center_crossing_45(self):
        """Test line through center at phi=45, theta=45."""
        row, col = self.processor.map_polar_to_sinogram(45, 45)
        expected_row = self.processor.sinogram.shape[0] / 2
        expected_col = self.processor.sinogram.shape[1] / 4
        self.assertAlmostEqual(row, expected_row, places=1)
        self.assertAlmostEqual(col, expected_col, places=1)

    def test_tangent_top_edge(self):
        """Test tangent line at top edge (phi=0, theta=90)."""
        row, col = self.processor.map_polar_to_sinogram(0, 90)
        # r = sin(0 - 90) = -1, should map to row ~0
        self.assertAlmostEqual(row, 0.0, places=1)
        expected_col = self.processor.sinogram.shape[1] / 2
        self.assertAlmostEqual(col, expected_col, places=1)

    def test_tangent_bottom_edge(self):
        """Test tangent line at bottom edge (phi=90, theta=0)."""
        row, col = self.processor.map_polar_to_sinogram(90, 0)
        # r = sin(90 - 0) = 1, should map to row ~num_rows
        expected_row = self.processor.sinogram.shape[0]
        self.assertAlmostEqual(row, expected_row, places=1)
        self.assertAlmostEqual(col, 0.0, places=1)

    def test_angle_normalization_phi(self):
        """Test that phi angles normalize correctly."""
        row1, col1 = self.processor.map_polar_to_sinogram(0, 0)
        row2, col2 = self.processor.map_polar_to_sinogram(360, 0)
        self.assertAlmostEqual(row1, row2, places=5)
        self.assertAlmostEqual(col1, col2, places=5)

    def test_angle_normalization_theta(self):
        """Test that theta angles normalize correctly."""
        row1, col1 = self.processor.map_polar_to_sinogram(0, 0)
        row2, col2 = self.processor.map_polar_to_sinogram(0, 180)
        self.assertAlmostEqual(row1, row2, places=5)
        self.assertAlmostEqual(col1, col2, places=5)

    def test_fractional_coordinates(self):
        """Test that fractional angles produce float coordinates."""
        row, col = self.processor.map_polar_to_sinogram(45.5, 67.3)
        self.assertIsInstance(row, (float, np.floating))
        self.assertIsInstance(col, (float, np.floating))
        # Check that coordinates are not integers
        self.assertNotEqual(row, int(row))
        self.assertNotEqual(col, int(col))

    def test_no_sinogram_raises_error(self):
        """Test that calling without sinogram raises ValueError."""
        processor = ImageProcessor()
        with self.assertRaises(ValueError) as context:
            processor.map_polar_to_sinogram(0, 0)
        self.assertIn("No sinogram available", str(context.exception))

    def test_coordinates_within_bounds(self):
        """Test that all reasonable inputs produce valid coordinates."""
        test_cases = [
            (0, 0), (90, 90), (180, 90), (270, 45),
            (45, 30), (135, 120), (225, 150), (315, 60)
        ]
        for phi, theta in test_cases:
            row, col = self.processor.map_polar_to_sinogram(phi, theta)
            self.assertGreaterEqual(row, 0, f"Row out of bounds for phi={phi}, theta={theta}")
            self.assertLessEqual(row, self.processor.sinogram.shape[0])
            self.assertGreaterEqual(col, 0)
            self.assertLess(col, self.processor.sinogram.shape[1])


class TestGetExitPoint(unittest.TestCase):
    """Tests for get_exit_point method."""

    @classmethod
    def setUpClass(cls):
        """Set up processor once for all tests."""
        cls.processor = ImageProcessor()

    def test_center_crossing_zero(self):
        """Test line through center at phi=0 exits at 180."""
        exit_angle = self.processor.get_exit_point(0, 0)
        self.assertAlmostEqual(exit_angle, 180.0, places=5)

    def test_center_crossing_90(self):
        """Test line through center at phi=90 exits at 270."""
        exit_angle = self.processor.get_exit_point(90, 90)
        self.assertAlmostEqual(exit_angle, 270.0, places=5)

    def test_center_crossing_45(self):
        """Test line through center at phi=45 exits at 225."""
        exit_angle = self.processor.get_exit_point(45, 45)
        self.assertAlmostEqual(exit_angle, 225.0, places=5)

    def test_center_crossing_arbitrary(self):
        """Test that center crossings always exit 180 degrees opposite."""
        test_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        for angle in test_angles:
            exit_angle = self.processor.get_exit_point(angle, angle)
            expected_exit = (angle + 180) % 360
            self.assertAlmostEqual(exit_angle, expected_exit, places=5,
                                   msg=f"Failed for angle={angle}")

    def test_non_center_crossing(self):
        """Test non-center crossing produces valid exit point on circle."""
        phi, theta = 30, 60
        exit_angle = self.processor.get_exit_point(phi, theta)

        # Verify exit point is on unit circle
        exit_x = np.cos(exit_angle * np.pi / 180)
        exit_y = np.sin(exit_angle * np.pi / 180)
        radius = np.sqrt(exit_x**2 + exit_y**2)
        self.assertAlmostEqual(radius, 1.0, places=5)

    def test_symmetry(self):
        """Test that reversing direction from exit returns to entry."""
        phi1, theta1 = 30, 60
        exit1 = self.processor.get_exit_point(phi1, theta1)
        # Reverse direction (theta + 180)
        exit2 = self.processor.get_exit_point(exit1, (theta1 + 180) % 360)
        self.assertAlmostEqual(exit2, phi1, places=5)

    def test_tangent_raises_error(self):
        """Test that tangent line raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.processor.get_exit_point(0, 90)
        self.assertIn("tangent", str(context.exception).lower())

    def test_tangent_45_135_raises_error(self):
        """Test another tangent configuration raises ValueError."""
        with self.assertRaises(ValueError):
            self.processor.get_exit_point(45, 135)

    def test_angle_normalization_phi(self):
        """Test phi angle normalization."""
        exit1 = self.processor.get_exit_point(0, 0)
        exit2 = self.processor.get_exit_point(360, 0)
        self.assertAlmostEqual(exit1, exit2, places=5)

    def test_angle_normalization_theta(self):
        """Test theta angle normalization."""
        exit1 = self.processor.get_exit_point(45, 45)
        exit2 = self.processor.get_exit_point(45, 225)  # 225 = 45 + 180
        # Should give same result since theta normalizes to [0, 180)
        self.assertAlmostEqual(exit1, exit2, places=5)

    def test_exit_angle_range(self):
        """Test that exit angle is always in [0, 360)."""
        test_cases = [
            (0, 0), (90, 90), (180, 90), (270, 45),
            (45, 30), (135, 120), (225, 150), (315, 60)
        ]
        for phi, theta in test_cases:
            try:
                exit_angle = self.processor.get_exit_point(phi, theta)
                self.assertGreaterEqual(exit_angle, 0, f"Exit angle < 0 for phi={phi}, theta={theta}")
                self.assertLess(exit_angle, 360, f"Exit angle >= 360 for phi={phi}, theta={theta}")
            except ValueError:
                # Skip tangent cases
                pass


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with dummy image and sinogram."""
        cls.processor = ImageProcessor()
        cls.processor.processed_image = np.ones((100, 100), dtype=np.float32)
        cls.processor.calculate_sinogram(1000)

    def test_entry_exit_same_sinogram_coordinates(self):
        """Test that entry and exit points map to same sinogram row (same line)."""
        phi, theta = 30, 60

        # Get sinogram coordinates for entry point
        row1, col1 = self.processor.map_polar_to_sinogram(phi, theta)

        # Get exit point
        exit_phi = self.processor.get_exit_point(phi, theta)

        # Get sinogram coordinates for exit point (same theta direction)
        row2, col2 = self.processor.map_polar_to_sinogram(exit_phi, theta)

        # Same line should map to same row (perpendicular distance)
        self.assertAlmostEqual(row1, row2, places=2,
                               msg="Entry and exit should map to same sinogram row")
        # Same theta should map to same column
        self.assertAlmostEqual(col1, col2, places=2,
                               msg="Same theta should map to same sinogram column")

    def test_center_crossings_same_row(self):
        """Test that opposite points on diameter map to same sinogram row."""
        theta = 45
        # Points on opposite sides of diameter
        phi1 = theta
        phi2 = (theta + 180) % 360

        row1, col1 = self.processor.map_polar_to_sinogram(phi1, theta)
        row2, col2 = self.processor.map_polar_to_sinogram(phi2, theta)

        # Both should map to center row (r=0)
        expected_row = self.processor.sinogram.shape[0] / 2
        self.assertAlmostEqual(row1, expected_row, places=1)
        self.assertAlmostEqual(row2, expected_row, places=1)
        self.assertAlmostEqual(col1, col2, places=1)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    @classmethod
    def setUpClass(cls):
        """Set up processor with dummy image and sinogram."""
        cls.processor = ImageProcessor()
        cls.processor.processed_image = np.ones((100, 100), dtype=np.float32)
        cls.processor.calculate_sinogram(100)  # Smaller sinogram for edge testing

    def test_large_phi_values(self):
        """Test that very large phi values normalize correctly."""
        row1, col1 = self.processor.map_polar_to_sinogram(45, 30)
        row2, col2 = self.processor.map_polar_to_sinogram(45 + 720, 30)  # 720 = 2*360
        self.assertAlmostEqual(row1, row2, places=5)
        self.assertAlmostEqual(col1, col2, places=5)

    def test_negative_phi_values(self):
        """Test that negative phi values work correctly."""
        row1, col1 = self.processor.map_polar_to_sinogram(0, 0)
        row2, col2 = self.processor.map_polar_to_sinogram(-360, 0)
        self.assertAlmostEqual(row1, row2, places=5)
        self.assertAlmostEqual(col1, col2, places=5)

    def test_near_tangent_lines(self):
        """Test lines very close to tangent (but not exactly)."""
        # Slightly off from tangent
        phi, theta = 0, 89.9999
        try:
            exit_angle = self.processor.get_exit_point(phi, theta)
            # Should not raise error, should return valid angle
            self.assertGreaterEqual(exit_angle, 0)
            self.assertLess(exit_angle, 360)
        except ValueError:
            self.fail("Near-tangent line should not raise ValueError")


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMapPolarToSinogram))
    suite.addTests(loader.loadTestsFromTestCase(TestGetExitPoint))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result


if __name__ == "__main__":
    run_tests()
