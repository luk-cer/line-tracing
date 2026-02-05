import numpy as np
from app import ImageProcessor

# Create processor and load image
processor = ImageProcessor()
processor.load("images/timik.jpg")
processor.calculate_sinogram(1000)

print(f"Sinogram shape: {processor.sinogram.shape}")
print(f"Num rows: {processor.sinogram.shape[0]}, Num cols: {processor.sinogram.shape[1]}")
print()

# Test 1: Center crossing (phi=0, theta=0)
row, col = processor.map_polar_to_sinogram(0, 0)
print(f"Test 1 - phi=0deg, theta=0deg (center crossing):")
print(f"  Result: row={row:.2f}, col={col:.2f}")
print(f"  Expected: row ~ {processor.sinogram.shape[0]/2:.2f}, col ~ 0")
print()

# Test 2: Center crossing (phi=90, theta=90)
row, col = processor.map_polar_to_sinogram(90, 90)
print(f"Test 2 - phi=90deg, theta=90deg (center crossing):")
print(f"  Result: row={row:.2f}, col={col:.2f}")
print(f"  Expected: row ~ {processor.sinogram.shape[0]/2:.2f}, col ~ {processor.sinogram.shape[1]/2:.2f}")
print()

# Test 3: Tangent line (phi=0, theta=90)
row, col = processor.map_polar_to_sinogram(0, 90)
print(f"Test 3 - phi=0deg, theta=90deg (tangent, r=-1):")
print(f"  Result: row={row:.2f}, col={col:.2f}")
print(f"  Expected: row ~ 0 (top edge), col ~ {processor.sinogram.shape[1]/2:.2f}")
print()

# Test 4: Tangent line (phi=90, theta=0)
row, col = processor.map_polar_to_sinogram(90, 0)
print(f"Test 4 - phi=90deg, theta=0deg (tangent, r=1):")
print(f"  Result: row={row:.2f}, col={col:.2f}")
print(f"  Expected: row ~ {processor.sinogram.shape[0]:.2f} (bottom edge), col ~ 0")
print()

# Test 5: Angle normalization (phi=360, theta=180)
row, col = processor.map_polar_to_sinogram(360, 180)
print(f"Test 5 - phi=360deg, theta=180deg (normalized to 0, 0):")
print(f"  Result: row={row:.2f}, col={col:.2f}")
row_check, col_check = processor.map_polar_to_sinogram(0, 0)
print(f"  Same as phi=0deg, theta=0deg: row={row_check:.2f}, col={col_check:.2f}")
print(f"  Match: {np.isclose(row, row_check) and np.isclose(col, col_check)}")
print()

# Test 6: Fractional coordinates
row, col = processor.map_polar_to_sinogram(45.5, 67.3)
print(f"Test 6 - phi=45.5deg, theta=67.3deg (fractional angles):")
print(f"  Result: row={row:.4f}, col={col:.4f}")
print(f"  Coordinates are floats: {isinstance(row, (float, np.floating)) and isinstance(col, (float, np.floating))}")
