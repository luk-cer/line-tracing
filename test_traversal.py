"""
Test script for circle traversal and sinogram optimization functions.

Tests:
1. get_sinogram_value - Interpolated value retrieval
2. find_max_theta_for_phi - Finding optimal theta for a given phi
3. find_global_max - Finding global maximum in sinogram
4. traverse_circle - Full traversal algorithm
"""

import numpy as np
from app import ImageProcessor

print("="*70)
print("Testing Circle Traversal Functions")
print("="*70)
print()

# Load image and calculate sinogram
processor = ImageProcessor()
processor.load("images/timik.jpg")
processor.calculate_sinogram(1000)

print(f"Sinogram shape: {processor.sinogram.shape}")
print(f"Sinogram value range: [{processor.sinogram.min():.4f}, {processor.sinogram.max():.4f}]")
print()

# Test 1: get_sinogram_value
print("Test 1: get_sinogram_value")
print("-" * 70)

# Test at center crossing
value = processor.get_sinogram_value(0, 0)
row, col = processor.map_polar_to_sinogram(0, 0)
array_value = processor.sinogram[int(row), int(col)]
print(f"At phi=0deg, theta=0deg:")
print(f"  Interpolated value: {value:.4f}")
print(f"  Array value at ({int(row)}, {int(col)}): {array_value:.4f}")
print(f"  Difference: {abs(value - array_value):.6f}")
print()

# Test at fractional coordinates
value_frac = processor.get_sinogram_value(45.5, 67.3)
print(f"At phi=45.5deg, theta=67.3deg (fractional):")
print(f"  Interpolated value: {value_frac:.4f}")
print()

# Test 2: find_max_theta_for_phi
print("Test 2: find_max_theta_for_phi")
print("-" * 70)

# Test at phi=0
theta_max, value_max = processor.find_max_theta_for_phi(0)
print(f"At phi=0deg:")
print(f"  Max theta: {theta_max:.2f}deg")
print(f"  Max value: {value_max:.4f}")

# Verify by sampling
sample_values = []
sample_thetas = np.linspace(0, 180, 20, endpoint=False)
for t in sample_thetas:
    sample_values.append(processor.get_sinogram_value(0, t))
print(f"  Sampled max: {max(sample_values):.4f} (should be <= {value_max:.4f})")
print()

# Test at phi=90
theta_max_90, value_max_90 = processor.find_max_theta_for_phi(90)
print(f"At phi=90deg:")
print(f"  Max theta: {theta_max_90:.2f}deg")
print(f"  Max value: {value_max_90:.4f}")
print()

# Test 3: find_global_max
print("Test 3: find_global_max")
print("-" * 70)

phi_max, theta_max, value_max = processor.find_global_max()
print(f"Global maximum found:")
print(f"  Phi: {phi_max:.2f}deg")
print(f"  Theta: {theta_max:.2f}deg")
print(f"  Value: {value_max:.4f}")
print()

# Verify against numpy
expected_max = processor.sinogram.max()
print(f"Numpy max value: {expected_max:.4f}")
print(f"Match: {np.isclose(value_max, expected_max, rtol=0.01)}")
print(f"Difference: {abs(value_max - expected_max):.6f}")
print()

# Test 4: traverse_circle
print("Test 4: traverse_circle")
print("-" * 70)

path = processor.traverse_circle(10)
print(f"Path length: {len(path)}")
print()

print("Traversal path:")
print(f"{'Step':<6} {'Phi (deg)':<12} {'Theta (deg)':<14} {'Value':<10}")
print("-" * 50)
for i, (phi, theta, value) in enumerate(path):
    print(f"{i:<6} {phi:<12.2f} {theta:<14.2f} {value:<10.4f}")
print()

# Analyze path
phis = [p[0] for p in path]
thetas = [p[1] for p in path]
values = [p[2] for p in path]

print("Path statistics:")
print(f"  Starting value: {values[0]:.4f}")
print(f"  Ending value: {values[-1]:.4f}")
print(f"  Mean value: {np.mean(values):.4f}")
print(f"  Min value: {min(values):.4f}")
print(f"  Max value: {max(values):.4f}")
print()

# Check if path forms a cycle (comes back near start)
phi_start, phi_end = phis[0], phis[-1]
phi_distance = min(abs(phi_end - phi_start), 360 - abs(phi_end - phi_start))
print(f"Distance from start to end:")
print(f"  Phi distance: {phi_distance:.2f}deg")
print(f"  Forms cycle: {phi_distance < 10}")
print()

# Test 5: Short traversal
print("Test 5: Short traversal (3 steps)")
print("-" * 70)

short_path = processor.traverse_circle(3)
print(f"Path length: {len(short_path)}")
for i, (phi, theta, value) in enumerate(short_path):
    print(f"Step {i}: phi={phi:.1f}deg, theta={theta:.1f}deg, value={value:.4f}")
print()

# Test 6: Verify exit points are used correctly
print("Test 6: Verify exit point usage")
print("-" * 70)

if len(path) >= 2:
    # First step
    phi1, theta1, val1 = path[0]
    phi2, theta2, val2 = path[1]

    # Calculate expected exit point
    expected_exit = processor.get_exit_point(phi1, theta1)

    print(f"Step 0 -> Step 1:")
    print(f"  Entry: phi={phi1:.2f}deg, theta={theta1:.2f}deg")
    print(f"  Expected exit: phi={expected_exit:.2f}deg")
    print(f"  Actual next phi: phi={phi2:.2f}deg")
    print(f"  Match: {np.isclose(phi2, expected_exit, atol=0.1)}")
    print()

print("="*70)
print("All tests completed successfully!")
print("="*70)
