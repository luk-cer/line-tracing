import numpy as np
from app import ImageProcessor

# Create processor (no need to load image for this test)
processor = ImageProcessor()

print("Testing get_exit_point function")
print("="*60)
print()

# Test 1: Line through center (phi=0, theta=0)
phi, theta = 0, 0
exit_angle = processor.get_exit_point(phi, theta)
print(f"Test 1 - phi={phi}deg, theta={theta}deg (through center):")
print(f"  Entry: {phi}deg, Exit: {exit_angle:.2f}deg")
print(f"  Expected: 180deg, Correct: {np.isclose(exit_angle, 180)}")
print()

# Test 2: Line through center (phi=90, theta=90)
phi, theta = 90, 90
exit_angle = processor.get_exit_point(phi, theta)
print(f"Test 2 - phi={phi}deg, theta={theta}deg (through center):")
print(f"  Entry: {phi}deg, Exit: {exit_angle:.2f}deg")
print(f"  Expected: 270deg, Correct: {np.isclose(exit_angle, 270)}")
print()

# Test 3: Line through center (phi=45, theta=45)
phi, theta = 45, 45
exit_angle = processor.get_exit_point(phi, theta)
print(f"Test 3 - phi={phi}deg, theta={theta}deg (through center):")
print(f"  Entry: {phi}deg, Exit: {exit_angle:.2f}deg")
print(f"  Expected: 225deg, Correct: {np.isclose(exit_angle, 225)}")
print()

# Test 4: Non-center crossing line
phi, theta = 30, 60
exit_angle = processor.get_exit_point(phi, theta)
entry_x = np.cos(phi * np.pi / 180)
entry_y = np.sin(phi * np.pi / 180)
exit_x = np.cos(exit_angle * np.pi / 180)
exit_y = np.sin(exit_angle * np.pi / 180)
print(f"Test 4 - phi={phi}deg, theta={theta}deg (not through center):")
print(f"  Entry: {phi}deg at ({entry_x:.3f}, {entry_y:.3f})")
print(f"  Exit: {exit_angle:.2f}deg at ({exit_x:.3f}, {exit_y:.3f})")
# Verify both points are on unit circle
print(f"  Entry on circle: {np.isclose(entry_x**2 + entry_y**2, 1)}")
print(f"  Exit on circle: {np.isclose(exit_x**2 + exit_y**2, 1)}")
print()

# Test 5: Verify symmetry - reversing direction should give same pair
phi1, theta1 = 30, 60
exit1 = processor.get_exit_point(phi1, theta1)
# Start from exit point, reverse direction (theta + 180)
exit2 = processor.get_exit_point(exit1, (theta1 + 180) % 360)
print(f"Test 5 - Symmetry check:")
print(f"  Forward: phi={phi1}deg, theta={theta1}deg -> exit={exit1:.2f}deg")
print(f"  Reverse: phi={exit1:.2f}deg, theta={(theta1+180)%360}deg -> exit={exit2:.2f}deg")
print(f"  Returns to start: {np.isclose(exit2, phi1)}")
print()

# Test 6: Tangent line (should raise ValueError)
phi, theta = 0, 90
print(f"Test 6 - phi={phi}deg, theta={theta}deg (tangent line):")
try:
    exit_angle = processor.get_exit_point(phi, theta)
    print(f"  ERROR: Should have raised ValueError but got {exit_angle}")
except ValueError as e:
    print(f"  Correctly raised ValueError: {str(e)[:60]}...")
print()

# Test 7: Another tangent line
phi, theta = 45, 135
print(f"Test 7 - phi={phi}deg, theta={theta}deg (tangent line):")
try:
    exit_angle = processor.get_exit_point(phi, theta)
    print(f"  ERROR: Should have raised ValueError but got {exit_angle}")
except ValueError as e:
    print(f"  Correctly raised ValueError")
print()

# Test 8: Angle normalization
phi, theta = 360, 180
exit_angle = processor.get_exit_point(phi, theta)
exit_check = processor.get_exit_point(0, 0)
print(f"Test 8 - Angle normalization (phi=360, theta=180):")
print(f"  Normalized exit: {exit_angle:.2f}deg")
print(f"  Same as (0, 0): {exit_check:.2f}deg")
print(f"  Match: {np.isclose(exit_angle, exit_check)}")
