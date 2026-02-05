# Polar Coordinate Functions - Unit Tests

Comprehensive unit tests for the ImageProcessor polar coordinate mapping functions.

## Test Coverage

### TestMapPolarToSinogram (10 tests)
Tests for mapping `(phi, theta)` coordinates to sinogram space:
- Center crossing lines at various angles
- Tangent lines (top and bottom edges)
- Angle normalization (phi and theta)
- Fractional coordinate handling
- Boundary validation
- Error handling when sinogram not computed

### TestGetExitPoint (11 tests)
Tests for finding exit angles when lines cross the circle:
- Center crossings (opposite side exits)
- Non-center crossings
- Symmetry (reversing direction returns to start)
- Tangent detection and error raising
- Angle normalization
- Exit angle range validation

### TestIntegration (2 tests)
Integration tests combining multiple functions:
- Entry/exit points mapping to same sinogram coordinates
- Opposite diameter points mapping to same row

### TestEdgeCases (3 tests)
Edge cases and boundary conditions:
- Large phi values (>360°)
- Negative phi values
- Near-tangent lines

## Running Tests

### Run all tests:
```bash
python test_polar_functions.py
```

### Run specific test class:
```bash
python -m unittest test_polar_functions.TestMapPolarToSinogram
```

### Run specific test:
```bash
python -m unittest test_polar_functions.TestGetExitPoint.test_center_crossing_zero
```

### Run with verbose output:
```bash
python -m unittest test_polar_functions -v
```

## Test Results Summary

```
Tests run: 26
Successes: 26
Failures: 0
Errors: 0
```

## Key Test Scenarios

### 1. Center Crossings
Lines passing through the circle center exit 180° opposite:
- phi=0°, theta=0° → exit at 180°
- phi=90°, theta=90° → exit at 270°

### 2. Tangent Lines
Lines tangent to circle raise ValueError (no exit point):
- phi=0°, theta=90° → ValueError
- phi=45°, theta=135° → ValueError

### 3. Sinogram Mapping
Entry and exit points of the same line map to:
- Same sinogram row (same perpendicular distance)
- Same sinogram column (same theta angle)

### 4. Angle Normalization
- phi values normalize to [0, 360): phi=360° ≡ phi=0°
- theta values normalize to [0, 180): theta=180° ≡ theta=0°

## Dependencies

- numpy
- PIL (Pillow)
- scikit-image
- unittest (built-in)

## File Structure

```
line-tracing/
├── app.py                      # Main ImageProcessor class
├── test_polar_functions.py     # Comprehensive unit tests
├── test_mapping.py             # Manual verification tests
├── test_exit_point.py          # Manual verification tests
└── README_TESTS.md            # This file
```
