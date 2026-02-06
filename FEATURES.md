# New Features Summary

## MSE (Mean Squared Error) Calculation

**What it does:**
- Compares rendered lines against the original grayscale image
- Provides a quantitative measure of how well the traced lines represent the original image
- Lower MSE = better approximation of the original

**How it works:**
- Renders all traced lines on white background
- Converts to grayscale [0, 1] range
- Calculates MSE: `mean((rendered - original)²)`

**Example output:**
```
MSE vs original: 0.292202
```

## Radon Transform Recalculation

**What it does:**
- Periodically erases traced lines from the processed image
- Recalculates the Radon transform on the updated image
- Helps find lines that were previously hidden by already-traced lines

**How it works:**
1. Every N lines (default: 50), the algorithm:
   - Resets the processed image from backup
   - Erases the last N traced lines by setting pixels to 0 (white)
   - Recalculates the Radon transform
2. Continues tracing with the updated sinogram

**Configuration:**
```bash
# Default: recalculate every 50 lines
python main.py -n 200

# Custom interval
python main.py -n 200 -r 30    # Every 30 lines
python main.py -n 200 -r 100   # Every 100 lines

# Disable recalculation
python main.py -n 200 -r 0
```

## Dual Tracking System

The algorithm now uses **both** tracking methods:

1. **Exact tracking** (traced_lines set)
   - Tracks (row, col) coordinates in sinogram
   - Prevents exact duplicates
   - Fast lookup

2. **Image-based tracking** (Radon recalculation)
   - Erases lines from processed image
   - Updates sinogram to reflect traced content
   - Finds lines previously hidden by other lines

This hybrid approach ensures:
- No exact duplicate lines (via set tracking)
- Discovery of hidden/overlapping lines (via image updates)
- Robust line detection across the full image

## Impact on Results

### Without Radon Recalculation (`-r 0`)
```
100 lines, no recalculation:
- Value drop: 5-8%
- Quality: 85-100% high quality
- Speed: ~10 seconds trace time
```

### With Radon Recalculation (`-r 50`)
```
100 lines, recalc every 50:
- Value drop: 25-35% (expected - finding weaker lines)
- Quality: 50% high quality, 50% medium quality
- Speed: ~60 seconds trace time (recalc overhead)
- MSE: 0.29
```

## Performance Considerations

### Recalculation Overhead
- Each Radon transform takes ~24 seconds (1000 angles)
- Recommended intervals:
  - Fast: `-r 0` (no recalc)
  - Balanced: `-r 50` (default)
  - Thorough: `-r 25`
  - Very thorough: `-r 10`

### MSE Calculation
- Minimal overhead (~0.1 seconds)
- Calculated once after all lines traced
- Always enabled

## Best Practices

### For Speed (Quick Preview)
```bash
python main.py -n 100 -r 0
# Fast, finds strongest lines only
```

### For Balanced Results (Recommended)
```bash
python main.py -n 200 -r 50 -v
# Good balance of quality and speed
```

### For Comprehensive Tracing
```bash
python main.py -n 1000 -r 25 -v
# Thorough, finds hidden lines
```

### For Maximum Detail
```bash
python main.py -n 3000 -r 10 -s 2000 -v
# Very thorough, high sinogram resolution
# Warning: Very slow (~30 minutes)
```

## Understanding the Metrics

### Efficiency
- Shows % of unique lines traced
- Always 100% with line avoidance enabled
- **Note:** Not a quality metric - just tracks duplicates

### MSE (Mean Squared Error)
- **Primary quality metric**
- Measures reconstruction accuracy
- Lower is better
- Typical values:
  - 0.1-0.2: Excellent reconstruction
  - 0.2-0.3: Good reconstruction
  - 0.3-0.4: Moderate reconstruction
  - >0.4: Poor reconstruction

### Value Statistics
- Shows sinogram values for traced lines
- Higher values = stronger/more prominent lines
- Value drop indicates exploring weaker lines

### Quality Distribution
- High (>99%): Very strong lines
- Medium (95-99%): Moderate lines
- Low (<95%): Weak lines

## Example Workflows

### Workflow 1: Quick Line Detection
```bash
# Find top 50 strongest lines quickly
python main.py -n 50 -r 0 -o quick_results/
```

### Workflow 2: Standard Reconstruction
```bash
# Balanced 200-line reconstruction
python main.py -n 200 -r 50 -v -o standard_results/
```

### Workflow 3: Detailed Analysis
```bash
# Comprehensive 1000-line analysis
python main.py -n 1000 -r 25 -v -o detailed_results/
```

### Workflow 4: Custom Image Processing
```bash
# Process custom image with fine-tuned parameters
python main.py -n 500 -i myimage.jpg -r 30 -s 1500 -v -o myimage_results/
```

## Command Line Reference

```
Options:
  -n, --num-lines INTEGER         Number of lines to trace [default: 100]
  -i, --image PATH                Input image path [default: images/timik.jpg]
  -o, --output-dir PATH           Output directory [default: output]
  -s, --sinogram-angles INTEGER   Sinogram resolution [default: 1000]
  -r, --recalc-radon-every INT    Radon recalc interval [default: 50]
  --avoid-traced / --no-avoid     Enable/disable line avoidance [default: True]
  -v, --verbose                   Detailed output
  --help                          Show help message
```
