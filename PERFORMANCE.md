# Performance Optimizations & Analysis

## Implemented Optimizations

### 1. Coordinate Caching
**What it does:**
- Pre-calculates sinogram coordinates for all (phi, theta) combinations
- Builds cache after initial sinogram calculation and after each Radon recalculation
- Cache contains ~360,000 entries (360 phi values × 1000 theta values)

**Impact:**
- `map_polar_to_sinogram` now O(1) lookup instead of trigonometric calculation
- Eliminates repeated calculation of `sin((phi - theta) * π/180)`
- Eliminates repeated row/col mapping calculations

**Code location:**
- Cache storage: `self.coord_cache`
- Cache building: `_build_coord_cache()` method
- Cache usage: `map_polar_to_sinogram()` method

## Performance Bottlenecks Analysis

### Current Time Consumers (Ranked)

#### 1. **Radon Transform Recalculation** ⚠️ MAJOR
- **Time per call:** ~24 seconds
- **Frequency:** Every N lines (default: every 10-50 lines)
- **Total impact:** For 5000 lines with N=10: 500 recalculations = **12,000 seconds (3.3 hours)**

**Why it's slow:**
- Computes projection at 1000 angles
- Processes entire image (1333×1048 pixels)
- Unavoidable for accuracy

**Mitigation options:**
- Increase recalc interval (trade accuracy for speed)
- Reduce sinogram_angles (trade resolution for speed)
- **Recommended:** Use `-r 50` or `-r 0` for faster runs

#### 2. **find_global_max()** ⚠️ MAJOR (when recalc enabled)
- **Time per call:** ~2-3 seconds (360 phi × 1000 theta searches)
- **Frequency:** Once per Radon recalculation
- **Total impact:** For 5000 lines with N=10: 500 calls = **1,000-1,500 seconds**

**Why it's slow:**
- Searches 360 phi values
- For each phi, searches 1000 theta values
- Total: 360,000 sinogram value lookups per call
- Includes interpolation for each lookup

**Potential optimizations:**
- Sample phi values less densely (every 5° instead of every 1°)
- Use coarse-to-fine search strategy
- Cache sinogram values (but invalidated after each recalc)

#### 3. **find_max_theta_for_phi()** 🔶 MODERATE
- **Time per call:** ~0.01 seconds (1000 theta searches)
- **Frequency:** Once per line traced
- **Total impact:** For 5000 lines: 5000 calls = **50 seconds**

**Why it's slow:**
- Loops through 1000 theta values
- For each theta:
  - Calls `map_polar_to_sinogram` (now O(1) via cache ✓)
  - Calls `get_sinogram_value` with interpolation
  - Checks traced_lines set

**Potential optimizations:**
- Pre-compute all sinogram values in a 2D array during cache building
- Use vectorized numpy operations instead of Python loops

#### 4. **Scipy map_coordinates (interpolation)** 🔶 MODERATE
- **Time per call:** ~0.00001 seconds (very fast)
- **Frequency:** Once per theta in find_max_theta_for_phi
- **Total impact:** For 5000 lines: 5,000,000 calls = **50 seconds**

**Why it's relevant:**
- Called millions of times during traversal
- Bilinear interpolation for sinogram values
- Already highly optimized (C code in scipy)

**Potential optimizations:**
- Pre-compute interpolated values for integer coordinates
- Use nearest-neighbor instead of bilinear (less accurate)

#### 5. **erase_lines_from_image()** 🟢 MINOR
- **Time per call:** ~0.1 seconds (depends on N lines to erase)
- **Frequency:** Once per Radon recalculation
- **Total impact:** For 5000 lines with N=10: 500 calls = **50 seconds**

**Why it's slow:**
- Bresenham line algorithm for each line
- Sets multiple pixels per line (line_width=2)

**Potential optimizations:**
- Use vectorized line drawing
- Reduce line_width (may miss some pixels)

#### 6. **MSE Calculation** 🟢 MINOR
- **Time per call:** ~1 second (once at end)
- **Frequency:** Once per run
- **Total impact:** Negligible

## Performance Recommendations

### For Different Use Cases

#### Quick Preview (< 1 minute)
```bash
python main.py -n 100 -r 0 -s 500
# No recalc, low resolution
# Finds strongest 100 lines quickly
```

#### Standard Run (5-10 minutes)
```bash
python main.py -n 500 -r 50 -s 1000
# Moderate recalc, standard resolution
# Good balance of quality and speed
```

#### Detailed Analysis (30-60 minutes)
```bash
python main.py -n 2000 -r 25 -s 1500
# Frequent recalc, high resolution
# Thorough line detection
```

#### Comprehensive (2-4 hours)
```bash
python main.py -n 5000 -r 10 -s 2000
# Very frequent recalc, very high resolution
# Maximum detail and accuracy
```

## Optimization Strategies

### Already Implemented ✓
1. Coordinate caching for `map_polar_to_sinogram`
2. Configurable Radon recalculation interval
3. Dual tracking (set + image-based)

### Potential Future Optimizations

#### High Impact 🚀
1. **Parallel Radon Transform**
   - Use multiple CPU cores
   - Could reduce Radon time by 4-8x
   - Libraries: `multiprocessing`, `joblib`

2. **GPU Acceleration**
   - Use CUDA/OpenCL for Radon transform
   - Could reduce Radon time by 10-100x
   - Libraries: `cupy`, `pyopencl`

3. **Vectorized find_max_theta_for_phi**
   - Pre-compute all interpolated sinogram values
   - Use numpy array operations instead of loops
   - Could reduce search time by 10-20x

4. **Adaptive Radon Recalculation**
   - Only recalc when value drop exceeds threshold
   - Skip recalc in low-density areas
   - Could reduce recalc calls by 2-5x

#### Medium Impact 📈
1. **Coarse-to-Fine find_global_max**
   - First search at 10° resolution
   - Refine around promising areas
   - Could reduce global search time by 5-10x

2. **Sinogram Value Caching**
   - Cache interpolated values between recalcs
   - Invalidate cache on Radon recalc
   - Could speed up theta searches by 2-3x

3. **Optimized Line Erasing**
   - Use scipy.ndimage.draw_line
   - Vectorized operations
   - Could reduce erase time by 3-5x

## Current Bottleneck Breakdown (5000 lines, recalc every 10)

| Operation | Time (est) | % of Total |
|-----------|------------|------------|
| Radon recalculation (500×) | 12,000s | 85% |
| find_global_max (500×) | 1,250s | 9% |
| find_max_theta_for_phi (5000×) | 50s | <1% |
| Interpolation (5M×) | 50s | <1% |
| Line erasing (500×) | 50s | <1% |
| Other | 600s | 4% |
| **Total** | **~14,000s (3.9 hours)** | **100%** |

## Conclusion

**For your 5000-line run with recalc_radon_every=10:**
- Expected time: **3-4 hours**
- Main bottleneck: Radon recalculation (85% of time)
- Current optimizations help with coordinate lookup overhead
- Further speedup requires parallelization or GPU acceleration

**Recommendation:**
- Use `-r 50` instead of `-r 10` to reduce runtime from 4 hours to **~45 minutes**
- Or use `-r 0` for fastest run at **~5 minutes** (but less thorough)
