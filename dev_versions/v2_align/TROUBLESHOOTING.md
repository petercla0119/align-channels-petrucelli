# Troubleshooting Alignment Issues

## When Automatic Alignment Fails

Some images may fail to align correctly even with the hybrid method. This document explains why this happens and what you can do about it.

### Why Alignment Fails

**Common Scenarios:**

1. **Images with very small shifts (< 5 pixels)**
   - ECC may fail to converge on near-identical images
   - Phase correlation may detect spurious shifts from noise
   
2. **Different staining intensity patterns**
   - Images have similar nuclei positions but different DAPI brightness/contrast
   - Low correlation coefficient (< 0.3) despite good spatial alignment
   
3. **Wrapping artifacts**
   - Phase correlation detects shift of 1163 pixels instead of -2959 (or vice versa)
   - System auto-corrects most cases, but some remain ambiguous

### Solutions for Problem Images

#### Option 1: Try Inverse Transform

Sometimes the alignment direction needs to be swapped:

```bash
python align_channels.py \
  --fixed path/to/fixed.tif \
  --moving path/to/moving.tif \
  --output results/ \
  --method hybrid \
  --invert-transform
```

This swaps the source and destination landmarks, effectively reversing the transformation direction.

#### Option 2: Accept Identity Transform (No Alignment)

If images are already well-aligned and the system applies a suspicious shift, you can manually verify they need no transformation. The identity transform (no shift) is automatically applied when:
- ECC fails AND
- Phase correlation detects suspicious shift (> 50 pixels) AND  
- Image correlation is high (> 0.95)

#### Option 3: Adjust Suspicious Shift Threshold

The default threshold is 50 pixels. For more aggressive filtering, edit `src/registration.py`:

**Current setting:**
```python
suspicious_shift_threshold = 50  # Line ~703
```

**For stricter filtering (30 pixels):**
```python
suspicious_shift_threshold = 30
```

**For looser filtering (80 pixels):**
```python
suspicious_shift_threshold = 80
```

Lower values = more images flagged as suspicious
Higher values = more shifts accepted as valid

#### Option 4: Use Pure ECC (No Fallback)

If phase correlation is causing more problems than it solves:

```bash
python align_channels.py \
  --fixed path/to/fixed.tif \
  --moving path/to/moving.tif \
  --output results/ \
  --method ecc
```

This will fail on some images but won't give you spurious shifts.

#### Option 5: Manual Inspection & Re-processing

For images flagged with warnings like:
```
WARNING - Manual inspection STRONGLY recommended for this pair
```

**Steps:**
1. Open QC overlays in `output/qc/overlay_after.png`
2. Check if nuclei align in the overlay (cyan + magenta DAPI)
3. If misaligned:
   - Try `--invert-transform`
   - Try adjusting threshold
   - Consider manual landmark placement (future feature)

### Understanding the Warnings

**Warning: "Phase correlation shift magnitude X pixels seems unreliable"**
- Detected shift is suspiciously large given that ECC already failed
- System checking image correlation to make intelligent decision

**Warning: "High correlation (0.9X) suggests images are nearly identical"**
- Images are very similar (> 95% correlation)
- Applying identity transform (no shift)
- Likely already aligned or need sub-pixel shift

**Warning: "Moderate correlation (0.XX) suggests images differ significantly"**
- Images are quite different (< 95% correlation)
- Keeping phase correlation shift as best guess
- **MANUAL INSPECTION REQUIRED**

**Warning: "Applying identity transform (no shift) - images may already be aligned"**
- No transformation applied
- Images remain in original positions
- Check QC overlay to verify alignment quality

### Batch Mode: Finding Problem Images

After batch processing, check the log for warnings:

```bash
grep -i "manual inspection" batch_alignment_results/run.log
grep -i "seems unreliable" batch_alignment_results/run.log
```

Or look at the summary at the end of the log for failure counts.

### Statistics on Failure Rates

**Typical results with hybrid method:**
- ~95-98% success rate with good alignment
- ~2-5% require manual inspection
- ~0-2% complete failures

**Problem image characteristics:**
- Very small actual shifts (< 3 pixels)
- Different staining quality between rounds
- Very uniform DAPI regions (few features)

### Future Improvements (Not Yet Implemented)

1. **Manual landmark specification**: Provide your own control points
2. **Interactive alignment GUI**: Visual landmark selection
3. **Multi-scale ORB features**: Better feature detection for difficult images
4. **Mutual information**: Alternative to correlation-based methods

## Need Help?

If you encounter persistent alignment issues:
1. Check the QC overlays first
2. Review the transform JSON files
3. Look at the correlation values in the logs
4. Try the options above systematically
5. Document which images fail consistently

