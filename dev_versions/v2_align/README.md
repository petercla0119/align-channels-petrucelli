# Two-Channel DAPI+Protein Alignment

**Pure Python implementation** of rigid-body image registration for multi-round microscopy.  
**NO FIJI/ImageJ required** - all algorithms ported from MultiStackReg/TurboReg Java source.

## Quick Start

```bash
# Activate virtual environment
cd /Users/cspeters/projects/align-channels-petrucelli/v2_align
source venv/bin/activate

# Run alignment
python align_channels.py \
  --fixed ../images/4.1/4.1_Stack01-Ctx-MIP.tiff \
  --moving ../images/4-2.1/4.1_Stack01-TMEM-Ctx-MIP.tiff \
  --output ./results
```

## Features

✅ **Pure Python** - No FIJI/ImageJ dependency  
✅ **Automatic registration** - ORB keypoint detection + RANSAC  
✅ **Rigid-body default** - 3 landmarks, rotation + translation, NO scaling  
✅ **Quality control** - Before/after overlays and RGB composites  
✅ **Full provenance** - Transform matrices saved as JSON  

## Input Format

Two multichannel TIFF files, each with **exactly 2 channels**:
- **Channel 0**: Protein of interest (marker)
- **Channel 1**: DAPI (used for alignment)

Images must have:
- Same dimensions (H × W)
- Same pixel size/FOV
- 2D or 2D MIP

## Output Structure

```
output_dir/
├── aligned/
│   ├── fixed.tif                  # Copy of fixed image (YXC metadata)
│   └── moving_aligned.tif         # Aligned moving image (both channels transformed)
├── transforms/
│   └── rigid_transform.json       # Transform parameters, matrices, landmarks
├── composite/
│   ├── composite_RGB.png          # DAPI (blue) + fixed (green) + moving (magenta)
│   └── channels/                  # Individual channel TIFFs
│       ├── fixed_protein.tif
│       ├── fixed_DAPI.tif
│       ├── moving_aligned_protein.tif
│       └── moving_aligned_DAPI.tif
└── qc/
    ├── overlay_before.png         # Pre-alignment: DAPI cyan vs magenta
    └── overlay_after.png          # Post-alignment: DAPI cyan vs magenta
```

## Usage

### Basic Usage

```bash
python align_channels.py --fixed FIXED.tif --moving MOVING.tif --output results/
```

### With Verbose Logging

```bash
python align_channels.py \
  --fixed fixed.tif \
  --moving moving.tif \
  --output results/ \
  --verbose
```

### Translation Only (Faster)

```bash
python align_channels.py \
  --fixed fixed.tif \
  --moving moving.tif \
  --output results/ \
  --method phase  # Phase correlation: translation only, no rotation
```

### Full Affine Transform

```bash
python align_channels.py \
  --fixed fixed.tif \
  --moving moving.tif \
  --output results/ \
  --transform AFFINE  # 6-DOF: rotation, scaling, shearing, translation
```

## Transform Types

| Type | Landmarks | DOF | Parameters | Use Case |
|------|-----------|-----|------------|----------|
| **RIGID_BODY** (default) | 3 | 3 | Rotation + translation | Multi-round microscopy |
| TRANSLATION | 1 | 2 | Translation only | Small shifts |
| SCALED_ROTATION | 2 | 4 | Rotation + uniform scaling + translation | Different magnifications |
| AFFINE | 3 | 6 | Full affine (+ shear) | Maximum flexibility |

**Default is RIGID_BODY**: preserves distances, no scaling, best for microscopy.

## Registration Methods

### Feature-Based (Default)

Automatic keypoint detection using ORB + RANSAC:
- Detects ~500 rotation-invariant keypoints per image
- Robust descriptor matching with cross-checking
- RANSAC filters outliers (threshold: 2 pixels)
- Selects 3 well-distributed landmarks for RIGID_BODY

**Pros**: Handles rotation, very robust  
**Cons**: Slower (~5-10 seconds)

```bash
--method feature  # Default
```

### Phase Correlation (Fallback)

Fast Fourier-based translation estimation:
- Sub-pixel precision (10× upsampling)
- Very fast (~1 second)
- **Translation only** (no rotation)

**Pros**: Fast, simple  
**Cons**: Cannot handle rotation

```bash
--method phase
```

## Command-Line Options

```
Required:
  --fixed PATH          Fixed/reference image (2-channel TIFF)
  --moving PATH         Moving image to align (2-channel TIFF)
  --output PATH         Output directory

Optional:
  --transform TYPE      Transform type (default: RIGID_BODY)
                        Choices: TRANSLATION, RIGID_BODY, SCALED_ROTATION, AFFINE
  --method METHOD       Registration method (default: feature)
                        Choices: feature (ORB+RANSAC), phase (translation only)
  --verbose, -v         Enable INFO logging
  --debug               Enable DEBUG logging
  --version             Show version and exit
  --help                Show this help message
```

## Python API

Use programmatically from your scripts:

```python
from src.pipeline import align_two_channel_images

results = align_two_channel_images(
    fixed_path='fixed.tif',
    moving_path='moving.tif',
    output_dir='results/',
    transform_type='RIGID_BODY',  # Default
    registration_method='feature'  # ORB+RANSAC
)

print(f"Rotation: {results['rotation_angle_degrees']:.3f}°")
print(f"Landmarks: {results['num_landmarks']}")
print(f"Time: {results['elapsed_time_seconds']:.2f}s")
```

## Technical Details

### Algorithms Implemented

1. **Transformation Matrix Computation** (`src/rigid_transform.py`)
   - Ported from `MultiStackReg_.java` lines 1328-1432
   - Gaussian elimination (`invertGauss()`) lines 1435-1507
   - Supports all 4 transform types

2. **Automatic Registration** (`src/registration.py`)
   - ORB feature detection (Oriented FAST + Rotated BRIEF)
   - Descriptor matching with Hamming distance
   - RANSAC outlier rejection
   - Spatial landmark distribution

3. **Transform Application** (`src/rigid_transform.py`)
   - Uses `scipy.ndimage.affine_transform`
   - Bilinear interpolation (order=1)
   - Applied to BOTH channels (protein + DAPI)

### Dependencies

```
numpy>=1.21.0        # Array operations
scipy>=1.7.0         # Affine transforms
Pillow>=9.0.0        # PNG export
tifffile>=2021.11.2  # TIFF I/O with metadata
scikit-image>=0.19.0 # Feature detection, registration
```

Install with:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### "No keypoints detected"

**Cause**: DAPI channel is too uniform (no features)  
**Solution**: Try phase correlation instead:
```bash
--method phase
```

### "Insufficient matches"

**Cause**: Images are too different or poorly focused  
**Solutions**:
1. Check that images are from the same tissue/FOV
2. Try different transform type: `--transform AFFINE`
3. Check DAPI channel quality

### "Image shapes don't match"

**Cause**: Fixed and moving images have different dimensions  
**Solution**: Crop or resize images to match before alignment

### "Expected 2 channels, found X"

**Cause**: Input TIFF doesn't have exactly 2 channels  
**Solutions**:
1. Split multi-channel images first
2. Stack channels if currently separate
3. Check channel order (should be [protein, DAPI])

## Performance

Typical performance on 2048×2048 images:

| Method | Time | Landmarks | Use Case |
|--------|------|-----------|----------|
| Feature (RIGID_BODY) | ~5-10s | 3 (from ~50-200 matches) | Default, best quality |
| Phase (TRANSLATION) | ~1s | 1 | Fast, translation only |

## Architecture

**100% Pure Python** - no external binaries:

```
align_channels.py         # CLI entry point
├── src/
│   ├── io_utils.py       # TIFF I/O, validation
│   ├── registration.py   # ORB detection, RANSAC matching
│   ├── rigid_transform.py # Matrix computation (ported from Java)
│   ├── qc_generator.py   # Overlays, composites
│   ├── pipeline.py       # Main orchestration
│   └── transform_parser.py # MultiStackReg file parser (optional)
└── tests/                # Unit tests (future)
```

## References

### Source Code Ports
- **MultiStackReg** (EMBL-CMCI): Transformation matrix computation
- **TurboReg** (EPFL/BIG): Original algorithms by P. Thevenaz

### Publications
- Thévenaz et al., "A Pyramid Approach to Subpixel Registration Based on Intensity", IEEE TIP 1998
- Rublee et al., "ORB: An efficient alternative to SIFT or SURF", ICCV 2011
- Fischler & Bolles, "Random sample consensus: a paradigm for model fitting", CACM 1981

## License

Uses algorithms from:
- MultiStackReg (modified by Brad Busse, Kota Miura)
- TurboReg (Philippe Thévenaz, EPFL)

See individual source files for licensing details.

## Support

For questions or issues, refer to:
- `AGENT.md` - Original specification
- `PLAN.md` - Implementation details
- Source code comments

---

**Version**: 0.1.0  
**Implementation**: Pure Python (NO FIJI/ImageJ)  
**Default**: RIGID_BODY transform (3 landmarks, rotation + translation, NO scaling)

