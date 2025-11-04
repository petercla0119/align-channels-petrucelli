# ChannelAlign — Multi-Channel TIFF Registration Tools

Tools for aligning multi-channel TIFF images across imaging rounds using nuclei as spatial reference.

> ✅ **Translation-only registration** with sub-pixel precision  
> ✅ **Dual-channel QC images** (before/after) for validation  
> ✅ **Consistent Ch0=Nuclei, Ch1=Protein** labeling throughout

---

## 🚀 Quick Start

### Installation

**Requirements:** Python 3.9+ with pip

```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch GUI
python channel_align_tk.py
```

**Dependencies:** numpy, tifffile, scikit-image, scipy, matplotlib (see [requirements.txt](requirements.txt))

### Verify Tkinter (GUI requirement)

```bash
python -m tkinter
```

Should open a small test window. If not:
- **Linux:** `sudo apt-get install python3-tk`
- **macOS:** Python from python.org includes Tk
- **Windows:** Reinstall Python from python.org

---

## ⚠️ CRITICAL: Channel Naming Convention

**All tools use this consistent labeling:**

```
Channel 0 (Ch0) = Nuclei (DAPI)     → Row 0 in all QC images
Channel 1 (Ch1) = Protein marker    → Row 1 in all QC images
```

This is consistent across:
- GUI parameters
- QC image labels
- Function calls
- Documentation
- Output files

---

## 🛠️ Available Tools

### 1. GUI Application: `channel_align_tk.py`

Interactive Tkinter interface with multiple alignment modes.

```bash
python channel_align_tk.py
```

**Features:**
- Four alignment workflows (see below)
- Automatic QC generation (before/after)
- Progress tracking with cancel/stop
- CSV logging of shifts
- Per-channel TIFF exports

**Four Workflows:**

#### A. Per-Channel Folders (Batch)
- Put each channel's TIFFs in separate folders
- Files paired by sorted filename order
- Aligns channels within each set

#### B. Multi-Channel TIFF(s)
- Load files that already contain multiple channels
- Aligns channels within each file
- Auto-detects channel axis

#### C. Dataset Pair (Shared Reference)
- Two folders with matching files (e.g., different imaging rounds)
- **Use Channel 0 (nuclei) as reference**
- Apply computed shift to both channels

#### D. Split Channels
- Extract individual channels from multi-channel TIFF
- Saves each as separate grayscale file

### 2. Standalone Script: `align_using_nuclei.py`

Command-line tool for single image pairs with automatic nuclei-based alignment.

```bash
python align_using_nuclei.py fixed.tiff moving.tiff output_dir
```

**Features:**
- Uses Ch0 (nuclei) for alignment automatically
- Applies shift to both channels
- Generates comprehensive QC images
- Fallback strategies for failed alignment
- Can be imported as module

---

## 📋 Recommended Settings

### For Multi-Round Imaging (Your Use Case)

**GUI Settings:**
1. Open **"Dataset pair (Channel 0→1)"** tab
2. Set parameters:
   - **Reference channel: 0** (nuclei/DAPI)
   - **Target channel: 1** (protein marker)
   - **Upsample factor: 100** (0.01 pixel precision)
   - **Interpolation order: 3** (cubic - smoother)
   - **Crop to overlap: ✓ checked**

**Command Line:**
```bash
python channel_align_tk.py --align-channel1 \
    --fixed-dir 23 \
    --moving-dir 23-2 \
    --output-dir aligned_output \
    --ref-index 0 \
    --target-index 1 \
    --upsample 100 \
    --interp-order 3
```
Replace `23` and `23-2` with your directory of choice.

**Or use standalone script:**
```bash
python align_using_nuclei.py \
    23/23.1_Stack01-Ctx-MIP.tiff \
    23-2/23.1_Stack01-TMEM-Ctx-MIP.tiff \
    aligned_output
```
Replace `23/23.1_Stack01-Ctx-MIP.tiff ` and `23-2/23.1_Stack01-TMEM-Ctx-MIP.tiff ` with your images.


### Why Channel 0 (Nuclei) as Reference?

For multi-round imaging:
- ✅ **Nuclei positions are stable** between rounds
- ✅ **Provides reliable spatial reference**
- ✅ **Should show magenta overlay** if same field of view
- ⚠️ **Channel 1 varies** (different protein markers each round)

Expected behavior:
- Ch0 (nuclei) should align well (magenta in overlay)
- Ch1 (protein) may NOT align (different markers = expected!)

---

## 📊 Output Files

### QC Images (NEW!)

**Two QC images generated per alignment:**

1. **`*_raw_comparison.png`** - Before alignment
   ```
   Row 0: Ch0 (Nuclei) - Fixed | Moving | Overlay
   Row 1: Ch1 (Protein) - Fixed | Moving | Overlay
   ```

2. **`*_qc.png`** - After alignment
   ```
   Row 0: Ch0 (Nuclei) - Fixed | Aligned | Overlay
   Row 1: Ch1 (Protein) - Fixed | Aligned | Overlay
   ```

**Overlay color interpretation:**
- 🟣 **Magenta** (red + blue) = Perfect overlap ✓
- 🔴 **Red** = Signal only in moving/aligned image
- 🔵 **Blue** = Signal only in fixed image

**Compare the two QC images to verify alignment improved!**

### Other Outputs

- **Composite TIFF:** `*_aligned.tif` (C,Y,X format, axes='CYX')
- **Per-channel TIFFs:** `*_ch0.tif`, `*_ch1.tif` (grayscale, 2D)
- **Alignment log:** `alignment_log.csv` or `alignment_channel1_log.csv`
  - Contains: dy, dx, distance, error, parameters
- **QC directory:** `qc/` folder with PNG overlays

---

## 📖 Detailed Usage

### Option A: Batch (Per-Channel Folders)

1. Click **Add Channel** for each channel (e.g., DAPI, FITC, TRITC)
2. Set **Name** and **Folder** for each channel
3. Set **Reference channel name** (must match one of your rows)
4. Choose **Output folder** and suffix
5. Click **Run Batch**

**Important:** Files are paired by sorted filename order. Ensure:
- Same number of files in each folder
- Filenames sort in matching order

### Option B: Multi-Channel TIFF(s)

1. Click **Select file(s)** to pick multi-channel TIFFs
2. (Optional) Enter **channel names** (comma-separated)
3. Set **reference channel index** (0 for nuclei)
4. Choose **Output folder**
5. Click **Run**

Auto-detects channel axis (prefers axis size 2-8).

### Option C: Dataset Pair (Multi-Round Imaging)

**Use this for your data!**

1. Set **Dataset A** (fixed/reference, e.g., `23/`)
2. Set **Dataset B** (moving, e.g., `23-2/`)
3. Set **filename pattern** (default: `*.tif`)
4. Set **Reference channel: 0** (nuclei)
5. Set **Target channel: 1** (protein)
6. Choose **Output folder**
7. Click **Run pair alignment**

**Outputs per pair:**
- 4-channel stack: [fixed_ch0, fixed_ch1, aligned_ch0, aligned_ch1]
- Per-channel grayscale TIFFs
- Raw comparison QC (before)
- Aligned QC (after)
- CSV log with shifts

### Option D: Split Channels

1. Choose **Input file** (multi-channel TIFF)
2. (Optional) Enter **channel names**
3. Choose **Output folder**
4. Click **Run split**

Saves each channel as `<basename>_chName.tif`.

---

## 🔧 Advanced Usage

### Command-Line Interface

**Folder pair alignment:**
```bash
python channel_align_tk.py --align-channel1 \
    --fixed-dir 4 \
    --moving-dir 4-2 \
    --output-dir pair_outputs \
    --ref-index 0 \
    --target-index 1 \
    --upsample 100 \
    --interp-order 3
```

# TODO: make note about 4/4-2

**Single file pair:**
```bash
python channel_align_tk.py --align-files \
    --fixed-file 4/sample1.tiff \
    --moving-file 4-2/sample1.tiff \
    --align-files-outdir aligned_pair \
    --ref-index 0 \
    --target-index 1
```

**Split channels:**
```bash
python channel_align_tk.py --split-channels \
    --split-input sample.tif \
    --split-output-dir split_output
```

Add `--no-gui` to suppress GUI, or `--with-gui` to launch GUI after command.

### Python API

Import functions for scripting:

```python
from channel_align_tk import align_channel1_between_dirs

# Align two directories (nuclei-based)
align_channel1_between_dirs(
    fixed_dir="23",
    moving_dir="23-2",
    out_dir="aligned_output",
    pattern="*.tiff",
    ref_index=0,        # Use Ch0 (nuclei)
    target_index=1,     # Align Ch1 (protein)
    upsample_factor=100,
    interpolation_order=3
)
```
# TODO: make note about 23/23-2


Or use standalone script:

```python
from align_using_nuclei import align_pair_using_nuclei

result = align_pair_using_nuclei(
    fixed_path="23/sample.tiff",
    moving_path="23-2/sample.tiff",
    output_dir="aligned_output",
    upsample_factor=100,
    interpolation_order=3,
    generate_qc=True,
    verbose=True
)

print(f"Shift: dy={result['dy']:.3f}, dx={result['dx']:.3f}")
```

# TODO: make note about 23/23-2


---

## 💡 Tips & Notes

### Alignment Parameters

- **Upsample factor:** Higher = more precise but slower
  - 10 = 0.1 pixel precision (default, fast)
  - 100 = 0.01 pixel precision (recommended for high quality)
  - 200 = 0.005 pixel precision (slow, diminishing returns)

- **Interpolation order:** 
  - 0 = nearest neighbor (preserves integers)
  - 1 = bilinear (default, fast)
  - 3 = bicubic (recommended for smooth results)

- **Crop to overlap:** Remove padded edges after alignment
  - ✓ Checked (recommended): No black borders
  - ☐ Unchecked: Keeps original size with padding

### Data Preparation

- **File pairing:** Ensure filenames sort in matching order
  - Good: `sample01_*.tiff`, `sample02_*.tiff`
  - Bad: Random names that don't sort consistently

- **Size differences:** Images auto-cropped to common size (center crop)

- **Channel detection:** Axis with size 2-8 assumed to be channels

### Understanding Results

**Check raw comparison QC first!**

If Ch0 (nuclei) overlay is **magenta:**
- ✅ Images already well-aligned
- ✅ Same field of view confirmed
- ✅ Alignment may not be necessary

If Ch0 (nuclei) overlay is **red/blue:**
- ⚠️ Significant misalignment present
- ⚠️ Verify same field of view
- ⚠️ Check if alignment improves after processing

If alignment **fails (error=1.0):**
- Images might already be perfectly aligned
- Images might not correspond (different fields)
- Try higher upsample factor or center region only

---

## ⚠️ Limitations

### Translation-Only Alignment

**Can correct:**
- ✅ X-Y shifts (translation)
- ✅ Chromatic aberration (if translation-only)
- ✅ Stage drift between rounds

**Cannot correct:**
- ❌ Rotation
- ❌ Scaling/magnification differences  
- ❌ Non-rigid deformation (tissue warping)
- ❌ Local distortions
- ❌ Z-axis misalignment

### When Translation Isn't Enough

Remaining misalignment after optimal translation may be due to:

1. **Non-rigid tissue deformation** between rounds
2. **Optical effects** (PSF differences, chromatic aberration)
3. **Biological changes** in sample
4. **Focal plane differences**

For these cases, you would need:
- Affine registration (adds rotation + scale)
- Non-rigid registration (e.g., optical flow, B-splines)
- Manual curation

**Some misalignment is unavoidable with translation-only**

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Expected 2D grayscale"** | Each file must be single-plane (Y×X) per channel in batch mode |
| **Files don't pair correctly** | Rename files so they sort in matching order |
| **Outputs too cropped** | Uncheck "Crop to common overlap" |
| **High error (1.0)** | Check raw QC - might already be aligned! |
| **Huge shift (>1000px)** | Likely alignment failed; verify same field of view |
| **No QC images** | Install matplotlib: `pip install matplotlib` |
| **GUI doesn't open** | Install/verify tkinter (see installation section) |
| **Ch1 doesn't align** | Expected if different protein markers! Check Ch0 instead |

---

## 📚 Additional Documentation

Detailed documentation available:

| File | Purpose |
|------|---------|
| **[CHANNEL_NAMING_STANDARD.md](CHANNEL_NAMING_STANDARD.md)** | Channel definitions and consistency rules |
| **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** | Complete project overview |
| **[ALIGNMENT_SUMMARY.md](ALIGNMENT_SUMMARY.md)** | Data analysis guide |
| **[NEW_FEATURES.md](NEW_FEATURES.md)** | QC function documentation |

---

## 📝 License

MIT — feel free to adapt to your workflow.
