# ChannelAlign — Multi‑Channel TIFF Registration (translation‑only, Tk GUI)

This is a simple, friendly GUI tool that **registers TIFF images across multiple channels** and exports aligned multi‑channel TIFFs, plus a **CSV log** of the corrections per channel.

> ✅ Designed for non‑programmers. No command line required.

---

## What you get
- `channel_align_tk.py` — the Tkinter app (double‑clickable if associated with Python)
- `requirements.txt` — the pip dependencies
- A GUI with two workflows:
  1. **Per‑channel folders (Batch)** — put each channel's TIFFs in its own folder. The app pairs files by sorted order across folders and aligns each set.
  2. **Multi‑channel TIFF(s)** — if your files already contain multiple channels, load them and align channels within each file.

The app logs:
- Image set/file processed
- Which **channels** were aligned and which was the **reference**
- The **correction distance** (dy, dx in pixels and |Δ| magnitude)
- The sizes used and parameters (upsample factor, interpolation order)

Outputs:
- An **aligned multi‑channel TIFF** per image (axes `'CYX'`), with optional channel names
- An `alignment_log.csv` in your chosen output folder

> **Note:** The registration is **translation‑only** (no rotation or scaling). This is ideal for correcting chromatic shifts or small XY misalignments between channels.

---

## Installation (Windows/Mac/Linux)

1) **Install Python 3.9+** (if you don't have it). From https://www.python.org/downloads/  
   When installing on Windows, check “Add Python to PATH”.

2) Open a terminal (Windows: *Command Prompt* or *PowerShell*; Mac: *Terminal*).  
   Navigate to the folder where you downloaded these files, then run:

```bash
python -m venv .venv
# Activate the environment:
# Windows (PowerShell):
. .venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> If you prefer not to use a virtual environment, you can skip the first two lines and just run `pip install -r requirements.txt`.

3) Run the app:

```bash
python channel_align_tk.py
```

> On Windows, after you’ve installed the requirements once, you can usually **double‑click** `channel_align_tk.py` to launch (depending on your file associations).
>
> Using Conda? Activate your environment (e.g. `conda activate gen-env`) and run the same command.

---

## Using the app

### Option A — Batch (Per‑channel folders)
1. Click **Add Channel** to create one row per channel (e.g., `DAPI`, `FITC`, `TRITC`).
2. For each row, choose the **Folder** containing that channel’s TIFFs. Leave the pattern as `*.tif` (or adjust).
3. Set the **Reference channel name** (must match one of your rows; default: `ch0`).
4. Choose an **Output folder** and optional suffix (default `_aligned`).
5. Keep **Crop to common overlap** checked to avoid padded edges in the output.
6. Click **Run Batch**.

**Important pairing rule:** files are paired **by sorted filename order** across the channel folders. Make sure the lists line up (same number and order of files). If counts differ, the app processes only up to the smaller count.

### Option B — Multi‑channel TIFF(s)
1. Click **Select file(s)** to pick one or more TIFFs that already contain multiple channels.
2. (Optional) Enter **channel names** (comma‑separated) to label channels in the saved TIFFs.
3. Choose the **reference channel index** (0‑based; default 0).
4. Click **Run**.

The app detects the channel axis automatically by preferring an axis whose size is between 2 and 8. If ambiguous, it falls back to the first axis.

---

## Output format

- TIFFs are saved as **(C, Y, X)** with metadata `axes='CYX'` which many tools (including ImageJ/Fiji) can read.
- If you supplied channel names, they’re stored in the TIFF metadata under the key `Channel`.
- The CSV log file `alignment_log.csv` contains per-channel rows with:
  - timestamp
  - mode (`single-set`, `single-file`, and `batch-summary`)
  - sample_id (best‑effort base of input names in batch mode)
  - channel, reference_channel
  - dy_px, dx_px, distance_px (magnitude)
  - H_used, W_used (size after initial equalization)
  - upsample_factor, interp_order, reg_error

---

## Tips & notes

- **Sub‑pixel accuracy:** Increase the **upsample factor** (e.g., 20–50) for higher precision (slower).
- **Interpolation order:** 1 = bilinear (default), 3 = bicubic (sharper, slower), 0 = nearest (keeps integers).
- If your channels differ slightly in size, the app **center‑crops** all images to the smallest H×W before alignment. This equalizes shapes and keeps things simple and robust.
- If you need to export in ImageJ style (hyperstack), you can open the saved TIFFs in Fiji and export accordingly.
- If you need rotation/scale corrections, ask and we can extend the tool with rigid/affine registration.

---

## Troubleshooting

- **“Expected 2D grayscale”** in batch mode: make sure each file is a single-plane (Y×X) grayscale TIFF per channel.
- **Files don’t pair correctly:** ensure the folder lists are in the same order when sorted (rename as needed).
- **Outputs look cropped too much:** you can uncheck “Crop to common overlap” to keep original dimensions (with padded borders).
- **Headless runs:** Add the project folder (`lipofusin-tmem/`) to `PYTHONPATH` before importing `channel_align_tk` if you automate tasks in scripts.

---

## Headless / scripted usage (optional)

All of the GUI logic is built on reusable helpers inside `channel_align_tk.py`. You can call them directly to automate workflows. For example, this snippet aligns matching 9×9 crops stored under `23/` and `23-2/`, registering to the **second channel (index 1)** of the `23-2` images, and writes an `alignment_log.csv` just like the GUI:

```python
import glob
import math
import os
from tifffile import imread
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from channel_align_tk import (
    compute_overlap_bounds,
    make_log_writer,
    natural_sort_key,
    stack_and_save_tiff,
    _timestamp,
)

root = "/path/to/lipofusin-tmem"
folder_ctx = os.path.join(root, "23")
folder_tmem = os.path.join(root, "23-2")
pattern = "*_crop_r??_c??.tiff"
files_ctx = sorted(glob.glob(os.path.join(folder_ctx, pattern)), key=natural_sort_key)
files_tmem = sorted(glob.glob(os.path.join(folder_tmem, pattern)), key=natural_sort_key)

outdir = os.path.join(folder_ctx, "aligned_to_channel1")
os.makedirs(outdir, exist_ok=True)
writer, log_file = make_log_writer(os.path.join(outdir, "alignment_log.csv"))

for idx, (f_ctx, f_tmem) in enumerate(zip(files_ctx, files_tmem), start=1):
    arr_ctx = imread(f_ctx)  # shape: (C, Y, X)
    arr_tmem = imread(f_tmem)
    shift, error, _ = phase_cross_correlation(arr_tmem[1], arr_ctx[1], upsample_factor=10)
    dy, dx = float(shift[0]), float(shift[1])
    aligned_ctx = ndimage.shift(arr_ctx, shift=(0, dy, dx), order=1, mode="constant", cval=0.0)

    stack = [aligned_ctx[0], aligned_ctx[1], arr_tmem[0], arr_tmem[1]]
    H, W = arr_ctx.shape[1:]
    y0, y1, x0, x1 = compute_overlap_bounds(H, W, [(dy, dx)] * 2 + [(0.0, 0.0)] * 2)
    stack = [ch[y0:y1, x0:x1] for ch in stack]

    out_name = f"{os.path.splitext(os.path.basename(f_ctx))[0]}_aligned.tiff"
    stack_and_save_tiff(
        os.path.join(outdir, out_name),
        stack,
        channel_names=["23_ch0_aligned", "23_ch1_aligned", "23-2_ch0_ref", "23-2_ch1_ref"],
    )

    writer.writerow([
        _timestamp(),
        "pairwise-crops",
        os.path.basename(f_ctx),
        "23_aligned",
        "23-2_ref",
        dy,
        dx,
        math.hypot(dy, dx),
        y1 - y0,
        x1 - x0,
        10,
        1,
        error,
    ])
    log_file.flush()
    print(f"[{idx}/{len(files_ctx)}] {out_name}: dy={dy:.2f}, dx={dx:.2f}")

log_file.close()
```

Feel free to adapt it to other use cases—e.g. different reference channels, additional preprocessing, or writing multi-page TIFFs.

---

## License

MIT — feel free to adapt to your workflow.
