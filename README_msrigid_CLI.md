# msrigid CLI — TurboReg rigid-body alignment without ImageJ

`msrigid_cli.py` is a lightweight command-line entry point that reuses the
TurboReg/MultiStackReg rigid-body math we ported to Python. It can:

- Parse existing MultiStackReg transformation files and apply them directly to
  TIFF stacks (no ImageJ runtime necessary).
- Estimate rigid transforms from shared DAPI channels via Fourier–polar + PCC
  heuristics when no `.txt` file is available.
- Emit the full artifact bundle required by AGENT (`*_aligned.tif`,
  `*_aligned_rgb.tif`, `*_qc.png`, and `*_aligned.json`).

## Requirements

```
pip install numpy tifffile scipy scikit-image matplotlib pillow
```

Matplotlib is imported lazily; if your home directory is read-only set
`MPLCONFIGDIR` to a writable location (the CLI also auto-creates a temporary
cache inside the system temp folder when needed).

## Quick start

### 1. Estimate a rigid transform from the provided DAPI MIPs

```bash
python msrigid_cli.py estimate \
  --fixed images/Lipofuscin-selected/23.1/23.1_Stack01-Ctx-MIP.tiff \
  --moving "images/TMEM Core-selected/23.1/23.1_Stack01-TMEM-Ctx-MIP.tiff" \
  --order 1 --downsample-to 640 --angle-min -30 --angle-max 30 \
  --json outputs/23_1_estimate.json
```

`--json` saves the 2×3 forward matrix, ImageJ macro fields, and estimator
metadata so you can reuse the transform later.

### 2. Align a single pair and emit all artifacts

```bash
python msrigid_cli.py single \
  --fixed images/Lipofuscin-selected/23.1/23.1_Stack01-Ctx-MIP.tiff \
  --moving "images/TMEM Core-selected/23.1/23.1_Stack01-TMEM-Ctx-MIP.tiff" \
  --fixed-ref-channel 0 --fixed-signal-channel 1 \
  --moving-ref-channel 0 --moving-signal-channel 1 \
  --outdir outputs/23_1_run --stem 23_1_stack01 \
  --order 1 --coord-base 0.0 --method turbo
```

- If you already have a MultiStackReg `.txt`, add `--msreg-file path/to/file.txt`
  (and optionally `--block 3`) to reuse the saved landmarks.
- Pass `--force-estimate` to ignore the `.txt` and recompute from the TIFFs.

The command writes:

| File | Purpose |
| --- | --- |
| `23_1_stack01_aligned.tif` | 3-channel (CYX) TIFF = `[DAPI, fixed signal, aligned moving signal]` |
| `23_1_stack01_aligned_rgb.tif` | RGB composite with DAPI→blue, fixed signal→green, moving signal→red |
| `23_1_stack01_qc.png` | 2×3 QC grid (before/after overlays) |
| `23_1_stack01_aligned.json` | Forward matrix (`m00…m12`), theta, landmarks, RMS, paths, outputs |

### 3. Batch mode (natural numeric order, 1-based indices)

```bash
python msrigid_cli.py batch \
  --fixed-dir images/Lipofuscin-selected/23.1 \
  --moving-dir "images/TMEM Core-selected/23.1" \
  --pattern "*.tiff" --start 1 --stop 5 \
  --fixed-ref-channel 0 --fixed-signal-channel 1 \
  --moving-ref-channel 0 --moving-signal-channel 1 \
  --outdir outputs/23_1_batch --method turbo
```

- The CLI sorts filenames using natural order; pair `k` uses `files[k-1]`.
- Provide `--msreg-file transforms/23_1_msreg.txt` to map `Source img: k`
  entries onto the sorted list; use `--force-estimate` to fall back to DAPI
  alignment when a block is missing.

## JSON structure

`*_aligned.json` uses the helper in `msrigid.save_transform_json()` and always
includes:

```json
{
  "transform": {
    "matrix": [[m00, m01, m02], [m10, m11, m12]],
    "macro": {"m00": …, "theta_deg": …},
    "convention": "fixed = R * moving + t"
  },
  "meta": {
    "paths": {"fixed": "…", "moving": "…"},
    "landmarks": {"mode": "msreg_file"|"synthetic_anchor", "moving": [[x,y],…]},
    "estimation": {"theta_deg": …, …} | null,
    "rms_error": 3.2e-07,
    "outputs": {"stack": "…", "rgb": "…", "qc": "…"}
  }
}
```

Use the `macro` fields directly inside ImageJ macros (`m00..m12`) if you still
need to drive other tooling.

## Validation + safety checklist

- Landmark RMS is reported for every msreg-driven block and compared against
  `--rms-max` (default `1e-6`).
- All TIFF writes go through `tifffile` with explicit output directories; the
  CLI never overwrites source data.
- No network access, subprocesses, or dynamic code execution are used.
- `apply_transform` clamps numeric ranges before casting back to integer types
  so aligned stacks respect the original bit depth.
