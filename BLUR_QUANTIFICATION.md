# Blur Quantification Integration

## Overview

The pipeline now includes **blur quantification** using two methods:

1. **FFT High-Frequency Energy** — Global sharpness metric
2. **Spatially Varying Blur Estimation** — Dense blur map with per-pixel sigma values

## Usage

### Basic Pipeline (without blur quantification)

```bash
python pipeline.py \
    --video input.mp4 \
    --efl 24.0 \
    --pitch 0.00112 \
    --shift-x 0.15 \
    --shift-y -0.05 \
    --out-dir ./output
```

### With Blur Quantification at Specific Point

To measure blur at edge position (e.g., x=1920, y=1080):

```bash
python pipeline.py \
    --video input.mp4 \
    --efl 24.0 \
    --pitch 0.00112 \
    --shift-x 0.15 \
    --shift-y -0.05 \
    --blur-query-x 1920 \
    --blur-query-y 1080 \
    --out-dir ./output
```

### Advanced Options

```bash
python pipeline.py \
    --video input.mp4 \
    --efl 24.0 \
    --pitch 0.00112 \
    --shift-x 0.15 \
    --shift-y -0.05 \
    --blur-query-x 1920 \
    --blur-query-y 1080 \
    --blur-model path/to/model.pth \  # Optional: use trained blur-kernel-estimation model
    --blur-window 32 \                 # Sliding window size (default: 32)
    --blur-stride 16 \                 # Stride for sliding window (default: 16)
    --out-dir ./output
```

## Output

### Console Output Example

```
[Stage 4] Blur quantification at query point (1920, 1080) …

=== Spatial Blur Report [blur_before_correction] ===
  Mean blur       : 3.245 px
  95th pct blur   : 8.123 px
  Max  blur       : 15.678 px
  Edge/Centre     : 3.21x
  Query at (1920, 1080): 12.000 px

=== Spatial Blur Report [blur_after_correction] ===
  Mean blur       : 1.123 px
  95th pct blur   : 2.456 px
  Max  blur       : 4.789 px
  Edge/Centre     : 1.45x
  Query at (1920, 1080): 2.000 px

====================================================
  Blur at (1920, 1080):
    Before : 12.0 px
    After  : 2.0 px
    Reduction : 83.3%
====================================================
```

### Generated Files

When blur quantification is enabled:

- `blur_heatmap_before.png` — Blur heatmap before correction
- `blur_heatmap_after.png` — Blur heatmap after correction
- `report.json` — Extended JSON report with blur metrics

### JSON Report Structure

```json
{
  "camera_params": { ... },
  "before_correction": { ... },
  "after_correction": { ... },
  "reduction_pct": { ... },

  "blur_before_correction": {
    "mean_sigma_px": 3.245,
    "p95_sigma_px": 8.123,
    "max_sigma_px": 15.678,
    "edge_centre_ratio": 3.21
  },
  "blur_after_correction": {
    "mean_sigma_px": 1.123,
    "p95_sigma_px": 2.456,
    "max_sigma_px": 4.789,
    "edge_centre_ratio": 1.45
  },
  "blur_reduction_pct": {
    "mean": 65.4,
    "p95": 69.8,
    "max": 69.5
  },
  "blur_query_point": {
    "coords": [1920, 1080],
    "before_px": 12.0,
    "after_px": 2.0,
    "reduction_pct": 83.3
  }
}
```

## Methods

### Method A: FFT High-Frequency Energy

- **Use case**: Quick global sharpness assessment
- **Output**: Single scalar value (0-100, higher = sharper)
- **Speed**: Fast (~10ms per frame)

```python
from blur_quantifier import quantify_blur_fft

report = quantify_blur_fft(frame)
print(f"Sharpness: {report.sharpness_score:.2f}/100")
```

### Method B: Spatially Varying Blur Estimation

- **Use case**: Detailed spatial blur analysis (edge vs center)
- **Output**: Dense H×W blur map (sigma in pixels)
- **Speed**: Moderate (~100-500ms per frame, depends on window size)

```python
from blur_quantifier import quantify_blur_spatial

report = quantify_blur_spatial(
    frame,
    window_size=32,
    stride=16,
    query_coords=(1920, 1080)
)
print(f"Blur at edge: {report.query_sigma:.1f} px")
```

## Integration with blur-kernel-estimation

To use the trained model from [arunpatro/blur-kernel-estimation](https://github.com/arunpatro/blur-kernel-estimation):

1. Clone the repository:
   ```bash
   git clone https://github.com/arunpatro/blur-kernel-estimation
   cd blur-kernel-estimation
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision
   ```

3. Download or train the model to get `model.pth`

4. Integrate model loading in `blur_quantifier.py`:
   ```python
   def _quantify_blur_with_model(frame, model_path, query_coords):
       import torch
       from blur_model import BlurNet  # Import model architecture

       model = BlurNet()
       model.load_state_dict(torch.load(model_path))
       model.eval()

       # Run inference
       with torch.no_grad():
           sigma_map = model(preprocess(frame))

       return _make_spatial_report(sigma_map, query_sigma, query_coords)
   ```

5. Run pipeline with model:
   ```bash
   python pipeline.py \
       --video input.mp4 \
       --blur-query-x 1920 \
       --blur-query-y 1080 \
       --blur-model blur-kernel-estimation/model.pth
   ```

## Fallback Method

When no model is provided, the system uses **Laplacian variance** in sliding windows:

- Computes local sharpness via Laplacian operator
- Inverts to get "blur sigma" (lower variance = higher blur)
- Empirical scaling: `sigma = 50.0 / (variance + 10.0)`

This fallback is less accurate than the trained model but requires no external dependencies.

## Performance Tips

1. **Reduce window stride** for higher resolution blur maps (slower)
2. **Increase window size** for smoother blur estimates
3. **Use model-based method** for production accuracy
4. **Use fallback method** for quick prototyping

## Citation

If using the blur-kernel-estimation model, please cite:

```bibtex
@inproceedings{patro2020blur,
  title={Spatially-Varying Blur Detection Based on Multiscale Fused and Sorted Transform Coefficients of Gradient Magnitudes},
  author={Patro, Arun and others},
  booktitle={CVPR Workshops},
  year={2020}
}
```
