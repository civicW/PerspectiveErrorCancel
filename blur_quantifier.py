"""
blur_quantifier.py
==================
Quantifies blur using two complementary methods:

  A) FFT High-Frequency Energy — global sharpness metric
  B) Spatially Varying Blur Kernel Estimation — dense blur map (sigma map)

Usage
-----
  from blur_quantifier import quantify_blur_fft, quantify_blur_spatial

  # Method A: FFT
  report_fft = quantify_blur_fft(frame)
  print(f"High-freq energy: {report_fft.hf_energy:.2f}")

  # Method B: Spatial (requires blur-kernel-estimation repo)
  report_spatial = quantify_blur_spatial(frame, model_path="path/to/model.pth")
  print(f"Blur at (32,4): {report_spatial.blur_map[4, 32]:.3f}")
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence
from pathlib import Path


@dataclass
class BlurReportFFT:
    """FFT-based blur quantification (global metric)."""
    hf_energy: float              # High-frequency energy (higher = sharper)
    hf_ratio: float               # HF / Total energy ratio
    mean_magnitude: float         # Mean FFT magnitude
    sharpness_score: float        # Normalized sharpness (0-100, higher = sharper)
    label: str = "fft_blur"

    def print_summary(self):
        print(f"\n=== FFT Blur Report [{self.label}] ===")
        print(f"  HF Energy       : {self.hf_energy:.2e}")
        print(f"  HF Ratio        : {self.hf_ratio:.4f}")
        print(f"  Sharpness Score : {self.sharpness_score:.2f} / 100")


@dataclass
class BlurReportSpatial:
    """Spatially-varying blur estimation (dense sigma map)."""
    blur_map: np.ndarray          # H×W float32, per-pixel blur sigma (in pixels)
    mean_sigma: float
    p95_sigma: float
    max_sigma: float
    edge_centre_ratio: float      # avg-edge-sigma / avg-centre-sigma
    query_sigma: Optional[float] = None  # Sigma at specific query point
    query_coords: Optional[tuple[int, int]] = None
    label: str = "spatial_blur"

    def print_summary(self):
        print(f"\n=== Spatial Blur Report [{self.label}] ===")
        print(f"  Mean blur       : {self.mean_sigma:.3f} px")
        print(f"  95th pct blur   : {self.p95_sigma:.3f} px")
        print(f"  Max  blur       : {self.max_sigma:.3f} px")
        print(f"  Edge/Centre     : {self.edge_centre_ratio:.2f}x")
        if self.query_sigma is not None and self.query_coords is not None:
            print(f"  Query at {self.query_coords}: {self.query_sigma:.3f} px")


# ── Method A: FFT High-Frequency Energy ───────────────────────────────────────

def quantify_blur_fft(
    frame: np.ndarray,
    hf_radius_ratio: float = 0.3,
) -> BlurReportFFT:
    """
    Quantify blur via FFT power spectrum analysis.

    Uses two complementary metrics:
    1. Low-frequency concentration ratio: blurred images have more energy near DC
    2. High-frequency energy ratio: sharpened images have more outer-ring energy

    Args:
        frame: BGR image (H×W×3)
        hf_radius_ratio: Fraction of radius to consider as "high frequency"
                         (0.3 means outer 30% of frequency spectrum)

    Returns:
        BlurReportFFT with sharpness metrics
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    h, w = gray.shape

    # FFT and shift zero frequency to center
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2  # Power spectrum

    # Build radius map
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    max_radius = min(cy, cx)
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)

    # ── Metric 1: LF concentration (how much energy sits near DC) ────────────
    # A blurrier image concentrates more power in the low-frequency core.
    lf_mask = radius < max_radius * 0.1      # inner 10% of spectrum
    hf_mask = radius >= max_radius * (1 - hf_radius_ratio)

    total_energy = np.sum(power) + 1e-9
    lf_energy    = np.sum(power[lf_mask])
    hf_energy    = np.sum(power[hf_mask])

    lf_ratio = lf_energy / total_energy   # higher → blurrier
    hf_ratio = hf_energy / total_energy   # higher → sharper

    # ── Metric 2: Power-weighted mean spatial frequency ───────────────────────
    # Sharper images have their power centre of mass further from DC.
    # Exclude DC pixel itself to avoid it dominating.
    dc_exclude = radius > 1
    weighted_radius = float(
        np.sum(radius[dc_exclude] * power[dc_exclude]) /
        np.sum(power[dc_exclude])
    )
    # Normalise to 0-100: 0 = all energy at DC, 100 = all energy at max radius
    mean_freq_score = float(weighted_radius / max_radius * 100)

    # ── Combined sharpness score ───────────────────────────────────────────────
    # 0 = very blurry, 100 = very sharp
    sharpness_score = float((mean_freq_score * 0.7) + (hf_ratio / (lf_ratio + 1e-6) * 3))
    sharpness_score = min(100.0, max(0.0, sharpness_score))

    return BlurReportFFT(
        hf_energy=float(hf_energy),
        hf_ratio=float(hf_ratio),
        mean_magnitude=float(weighted_radius),   # reuse field as mean-freq proxy
        sharpness_score=sharpness_score,
    )


# ── Method B: Spatially Varying Blur Kernel Estimation ────────────────────────

def quantify_blur_spatial(
    frame: np.ndarray,
    model_path: Optional[str] = None,
    window_size: int = 32,
    stride: int = 16,
    query_coords: Optional[tuple[int, int]] = None,
) -> BlurReportSpatial:
    """
    Quantify spatially-varying blur using sliding window estimation.

    This function provides a fallback implementation using Laplacian variance
    in sliding windows. For production use, integrate with the actual
    blur-kernel-estimation model from arunpatro/blur-kernel-estimation.

    Args:
        frame: BGR image (H×W×3)
        model_path: Path to trained model (if None, uses fallback method)
        window_size: Sliding window size (default 32×32)
        stride: Stride for sliding window
        query_coords: Optional (x, y) to query specific location

    Returns:
        BlurReportSpatial with dense blur map
    """
    if model_path is not None and Path(model_path).exists():
        return _quantify_blur_with_model(frame, model_path, query_coords)
    else:
        return _quantify_blur_fallback(frame, window_size, stride, query_coords)


def _quantify_blur_with_model(
    frame: np.ndarray,
    model_path: str,
    query_coords: Optional[tuple[int, int]],
) -> BlurReportSpatial:
    """
    Use trained blur-kernel-estimation model for inference.

    Integration steps:
    1. Clone repo: git clone https://github.com/arunpatro/blur-kernel-estimation
    2. Install dependencies: pip install torch torchvision
    3. Load model checkpoint
    4. Run inference to get dense sigma map
    """
    try:
        import torch
        # TODO: Load model architecture and weights
        # model = load_blur_model(model_path)
        # sigma_map = model.predict(frame)
        raise NotImplementedError(
            "Model-based blur estimation not yet integrated. "
            "Please implement model loading and inference, or use fallback method."
        )
    except ImportError:
        print("Warning: PyTorch not installed. Falling back to Laplacian method.")
        return _quantify_blur_fallback(frame, 32, 16, query_coords)


def _quantify_blur_fallback(
    frame: np.ndarray,
    window_size: int,
    stride: int,
    query_coords: Optional[tuple[int, int]],
) -> BlurReportSpatial:
    """
    Fallback method: estimate blur using Laplacian variance in sliding windows.

    Lower variance = more blur. We invert and normalize to get "sigma-like" metric.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    h, w = gray.shape

    # Initialize blur map
    blur_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.int32)

    # Sliding window
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            window = gray[y:y+window_size, x:x+window_size]

            # Laplacian variance (higher = sharper)
            laplacian = cv2.Laplacian(window, cv2.CV_64F)
            variance = laplacian.var()

            # Convert to "blur sigma" (invert and scale)
            # Empirical scaling: high variance (~1000) → low sigma (~0.5)
            #                    low variance (~10) → high sigma (~5.0)
            sigma = 50.0 / (variance + 10.0)

            # Accumulate
            blur_map[y:y+window_size, x:x+window_size] += sigma
            count_map[y:y+window_size, x:x+window_size] += 1

    # Average overlapping windows
    valid = count_map > 0
    blur_map[valid] = blur_map[valid] / count_map[valid]

    # Fill any gaps (edges) with nearest neighbor
    mask = (count_map == 0).astype(np.uint8)
    if mask.any():
        blur_map = cv2.inpaint(blur_map, mask, 3, cv2.INPAINT_NS)

    # Query specific location if requested
    query_sigma = None
    if query_coords is not None:
        qx, qy = query_coords
        if 0 <= qy < h and 0 <= qx < w:
            query_sigma = float(blur_map[qy, qx])

    return _make_spatial_report(blur_map, query_sigma, query_coords)


def _make_spatial_report(
    blur_map: np.ndarray,
    query_sigma: Optional[float],
    query_coords: Optional[tuple[int, int]],
) -> BlurReportSpatial:
    """Generate report from blur map."""
    h, w = blur_map.shape
    flat = blur_map.ravel()

    # Edge ring: outermost 10% of width/height
    bx = max(1, int(w * 0.10))
    by = max(1, int(h * 0.10))
    edge_mask = np.zeros((h, w), bool)
    edge_mask[:by, :] = True
    edge_mask[-by:, :] = True
    edge_mask[:, :bx] = True
    edge_mask[:, -bx:] = True

    centre_val = float(np.mean(blur_map[~edge_mask])) + 1e-9
    edge_val = float(np.mean(blur_map[edge_mask]))

    return BlurReportSpatial(
        blur_map=blur_map,
        mean_sigma=float(np.mean(flat)),
        p95_sigma=float(np.percentile(flat, 95)),
        max_sigma=float(np.max(flat)),
        edge_centre_ratio=edge_val / centre_val,
        query_sigma=query_sigma,
        query_coords=query_coords,
    )


# ── Batch processing for videos ───────────────────────────────────────────────

def quantify_video_blur_fft(
    frames: Sequence[np.ndarray],
    max_frames: Optional[int] = None,
) -> list[BlurReportFFT]:
    """Quantify FFT blur for each frame in video."""
    if max_frames:
        frames = frames[:max_frames]
    return [quantify_blur_fft(f) for f in frames]


def quantify_video_blur_spatial(
    frames: Sequence[np.ndarray],
    model_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    query_coords: Optional[tuple[int, int]] = None,
) -> list[BlurReportSpatial]:
    """Quantify spatial blur for each frame in video."""
    if max_frames:
        frames = frames[:max_frames]
    return [
        quantify_blur_spatial(f, model_path, query_coords=query_coords)
        for f in frames
    ]
