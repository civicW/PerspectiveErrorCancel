"""
perspective_quantifier.py
==========================
Quantifies perspective error as WARPED PIXEL COUNT.

Two methods:
  A) VIDEO  — optical flow magnitude after subtracting global translation.
              After RS correction the remaining spatially-varying motion that
              grows toward frame edges is "perspective jitter".
  B) STILL  — project a virtual grid through the camera model and measure
              reprojection error (useful with synthetic / calibration data).

Metric output (PerspErrorReport):
  warp_map        : H×W float32, per-pixel warp magnitude in pixels
  mean/p95/max    : scalar statistics
  edge_centre_ratio: avg-edge-warp / avg-centre-warp  (>1 means edge is worse)
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Sequence


@dataclass
class PerspErrorReport:
    warp_map: np.ndarray          # H×W float32, magnitude in pixels
    mean_warp_px: float
    p95_warp_px: float
    max_warp_px: float
    edge_centre_ratio: float      # avg-edge / avg-centre warp
    label: str = ""

    def print_summary(self):
        print(f"\n=== Perspective Error Report [{self.label}] ===")
        print(f"  Mean warp     : {self.mean_warp_px:.3f} px")
        print(f"  95th pct warp : {self.p95_warp_px:.3f} px")
        print(f"  Max  warp     : {self.max_warp_px:.3f} px")
        print(f"  Edge/Centre   : {self.edge_centre_ratio:.2f}x")


# ── Method A: optical-flow based (video) ──────────────────────────────────────

def _dense_flow(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Farneback dense optical flow → H×W×2 float32."""
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=5, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )


def _global_translation(flow: np.ndarray) -> np.ndarray:
    """Spatially-uniform (DC) component of the flow — use median for robustness."""
    return np.median(flow.reshape(-1, 2), axis=0)   # shape (2,)


def quantify_video(
    frames: Sequence[np.ndarray],
    window: int = 5,
) -> PerspErrorReport:
    """
    Accumulate optical flow between consecutive frame pairs.
    The global translation component is subtracted — what remains is the
    spatially-varying distortion (perspective jitter) after RS correction.

    window : number of frame pairs to average for stability.
    """
    if len(frames) < 2:
        h, w = frames[0].shape[:2]
        zero = np.zeros((h, w), np.float32)
        return PerspErrorReport(zero, 0., 0., 0., 1.0)

    h, w = frames[0].shape[:2]
    accum = np.zeros((h, w), np.float32)
    pairs = min(window, len(frames) - 1)

    for i in range(pairs):
        flow = _dense_flow(frames[i], frames[i + 1])          # H×W×2
        dc = _global_translation(flow)
        residual = flow - dc[np.newaxis, np.newaxis, :]       # remove global shift
        accum += np.linalg.norm(residual, axis=2)             # H×W magnitude

    warp_map = accum / max(pairs, 1)
    return _make_report(warp_map, label="video_optical_flow")


# ── Method B: synthetic grid reprojection (still / calibration) ───────────────

def _project_grid(
    K: np.ndarray,
    dist: np.ndarray,
    shift_px: tuple[float, float],
    grid_size: int,
    img_size: tuple[int, int],
) -> np.ndarray:
    """Project a flat grid of 3-D points with camera K + distortion + shift."""
    w, h = img_size
    xs = np.linspace(-1.0, 1.0, grid_size) * w * 0.4
    ys = np.linspace(-1.0, 1.0, grid_size) * h * 0.4
    xx, yy = np.meshgrid(xs, ys)
    pts3d = np.stack([xx.ravel(), yy.ravel(),
                      np.ones(xx.size) * K[0, 0]], axis=1).astype(np.float64)

    K_shifted = K.copy()
    K_shifted[0, 2] += shift_px[0]
    K_shifted[1, 2] += shift_px[1]

    rvec = np.zeros(3, np.float64)
    tvec = np.zeros(3, np.float64)
    projected, _ = cv2.projectPoints(pts3d, rvec, tvec, K_shifted, dist)
    return projected.reshape(-1, 2)


def quantify_checkerboard(
    K: np.ndarray,
    dist: np.ndarray,
    shift_px: tuple[float, float],
    img_size: tuple[int, int],
    grid_size: int = 30,
) -> PerspErrorReport:
    """
    Compare ideal (no-shift) vs shifted projection on a dense grid.
    The difference is the perspective error introduced by shift.
    """
    ideal   = _project_grid(K, dist, (0., 0.),  grid_size, img_size)
    shifted = _project_grid(K, dist, shift_px,  grid_size, img_size)

    disp = shifted - ideal                       # Nx2
    mag  = np.linalg.norm(disp, axis=1)          # N

    # scatter → dense map via rasterise
    w, h = img_size
    warp_map = np.zeros((h, w), np.float32)
    xs_i = ((ideal[:, 0] + w / 2) / w * (w - 1)).astype(int).clip(0, w - 1)
    ys_i = ((ideal[:, 1] + h / 2) / h * (h - 1)).astype(int).clip(0, h - 1)
    np.maximum.at(warp_map, (ys_i, xs_i), mag.astype(np.float32))

    # fill gaps with nearest-neighbour inpainting
    mask = (warp_map == 0).astype(np.uint8)
    warp_map = cv2.inpaint(warp_map, mask, 3, cv2.INPAINT_NS)

    return _make_report(warp_map, label="checkerboard_reprojection")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_report(warp_map: np.ndarray, label: str = "") -> PerspErrorReport:
    h, w = warp_map.shape
    flat = warp_map.ravel()

    # edge ring: outermost 10% of width / height
    bx = max(1, int(w * 0.10))
    by = max(1, int(h * 0.10))
    edge_mask = np.zeros((h, w), bool)
    edge_mask[:by, :] = True
    edge_mask[-by:, :] = True
    edge_mask[:, :bx] = True
    edge_mask[:, -bx:] = True

    centre_val = float(np.mean(warp_map[~edge_mask])) + 1e-9
    edge_val   = float(np.mean(warp_map[edge_mask]))

    return PerspErrorReport(
        warp_map=warp_map,
        mean_warp_px=float(np.mean(flat)),
        p95_warp_px=float(np.percentile(flat, 95)),
        max_warp_px=float(np.max(flat)),
        edge_centre_ratio=edge_val / centre_val,
        label=label,
    )
