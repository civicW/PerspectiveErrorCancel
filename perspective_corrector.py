"""
perspective_corrector.py
========================
Corrects perspective error introduced by lens shift or sensor shift.

Physical model
--------------
Camera parameters:
  EFL      f   [mm]   — Equivalent Focal Length
  pitch    p   [mm/px]— Sensor pixel pitch  →  f_px = f / p
  shift  (sx, sy) [mm]— Lens or sensor shift; positive = right / down

The shift moves the optical-axis footprint from the image centre (cx, cy) to
(cx + sx/p, cy + sy/p), creating trapezoidal / keystone perspective distortion
that grows toward the frame edges.

Correction mapping
------------------
For each OUTPUT pixel (u, v) we find the INPUT pixel to sample:

  1. Compute the ray direction assuming a CENTRED principal point:
         dx = (u - cx) / f_px
         dy = (v - cy) / f_px     (dz = 1, standard pinhole)

  2. Find where that ray hits the SHIFTED sensor plane:
         src_x = f_px * dx + cx + sx_px   =  u + sx_px
         src_y = f_px * dy + cy + sy_px   =  v + sy_px

  To first order this is just a translation, but the full formula handles wide
  FoV correctly when combined with radial distortion (see
  build_correction_maps_with_distortion).

Usage
-----
  params = LensSensorParams(efl_mm=24, pitch_mm=0.00112,
                             shift_x_mm=0.15, shift_y_mm=-0.05,
                             img_w=1920, img_h=1080)
  corrected_frames = correct_frames(frames, params, k1=-0.05)
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class LensSensorParams:
    """
    efl_mm      : Equivalent Focal Length [mm]
    pitch_mm    : Sensor pixel pitch [mm/px]  e.g. 0.00112 for 1.12 µm pixels
    shift_x_mm  : Lens/sensor shift X [mm], positive = shift right
    shift_y_mm  : Lens/sensor shift Y [mm], positive = shift down
    img_w, img_h: Image dimensions [px]
    """
    efl_mm: float
    pitch_mm: float
    shift_x_mm: float
    shift_y_mm: float
    img_w: int
    img_h: int

    @property
    def f_px(self) -> float:
        return self.efl_mm / self.pitch_mm

    @property
    def shift_x_px(self) -> float:
        return self.shift_x_mm / self.pitch_mm

    @property
    def shift_y_px(self) -> float:
        return self.shift_y_mm / self.pitch_mm

    @property
    def cx(self) -> float:
        return self.img_w / 2.0

    @property
    def cy(self) -> float:
        return self.img_h / 2.0


def build_correction_maps(
    params: LensSensorParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build OpenCV remap maps (map_x, map_y) that correct shift-induced
    perspective distortion (no radial distortion).

    Returns map_x, map_y — float32, shape (H, W).
    """
    h, w = params.img_h, params.img_w
    f    = params.f_px
    cx, cy = params.cx, params.cy
    sx, sy = params.shift_x_px, params.shift_y_px

    u_out, v_out = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    # normalised ray directions from centred principal point
    dx = (u_out - cx) / f
    dy = (v_out - cy) / f

    # source pixel on the shifted sensor
    map_x = f * dx + cx + sx    # = u_out + sx  (exact for pure shift, pinhole)
    map_y = f * dy + cy + sy    # = v_out + sy

    return map_x.astype(np.float32), map_y.astype(np.float32)


def build_correction_maps_with_distortion(
    params: LensSensorParams,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combined shift-correction + radial/tangential undistortion.

    Uses cv2.initUndistortRectifyMap with the shifted principal point as
    the source camera and the centred principal point as the target camera.
    This is exact for the full OpenCV distortion model.
    """
    h, w = params.img_h, params.img_w
    f    = params.f_px
    cx, cy = params.cx, params.cy
    sx, sy = params.shift_x_px, params.shift_y_px

    # camera matrix of the ACTUAL (shifted) sensor
    K_src = np.array([
        [f, 0, cx + sx],
        [0, f, cy + sy],
        [0, 0, 1      ],
    ], dtype=np.float64)

    # desired output camera matrix (centred, no shift)
    K_dst = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1 ],
    ], dtype=np.float64)

    dist = np.array([k1, k2, p1, p2, 0], dtype=np.float64)

    map_x, map_y = cv2.initUndistortRectifyMap(
        K_src, dist, None, K_dst, (w, h), cv2.CV_32FC1
    )
    return map_x, map_y


def apply_correction(
    frame: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """Remap a single frame using precomputed correction maps."""
    return cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=border_mode,
    )


def correct_frames(
    frames: list[np.ndarray],
    params: LensSensorParams,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> list[np.ndarray]:
    """Correct a list of frames in one call."""
    if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0:
        map_x, map_y = build_correction_maps(params)
    else:
        map_x, map_y = build_correction_maps_with_distortion(
            params, k1=k1, k2=k2, p1=p1, p2=p2
        )
    return [apply_correction(f, map_x, map_y) for f in frames]


def visualise_warp_field(
    params: LensSensorParams,
    stride: int = 30,
) -> np.ndarray:
    """
    Return an RGB image showing the correction vector field as arrows.
    stride : grid spacing in pixels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    map_x, map_y = build_correction_maps(params)
    h, w = params.img_h, params.img_w

    fig, ax = plt.subplots(figsize=(8, 8 * h / w))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(
        f"Shift correction warp field\n"
        f"EFL={params.efl_mm} mm  "
        f"shift=({params.shift_x_mm:.3f}, {params.shift_y_mm:.3f}) mm"
    )

    ys = np.arange(stride // 2, h, stride)
    xs = np.arange(stride // 2, w, stride)
    for y in ys:
        for x in xs:
            dx = map_x[y, x] - x
            dy = map_y[y, x] - y
            if abs(dx) + abs(dy) > 0.05:
                ax.annotate(
                    "", xy=(x + dx * 5, y + dy * 5), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="royalblue", lw=0.8),
                )

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf
