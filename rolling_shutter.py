"""
rolling_shutter.py
==================
Wraps gyroflow-toolbox to apply rolling-shutter / stabilisation correction
frame-by-frame.

If gyroflow-toolbox is not available, falls back to a pure-OpenCV EIS stub
(ORB feature matching + homography) so the rest of the pipeline can still run.
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class RSCorrectionResult:
    frames: list[np.ndarray]          # corrected frames (BGR, uint8)
    residual_map: np.ndarray | None   # per-frame RS residual magnitude (float32)
    gyro_available: bool = False


# ── gyroflow binding (optional) ───────────────────────────────────────────────
def _try_import_gyroflow():
    try:
        import gyroflow_toolbox as gf   # pip install gyroflow-toolbox
        return gf
    except ImportError:
        return None


class GyroflowRSCorrector:
    """
    Use gyroflow-toolbox to load a .gyroflow project and render stabilised
    frames that have rolling-shutter compensation baked in.

    gyroflow_project : path to the *.gyroflow file exported from the desktop app
    video_path       : source video (must match the project)
    """

    def __init__(self, gyroflow_project: str | Path, video_path: str | Path):
        gf = _try_import_gyroflow()
        if gf is None:
            raise RuntimeError(
                "gyroflow-toolbox not found. Install with:\n"
                "  pip install gyroflow-toolbox\n"
                "or use FallbackRSCorrector for a no-gyro path."
            )
        self.gf = gf
        self.project_path = str(gyroflow_project)
        self.video_path = str(video_path)

    def correct(self, max_frames: int | None = None) -> RSCorrectionResult:
        """
        Render all (or up to max_frames) stabilised + RS-corrected frames.

        gyroflow-toolbox API (v0.1.x):
            manager = gf.GyroflowManager()
            manager.load_project(path)
            frame_bgr = manager.get_frame(frame_index)
        """
        manager = self.gf.GyroflowManager()
        manager.load_project(self.project_path)

        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        n = min(total, max_frames) if max_frames else total
        corrected: list[np.ndarray] = []
        for i in range(n):
            frame = manager.get_frame(i)   # ndarray H×W×3 BGR
            corrected.append(frame)

        return RSCorrectionResult(
            frames=corrected,
            residual_map=None,
            gyro_available=True,
        )


class FallbackRSCorrector:
    """
    No-gyro fallback: estimate inter-frame homography via ORB feature matching
    and warp each frame to the first frame's reference.

    This does NOT compensate rolling shutter on a per-scanline basis, but
    removes global camera shake so the perspective-error stage can be isolated.
    """

    def __init__(self, video_path: str | Path):
        self.video_path = str(video_path)

    def _read_frames(self, max_frames: int | None) -> list[np.ndarray]:
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        return frames

    def correct(self, max_frames: int | None = None) -> RSCorrectionResult:
        raw = self._read_frames(max_frames)
        if len(raw) == 0:
            return RSCorrectionResult(frames=[], residual_map=None)

        ref_gray = cv2.cvtColor(raw[0], cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(2000)
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        h, w = raw[0].shape[:2]

        corrected = [raw[0].copy()]
        warp_magnitudes: list[float] = [0.0]

        for frame in raw[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)
            if des is None or des_ref is None or len(des) < 10:
                corrected.append(frame)
                warp_magnitudes.append(0.0)
                continue

            matches = bf.match(des_ref, des)
            matches = sorted(matches, key=lambda m: m.distance)[:200]

            if len(matches) < 10:
                corrected.append(frame)
                warp_magnitudes.append(0.0)
                continue

            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)

            if H is None:
                corrected.append(frame)
                warp_magnitudes.append(0.0)
                continue

            warped = cv2.warpPerspective(frame, H, (w, h))
            corrected.append(warped)

            # warp magnitude at frame corners → rolling-shutter residual proxy
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(corners, H)
            mag = float(np.mean(np.linalg.norm(mapped - corners, axis=2)))
            warp_magnitudes.append(mag)

        residual = np.array(warp_magnitudes, dtype=np.float32)
        return RSCorrectionResult(
            frames=corrected,
            residual_map=residual,
            gyro_available=False,
        )


def load_corrector(
    video_path: str | Path,
    gyroflow_project: str | Path | None = None,
) -> GyroflowRSCorrector | FallbackRSCorrector:
    if gyroflow_project is not None:
        return GyroflowRSCorrector(gyroflow_project, video_path)
    return FallbackRSCorrector(video_path)
