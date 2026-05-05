"""
generate_test_data.py
=====================
生成测试用合成视频和 shift CSV，模拟：
  - 4K 60fps 1/1000s 快门，1 秒（60 帧）
  - OIS 执行器逐帧 shift（低频漂移 + 高频抖动）

生成文件：
  test_input/
  ├── test_video.mp4     — 棋盘格背景 + shift 透视扭曲
  └── shift_log.csv      — 每帧 shift_x_mm / shift_y_mm

CSV 格式
--------
  frame, timestamp_s, shift_x_mm, shift_y_mm, shift_x_px, shift_y_px

运行：
  python generate_test_data.py [--out-dir test_input] [--efl 24] [--pitch 0.00112]
"""

from __future__ import annotations
import argparse, csv, os
import cv2
import numpy as np

# ── 默认参数 ──────────────────────────────────────────────────────────────────
W, H   = 3840, 2160
FPS    = 60
N      = 60            # 1 秒

def make_shift_csv(path: str, n: int, f_px: float, pitch: float,
                   rng: np.random.Generator) -> list[dict]:
    t  = np.linspace(0, 1, n, endpoint=False)
    sx = 0.12 * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0, 0.015, n)
    sy = 0.08 * np.sin(2 * np.pi * 0.4 * t + 0.5) + rng.normal(0, 0.015, n)

    rows = []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame", "timestamp_s",
            "shift_x_mm", "shift_y_mm",
            "shift_x_px", "shift_y_px",
        ])
        w.writeheader()
        for i in range(n):
            row = dict(
                frame=i,
                timestamp_s=round(float(t[i]), 6),
                shift_x_mm=round(float(sx[i]), 6),
                shift_y_mm=round(float(sy[i]), 6),
                shift_x_px=round(float(sx[i] / pitch), 3),
                shift_y_px=round(float(sy[i] / pitch), 3),
            )
            w.writerow(row)
            rows.append(row)
    print(f"  [CSV]   {path}  ({n} rows)")
    return rows


def _checkerboard(w: int, h: int, sq: int = 120) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, sq):
        for c in range(0, w, sq):
            if ((r // sq) + (c // sq)) % 2 == 0:
                img[r:r+sq, c:c+sq] = 210
    return img


def _apply_shift_warp(frame: np.ndarray,
                      sx_px: float, sy_px: float,
                      f_px: float) -> np.ndarray:
    """Forward warp: 模拟 shift 导致的梯形透视失真（corrector 要消除的）。"""
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    K_s = np.array([[f_px, 0, cx+sx_px],
                    [0, f_px, cy+sy_px],
                    [0, 0, 1]], dtype=np.float64)
    K_i = np.array([[f_px, 0, cx],
                    [0, f_px, cy],
                    [0, 0, 1]], dtype=np.float64)
    H = K_s @ np.linalg.inv(K_i)
    return cv2.warpPerspective(frame, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def make_test_video(path: str, shift_rows: list[dict], f_px: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    bg = _checkerboard(W, H)
    for row in shift_rows:
        frame = bg.copy()
        cv2.putText(frame,
                    f"frame {row['frame']:03d}  "
                    f"sx={row['shift_x_mm']:+.3f}mm  "
                    f"sy={row['shift_y_mm']:+.3f}mm",
                    (60, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    2.2, (0, 180, 255), 4, cv2.LINE_AA)
        distorted = _apply_shift_warp(frame,
                                      float(row["shift_x_px"]),
                                      float(row["shift_y_px"]),
                                      f_px)
        writer.write(distorted)
    writer.release()
    print(f"  [Video] {path}  ({len(shift_rows)} frames, {W}×{H} @ {FPS}fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="test_input")
    ap.add_argument("--efl",     type=float, default=24.0)
    ap.add_argument("--pitch",   type=float, default=0.00112)
    ap.add_argument("--seed",    type=int,   default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    f_px = args.efl / args.pitch
    rng  = np.random.default_rng(args.seed)

    print(f"\nGenerating test data  (EFL={args.efl}mm  pitch={args.pitch}mm  f={f_px:.0f}px)\n")
    rows = make_shift_csv(f"{args.out_dir}/shift_log.csv", N, f_px, args.pitch, rng)
    make_test_video(f"{args.out_dir}/test_video.mp4", rows, f_px)

    xs = [r['shift_x_mm'] for r in rows]
    ys = [r['shift_y_mm'] for r in rows]
    print(f"\n  shift X: [{min(xs):+.3f}, {max(xs):+.3f}] mm")
    print(f"  shift Y: [{min(ys):+.3f}, {max(ys):+.3f}] mm")
    print(f"\nDone → ./{args.out_dir}/")


if __name__ == "__main__":
    main()
