"""
pipeline_csv.py
===============
和 pipeline.py 相同的三阶段流程，但支持逐帧 shift CSV：

  Stage 1  Rolling-shutter 校正（gyroflow / fallback）
  Stage 2a 透视误差量化 BEFORE
  Stage 3  按 CSV 逐帧 perspective 校正
  Stage 2b 透视误差量化 AFTER

CSV 格式（至少包含以下列，其余可忽略）：
  frame, shift_x_mm, shift_y_mm

用法
----
  # 使用合成测试数据（generate_test_data.py 生成）：
  python pipeline_csv.py \\
      --video   test_input/test_video.mp4 \\
      --csv     test_input/shift_log.csv  \\
      --efl     24.0    \\
      --pitch   0.00112 \\
      --out-dir output_csv

  # 真实素材 + 你的 shift log：
  python pipeline_csv.py \\
      --video   my_footage.mp4 \\
      --csv     my_shift_log.csv \\
      --efl     24.0 --pitch 0.00112 \\
      --gyroflow my_footage.gyroflow   # 可选
"""

from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 把仓库路径加入 sys.path ───────────────────────────────────────────────────
REPO = Path(__file__).parent.parent / "PerspectiveErrorCancel"
# 如果脚本和仓库在同一目录，直接用当前目录
for candidate in [REPO, Path(__file__).parent,
                  Path("/Volumes/SSD1T/Project/OIS/PerspectiveErrorCancel")]:
    if (candidate / "rolling_shutter.py").exists():
        sys.path.insert(0, str(candidate))
        break

from rolling_shutter import load_corrector, RSCorrectionResult
from perspective_quantifier import quantify_video, PerspErrorReport
from perspective_corrector import (
    LensSensorParams, build_correction_maps,
    build_correction_maps_with_distortion, apply_correction,
    visualise_warp_field,
)


# ── CSV 加载 ──────────────────────────────────────────────────────────────────

def load_shift_csv(path: str) -> list[dict]:
    """
    读取 shift CSV，返回按 frame 排序的 list[dict]。
    必须包含列: frame, shift_x_mm, shift_y_mm
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame":       int(row["frame"]),
                "shift_x_mm":  float(row["shift_x_mm"]),
                "shift_y_mm":  float(row["shift_y_mm"]),
            })
    rows.sort(key=lambda r: r["frame"])
    return rows


def get_shift_for_frame(rows: list[dict], frame_idx: int) -> tuple[float, float]:
    """线性插值，允许 CSV 帧数和视频帧数不完全对齐。"""
    if frame_idx <= rows[0]["frame"]:
        return rows[0]["shift_x_mm"], rows[0]["shift_y_mm"]
    if frame_idx >= rows[-1]["frame"]:
        return rows[-1]["shift_x_mm"], rows[-1]["shift_y_mm"]
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if a["frame"] <= frame_idx <= b["frame"]:
            t = (frame_idx - a["frame"]) / max(b["frame"] - a["frame"], 1)
            sx = a["shift_x_mm"] + t * (b["shift_x_mm"] - a["shift_x_mm"])
            sy = a["shift_y_mm"] + t * (b["shift_y_mm"] - a["shift_y_mm"])
            return sx, sy
    return 0.0, 0.0


# ── 逐帧校正 ──────────────────────────────────────────────────────────────────

def correct_frames_perframe(
    frames: list[np.ndarray],
    shift_rows: list[dict],
    efl_mm: float,
    pitch_mm: float,
    k1: float = 0.0, k2: float = 0.0,
    p1: float = 0.0, p2: float = 0.0,
) -> list[np.ndarray]:
    """
    逐帧构建 remap map（每帧 shift 不同），应用透视校正。
    """
    corrected = []
    img_h, img_w = frames[0].shape[:2]
    cache: dict[tuple, tuple] = {}   # (sx_rounded, sy_rounded) → (map_x, map_y)

    for i, frame in enumerate(frames):
        sx_mm, sy_mm = get_shift_for_frame(shift_rows, i)

        # round to 3 decimal places for cache key (0.001 mm ≈ 0.9 px)
        key = (round(sx_mm, 3), round(sy_mm, 3))
        if key not in cache:
            params = LensSensorParams(
                efl_mm=efl_mm, pitch_mm=pitch_mm,
                shift_x_mm=sx_mm, shift_y_mm=sy_mm,
                img_w=img_w, img_h=img_h,
            )
            if k1 == 0.0 and k2 == 0.0:
                mx, my = build_correction_maps(params)
            else:
                mx, my = build_correction_maps_with_distortion(
                    params, k1=k1, k2=k2, p1=p1, p2=p2)
            cache[key] = (mx, my)

        map_x, map_y = cache[key]
        corrected.append(apply_correction(frame, map_x, map_y))

    return corrected


# ── 可视化 shift 轨迹 ─────────────────────────────────────────────────────────

def plot_shift_trajectory(shift_rows: list[dict], path: str):
    frames = [r["frame"]      for r in shift_rows]
    sx     = [r["shift_x_mm"] for r in shift_rows]
    sy     = [r["shift_y_mm"] for r in shift_rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(frames, sx, color="royalblue", lw=1.5)
    axes[0].axhline(0, color="gray", lw=0.7, ls="--")
    axes[0].set_ylabel("shift X (mm)")
    axes[0].set_title("OIS shift trajectory from CSV")
    axes[1].plot(frames, sy, color="tomato", lw=1.5)
    axes[1].axhline(0, color="gray", lw=0.7, ls="--")
    axes[1].set_ylabel("shift Y (mm)")
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved shift trajectory → {path}")


# ── 热图 / 报告工具 ───────────────────────────────────────────────────────────

def save_heatmap(report: PerspErrorReport, path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(report.warp_map, cmap="hot", aspect="auto",
                   vmin=0, vmax=max(report.max_warp_px, 0.1))
    plt.colorbar(im, ax=ax, label="warp (px)")
    ax.set_title(title or report.label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap → {path}")


def save_comparison(before: PerspErrorReport, after: PerspErrorReport, path: str):
    vmax = max(before.max_warp_px, after.max_warp_px, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, r, title in zip(axes, [before, after],
                             ["BEFORE correction", "AFTER correction"]):
        im = ax.imshow(r.warp_map, cmap="hot", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"{title}\nmean={r.mean_warp_px:.3f} px  p95={r.p95_warp_px:.3f} px")
        plt.colorbar(im, ax=ax, label="warp (px)")
    fig.suptitle("Perspective error: before vs after (per-frame shift correction)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison → {path}")


def save_video(frames: list[np.ndarray], path: str, fps: float = 60.0):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  Saved video → {path}")


def save_report(before: PerspErrorReport, after: PerspErrorReport,
                shift_rows: list[dict], path: str) -> dict:
    eps = 1e-9
    shifts_x = [r["shift_x_mm"] for r in shift_rows]
    shifts_y = [r["shift_y_mm"] for r in shift_rows]
    data = {
        "shift_stats_mm": {
            "x_range": [round(min(shifts_x),4), round(max(shifts_x),4)],
            "y_range": [round(min(shifts_y),4), round(max(shifts_y),4)],
            "x_rms":   round(float(np.sqrt(np.mean(np.square(shifts_x)))), 4),
            "y_rms":   round(float(np.sqrt(np.mean(np.square(shifts_y)))), 4),
        },
        "before": {
            "mean_warp_px": round(before.mean_warp_px, 4),
            "p95_warp_px":  round(before.p95_warp_px,  4),
            "max_warp_px":  round(before.max_warp_px,  4),
            "edge_centre_ratio": round(before.edge_centre_ratio, 3),
        },
        "after": {
            "mean_warp_px": round(after.mean_warp_px, 4),
            "p95_warp_px":  round(after.p95_warp_px,  4),
            "max_warp_px":  round(after.max_warp_px,  4),
            "edge_centre_ratio": round(after.edge_centre_ratio, 3),
        },
        "reduction_pct": {
            "mean": round((1 - after.mean_warp_px/(before.mean_warp_px+eps))*100, 1),
            "p95":  round((1 - after.p95_warp_px /(before.p95_warp_px +eps))*100, 1),
            "max":  round((1 - after.max_warp_px  /(before.max_warp_px +eps))*100, 1),
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved report → {path}")
    return data


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stage 1 ─ Rolling shutter
    print("\n[Stage 1] Rolling-shutter correction …")
    corrector = load_corrector(args.video, args.gyroflow)
    rs: RSCorrectionResult = corrector.correct(max_frames=args.max_frames)
    print(f"  Frames: {len(rs.frames)}  |  gyro: {rs.gyro_available}")
    save_video(rs.frames, str(out / "s1_rs_corrected.mp4"), fps=args.fps)

    # Load shift CSV
    print(f"\n[CSV] Loading shift log: {args.csv}")
    shift_rows = load_shift_csv(args.csv)
    print(f"  {len(shift_rows)} entries  "
          f"| X rms={np.sqrt(np.mean(np.square([r['shift_x_mm'] for r in shift_rows]))):.3f}mm  "
          f"| Y rms={np.sqrt(np.mean(np.square([r['shift_y_mm'] for r in shift_rows]))):.3f}mm")
    plot_shift_trajectory(shift_rows, str(out / "shift_trajectory.png"))

    # Stage 2a ─ Quantify BEFORE
    print("\n[Stage 2a] Quantify perspective error BEFORE correction …")
    window = min(10, len(rs.frames) - 1)
    before = quantify_video(rs.frames, window=window)
    before.label = "before"
    before.print_summary()
    save_heatmap(before, str(out / "heatmap_before.png"),
                 "Perspective error BEFORE correction (px)")

    # Stage 3 ─ Per-frame perspective correction
    print("\n[Stage 3] Per-frame perspective correction …")
    corrected = correct_frames_perframe(
        rs.frames, shift_rows,
        efl_mm=args.efl, pitch_mm=args.pitch,
        k1=args.k1, k2=args.k2, p1=args.p1, p2=args.p2,
    )
    save_video(corrected, str(out / "s3_persp_corrected.mp4"), fps=args.fps)

    # Stage 2b ─ Quantify AFTER
    print("\n[Stage 2b] Quantify perspective error AFTER correction …")
    after = quantify_video(corrected, window=window)
    after.label = "after"
    after.print_summary()
    save_heatmap(after, str(out / "heatmap_after.png"),
                 "Perspective error AFTER correction (px)")

    save_comparison(before, after, str(out / "heatmap_comparison.png"))

    # Report
    print("\n[Report]")
    data = save_report(before, after, shift_rows, str(out / "report.json"))
    r = data["reduction_pct"]
    print(f"\n{'='*50}")
    print(f"  Mean warp reduction : {r['mean']:>6.1f}%")
    print(f"  P95  warp reduction : {r['p95']:>6.1f}%")
    print(f"  Max  warp reduction : {r['max']:>6.1f}%")
    print(f"{'='*50}")
    print(f"\nAll outputs → {out.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True)
    ap.add_argument("--csv",        required=True,  help="Per-frame shift CSV")
    ap.add_argument("--gyroflow",   default=None)
    ap.add_argument("--efl",        type=float, default=24.0)
    ap.add_argument("--pitch",      type=float, default=0.00112)
    ap.add_argument("--k1",         type=float, default=0.0)
    ap.add_argument("--k2",         type=float, default=0.0)
    ap.add_argument("--p1",         type=float, default=0.0)
    ap.add_argument("--p2",         type=float, default=0.0)
    ap.add_argument("--fps",        type=float, default=60.0)
    ap.add_argument("--max-frames", type=int,   default=None)
    ap.add_argument("--out-dir",    default="output_csv")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
