"""
pipeline.py
===========
Full three-stage pipeline:

  Stage 1  Rolling-shutter (果冻效应) correction via gyroflow / fallback EIS
  Stage 2a Perspective error quantification BEFORE correction
  Stage 3  Perspective correction (lens/sensor shift + EFL model)
  Stage 2b Perspective error re-quantification AFTER correction

Usage
-----
  python pipeline.py \\
      --video      input.mp4          \\
      --gyroflow   input.gyroflow     \\   # optional
      --efl        24.0               \\   # EFL mm
      --pitch      0.00112            \\   # pixel pitch mm
      --shift-x    0.15               \\   # lens/sensor shift X mm
      --shift-y   -0.05               \\   # lens/sensor shift Y mm
      --k1        -0.05               \\   # radial distortion k1 (optional)
      --out-dir    ./output
"""

from __future__ import annotations
import argparse
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from rolling_shutter import load_corrector, RSCorrectionResult
from perspective_quantifier import quantify_video, PerspErrorReport
from perspective_corrector import (
    LensSensorParams,
    correct_frames,
    visualise_warp_field,
)
from blur_quantifier import (
    quantify_blur_fft,
    quantify_blur_spatial,
    BlurReportFFT,
    BlurReportSpatial,
)
from quality_analyzer import QualityAnalyzer, QualityReport


# ── helpers ───────────────────────────────────────────────────────────────────

def save_heatmap(report: PerspErrorReport, path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(report.warp_map, cmap="hot", aspect="auto",
                   vmin=0, vmax=max(report.max_warp_px, 0.1))
    plt.colorbar(im, ax=ax, label="warp magnitude (px)")
    ax.set_title(title or report.label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap → {path}")


def save_blur_heatmap(report: BlurReportSpatial, path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(report.blur_map, cmap="hot", aspect="auto",
                   vmin=0, vmax=max(report.max_sigma, 0.1))
    plt.colorbar(im, ax=ax, label="blur sigma (px)")
    ax.set_title(title or report.label)

    # Mark query point if available
    if report.query_coords is not None:
        qx, qy = report.query_coords
        ax.plot(qx, qy, 'c*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
        ax.text(qx + 20, qy, f'{report.query_sigma:.1f} px',
                color='cyan', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved blur heatmap → {path}")


def save_comparison_heatmap(
    before: PerspErrorReport,
    after: PerspErrorReport,
    path: str,
):
    vmax = max(before.max_warp_px, after.max_warp_px, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, report, title in zip(
        axes,
        [before, after],
        ["Before correction", "After correction"],
    ):
        im = ax.imshow(report.warp_map, cmap="hot", aspect="auto",
                       vmin=0, vmax=vmax)
        ax.set_title(f"{title}\nmean={report.mean_warp_px:.3f} px  "
                     f"p95={report.p95_warp_px:.3f} px")
        plt.colorbar(im, ax=ax, label="warp (px)")
    fig.suptitle("Perspective error: before vs after shift correction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison → {path}")


def save_video(frames: list[np.ndarray], path: str, fps: float = 30.0):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  Saved video → {path}")


def save_report(
    before: PerspErrorReport,
    after: PerspErrorReport,
    params: LensSensorParams,
    path: str,
    blur_before: Optional[BlurReportSpatial] = None,
    blur_after: Optional[BlurReportSpatial] = None,
) -> dict:
    eps = 1e-9
    data = {
        "camera_params": {
            "efl_mm":      params.efl_mm,
            "pitch_mm":    params.pitch_mm,
            "f_px":        params.f_px,
            "shift_x_mm":  params.shift_x_mm,
            "shift_y_mm":  params.shift_y_mm,
            "shift_x_px":  params.shift_x_px,
            "shift_y_px":  params.shift_y_px,
            "img_w":       params.img_w,
            "img_h":       params.img_h,
        },
        "before_correction": {
            "mean_warp_px":      round(before.mean_warp_px, 4),
            "p95_warp_px":       round(before.p95_warp_px,  4),
            "max_warp_px":       round(before.max_warp_px,  4),
            "edge_centre_ratio": round(before.edge_centre_ratio, 3),
        },
        "after_correction": {
            "mean_warp_px":      round(after.mean_warp_px,  4),
            "p95_warp_px":       round(after.p95_warp_px,   4),
            "max_warp_px":       round(after.max_warp_px,   4),
            "edge_centre_ratio": round(after.edge_centre_ratio, 3),
        },
        "reduction_pct": {
            "mean": round((1 - after.mean_warp_px / (before.mean_warp_px + eps)) * 100, 1),
            "p95":  round((1 - after.p95_warp_px  / (before.p95_warp_px  + eps)) * 100, 1),
            "max":  round((1 - after.max_warp_px   / (before.max_warp_px  + eps)) * 100, 1),
        },
    }

    # Add blur metrics if available
    if blur_before is not None and blur_after is not None:
        data["blur_before_correction"] = {
            "mean_sigma_px":     round(blur_before.mean_sigma, 4),
            "p95_sigma_px":      round(blur_before.p95_sigma, 4),
            "max_sigma_px":      round(blur_before.max_sigma, 4),
            "edge_centre_ratio": round(blur_before.edge_centre_ratio, 3),
        }
        data["blur_after_correction"] = {
            "mean_sigma_px":     round(blur_after.mean_sigma, 4),
            "p95_sigma_px":      round(blur_after.p95_sigma, 4),
            "max_sigma_px":      round(blur_after.max_sigma, 4),
            "edge_centre_ratio": round(blur_after.edge_centre_ratio, 3),
        }
        data["blur_reduction_pct"] = {
            "mean": round((1 - blur_after.mean_sigma / (blur_before.mean_sigma + eps)) * 100, 1),
            "p95":  round((1 - blur_after.p95_sigma  / (blur_before.p95_sigma  + eps)) * 100, 1),
            "max":  round((1 - blur_after.max_sigma   / (blur_before.max_sigma  + eps)) * 100, 1),
        }

        # Add query point data if available
        if blur_before.query_sigma is not None and blur_after.query_sigma is not None:
            data["blur_query_point"] = {
                "coords": blur_before.query_coords,
                "before_px": round(blur_before.query_sigma, 2),
                "after_px": round(blur_after.query_sigma, 2),
                "reduction_pct": round((1 - blur_after.query_sigma / (blur_before.query_sigma + eps)) * 100, 1),
            }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved report → {path}")
    return data


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    analyzer = QualityAnalyzer()

    # ── Stage 1: Rolling-shutter correction ───────────────────────────────────
    print("\n[Stage 1] Rolling-shutter correction …")
    corrector = load_corrector(args.video, args.gyroflow)
    rs_result: RSCorrectionResult = corrector.correct(max_frames=args.max_frames)
    rs_frames = rs_result.frames
    print(f"  Frames processed : {len(rs_frames)}")
    print(f"  Gyro available   : {rs_result.gyro_available}")
    if not rs_result.gyro_available:
        print("  (Using ORB-homography EIS fallback — "
              "install gyroflow-toolbox for per-scanline RS correction)")

    save_video(rs_frames, str(out / "stage1_rs_corrected.mp4"))

    # ── Stage 2a: Quantify perspective error BEFORE correction ────────────────
    print("\n[Stage 2a] Quantifying perspective error BEFORE correction …")
    window = min(10, len(rs_frames) - 1)
    report_before = quantify_video(rs_frames, window=window)
    report_before.label = "before_correction"
    report_before.print_summary()
    save_heatmap(report_before, str(out / "heatmap_before.png"),
                 "Perspective error BEFORE correction (px)")

    # ── Stage 3: Perspective correction ───────────────────────────────────────
    print("\n[Stage 3] Applying perspective correction …")
    params = LensSensorParams(
        efl_mm=args.efl,
        pitch_mm=args.pitch,
        shift_x_mm=args.shift_x,
        shift_y_mm=args.shift_y,
        img_w=rs_frames[0].shape[1],
        img_h=rs_frames[0].shape[0],
    )
    print(f"  f_px        : {params.f_px:.1f} px")
    print(f"  shift_px    : ({params.shift_x_px:.3f}, {params.shift_y_px:.3f}) px")

    corrected_frames = correct_frames(
        rs_frames, params,
        k1=args.k1, k2=args.k2, p1=args.p1, p2=args.p2,
    )
    save_video(corrected_frames, str(out / "stage3_perspective_corrected.mp4"))

    # warp field visualisation
    wf_img = visualise_warp_field(params)
    cv2.imwrite(str(out / "warp_field.png"), cv2.cvtColor(wf_img, cv2.COLOR_RGB2BGR))
    print(f"  Saved warp field → {out / 'warp_field.png'}")

    # ── Stage 2b: Re-quantify AFTER correction ────────────────────────────────
    print("\n[Stage 2b] Re-quantifying perspective error AFTER correction …")
    report_after = quantify_video(corrected_frames, window=window)
    report_after.label = "after_correction"
    report_after.print_summary()
    save_heatmap(report_after, str(out / "heatmap_after.png"),
                 "Perspective error AFTER correction (px)")

    save_comparison_heatmap(report_before, report_after,
                            str(out / "heatmap_comparison.png"))

    # ── Stage 4: Blur quantification ──────────────────────────────────────────
    blur_before = None
    blur_after = None

    if args.blur_query_x is not None and args.blur_query_y is not None:
        query_coords = (args.blur_query_x, args.blur_query_y)
        print(f"\n[Stage 4] Blur quantification at query point {query_coords} …")

        # Quantify blur BEFORE correction (use middle frame)
        mid_idx = len(rs_frames) // 2
        blur_before = quantify_blur_spatial(
            rs_frames[mid_idx],
            model_path=args.blur_model,
            window_size=args.blur_window,
            stride=args.blur_stride,
            query_coords=query_coords,
        )
        blur_before.label = "blur_before_correction"
        blur_before.print_summary()
        save_blur_heatmap(blur_before, str(out / "blur_heatmap_before.png"),
                         "Blur BEFORE correction (px)")

        # Quantify blur AFTER correction
        blur_after = quantify_blur_spatial(
            corrected_frames[mid_idx],
            model_path=args.blur_model,
            window_size=args.blur_window,
            stride=args.blur_stride,
            query_coords=query_coords,
        )
        blur_after.label = "blur_after_correction"
        blur_after.print_summary()
        save_blur_heatmap(blur_after, str(out / "blur_heatmap_after.png"),
                         "Blur AFTER correction (px)")

        # Print blur reduction at query point
        if blur_before.query_sigma and blur_after.query_sigma:
            reduction = (1 - blur_after.query_sigma / blur_before.query_sigma) * 100
            print(f"\n{'='*52}")
            print(f"  Blur at {query_coords}:")
            print(f"    Before : {blur_before.query_sigma:.1f} px")
            print(f"    After  : {blur_after.query_sigma:.1f} px")
            print(f"    Reduction : {reduction:.1f}%")
            print(f"{'='*52}")

    # ── Stage 5: Quality assessment ───────────────────────────────────────────
    print("\n[Stage 5] Computing IQA quality scores …")
    quality_report = analyzer.analyze(
        rs_result=rs_result,
        persp_before=report_before,
        persp_after=report_after,
        blur_before=blur_before,
        blur_after=blur_after,
    )
    print(f"  IQA score before : {quality_report.score_before:.1f} / 100")
    print(f"  IQA score after  : {quality_report.score_after:.1f} / 100")
    print(f"  Correction gain  : {quality_report.gain_pct:+.1f}%")
    print(f"  Defect weights   : RS={quality_report.weight_rs:.2f}  "
          f"Persp={quality_report.weight_persp:.2f}  "
          f"Blur={quality_report.weight_blur:.2f}")
    analyzer.generate_summary_report(
        quality_report, str(out / "summary_report.png")
    )

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n[Report] Writing JSON summary …")
    data = save_report(report_before, report_after, params,
                       str(out / "report.json"),
                       blur_before, blur_after)

    # Merge quality assessment into report.json
    quality_dict = analyzer.to_dict(quality_report)
    data.update(quality_dict)
    with open(str(out / "report.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Updated report (with IQA) → {out / 'report.json'}")

    r = data["reduction_pct"]
    print(f"\n{'='*52}")
    print(f"  Mean warp reduction : {r['mean']:>6.1f}%")
    print(f"  P95  warp reduction : {r['p95']:>6.1f}%")
    print(f"  Max  warp reduction : {r['max']:>6.1f}%")

    if "blur_reduction_pct" in data:
        br = data["blur_reduction_pct"]
        print(f"\n  Mean blur reduction : {br['mean']:>6.1f}%")
        print(f"  P95  blur reduction : {br['p95']:>6.1f}%")
        print(f"  Max  blur reduction : {br['max']:>6.1f}%")

    qa = data["quality_assessment"]
    print(f"\n  IQA score before : {qa['iqa_score_before']:>6.2f} / 100")
    print(f"  IQA score after  : {qa['iqa_score_after']:>6.2f} / 100")
    print(f"  Correction gain  : {qa['correction_gain_pct']:>+6.2f}%")

    print(f"{'='*52}")
    print(f"\nAll outputs saved to: {out.resolve()}")


def main():
    p = argparse.ArgumentParser(
        description="Rolling-shutter + perspective-error cancel pipeline"
    )
    p.add_argument("--video",      required=True,               help="Input video path")
    p.add_argument("--gyroflow",   default=None,                help="*.gyroflow project (optional)")
    p.add_argument("--efl",        type=float, default=24.0,    help="EFL in mm")
    p.add_argument("--pitch",      type=float, default=0.00112, help="Pixel pitch in mm")
    p.add_argument("--shift-x",    type=float, default=0.0,     help="Lens/sensor shift X in mm")
    p.add_argument("--shift-y",    type=float, default=0.0,     help="Lens/sensor shift Y in mm")
    p.add_argument("--k1",         type=float, default=0.0,     help="Radial distortion k1")
    p.add_argument("--k2",         type=float, default=0.0,     help="Radial distortion k2")
    p.add_argument("--p1",         type=float, default=0.0,     help="Tangential distortion p1")
    p.add_argument("--p2",         type=float, default=0.0,     help="Tangential distortion p2")
    p.add_argument("--max-frames", type=int,   default=None,    help="Limit frames processed")
    p.add_argument("--out-dir",    default="./output",          help="Output directory")

    # Blur quantification options
    p.add_argument("--blur-query-x", type=int, default=None,    help="Query point X coordinate for blur measurement")
    p.add_argument("--blur-query-y", type=int, default=None,    help="Query point Y coordinate for blur measurement")
    p.add_argument("--blur-model",   default=None,              help="Path to blur-kernel-estimation model (optional)")
    p.add_argument("--blur-window",  type=int, default=32,      help="Sliding window size for blur estimation")
    p.add_argument("--blur-stride",  type=int, default=16,      help="Stride for sliding window")

    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
