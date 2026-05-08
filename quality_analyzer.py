"""
quality_analyzer.py
===================
Unified Image Quality Assessment (IQA) module that aggregates rolling-shutter,
perspective, and blur metrics into a single 0-100 score, composition weights,
before/after correction gain, and a summary dashboard PNG.

Usage
-----
  from quality_analyzer import QualityAnalyzer

  analyzer = QualityAnalyzer()
  report = analyzer.analyze(
      rs_result=rs_result,
      persp_before=report_before,
      persp_after=report_after,
      blur_before=blur_before,
      blur_after=blur_after,
  )
  print(f"IQA: {report.score_before:.1f} → {report.score_after:.1f}")
  analyzer.generate_summary_report(report, "output/summary_report.png")
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

from rolling_shutter import RSCorrectionResult
from perspective_quantifier import PerspErrorReport
from blur_quantifier import BlurReportFFT, BlurReportSpatial


@dataclass
class QualityReport:
    rs_loss_before: float        # [0,1] rolling-shutter loss before correction
    persp_loss_before: float     # [0,1] perspective loss before correction
    blur_loss_before: float      # [0,1] blur loss before correction (0.0 if unavailable)
    blur_available: bool
    rs_loss_after: float
    persp_loss_after: float
    blur_loss_after: float
    weight_rs: float             # composition weight, sums to 1.0 with others
    weight_persp: float
    weight_blur: float
    score_before: float          # [0,100] IQA score
    score_after: float
    gain_pct: float              # percentage improvement
    component_scores: dict       # {"RS": float, "Perspective": float, "Blur": float|None}


class QualityAnalyzer:
    # ── Physical constants (tunable) ──────────────────────────────────────────
    RS_BAD_THRESHOLD_PX  = 10.0    # corner warp = "max bad" for RS
    RS_GYRO_ASSUMED_LOSS = 0.05    # gyroflow leaves ~5% residual
    PERSP_SCALE          = 4.33    # exponential scale: 5px combined → loss 0.68
    BLUR_SCALE           = 2.16    # exponential scale: 3px sigma → loss 0.75
    POWER                = 1.5     # power-law for composition emphasis
    PERCEPTUAL_WEIGHTS   = np.array([0.8, 1.2, 1.0])  # RS, Perspective, Blur

    # ── Normalization helpers ─────────────────────────────────────────────────

    def _rs_loss(self, rs_result: RSCorrectionResult) -> float:
        """Compute normalized RS loss in [0, 1]."""
        if rs_result.residual_map is None:
            # gyroflow path: assume small residual
            return self.RS_GYRO_ASSUMED_LOSS
        mean_val = float(np.mean(rs_result.residual_map))
        return float(np.clip(mean_val / self.RS_BAD_THRESHOLD_PX, 0.0, 1.0))

    def _persp_loss(self, report: PerspErrorReport) -> float:
        """Compute normalized perspective loss in [0, 1]."""
        combined = 0.6 * report.mean_warp_px + 0.4 * report.p95_warp_px
        return float(1.0 - np.exp(-combined / self.PERSP_SCALE))

    def _blur_loss(
        self,
        blur_spatial: Optional[BlurReportSpatial],
        blur_fft: Optional[BlurReportFFT],
    ) -> float:
        """Compute normalized blur loss in [0, 1]."""
        if blur_spatial is not None:
            combined = 0.6 * blur_spatial.mean_sigma + 0.4 * blur_spatial.p95_sigma
            return float(1.0 - np.exp(-combined / self.BLUR_SCALE))
        if blur_fft is not None:
            return float((100.0 - blur_fft.sharpness_score) / 100.0)
        return 0.0

    # ── Composition weights ───────────────────────────────────────────────────

    def analyze_composition(
        self,
        rs_loss: float,
        persp_loss: float,
        blur_loss: float,
        blur_available: bool,
    ) -> tuple[float, float, float]:
        """Return (w_rs, w_persp, w_blur) summing to 1.0."""
        losses = np.array([rs_loss, persp_loss, blur_loss], dtype=float)

        # Mask blur if not available
        perceptual = self.PERCEPTUAL_WEIGHTS.copy()
        if not blur_available:
            perceptual[2] = 0.0

        raw = (losses ** self.POWER) * perceptual
        total = float(np.sum(raw))

        if total < 1e-12:
            # All losses near zero → equal split
            if blur_available:
                return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            else:
                return (0.5, 0.5, 0.0)

        weights = raw / total
        return (float(weights[0]), float(weights[1]), float(weights[2]))

    # ── IQA score ─────────────────────────────────────────────────────────────

    def calculate_total_score(
        self,
        w_rs: float, rs_loss: float,
        w_persp: float, persp_loss: float,
        w_blur: float, blur_loss: float,
    ) -> float:
        """Return IQA score in [0, 100]."""
        weighted_loss = w_rs * rs_loss + w_persp * persp_loss + w_blur * blur_loss
        return float(np.clip(100.0 * (1.0 - weighted_loss), 0.0, 100.0))

    # ── Main entry ────────────────────────────────────────────────────────────

    def analyze(
        self,
        rs_result: RSCorrectionResult,
        persp_before: PerspErrorReport,
        persp_after: PerspErrorReport,
        blur_before: Optional[BlurReportSpatial] = None,
        blur_after: Optional[BlurReportSpatial] = None,
        blur_fft_before: Optional[BlurReportFFT] = None,
        blur_fft_after: Optional[BlurReportFFT] = None,
    ) -> QualityReport:
        blur_available = (blur_before is not None) or (blur_fft_before is not None)

        # ── Compute losses ────────────────────────────────────────────────────
        rs_loss = self._rs_loss(rs_result)
        persp_loss_before = self._persp_loss(persp_before)
        persp_loss_after  = self._persp_loss(persp_after)
        blur_loss_before  = self._blur_loss(blur_before, blur_fft_before)
        blur_loss_after   = self._blur_loss(blur_after,  blur_fft_after)

        # RS loss does not change after perspective correction
        rs_loss_after = rs_loss

        # ── Composition weights from BEFORE losses (diagnosis of original) ────
        w_rs, w_persp, w_blur = self.analyze_composition(
            rs_loss, persp_loss_before, blur_loss_before, blur_available
        )

        # ── IQA scores ────────────────────────────────────────────────────────
        score_before = self.calculate_total_score(
            w_rs, rs_loss, w_persp, persp_loss_before, w_blur, blur_loss_before
        )
        score_after = self.calculate_total_score(
            w_rs, rs_loss_after, w_persp, persp_loss_after, w_blur, blur_loss_after
        )
        gain_pct = (score_after - score_before) / max(score_before, 1.0) * 100.0

        # ── Per-component scores (after) ──────────────────────────────────────
        component_scores = {
            "RS":          float(np.clip(100.0 * (1.0 - rs_loss_after),        0, 100)),
            "Perspective": float(np.clip(100.0 * (1.0 - persp_loss_after),     0, 100)),
            "Blur":        float(np.clip(100.0 * (1.0 - blur_loss_after),      0, 100))
                           if blur_available else None,
        }

        return QualityReport(
            rs_loss_before=rs_loss,
            persp_loss_before=persp_loss_before,
            blur_loss_before=blur_loss_before,
            blur_available=blur_available,
            rs_loss_after=rs_loss_after,
            persp_loss_after=persp_loss_after,
            blur_loss_after=blur_loss_after,
            weight_rs=w_rs,
            weight_persp=w_persp,
            weight_blur=w_blur,
            score_before=score_before,
            score_after=score_after,
            gain_pct=gain_pct,
            component_scores=component_scores,
        )

    # ── Visualization ─────────────────────────────────────────────────────────

    def generate_summary_report(self, report: QualityReport, path: str) -> None:
        """Generate a 3-panel summary dashboard and save to path."""
        fig = plt.figure(figsize=(14, 5), dpi=150, facecolor="#1a1a1a")

        hot = plt.cm.hot

        # ── Panel 1: Pie chart — defect composition ───────────────────────────
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_facecolor("#1a1a1a")

        labels = ["RS", "Perspective"]
        contributions = [
            report.weight_rs * report.rs_loss_before,
            report.weight_persp * report.persp_loss_before,
        ]
        colors = [hot(0.25), hot(0.55)]

        if report.blur_available:
            labels.append("Blur")
            contributions.append(report.weight_blur * report.blur_loss_before)
            colors.append(hot(0.80))

        # Remove zero-contribution slices to avoid matplotlib warnings
        filtered = [(l, c, col) for l, c, col in zip(labels, contributions, colors) if c > 1e-9]
        if filtered:
            labels, contributions, colors = zip(*filtered)
        else:
            labels, contributions, colors = ["No defects"], [1.0], [hot(0.55)]

        explode = [0.05] * len(labels)
        wedges, texts, autotexts = ax1.pie(
            contributions,
            labels=labels,
            colors=colors,
            explode=explode,
            autopct="%1.1f%%",
            textprops={"color": "white", "fontsize": 9},
            wedgeprops={"linewidth": 0.5, "edgecolor": "#1a1a1a"},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(8)
        ax1.set_title("Defect Contribution", color="white", fontsize=10, pad=8)

        # ── Panel 2: Half-donut gauge ─────────────────────────────────────────
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_facecolor("#1a1a1a")
        ax2.set_aspect("equal")
        ax2.axis("off")

        score = report.score_after
        color = plt.cm.RdYlGn(score / 100.0)

        # Background arc
        theta_bg = np.linspace(np.pi, 0, 200)
        ax2.plot(np.cos(theta_bg), np.sin(theta_bg),
                 color="#333333", linewidth=18, solid_capstyle="round")

        # Filled arc up to score
        frac = score / 100.0
        theta_fill = np.linspace(np.pi, np.pi - frac * np.pi, 200)
        ax2.plot(np.cos(theta_fill), np.sin(theta_fill),
                 color=color, linewidth=18, solid_capstyle="round")

        # Needle
        angle = np.pi - frac * np.pi
        ax2.annotate(
            "",
            xy=(0.75 * np.cos(angle), 0.75 * np.sin(angle)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2,
                            mutation_scale=15),
        )
        ax2.plot(0, 0, "o", color="white", markersize=6, zorder=5)

        # Score text
        ax2.text(0, -0.25, f"{score:.1f}",
                 ha="center", va="center", color="white",
                 fontsize=22, fontweight="bold")
        ax2.text(0, -0.45, "/ 100",
                 ha="center", va="center", color="#aaaaaa", fontsize=10)
        ax2.text(0, -0.65,
                 f"Before: {report.score_before:.1f}  |  Gain: {report.gain_pct:+.1f}%",
                 ha="center", va="center", color="#cccccc", fontsize=8)

        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-0.8, 1.2)
        ax2.set_title("IQA Score (After)", color="white", fontsize=10, pad=8)

        # ── Panel 3: Grouped bar chart — per-component scores ─────────────────
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_facecolor("#1a1a1a")

        components = ["RS", "Perspective"]
        scores_before = [
            float(np.clip(100.0 * (1.0 - report.rs_loss_before),        0, 100)),
            float(np.clip(100.0 * (1.0 - report.persp_loss_before),     0, 100)),
        ]
        scores_after_ = [
            report.component_scores["RS"],
            report.component_scores["Perspective"],
        ]

        if report.blur_available:
            components.append("Blur")
            scores_before.append(float(np.clip(100.0 * (1.0 - report.blur_loss_before), 0, 100)))
            scores_after_.append(report.component_scores["Blur"])

        x = np.arange(len(components))
        width = 0.35

        bars_b = ax3.bar(x - width / 2, scores_before, width,
                         color=hot(0.35), label="Before")
        bars_a = ax3.bar(x + width / 2, scores_after_, width,
                         color=hot(0.65), label="After")

        # Value labels on bars
        for bar in bars_b:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1,
                     f"{h:.0f}", ha="center", va="bottom",
                     color="white", fontsize=8)
        for bar in bars_a:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1,
                     f"{h:.0f}", ha="center", va="bottom",
                     color="white", fontsize=8)

        ax3.set_xticks(x)
        ax3.set_xticklabels(components, color="white", fontsize=9)
        ax3.set_yticks(range(0, 101, 20))
        ax3.set_yticklabels([str(v) for v in range(0, 101, 20)], color="#aaaaaa", fontsize=8)
        ax3.set_ylim(0, 115)
        ax3.set_title("Component Scores", color="white", fontsize=10, pad=8)
        ax3.tick_params(colors="white")
        ax3.spines[:].set_color("#444444")
        legend = ax3.legend(fontsize=8, facecolor="#2a2a2a",
                            edgecolor="#555555", labelcolor="white")

        fig.suptitle("Pipeline Quality Assessment Report",
                     color="white", fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig)
        print(f"  Saved summary_report → {path}")

    # ── JSON serialization ────────────────────────────────────────────────────

    def to_dict(self, report: QualityReport) -> dict:
        """Return dict with key 'quality_assessment' suitable for JSON merge."""
        return {
            "quality_assessment": {
                "losses_before": {
                    "rs":          round(report.rs_loss_before,    4),
                    "perspective": round(report.persp_loss_before, 4),
                    "blur":        round(report.blur_loss_before,  4),
                },
                "losses_after": {
                    "rs":          round(report.rs_loss_after,    4),
                    "perspective": round(report.persp_loss_after, 4),
                    "blur":        round(report.blur_loss_after,  4),
                },
                "composition_weights": {
                    "rs":          round(report.weight_rs,    4),
                    "perspective": round(report.weight_persp, 4),
                    "blur":        round(report.weight_blur,  4),
                },
                "iqa_score_before":    round(report.score_before, 2),
                "iqa_score_after":     round(report.score_after,  2),
                "correction_gain_pct": round(report.gain_pct,     2),
                "component_scores_after": {
                    "RS":          round(report.component_scores["RS"],          2),
                    "Perspective": round(report.component_scores["Perspective"], 2),
                    "Blur":        round(report.component_scores["Blur"],        2)
                                   if report.component_scores["Blur"] is not None else None,
                },
                "blur_data_available": report.blur_available,
            }
        }
