#!/usr/bin/env python3
"""Build reproducible static figures for the A2CR/CAR paper and Habr article."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS = ROOT / "benchmark_results" / "publication-statistics.json"
DEFAULT_OUT = ROOT / "SimpleDeHaze" / "docs" / "research" / "figures"

BLUE = "#276FBF"
ORANGE = "#D97706"
INK = "#20262E"
GRID = "#D9DEE5"
ZERO = "#59636E"


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_diode_intervals(stats_path: Path, out_dir: Path) -> None:
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    records = [
        row for row in payload["records"]
        if row["analysis"] == "diode_frame_cluster_bootstrap"
        and row["contrast"] in {"B9 - B0", "B9 - B8"}
    ]
    metrics = [
        ("psnr", "PSNR (dB)", True),
        ("ssim", "SSIM", True),
        ("ciede2000", "CIEDE2000", False),
        ("chroma_error", "Lab chroma error", False),
        ("flat_noise_output_sigma", "Flat-region noise sigma", False),
        ("ms", "CPU time (ms)", False),
    ]
    by_key = {(row["contrast"], row["metric"]): row for row in records}

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.2), constrained_layout=True)
    for ax, (metric, label, higher) in zip(axes.flat, metrics):
        for y, (contrast, color) in enumerate((("B9 - B0", BLUE), ("B9 - B8", ORANGE))):
            row = by_key[(contrast, metric)]
            x = row["difference"]
            ax.errorbar(
                x,
                y,
                xerr=[[x - row["ci95_low"]], [row["ci95_high"] - x]],
                fmt="o",
                color=color,
                markeredgecolor=INK,
                markeredgewidth=0.6,
                capsize=3,
                linewidth=2,
                markersize=6,
                zorder=3,
            )
        ax.axvline(0, color=ZERO, linewidth=1, linestyle="--", zorder=1)
        ax.set_yticks([0, 1], ["B9 - B0", "B9 - B8"])
        ax.set_title(label, color=INK, fontsize=10.5, loc="left", pad=7)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color(GRID)
        direction = "higher is better" if higher else "lower is better"
        ax.set_xlabel(f"target - baseline; {direction}", fontsize=7.5, color=ZERO)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))

    fig.suptitle("DIODE controlled benchmark: paired frame-level effects", fontsize=15, color=INK)
    fig.text(
        0.5,
        -0.015,
        "Mean difference and 95% stratified bootstrap CI; 500 frames, 45 haze recipes averaged within frame. "
        "Runtime is the expected trade-off.",
        ha="center",
        fontsize=8.5,
        color=ZERO,
    )
    _save(fig, out_dir / "diode-bootstrap-effects")


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_scene8_montage(out_dir: Path) -> None:
    default_dir = ROOT / "benchmark_results" / "scene8-car-final-default-800"
    tuned_dir = ROOT / "benchmark_results" / "scene8-car-reference-autotuned-final-800"
    car_name = "01_Chromatic_Airlight_Residual_(CAR-Dehaze,_эксперимент)"
    full = [
        _open_rgb(default_dir / "00_hazy_full.png"),
        _open_rgb(default_dir / f"{car_name}_full.png"),
        _open_rgb(tuned_dir / f"{car_name}_full.png"),
        _open_rgb(default_dir / "00_gt_full.png"),
    ]
    crops = [
        _open_rgb(default_dir / "00_hazy_wall.png"),
        _open_rgb(default_dir / f"{car_name}_wall.png"),
        _open_rgb(tuned_dir / f"{car_name}_wall.png"),
        _open_rgb(default_dir / "00_gt_wall.png"),
    ]
    titles = ["Hazy input", "CAR default", "CAR globally tuned", "Ground truth"]
    subtitles = [
        "wall a* = -0.45",
        "wall a* = 1.81",
        "wall a* = -0.03",
        "wall a* = 2.44",
    ]

    fig = plt.figure(figsize=(13.8, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[3.6, 1.25])
    for col, (image, crop, title, subtitle) in enumerate(zip(full, crops, titles, subtitles)):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(image)
        ax.set_title(title, fontsize=11, color=INK, pad=6)
        ax.axis("off")
        crop_ax = fig.add_subplot(grid[1, col])
        crop_ax.imshow(crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST))
        crop_ax.set_title(subtitle, fontsize=9, color=INK, pad=4)
        crop_ax.axis("off")
        for spine in crop_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(ORANGE if col in (1, 3) else GRID)
            spine.set_linewidth(2 if col in (1, 3) else 1)

    fig.suptitle("Scene 08: global fidelity and subtle red-wall recovery are different objectives", fontsize=14.5, color=INK)
    fig.text(
        0.5,
        -0.018,
        "The fixed default restores the wall's positive a* more faithfully; reference-guided global tuning improves full-image "
        "PSNR/SSIM/CIEDE2000 but weakens this local cue. Scene 08 is a development case, not a blind test.",
        ha="center",
        fontsize=8.5,
        color=ZERO,
    )
    _save(fig, out_dir / "scene8-car-wall")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_diode_intervals(args.stats.resolve(), args.out.resolve())
    build_scene8_montage(args.out.resolve())
    print(args.out.resolve())


if __name__ == "__main__":
    main()
