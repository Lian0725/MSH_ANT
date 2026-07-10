#!/usr/bin/env python3
"""Render Wang Figure 3 panels directly from saved NPZ numeric data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


DEFAULT_OUTPUT = Path("/Users/lgx/Projects/MSH_ANT/outputs/work_wang_figure3_uniform_panels_from_npz_20260629")
PANEL_A = Path("/Users/lgx/Projects/MSH_ANT/outputs/work_1d_wang_figure3a_server_rerun_stack_20260623_regen/wang_figure3a_panel_data.npz")
PANEL_B = Path("/Users/lgx/Projects/MSH_ANT/outputs/lenovo_1d_wang_figure3b_phase_subset_20140723_25_20260620_regen/subset_strict_data.npz")
PANEL_C = Path("/Users/lgx/Projects/MSH_ANT/outputs/work_1d_wang_figure3a_server_rerun_stack_template_subtract_diagfit_20260625_regen/distance_bin_wiggle_panel_data.npz")
PANEL_D = Path("/Users/lgx/Projects/MSH_ANT/outputs/work_1d_wang_figure3d_fill_scaled_2p0_no_fill_cached_20260629_regen/wang_figure3d_bandpassed_fill_scaled_2p00_no_fill_data.npz")

PANEL_SIZE_IN = (5.4, 4.0)
PANEL_DPI = 600
TIME_LABEL = "Time (sec)"
DISTANCE_LABEL = "Distance (km)"
TOP_PADDING_KM = 0.85
BOTTOM_PADDING_KM = 1.00
PANELS_AC_SCALE_MULTIPLIER = 2.0


def report_local_defaults(script_path: Path | None = None) -> dict[str, Path] | None:
    path = (script_path or Path(__file__)).resolve()
    if path.parent.name != "wang_figure3_uniform":
        return None
    if path.parent.parent.name != "code":
        return None
    reports_root = path.parent.parent.parent
    image_root = reports_root / "images" / "wang_figure3_uniform"
    panel_a = image_root / "wang_figure3a_panel_data.npz"
    panel_b = image_root / "subset_strict_data.npz"
    panel_c = image_root / "distance_bin_wiggle_panel_data.npz"
    panel_d = image_root / "wang_figure3d_bandpassed_fill_scaled_2p00_no_fill_data.npz"
    if not all(asset.exists() for asset in (panel_a, panel_b, panel_c, panel_d)):
        return None
    return {
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_c": panel_c,
        "panel_d": panel_d,
        "output": image_root,
    }


def default_paths(script_path: Path | None = None) -> dict[str, Path]:
    local = report_local_defaults(script_path)
    if local is not None:
        return local
    return {
        "panel_a": PANEL_A,
        "panel_b": PANEL_B,
        "panel_c": PANEL_C,
        "panel_d": PANEL_D,
        "output": DEFAULT_OUTPUT,
    }


def figure_rcparams() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }


def create_panel_figure():
    plt.rcParams.update(figure_rcparams())
    fig, ax = plt.subplots(figsize=PANEL_SIZE_IN)
    fig.subplots_adjust(left=0.16, right=0.965, bottom=0.16, top=0.93)
    return fig, ax


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PANEL_DPI)
    plt.close(fig)
    return path


def add_common_ticks(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True, length=3, width=0.8)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.01, 1.02, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)


def plot_distance_panel(ax: plt.Axes, *, time: np.ndarray, data: np.ndarray, valid: np.ndarray, bin_width_km: float, amplitude_scale: float, label: str, guide_offset_s: float | None = None, guide_seconds: range | None = None, half_second_guides: bool = False, line_width: float = 0.38) -> None:
    for index in valid:
        trace = data[index]
        if not np.all(np.isfinite(trace)):
            continue
        distance = (index + 0.5) * bin_width_km
        ax.plot(time, distance + trace * bin_width_km * amplitude_scale, color="black", lw=line_width)
    if guide_seconds is not None and guide_offset_s is not None:
        for second in guide_seconds:
            ax.axvline(second + guide_offset_s, color="#dc2626", lw=0.25, alpha=0.25)
    if half_second_guides:
        for second in range(0, 17):
            ax.axvline(second + 0.5, color="#dc2626", lw=0.25, alpha=0.25)
    ax.set_xlim(0, 16)
    ax.set_ylim(25 + BOTTOM_PADDING_KM, -TOP_PADDING_KM)
    ax.set_xticks(np.arange(0, 17, 2))
    ax.set_yticks(np.arange(0, 26, 5))
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(DISTANCE_LABEL)
    add_common_ticks(ax)
    add_panel_label(ax, label)


def render_panel_a(npz_path: Path, output_path: Path) -> Path:
    payload = np.load(npz_path)
    fig, ax = create_panel_figure()
    plot_distance_panel(
        ax,
        time=payload["time"],
        data=payload["diagnostic"],
        valid=payload["valid"],
        bin_width_km=float(payload["bin_width_km"]),
        amplitude_scale=float(payload["display_amplitude_scale"]) * PANELS_AC_SCALE_MULTIPLIER,
        label="(a)",
        half_second_guides=True,
        line_width=0.38,
    )
    return save_figure(fig, output_path)


def render_panel_b(npz_path: Path, output_path: Path) -> Path:
    payload = np.load(npz_path)
    fig, ax = create_panel_figure()
    ax.plot(payload["time_s"], payload["waveform"], color="#0f172a", linewidth=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(str(payload["ylabel"]))
    add_common_ticks(ax)
    add_panel_label(ax, "(b)")
    return save_figure(fig, output_path)


def render_panel_c(npz_path: Path, output_path: Path) -> Path:
    payload = np.load(npz_path)
    fig, ax = create_panel_figure()
    plot_distance_panel(
        ax,
        time=payload["time"],
        data=payload["corrected_after"],
        valid=payload["valid"],
        bin_width_km=float(payload["bin_width_km"]),
        amplitude_scale=float(payload["display_amplitude_scale"]) * PANELS_AC_SCALE_MULTIPLIER,
        label="(c)",
        guide_offset_s=float(payload["best_phase_s"]),
        guide_seconds=range(1, 16),
        line_width=0.38,
    )
    return save_figure(fig, output_path)


def render_panel_d(npz_path: Path, output_path: Path) -> Path:
    payload = np.load(npz_path)
    fig, ax = create_panel_figure()
    time = payload["time"]
    data = payload["display_bins"]
    valid = payload["valid"]
    bin_width_km = float(payload["bin_width_km"])
    amplitude_scale = float(payload["amplitude_scale"])
    line_width = float(payload["line_width"])
    for index in valid:
        trace = data[index]
        if not np.all(np.isfinite(trace)):
            continue
        distance = (index + 0.5) * bin_width_km
        display = distance + trace * bin_width_km * amplitude_scale
        ax.plot(time, display, color="black", lw=line_width)
    ax.set_xlim(0, 16)
    ax.set_ylim(
        25 + max(float(payload["bottom_padding_km"]), BOTTOM_PADDING_KM),
        -max(float(payload["top_padding_km"]), TOP_PADDING_KM),
    )
    ax.set_xticks(np.arange(0, 17, 2))
    ax.set_yticks(np.arange(0, 26, 5))
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(DISTANCE_LABEL)
    add_common_ticks(ax)
    add_panel_label(ax, "(d)")
    return save_figure(fig, output_path)


def compose_uniform_grid(panel_paths: list[Path], output_path: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in panel_paths]
    try:
        width, height = images[0].size
        if any(image.size != (width, height) for image in images[1:]):
            raise ValueError(f"Panel sizes do not match: {[image.size for image in images]}")
        gap = 56
        margin = 40
        canvas = Image.new("RGB", (margin * 2 + width * 2 + gap, margin * 2 + height * 2 + gap), "white")
        positions = [
            (margin, margin),
            (margin + width + gap, margin),
            (margin, margin + height + gap),
            (margin + width + gap, margin + height + gap),
        ]
        for image, pos in zip(images, positions):
            canvas.paste(image, pos)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return output_path
    finally:
        for image in images:
            image.close()


def render_all_panels(*, panel_a: Path, panel_b: Path, panel_c: Path, panel_d: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        render_panel_a(panel_a, output_dir / "panel_a.png"),
        render_panel_b(panel_b, output_dir / "panel_b.png"),
        render_panel_c(panel_c, output_dir / "panel_c.png"),
        render_panel_d(panel_d, output_dir / "panel_d.png"),
    ]
    return compose_uniform_grid(outputs, output_dir / "wang_figure3_four_panel_uniform.png")


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-a", type=Path, default=defaults["panel_a"])
    parser.add_argument("--panel-b", type=Path, default=defaults["panel_b"])
    parser.add_argument("--panel-c", type=Path, default=defaults["panel_c"])
    parser.add_argument("--panel-d", type=Path, default=defaults["panel_d"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = render_all_panels(
        panel_a=args.panel_a,
        panel_b=args.panel_b,
        panel_c=args.panel_c,
        panel_d=args.panel_d,
        output_dir=args.output,
    )
    print(combined)


if __name__ == "__main__":
    main()
