#!/usr/bin/env python3
"""Compose four already-regenerated Wang Figure 3 panels into one 2x2 plate."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


DEFAULT_OUTPUT = Path("/Users/lgx/Projects/MSH_ANT/outputs/work_wang_figure3_uniform_panels_regenerated_20260629")
SOURCE_A = Path("/Users/lgx/Projects/MSH_ANT/outputs/work_1d_wang_figure3a_server_rerun_stack_20260623_regen/wang_figure3a_style_diagnostic_residual.png")
SOURCE_B = Path("/Users/lgx/Projects/MSH_ANT/outputs/lenovo_1d_wang_figure3b_phase_subset_20140723_25_20260620_regen/subset_strict.png")
SOURCE_C = Path("/Users/lgx/Projects/MSH_ANT/outputs/work_1d_wang_figure3a_server_rerun_stack_template_subtract_diagfit_20260625_regen/distance_bin_wiggle_after_only.png")
SOURCE_D = Path("/Users/lgx/Projects/MSH_ANT/outputs/work_1d_wang_figure3d_fill_scaled_2p0_no_fill_cached_20260629_regen/wang_figure3d_bandpassed_fill_scaled_2p00_no_fill.png")


def report_local_defaults(script_path: Path | None = None) -> dict[str, Path] | None:
    path = (script_path or Path(__file__)).resolve()
    if path.parent.name != "wang_figure3_uniform":
        return None
    if path.parent.parent.name != "code":
        return None
    reports_root = path.parent.parent.parent
    image_root = reports_root / "images" / "wang_figure3_uniform"
    panel_a = image_root / "panel_a.png"
    panel_b = image_root / "panel_b.png"
    panel_c = image_root / "panel_c.png"
    panel_d = image_root / "panel_d.png"
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
        "panel_a": SOURCE_A,
        "panel_b": SOURCE_B,
        "panel_c": SOURCE_C,
        "panel_d": SOURCE_D,
        "output": DEFAULT_OUTPUT,
    }


def panel_data_path(figure_path: Path) -> Path:
    return figure_path.with_name(f"{figure_path.stem}_data.npz")


def ensure_uniform_sizes(paths: list[Path], ratio_tolerance: float = 1.0e-3) -> tuple[int, int]:
    sizes = []
    ratios = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
            ratios.append(image.size[0] / image.size[1])
    first_ratio = ratios[0]
    if any(abs(ratio - first_ratio) > ratio_tolerance for ratio in ratios[1:]):
        raise ValueError(f"Panel ratios are not identical enough: {ratios}")
    return max(width for width, _ in sizes), max(height for _, height in sizes)


def compose_grid(
    panel_paths: list[Path],
    output: Path,
    *,
    gap: int = 56,
    outer_margin: int = 40,
) -> Path:
    panel_width, panel_height = ensure_uniform_sizes(panel_paths)
    width = outer_margin * 2 + gap + panel_width * 2
    height = outer_margin * 2 + gap + panel_height * 2
    canvas = Image.new("RGB", (width, height), "white")
    positions = [
        (outer_margin, outer_margin),
        (outer_margin + panel_width + gap, outer_margin),
        (outer_margin, outer_margin + panel_height + gap),
        (outer_margin + panel_width + gap, outer_margin + panel_height + gap),
    ]
    for path, (x, y) in zip(panel_paths, positions):
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            canvas.paste(resized, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


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
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    panel_paths = [args.panel_a, args.panel_b, args.panel_c, args.panel_d]
    for source, name in zip(panel_paths, ["panel_a.png", "panel_b.png", "panel_c.png", "panel_d.png"]):
        target = output / name
        target.write_bytes(source.read_bytes())
    combined = compose_grid(panel_paths, output / "wang_figure3_four_panel_uniform.png")
    print(combined)


if __name__ == "__main__":
    main()
