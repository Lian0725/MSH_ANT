#!/usr/bin/env python3
"""Replot Wang Figure 4 with a no-distance-filter left column.

The existing Figure 4 right column is deliberately read, not recomputed.  This
keeps its cycle correction, one-wavelength selection, fit, and numerical
annotations bit-for-bit tied to the approved EGF-convention result.  The left
column combines the existing measurements with a separately reprocessed set of
short paths that passed only the Wang SNR and group-velocity conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PERIODS_S = (3.0, 4.0, 5.0)
# Measured from the Wang et al. Figure 4 panels: height / width ≈ 1.23.
WANG_AXIS_BOX_ASPECT = 1.23
WANG_FONT_FAMILY = "Times New Roman"
DEFAULT_ROOT = Path(
    "/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_dat_20260724"
)


def row_key(row: Mapping[str, object]) -> tuple[str, float]:
    """Return the immutable identity used to de-duplicate a pair-period row."""

    return str(row["pair_name"]), float(row["period_s"])


def merge_left_only_rows(
    original_left_rows: Iterable[Mapping[str, object]],
    reprocessed_short_rows: Iterable[Mapping[str, object]],
    original_right_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], Sequence[Mapping[str, object]]]:
    """Append only missing short-path rows while preserving right-column data."""

    merged = [dict(row) for row in original_left_rows]
    present = {row_key(row) for row in merged}
    for row in reprocessed_short_rows:
        key = row_key(row)
        if key not in present:
            merged.append(dict(row))
            present.add(key)
    return merged, original_right_rows


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_raw_rows(rows: Iterable[Mapping[str, object]], period_s: float):
    return [
        dict(row)
        for row in rows
        if float(row["period_s"]) == period_s
        and math.isfinite(float(row["raw_travel_time_s"]))
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_wang_figure4_axis_aspect(axes: Iterable[object]) -> None:
    """Make every Figure-4 panel match Wang's tall plotting box."""

    for axis in axes:
        axis.set_box_aspect(WANG_AXIS_BOX_ASPECT)


def selected_cluster_boundary_offsets(period_s: float) -> tuple[float, float]:
    """Return the two half-period boundaries enclosing the selected branch."""

    return (-0.5 * period_s, 0.5 * period_s)


def configure_wang_font(font_path: Path, plt: object) -> None:
    """Register and select the exact font requested for the Wang reproduction."""

    if not font_path.is_file():
        raise FileNotFoundError(f"Times New Roman font file is missing: {font_path}")
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = WANG_FONT_FAMILY


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "egf_convention_check_no_preleft",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path(__file__).resolve().parent / "Times New Roman.ttf",
    )
    args = parser.parse_args()
    configure_wang_font(args.font_path, plt)

    root = args.root
    original_dir = root / "egf_convention_check"
    base_measurements = root / "fixed_bensen_alpha12_b1_b2_1/target_measurements.jsonl"
    short_measurements = args.output_dir / "short_measurements.jsonl"
    original_right = original_dir / "measurements_corrected.jsonl"
    original_metadata = original_dir / "metadata.json"
    required = (base_measurements, short_measurements, original_right, original_metadata)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required input is missing: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_jsonl(base_measurements)
    recovered_rows = read_jsonl(short_measurements)
    original_right_rows = read_jsonl(original_right)
    merged_left_rows, right_rows = merge_left_only_rows(
        base_rows, recovered_rows, original_right_rows
    )
    metadata = json.loads(original_metadata.read_text())
    summary_by_period = {
        float(row["period_s"]): row for row in metadata["summary"]
    }

    figure, axes = plt.subplots(3, 2, figsize=(6.8, 11.5), sharex=True, sharey=True)
    set_wang_figure4_axis_aspect(axes.flat)
    output_summary = []
    for index, period_s in enumerate(PERIODS_S):
        left_rows = finite_raw_rows(merged_left_rows, period_s)
        right_period_rows = [
            row
            for row in right_rows
            if float(row["period_s"]) == period_s
            and bool(row["one_wavelength_accepted"])
        ]
        reference = summary_by_period[period_s]
        v_ref = float(reference["reference_velocity_km_s"])
        v_fit = float(reference["right_fit_velocity_km_s"])
        v_std = float(reference["right_phase_velocity_std_km_s"])
        left_distance = np.asarray([row["distance_km"] for row in left_rows], dtype=float)
        left_time = np.asarray(
            [float(row["raw_travel_time_s"]) + period_s / 4.0 for row in left_rows],
            dtype=float,
        )
        right_distance = np.asarray(
            [row["distance_km"] for row in right_period_rows], dtype=float
        )
        right_time = np.asarray(
            [row["corrected_travel_time_s"] for row in right_period_rows], dtype=float
        )
        left_axis, right_axis = axes[index]
        left_axis.plot(left_distance, left_time, ".", ms=1.5, alpha=0.3, color="blue")
        right_axis.plot(right_distance, right_time, ".", ms=1.5, alpha=0.3, color="blue")
        x_line = np.asarray([0.0, 25.0])
        right_axis.plot(x_line, x_line / v_fit, "k-", lw=1.0)
        for axis in (left_axis, right_axis):
            for offset in selected_cluster_boundary_offsets(period_s):
                axis.plot(x_line, x_line / v_ref + offset, "--", color="limegreen", lw=0.8)
            axis.set_xlim(0, 25)
            axis.set_ylim(0, 10)
            axis.grid(alpha=0.25, ls=":")
        left_axis.set_ylabel("Travel Time (s)")
        left_axis.text(0.04, 0.93, f"({'abc'[index]}) {period_s:g} s", transform=left_axis.transAxes, fontsize=11)
        right_axis.text(0.04, 0.93, f"V = {v_fit:.2f} km/s\nSTDV = {v_std:.2f} km/s", transform=right_axis.transAxes, fontsize=10, va="top")
        output_summary.append(
            {
                "period_s": period_s,
                "left_count_no_preleft": len(left_rows),
                "right_count_reused": len(right_period_rows),
                "reference_velocity_km_s_reused": v_ref,
                "right_fit_velocity_km_s_reused": v_fit,
                "right_phase_velocity_std_km_s_reused": v_std,
            }
        )
    axes[-1, 0].set_xlabel("Distance (km)")
    axes[-1, 1].set_xlabel("Distance (km)")
    figure.suptitle("Wang et al. (2017) Figure 4 reproduction - no distance filter in left column", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(args.output_dir / "wang_figure4_egf_no_preleft.png", dpi=220)
    plt.close(figure)

    with (args.output_dir / "left_measurements_no_preleft.jsonl").open("w", encoding="utf-8") as handle:
        for row in merged_left_rows:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "fit_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_summary[0]))
        writer.writeheader()
        writer.writerows(output_summary)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "left_column": "all SNR/group-velocity-qualified measurements; no distance filter",
                "right_column": "reused unchanged from egf_convention_check",
                "source_sha256": {
                    "base_measurements": sha256(base_measurements),
                    "short_measurements": sha256(short_measurements),
                    "right_measurements": sha256(original_right),
                },
                "summary": output_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
