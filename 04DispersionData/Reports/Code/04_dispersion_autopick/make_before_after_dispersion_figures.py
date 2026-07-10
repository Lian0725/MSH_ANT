#!/usr/bin/env python3
"""Create before/after dispersion-energy comparison figures for matched station pairs."""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}


@dataclass(frozen=True)
class PairPaths:
    pair: str
    non_npz: Path
    rem_npz: Path
    non_g: Path
    non_c: Path
    rem_g: Path
    rem_c: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path("/mnt/data_hdd/MSH_ANT_Final"),
        help="Root containing 03CC_StackData and 04DispersionData",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data_hdd/MSH_ANT_Final/resources/figures/2014_dispersion_before_after"),
    )
    parser.add_argument("--figure-count", type=int, default=16)
    parser.add_argument(
        "--candidate-sample-size",
        type=int,
        default=6000,
        help="Random sample size used to rank interesting pairs.",
    )
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument(
        "--min-valid-phase-points",
        type=int,
        default=6,
        help="Minimum non-zero CDisp points required in both before/after curves.",
    )
    return parser.parse_args()


def paths_from_root(final_root: Path):
    non_base = final_root / "04DispersionData" / "2014" / "1D" / "NonRemoveSpikes"
    rem_base = final_root / "04DispersionData" / "2014" / "1D" / "RemoveSpikes"
    return {
        "non_npz_dir": non_base / "DispersionNPZ",
        "rem_npz_dir": rem_base / "DispersionNPZ" / "full_pixel_data_all",
        "non_curves_dir": non_base / "Curves" / "curves_all_finalct001",
        "rem_curves_dir": rem_base / "Curves" / "curves_all_finalct001",
    }


def load_curve(curve_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    periods = []
    velocities = []
    snr = []
    with curve_path.open("r", encoding="utf-8") as handle:
        lines = handle.read().strip().splitlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) != 4:
            continue
        periods.append(float(parts[0]))
        velocities.append(float(parts[1]))
        snr.append(float(parts[2]))
    return np.asarray(periods, dtype=float), np.asarray(velocities, dtype=float), np.asarray(snr, dtype=float)


def valid_phase_points(curve_path: Path) -> int:
    _, velocity, _ = load_curve(curve_path)
    return int(np.count_nonzero(np.isfinite(velocity) & (velocity > 0)))


def common_pair_paths(final_root: Path) -> list[PairPaths]:
    p = paths_from_root(final_root)
    non_pairs = {path.stem for path in p["non_npz_dir"].glob("*.npz")}
    rem_pairs = {path.stem for path in p["rem_npz_dir"].glob("*.npz")}
    common = sorted(non_pairs & rem_pairs)
    rows: list[PairPaths] = []
    for pair in common:
        rows.append(
            PairPaths(
                pair=pair,
                non_npz=p["non_npz_dir"] / f"{pair}.npz",
                rem_npz=p["rem_npz_dir"] / f"{pair}.npz",
                non_g=p["non_curves_dir"] / f"GDisp.{pair}.txt",
                non_c=p["non_curves_dir"] / f"CDisp.{pair}.txt",
                rem_g=p["rem_curves_dir"] / f"GDisp.{pair}.txt",
                rem_c=p["rem_curves_dir"] / f"CDisp.{pair}.txt",
            )
        )
    return rows


def load_phase_payload(npz_path: Path) -> dict[str, np.ndarray | float]:
    with np.load(npz_path) as data:
        phase_image = np.asarray(data["phase_image"], dtype=float)
        periods = np.asarray(data["periods"], dtype=float)
        velocities = np.asarray(data["velocity_axis_km_s"], dtype=float)
        actual_start_v = float(data["actual_start_v"])
        distance_km = float(data["distance_km"])
        delta_v = float(data["delta_v"]) if "delta_v" in data else float(np.median(np.diff(velocities)))
    offset = int(round((actual_start_v - float(velocities[0])) / delta_v))
    cropped = phase_image[offset:, :]
    cropped_vel = velocities[offset:]
    return {
        "phase_image": cropped,
        "periods": periods,
        "velocities": cropped_vel,
        "distance_km": distance_km,
        "actual_start_v": actual_start_v,
    }


def pair_difference_score(paths: PairPaths, min_valid_phase_points: int) -> float | None:
    if not all(path.exists() for path in (paths.non_npz, paths.rem_npz, paths.non_g, paths.non_c, paths.rem_g, paths.rem_c)):
        return None
    if valid_phase_points(paths.non_c) < min_valid_phase_points:
        return None
    if valid_phase_points(paths.rem_c) < min_valid_phase_points:
        return None

    before = load_phase_payload(paths.non_npz)
    after = load_phase_payload(paths.rem_npz)
    img_before = np.asarray(before["phase_image"], dtype=float)
    img_after = np.asarray(after["phase_image"], dtype=float)
    if img_before.shape != img_after.shape:
        min_rows = min(img_before.shape[0], img_after.shape[0])
        min_cols = min(img_before.shape[1], img_after.shape[1])
        img_before = img_before[:min_rows, :min_cols]
        img_after = img_after[:min_rows, :min_cols]
    diff = np.mean(np.abs(img_after - img_before))
    energy = np.mean(np.abs(img_before)) + np.mean(np.abs(img_after))
    return float(diff + 0.15 * energy)


def select_interesting_pairs(
    all_pairs: list[PairPaths],
    figure_count: int,
    candidate_sample_size: int,
    seed: int,
    min_valid_phase_points: int,
) -> list[tuple[float, PairPaths]]:
    rng = random.Random(seed)
    if len(all_pairs) > candidate_sample_size:
        candidates = rng.sample(all_pairs, candidate_sample_size)
    else:
        candidates = list(all_pairs)

    scored: list[tuple[float, PairPaths]] = []
    for row in candidates:
        score = pair_difference_score(row, min_valid_phase_points=min_valid_phase_points)
        if score is not None and math.isfinite(score):
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:figure_count]


def add_curve(ax, curve_path: Path, color: str, label: str, linestyle: str = "-") -> None:
    periods, velocities, _ = load_curve(curve_path)
    mask = np.isfinite(velocities) & (velocities > 0)
    if not np.any(mask):
        return
    line = ax.plot(
        periods[mask],
        velocities[mask],
        linestyle=linestyle,
        color=color,
        linewidth=2.2,
        label=label,
    )[0]
    line.set_path_effects([pe.Stroke(linewidth=3.4, foreground="black", alpha=0.55), pe.Normal()])


def draw_one_pair(paths: PairPaths, score: float, output_path: Path) -> None:
    before = load_phase_payload(paths.non_npz)
    after = load_phase_payload(paths.rem_npz)
    image_before = np.asarray(before["phase_image"], dtype=float)
    image_after = np.asarray(after["phase_image"], dtype=float)

    if image_before.shape != image_after.shape:
        min_rows = min(image_before.shape[0], image_after.shape[0])
        min_cols = min(image_before.shape[1], image_after.shape[1])
        image_before = image_before[:min_rows, :min_cols]
        image_after = image_after[:min_rows, :min_cols]
        vel_before = np.asarray(before["velocities"], dtype=float)[:min_rows]
        vel_after = np.asarray(after["velocities"], dtype=float)[:min_rows]
        periods = np.asarray(before["periods"], dtype=float)[:min_cols]
    else:
        vel_before = np.asarray(before["velocities"], dtype=float)
        vel_after = np.asarray(after["velocities"], dtype=float)
        periods = np.asarray(before["periods"], dtype=float)

    vmax = max(np.nanpercentile(np.abs(image_before), 99), np.nanpercentile(np.abs(image_after), 99), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True, sharey=False, constrained_layout=True)
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})

    panels = [
        ("Before Spike Removal", image_before, vel_before, paths.non_g, paths.non_c),
        ("After Spike Removal", image_after, vel_after, paths.rem_g, paths.rem_c),
    ]
    for ax, (title, image, velocity_axis, g_curve, c_curve) in zip(axes, panels):
        mesh = ax.pcolormesh(
            periods,
            velocity_axis,
            image,
            shading="auto",
            cmap="cividis",
            vmin=-vmax,
            vmax=vmax,
        )
        add_curve(ax, g_curve, OKABE_ITO["sky_blue"], "Group", linestyle="--")
        add_curve(ax, c_curve, OKABE_ITO["vermillion"], "Phase", linestyle="-")
        ax.set_title(title)
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Velocity (km/s)")
        ax.set_xlim(float(periods[0]), float(periods[-1]))
        ax.set_ylim(float(velocity_axis[0]), float(velocity_axis[-1]))
        ax.grid(False)

    cbar = fig.colorbar(mesh, ax=axes, shrink=0.94, pad=0.02)
    cbar.set_label("Phase dispersion energy")
    axes[1].legend(loc="upper right", frameon=True)

    distance = float(before["distance_km"])
    fig.suptitle(
        f"{paths.pair}  |  distance={distance:.2f} km  |  difference score={score:.5f}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_index(rows: list[tuple[float, PairPaths]], output_dir: Path) -> Path:
    csv_path = output_dir / "selected_pairs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "pair", "difference_score", "filename"])
        for index, (score, paths) in enumerate(rows, start=1):
            writer.writerow([index, paths.pair, f"{score:.8f}", f"{index:02d}_{paths.pair}.png"])
    return csv_path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = common_pair_paths(args.final_root)
    selected = select_interesting_pairs(
        all_pairs=all_pairs,
        figure_count=args.figure_count,
        candidate_sample_size=args.candidate_sample_size,
        seed=args.seed,
        min_valid_phase_points=args.min_valid_phase_points,
    )
    if not selected:
        raise RuntimeError("No suitable before/after pair was found.")

    for index, (score, paths) in enumerate(selected, start=1):
        out_path = output_dir / f"{index:02d}_{paths.pair}.png"
        draw_one_pair(paths, score=score, output_path=out_path)

    write_index(selected, output_dir)
    print(f"Wrote {len(selected)} figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
