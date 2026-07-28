#!/usr/bin/env python3
"""Replot Wang-style Figure 6 from saved NPZ grids with paper-style colors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.ndimage import gaussian_filter, zoom

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


DEFAULT_LIMITS = {
    3.0: (2.4, 3.3),
    3.5: (2.5, 3.4),
    4.0: (2.5, 3.5),
}

PAPER_CMAP_STOPS = (
    (0.00, "#050505"),
    (0.10, "#5f0000"),
    (0.22, "#f00000"),
    (0.34, "#ff6a00"),
    (0.46, "#ffe500"),
    (0.56, "#fff9df"),
    (0.64, "#baf5cb"),
    (0.73, "#27dbc6"),
    (0.82, "#14afe7"),
    (0.91, "#6273d4"),
    (1.00, "#8e1bb4"),
)
PAPER_X_TICKS = (-122.3, -122.2, -122.1)
PAPER_Y_TICKS = (46.1, 46.2, 46.3)
DEFAULT_SMOOTH_SIGMA_CELLS = 1.15
DEFAULT_UPSAMPLE_FACTOR = 12


def parse_periods(value: str) -> Tuple[float, ...]:
    periods = tuple(round(float(part.strip()), 1) for part in value.split(",") if part.strip())
    if not periods:
        raise argparse.ArgumentTypeError("Use at least one period, e.g. 3.0,3.5,4.0")
    return periods


def period_label(period: float) -> str:
    return f"{float(period):.1f}"


def load_stations(path: Path) -> List[Dict[str, float]]:
    stations: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stations.append(
                {
                    "longitude": float(row["longitude"]),
                    "latitude": float(row["latitude"]),
                }
            )
    return stations


def station_points(stations: Sequence[Mapping[str, float]], bbox: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    lon: List[float] = []
    lat: List[float] = []
    for station in stations:
        station_lon = float(station["longitude"])
        station_lat = float(station["latitude"])
        if (
            float(bbox["minlon"]) <= station_lon <= float(bbox["maxlon"])
            and float(bbox["minlat"]) <= station_lat <= float(bbox["maxlat"])
        ):
            lon.append(station_lon)
            lat.append(station_lat)
    return np.asarray(lon), np.asarray(lat)


def load_period_grid(data_root: Path, period: float) -> Dict[str, np.ndarray]:
    label = period_label(period)
    npz_path = data_root / f"period_{label}s" / "phase_velocity_model_grid.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing model grid NPZ: {npz_path}")
    data = np.load(npz_path)
    velocity = np.asarray(data["velocity_km_s"], dtype=float)
    ray_count = np.asarray(data["ray_count"], dtype=float)
    lon_edges = np.asarray(data["lon_edges"], dtype=float)
    lat_edges = np.asarray(data["lat_edges"], dtype=float)
    ny = len(lat_edges) - 1
    nx = len(lon_edges) - 1
    return {
        "velocity": velocity.reshape(ny, nx),
        "ray_count": ray_count.reshape(ny, nx),
        "lon_edges": lon_edges,
        "lat_edges": lat_edges,
    }


def velocity_limits(period: float) -> Tuple[float, float]:
    return DEFAULT_LIMITS.get(round(float(period), 1), (2.4, 3.5))


def paper_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("wang_paper_phase_velocity", PAPER_CMAP_STOPS, N=256)


def degree_formatter(value: float, _position: object) -> str:
    return rf"{value:.1f}$^\circ$"


def colorbar_ticks(vmin: float, vmax: float) -> np.ndarray:
    start = np.ceil(vmin * 10.0) / 10.0
    stop = np.floor(vmax * 10.0) / 10.0
    return np.round(np.arange(start, stop + 0.05, 0.1), 1)


def nan_gaussian_filter(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if sigma_cells <= 0:
        return values.copy()
    finite = np.isfinite(values)
    if not np.any(finite):
        return values.copy()
    filled = np.where(finite, values, 0.0)
    weights = gaussian_filter(finite.astype(float), sigma=sigma_cells, mode="nearest")
    smoothed = gaussian_filter(filled, sigma=sigma_cells, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed / weights
    smoothed[weights <= 1.0e-8] = np.nan
    return smoothed


def display_grid(values: np.ndarray, *, sigma_cells: float, upsample_factor: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    smoothed = nan_gaussian_filter(values, sigma_cells)
    if upsample_factor > 1:
        smoothed = zoom(smoothed, zoom=int(upsample_factor), order=3, mode="nearest")
    finite_values = values[np.isfinite(values)]
    if finite_values.size and np.any(np.isfinite(smoothed)):
        smoothed = np.clip(smoothed, np.nanmin(finite_values), np.nanmax(finite_values))
    return smoothed


def display_axis(edges: np.ndarray, n_points: int) -> np.ndarray:
    return np.linspace(float(edges[0]), float(edges[-1]), int(n_points))


def save_combined_npz(
    path: Path,
    *,
    periods: Sequence[float],
    grids: Mapping[float, Mapping[str, np.ndarray]],
    coverage_contour: float,
    cmap_name: str,
    bbox: Mapping[str, float],
    smooth_sigma_cells: float,
    upsample_factor: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = grids[round(float(periods[0]), 1)]
    velocities = np.stack([np.asarray(grids[round(float(period), 1)]["velocity"], dtype=float) for period in periods])
    ray_counts = np.stack([np.asarray(grids[round(float(period), 1)]["ray_count"], dtype=float) for period in periods])
    display_velocities = np.stack(
        [
            display_grid(
                np.asarray(grids[round(float(period), 1)]["velocity"], dtype=float),
                sigma_cells=smooth_sigma_cells,
                upsample_factor=upsample_factor,
            )
            for period in periods
        ]
    )
    limits = np.asarray([velocity_limits(period) for period in periods], dtype=float)
    display_lon = display_axis(np.asarray(first["lon_edges"], dtype=float), display_velocities.shape[2])
    display_lat = display_axis(np.asarray(first["lat_edges"], dtype=float), display_velocities.shape[1])
    np.savez_compressed(
        path,
        periods_s=np.asarray(periods, dtype=float),
        velocity_km_s=velocities,
        display_velocity_km_s=display_velocities,
        ray_count=ray_counts,
        lon_edges=np.asarray(first["lon_edges"], dtype=float),
        lat_edges=np.asarray(first["lat_edges"], dtype=float),
        display_lon=display_lon,
        display_lat=display_lat,
        color_limits_km_s=limits,
        color_centers_km_s=np.mean(limits, axis=1),
        coverage_contour=float(coverage_contour),
        cmap_name=str(cmap_name),
        cmap_positions=np.asarray([position for position, _color in PAPER_CMAP_STOPS], dtype=float),
        cmap_colors=np.asarray([color for _position, color in PAPER_CMAP_STOPS], dtype="U16"),
        plot_bbox=np.asarray(
            [float(bbox["minlon"]), float(bbox["maxlon"]), float(bbox["minlat"]), float(bbox["maxlat"])],
            dtype=float,
        ),
        x_ticks=np.asarray(PAPER_X_TICKS, dtype=float),
        y_ticks=np.asarray(PAPER_Y_TICKS, dtype=float),
        square_panel_mode=np.asarray("matplotlib_axes_box_aspect_1", dtype="U32"),
        font_family=np.asarray("Times New Roman", dtype="U32"),
        display_method=np.asarray("nan_gaussian_filter_plus_cubic_upsampling", dtype="U64"),
        smooth_sigma_cells=float(smooth_sigma_cells),
        upsample_factor=int(upsample_factor),
    )


def plot_figure6(
    output_path: Path,
    *,
    periods: Sequence[float],
    grids: Mapping[float, Mapping[str, np.ndarray]],
    stations: Sequence[Mapping[str, float]],
    bbox: Mapping[str, float],
    coverage_contour: float,
    dpi: int,
    smooth_sigma_cells: float,
    upsample_factor: int,
) -> None:
    cmap = paper_colormap()
    fig, axes = plt.subplots(1, len(periods), figsize=(12.1, 4.75), sharex=True, sharey=True)
    if len(periods) == 1:
        axes = np.asarray([axes])
    station_lon, station_lat = station_points(stations, bbox)
    formatter = FuncFormatter(degree_formatter)

    for col, period in enumerate(periods):
        period_key = round(float(period), 1)
        grid = grids[period_key]
        velocity = np.asarray(grid["velocity"], dtype=float)
        ray_count = np.asarray(grid["ray_count"], dtype=float)
        lon_edges = np.asarray(grid["lon_edges"], dtype=float)
        lat_edges = np.asarray(grid["lat_edges"], dtype=float)
        vmin, vmax = velocity_limits(period_key)
        norm = Normalize(vmin=vmin, vmax=vmax)
        display_velocity = display_grid(velocity, sigma_cells=smooth_sigma_cells, upsample_factor=upsample_factor)

        ax = axes[col]
        mesh = ax.imshow(
            display_velocity,
            extent=(float(lon_edges[0]), float(lon_edges[-1]), float(lat_edges[0]), float(lat_edges[-1])),
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="bicubic",
        )
        if np.nanmax(ray_count) >= coverage_contour:
            display_ray_count = display_grid(
                ray_count,
                sigma_cells=max(0.5, smooth_sigma_cells),
                upsample_factor=upsample_factor,
            )
            display_lon = display_axis(lon_edges, display_ray_count.shape[1])
            display_lat = display_axis(lat_edges, display_ray_count.shape[0])
            ax.contour(
                display_lon,
                display_lat,
                display_ray_count,
                levels=[float(coverage_contour)],
                colors="0.55",
                linewidths=1.3,
            )
        if station_lon.size:
            ax.scatter(station_lon, station_lat, marker="^", s=10, color="black", linewidths=0, zorder=4)
        ax.set_xlim(float(bbox["minlon"]), float(bbox["maxlon"]))
        ax.set_ylim(float(bbox["minlat"]), float(bbox["maxlat"]))
        ax.set_aspect("auto")
        ax.set_box_aspect(1)
        ax.xaxis.set_major_locator(FixedLocator(PAPER_X_TICKS))
        ax.yaxis.set_major_locator(FixedLocator(PAPER_Y_TICKS))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(
            labelsize=12,
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            labelleft=(col == 0),
            direction="in",
            length=4.0,
            width=0.9,
        )
        ax.text(0.00, 1.035, f"({chr(ord('a') + col)})", transform=ax.transAxes, fontsize=15, ha="left", va="bottom")
        ax.text(0.50, 1.035, f"{period:g} s", transform=ax.transAxes, fontsize=15, ha="center", va="bottom")

        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.065, pad=0.12)
        cbar.set_ticks(colorbar_ticks(vmin, vmax))
        cbar.ax.tick_params(labelsize=10, direction="in", length=3.0)
        cbar.set_label("Phase Velocity (km/s)", fontsize=13, labelpad=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, object]:
    summary_path = args.input_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bbox = summary["parameters"]["bbox"]
    periods = tuple(round(float(period), 1) for period in args.periods)
    data_root = args.input_dir / "data"
    grids = {period: load_period_grid(data_root, period) for period in periods}
    stations = load_stations(data_root / "stations.csv")

    output_path = args.output_png or (
        args.input_dir / "figures" / "wang_figure6_style_phase_velocity_maps_paper_smooth_square.png"
    )
    combined_npz = args.output_npz or (
        args.input_dir / "data" / "wang_figure6_plot_data_paper_smooth_square.npz"
    )
    plot_figure6(
        output_path,
        periods=periods,
        grids=grids,
        stations=stations,
        bbox=bbox,
        coverage_contour=args.coverage_contour,
        dpi=args.dpi,
        smooth_sigma_cells=args.smooth_sigma_cells,
        upsample_factor=args.upsample_factor,
    )
    save_combined_npz(
        combined_npz,
        periods=periods,
        grids=grids,
        coverage_contour=args.coverage_contour,
        cmap_name="wang_paper_phase_velocity",
        bbox=bbox,
        smooth_sigma_cells=args.smooth_sigma_cells,
        upsample_factor=args.upsample_factor,
    )
    result = {
        "output_png": str(output_path),
        "output_npz": str(combined_npz),
        "periods": list(periods),
        "coverage_contour": float(args.coverage_contour),
        "colormap": "paper-like black-red-yellow-white-cyan-blue-purple",
        "x_ticks": list(PAPER_X_TICKS),
        "y_ticks": list(PAPER_Y_TICKS),
        "font_family": "Times New Roman",
        "square_panel_mode": "matplotlib axes box_aspect=1",
        "display_method": "nan Gaussian smoothing + cubic upsampling",
        "smooth_sigma_cells": float(args.smooth_sigma_cells),
        "upsample_factor": int(args.upsample_factor),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--periods", type=parse_periods, default=(3.0, 3.5, 4.0))
    parser.add_argument("--coverage-contour", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--smooth-sigma-cells", type=float, default=DEFAULT_SMOOTH_SIGMA_CELLS)
    parser.add_argument("--upsample-factor", type=int, default=DEFAULT_UPSAMPLE_FACTOR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
