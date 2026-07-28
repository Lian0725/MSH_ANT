#!/usr/bin/env python3
"""Re-run Figure 9 checkerboard recovery using physical kilometre squares."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "run_fig9_resolution_tests.py"
DEFAULT_MEASUREMENTS = Path(
    "/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_dat_20260724/egf_convention_check/measurements_fig56_egf.csv"
)
DEFAULT_STATIONS = Path(
    "/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_fig9_azimuth_corrected_egf_convention_20260724/data/stations.csv"
)
PERIODS = (3.0, 3.5, 4.0)
CHECKERBOARD_SIDES_KM = (2.0, 3.0, 4.0)
COVERAGE_MIN_RAYS = 20.0
RESULT_INTERPOLATION = "bilinear"
FIGURE_WIDTH_IN = 8.9
FIGURE_HEIGHT_IN = 8.6
HORIZONTAL_SPACING = 0.015
INPUT_CHECKERBOARD_CMAP = "RdBu"
RESULT_CMAP = "RdBu"
COLORBAR_LABEL = "Phase velocity (km/s; 3.0 s / 3.5 s / 4.0 s)"


def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_fig9_resolution_runner_for_km_checkers", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUN = load_runner()


def checkerboard_pattern_from_km(
    east_km: np.ndarray,
    north_km: np.ndarray,
    *,
    side_km: float,
    amplitude: float,
) -> np.ndarray:
    """Return alternating slowness fractions in physically square kilometre bins."""
    if side_km <= 0:
        raise ValueError("side_km must be positive")
    bins = np.floor(east_km / side_km).astype(int) + np.floor(north_km / side_km).astype(int)
    return np.where(bins % 2 == 0, amplitude, -amplitude).astype(float)


def checkerboard_side_labels(sides_km: Sequence[float]) -> list[str]:
    return [f"{side:.1f} km" for side in sides_km]


def absolute_velocity_tick_labels(
    perturbation_ticks_km_s: np.ndarray,
    reference_velocity_by_period: dict[float, float],
) -> list[str]:
    """Show the true speed of one colour at each displayed period."""
    return [
        " / ".join(f"{reference_velocity_by_period[period] + float(tick):.2f}" for period in PERIODS)
        for tick in perturbation_ticks_km_s
    ]


def slowness_fraction_to_velocity(frac: np.ndarray, *, reference_velocity_km_s: float) -> np.ndarray:
    denominator = 1.0 + np.asarray(frac, dtype=float)
    if np.any(denominator <= 0):
        raise ValueError("slowness fractions must remain above -1")
    return float(reference_velocity_km_s) / denominator


def phase_velocity_perturbation(frac: np.ndarray, *, reference_velocity_km_s: float) -> np.ndarray:
    """Return phase-velocity change in km/s, centred at zero for zero slowness."""
    return slowness_fraction_to_velocity(frac, reference_velocity_km_s=reference_velocity_km_s) - float(reference_velocity_km_s)


def physical_offsets_km(grid: object) -> tuple[np.ndarray, np.ndarray]:
    """Return east/north distances from the southwest grid corner at cell centres."""
    lon_centres, lat_centres = RUN.cell_centers(grid)
    lon0 = float(grid.lon_edges[0])
    lat0 = float(grid.lat_edges[0])
    east = np.empty((grid.ny, grid.nx), dtype=float)
    north = np.empty((grid.ny, grid.nx), dtype=float)
    for iy, lat in enumerate(lat_centres):
        north[iy, :] = RUN.AANT.distance_km(lon0, lat0, lon0, float(lat))
        for ix, lon in enumerate(lon_centres):
            east[iy, ix] = RUN.AANT.distance_km(lon0, float(lat), float(lon), float(lat))
    return east, north


def checkerboard_model_km(grid: object, *, side_km: float, amplitude: float) -> np.ndarray:
    east_km, north_km = physical_offsets_km(grid)
    return checkerboard_pattern_from_km(east_km, north_km, side_km=side_km, amplitude=amplitude)


def physical_checkerboard_raster(
    grid: object,
    *,
    side_km: float,
    amplitude: float,
    samples_per_km: int = 24,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Render physical square checkerboard input more finely than the inversion grid."""
    if samples_per_km <= 0:
        raise ValueError("samples_per_km must be positive")
    lon0, lon1 = float(grid.lon_edges[0]), float(grid.lon_edges[-1])
    lat0, lat1 = float(grid.lat_edges[0]), float(grid.lat_edges[-1])
    mean_lat = 0.5 * (lat0 + lat1)
    north_extent = 6371.0 * math.radians(lat1 - lat0)
    east_extent = 6371.0 * math.cos(math.radians(mean_lat)) * math.radians(lon1 - lon0)
    ny = max(2, int(math.ceil(north_extent * samples_per_km)))
    nx = max(2, int(math.ceil(east_extent * samples_per_km)))
    lons = lon0 + (np.arange(nx, dtype=float) + 0.5) * (lon1 - lon0) / nx
    lats = lat0 + (np.arange(ny, dtype=float) + 0.5) * (lat1 - lat0) / ny
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    north_km = 6371.0 * np.radians(lat_grid - lat0)
    east_km = 6371.0 * np.cos(np.radians(lat_grid)) * np.radians(lon_grid - lon0)
    return (
        checkerboard_pattern_from_km(east_km, north_km, side_km=side_km, amplitude=amplitude),
        (lon0, lon1, lat0, lat1),
    )


def load_stations(path: Path, bbox: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Load every station in the supplied nodal inventory.

    Four stations lie just east of the fixed inversion bbox.  Retaining them
    here preserves the complete 896-station inventory; Matplotlib clips their
    out-of-bounds markers at the map frame without changing the inversion grid.
    """
    lons: list[float] = []
    lats: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            lon = float(row["longitude"])
            lat = float(row["latitude"])
            lons.append(lon)
            lats.append(lat)
    if not lons:
        raise RuntimeError(f"No station coordinates inside the Figure 9 bbox: {path}")
    return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)


def panel_box_aspect(grid: object) -> float:
    """Return the north/east physical aspect ratio for a map grid."""
    lon_span = float(grid.lon_edges[-1] - grid.lon_edges[0])
    lat_span = float(grid.lat_edges[-1] - grid.lat_edges[0])
    centre_lat = 0.5 * (float(grid.lat_edges[0]) + float(grid.lat_edges[-1]))
    return lat_span / (lon_span * math.cos(math.radians(centre_lat)))


def draw_panel(
    ax: plt.Axes,
    *,
    phase_velocity_km_s: np.ndarray,
    grid: object,
    ray_count: np.ndarray,
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    interpolation: str,
    cmap: object = "RdBu_r",
) -> object:
    image = ax.imshow(
        np.ma.masked_invalid(phase_velocity_km_s),
        extent=(float(grid.lon_edges[0]), float(grid.lon_edges[-1]), float(grid.lat_edges[0]), float(grid.lat_edges[-1])),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        aspect="auto",
    )
    lon_centres, lat_centres = RUN.cell_centers(grid)
    if float(np.nanmax(ray_count)) >= COVERAGE_MIN_RAYS:
        ax.contour(
            lon_centres,
            lat_centres,
            ray_count,
            levels=[COVERAGE_MIN_RAYS],
            colors="#1a1a1a",
            linewidths=0.8,
            linestyles="--",
            zorder=4,
        )
    ax.scatter(station_lons, station_lats, marker="^", s=3.0, c="#111111", alpha=0.72, linewidths=0, zorder=5)
    # Four stations lie just beyond the inversion longitude edge.  Keep their
    # full inventory in the input, but clip the map frame to the inversion grid
    # so no outside-station extent becomes a blank strip inside the black box.
    ax.set_xlim(float(grid.lon_edges[0]), float(grid.lon_edges[-1]))
    ax.set_ylim(float(grid.lat_edges[0]), float(grid.lat_edges[-1]))
    ax.set_title(title, fontsize=11, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    # Lock the axes box to the physical map ratio.  Without this, imshow keeps
    # the map ratio inside a wider subplot cell and leaves an empty right strip.
    ax.set_box_aspect(panel_box_aspect(grid))
    return image


def plot_period_comparison(
    output_path: Path,
    *,
    grid: object,
    sides_km: Sequence[float],
    input_phase_velocity_perturbation_km_s: np.ndarray,
    recovered_phase_velocity_perturbation_by_period: dict[float, np.ndarray],
    reference_velocity_by_period: dict[float, float],
    ray_count_by_period: dict[float, np.ndarray],
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    dpi: int,
) -> None:
    all_perturbations = [input_phase_velocity_perturbation_km_s, *[recovered_phase_velocity_perturbation_by_period[period] for period in PERIODS]]
    vlim = max(float(np.nanmax(np.abs(values))) for values in all_perturbations)
    vlim *= 1.03
    vmin, vmax = -vlim, vlim
    common_ray_count = np.minimum.reduce([ray_count_by_period[period] for period in PERIODS])
    fig, axes = plt.subplots(3, 4, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=dpi, sharex=True, sharey=True)
    titles = ("Input", "3.0 s", "3.5 s", "4.0 s")
    image = None
    for row, side_label in enumerate(checkerboard_side_labels(sides_km)):
        panels = [
            (input_phase_velocity_perturbation_km_s[row], common_ray_count),
            (recovered_phase_velocity_perturbation_by_period[3.0][row], ray_count_by_period[3.0]),
            (recovered_phase_velocity_perturbation_by_period[3.5][row], ray_count_by_period[3.5]),
            (recovered_phase_velocity_perturbation_by_period[4.0][row], ray_count_by_period[4.0]),
        ]
        for col, (velocity, ray_count) in enumerate(panels):
            image = draw_panel(
                axes[row, col],
                phase_velocity_km_s=velocity,
                grid=grid,
                ray_count=ray_count,
                station_lons=station_lons,
                station_lats=station_lats,
                vmin=vmin,
                vmax=vmax,
                title=titles[col] if row == 0 else "",
                interpolation="nearest" if col == 0 else RESULT_INTERPOLATION,
                cmap=INPUT_CHECKERBOARD_CMAP if col == 0 else RESULT_CMAP,
            )
        axes[row, 0].set_ylabel(side_label, rotation=0, ha="right", va="center", labelpad=31, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.095, top=0.94, wspace=HORIZONTAL_SPACING, hspace=0.07)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.033, pad=0.045)
    colorbar_ticks = np.linspace(vmin, vmax, 5)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(absolute_velocity_tick_labels(colorbar_ticks, reference_velocity_by_period))
    colorbar.set_label(COLORBAR_LABEL, fontsize=11)
    colorbar.ax.tick_params(labelsize=7)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    if tuple(args.checkerboard_sides_km) != CHECKERBOARD_SIDES_KM:
        raise ValueError("This comparison is defined for 2.0, 3.0, and 4.0 km checkerboard sides")
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite an existing report directory: {args.output_dir}")
    bbox = RUN.AANT.normalized_bbox(RUN.FIG9.DEFAULT_BBOX)
    grid = RUN.FIG9.build_degree_grid(bbox, args.grid_spacing_deg)
    rows_by_period = RUN.FIG9.load_corrected_rows(args.measurements_csv, curves_dir=None, periods=PERIODS)
    station_lons, station_lats = load_stations(args.stations_csv, bbox)
    if station_lons.size != 896:
        raise RuntimeError(f"Expected the Figure 9 nodal inventory of 896 stations, found {station_lons.size}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    data_dir = args.output_dir / "data"
    figures_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    truths: list[np.ndarray] = [
        checkerboard_model_km(grid, side_km=side_km, amplitude=args.checker_amplitude)
        for side_km in args.checkerboard_sides_km
    ]
    recovered_frac_by_period: dict[float, np.ndarray] = {}
    recovered_velocity_by_period: dict[float, np.ndarray] = {}
    recovered_velocity_perturbation_by_period: dict[float, np.ndarray] = {}
    ray_count_by_period: dict[float, np.ndarray] = {}
    ref_velocity_by_period: dict[float, float] = {}
    noise_sd_by_period: dict[float, float] = {}
    for period in PERIODS:
        geometry = RUN.assemble_period_geometry(
            rows_by_period[period],
            bbox=bbox,
            grid=grid,
            min_inside_km=args.min_inside_km,
            sample_step_km=args.sample_step_km,
        )
        base_real = RUN.invert_linear(
            geometry["matrix"],
            geometry["rhs_data"],
            grid,
            damping=args.damping,
            smoothing=args.smoothing,
            robust_iterations=args.robust_iterations_real,
        )
        noise_sd = float(args.noise_sd) if args.noise_sd is not None else float(base_real["residual_mad_s"])
        recovered: list[np.ndarray] = []
        for index, truth in enumerate(truths):
            rng = np.random.default_rng(args.seed + int(round(period * 100)) + index)
            recovered.append(
                RUN.synthetic_inversion(
                    geometry,
                    grid,
                    truth,
                    damping=args.damping,
                    smoothing=args.smoothing,
                    km_normalized=False,
                    noise_sd=noise_sd,
                    rng=rng,
                )
            )
        ref_velocity = float(geometry["ref_velocity"])
        recovered_frac = np.stack(recovered)
        recovered_frac_by_period[period] = recovered_frac
        recovered_velocity_by_period[period] = slowness_fraction_to_velocity(
            recovered_frac,
            reference_velocity_km_s=ref_velocity,
        )
        recovered_velocity_perturbation_by_period[period] = phase_velocity_perturbation(
            recovered_frac,
            reference_velocity_km_s=ref_velocity,
        )
        ray_count_by_period[period] = np.asarray(geometry["ray_count"], dtype=float).reshape(grid.ny, grid.nx)
        ref_velocity_by_period[period] = ref_velocity
        noise_sd_by_period[period] = noise_sd

    input_reference_velocity = ref_velocity_by_period[3.0]
    input_velocity = slowness_fraction_to_velocity(
        np.stack(truths),
        reference_velocity_km_s=input_reference_velocity,
    )
    input_rasters = [
        physical_checkerboard_raster(grid, side_km=side_km, amplitude=args.checker_amplitude)[0]
        for side_km in args.checkerboard_sides_km
    ]
    input_display_phase_velocity_perturbation = np.stack([
        phase_velocity_perturbation(raster, reference_velocity_km_s=input_reference_velocity)
        for raster in input_rasters
    ])
    figure_path = figures_dir / "checkerboard_recovery_by_period_km_2_3_4.png"
    plot_period_comparison(
        figure_path,
        grid=grid,
        sides_km=args.checkerboard_sides_km,
        input_phase_velocity_perturbation_km_s=input_display_phase_velocity_perturbation,
        recovered_phase_velocity_perturbation_by_period=recovered_velocity_perturbation_by_period,
        reference_velocity_by_period=ref_velocity_by_period,
        ray_count_by_period=ray_count_by_period,
        station_lons=station_lons,
        station_lats=station_lats,
        dpi=args.dpi,
    )
    np.savez_compressed(
        data_dir / "checkerboard_km_recovery_data.npz",
        lon_edges=grid.lon_edges,
        lat_edges=grid.lat_edges,
        checkerboard_sides_km=np.asarray(args.checkerboard_sides_km, dtype=float),
        input_slowness_fraction=np.stack(truths),
        input_phase_velocity_km_s=input_velocity,
        input_display_phase_velocity_perturbation_km_s=input_display_phase_velocity_perturbation,
        recovery_slowness_fraction_3s=recovered_frac_by_period[3.0],
        recovery_slowness_fraction_3p5s=recovered_frac_by_period[3.5],
        recovery_slowness_fraction_4s=recovered_frac_by_period[4.0],
        recovery_phase_velocity_km_s_3s=recovered_velocity_by_period[3.0],
        recovery_phase_velocity_km_s_3p5s=recovered_velocity_by_period[3.5],
        recovery_phase_velocity_km_s_4s=recovered_velocity_by_period[4.0],
        recovery_phase_velocity_perturbation_km_s_3s=recovered_velocity_perturbation_by_period[3.0],
        recovery_phase_velocity_perturbation_km_s_3p5s=recovered_velocity_perturbation_by_period[3.5],
        recovery_phase_velocity_perturbation_km_s_4s=recovered_velocity_perturbation_by_period[4.0],
        ray_count_3s=ray_count_by_period[3.0],
        ray_count_3p5s=ray_count_by_period[3.5],
        ray_count_4s=ray_count_by_period[4.0],
        station_lon=station_lons,
        station_lat=station_lats,
    )
    summary = {
        "purpose": "Figure 9 physical-kilometre checkerboard recovery comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "measurements_csv": str(args.measurements_csv),
        "stations_csv": str(args.stations_csv),
        "station_count": int(station_lons.size),
        "periods_s": list(PERIODS),
        "checkerboard_side_km": list(args.checkerboard_sides_km),
        "checker_amplitude_fractional_slowness": float(args.checker_amplitude),
        "inversion_parameters": {
            "damping": float(args.damping),
            "smoothing": float(args.smoothing),
            "real_data_robust_iterations": int(args.robust_iterations_real),
            "synthetic_robust_iterations": 1,
        },
        "reference_velocity_km_s": {f"{period:g}": ref_velocity_by_period[period] for period in PERIODS},
        "noise_sd_s": {f"{period:g}": noise_sd_by_period[period] for period in PERIODS},
        "coverage_boundary": "dashed contour at ray_count = 20 in every panel; Input uses the all-period intersection",
        "display": "high-resolution physical-square input; bilinear recovered grids; zero-centred colours with true phase-velocity tick values (3.0 s / 3.5 s / 4.0 s) in km/s",
        "figure": str(figure_path),
        "data": str(data_dir / "checkerboard_km_recovery_data.npz"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements-csv", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkerboard-sides-km", type=float, nargs="+", default=CHECKERBOARD_SIDES_KM)
    parser.add_argument("--checker-amplitude", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=8.0)
    parser.add_argument("--smoothing", type=float, default=22.0)
    parser.add_argument("--grid-spacing-deg", type=float, default=0.01)
    parser.add_argument("--min-inside-km", type=float, default=0.5)
    parser.add_argument("--sample-step-km", type=float, default=0.25)
    parser.add_argument("--robust-iterations-real", type=int, default=4)
    parser.add_argument("--noise-sd", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the files in an existing km-checkerboard report directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(build_arg_parser().parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
