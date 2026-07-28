#!/usr/bin/env python3
"""Apply Wang et al. (2017) azimuthal traveltime correction and redraw phase-velocity maps.

Reproduces the Wang et al. (2017, JGR) Figure 8 diagnostic and Figure 9 corrected
phase-velocity maps:
  1. residual = observed (N*T-corrected) traveltime - D / c_ref, c_ref = constant
     through-origin fit velocity per period (NOT a tomographic model);
  2. bin residuals by 20 deg back-azimuth bins, take bin means;
  3. least-squares fit f(theta) = a + b*cos(2t) + c*sin(2t) + d*cos(4t) + e*sin(4t)
     per period (even harmonics only, 180 deg symmetry);
  4. subtract f(theta_path) from every traveltime, re-run the same straight-ray
     Barmin-style inversion, and replot maps with the paper Figure 9 color limits.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator, FuncFormatter

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    module_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LOCAL = _load_module("_local_phase_velocity_maps_for_fig9", "local_phase_velocity_maps.py")
REPLOT = _load_module("_replot_wang_figure6_for_fig9", "replot_wang_figure6_colormap.py")
AANT = LOCAL.AANT

DEFAULT_PERIODS = (3.0, 3.5, 4.0)
DEFAULT_BBOX = {
    "minlon": -122.34,
    "minlat": 46.08,
    "maxlon": -122.04,
    "maxlat": 46.32,
}
FIGURE6_VELOCITY_LIMITS = {
    3.0: (2.4, 3.3),
    3.5: (2.5, 3.4),
    4.0: (2.5, 3.5),
}
FIGURE9_VELOCITY_LIMITS = {
    3.0: (2.4, 3.3),
    3.5: (2.5, 3.4),
    4.0: (2.5, 3.5),
}
COMPARISON_ROW_LABEL_X = 0.055
COMPARISON_ROW_LABEL_FONT_PATH = Path(
    "/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_fig78_20260712/fonts/Times New Roman.ttf"
)


def comparison_row_label_text_kwargs() -> Dict[str, object]:
    """Style the Figure 6/9 row labels independently from axis tick labels."""
    return {
        "fontsize": 12,
        "ha": "center",
        "va": "center",
        "rotation": 90,
        "fontproperties": FontProperties(fname=COMPARISON_ROW_LABEL_FONT_PATH),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_periods(value: str) -> Tuple[float, ...]:
    periods = tuple(round(float(part.strip()), 1) for part in value.split(",") if part.strip())
    if not periods:
        raise argparse.ArgumentTypeError("Use at least one period, e.g. 3.0,3.5,4.0")
    return periods


def parse_bbox(value: str) -> Dict[str, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be minlon,minlat,maxlon,maxlat")
    minlon, minlat, maxlon, maxlat = parts
    if minlon >= maxlon or minlat >= maxlat:
        raise argparse.ArgumentTypeError("--bbox min values must be smaller than max values")
    return {"minlon": minlon, "minlat": minlat, "maxlon": maxlon, "maxlat": maxlat}


def normalize_lon(lon: float) -> float:
    lon = float(lon)
    return lon - 360.0 if lon > 180.0 else lon


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def pair_to_codes(pair_name: str) -> Tuple[str, str]:
    if "__" not in pair_name:
        raise ValueError(f"Unsupported pair name: {pair_name}")
    source_code, receiver_code = pair_name.split("__", 1)
    return source_code, receiver_code


def parse_curve_coords(curves_dir: Path, pair_name: str) -> Tuple[float, float, float, float]:
    path = curves_dir / f"CDisp.{pair_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing curve file for coordinates: {path}")
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
        second = handle.readline().split()
    if len(first) < 2 or len(second) < 2:
        raise ValueError(f"Curve file is missing coordinate headers: {path}")
    source_lon, source_lat = normalize_lon(float(first[0])), float(first[1])
    receiver_lon, receiver_lat = normalize_lon(float(second[0])), float(second[1])
    return source_lon, source_lat, receiver_lon, receiver_lat


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * 6371.0088 * math.asin(min(1.0, math.sqrt(a)))


def great_circle_azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth (clockwise from north) of the path from point 1 to point 2."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def row_inline_coords(row: Mapping[str, object]) -> Optional[Tuple[float, float, float, float]]:
    source_lon = finite_float(row.get("source_lon"))
    source_lat = finite_float(row.get("source_lat"))
    receiver_lon = finite_float(row.get("receiver_lon"))
    receiver_lat = finite_float(row.get("receiver_lat"))
    coords = (source_lon, source_lat, receiver_lon, receiver_lat)
    if not all(math.isfinite(value) for value in coords):
        return None
    return normalize_lon(source_lon), source_lat, normalize_lon(receiver_lon), receiver_lat


def load_corrected_rows(
    path: Path,
    *,
    curves_dir: Optional[Path],
    periods: Sequence[float],
) -> Dict[float, List[Dict[str, object]]]:
    wanted = {round(float(period), 1) for period in periods}
    coord_cache: Dict[str, Tuple[float, float, float, float]] = {}
    rows_by_period: Dict[float, List[Dict[str, object]]] = {round(float(period), 1): [] for period in periods}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            period = round(finite_float(row.get("period_s")), 1)
            if period not in wanted:
                continue
            pair_name = str(row.get("pair_name", "")).strip()
            if not pair_name:
                continue
            travel_time = finite_float(row.get("corrected_travel_time_s"))
            if not (travel_time > 0):
                continue
            inline_coords = row_inline_coords(row)
            if inline_coords is not None:
                coord_cache[pair_name] = inline_coords
            elif pair_name not in coord_cache:
                if curves_dir is None:
                    raise ValueError(
                        f"Row for {pair_name} has no inline coordinates and --curves-dir was not given"
                    )
                coord_cache[pair_name] = parse_curve_coords(curves_dir, pair_name)
            source_code, receiver_code = pair_to_codes(pair_name)
            source_lon, source_lat, receiver_lon, receiver_lat = coord_cache[pair_name]
            distance_km = great_circle_km(source_lat, source_lon, receiver_lat, receiver_lon)
            if not (distance_km > 0):
                continue
            azimuth_deg = great_circle_azimuth_deg(source_lat, source_lon, receiver_lat, receiver_lon)
            rows_by_period[period].append(
                {
                    "pair_name": pair_name,
                    "source_code": source_code,
                    "receiver_code": receiver_code,
                    "source_lon": source_lon,
                    "source_lat": source_lat,
                    "receiver_lon": receiver_lon,
                    "receiver_lat": receiver_lat,
                    "period_s": period,
                    "distance_km": distance_km,
                    "azimuth_deg": azimuth_deg,
                    "travel_time_s": travel_time,
                    "phase_velocity_km_s": distance_km / travel_time,
                    "raw_travel_time_s": finite_float(row.get("raw_travel_time_s")),
                    "branch_n": int(finite_float(row.get("branch_n"), 0.0)),
                    "snr": finite_float(row.get("snr")),
                    "group_velocity_km_s": finite_float(row.get("group_velocity_km_s")),
                }
            )
    for rows in rows_by_period.values():
        rows.sort(key=lambda item: str(item["pair_name"]))
    return rows_by_period


def fit_reference_velocity(distance_km: Sequence[float], travel_time_s: Sequence[float]) -> float:
    """Through-origin least-squares fit of t = D / c."""
    distance = np.asarray(distance_km, dtype=float)
    time = np.asarray(travel_time_s, dtype=float)
    valid = np.isfinite(distance) & np.isfinite(time) & (distance > 0) & (time > 0)
    if np.count_nonzero(valid) == 0:
        return float("nan")
    slope = float(np.dot(distance[valid], time[valid]) / np.dot(distance[valid], distance[valid]))
    return 1.0 / slope if math.isfinite(slope) and slope > 0 else float("nan")


def fourier_design(azimuth_deg: np.ndarray) -> np.ndarray:
    theta = np.radians(azimuth_deg)
    return np.column_stack(
        [
            np.ones_like(theta),
            np.cos(2.0 * theta),
            np.sin(2.0 * theta),
            np.cos(4.0 * theta),
            np.sin(4.0 * theta),
        ]
    )


def fit_azimuthal_correction(
    rows: Sequence[Mapping[str, object]],
    *,
    bin_width_deg: float,
) -> Dict[str, object]:
    """Wang et al. (2017) correction: residuals vs constant reference velocity,
    binned means per azimuth bin, Fourier fit f(t)=a+b cos2t+c sin2t+d cos4t+e sin4t.

    Azimuths are folded to [0, 180) before binning: with symmetric CCFs theta and
    theta+180 are the same path (pair ordering is arbitrary), so unfolded bins
    break the 180-degree symmetry the paper relies on. Bin means are fitted with
    count weighting so sparse bins cannot distort the correction."""
    azimuth = np.asarray([float(row["azimuth_deg"]) for row in rows], dtype=float)
    distance = np.asarray([float(row["distance_km"]) for row in rows], dtype=float)
    travel_time = np.asarray([float(row["travel_time_s"]) for row in rows], dtype=float)
    ref_velocity = fit_reference_velocity(distance, travel_time)
    residual = travel_time - distance / ref_velocity

    folded = azimuth % 180.0
    n_bins = int(round(180.0 / float(bin_width_deg)))
    bin_index = np.floor(folded / float(bin_width_deg)).astype(int) % n_bins
    fold_centers = (np.arange(n_bins) + 0.5) * float(bin_width_deg)
    fold_mean = np.full(n_bins, np.nan)
    fold_sem = np.full(n_bins, np.nan)
    fold_count = np.zeros(n_bins, dtype=int)
    for ibin in range(n_bins):
        values = residual[bin_index == ibin]
        fold_count[ibin] = values.size
        if values.size:
            fold_mean[ibin] = float(np.mean(values))
            fold_sem[ibin] = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0

    valid = fold_count > 0
    weight = np.sqrt(fold_count[valid].astype(float))
    design = fourier_design(fold_centers[valid])
    coeffs, *_ = np.linalg.lstsq(design * weight[:, None], fold_mean[valid] * weight, rcond=None)
    fitted_all = fourier_design(azimuth) @ coeffs

    # mirror folded bins back to 0-360 for Figure 8 style plotting
    centers = np.concatenate([fold_centers, fold_centers + 180.0])
    bin_mean = np.concatenate([fold_mean, fold_mean])
    bin_sem = np.concatenate([fold_sem, fold_sem])
    bin_count = np.concatenate([fold_count, fold_count])

    corrected_time = travel_time - fitted_all
    sd_before = float(np.std(residual, ddof=1))
    sd_after = float(np.std(residual - fitted_all, ddof=1))
    return {
        "reference_velocity_km_s": float(ref_velocity),
        "coefficients": {"a": float(coeffs[0]), "b_cos2": float(coeffs[1]), "c_sin2": float(coeffs[2]), "d_cos4": float(coeffs[3]), "e_sin4": float(coeffs[4])},
        "bin_width_deg": float(bin_width_deg),
        "bin_centers_deg": [float(v) for v in centers],
        "bin_count": [int(v) for v in bin_count],
        "bin_mean_s": [float(v) if math.isfinite(v) else None for v in bin_mean],
        "bin_sem_s": [float(v) if math.isfinite(v) else None for v in bin_sem],
        "residual_sd_before_s": sd_before,
        "residual_sd_after_s": sd_after,
        "variance_reduction_fraction": 1.0 - (sd_after**2) / (sd_before**2) if sd_before > 0 else float("nan"),
        "azimuth_deg": azimuth,
        "residual_s": residual,
        "correction_s": fitted_all,
        "corrected_travel_time_s": corrected_time,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_degree_grid(bbox: Mapping[str, float], spacing_deg: float) -> object:
    box = AANT.normalized_bbox(bbox)
    lon_edges = np.arange(box["minlon"], box["maxlon"] + 0.5 * spacing_deg, spacing_deg)
    lat_edges = np.arange(box["minlat"], box["maxlat"] + 0.5 * spacing_deg, spacing_deg)
    if lon_edges[-1] < box["maxlon"]:
        lon_edges = np.append(lon_edges, box["maxlon"])
    if lat_edges[-1] < box["maxlat"]:
        lat_edges = np.append(lat_edges, box["maxlat"])
    lon_edges[-1] = max(float(lon_edges[-1]), box["maxlon"])
    lat_edges[-1] = max(float(lat_edges[-1]), box["maxlat"])
    mid_lat = 0.5 * (box["minlat"] + box["maxlat"])
    cell_size_km = 0.5 * (
        float(spacing_deg) * AANT.lon_degree_width_km(mid_lat)
        + float(spacing_deg) * AANT.lat_degree_height_km()
    )
    return LOCAL.RegularGrid(
        bbox=box,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        nx=len(lon_edges) - 1,
        ny=len(lat_edges) - 1,
        cell_size_km=cell_size_km,
    )


def clipped_rows_for_bbox(
    rows: Sequence[Mapping[str, object]],
    *,
    bbox: Mapping[str, float],
    min_inside_km: float,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        observation = LOCAL.row_to_local_observation(
            row,
            bbox=bbox,
            mode="segment",
            min_inside_km=min_inside_km,
            min_inside_fraction=0.0,
        )
        if observation is not None:
            out.append(observation)
    return out


def ray_count_grid(rows: Sequence[Mapping[str, object]], grid: object, *, sample_step_km: float) -> np.ndarray:
    counts = np.zeros(grid.cell_count, dtype=float)
    for row in rows:
        source_lon = float(row["source_lon"])
        source_lat = float(row["source_lat"])
        receiver_lon = float(row["receiver_lon"])
        receiver_lat = float(row["receiver_lat"])
        inside_km = float(row.get("inside_km") or AANT.distance_km(source_lon, source_lat, receiver_lon, receiver_lat))
        sample_count = max(4, int(math.ceil(inside_km / float(sample_step_km))))
        touched: set[int] = set()
        prev_lon, prev_lat = source_lon, source_lat
        for sample_index in range(1, sample_count + 1):
            fraction = sample_index / sample_count
            cur_lon = source_lon + (receiver_lon - source_lon) * fraction
            cur_lat = source_lat + (receiver_lat - source_lat) * fraction
            mid_lon = 0.5 * (prev_lon + cur_lon)
            mid_lat = 0.5 * (prev_lat + cur_lat)
            ix = int(np.searchsorted(grid.lon_edges, mid_lon, side="right") - 1)
            iy = int(np.searchsorted(grid.lat_edges, mid_lat, side="right") - 1)
            if 0 <= ix < grid.nx and 0 <= iy < grid.ny:
                touched.add(iy * grid.nx + ix)
            prev_lon, prev_lat = cur_lon, cur_lat
        for cell in touched:
            counts[cell] += 1.0
    return counts


def invert_period(
    rows: Sequence[Mapping[str, object]],
    *,
    bbox: Mapping[str, float],
    grid: object,
    min_inside_km: float,
    damping: float,
    smoothing: float,
    sample_step_km: float,
    robust_iterations: int,
) -> Dict[str, object]:
    clipped = clipped_rows_for_bbox(rows, bbox=bbox, min_inside_km=min_inside_km)
    ray_counts = ray_count_grid(clipped, grid, sample_step_km=sample_step_km)
    model = LOCAL.solve_slowness_model(
        clipped,
        grid,
        damping=damping,
        smoothing=smoothing,
        sample_step_km=sample_step_km,
        robust_iterations=robust_iterations,
    )
    return {
        "clipped": clipped,
        "ray_count": ray_counts,
        "model": model,
        "velocity": np.asarray(model["velocity_km_s"], dtype=float).reshape(grid.ny, grid.nx),
    }


def plot_figure8_style(
    path: Path,
    *,
    periods: Sequence[float],
    fit_by_period: Mapping[float, Mapping[str, object]],
    dpi: int,
) -> None:
    """Wang Figure 8 style: traveltime residual vs back azimuth with binned means
    (stars, error bars = 5x SEM) and the best-fitting Fourier curve (red)."""
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, len(periods), figsize=(12.6, 4.4), sharex=True, sharey=True)
    if len(periods) == 1:
        axes = np.asarray([axes])
    dense_az = np.linspace(0.0, 360.0, 721)
    for col, period in enumerate(periods):
        period_key = round(float(period), 1)
        fit = fit_by_period[period_key]
        ax = axes[col]
        azimuth = np.asarray(fit["azimuth_deg"], dtype=float)
        residual = np.asarray(fit["residual_s"], dtype=float)
        ax.scatter(azimuth, residual, s=2, color="#9db8dd", alpha=0.12, linewidths=0, zorder=1)
        centers = np.asarray(fit["bin_centers_deg"], dtype=float)
        means = np.asarray([np.nan if v is None else float(v) for v in fit["bin_mean_s"]], dtype=float)
        sems = np.asarray([np.nan if v is None else float(v) for v in fit["bin_sem_s"]], dtype=float)
        ax.errorbar(
            centers,
            means,
            yerr=5.0 * sems,
            fmt="*",
            color="#1f5fc4",
            ecolor="#7fa6e3",
            elinewidth=1.0,
            capsize=2.5,
            markersize=7,
            zorder=3,
        )
        coeffs = fit["coefficients"]
        coeff_vec = np.asarray([coeffs["a"], coeffs["b_cos2"], coeffs["c_sin2"], coeffs["d_cos4"], coeffs["e_sin4"]])
        dense_fit = fourier_design(dense_az) @ coeff_vec
        ax.plot(dense_az, dense_fit, color="red", lw=1.6, zorder=4)
        ax.axhline(0.0, color="0.4", lw=0.7, zorder=2)
        ax.set_xlim(0.0, 360.0)
        ax.set_xticks(np.arange(0.0, 361.0, 50.0))
        ax.grid(True, which="major", color="#e2e2e2", linewidth=0.6)
        ax.set_title(f"({chr(ord('a') + col)}) {period:g} s", fontsize=13)
        ax.set_xlabel("Back-azimuth (deg)", fontsize=12)
        if col == 0:
            ax.set_ylabel("Travel Time Residuals (s)", fontsize=12)
        ymax = max(1.0, float(np.nanpercentile(np.abs(residual), 99.5)) * 1.15)
        ax.set_ylim(-ymax, ymax)
    fig.suptitle(
        "Azimuthal traveltime residuals (vs constant reference velocity) and Fourier fit",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_paper_maps(
    path: Path,
    *,
    periods: Sequence[float],
    grids: Mapping[float, Mapping[str, np.ndarray]],
    stations: Sequence[Mapping[str, float]],
    bbox: Mapping[str, float],
    coverage_contour: float,
    velocity_limits: Mapping[float, Tuple[float, float]],
    dpi: int,
    smooth_sigma_cells: float,
    upsample_factor: int,
    suptitle: Optional[str] = None,
) -> None:
    """Paper-style smooth square maps (same renderer as the Figure 6 replot)."""
    ensure_dir(path.parent)
    cmap = REPLOT.paper_colormap()
    fig, axes = plt.subplots(1, len(periods), figsize=(12.1, 4.75), sharex=True, sharey=True)
    if len(periods) == 1:
        axes = np.asarray([axes])
    station_lon, station_lat = REPLOT.station_points(stations, bbox)
    formatter = FuncFormatter(REPLOT.degree_formatter)
    for col, period in enumerate(periods):
        period_key = round(float(period), 1)
        grid = grids[period_key]
        velocity = np.asarray(grid["velocity"], dtype=float)
        ray_count = np.asarray(grid["ray_count"], dtype=float)
        lon_edges = np.asarray(grid["lon_edges"], dtype=float)
        lat_edges = np.asarray(grid["lat_edges"], dtype=float)
        vmin, vmax = velocity_limits.get(period_key, (2.4, 3.5))
        norm = Normalize(vmin=vmin, vmax=vmax)
        display_velocity = REPLOT.display_grid(
            velocity, sigma_cells=smooth_sigma_cells, upsample_factor=upsample_factor
        )
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
            display_ray_count = REPLOT.display_grid(
                ray_count, sigma_cells=max(0.5, smooth_sigma_cells), upsample_factor=upsample_factor
            )
            display_lon = REPLOT.display_axis(lon_edges, display_ray_count.shape[1])
            display_lat = REPLOT.display_axis(lat_edges, display_ray_count.shape[0])
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
        ax.xaxis.set_major_locator(FixedLocator(REPLOT.PAPER_X_TICKS))
        ax.yaxis.set_major_locator(FixedLocator(REPLOT.PAPER_Y_TICKS))
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
        cbar.set_ticks(REPLOT.colorbar_ticks(vmin, vmax))
        cbar.ax.tick_params(labelsize=10, direction="in", length=3.0)
        cbar.set_label("Phase Velocity (km/s)", fontsize=13, labelpad=4)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_before_after_comparison(
    path: Path,
    *,
    periods: Sequence[float],
    grids_before: Mapping[float, Mapping[str, np.ndarray]],
    grids_after: Mapping[float, Mapping[str, np.ndarray]],
    stations: Sequence[Mapping[str, float]],
    bbox: Mapping[str, float],
    coverage_contour: float,
    limits_before: Mapping[float, Tuple[float, float]],
    limits_after: Mapping[float, Tuple[float, float]],
    dpi: int,
    smooth_sigma_cells: float,
    upsample_factor: int,
) -> None:
    ensure_dir(path.parent)
    cmap = REPLOT.paper_colormap()
    fig, axes = plt.subplots(2, len(periods), figsize=(12.1, 9.6), sharex=True, sharey=True)
    if len(periods) == 1:
        axes = axes.reshape(2, 1)
    station_lon, station_lat = REPLOT.station_points(stations, bbox)
    formatter = FuncFormatter(REPLOT.degree_formatter)
    row_label_axes: List[Tuple[object, str]] = []
    for row_index, (grids, limits, row_label) in enumerate(
        (
            (grids_before, limits_before, "before azimuthal correction"),
            (grids_after, limits_after, "after azimuthal correction (Fig.9 style)"),
        )
    ):
        for col, period in enumerate(periods):
            period_key = round(float(period), 1)
            grid = grids[period_key]
            velocity = np.asarray(grid["velocity"], dtype=float)
            ray_count = np.asarray(grid["ray_count"], dtype=float)
            lon_edges = np.asarray(grid["lon_edges"], dtype=float)
            lat_edges = np.asarray(grid["lat_edges"], dtype=float)
            vmin, vmax = limits.get(period_key, (2.4, 3.5))
            norm = Normalize(vmin=vmin, vmax=vmax)
            display_velocity = REPLOT.display_grid(
                velocity, sigma_cells=smooth_sigma_cells, upsample_factor=upsample_factor
            )
            ax = axes[row_index, col]
            mesh = ax.imshow(
                display_velocity,
                extent=(float(lon_edges[0]), float(lon_edges[-1]), float(lat_edges[0]), float(lat_edges[-1])),
                origin="lower",
                cmap=cmap,
                norm=norm,
                interpolation="bicubic",
            )
            if np.nanmax(ray_count) >= coverage_contour:
                display_ray_count = REPLOT.display_grid(
                    ray_count, sigma_cells=max(0.5, smooth_sigma_cells), upsample_factor=upsample_factor
                )
                display_lon = REPLOT.display_axis(lon_edges, display_ray_count.shape[1])
                display_lat = REPLOT.display_axis(lat_edges, display_ray_count.shape[0])
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
            ax.xaxis.set_major_locator(FixedLocator(REPLOT.PAPER_X_TICKS))
            ax.yaxis.set_major_locator(FixedLocator(REPLOT.PAPER_Y_TICKS))
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(
                labelsize=11,
                top=True,
                right=True,
                labeltop=False,
                labelright=False,
                labelleft=(col == 0),
                labelbottom=(row_index == 1),
                direction="in",
                length=4.0,
                width=0.9,
            )
            if row_index == 0:
                ax.text(0.50, 1.03, f"{period:g} s", transform=ax.transAxes, fontsize=14, ha="center", va="bottom")
            if col == 0:
                row_label_axes.append((ax, row_label))
            cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.065, pad=0.10)
            cbar.set_ticks(REPLOT.colorbar_ticks(vmin, vmax))
            cbar.ax.tick_params(labelsize=9, direction="in", length=3.0)
            cbar.set_label("Phase Velocity (km/s)", fontsize=11, labelpad=3)
    for label_ax, row_label in row_label_axes:
        axes_position = label_ax.get_position()
        fig.text(
            COMPARISON_ROW_LABEL_X,
            0.5 * (axes_position.y0 + axes_position.y1),
            row_label,
            transform=fig.transFigure,
            **comparison_row_label_text_kwargs(),
        )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_stations_csv(path: Path, rows_by_period: Mapping[float, Sequence[Mapping[str, object]]]) -> List[Dict[str, object]]:
    stations: Dict[str, Dict[str, object]] = {}
    for rows in rows_by_period.values():
        for row in rows:
            stations.setdefault(
                str(row["source_code"]),
                {"station_code": row["source_code"], "longitude": float(row["source_lon"]), "latitude": float(row["source_lat"])},
            )
            stations.setdefault(
                str(row["receiver_code"]),
                {"station_code": row["receiver_code"], "longitude": float(row["receiver_lon"]), "latitude": float(row["receiver_lat"])},
            )
    station_rows = [stations[key] for key in sorted(stations)]
    write_csv(path, station_rows)
    return station_rows


def run(args: argparse.Namespace) -> Dict[str, object]:
    periods = tuple(round(float(period), 1) for period in args.periods)
    bbox = AANT.normalized_bbox(args.bbox)
    ensure_dir(args.output_dir)
    figures_dir = args.output_dir / "figures"
    data_dir = args.output_dir / "data"
    ensure_dir(figures_dir)
    ensure_dir(data_dir)

    rows_by_period = load_corrected_rows(args.measurements_csv, curves_dir=args.curves_dir, periods=periods)
    stations = write_stations_csv(data_dir / "stations.csv", rows_by_period)
    grid = build_degree_grid(bbox, args.grid_spacing_deg)

    fit_by_period: Dict[float, Dict[str, object]] = {}
    corrected_rows_by_period: Dict[float, List[Dict[str, object]]] = {}
    grids_before: Dict[float, Dict[str, np.ndarray]] = {}
    grids_after: Dict[float, Dict[str, np.ndarray]] = {}
    summary_periods: Dict[str, object] = {}
    correction_table_rows: List[Dict[str, object]] = []

    for period in periods:
        period_key = round(float(period), 1)
        rows = rows_by_period[period_key]
        fit = fit_azimuthal_correction(rows, bin_width_deg=args.az_bin_deg)
        fit_by_period[period_key] = fit

        corrected_rows: List[Dict[str, object]] = []
        for row, t_corr, f_theta, residual in zip(
            rows,
            fit["corrected_travel_time_s"],
            fit["correction_s"],
            fit["residual_s"],
        ):
            t_corr = float(t_corr)
            if not (math.isfinite(t_corr) and t_corr > 0.2):
                continue
            new_row = dict(row)
            new_row["residual_s"] = float(residual)
            new_row["azimuthal_correction_s"] = float(f_theta)
            new_row["azimuth_corrected_travel_time_s"] = t_corr
            new_row["phase_velocity_km_s"] = float(row["distance_km"]) / t_corr
            corrected_rows.append(new_row)
            correction_table_rows.append(
                {
                    "pair_name": row["pair_name"],
                    "period_s": period_key,
                    "distance_km": float(row["distance_km"]),
                    "azimuth_deg": float(row["azimuth_deg"]),
                    "travel_time_s": float(row["travel_time_s"]),
                    "residual_s": float(residual),
                    "azimuthal_correction_s": float(f_theta),
                    "azimuth_corrected_travel_time_s": t_corr,
                    "azimuth_corrected_phase_velocity_km_s": float(row["distance_km"]) / t_corr,
                }
            )
        corrected_rows_by_period[period_key] = corrected_rows

        common = dict(
            bbox=bbox,
            grid=grid,
            min_inside_km=args.min_inside_km,
            damping=args.damping,
            smoothing=args.smoothing,
            sample_step_km=args.sample_step_km,
            robust_iterations=args.robust_iterations,
        )
        result_before = invert_period(rows, **common)
        result_after = invert_period(corrected_rows, **common)
        period_label = f"{period_key:.1f}"
        for tag, result in (("before", result_before), ("after", result_after)):
            period_dir = data_dir / f"period_{period_label}s_{tag}"
            ensure_dir(period_dir)
            np.savez_compressed(
                period_dir / "phase_velocity_model_grid.npz",
                velocity_km_s=np.asarray(result["velocity"], dtype=float).ravel(),
                ray_count=result["ray_count"],
                lon_edges=grid.lon_edges,
                lat_edges=grid.lat_edges,
                residual_s=np.asarray(result["model"]["residual_s"], dtype=float),
                robust_weights=np.asarray(result["model"]["robust_weights"], dtype=float),
            )
            write_csv(period_dir / "paths.csv", result["clipped"])
        grids_before[period_key] = {
            "velocity": result_before["velocity"],
            "ray_count": result_before["ray_count"].reshape(grid.ny, grid.nx),
            "lon_edges": grid.lon_edges,
            "lat_edges": grid.lat_edges,
        }
        grids_after[period_key] = {
            "velocity": result_after["velocity"],
            "ray_count": result_after["ray_count"].reshape(grid.ny, grid.nx),
            "lon_edges": grid.lon_edges,
            "lat_edges": grid.lat_edges,
        }
        summary_periods[str(period_key)] = {
            "input_count": len(rows),
            "corrected_count": len(corrected_rows),
            "reference_velocity_km_s": fit["reference_velocity_km_s"],
            "fourier_coefficients_s": fit["coefficients"],
            "residual_sd_before_s": fit["residual_sd_before_s"],
            "residual_sd_after_s": fit["residual_sd_after_s"],
            "variance_reduction_fraction": fit["variance_reduction_fraction"],
            "inversion_residual_mad_before_s": float(result_before["model"]["residual_mad_s"]),
            "inversion_residual_mad_after_s": float(result_after["model"]["residual_mad_s"]),
        }

    write_csv(data_dir / "measurements_azimuth_corrected.csv", correction_table_rows)

    figure8_path = figures_dir / "wang_figure8_style_azimuthal_residual_fit.png"
    plot_figure8_style(figure8_path, periods=periods, fit_by_period=fit_by_period, dpi=args.dpi)

    figure9_path = figures_dir / "wang_figure9_style_phase_velocity_maps_azimuthal_corrected.png"
    plot_paper_maps(
        figure9_path,
        periods=periods,
        grids=grids_after,
        stations=stations,
        bbox=bbox,
        coverage_contour=args.coverage_contour,
        velocity_limits=FIGURE9_VELOCITY_LIMITS,
        dpi=max(args.dpi, 240),
        smooth_sigma_cells=args.smooth_sigma_cells,
        upsample_factor=args.upsample_factor,
    )

    comparison_path = figures_dir / "wang_figure6_vs_figure9_before_after_comparison.png"
    plot_before_after_comparison(
        comparison_path,
        periods=periods,
        grids_before=grids_before,
        grids_after=grids_after,
        stations=stations,
        bbox=bbox,
        coverage_contour=args.coverage_contour,
        limits_before=FIGURE6_VELOCITY_LIMITS,
        limits_after=FIGURE9_VELOCITY_LIMITS,
        dpi=max(args.dpi, 240),
        smooth_sigma_cells=args.smooth_sigma_cells,
        upsample_factor=args.upsample_factor,
    )

    correction_json = {
        str(period): {
            key: value
            for key, value in fit_by_period[period].items()
            if key not in ("azimuth_deg", "residual_s", "correction_s", "corrected_travel_time_s")
        }
        for period in periods
    }
    (data_dir / "azimuthal_correction.json").write_text(
        json.dumps(correction_json, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    summary: Dict[str, object] = {
        "method": "Wang et al. (2017) azimuthal traveltime correction + Barmin-style re-inversion (Figure 8/9 style)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "measurements_csv": str(args.measurements_csv),
        "curves_dir": str(args.curves_dir),
        "output_dir": str(args.output_dir),
        "correction": {
            "residual_definition": "travel_time - distance / c_ref; c_ref = through-origin fit per period (constant, NOT a tomographic model)",
            "bin_width_deg": float(args.az_bin_deg),
            "fit_function": "f(theta) = a + b*cos(2t) + c*sin(2t) + d*cos(4t) + e*sin(4t); least squares on 20-deg bin means; even harmonics only",
            "application": "corrected_travel_time = travel_time - f(azimuth); applied to ALL measurements before re-inversion",
            "reference": "Wang, Lin, Schmandt & Farrell (2017), JGR Solid Earth, doi:10.1002/2016JB013769, eq. (1)",
        },
        "periods": summary_periods,
        "parameters": {
            "periods": list(periods),
            "bbox": bbox,
            "grid_spacing_deg": float(args.grid_spacing_deg),
            "min_inside_km": float(args.min_inside_km),
            "sample_step_km": float(args.sample_step_km),
            "damping": float(args.damping),
            "smoothing": float(args.smoothing),
            "robust_iterations": int(args.robust_iterations),
            "coverage_contour": float(args.coverage_contour),
            "smooth_sigma_cells": float(args.smooth_sigma_cells),
            "upsample_factor": int(args.upsample_factor),
            "figure6_velocity_limits": {str(k): list(v) for k, v in FIGURE6_VELOCITY_LIMITS.items()},
            "figure9_velocity_limits": {str(k): list(v) for k, v in FIGURE9_VELOCITY_LIMITS.items()},
        },
        "figures": {
            "figure8_style_azimuthal_fit": str(figure8_path),
            "figure9_style_corrected_maps": str(figure9_path),
            "before_after_comparison": str(comparison_path),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "periods": summary_periods}, indent=2, ensure_ascii=True))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements-csv", type=Path, required=True)
    parser.add_argument(
        "--curves-dir",
        type=Path,
        default=None,
        help="CDisp curve directory for coordinates; optional if the CSV has inline coordinates.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--periods", type=parse_periods, default=DEFAULT_PERIODS)
    parser.add_argument("--bbox", type=parse_bbox, default=DEFAULT_BBOX)
    parser.add_argument("--az-bin-deg", type=float, default=20.0, help="Azimuth bin width for residual means (paper: 20 deg).")
    parser.add_argument("--grid-spacing-deg", type=float, default=0.01)
    parser.add_argument("--min-inside-km", type=float, default=0.5)
    parser.add_argument("--sample-step-km", type=float, default=0.25)
    parser.add_argument("--damping", type=float, default=8.0)
    parser.add_argument("--smoothing", type=float, default=30.0)
    parser.add_argument("--robust-iterations", type=int, default=4)
    parser.add_argument("--coverage-contour", type=float, default=20.0)
    parser.add_argument(
        "--smooth-sigma-cells",
        type=float,
        default=REPLOT.DEFAULT_SMOOTH_SIGMA_CELLS,
        help="Gaussian sigma (grid cells) for map rendering; large values dilute compact anomalies.",
    )
    parser.add_argument("--upsample-factor", type=int, default=REPLOT.DEFAULT_UPSAMPLE_FACTOR)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
