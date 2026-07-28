#!/usr/bin/env python3
"""Render Wang-style Figure 5/6 maps from DisperPicker screened paths."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_local_maps_module():
    module_path = SCRIPT_DIR / "local_phase_velocity_maps.py"
    spec = importlib.util.spec_from_file_location("_local_phase_velocity_maps_for_wang_figs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_replot_module():
    module_path = SCRIPT_DIR / "replot_wang_figure6_colormap.py"
    spec = importlib.util.spec_from_file_location("_replot_wang_figure6_colormap", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LOCAL = _load_local_maps_module()
AANT = LOCAL.AANT
REPLOT = _load_replot_module()

DEFAULT_PERIODS = (3.0, 3.5, 4.0)
DEFAULT_BBOX = {
    "minlon": -122.34,
    "minlat": 46.08,
    "maxlon": -122.04,
    "maxlat": 46.32,
}
DEFAULT_VELOCITY_LIMITS = {
    3.0: (2.4, 3.3),
    3.5: (2.5, 3.4),
    4.0: (2.5, 3.5),
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
            velocity = finite_float(
                row.get("corrected_phase_velocity_km_s"),
                finite_float(row.get("phase_velocity_km_s")),
            )
            if not (velocity > 0):
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
            rows_by_period[period].append(
                {
                    "pair_name": pair_name,
                    "source_code": source_code,
                    "receiver_code": receiver_code,
                    "source_lon": source_lon,
                    "source_lat": source_lat,
                    "receiver_lon": receiver_lon,
                    "receiver_lat": receiver_lat,
                    "period": period,
                    "period_s": period,
                    "phase_velocity_km_s": velocity,
                    "velocity_km_s": velocity,
                    "raw_phase_velocity_km_s": finite_float(row.get("phase_velocity_km_s")),
                    "raw_travel_time_s": finite_float(row.get("raw_travel_time_s")),
                    "corrected_travel_time_s": finite_float(row.get("corrected_travel_time_s")),
                    "branch_n": int(finite_float(row.get("branch_n"), 0.0)),
                    "snr": finite_float(row.get("snr")),
                    "confidence": finite_float(row.get("confidence")),
                    "group_velocity_km_s": finite_float(row.get("group_velocity_km_s")),
                    "distance_over_lambda": finite_float(row.get("distance_over_lambda")),
                }
            )
    for rows in rows_by_period.values():
        rows.sort(key=lambda item: (str(item["source_code"]), str(item["receiver_code"])))
    return rows_by_period


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


def write_stations_csv(path: Path, rows_by_period: Mapping[float, Sequence[Mapping[str, object]]]) -> List[Dict[str, object]]:
    stations: Dict[str, Dict[str, object]] = {}
    for rows in rows_by_period.values():
        for row in rows:
            stations.setdefault(
                str(row["source_code"]),
                {
                    "station_code": row["source_code"],
                    "longitude": float(row["source_lon"]),
                    "latitude": float(row["source_lat"]),
                },
            )
            stations.setdefault(
                str(row["receiver_code"]),
                {
                    "station_code": row["receiver_code"],
                    "longitude": float(row["receiver_lon"]),
                    "latitude": float(row["receiver_lat"]),
                },
            )
    station_rows = [stations[key] for key in sorted(stations)]
    write_csv(path, station_rows)
    return station_rows


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


def ray_count_grid(
    rows: Sequence[Mapping[str, object]],
    grid: object,
    *,
    sample_step_km: float,
) -> np.ndarray:
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


def write_grid_csv(
    path: Path,
    grid: object,
    *,
    ray_count: np.ndarray,
    velocity: Optional[np.ndarray] = None,
) -> None:
    rows: List[Dict[str, object]] = []
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            cell = iy * grid.nx + ix
            row = {
                "cell_index": cell,
                "lon_min": float(grid.lon_edges[ix]),
                "lon_max": float(grid.lon_edges[ix + 1]),
                "lat_min": float(grid.lat_edges[iy]),
                "lat_max": float(grid.lat_edges[iy + 1]),
                "ray_count": float(ray_count[cell]),
            }
            if velocity is not None:
                row["phase_velocity_km_s"] = float(velocity[cell])
            rows.append(row)
    write_csv(path, rows)


def period_velocity_limits(period: float) -> Tuple[float, float]:
    return DEFAULT_VELOCITY_LIMITS.get(round(float(period), 1), (2.4, 3.5))


def set_geo_axes(ax, bbox: Mapping[str, float]) -> None:
    box = AANT.normalized_bbox(bbox)
    ax.set_xlim(box["minlon"], box["maxlon"])
    ax.set_ylim(box["minlat"], box["maxlat"])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8, top=True, right=True, direction="in")


def station_points(stations: Sequence[Mapping[str, object]], bbox: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    box = AANT.normalized_bbox(bbox)
    lon: List[float] = []
    lat: List[float] = []
    for station in stations:
        station_lon = float(station["longitude"])
        station_lat = float(station["latitude"])
        if box["minlon"] <= station_lon <= box["maxlon"] and box["minlat"] <= station_lat <= box["maxlat"]:
            lon.append(station_lon)
            lat.append(station_lat)
    return np.asarray(lon), np.asarray(lat)


def plot_figure5(
    path: Path,
    *,
    periods: Sequence[float],
    rows_by_period: Mapping[float, Sequence[Mapping[str, object]]],
    coverage_by_period: Mapping[float, np.ndarray],
    grid: object,
    stations: Sequence[Mapping[str, object]],
    bbox: Mapping[str, float],
    coverage_vmax: float,
    dpi: int,
) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(2, len(periods), figsize=(11.2, 6.8), sharex=True, sharey=True)
    if len(periods) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])
    velocity_norm = Normalize(vmin=2.5, vmax=3.4)
    coverage_norm = Normalize(vmin=0.0, vmax=float(coverage_vmax))
    velocity_mappable = None
    coverage_mappable = None
    for col, period in enumerate(periods):
        rows = rows_by_period[round(float(period), 1)]
        segments = [
            [
                (float(row["source_lon"]), float(row["source_lat"])),
                (float(row["receiver_lon"]), float(row["receiver_lat"])),
            ]
            for row in rows
        ]
        velocities = np.asarray([float(row["phase_velocity_km_s"]) for row in rows], dtype=float)
        ax_top = axes[0, col]
        if segments:
            line_collection = LineCollection(
                segments,
                cmap="jet_r",
                norm=velocity_norm,
                linewidths=0.38,
                alpha=0.86,
            )
            line_collection.set_array(velocities)
            ax_top.add_collection(line_collection)
            velocity_mappable = line_collection
        set_geo_axes(ax_top, bbox)
        ax_top.set_title(f"({chr(ord('a') + col)}) {period:g} s", fontsize=10)

        ax_bottom = axes[1, col]
        values = coverage_by_period[round(float(period), 1)].reshape(grid.ny, grid.nx)
        masked = np.ma.masked_where(values <= 0, values)
        mesh = ax_bottom.pcolormesh(
            grid.lon_edges,
            grid.lat_edges,
            masked,
            cmap="YlOrRd",
            norm=coverage_norm,
            shading="flat",
        )
        coverage_mappable = mesh
        if segments:
            step = max(1, len(segments) // 2500)
            coverage_lines = LineCollection(
                segments[::step],
                colors="#214f9c",
                linewidths=0.12,
                alpha=0.025,
            )
            ax_bottom.add_collection(coverage_lines)
        set_geo_axes(ax_bottom, bbox)
        ax_bottom.set_title(f"({chr(ord('d') + col)})", fontsize=10)
        ax_bottom.set_facecolor("0.58")

    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude")
    for ax in axes[1, :]:
        ax.set_xlabel("Longitude")
    if velocity_mappable is not None:
        cbar = fig.colorbar(velocity_mappable, ax=axes[0, :].ravel().tolist(), fraction=0.025, pad=0.018)
        cbar.set_label("Phase Velocity (km/s)")
    if coverage_mappable is not None:
        cbar = fig.colorbar(coverage_mappable, ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.018)
        cbar.set_label("Number of Ray Path")
    fig.suptitle("Wang Figure 5 style: phase-velocity measurements and ray coverage density", fontsize=12)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def solve_period_model(
    rows: Sequence[Mapping[str, object]],
    *,
    grid: object,
    damping: float,
    smoothing: float,
    sample_step_km: float,
    robust_iterations: int,
) -> Dict[str, object]:
    return LOCAL.solve_slowness_model(
        rows,
        grid,
        damping=damping,
        smoothing=smoothing,
        sample_step_km=sample_step_km,
        robust_iterations=robust_iterations,
    )


def plot_figure6(
    path: Path,
    *,
    periods: Sequence[float],
    model_by_period: Mapping[float, Mapping[str, object]],
    coverage_by_period: Mapping[float, np.ndarray],
    grid: object,
    stations: Sequence[Mapping[str, object]],
    bbox: Mapping[str, float],
    coverage_contour: float,
    dpi: int,
) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, len(periods), figsize=(12.0, 4.6), sharex=True, sharey=True)
    if len(periods) == 1:
        axes = np.asarray([axes])
    station_lon, station_lat = station_points(stations, bbox)
    for col, period in enumerate(periods):
        period_key = round(float(period), 1)
        ax = axes[col]
        velocity = np.asarray(model_by_period[period_key]["velocity_km_s"], dtype=float).reshape(grid.ny, grid.nx)
        vmin, vmax = period_velocity_limits(period_key)
        mesh = ax.pcolormesh(
            grid.lon_edges,
            grid.lat_edges,
            velocity,
            cmap="jet_r",
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        coverage = np.asarray(coverage_by_period[period_key], dtype=float).reshape(grid.ny, grid.nx)
        lon_centers = 0.5 * (grid.lon_edges[:-1] + grid.lon_edges[1:])
        lat_centers = 0.5 * (grid.lat_edges[:-1] + grid.lat_edges[1:])
        if np.nanmax(coverage) >= coverage_contour:
            ax.contour(
                lon_centers,
                lat_centers,
                coverage,
                levels=[float(coverage_contour)],
                colors="0.55",
                linewidths=1.3,
            )
        if station_lon.size:
            ax.scatter(station_lon, station_lat, marker="^", s=10, color="black", linewidths=0, zorder=4)
        set_geo_axes(ax, bbox)
        ax.set_title(f"({chr(ord('a') + col)}) {period:g} s", fontsize=10)
        ax.set_xlabel("Longitude")
        if col == 0:
            ax.set_ylabel("Latitude")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Phase Velocity (km/s)")
    fig.suptitle("Wang Figure 6 style: Rayleigh-wave phase velocity maps", fontsize=12)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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

    clipped_by_period: Dict[float, List[Dict[str, object]]] = {}
    coverage_by_period: Dict[float, np.ndarray] = {}
    model_by_period: Dict[float, Mapping[str, object]] = {}
    summary_periods: Dict[str, object] = {}

    for period in periods:
        period_key = round(float(period), 1)
        period_label = AANT.format_period(period_key)
        period_dir = data_dir / f"period_{period_label}s"
        ensure_dir(period_dir)
        clipped = clipped_rows_for_bbox(
            rows_by_period[period_key],
            bbox=bbox,
            min_inside_km=args.min_inside_km,
        )
        clipped_by_period[period_key] = clipped
        write_csv(period_dir / "paths.csv", clipped)

        ray_counts = ray_count_grid(clipped, grid, sample_step_km=args.sample_step_km)
        coverage_by_period[period_key] = ray_counts
        write_grid_csv(period_dir / "ray_coverage_count_grid.csv", grid, ray_count=ray_counts)

        model = solve_period_model(
            clipped,
            grid=grid,
            damping=args.damping,
            smoothing=args.smoothing,
            sample_step_km=args.sample_step_km,
            robust_iterations=args.robust_iterations,
        )
        model_by_period[period_key] = model
        velocity = np.asarray(model["velocity_km_s"], dtype=float)
        write_grid_csv(period_dir / "phase_velocity_model_grid.csv", grid, ray_count=ray_counts, velocity=velocity)
        np.savez_compressed(
            period_dir / "phase_velocity_model_grid.npz",
            velocity_km_s=velocity,
            ray_count=ray_counts,
            lon_edges=grid.lon_edges,
            lat_edges=grid.lat_edges,
            residual_s=np.asarray(model["residual_s"], dtype=float),
            robust_weights=np.asarray(model["robust_weights"], dtype=float),
        )
        visible = velocity[(ray_counts >= args.coverage_contour) & np.isfinite(velocity)]
        path_velocities = np.asarray([float(row["phase_velocity_km_s"]) for row in clipped], dtype=float)
        summary_periods[str(period_key)] = {
            "input_corrected_count": len(rows_by_period[period_key]),
            "clipped_path_count": len(clipped),
            "ray_count_max_per_grid": float(np.nanmax(ray_counts)) if ray_counts.size else 0.0,
            "path_velocity_p05_p50_p95": [float(v) for v in np.percentile(path_velocities, [5, 50, 95])]
            if path_velocities.size
            else None,
            "visible_model_p02_p50_p98": [float(v) for v in np.percentile(visible, [2, 50, 98])]
            if visible.size
            else None,
            "ref_velocity_km_s": float(model["ref_velocity_km_s"]),
            "residual_mad_s": float(model["residual_mad_s"]),
        }

    figure5_path = figures_dir / "wang_figure5_style_measurements_and_ray_coverage.png"
    plot_figure5(
        figure5_path,
        periods=periods,
        rows_by_period=clipped_by_period,
        coverage_by_period=coverage_by_period,
        grid=grid,
        stations=stations,
        bbox=bbox,
        coverage_vmax=args.coverage_vmax,
        dpi=args.dpi,
    )
    figure6_path = figures_dir / "wang_figure6_style_phase_velocity_maps.png"
    figure6_pixel_path = figures_dir / "wang_figure6_style_phase_velocity_maps_pixel.png"
    figure6_smooth_path = figures_dir / "wang_figure6_style_phase_velocity_maps_paper_smooth_square.png"
    figure6_plot_data_path = data_dir / "wang_figure6_plot_data_paper_smooth_square.npz"
    plot_figure6(
        figure6_pixel_path,
        periods=periods,
        model_by_period=model_by_period,
        coverage_by_period=coverage_by_period,
        grid=grid,
        stations=stations,
        bbox=bbox,
        coverage_contour=args.coverage_contour,
        dpi=args.dpi,
    )
    summary: Dict[str, object] = {
        "method": "Wang Figure 5/6 style maps from DisperPicker paper-standard corrected measurements",
        "measurements_csv": str(args.measurements_csv),
        "curves_dir": str(args.curves_dir),
        "output_dir": str(args.output_dir),
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
            "coverage_vmax": float(args.coverage_vmax),
        },
        "figures": {
            "figure5_style": str(figure5_path),
            "figure6_style": str(figure6_path),
            "figure6_style_pixel": str(figure6_pixel_path),
            "figure6_style_paper_smooth_square": str(figure6_smooth_path),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    REPLOT.run(
        argparse.Namespace(
            input_dir=args.output_dir,
            output_png=figure6_smooth_path,
            output_npz=figure6_plot_data_path,
            periods=periods,
            coverage_contour=args.coverage_contour,
            dpi=max(args.dpi, 240),
            smooth_sigma_cells=REPLOT.DEFAULT_SMOOTH_SIGMA_CELLS,
            upsample_factor=REPLOT.DEFAULT_UPSAMPLE_FACTOR,
        )
    )
    shutil.copy2(figure6_smooth_path, figure6_path)
    summary["plot_data"] = {
        "figure6_paper_smooth_square_npz": str(figure6_plot_data_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
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
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        default=DEFAULT_BBOX,
        help="minlon,minlat,maxlon,maxlat; default matches the Wang MSH plotting window.",
    )
    parser.add_argument("--grid-spacing-deg", type=float, default=0.01)
    parser.add_argument("--min-inside-km", type=float, default=0.5)
    parser.add_argument("--sample-step-km", type=float, default=0.25)
    parser.add_argument("--damping", type=float, default=8.0)
    parser.add_argument("--smoothing", type=float, default=30.0)
    parser.add_argument("--robust-iterations", type=int, default=4)
    parser.add_argument("--coverage-contour", type=float, default=20.0)
    parser.add_argument("--coverage-vmax", type=float, default=1000.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
