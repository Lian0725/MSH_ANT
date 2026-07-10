#!/usr/bin/env python3
"""Local 2-D phase-velocity inversion from prepared XD-1D path CSV files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None
from matplotlib.collections import PolyCollection
from scipy import sparse
from scipy.sparse.linalg import lsqr


def _load_aant_helpers():
    module_path = Path(__file__).with_name("aant_2014_phase_maps.py")
    spec = importlib.util.spec_from_file_location("_aant_2014_phase_maps_for_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AANT = _load_aant_helpers()

DEFAULT_BBOX = {
    "minlon": -122.3807,
    "minlat": 46.0822,
    "maxlon": -122.003,
    "maxlat": 46.3453,
}
DEFAULT_PERIODS = tuple(round(1.0 + 0.5 * idx, 1) for idx in range(9))
DEFAULT_WINDOWS = {
    1.0: (1.7, 3.10),
    1.5: (1.7, 3.15),
    2.0: (1.7, 3.25),
    2.5: (1.7, 3.30),
    3.0: (1.7, 3.35),
    3.5: (1.7, 3.45),
    4.0: (1.7, 3.55),
    4.5: (1.7, 3.65),
    5.0: (1.7, 3.75),
}
STRICT_PRIMARY_WINDOWS = {
    1.0: (1.7, 3.30),
    1.5: (1.7, 3.30),
    2.0: (1.7, 3.25),
    2.5: (1.7, 3.35),
    3.0: (1.7, 2.78),
    3.5: (1.7, 2.9325),
    4.0: (1.7, 2.9725),
    4.5: (1.7, 3.05),
    5.0: (1.7, 3.15),
}
MSH_LON = -122.188
MSH_LAT = 46.191


@dataclass(frozen=True)
class RegularGrid:
    bbox: Dict[str, float]
    lon_edges: np.ndarray
    lat_edges: np.ndarray
    nx: int
    ny: int
    cell_size_km: float

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny

    def polygons(self) -> list[list[tuple[float, float]]]:
        polygons: list[list[tuple[float, float]]] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                polygons.append(
                    [
                        (float(self.lon_edges[ix]), float(self.lat_edges[iy])),
                        (float(self.lon_edges[ix + 1]), float(self.lat_edges[iy])),
                        (float(self.lon_edges[ix + 1]), float(self.lat_edges[iy + 1])),
                        (float(self.lon_edges[ix]), float(self.lat_edges[iy + 1])),
                    ]
                )
        return polygons


def period_key(period: float) -> float:
    return round(float(period), 3)


def build_regular_grid(
    bbox: Mapping[str, float],
    *,
    cell_size_km: float = 1.0,
) -> RegularGrid:
    box = AANT.normalized_bbox(bbox)
    mid_lat = 0.5 * (box["minlat"] + box["maxlat"])
    dlon = float(cell_size_km) / AANT.lon_degree_width_km(mid_lat)
    dlat = float(cell_size_km) / AANT.lat_degree_height_km()
    lon_edges = np.arange(box["minlon"], box["maxlon"] + 0.999 * dlon, dlon)
    lat_edges = np.arange(box["minlat"], box["maxlat"] + 0.999 * dlat, dlat)
    if lon_edges[-1] < box["maxlon"]:
        lon_edges = np.append(lon_edges, box["maxlon"])
    if lat_edges[-1] < box["maxlat"]:
        lat_edges = np.append(lat_edges, box["maxlat"])
    return RegularGrid(
        bbox=box,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        nx=len(lon_edges) - 1,
        ny=len(lat_edges) - 1,
        cell_size_km=float(cell_size_km),
    )


def row_velocity(row: Mapping[str, object]) -> float:
    if "local_velocity_km_s" in row:
        return float(row["local_velocity_km_s"])
    if "phase_velocity_km_s" in row:
        return float(row["phase_velocity_km_s"])
    return float(row["velocity_km_s"])


def filter_rows_by_velocity_window(
    rows: Sequence[Mapping[str, object]],
    *,
    period: float,
    windows: Mapping[float, tuple[float, float]],
) -> list[Dict[str, object]]:
    window = windows.get(period_key(period))
    if window is None:
        return [dict(row) for row in rows]
    vmin, vmax = window
    kept: list[Dict[str, object]] = []
    for row in rows:
        velocity = row_velocity(row)
        if math.isfinite(velocity) and float(vmin) <= velocity <= float(vmax):
            kept.append(dict(row))
    return kept


def row_to_local_observation(
    row: Mapping[str, object],
    *,
    bbox: Mapping[str, float],
    mode: str = "segment",
    external_velocity_km_s: Optional[float] = None,
    min_inside_km: float = 2.0,
    min_inside_fraction: float = 0.0,
) -> Optional[Dict[str, object]]:
    box = AANT.normalized_bbox(bbox)
    source_lon = float(row["source_lon"])
    source_lat = float(row["source_lat"])
    receiver_lon = float(row["receiver_lon"])
    receiver_lat = float(row["receiver_lat"])
    clipped = AANT.clip_segment_to_bbox(source_lon, source_lat, receiver_lon, receiver_lat, box)
    if clipped is None:
        return None

    clipped_source_lon, clipped_source_lat, clipped_receiver_lon, clipped_receiver_lat = clipped
    inside_km = AANT.distance_km(
        clipped_source_lon,
        clipped_source_lat,
        clipped_receiver_lon,
        clipped_receiver_lat,
    )
    total_km = AANT.distance_km(source_lon, source_lat, receiver_lon, receiver_lat)
    if inside_km < float(min_inside_km) or total_km <= 0.0:
        return None
    inside_fraction = inside_km / total_km
    if inside_fraction < float(min_inside_fraction):
        return None

    observed_velocity = row_velocity(row)
    if observed_velocity <= 0.0 or not math.isfinite(observed_velocity):
        return None

    if mode == "segment":
        local_velocity = observed_velocity
    elif mode == "external_reference":
        if external_velocity_km_s is None or external_velocity_km_s <= 0.0:
            raise ValueError("external_velocity_km_s must be positive in external_reference mode.")
        outside_km = max(0.0, total_km - inside_km)
        local_time_s = total_km / observed_velocity - outside_km / float(external_velocity_km_s)
        if local_time_s <= 0.0 or not math.isfinite(local_time_s):
            return None
        local_velocity = inside_km / local_time_s
    else:
        raise ValueError(f"Unsupported local observation mode: {mode}")

    if local_velocity <= 0.0 or not math.isfinite(local_velocity):
        return None

    output = dict(row)
    output.update(
        {
            "original_source_lon": source_lon,
            "original_source_lat": source_lat,
            "original_receiver_lon": receiver_lon,
            "original_receiver_lat": receiver_lat,
            "source_lon": clipped_source_lon,
            "source_lat": clipped_source_lat,
            "receiver_lon": clipped_receiver_lon,
            "receiver_lat": clipped_receiver_lat,
            "inside_km": inside_km,
            "total_km": total_km,
            "inside_fraction": inside_fraction,
            "local_velocity_km_s": local_velocity,
        }
    )
    return output


def build_ray_matrix(
    rows: Sequence[Mapping[str, object]],
    grid: RegularGrid,
    *,
    sample_step_km: float = 0.25,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_vals: list[float] = []
    coverage = np.zeros(grid.cell_count, dtype=float)

    for row_index, row in enumerate(rows):
        source_lon = float(row["source_lon"])
        source_lat = float(row["source_lat"])
        receiver_lon = float(row["receiver_lon"])
        receiver_lat = float(row["receiver_lat"])
        inside_km = float(row.get("inside_km") or AANT.distance_km(source_lon, source_lat, receiver_lon, receiver_lat))
        sample_count = max(4, int(math.ceil(inside_km / float(sample_step_km))))
        accum: Dict[int, float] = {}
        prev_lon, prev_lat = source_lon, source_lat
        for sample_index in range(1, sample_count + 1):
            fraction = sample_index / sample_count
            cur_lon = source_lon + (receiver_lon - source_lon) * fraction
            cur_lat = source_lat + (receiver_lat - source_lat) * fraction
            mid_lon = 0.5 * (prev_lon + cur_lon)
            mid_lat = 0.5 * (prev_lat + cur_lat)
            segment_km = AANT.distance_km(prev_lon, prev_lat, cur_lon, cur_lat)
            ix = int(np.searchsorted(grid.lon_edges, mid_lon, side="right") - 1)
            iy = int(np.searchsorted(grid.lat_edges, mid_lat, side="right") - 1)
            if 0 <= ix < grid.nx and 0 <= iy < grid.ny:
                cell = iy * grid.nx + ix
                accum[cell] = accum.get(cell, 0.0) + segment_km
                coverage[cell] += segment_km
            prev_lon, prev_lat = cur_lon, cur_lat

        for cell, length_km in accum.items():
            matrix_rows.append(row_index)
            matrix_cols.append(cell)
            matrix_vals.append(length_km)

    matrix = sparse.coo_matrix(
        (matrix_vals, (matrix_rows, matrix_cols)),
        shape=(len(rows), grid.cell_count),
    ).tocsr()
    return matrix, coverage


def build_regularization(
    grid: RegularGrid,
    *,
    damping: float,
    smoothing: float,
) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    row_index = 0
    for cell in range(grid.cell_count):
        rows.append(row_index)
        cols.append(cell)
        vals.append(float(damping))
        row_index += 1
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            cell = iy * grid.nx + ix
            if ix + 1 < grid.nx:
                rows.extend([row_index, row_index])
                cols.extend([cell, cell + 1])
                vals.extend([float(smoothing), -float(smoothing)])
                row_index += 1
            if iy + 1 < grid.ny:
                rows.extend([row_index, row_index])
                cols.extend([cell, cell + grid.nx])
                vals.extend([float(smoothing), -float(smoothing)])
                row_index += 1
    return sparse.coo_matrix((vals, (rows, cols)), shape=(row_index, grid.cell_count)).tocsr()


def solve_slowness_model(
    rows: Sequence[Mapping[str, object]],
    grid: RegularGrid,
    *,
    damping: float = 8.0,
    smoothing: float = 30.0,
    sample_step_km: float = 0.25,
    ref_velocity_km_s: Optional[float] = None,
    robust_iterations: int = 4,
) -> Dict[str, object]:
    if not rows:
        raise ValueError("Cannot invert an empty row set.")

    matrix, coverage = build_ray_matrix(rows, grid, sample_step_km=sample_step_km)
    velocities = np.asarray([row_velocity(row) for row in rows], dtype=float)
    distances = np.asarray([float(row["inside_km"]) for row in rows], dtype=float)
    data_time = distances / velocities
    if ref_velocity_km_s is None:
        ref_velocity_km_s = float(np.median(velocities))
    ref_slowness = 1.0 / float(ref_velocity_km_s)
    rhs_data = data_time - matrix @ np.full(grid.cell_count, ref_slowness, dtype=float)
    reg = build_regularization(grid, damping=damping, smoothing=smoothing)
    reg_rhs = np.zeros(reg.shape[0], dtype=float)

    weights = np.ones(len(rows), dtype=float)
    solution = np.zeros(grid.cell_count, dtype=float)
    residual = np.zeros(len(rows), dtype=float)
    mad = math.nan
    for _ in range(max(1, int(robust_iterations))):
        weighted_matrix = sparse.diags(weights) @ matrix
        system = sparse.vstack([weighted_matrix, reg], format="csr")
        rhs = np.concatenate([weights * rhs_data, reg_rhs])
        solution = lsqr(system, rhs, atol=1e-6, btol=1e-6, iter_lim=800)[0]
        residual = matrix @ solution - rhs_data
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)) * 1.4826 + 1e-6)
        weights = np.minimum(1.0, (2.5 * mad) / (np.abs(residual - median) + 1e-12))

    slowness = ref_slowness + solution
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = 1.0 / slowness
    return {
        "matrix": matrix,
        "coverage_km": coverage,
        "velocity_km_s": velocity,
        "slowness_s_km": slowness,
        "ref_velocity_km_s": float(ref_velocity_km_s),
        "residual_s": residual,
        "robust_weights": weights,
        "residual_mad_s": float(mad),
    }


def write_local_observations(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "source_code",
        "receiver_code",
        "period",
        "phase_velocity_km_s",
        "local_velocity_km_s",
        "inside_km",
        "total_km",
        "inside_fraction",
        "source_lon",
        "source_lat",
        "receiver_lon",
        "receiver_lat",
        "original_source_lon",
        "original_source_lat",
        "original_receiver_lon",
        "original_receiver_lat",
    ]
    extras = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fields = [field for field in preferred if any(field in row for row in rows)] + extras
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_model_csv(path: Path, grid: RegularGrid, velocity: np.ndarray, coverage: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cell_index",
                "lat_min",
                "lat_max",
                "lon_min",
                "lon_max",
                "coverage_km",
                "phase_velocity_km_s",
            ],
        )
        writer.writeheader()
        for iy in range(grid.ny):
            for ix in range(grid.nx):
                cell = iy * grid.nx + ix
                writer.writerow(
                    {
                        "cell_index": cell,
                        "lat_min": f"{grid.lat_edges[iy]:.6f}",
                        "lat_max": f"{grid.lat_edges[iy + 1]:.6f}",
                        "lon_min": f"{grid.lon_edges[ix]:.6f}",
                        "lon_max": f"{grid.lon_edges[ix + 1]:.6f}",
                        "coverage_km": f"{float(coverage[cell]):.6f}",
                        "phase_velocity_km_s": f"{float(velocity[cell]):.6f}",
                    }
                )


def plot_velocity_map(
    path: Path,
    *,
    grid: RegularGrid,
    velocity: np.ndarray,
    coverage: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    period: float,
    min_coverage_km: float,
    cmap: str = "RdBu",
    dpi: int = 220,
) -> None:
    values = np.asarray(velocity, dtype=float).copy()
    values[coverage < float(min_coverage_km)] = np.nan
    finite = values[np.isfinite(values)]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [2, 98])
    else:
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    collection = PolyCollection(
        grid.polygons(),
        array=np.ma.masked_invalid(values),
        cmap=cmap,
        edgecolors="none",
        linewidths=0.0,
    )
    if vmin is not None and vmax is not None and vmin < vmax:
        collection.set_clim(float(vmin), float(vmax))
    ax.add_collection(collection)

    if rows:
        step = max(1, len(rows) // 1800)
        for row in rows[::step]:
            ax.plot(
                [float(row["source_lon"]), float(row["receiver_lon"])],
                [float(row["source_lat"]), float(row["receiver_lat"])],
                color="0.15",
                alpha=0.035,
                linewidth=0.25,
                zorder=3,
            )
    ax.scatter([MSH_LON], [MSH_LAT], marker="*", s=130, color="gold", edgecolor="black", zorder=5)
    box = grid.bbox
    ax.set_xlim(box["minlon"], box["maxlon"])
    ax.set_ylim(box["minlat"], box["maxlat"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Local XD-1D phase velocity at {float(period):.1f} s")
    cbar = fig.colorbar(collection, ax=ax, pad=0.02)
    cbar.set_label("Phase velocity (km/s); red=low, blue=high")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run_period(
    *,
    path_csv: Path,
    output_dir: Path,
    period: float,
    bbox: Mapping[str, float],
    windows: Mapping[float, tuple[float, float]],
    mode: str,
    external_velocity_km_s: Optional[float],
    source_median_external: bool,
    min_inside_km: float,
    min_inside_fraction: float,
    cell_size_km: float,
    damping: float,
    smoothing: float,
    min_coverage_km: float,
    sample_step_km: float,
    robust_iterations: int,
    plot_dpi: int,
) -> Dict[str, object]:
    if pd is None:
        raise ModuleNotFoundError("pandas is required when running local_phase_velocity_maps.py directly")
    dataframe = pd.read_csv(path_csv)
    raw_rows = dataframe.to_dict("records")
    window_rows = filter_rows_by_velocity_window(raw_rows, period=period, windows=windows)

    external_by_source: Dict[str, float] = {}
    if source_median_external:
        filtered = pd.DataFrame(window_rows)
        if not filtered.empty:
            external_by_source = (
                filtered.groupby("source_code")["phase_velocity_km_s"].median().astype(float).to_dict()
            )

    local_rows: list[Dict[str, object]] = []
    for row in window_rows:
        row_external_velocity = external_velocity_km_s
        if source_median_external:
            row_external_velocity = external_by_source.get(str(row.get("source_code")))
        observation = row_to_local_observation(
            row,
            bbox=bbox,
            mode=mode,
            external_velocity_km_s=row_external_velocity,
            min_inside_km=min_inside_km,
            min_inside_fraction=min_inside_fraction,
        )
        if observation is not None:
            local_rows.append(observation)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_local_observations(output_dir / "local_observations.csv", local_rows)
    if not local_rows:
        return {
            "status": "insufficient_paths",
            "raw_path_count": len(raw_rows),
            "window_path_count": len(window_rows),
            "local_path_count": 0,
        }

    grid = build_regular_grid(bbox, cell_size_km=cell_size_km)
    result = solve_slowness_model(
        local_rows,
        grid,
        damping=damping,
        smoothing=smoothing,
        sample_step_km=sample_step_km,
        robust_iterations=robust_iterations,
    )
    velocity = np.asarray(result["velocity_km_s"], dtype=float)
    coverage = np.asarray(result["coverage_km"], dtype=float)
    model_csv = output_dir / "local_phase_velocity_model.csv"
    write_model_csv(model_csv, grid, velocity, coverage)
    model_npz = output_dir / "local_phase_velocity_model.npz"
    np.savez_compressed(
        model_npz,
        velocity_km_s=velocity,
        coverage_km=coverage,
        lon_edges=grid.lon_edges,
        lat_edges=grid.lat_edges,
        residual_s=result["residual_s"],
        robust_weights=result["robust_weights"],
    )
    map_png = output_dir / f"local_phase_velocity_{AANT.format_period(period)}s.png"
    plot_velocity_map(
        map_png,
        grid=grid,
        velocity=velocity,
        coverage=coverage,
        rows=local_rows,
        period=period,
        min_coverage_km=min_coverage_km,
        dpi=plot_dpi,
    )
    finite_visible = velocity[(coverage >= float(min_coverage_km)) & np.isfinite(velocity)]
    local_velocities = np.asarray([row["local_velocity_km_s"] for row in local_rows], dtype=float)
    return {
        "status": "inverted",
        "raw_path_count": len(raw_rows),
        "window_path_count": len(window_rows),
        "local_path_count": len(local_rows),
        "local_velocity_p05_p50_p95": [float(value) for value in np.percentile(local_velocities, [5, 50, 95])],
        "model_p02_p50_p98": [float(value) for value in np.percentile(finite_visible, [2, 50, 98])]
        if finite_visible.size
        else None,
        "model_csv": str(model_csv.resolve()),
        "model_npz": str(model_npz.resolve()),
        "map_png": str(map_png.resolve()),
        "ref_velocity_km_s": float(result["ref_velocity_km_s"]),
        "residual_mad_s": float(result["residual_mad_s"]),
        "robust_weight_gt_0p5": int(np.count_nonzero(np.asarray(result["robust_weights"]) > 0.5)),
        "grid": {"nx": grid.nx, "ny": grid.ny, "cell_count": grid.cell_count},
    }


def plot_montage(summary: Mapping[str, object], output_path: Path) -> None:
    periods = [
        (float(period), info)
        for period, info in sorted(summary["periods"].items(), key=lambda item: float(item[0]))
        if isinstance(info, Mapping) and info.get("status") == "inverted"
    ]
    if not periods:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (period, info) in zip(axes.ravel(), periods):
        data = np.load(str(info["model_npz"]))
        velocity = data["velocity_km_s"]
        coverage = data["coverage_km"]
        lon_edges = data["lon_edges"]
        lat_edges = data["lat_edges"]
        grid = RegularGrid(
            bbox=dict(summary["parameters"]["bbox"]),
            lon_edges=lon_edges,
            lat_edges=lat_edges,
            nx=len(lon_edges) - 1,
            ny=len(lat_edges) - 1,
            cell_size_km=float(summary["parameters"]["cell_size_km"]),
        )
        values = velocity.copy()
        values[coverage < float(summary["parameters"]["min_coverage_km"])] = np.nan
        finite = values[np.isfinite(values)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [2, 98])
        else:
            vmin, vmax = None, None
        collection = PolyCollection(
            grid.polygons(),
            array=np.ma.masked_invalid(values),
            cmap="RdBu",
            edgecolors="none",
            linewidths=0.0,
        )
        if vmin is not None and vmax is not None and vmin < vmax:
            collection.set_clim(float(vmin), float(vmax))
        ax.add_collection(collection)
        ax.scatter([MSH_LON], [MSH_LAT], marker="*", s=80, color="gold", edgecolor="black", zorder=5)
        box = grid.bbox
        ax.set_xlim(box["minlon"], box["maxlon"])
        ax.set_ylim(box["minlat"], box["maxlat"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{period:.1f} s")
        ax.axis("on")
        cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("km/s")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(summary["parameters"].get("plot_dpi", 220)))
    plt.close(fig)


def parse_periods(value: str) -> list[float]:
    periods = [round(float(item.strip()), 1) for item in value.split(",") if item.strip()]
    if not periods:
        raise argparse.ArgumentTypeError("Use at least one period, e.g. 1.0,1.5")
    return periods


def parse_bbox(value: str) -> Dict[str, float]:
    return AANT.parse_bbox(value)


def window_preset(name: str) -> Mapping[float, tuple[float, float]]:
    if name == "soft":
        return DEFAULT_WINDOWS
    if name == "strict-primary":
        return STRICT_PRIMARY_WINDOWS
    if name == "none":
        return {}
    raise ValueError(f"Unsupported window preset: {name}")


def run_workspace(
    *,
    input_root: Path,
    output_root: Path,
    periods: Sequence[float],
    bbox: Mapping[str, float],
    windows: Mapping[float, tuple[float, float]],
    mode: str,
    external_velocity_km_s: Optional[float],
    source_median_external: bool,
    min_inside_km: float,
    min_inside_fraction: float,
    cell_size_km: float,
    damping: float,
    smoothing: float,
    min_coverage_km: float,
    sample_step_km: float,
    robust_iterations: int,
    plot_dpi: int,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {
        "method": "local clipped-path LSQR phase-velocity inversion",
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "parameters": {
            "periods": [round(float(period), 1) for period in periods],
            "bbox": AANT.normalized_bbox(bbox),
            "windows": {str(period): list(window) for period, window in windows.items()},
            "mode": mode,
            "external_velocity_km_s": external_velocity_km_s,
            "source_median_external": bool(source_median_external),
            "min_inside_km": float(min_inside_km),
            "min_inside_fraction": float(min_inside_fraction),
            "cell_size_km": float(cell_size_km),
            "damping": float(damping),
            "smoothing": float(smoothing),
            "min_coverage_km": float(min_coverage_km),
            "sample_step_km": float(sample_step_km),
            "robust_iterations": int(robust_iterations),
            "plot_dpi": int(plot_dpi),
        },
        "periods": {},
    }
    for period in periods:
        key = AANT.format_period(period)
        path_csv = input_root / f"period_{key}s" / "paths.csv"
        period_output = output_root / f"period_{key}s"
        if not path_csv.exists():
            summary["periods"][key] = {"status": "missing_paths_csv", "path_csv": str(path_csv)}
            continue
        summary["periods"][key] = run_period(
            path_csv=path_csv,
            output_dir=period_output,
            period=float(period),
            bbox=bbox,
            windows=windows,
            mode=mode,
            external_velocity_km_s=external_velocity_km_s,
            source_median_external=source_median_external,
            min_inside_km=min_inside_km,
            min_inside_fraction=min_inside_fraction,
            cell_size_km=cell_size_km,
            damping=damping,
            smoothing=smoothing,
            min_coverage_km=min_coverage_km,
            sample_step_km=sample_step_km,
            robust_iterations=robust_iterations,
            plot_dpi=plot_dpi,
        )
    montage = output_root / "local_phase_velocity_1to5s_montage.png"
    summary["montage_png"] = str(montage.resolve())
    plot_montage(summary, montage)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Invert local XD-1D phase-velocity maps from paths.csv files.")
    parser.add_argument("input_root", type=Path, help="Prepared AANT path workspace with period_*s/paths.csv files.")
    parser.add_argument("output_root", type=Path, help="Output directory.")
    parser.add_argument("--periods", type=parse_periods, default=list(DEFAULT_PERIODS))
    parser.add_argument("--bbox", type=parse_bbox, default=DEFAULT_BBOX)
    parser.add_argument(
        "--window-preset",
        choices=["soft", "strict-primary", "none"],
        default="soft",
        help="Period velocity windows applied before inversion.",
    )
    parser.add_argument(
        "--mode",
        choices=["segment", "external_reference"],
        default="segment",
        help="How whole-path velocities are converted to local clipped-path observations.",
    )
    parser.add_argument("--external-velocity-km-s", type=float)
    parser.add_argument(
        "--source-median-external",
        action="store_true",
        help="Use each XD source's period median velocity as the external reference velocity.",
    )
    parser.add_argument("--min-inside-km", type=float, default=2.0)
    parser.add_argument("--min-inside-fraction", type=float, default=0.0)
    parser.add_argument("--cell-size-km", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=8.0)
    parser.add_argument("--smoothing", type=float, default=30.0)
    parser.add_argument("--min-coverage-km", type=float, default=10.0)
    parser.add_argument("--sample-step-km", type=float, default=0.25)
    parser.add_argument("--robust-iterations", type=int, default=4)
    parser.add_argument("--plot-dpi", type=int, default=220)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.mode == "external_reference" and not args.source_median_external and args.external_velocity_km_s is None:
        parser.error("--mode external_reference requires --external-velocity-km-s or --source-median-external")
    summary = run_workspace(
        input_root=args.input_root,
        output_root=args.output_root,
        periods=args.periods,
        bbox=args.bbox,
        windows=window_preset(args.window_preset),
        mode=args.mode,
        external_velocity_km_s=args.external_velocity_km_s,
        source_median_external=args.source_median_external,
        min_inside_km=args.min_inside_km,
        min_inside_fraction=args.min_inside_fraction,
        cell_size_km=args.cell_size_km,
        damping=args.damping,
        smoothing=args.smoothing,
        min_coverage_km=args.min_coverage_km,
        sample_step_km=args.sample_step_km,
        robust_iterations=args.robust_iterations,
        plot_dpi=args.plot_dpi,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
