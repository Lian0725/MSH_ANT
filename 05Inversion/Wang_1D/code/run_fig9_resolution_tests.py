#!/usr/bin/env python3
"""Resolution / uncertainty tests for the Wang Fig.9 phase-velocity inversion.

Addresses review items 2-4 of the fig9 pipeline critique:
  2. synthetic recovery tests (checkerboard, crater low-velocity anomaly,
     single-cell spike / point-spread function) with realistic noise;
  3. objective damping/smoothing selection: parameter scan scored by
     checkerboard recovery + L-curve on the real data;
  4. km-normalized first-difference smoothing operator (isotropic physical
     smoothing length) compared against the legacy grid-unit operator.

Uses the exact same ray geometry as the production Fig.9 run: measurements
CSV -> azimuthal correction -> bbox clipping -> straight-ray matrix. All maps
are rendered on the raw inversion grid (no gaussian smoothing / upsampling).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

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


FIG9 = _load_module("_fig9_pipeline_for_resolution_tests", "plot_wang_fig9_azimuthal_correction.py")
LOCAL = FIG9.LOCAL
AANT = FIG9.AANT

CRATER_LON = LOCAL.MSH_LON
CRATER_LAT = LOCAL.MSH_LAT
COVERAGE_MIN_RAYS = 20.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_float_list(value: str) -> Tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_int_list(value: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


# ---------------------------------------------------------------------------
# geometry / data assembly (mirrors plot_wang_fig9_azimuthal_correction.run)
# ---------------------------------------------------------------------------

def corrected_rows_for_period(rows: Sequence[Mapping[str, object]], fit: Mapping[str, object]) -> List[Dict[str, object]]:
    corrected: List[Dict[str, object]] = []
    for row, t_corr in zip(rows, fit["corrected_travel_time_s"]):
        t_corr = float(t_corr)
        if not (math.isfinite(t_corr) and t_corr > 0.2):
            continue
        new_row = dict(row)
        new_row["travel_time_s"] = t_corr
        new_row["phase_velocity_km_s"] = float(row["distance_km"]) / t_corr
        corrected.append(new_row)
    return corrected


def assemble_period_geometry(
    rows: Sequence[Mapping[str, object]],
    *,
    bbox: Mapping[str, float],
    grid: object,
    min_inside_km: float,
    sample_step_km: float,
) -> Dict[str, object]:
    fit = FIG9.fit_azimuthal_correction(rows, bin_width_deg=20.0)
    corrected = corrected_rows_for_period(rows, fit)
    clipped = FIG9.clipped_rows_for_bbox(corrected, bbox=bbox, min_inside_km=min_inside_km)
    matrix, coverage_km = LOCAL.build_ray_matrix(clipped, grid, sample_step_km=sample_step_km)
    ray_count = FIG9.ray_count_grid(clipped, grid, sample_step_km=sample_step_km)
    velocities = np.asarray([LOCAL.row_velocity(row) for row in clipped], dtype=float)
    distances = np.asarray([float(row["inside_km"]) for row in clipped], dtype=float)
    data_time = distances / velocities
    ref_velocity = float(np.median(velocities))
    ref_slowness = 1.0 / ref_velocity
    rhs_data = data_time - matrix @ np.full(grid.cell_count, ref_slowness, dtype=float)
    return {
        "matrix": matrix,
        "coverage_km": coverage_km,
        "ray_count": ray_count,
        "rhs_data": rhs_data,
        "ref_velocity": ref_velocity,
        "ref_slowness": ref_slowness,
        "n_paths": len(clipped),
    }


# ---------------------------------------------------------------------------
# inversion core (same objective as LOCAL.solve_slowness_model)
# ---------------------------------------------------------------------------

def invert_linear(
    matrix: sparse.csr_matrix,
    rhs_data: np.ndarray,
    grid: object,
    *,
    damping: float,
    smoothing: float,
    km_normalized: bool = False,
    robust_iterations: int = 1,
) -> Dict[str, object]:
    """robust_iterations=1 -> plain least squares (weights stay 1)."""
    reg = LOCAL.build_regularization(grid, damping=damping, smoothing=smoothing, km_normalized=km_normalized)
    reg_rhs = np.zeros(reg.shape[0], dtype=float)
    weights = np.ones(matrix.shape[0], dtype=float)
    solution = np.zeros(grid.cell_count, dtype=float)
    residual = np.zeros(matrix.shape[0], dtype=float)
    for _ in range(max(1, int(robust_iterations))):
        weighted_matrix = sparse.diags(weights) @ matrix
        system = sparse.vstack([weighted_matrix, reg], format="csr")
        rhs = np.concatenate([weights * rhs_data, reg_rhs])
        solution = lsqr(system, rhs, atol=1e-6, btol=1e-6, iter_lim=800)[0]
        residual = matrix @ solution - rhs_data
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)) * 1.4826 + 1e-6)
        weights = np.minimum(1.0, (2.5 * mad) / (np.abs(residual - median) + 1e-12))
    return {
        "solution": solution,
        "residual": residual,
        "misfit_rms_s": float(np.sqrt(np.mean(residual**2))),
        "model_norm": float(np.linalg.norm(solution)),
        "residual_mad_s": float(np.median(np.abs(residual - np.median(residual))) * 1.4826),
    }


# ---------------------------------------------------------------------------
# synthetic models (fractional slowness perturbation grids, shape ny x nx)
# ---------------------------------------------------------------------------

def cell_centers(grid: object) -> Tuple[np.ndarray, np.ndarray]:
    lon_c = 0.5 * (np.asarray(grid.lon_edges[:-1], dtype=float) + np.asarray(grid.lon_edges[1:], dtype=float))
    lat_c = 0.5 * (np.asarray(grid.lat_edges[:-1], dtype=float) + np.asarray(grid.lat_edges[1:], dtype=float))
    return lon_c, lat_c


def checkerboard_model(grid: object, size_cells: int, amplitude: float) -> np.ndarray:
    iy, ix = np.mgrid[0:grid.ny, 0:grid.nx]
    sign = np.where(((ix // size_cells) + (iy // size_cells)) % 2 == 0, 1.0, -1.0)
    return amplitude * sign


def crater_gaussian_model(grid: object, amplitude: float, sigma_km: float) -> np.ndarray:
    lon_c, lat_c = cell_centers(grid)
    model = np.zeros((grid.ny, grid.nx), dtype=float)
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            r_km = AANT.distance_km(float(lon_c[ix]), float(lat_c[iy]), CRATER_LON, CRATER_LAT)
            model[iy, ix] = amplitude * math.exp(-0.5 * (r_km / sigma_km) ** 2)
    return model


def spike_model(grid: object, amplitude: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    ix = int(np.searchsorted(grid.lon_edges, CRATER_LON, side="right") - 1)
    iy = int(np.searchsorted(grid.lat_edges, CRATER_LAT, side="right") - 1)
    ix = min(max(ix, 0), grid.nx - 1)
    iy = min(max(iy, 0), grid.ny - 1)
    model = np.zeros((grid.ny, grid.nx), dtype=float)
    model[iy, ix] = amplitude
    return model, (iy, ix)


def synthetic_inversion(
    geometry: Mapping[str, object],
    grid: object,
    true_frac: np.ndarray,
    *,
    damping: float,
    smoothing: float,
    km_normalized: bool,
    noise_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    ref_slowness = float(geometry["ref_slowness"])
    ds_true = (true_frac * ref_slowness).ravel()
    data = geometry["matrix"] @ ds_true + rng.normal(0.0, noise_sd, geometry["matrix"].shape[0])
    result = invert_linear(
        geometry["matrix"], data, grid,
        damping=damping, smoothing=smoothing, km_normalized=km_normalized,
        robust_iterations=1,
    )
    return (result["solution"] / ref_slowness).reshape(grid.ny, grid.nx)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def recovery_metrics(true_frac: np.ndarray, rec_frac: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    t = true_frac[mask]
    r = rec_frac[mask]
    if t.size < 3 or float(np.dot(t, t)) <= 0:
        return {"correlation": float("nan"), "amplitude_slope": float("nan")}
    corr = float(np.corrcoef(t, r)[0, 1])
    slope = float(np.dot(t, r) / np.dot(t, t))
    return {"correlation": corr, "amplitude_slope": slope}


def crater_metrics(
    true_frac: np.ndarray,
    rec_frac: np.ndarray,
    mask: np.ndarray,
    grid: object,
    amplitude: float,
    radius_km: float = 6.0,
) -> Dict[str, float]:
    lon_c, lat_c = cell_centers(grid)
    dist = np.zeros((grid.ny, grid.nx), dtype=float)
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            dist[iy, ix] = AANT.distance_km(float(lon_c[ix]), float(lat_c[iy]), CRATER_LON, CRATER_LAT)
    region = mask & (dist <= radius_km)
    if not region.any():
        return {"peak_recovery_fraction": float("nan"), "centroid_offset_km": float("nan")}
    peak = float(np.max(rec_frac[region]))
    weights = np.clip(rec_frac, 0.0, None) * region
    if weights.sum() <= 0:
        centroid_offset = float("nan")
    else:
        lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)
        c_lon = float(np.sum(lon_grid * weights) / weights.sum())
        c_lat = float(np.sum(lat_grid * weights) / weights.sum())
        centroid_offset = AANT.distance_km(c_lon, c_lat, CRATER_LON, CRATER_LAT)
    return {
        "peak_recovery_fraction": peak / amplitude,
        "centroid_offset_km": float(centroid_offset),
    }


def psf_fwhm_km(rec_frac: np.ndarray, spike_cell: Tuple[int, int], grid: object) -> Dict[str, float]:
    lon_c, lat_c = cell_centers(grid)
    iy0, ix0 = spike_cell
    # locate actual recovered peak near the spike (within 3 cells)
    y0, y1 = max(0, iy0 - 3), min(grid.ny, iy0 + 4)
    x0, x1 = max(0, ix0 - 3), min(grid.nx, ix0 + 4)
    sub = rec_frac[y0:y1, x0:x1]
    dy, dx = np.unravel_index(int(np.argmax(sub)), sub.shape)
    py, px = y0 + dy, x0 + dx
    peak = float(rec_frac[py, px])
    if peak <= 0:
        return {"peak_recovered_frac": peak, "fwhm_ew_km": float("nan"), "fwhm_ns_km": float("nan")}
    half = 0.5 * peak

    def span(values: np.ndarray, centers_km: np.ndarray, pivot: int) -> float:
        lo = pivot
        while lo - 1 >= 0 and values[lo - 1] >= half:
            lo -= 1
        hi = pivot
        while hi + 1 < values.size and values[hi + 1] >= half:
            hi += 1
        return abs(centers_km[hi] - centers_km[lo]) + abs(
            centers_km[min(hi + 1, values.size - 1)] - centers_km[hi]
        )  # + one cell width

    ew_km = np.asarray(
        [AANT.distance_km(float(lon_c[0]), float(lat_c[py]), float(lon_c[i]), float(lat_c[py])) for i in range(grid.nx)]
    )
    ns_km = np.asarray(
        [AANT.distance_km(float(lon_c[px]), float(lat_c[0]), float(lon_c[px]), float(lat_c[i])) for i in range(grid.ny)]
    )
    return {
        "peak_recovered_frac": peak,
        "fwhm_ew_km": span(rec_frac[py, :], ew_km, px),
        "fwhm_ns_km": span(rec_frac[:, px], ns_km, py),
    }


# ---------------------------------------------------------------------------
# plotting (raw grid, no smoothing/upsampling)
# ---------------------------------------------------------------------------

def bilinear_display_values(raw: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return raw values and the display-only interpolation selected by the user."""
    return raw, "bilinear"


def draw_display_map(
    ax,
    frac: np.ndarray,
    grid: object,
    ray_count_2d: np.ndarray,
    *,
    vlim_pct: float,
    title: str,
) -> object:
    shown, interpolation = bilinear_display_values(frac)
    mesh = ax.imshow(
        np.ma.masked_invalid(100.0 * shown),
        extent=(grid.lon_edges[0], grid.lon_edges[-1], grid.lat_edges[0], grid.lat_edges[-1]),
        origin="lower", cmap="RdBu_r", vmin=-vlim_pct, vmax=vlim_pct,
        interpolation=interpolation,
    )
    lon_c, lat_c = cell_centers(grid)
    if np.nanmax(ray_count_2d) >= COVERAGE_MIN_RAYS:
        ax.contour(lon_c, lat_c, ray_count_2d, levels=[COVERAGE_MIN_RAYS], colors="#222222",
                   linewidths=0.9, linestyles="--")
    ax.plot(CRATER_LON, CRATER_LAT, marker="^", color="k", markersize=5, markerfacecolor="none")
    ax.set_facecolor("#ffffff")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0 / math.cos(math.radians(0.5 * (grid.lat_edges[0] + grid.lat_edges[-1]))))
    return mesh


def plot_display_recovery_panel(
    path: Path,
    *,
    grid: object,
    ray_count_2d: np.ndarray,
    rows: Sequence[Tuple[str, np.ndarray]],
    cols: Sequence[str],
    maps: Mapping[Tuple[int, int], np.ndarray],
    vlim_pct: float,
    suptitle: str,
    dpi: int,
) -> None:
    ensure_dir(path.parent)
    n_rows, n_cols = len(rows), len(cols) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.75 * n_cols, 3.2 * n_rows))
    axes = np.atleast_2d(axes)
    mesh = None
    for irow, (row_label, truth) in enumerate(rows):
        mesh = draw_display_map(axes[irow, 0], truth, grid, ray_count_2d, vlim_pct=vlim_pct, title=f"truth ({row_label})")
        for icol, col_label in enumerate(cols):
            rec = maps[(irow, icol)]
            draw_display_map(axes[irow, icol + 1], rec, grid, ray_count_2d, vlim_pct=vlim_pct, title=col_label)
    fig.suptitle(suptitle + "\n" + "bilinear display; metrics use raw grids", fontsize=10.5, y=1.03 if n_rows > 1 else 1.12)
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("slowness perturbation (%)", fontsize=9)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def draw_frac_map(
    ax,
    frac: np.ndarray,
    grid: object,
    ray_count_2d: np.ndarray,
    *,
    vlim_pct: float,
    title: str,
) -> object:
    mesh = ax.pcolormesh(
        grid.lon_edges,
        grid.lat_edges,
        100.0 * frac,
        cmap="RdBu_r",
        vmin=-vlim_pct,
        vmax=vlim_pct,
        shading="flat",
    )
    lon_c, lat_c = cell_centers(grid)
    if np.nanmax(ray_count_2d) >= COVERAGE_MIN_RAYS:
        ax.contour(lon_c, lat_c, ray_count_2d, levels=[COVERAGE_MIN_RAYS], colors="0.25", linewidths=1.0)
    ax.plot(CRATER_LON, CRATER_LAT, marker="^", color="k", markersize=6, markerfacecolor="none")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0 / math.cos(math.radians(0.5 * (grid.lat_edges[0] + grid.lat_edges[-1]))))
    return mesh


def plot_recovery_panel(
    path: Path,
    *,
    grid: object,
    ray_count_2d: np.ndarray,
    rows: Sequence[Tuple[str, np.ndarray]],
    cols: Sequence[str],
    maps: Mapping[Tuple[int, int], np.ndarray],
    vlim_pct: float,
    suptitle: str,
    dpi: int,
) -> None:
    ensure_dir(path.parent)
    n_rows, n_cols = len(rows), len(cols) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.1 * n_rows))
    axes = np.atleast_2d(axes)
    mesh = None
    for irow, (row_label, truth) in enumerate(rows):
        mesh = draw_frac_map(
            axes[irow, 0], truth, grid, ray_count_2d,
            vlim_pct=vlim_pct, title=f"truth ({row_label})",
        )
        for icol, col_label in enumerate(cols):
            rec = maps[(irow, icol)]
            peak = 100.0 * float(np.nanmax(np.abs(rec)))
            draw_frac_map(
                axes[irow, icol + 1], rec, grid, ray_count_2d,
                vlim_pct=vlim_pct, title=f"{col_label}\n|peak|={peak:.1f}%",
            )
    fig.suptitle(suptitle, fontsize=11, y=1.02 if n_rows > 1 else 1.12)
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("slowness perturbation (%)", fontsize=9)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_lcurves(
    path: Path,
    *,
    periods: Sequence[float],
    scan: Mapping[float, List[Dict[str, object]]],
    baseline: Tuple[float, float],
    best: Tuple[float, float],
    dpi: int,
) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, len(periods), figsize=(4.4 * len(periods), 4.0))
    axes = np.atleast_1d(axes)
    for col, period in enumerate(periods):
        ax = axes[col]
        entries = scan[period]
        dampings = sorted({e["damping"] for e in entries})
        cmap = plt.get_cmap("viridis")
        for i, damping in enumerate(dampings):
            pts = sorted([e for e in entries if e["damping"] == damping], key=lambda e: e["smoothing"])
            x = [e["real_misfit_rms_s"] for e in pts]
            y = [e["real_model_norm"] for e in pts]
            ax.plot(x, y, "o-", ms=3.5, lw=1.0, color=cmap(i / max(1, len(dampings) - 1)), label=f"damping={damping:g}")
            for e in pts:
                ax.annotate(f"{e['smoothing']:g}", (e["real_misfit_rms_s"], e["real_model_norm"]), fontsize=6, alpha=0.7)
        for combo, color, label in ((baseline, "red", "baseline"), (best, "limegreen", "scan best")):
            for e in entries:
                if e["damping"] == combo[0] and e["smoothing"] == combo[1]:
                    ax.plot(e["real_misfit_rms_s"], e["real_model_norm"], "*", ms=14, color=color, label=label, zorder=5)
        ax.set_xlabel("data misfit RMS (s)")
        if col == 0:
            ax.set_ylabel("model norm ||ds||")
        ax.set_title(f"{period:g} s")
        ax.legend(fontsize=7)
    fig.suptitle("L-curves (real corrected data, robust LS as in production)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_scan_heatmaps(
    path: Path,
    *,
    periods: Sequence[float],
    scan: Mapping[float, List[Dict[str, object]]],
    dampings: Sequence[float],
    smoothings: Sequence[float],
    baseline: Tuple[float, float],
    best: Tuple[float, float],
    dpi: int,
) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(2, len(periods), figsize=(3.9 * len(periods), 7.0))
    axes = np.asarray(axes).reshape(2, len(periods))
    for col, period in enumerate(periods):
        entries = {(e["damping"], e["smoothing"]): e for e in scan[period]}
        corr = np.full((len(dampings), len(smoothings)), np.nan)
        slope = np.full_like(corr, np.nan)
        for i, d in enumerate(dampings):
            for j, s in enumerate(smoothings):
                e = entries[(d, s)]
                corr[i, j] = e["checker_correlation"]
                slope[i, j] = e["checker_amplitude_slope"]
        for row_idx, (values, label, vmin, vmax) in enumerate(
            ((corr, "checkerboard correlation", 0.0, 1.0), (slope, "amplitude recovery slope", 0.0, 1.2))
        ):
            ax = axes[row_idx, col]
            im = ax.imshow(values, cmap="magma", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
            ax.set_xticks(range(len(smoothings)), [f"{s:g}" for s in smoothings], fontsize=8)
            ax.set_yticks(range(len(dampings)), [f"{d:g}" for d in dampings], fontsize=8)
            ax.set_xlabel("smoothing", fontsize=9)
            ax.set_ylabel("damping", fontsize=9)
            for i in range(len(dampings)):
                for j in range(len(smoothings)):
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if values[i, j] < 0.6 * vmax else "black")
            for combo, color in ((baseline, "red"), (best, "limegreen")):
                if combo[0] in dampings and combo[1] in smoothings:
                    ax.add_patch(plt.Rectangle(
                        (smoothings.index(combo[1]) - 0.5, dampings.index(combo[0]) - 0.5), 1, 1,
                        fill=False, edgecolor=color, lw=2.2,
                    ))
            ax.set_title(f"{period:g} s: {label}", fontsize=10)
        fig.colorbar(im, ax=axes[:, col].tolist(), fraction=0.03, pad=0.02)
    fig.suptitle("Checkerboard (3x3-cell) recovery vs damping/smoothing\nred = baseline (8, 22), green = scan best; scored on ray_count>=20 cells", fontsize=11)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------

def render_display_only(args: argparse.Namespace) -> None:
    """Render the nine display panels from saved grids without touching raw outputs."""
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    parameters = summary.get("parameters", {})
    baseline = parameters.get("baseline", {})
    scan_best = summary.get("scan_best", {})
    col_labels = [
        f"baseline\n(d={baseline.get('damping', 8):g}, s={baseline.get('smoothing', 22):g})",
        f"scan_best\n(d={scan_best.get('damping', 8):g}, s={scan_best.get('smoothing', 22):g})",
        f"baseline_km_norm\n(d={baseline.get('damping', 8):g}, s={baseline.get('smoothing', 22):g}, km)",
    ]
    checker_sizes = parameters.get("checker_sizes_cells", [2, 3, 4])
    display_dir = args.output_dir / "figures_display_smoothed"
    for npz_path in sorted((args.output_dir / "data").glob("recovery_grids_*s.npz")):
        period_token = npz_path.stem.removeprefix("recovery_grids_").removesuffix("s")
        with np.load(npz_path) as data:
            ray_count = np.asarray(data["ray_count"], dtype=float)
            lon_edges = np.asarray(data["lon_edges"], dtype=float)
            lat_edges = np.asarray(data["lat_edges"], dtype=float)
            grid = SimpleNamespace(lon_edges=lon_edges, lat_edges=lat_edges, nx=lon_edges.size - 1, ny=lat_edges.size - 1)
            checker_truths = np.asarray(data["checker_truths"], dtype=float)
            checker_recovered = np.asarray(data["checker_recovered"], dtype=float)
            checker_rows = [
                (f"{checker_sizes[index] if index < len(checker_sizes) else index + 1}x{checker_sizes[index] if index < len(checker_sizes) else index + 1} cells", truth)
                for index, truth in enumerate(checker_truths)
            ]
            checker_maps = {(row, col): checker_recovered[row, col] for row in range(checker_recovered.shape[0]) for col in range(checker_recovered.shape[1])}
            checker_limit = 100.0 * max(0.01, float(np.nanmax(np.abs(np.concatenate([checker_truths.ravel(), checker_recovered.ravel()])))))
            plot_display_recovery_panel(
                display_dir / f"checkerboard_recovery_{period_token}s.png", grid=grid, ray_count_2d=ray_count,
                rows=checker_rows, cols=col_labels, maps=checker_maps, vlim_pct=checker_limit,
                suptitle=f"Checkerboard recovery, T={period_token} s", dpi=args.dpi,
            )
            crater_truth = np.asarray(data["crater_truth"], dtype=float)
            crater_recovered = np.asarray(data["crater_recovered"], dtype=float)
            crater_maps = {(0, col): crater_recovered[col] for col in range(crater_recovered.shape[0])}
            crater_limit = 100.0 * max(0.01, float(np.nanmax(np.abs(np.concatenate([crater_truth.ravel(), crater_recovered.ravel()])))))
            plot_display_recovery_panel(
                display_dir / f"crater_anomaly_recovery_{period_token}s.png", grid=grid, ray_count_2d=ray_count,
                rows=[("Gaussian crater anomaly", crater_truth)], cols=col_labels, maps=crater_maps, vlim_pct=crater_limit,
                suptitle=f"Crater low-velocity anomaly recovery, T={period_token} s", dpi=args.dpi,
            )
            spike_truth = np.asarray(data["spike_truth"], dtype=float)
            spike_recovered = np.asarray(data["spike_recovered"], dtype=float)
            spike_maps = {(0, col): spike_recovered[col] for col in range(spike_recovered.shape[0])}
            spike_limit = 100.0 * max(0.005, float(np.nanmax(np.abs(np.concatenate([spike_truth.ravel(), spike_recovered.ravel()])))))
            plot_display_recovery_panel(
                display_dir / f"spike_psf_{period_token}s.png", grid=grid, ray_count_2d=ray_count,
                rows=[("single-cell spike", spike_truth)], cols=col_labels, maps=spike_maps, vlim_pct=spike_limit,
                suptitle=f"Point-spread function, T={period_token} s", dpi=args.dpi,
            )
        print(f"[display] period {period_token}s done")

def run(args: argparse.Namespace) -> None:
    periods = tuple(round(float(p), 1) for p in args.periods)
    dampings = list(args.dampings)
    smoothings = list(args.smoothings)
    bbox = AANT.normalized_bbox(FIG9.DEFAULT_BBOX)
    grid = FIG9.build_degree_grid(bbox, args.grid_spacing_deg)
    ensure_dir(args.output_dir)
    figures_dir = args.output_dir / "figures"
    display_figures_dir = args.output_dir / "figures_display_smoothed"
    data_dir = args.output_dir / "data"
    ensure_dir(figures_dir)
    ensure_dir(display_figures_dir)
    ensure_dir(data_dir)

    print(f"[load] {args.measurements_csv}")
    rows_by_period = FIG9.load_corrected_rows(args.measurements_csv, curves_dir=None, periods=periods)

    geometries: Dict[float, Dict[str, object]] = {}
    noise_sd_by_period: Dict[float, float] = {}
    baseline = (float(args.baseline_damping), float(args.baseline_smoothing))

    for period in periods:
        print(f"[geometry] period {period:g} s ...")
        geom = assemble_period_geometry(
            rows_by_period[period], bbox=bbox, grid=grid,
            min_inside_km=args.min_inside_km, sample_step_km=args.sample_step_km,
        )
        geometries[period] = geom
        base_real = invert_linear(
            geom["matrix"], geom["rhs_data"], grid,
            damping=baseline[0], smoothing=baseline[1], robust_iterations=args.robust_iterations,
        )
        noise_sd = float(args.noise_sd) if args.noise_sd else base_real["residual_mad_s"]
        noise_sd_by_period[period] = noise_sd
        print(
            f"  paths={geom['n_paths']}  ref_v={geom['ref_velocity']:.4f} km/s  "
            f"baseline misfit_rms={base_real['misfit_rms_s']:.3f} s  noise_sd={noise_sd:.3f} s"
        )

    # ---------------- parameter scan: real L-curve + checkerboard score ----
    scan: Dict[float, List[Dict[str, object]]] = {p: [] for p in periods}
    scan_checker_size = int(args.scan_checker_size)
    for period in periods:
        geom = geometries[period]
        ray_count_2d = np.asarray(geom["ray_count"], dtype=float).reshape(grid.ny, grid.nx)
        mask = ray_count_2d >= COVERAGE_MIN_RAYS
        truth = checkerboard_model(grid, scan_checker_size, args.checker_amp)
        for damping in dampings:
            for smoothing in smoothings:
                real = invert_linear(
                    geom["matrix"], geom["rhs_data"], grid,
                    damping=damping, smoothing=smoothing, robust_iterations=args.robust_iterations,
                )
                rng = np.random.default_rng(args.seed)
                rec = synthetic_inversion(
                    geom, grid, truth,
                    damping=damping, smoothing=smoothing, km_normalized=False,
                    noise_sd=noise_sd_by_period[period], rng=rng,
                )
                metrics = recovery_metrics(truth, rec, mask)
                scan[period].append({
                    "damping": damping,
                    "smoothing": smoothing,
                    "real_misfit_rms_s": real["misfit_rms_s"],
                    "real_model_norm": real["model_norm"],
                    "checker_correlation": metrics["correlation"],
                    "checker_amplitude_slope": metrics["amplitude_slope"],
                })
                print(
                    f"[scan] T={period:g}s d={damping:g} s={smoothing:g} "
                    f"misfit={real['misfit_rms_s']:.3f} |m|={real['model_norm']:.4f} "
                    f"corr={metrics['correlation']:.3f} slope={metrics['amplitude_slope']:.3f}"
                )

    # global best combo: max mean correlation subject to mean slope >= 0.5
    combo_stats: List[Dict[str, float]] = []
    for damping in dampings:
        for smoothing in smoothings:
            corrs, slopes = [], []
            for period in periods:
                e = next(x for x in scan[period] if x["damping"] == damping and x["smoothing"] == smoothing)
                corrs.append(e["checker_correlation"])
                slopes.append(e["checker_amplitude_slope"])
            combo_stats.append({
                "damping": damping, "smoothing": smoothing,
                "mean_correlation": float(np.mean(corrs)),
                "mean_slope": float(np.mean(slopes)),
            })
    eligible = [c for c in combo_stats if c["mean_slope"] >= 0.5]
    pool = eligible if eligible else combo_stats
    best_stat = max(pool, key=lambda c: c["mean_correlation"])
    best = (best_stat["damping"], best_stat["smoothing"])
    print(f"[scan] best combo (mean corr, slope>=0.5): damping={best[0]:g} smoothing={best[1]:g} "
          f"corr={best_stat['mean_correlation']:.3f} slope={best_stat['mean_slope']:.3f}")

    plot_lcurves(
        figures_dir / "real_data_lcurves.png",
        periods=periods, scan=scan, baseline=baseline, best=best, dpi=args.dpi,
    )
    plot_scan_heatmaps(
        figures_dir / "checkerboard_param_scan.png",
        periods=periods, scan=scan, dampings=dampings, smoothings=smoothings,
        baseline=baseline, best=best, dpi=args.dpi,
    )

    # ---------------- detailed recovery tests at 3 parameter sets ----------
    paramsets = [
        ("baseline", baseline[0], baseline[1], False),
        ("scan_best", best[0], best[1], False),
        ("baseline_km_norm", baseline[0], baseline[1], True),
    ]
    col_labels = [f"{name}\n(d={d:g}, s={s:g}{', km' if km else ''})" for name, d, s, km in paramsets]
    detail_metrics: Dict[str, object] = {}

    for period in periods:
        geom = geometries[period]
        ray_count_2d = np.asarray(geom["ray_count"], dtype=float).reshape(grid.ny, grid.nx)
        mask = ray_count_2d >= COVERAGE_MIN_RAYS
        noise_sd = noise_sd_by_period[period]
        period_metrics: Dict[str, object] = {}

        # checkerboards
        rows_spec = [(f"{size}x{size} cells", checkerboard_model(grid, size, args.checker_amp)) for size in args.checker_sizes]
        maps: Dict[Tuple[int, int], np.ndarray] = {}
        checker_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for irow, (label, truth) in enumerate(rows_spec):
            checker_metrics[label] = {}
            for icol, (name, d, s, km) in enumerate(paramsets):
                rng = np.random.default_rng(args.seed)
                rec = synthetic_inversion(geom, grid, truth, damping=d, smoothing=s, km_normalized=km, noise_sd=noise_sd, rng=rng)
                maps[(irow, icol)] = rec
                checker_metrics[label][name] = recovery_metrics(truth, rec, mask)
        plot_recovery_panel(
            figures_dir / f"checkerboard_recovery_{period:g}s.png",
            grid=grid, ray_count_2d=ray_count_2d, rows=rows_spec, cols=col_labels, maps=maps,
            vlim_pct=100.0 * args.checker_amp,
            suptitle=f"Checkerboard recovery, T={period:g} s (noise sd={noise_sd:.2f} s; contour = {COVERAGE_MIN_RAYS:g} rays)",
            dpi=args.dpi,
        )
        plot_display_recovery_panel(
            display_figures_dir / f"checkerboard_recovery_{period:g}s.png",
            grid=grid, ray_count_2d=ray_count_2d, rows=rows_spec, cols=col_labels, maps=maps,
            vlim_pct=max(6.0, 100.0 * args.checker_amp * 1.2),
            suptitle=f"Checkerboard recovery, T={period:g} s (coverage >= {COVERAGE_MIN_RAYS:g} rays)",
            dpi=args.dpi,
        )
        period_metrics["checkerboard"] = checker_metrics

        # crater gaussian anomaly
        truth_crater = crater_gaussian_model(grid, args.anomaly_amp, args.anomaly_sigma_km)
        maps_crater: Dict[Tuple[int, int], np.ndarray] = {}
        crater_res: Dict[str, Dict[str, float]] = {}
        for icol, (name, d, s, km) in enumerate(paramsets):
            rng = np.random.default_rng(args.seed)
            rec = synthetic_inversion(geom, grid, truth_crater, damping=d, smoothing=s, km_normalized=km, noise_sd=noise_sd, rng=rng)
            maps_crater[(0, icol)] = rec
            crater_res[name] = crater_metrics(truth_crater, rec, mask, grid, args.anomaly_amp)
        plot_recovery_panel(
            figures_dir / f"crater_anomaly_recovery_{period:g}s.png",
            grid=grid, ray_count_2d=ray_count_2d,
            rows=[(f"gaussian sigma={args.anomaly_sigma_km:g} km", truth_crater)],
            cols=col_labels, maps=maps_crater,
            vlim_pct=100.0 * args.anomaly_amp,
            suptitle=f"Crater low-velocity (slow) anomaly recovery, T={period:g} s (noise sd={noise_sd:.2f} s)",
            dpi=args.dpi,
        )
        plot_display_recovery_panel(
            display_figures_dir / f"crater_anomaly_recovery_{period:g}s.png",
            grid=grid, ray_count_2d=ray_count_2d,
            rows=[(f"gaussian sigma={args.anomaly_sigma_km:g} km", truth_crater)],
            cols=col_labels, maps=maps_crater,
            vlim_pct=8.0,
            suptitle=f"Crater low-velocity anomaly recovery, T={period:g} s (coverage >= {COVERAGE_MIN_RAYS:g} rays)",
            dpi=args.dpi,
        )
        period_metrics["crater_anomaly"] = crater_res

        # spike / PSF
        truth_spike, spike_cell = spike_model(grid, args.anomaly_amp)
        maps_spike: Dict[Tuple[int, int], np.ndarray] = {}
        spike_res: Dict[str, Dict[str, float]] = {}
        for icol, (name, d, s, km) in enumerate(paramsets):
            rng = np.random.default_rng(args.seed)
            rec = synthetic_inversion(geom, grid, truth_spike, damping=d, smoothing=s, km_normalized=km, noise_sd=noise_sd, rng=rng)
            maps_spike[(0, icol)] = rec
            spike_res[name] = psf_fwhm_km(rec, spike_cell, grid)
        spike_vlim = max(0.5, 100.0 * max(float(np.nanmax(np.abs(m))) for m in maps_spike.values()))
        plot_recovery_panel(
            figures_dir / f"spike_psf_{period:g}s.png",
            grid=grid, ray_count_2d=ray_count_2d,
            rows=[("single-cell spike", truth_spike)],
            cols=col_labels, maps=maps_spike,
            vlim_pct=spike_vlim,
            suptitle=f"Point-spread function (crater cell spike {100 * args.anomaly_amp:g}%), T={period:g} s; color range +-{spike_vlim:.1f}%",
            dpi=args.dpi,
        )
        plot_display_recovery_panel(
            display_figures_dir / f"spike_psf_{period:g}s.png",
            grid=grid, ray_count_2d=ray_count_2d,
            rows=[("single-cell spike", truth_spike)], cols=col_labels, maps=maps_spike,
            vlim_pct=8.0,
            suptitle=f"Point-spread function, T={period:g} s (coverage >= {COVERAGE_MIN_RAYS:g} rays)",
            dpi=args.dpi,
        )
        period_metrics["spike_psf"] = spike_res

        detail_metrics[f"{period:g}"] = period_metrics
        np.savez_compressed(
            data_dir / f"recovery_grids_{period:g}s.npz",
            ray_count=ray_count_2d,
            lon_edges=grid.lon_edges,
            lat_edges=grid.lat_edges,
            checker_truths=np.stack([t for _, t in rows_spec]),
            checker_recovered=np.stack([np.stack([maps[(i, j)] for j in range(len(paramsets))]) for i in range(len(rows_spec))]),
            crater_truth=truth_crater,
            crater_recovered=np.stack([maps_crater[(0, j)] for j in range(len(paramsets))]),
            spike_truth=truth_spike,
            spike_recovered=np.stack([maps_spike[(0, j)] for j in range(len(paramsets))]),
        )
        print(f"[detail] period {period:g}s done")

    summary = {
        "purpose": "resolution & regularization tests for the Wang Fig.9 phase-velocity inversion (review items 2-4)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "measurements_csv": str(args.measurements_csv),
        "grid": {"spacing_deg": args.grid_spacing_deg, "nx": grid.nx, "ny": grid.ny, "bbox": bbox},
        "parameters": {
            "periods": list(periods),
            "dampings": dampings,
            "smoothings": smoothings,
            "baseline": {"damping": baseline[0], "smoothing": baseline[1]},
            "scan_checker_size_cells": scan_checker_size,
            "checker_sizes_cells": list(args.checker_sizes),
            "checker_amp_frac": args.checker_amp,
            "anomaly_amp_frac": args.anomaly_amp,
            "anomaly_sigma_km": args.anomaly_sigma_km,
            "robust_iterations_real": args.robust_iterations,
            "robust_iterations_synthetic": 1,
            "seed": args.seed,
            "coverage_min_rays": COVERAGE_MIN_RAYS,
        },
        "noise_sd_s": {f"{p:g}": noise_sd_by_period[p] for p in periods},
        "scan_best": {"damping": best[0], "smoothing": best[1],
                      "mean_correlation": best_stat["mean_correlation"], "mean_slope": best_stat["mean_slope"]},
        "combo_stats": combo_stats,
        "scan": {f"{p:g}": scan[p] for p in periods},
        "detail_metrics": detail_metrics,
        "notes": [
            "synthetic inversions use plain LS (gaussian noise); real-data scan uses the production 4-iteration robust loop",
            "metrics are computed on cells with ray_count >= 20 only",
            "baseline_km_norm uses the km-normalized first-difference smoothing operator (isotropic physical smoothing)",
            "maps are raw inversion grids: no gaussian smoothing, no bicubic upsampling",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[done] outputs in {args.output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements-csv", type=Path,
                        default=Path("experiments/wang_ftan_dat_20260724/egf_convention_check/measurements_fig56_egf.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--display-only", action="store_true", help="Render display PNGs from saved NPZ grids only.")
    parser.add_argument("--periods", type=FIG9.parse_periods, default=(3.0, 3.5, 4.0))
    parser.add_argument("--grid-spacing-deg", type=float, default=0.01)
    parser.add_argument("--min-inside-km", type=float, default=0.5)
    parser.add_argument("--sample-step-km", type=float, default=0.25)
    parser.add_argument("--robust-iterations", type=int, default=4)
    parser.add_argument("--dampings", type=parse_float_list, default=(2.0, 4.0, 8.0, 16.0))
    parser.add_argument("--smoothings", type=parse_float_list, default=(5.0, 10.0, 15.0, 22.0, 30.0, 45.0))
    parser.add_argument("--baseline-damping", type=float, default=8.0)
    parser.add_argument("--baseline-smoothing", type=float, default=22.0)
    parser.add_argument("--scan-checker-size", type=int, default=3)
    parser.add_argument("--checker-sizes", type=parse_int_list, default=(2, 3, 4))
    parser.add_argument("--checker-amp", type=float, default=0.05, help="fractional slowness amplitude")
    parser.add_argument("--anomaly-amp", type=float, default=0.08)
    parser.add_argument("--anomaly-sigma-km", type=float, default=1.5)
    parser.add_argument("--noise-sd", type=float, default=None,
                        help="synthetic noise sd in s; default = per-period real baseline residual MAD*1.4826")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.display_only:
        render_display_only(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
