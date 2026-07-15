#!/usr/bin/env python3
"""Reproduce the Figure 7/8 azimuth diagnostics of Wang et al. (2017).

The script intentionally separates the two observations in the paper:

* Figure 7a--c: phase-traveltime residual as a function of interstation
  backazimuth and distance, relative to one reference velocity per period.
* Figure 7d--f: a narrow-band, frequency-domain implementation of trial
  azimuth/slowness beamforming on non-symmetric, spike-removed CCFs.
* Figure 8: 20 degree azimuth-bin means and the even (180 degree symmetric)
  harmonic correction ``a + b cos(2θ) + c sin(2θ) + d cos(4θ) + e sin(4θ)``.

The original paper used phase FTAN and the maximum amplitude of a time-domain
CCF stack.  This project has CDisp picks, so the first item is a faithful
implementation of the published residual/correction calculation on the
available picks.  The beam uses the corresponding Fourier-domain coherence
of shifted CCFs, which tests the same trial delays while remaining tractable
for the full archive.  The output metadata records this distinction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


PAPER_RESIDUAL_LIMIT_S = 0.3
PAPER_BEAM_AMPLITUDE_LIMIT = 2.0
PAPER_DISTANCE_LIMIT_KM = 25.0
PAPER_SLOWNESS_LIMIT_S_PER_KM = 0.6
PAPER_FONT_FAMILY = "Times New Roman"
POLAR_GRID_COLOR = "black"
PAPER_BEAM_COLOR_STOPS = [
    (0.00, "#123c8c"),
    (0.10, "#0067b7"),
    (0.15, "#008c78"),
    (0.27, "#00b547"),
    (0.39, "#b4db26"),
    (0.47, "#fff000"),
    (0.54, "#ffb000"),
    (0.60, "#ff2600"),
    (0.82, "#f00045"),
    (1.00, "#cc329d"),
]


def fit_reference_velocity(distance_km: np.ndarray, travel_time_s: np.ndarray) -> float:
    """Fit a constant velocity by least squares through the origin in slowness."""
    distance_km = np.asarray(distance_km, dtype=float)
    travel_time_s = np.asarray(travel_time_s, dtype=float)
    denominator = float(np.dot(distance_km, distance_km))
    if denominator <= 0.0:
        raise ValueError("Distances must contain a positive value for velocity fitting.")
    slowness_s_per_km = float(np.dot(distance_km, travel_time_s) / denominator)
    if slowness_s_per_km <= 0.0:
        raise ValueError("Fitted slowness is non-positive.")
    return 1.0 / slowness_s_per_km


def even_harmonic_design(azimuth_deg: np.ndarray) -> np.ndarray:
    """Return [1, cos2θ, sin2θ, cos4θ, sin4θ] for azimuths in degrees."""
    theta = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    return np.column_stack(
        (
            np.ones(theta.size),
            np.cos(2.0 * theta),
            np.sin(2.0 * theta),
            np.cos(4.0 * theta),
            np.sin(4.0 * theta),
        )
    )


def fit_even_harmonics(azimuth_deg: np.ndarray, residual_s: np.ndarray) -> np.ndarray:
    """Least-squares fit of the Wang et al. even azimuthal correction."""
    design = even_harmonic_design(azimuth_deg)
    coefficients, _, rank, _ = np.linalg.lstsq(design, np.asarray(residual_s, dtype=float), rcond=None)
    if rank != 5:
        raise ValueError("Azimuth samples do not constrain all five harmonic coefficients.")
    return coefficients


def predict_even_harmonics(azimuth_deg: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate a fitted even harmonic model."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (5,):
        raise ValueError("Expected five harmonic coefficients.")
    return even_harmonic_design(azimuth_deg) @ coefficients


def bin_azimuth_residuals(
    azimuth_deg: np.ndarray, residual_s: np.ndarray, bin_width_deg: float = 20.0
) -> np.ndarray:
    """Return fixed-width circular-bin statistics, including standard error."""
    if not 0.0 < bin_width_deg <= 360.0 or 360.0 % bin_width_deg:
        raise ValueError("bin_width_deg must divide 360 exactly.")
    azimuth = np.mod(np.asarray(azimuth_deg, dtype=float), 360.0)
    residual = np.asarray(residual_s, dtype=float)
    edges = np.arange(0.0, 360.0 + bin_width_deg, bin_width_deg)
    dtype = [("start_deg", float), ("center_deg", float), ("count", int), ("mean_s", float), ("sem_s", float)]
    output = np.empty(len(edges) - 1, dtype=dtype)
    for index, start in enumerate(edges[:-1]):
        end = edges[index + 1]
        include = (azimuth >= start) & (azimuth < end)
        values = residual[include]
        count = values.size
        output[index]["start_deg"] = start
        output[index]["center_deg"] = (start + end) / 2.0
        output[index]["count"] = count
        output[index]["mean_s"] = float(np.mean(values)) if count else np.nan
        output[index]["sem_s"] = float(np.std(values, ddof=1) / math.sqrt(count)) if count > 1 else np.nan
    return output


def beam_phase_delay_s(
    east_km: np.ndarray, north_km: np.ndarray, source_azimuth_deg: float, slowness_s_per_km: float
) -> np.ndarray:
    """Predicted CCF lag for a plane wave from ``source_azimuth_deg``.

    Azimuth is source-facing, clockwise from north.  Propagation is therefore
    opposite to it.  Input offsets point from the first station to the second.
    """
    propagation = math.radians((source_azimuth_deg + 180.0) % 360.0)
    projection_km = np.asarray(east_km) * math.sin(propagation) + np.asarray(north_km) * math.cos(propagation)
    return slowness_s_per_km * projection_km


def _read_stations(path: Path) -> dict[str, tuple[float, float]]:
    stations: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stations[row["station_code"]] = (float(row["latitude"]), float(row["longitude"]))
    if not stations:
        raise ValueError(f"No stations read from {path}")
    return stations


def _local_offsets_km(
    source_lat: np.ndarray, source_lon: np.ndarray, receiver_lat: np.ndarray, receiver_lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Small-aperture east/north offsets; adequate for the ~25 km MSH array."""
    mean_lat_rad = np.deg2rad((source_lat + receiver_lat) / 2.0)
    east = (receiver_lon - source_lon) * 111.320 * np.cos(mean_lat_rad)
    north = (receiver_lat - source_lat) * 110.574
    return east, north


def _azimuth_deg(east_km: np.ndarray, north_km: np.ndarray) -> np.ndarray:
    return np.mod(np.rad2deg(np.arctan2(east_km, north_km)), 360.0)


def polar_azimuth_theta(azimuth_deg: np.ndarray) -> np.ndarray:
    """Convert north-up clockwise azimuth values to polar-axis angles."""
    return np.deg2rad(np.mod(np.asarray(azimuth_deg, dtype=float), 360.0))


def _radial_edges(centers: np.ndarray) -> np.ndarray:
    """Build pcolormesh cell edges from an evenly spaced radial grid."""
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("At least two radial centers are required.")
    half_step = (centers[1] - centers[0]) / 2.0
    return np.concatenate(([max(0.0, centers[0] - half_step)], centers[:-1] + np.diff(centers) / 2.0, [centers[-1] + half_step]))


def normalize_beam_amplitude(beam_coherence: np.ndarray) -> np.ndarray:
    """Scale one beam panel to the 0--2 normalized-amplitude range in Figure 7."""
    beam_coherence = np.asarray(beam_coherence, dtype=float)
    peak = float(np.nanmax(beam_coherence))
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("Beam coherence must contain a finite positive value.")
    return PAPER_BEAM_AMPLITUDE_LIMIT * beam_coherence / peak


def build_slowness_grid(minimum_s_per_km: float, maximum_s_per_km: float, step_s_per_km: float) -> np.ndarray:
    """Build the beam grid, including zero slowness when requested."""
    if minimum_s_per_km < 0.0 or step_s_per_km <= 0.0 or maximum_s_per_km < minimum_s_per_km:
        raise ValueError("Invalid slowness grid")
    return np.arange(minimum_s_per_km, maximum_s_per_km + step_s_per_km / 2.0, step_s_per_km)


def _read_measurements(path: Path, stations: dict[str, tuple[float, float]], periods: Iterable[float]) -> dict[float, dict[str, np.ndarray]]:
    wanted = {float(period) for period in periods}
    values: dict[float, dict[str, list[float] | list[str]]] = {
        period: {"pair_name": [], "distance_km": [], "travel_time_s": [], "azimuth_deg": []} for period in wanted
    }
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            period = float(row["period_s"])
            if period not in wanted:
                continue
            source, receiver = row["pair_name"].split("__", 1)
            if source not in stations or receiver not in stations:
                continue
            source_lat, source_lon = stations[source]
            receiver_lat, receiver_lon = stations[receiver]
            east, north = _local_offsets_km(
                np.array([source_lat]), np.array([source_lon]), np.array([receiver_lat]), np.array([receiver_lon])
            )
            values[period]["pair_name"].append(row["pair_name"])
            values[period]["distance_km"].append(float(row["distance_km"]))
            values[period]["travel_time_s"].append(float(row["corrected_travel_time_s"]))
            values[period]["azimuth_deg"].append(float(_azimuth_deg(east, north)[0]))
    output: dict[float, dict[str, np.ndarray]] = {}
    for period, fields in values.items():
        if not fields["pair_name"]:
            raise ValueError(f"No usable measurements for {period:g} s.")
        output[period] = {key: np.asarray(value) for key, value in fields.items()}
    return output


def _distance_azimuth_grid(
    azimuth_deg: np.ndarray,
    distance_km: np.ndarray,
    residual_s: np.ndarray,
    distance_bin_km: float = 1.0,
    max_distance_km: float = PAPER_DISTANCE_LIMIT_KM,
    azimuth_bin_deg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_distance_km <= 0.0 or azimuth_bin_deg <= 0.0 or 360.0 % azimuth_bin_deg:
        raise ValueError("Invalid Figure 7 polar grid.")
    include = np.asarray(distance_km) <= max_distance_km
    az_edges = np.arange(0.0, 360.0 + azimuth_bin_deg, azimuth_bin_deg)
    distance_edges = np.arange(0.0, max_distance_km + distance_bin_km, distance_bin_km)
    sums, _, _ = np.histogram2d(distance_km[include], np.mod(azimuth_deg[include], 360.0), bins=(distance_edges, az_edges), weights=residual_s[include])
    counts, _, _ = np.histogram2d(distance_km[include], np.mod(azimuth_deg[include], 360.0), bins=(distance_edges, az_edges))
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return az_edges, distance_edges, means


def _ccf_paths(stack_root: Path, stations: dict[str, tuple[float, float]], sample_size: int, seed: int) -> list[Path]:
    candidates = []
    for path in stack_root.glob("*/*/stack_pws.h5"):
        source, receiver = path.parts[-3], path.parts[-2]
        if source in stations and receiver in stations:
            candidates.append(path)
    if not candidates:
        raise ValueError(f"No readable CCF files found below {stack_root}")
    candidates.sort()
    if sample_size <= 0 or sample_size >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(candidates), size=sample_size, replace=False))
    return [candidates[index] for index in indices]


def _load_ccf_spectra(paths: list[Path], stations: dict[str, tuple[float, float]], periods: list[float]) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray], float]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised on the work server
        raise RuntimeError("h5py is required for CCF beamforming.") from exc

    east_values: list[float] = []
    north_values: list[float] = []
    spectra: dict[float, list[complex]] = {period: [] for period in periods}
    dt: float | None = None
    for index, path in enumerate(paths, start=1):
        source, receiver = path.parts[-3], path.parts[-2]
        source_lat, source_lon = stations[source]
        receiver_lat, receiver_lon = stations[receiver]
        with h5py.File(path, "r") as handle:
            trace = np.asarray(handle["AuxiliaryData/Allstack_pws/ZZ"], dtype=float)
            trace_dt = float(handle["AuxiliaryData/Allstack_pws/ZZ"].attrs["dt"])
        if dt is None:
            dt = trace_dt
            frequencies = np.fft.rfftfreq(trace.size, d=dt)
            taper = np.hanning(trace.size)
            target_indices = {period: int(np.argmin(np.abs(frequencies - 1.0 / period))) for period in periods}
        elif not math.isclose(dt, trace_dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Mixed CCF sampling intervals: {dt} and {trace_dt}")
        transformed = np.fft.rfft(trace * taper)
        for period in periods:
            coefficient = transformed[target_indices[period]]
            spectra[period].append(coefficient / abs(coefficient) if abs(coefficient) else 0.0j)
        east, north = _local_offsets_km(
            np.array([source_lat]), np.array([source_lon]), np.array([receiver_lat]), np.array([receiver_lon])
        )
        east_values.append(float(east[0]))
        north_values.append(float(north[0]))
        if index % 2000 == 0:
            print(f"  loaded {index}/{len(paths)} CCFs", file=sys.stderr, flush=True)
    if dt is None:
        raise ValueError("No CCFs were loaded.")
    return np.asarray(east_values), np.asarray(north_values), {key: np.asarray(value) for key, value in spectra.items()}, dt


def _beamform(
    east_km: np.ndarray,
    north_km: np.ndarray,
    spectrum: np.ndarray,
    period_s: float,
    azimuths_deg: np.ndarray,
    slownesses_s_per_km: np.ndarray,
) -> np.ndarray:
    """Narrow-band coherence after the paper's trial time shifts."""
    frequency_hz = 1.0 / period_s
    output = np.empty((slownesses_s_per_km.size, azimuths_deg.size), dtype=float)
    for azimuth_index, azimuth in enumerate(azimuths_deg):
        delay_s = beam_phase_delay_s(east_km, north_km, float(azimuth), 1.0)
        phase = 2.0j * np.pi * frequency_hz * np.outer(slownesses_s_per_km, delay_s)
        output[:, azimuth_index] = np.abs(np.exp(phase) @ spectrum) / spectrum.size
    return output


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_matplotlib(font_file: Path | None) -> None:
    import matplotlib
    from matplotlib import font_manager

    if font_file is not None:
        font_manager.fontManager.addfont(str(font_file))
    matplotlib.rcParams.update({"font.family": PAPER_FONT_FAMILY, "font.weight": "normal", "axes.titleweight": "normal"})


def _style_polar_axis(axis: object, angle_ticks: list[int], angle_labels: list[str], radial_ticks: list[float], radial_label_angle: float) -> None:
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_thetagrids(angle_ticks, labels=angle_labels)
    axis.set_rticks(radial_ticks)
    axis.set_rlabel_position(radial_label_angle)
    axis.xaxis.grid(True, color=POLAR_GRID_COLOR, linestyle="-", linewidth=0.65, alpha=1.0, zorder=10)
    axis.yaxis.grid(True, color=POLAR_GRID_COLOR, linestyle="-", linewidth=0.65, alpha=1.0, zorder=10)
    axis.spines["polar"].set_color(POLAR_GRID_COLOR)
    axis.spines["polar"].set_linewidth(0.75)


def _plot_figure7(
    output_path: Path,
    residual_results: dict[float, dict[str, object]],
    beams: dict[float, np.ndarray],
    azimuths_deg: np.ndarray,
    slownesses_s_per_km: np.ndarray,
    font_file: Path | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    _configure_matplotlib(font_file)
    residual_periods = sorted(residual_results)
    beam_periods = sorted(beams)
    if len(residual_periods) != 3 or len(beam_periods) != 3:
        raise ValueError("Figure 7 requires three residual panels and three beam panels.")
    residual_cmap = LinearSegmentedColormap.from_list(
        "wang_residual",
        ["#762a9f", "#2864c2", "#00d7e9", "#f7f7f7", "#fff200", "#e60000", "#000000"],
        N=256,
    )
    residual_cmap.set_bad("#a6a6a6")
    beam_cmap = LinearSegmentedColormap.from_list("wang_beam", PAPER_BEAM_COLOR_STOPS, N=256)
    angle_ticks = list(range(0, 360, 30))
    angle_labels = [str(angle) if angle <= 180 else str(angle - 360) for angle in angle_ticks]
    figure, axes = plt.subplots(
        2, 3, figsize=(13, 8.5), constrained_layout=True, subplot_kw={"projection": "polar"}
    )
    residual_images = []
    for column, period in enumerate(residual_periods):
        result = residual_results[period]
        az_edges, distance_edges, means = _distance_azimuth_grid(result["azimuth_deg"], result["distance_km"], result["residual_s"])
        image = axes[0, column].pcolormesh(
            polar_azimuth_theta(az_edges), distance_edges, means, cmap=residual_cmap,
            norm=Normalize(-PAPER_RESIDUAL_LIMIT_S, PAPER_RESIDUAL_LIMIT_S), shading="auto"
        )
        axis = axes[0, column]
        axis.set_facecolor("#a6a6a6")
        axis.set(title=f"({chr(ord('a') + column)})     {period:g} s", ylim=(0, PAPER_DISTANCE_LIMIT_KM))
        _style_polar_axis(axis, angle_ticks, angle_labels, [5, 10, 15, 20, 25], radial_label_angle=45)
        residual_images.append(image)

    beam_images = []
    for column, period in enumerate(beam_periods):
        beam_amplitude = normalize_beam_amplitude(beams[period])
        azimuth_step = float(azimuths_deg[1] - azimuths_deg[0])
        beam_azimuth_edges = np.arange(-azimuth_step / 2.0, 360.0 + azimuth_step / 2.0, azimuth_step)
        axis = axes[1, column]
        beam_image = axis.pcolormesh(
            polar_azimuth_theta(beam_azimuth_edges), _radial_edges(slownesses_s_per_km), beam_amplitude,
            cmap=beam_cmap, vmin=0.0, vmax=PAPER_BEAM_AMPLITUDE_LIMIT, shading="auto"
        )
        axis.set(title=f"({chr(ord('d') + column)})     {period:g} s", ylim=(0, PAPER_SLOWNESS_LIMIT_S_PER_KM))
        _style_polar_axis(axis, angle_ticks, angle_labels, [0.2, 0.4, 0.6], radial_label_angle=45)
        beam_images.append(beam_image)

    figure.colorbar(
        residual_images[-1], ax=axes[0, :], label="Average residual (s)", ticks=np.arange(-0.3, 0.31, 0.1), shrink=0.86
    )
    figure.colorbar(
        beam_images[-1], ax=axes[1, :], label="Normalized amplitude", ticks=[0, 0.5, 1.0, 1.5, PAPER_BEAM_AMPLITUDE_LIMIT], shrink=0.86
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_figure8(output_path: Path, residual_results: dict[float, dict[str, object]], font_file: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib(font_file)
    periods = sorted(residual_results)
    figure, axes = plt.subplots(1, len(periods), figsize=(15, 4.5), sharey=True, constrained_layout=True)
    for axis, period in zip(axes, periods, strict=True):
        result = residual_results[period]
        bins = result["bins"]
        valid = bins["count"] > 0
        error = np.nan_to_num(bins["sem_s"], nan=0.0) * 5.0
        axis.errorbar(
            bins["center_deg"][valid], bins["mean_s"][valid], yerr=error[valid], fmt="o", color="black", ms=4,
            capsize=2, label="20° mean; bar = 5 × SEM",
        )
        azimuth = np.linspace(0.0, 360.0, 721)
        axis.plot(azimuth, predict_even_harmonics(azimuth, result["coefficients"]), color="#d62728", lw=2, label="LS 2θ + 4θ")
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set(title=f"{period:g} s", xlabel="Backazimuth (deg)", xlim=(0, 360))
        axis.set_xticks([0, 90, 180, 270, 360])
    axes[0].set_ylabel("Traveltime residual (s)")
    axes[-1].legend(loc="best", fontsize=8)
    figure.suptitle("Wang et al. (2017) Figure 8 style azimuthal correction")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _write_html(output_path: Path, metadata: dict[str, object]) -> None:
    residual_rows = "".join(
        f"<tr><td>{float(period):g} s</td><td>{values['n_measurements']:,}</td><td>{values['reference_velocity_km_s']:.4f}</td>"
        f"<td>{values['trend_peak_to_peak_s']:.4f}</td><td>{values['residual_std_before_s']:.4f}</td>"
        f"<td>{values['residual_std_after_s']:.4f}</td></tr>"
        for period, values in sorted(metadata["residual_period_results"].items(), key=lambda item: float(item[0]))
    )
    beam_rows = "".join(
        f"<tr><td>{float(period):g} s</td><td>{values['beam_peak_azimuth_deg']:.0f}</td>"
        f"<td>{values['beam_peak_slowness_s_per_km']:.3f}</td><td>{values['beam_peak_coherence']:.4f}</td></tr>"
        for period, values in sorted(metadata["beam_period_results"].items(), key=lambda item: float(item[0]))
    )
    output_path.write_text(
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Wang Fig. 7/8 reproduction</title>"
        "<style>body{font:16px system-ui;margin:2rem;max-width:1200px}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.45rem;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;border:1px solid #ddd}</style>"
        "<h1>Wang et al. (2017) 图7、图8数值复现</h1>"
        f"<p>图7上排为 3、3.5、4 s 相位走时残差的方位—距离分布；下排为 2、3、4 s 去尖峰、非对称 CCF 的窄带频域波束。"
        f"图7显示范围：残差 -0.3 至 +0.3 s，归一化振幅 0 至 {PAPER_BEAM_AMPLITUDE_LIMIT:g}，距离半径 0 至 25 km，慢度半径 0 至 0.6 s/km；网格为黑色实线。"
        "图8按论文以 20° 分箱均值拟合 <i>a+b cos(2θ)+c sin(2θ)+d cos(4θ)+e sin(4θ)</i>；误差条为 5×均值标准误。"
        "相速度拾取来自本项目 CDisp 结果，而非论文的 FTAN；波束是时域移时叠加的频域窄带等价诊断，故用于检验方位性、不能声称逐像素复刻原图。</p>"
        "<h2>残差统计（图7上排、图8）</h2><table><tr><th>Period</th><th>N</th><th>Vref (km/s)</th><th>Trend p-p (s)</th><th>Std before (s)</th><th>Std after (s)</th></tr>"
        + residual_rows
        + "</table><h2>波束峰值（图7下排）</h2><table><tr><th>Period</th><th>Peak azimuth</th><th>Peak slowness</th><th>Raw coherence</th></tr>"
        + beam_rows
        + "</table><h2>图7</h2><img src='figure7_wang_style.png'><h2>图8</h2><img src='figure8_wang_style.png'>"
        "<h2>可复查产物</h2><ul><li><code>measurements_azimuth_corrected.csv</code>：每条走时的残差、校正项及校正后走时。</li>"
        "<li><code>azimuth_bin_statistics.csv</code>：图8 20° 分箱。</li><li><code>metadata.json</code>：所有输入、网格与峰值。</li></ul></html>",
        encoding="utf-8",
    )


def run(arguments: argparse.Namespace) -> dict[str, object]:
    measurements_path = Path(arguments.measurements_csv).resolve()
    stations_path = Path(arguments.stations_csv).resolve()
    stack_root = Path(arguments.stack_root).resolve()
    output_dir = Path(arguments.output_dir).resolve()
    font_file = Path(arguments.font_file).resolve() if arguments.font_file else None
    for input_path in (measurements_path, stations_path, stack_root):
        if not input_path.exists():
            raise FileNotFoundError(input_path)
    if font_file is not None and not font_file.is_file():
        raise FileNotFoundError(font_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    residual_periods = sorted(float(period) for period in arguments.periods)
    beam_periods = sorted(float(period) for period in arguments.beam_periods)
    stations = _read_stations(stations_path)
    measurements = _read_measurements(measurements_path, stations, residual_periods)
    residual_results: dict[float, dict[str, object]] = {}
    correction_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    for period in residual_periods:
        data = measurements[period]
        reference_velocity = fit_reference_velocity(data["distance_km"], data["travel_time_s"])
        reference_time = data["distance_km"] / reference_velocity
        residual = data["travel_time_s"] - reference_time
        bins = bin_azimuth_residuals(data["azimuth_deg"], residual)
        constrained = bins["count"] > 0
        coefficients = fit_even_harmonics(bins["center_deg"][constrained], bins["mean_s"][constrained])
        correction = predict_even_harmonics(data["azimuth_deg"], coefficients)
        corrected_residual = residual - correction
        trend = predict_even_harmonics(np.linspace(0.0, 360.0, 3601), coefficients)
        residual_results[period] = {
            **data,
            "reference_velocity_km_s": reference_velocity,
            "reference_time_s": reference_time,
            "residual_s": residual,
            "bins": bins,
            "coefficients": coefficients,
            "correction_s": correction,
            "corrected_residual_s": corrected_residual,
            "trend_peak_to_peak_s": float(np.ptp(trend)),
        }
        for index in range(data["pair_name"].size):
            correction_rows.append(
                {
                    "pair_name": data["pair_name"][index], "period_s": period, "distance_km": data["distance_km"][index],
                    "backazimuth_deg": data["azimuth_deg"][index], "reference_velocity_km_s": reference_velocity,
                    "observed_travel_time_s": data["travel_time_s"][index], "reference_travel_time_s": reference_time[index],
                    "residual_s": residual[index], "azimuth_correction_s": correction[index],
                    "corrected_travel_time_s": data["travel_time_s"][index] - correction[index],
                    "residual_after_correction_s": corrected_residual[index],
                }
            )
        for bin_result in bins:
            bin_rows.append({"period_s": period, **{field: bin_result[field].item() for field in bins.dtype.names}})
    paths = _ccf_paths(stack_root, stations, arguments.beam_sample_size, arguments.seed)
    print(f"Loading {len(paths):,} spike-removed CCFs for beamforming...", file=sys.stderr, flush=True)
    east, north, spectra, _ = _load_ccf_spectra(paths, stations, beam_periods)
    azimuths = np.arange(0.0, 360.0, arguments.beam_azimuth_step_deg)
    slownesses = build_slowness_grid(arguments.beam_slowness_min, arguments.beam_slowness_max, arguments.beam_slowness_step)
    beams = {period: _beamform(east, north, spectra[period], period, azimuths, slownesses) for period in beam_periods}
    residual_period_metadata: dict[str, dict[str, object]] = {}
    for period in residual_periods:
        result = residual_results[period]
        coefficients = result["coefficients"]
        residual_period_metadata[f"{period:g}"] = {
            "n_measurements": int(result["pair_name"].size),
            "reference_velocity_km_s": float(result["reference_velocity_km_s"]),
            "harmonic_coefficients_s": {name: float(value) for name, value in zip(("a", "b_cos2", "c_sin2", "d_cos4", "e_sin4"), coefficients, strict=True)},
            "trend_peak_to_peak_s": float(result["trend_peak_to_peak_s"]),
            "residual_std_before_s": float(np.std(result["residual_s"])),
            "residual_std_after_s": float(np.std(result["corrected_residual_s"])),
        }
    beam_period_metadata: dict[str, dict[str, object]] = {}
    for period in beam_periods:
        beam = beams[period]
        row, column = np.unravel_index(np.argmax(beam), beam.shape)
        beam_period_metadata[f"{period:g}"] = {
            "beam_peak_azimuth_deg": float(azimuths[column]),
            "beam_peak_slowness_s_per_km": float(slownesses[row]),
            "beam_peak_coherence": float(beam[row, column]),
        }
    metadata: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "residuals": "CDisp corrected phase traveltimes, through-origin constant-slowness reference fit",
            "azimuth_correction": "20 degree bin means; unweighted least-squares a+b*cos(2theta)+c*sin(2theta)+d*cos(4theta)+e*sin(4theta)",
            "beam": "non-symmetric spike-removed CCFs; Hann-tapered narrow-band Fourier phase coherence after trial plane-wave time shifts, including zero slowness",
            "limitation": "Wang et al. (2017) used phase FTAN and the maximum amplitude of a time-domain shifted stack; this is a reproducible numerical analogue using available CDisp picks.",
        },
        "inputs": {"measurements_csv": str(measurements_path), "stations_csv": str(stations_path), "stack_root": str(stack_root)},
        "beam_grid": {"n_ccfs": len(paths), "sampling": "seeded uniform sample of available CCF files; all files if sample size is 0 or exceeds archive size", "seed": arguments.seed, "azimuth_step_deg": arguments.beam_azimuth_step_deg, "slowness_min_s_per_km": arguments.beam_slowness_min, "slowness_max_s_per_km": arguments.beam_slowness_max, "slowness_step_s_per_km": arguments.beam_slowness_step},
        "figure_7_display": {"residual_periods_s": residual_periods, "beam_periods_s": beam_periods, "residual_limits_s": [-PAPER_RESIDUAL_LIMIT_S, PAPER_RESIDUAL_LIMIT_S], "beam_amplitude_limits": [0.0, PAPER_BEAM_AMPLITUDE_LIMIT], "distance_limits_km": [0.0, PAPER_DISTANCE_LIMIT_KM], "slowness_limits_s_per_km": [0.0, PAPER_SLOWNESS_LIMIT_S_PER_KM], "font_family": PAPER_FONT_FAMILY, "font_file": str(font_file) if font_file else None, "polar_grid": "black solid", "beam_normalization": f"each period's raw coherence is divided by its own maximum and multiplied by {PAPER_BEAM_AMPLITUDE_LIMIT:g} for display", "beam_colormap": "short blue low-value interval, expanded red high-value interval modeled on Wang et al. Figure 7"},
        "residual_period_results": residual_period_metadata,
        "beam_period_results": beam_period_metadata,
    }
    _write_csv(output_dir / "measurements_azimuth_corrected.csv", correction_rows, list(correction_rows[0]))
    _write_csv(output_dir / "azimuth_bin_statistics.csv", bin_rows, list(bin_rows[0]))
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    _plot_figure7(output_dir / "figure7_wang_style.png", residual_results, beams, azimuths, slownesses, font_file)
    _plot_figure8(output_dir / "figure8_wang_style.png", residual_results, font_file)
    _write_html(output_dir / "report.html", metadata)
    return metadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements-csv", required=True)
    parser.add_argument("--stations-csv", required=True)
    parser.add_argument("--stack-root", required=True, help="Root containing <source>/<receiver>/stack_pws.h5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--font-file", help="Optional .ttf/.otf font file; required on hosts without Times New Roman")
    parser.add_argument("--periods", nargs="+", type=float, default=[3.0, 3.5, 4.0], help="Residual periods for Figure 7a-c and Figure 8")
    parser.add_argument("--beam-periods", nargs="+", type=float, default=[2.0, 3.0, 4.0], help="CCF beam periods for Figure 7d-f")
    parser.add_argument("--beam-sample-size", type=int, default=20000, help="0 means all available CCFs")
    parser.add_argument("--seed", type=int, default=20160708)
    parser.add_argument("--beam-azimuth-step-deg", type=float, default=1.0)
    parser.add_argument("--beam-slowness-min", type=float, default=0.0)
    parser.add_argument("--beam-slowness-max", type=float, default=0.60)
    parser.add_argument("--beam-slowness-step", type=float, default=0.005)
    arguments = parser.parse_args()
    if arguments.beam_sample_size < 0:
        parser.error("--beam-sample-size must be non-negative")
    if arguments.beam_azimuth_step_deg <= 0 or 360.0 % arguments.beam_azimuth_step_deg:
        parser.error("--beam-azimuth-step-deg must be a positive divisor of 360")
    if arguments.beam_slowness_min < 0 or arguments.beam_slowness_step <= 0 or arguments.beam_slowness_max < arguments.beam_slowness_min:
        parser.error("Invalid slowness grid")
    return arguments


def main() -> int:
    try:
        metadata = run(parse_arguments())
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"residual_period_results": metadata["residual_period_results"], "beam_period_results": metadata["beam_period_results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
