#!/usr/bin/env python3
"""Plot Wang-style Figure 4 from DisperPicker phase-velocity curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_PERIODS_S = (3.0, 4.0, 5.0)
WANG_REFERENCE_VELOCITIES = {3.0: 2.70, 4.0: 2.90, 5.0: 3.05}


@dataclass(frozen=True)
class Measurement:
    pair_name: str
    distance_km: float
    period_s: float
    phase_velocity_km_s: float
    travel_time_s: float
    snr: float
    confidence: float
    group_velocity_km_s: float = float("nan")


@dataclass(frozen=True)
class CorrectedMeasurement:
    pair_name: str
    distance_km: float
    period_s: float
    raw_travel_time_s: float
    corrected_travel_time_s: float
    branch_n: int
    residual_s: float
    phase_velocity_km_s: float
    corrected_phase_velocity_km_s: float
    snr: float
    confidence: float
    group_velocity_km_s: float
    distance_over_lambda: float
    reference_velocity_km_s: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_periods(value: str) -> Tuple[float, ...]:
    periods = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not periods:
        raise ValueError("At least one target period is required")
    return periods


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * 6371.0088 * math.asin(min(1.0, math.sqrt(a)))


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def pair_name_from_cdisp(path: Path) -> str:
    name = path.name
    if name.startswith("CDisp.") and name.endswith(".txt"):
        return name[len("CDisp.") : -len(".txt")]
    return path.stem


def read_cdisp_measurements(path: Path, target_periods: Set[float]) -> List[Measurement]:
    pair_name = pair_name_from_cdisp(path)
    rows: List[Measurement] = []
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
        second = handle.readline().split()
        if len(first) < 2 or len(second) < 2:
            return rows
        lon_a, lat_a = float(first[0]), float(first[1])
        lon_b, lat_b = float(second[0]), float(second[1])
        distance = great_circle_km(lat_a, lon_a, lat_b, lon_b)
        if not math.isfinite(distance) or distance <= 0:
            return rows
        target_by_centiperiod = {int(round(period * 100.0)): float(period) for period in target_periods}
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            period = finite_float(parts[0])
            key = int(round(period * 100.0))
            if key not in target_by_centiperiod:
                continue
            velocity = finite_float(parts[1])
            snr = finite_float(parts[2])
            confidence = finite_float(parts[3])
            if not (math.isfinite(velocity) and velocity > 0):
                continue
            rows.append(
                Measurement(
                    pair_name=pair_name,
                    distance_km=distance,
                    period_s=target_by_centiperiod[key],
                    phase_velocity_km_s=velocity,
                    travel_time_s=distance / velocity,
                    snr=snr,
                    confidence=confidence,
                )
            )
    return rows


def read_gdisp_by_period(path: Path, target_periods: Set[float]) -> Dict[int, Tuple[float, float, float]]:
    out: Dict[int, Tuple[float, float, float]] = {}
    if not path.exists():
        return out
    target_by_centiperiod = {int(round(period * 100.0)) for period in target_periods}
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        next(handle, None)
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            period = finite_float(parts[0])
            key = int(round(period * 100.0))
            if key not in target_by_centiperiod:
                continue
            out[key] = (finite_float(parts[1]), finite_float(parts[2]), finite_float(parts[3]))
    return out


def read_pair_measurements(c_path: Path, g_path: Path, target_periods: Set[float]) -> List[Measurement]:
    group_by_period = read_gdisp_by_period(g_path, target_periods)
    rows = []
    for row in read_cdisp_measurements(c_path, target_periods):
        key = int(round(row.period_s * 100.0))
        group_velocity, group_snr, _ = group_by_period.get(key, (float("nan"), float("nan"), float("nan")))
        snr = min(row.snr, group_snr) if math.isfinite(group_snr) else row.snr
        rows.append(
            Measurement(
                pair_name=row.pair_name,
                distance_km=row.distance_km,
                period_s=row.period_s,
                phase_velocity_km_s=row.phase_velocity_km_s,
                travel_time_s=row.travel_time_s,
                snr=snr,
                confidence=row.confidence,
                group_velocity_km_s=group_velocity,
            )
        )
    return rows


def _read_worker(args: Tuple[str, str, Tuple[float, ...]]) -> List[Measurement]:
    c_path_text, g_dir_text, periods = args
    c_path = Path(c_path_text)
    g_path = Path(g_dir_text) / ("GDisp." + pair_name_from_cdisp(c_path) + ".txt")
    return read_pair_measurements(c_path, g_path, set(periods))


def load_measurements(
    curves_dir: Path,
    target_periods: Tuple[float, ...],
    workers: int,
    group_curves_dir: Optional[Path] = None,
) -> List[Measurement]:
    paths = sorted(curves_dir.glob("CDisp.*.txt"))
    group_curves_dir = group_curves_dir or curves_dir
    if workers <= 1:
        out: List[Measurement] = []
        for path in paths:
            g_path = group_curves_dir / ("GDisp." + pair_name_from_cdisp(path) + ".txt")
            out.extend(read_pair_measurements(path, g_path, set(target_periods)))
        return out
    tasks = [(str(path), str(group_curves_dir), target_periods) for path in paths]
    out = []
    with Pool(processes=workers) as pool:
        for rows in pool.imap_unordered(_read_worker, tasks, chunksize=200):
            out.extend(rows)
    return out


def load_measurements_csv(path: Path, target_periods: Tuple[float, ...]) -> List[Measurement]:
    target_by_centiperiod = {int(round(period * 100.0)): float(period) for period in target_periods}
    rows: List[Measurement] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            period = finite_float(row.get("period_s"))
            key = int(round(period * 100.0))
            if key not in target_by_centiperiod:
                continue
            distance = finite_float(row.get("distance_km"))
            velocity = finite_float(row.get("phase_velocity_km_s"))
            travel_time = finite_float(row.get("travel_time_s"))
            if not math.isfinite(travel_time) and distance > 0 and velocity > 0:
                travel_time = distance / velocity
            if not (distance > 0 and velocity > 0 and travel_time > 0):
                continue
            rows.append(
                Measurement(
                    pair_name=str(row.get("pair_name", "")),
                    distance_km=distance,
                    period_s=target_by_centiperiod[key],
                    phase_velocity_km_s=velocity,
                    travel_time_s=travel_time,
                    snr=finite_float(row.get("snr")),
                    confidence=finite_float(row.get("confidence")),
                    group_velocity_km_s=finite_float(row.get("group_velocity_km_s")),
                )
            )
    return rows


def group_velocity_limit(period_s: float) -> float:
    return 3.0 if float(period_s) < 4.5 else 3.3


def screen_measurements(
    rows: Iterable[Measurement],
    *,
    min_snr: float,
    group_vmin: float,
    min_wavelengths: float = 0.0,
) -> List[Measurement]:
    screened = []
    for row in rows:
        if not (row.snr >= min_snr):
            continue
        group_velocity = row.group_velocity_km_s
        if not (math.isfinite(group_velocity) and group_vmin <= group_velocity <= group_velocity_limit(row.period_s)):
            continue
        if min_wavelengths > 0 and row.travel_time_s < min_wavelengths * row.period_s:
            continue
        screened.append(row)
    return screened


def reference_wavelength_filter(
    rows: Iterable[Measurement],
    *,
    period_s: float,
    min_wavelengths: float,
    reference_velocity_km_s: float,
) -> Tuple[List[Measurement], float]:
    buffered = list(rows)
    if min_wavelengths <= 0:
        return buffered, 0.0
    if not (math.isfinite(reference_velocity_km_s) and reference_velocity_km_s > 0):
        return [], float("nan")
    distance_cutoff_km = float(min_wavelengths) * float(reference_velocity_km_s) * float(period_s)
    return [row for row in buffered if row.distance_km >= distance_cutoff_km], distance_cutoff_km


def fit_velocity_through_origin(distance_km: Sequence[float], travel_time_s: Sequence[float]) -> float:
    distance = np.asarray(distance_km, dtype=float)
    time = np.asarray(travel_time_s, dtype=float)
    valid = np.isfinite(distance) & np.isfinite(time) & (distance > 0) & (time > 0)
    if np.count_nonzero(valid) == 0:
        return float("nan")
    slope = float(np.dot(distance[valid], time[valid]) / np.dot(distance[valid], distance[valid]))
    return 1.0 / slope if math.isfinite(slope) and slope > 0 else float("nan")


def correct_period_branches(
    rows: Iterable[Measurement],
    *,
    reference_velocity_km_s: float,
) -> List[CorrectedMeasurement]:
    corrected: List[CorrectedMeasurement] = []
    if not (math.isfinite(reference_velocity_km_s) and reference_velocity_km_s > 0):
        return corrected
    for row in rows:
        predicted_time = row.distance_km / reference_velocity_km_s
        branch_n = int(round((predicted_time - row.travel_time_s) / row.period_s))
        corrected_time = row.travel_time_s + branch_n * row.period_s
        if not (math.isfinite(corrected_time) and corrected_time > 0):
            continue
        corrected_velocity = row.distance_km / corrected_time
        if not (math.isfinite(corrected_velocity) and corrected_velocity > 0):
            continue
        corrected.append(
            CorrectedMeasurement(
                pair_name=row.pair_name,
                distance_km=row.distance_km,
                period_s=row.period_s,
                raw_travel_time_s=row.travel_time_s,
                corrected_travel_time_s=corrected_time,
                branch_n=branch_n,
                residual_s=corrected_time - predicted_time,
                phase_velocity_km_s=row.phase_velocity_km_s,
                corrected_phase_velocity_km_s=corrected_velocity,
                snr=row.snr,
                confidence=row.confidence,
                group_velocity_km_s=row.group_velocity_km_s,
                distance_over_lambda=row.travel_time_s / row.period_s,
                reference_velocity_km_s=reference_velocity_km_s,
            )
        )
    return corrected


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_dict(row: Measurement) -> Dict[str, object]:
    return asdict(row)


def corrected_row_dict(row: CorrectedMeasurement) -> Dict[str, object]:
    return asdict(row)


def summarize_period(period: float, initial: List[Measurement], screened: List[Measurement]) -> Dict[str, object]:
    fit_velocity = fit_velocity_through_origin(
        [row.distance_km for row in screened],
        [row.travel_time_s for row in screened],
    )
    velocities = np.asarray([row.phase_velocity_km_s for row in screened], dtype=float)
    return {
        "period_s": float(period),
        "initial_count": len(initial),
        "screened_count": len(screened),
        "fit_velocity_km_s": fit_velocity,
        "std_velocity_km_s": float(np.std(velocities)) if velocities.size else float("nan"),
        "median_velocity_km_s": float(np.median(velocities)) if velocities.size else float("nan"),
        "wang_velocity_km_s": WANG_REFERENCE_VELOCITIES.get(float(period), float("nan")),
    }


def summarize_paper_period(
    period: float,
    input_rows: List[Measurement],
    left_rows: List[Measurement],
    right_rows: List[CorrectedMeasurement],
    reference_velocity_km_s: float,
) -> Dict[str, object]:
    fit_velocity = fit_velocity_through_origin(
        [row.distance_km for row in right_rows],
        [row.corrected_travel_time_s for row in right_rows],
    )
    velocities = np.asarray([row.corrected_phase_velocity_km_s for row in right_rows], dtype=float)
    left_fit_velocity = fit_velocity_through_origin(
        [row.distance_km for row in left_rows],
        [row.travel_time_s for row in left_rows],
    )
    return {
        "period_s": float(period),
        "input_count": len(input_rows),
        "left_screened_count": len(left_rows),
        "corrected_count": len(right_rows),
        "reference_velocity_km_s": reference_velocity_km_s,
        "left_fit_velocity_km_s": left_fit_velocity,
        "fit_velocity_km_s": fit_velocity,
        "std_velocity_km_s": float(np.std(velocities)) if velocities.size else float("nan"),
        "median_velocity_km_s": float(np.median(velocities)) if velocities.size else float("nan"),
        "wang_velocity_km_s": WANG_REFERENCE_VELOCITIES.get(float(period), float("nan")),
    }


def plot_figure(
    path: Path,
    per_period: Dict[float, Dict[str, object]],
    *,
    target_periods: Tuple[float, ...],
    max_distance_km: float,
    max_time_s: float,
) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(len(target_periods), 2, figsize=(10.2, 4.75 * len(target_periods)), sharex=True, sharey=True)
    if len(target_periods) == 1:
        axes = np.asarray([axes])
    fig.patch.set_facecolor("white")
    xline = np.linspace(0.0, max_distance_km, 300)
    for row_index, period in enumerate(target_periods):
        payload = per_period[float(period)]
        initial = payload["initial"]
        screened = payload["screened"]
        summary = payload["summary"]
        wang_v = WANG_REFERENCE_VELOCITIES.get(float(period), float("nan"))
        fit_v = float(summary["fit_velocity_km_s"])
        std_v = float(summary["std_velocity_km_s"])

        ax_left = axes[row_index, 0]
        ax_right = axes[row_index, 1]
        if initial:
            ax_left.scatter(
                [row.distance_km for row in initial],
                [row.travel_time_s for row in initial],
                s=3,
                color="#0b51ff",
                alpha=0.18,
                linewidths=0,
            )
        if screened:
            ax_right.scatter(
                [row.distance_km for row in screened],
                [row.travel_time_s for row in screened],
                s=4,
                color="#0b51ff",
                alpha=0.55,
                linewidths=0,
            )

        if math.isfinite(wang_v) and wang_v > 0:
            center = xline / wang_v
            for axis in (ax_left, ax_right):
                axis.plot(xline, center - period / 2.0, "--", color="#66ff66", lw=1.0)
                axis.plot(xline, center + period / 2.0, "--", color="#66ff66", lw=1.0)
        if math.isfinite(fit_v) and fit_v > 0:
            ax_right.plot(xline, xline / fit_v, color="black", lw=0.9)

        label = chr(ord("a") + row_index)
        ax_left.text(0.6, max_time_s * 0.925, f"({label}) {period:g} s", fontsize=14)
        fit_text = "V = {:.2f} km/s\nSTDV = {:.2f} km/s".format(fit_v, std_v)
        ax_right.text(1.2, max_time_s * 0.83, fit_text, fontsize=11)
        ax_left.set_title("All nonzero CDisp picks", fontsize=12)
        ax_right.set_title("SNR / group velocity screened", fontsize=12)

        for axis in (ax_left, ax_right):
            axis.set_xlim(0.0, max_distance_km)
            axis.set_ylim(0.0, max_time_s)
            axis.set_xticks(np.arange(0, max_distance_km + 0.1, 5.0))
            axis.set_yticks(np.arange(0, max_time_s + 0.1, 1.0))
            axis.minorticks_on()
            axis.grid(True, which="major", color="#d7d7d7", linestyle="-", linewidth=0.7)
            axis.grid(True, which="minor", color="#e9e9e9", linestyle=":", linewidth=0.7)
            axis.tick_params(labelsize=11)
        ax_left.set_ylabel("Travel Time from CDisp (s)", fontsize=15)
    axes[-1, 0].set_xlabel("Distance (km)", fontsize=18)
    axes[-1, 1].set_xlabel("Distance (km)", fontsize=18)
    plt.tight_layout()
    fig.savefig(str(path), dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_paper_figure(
    path: Path,
    per_period: Dict[float, Dict[str, object]],
    *,
    target_periods: Tuple[float, ...],
    max_distance_km: float,
    max_time_s: float,
    min_wavelengths: float,
) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(len(target_periods), 2, figsize=(10.2, 4.75 * len(target_periods)), sharex=True, sharey=True)
    if len(target_periods) == 1:
        axes = np.asarray([axes])
    fig.patch.set_facecolor("white")
    xline = np.linspace(0.0, max_distance_km, 300)
    for row_index, period in enumerate(target_periods):
        payload = per_period[float(period)]
        left_rows = payload["left"]
        right_rows = payload["right"]
        summary = payload["summary"]
        reference_velocity = float(summary["reference_velocity_km_s"])
        fit_v = float(summary["fit_velocity_km_s"])
        std_v = float(summary["std_velocity_km_s"])

        ax_left = axes[row_index, 0]
        ax_right = axes[row_index, 1]
        if left_rows:
            ax_left.scatter(
                [row.distance_km for row in left_rows],
                [row.travel_time_s for row in left_rows],
                s=4,
                color="#0b51ff",
                alpha=0.45,
                linewidths=0,
            )
        if right_rows:
            ax_right.scatter(
                [row.distance_km for row in right_rows],
                [row.corrected_travel_time_s for row in right_rows],
                s=4,
                color="#0b51ff",
                alpha=0.55,
                linewidths=0,
            )

        if math.isfinite(reference_velocity) and reference_velocity > 0:
            center = xline / reference_velocity
            for axis in (ax_left, ax_right):
                axis.plot(xline, center - period / 2.0, "--", color="#66ff66", lw=1.0)
                axis.plot(xline, center + period / 2.0, "--", color="#66ff66", lw=1.0)
        if math.isfinite(fit_v) and fit_v > 0:
            ax_right.plot(xline, xline / fit_v, color="black", lw=0.9)

        label = chr(ord("a") + row_index)
        ax_left.text(0.6, max_time_s * 0.925, f"({label}) {period:g} s", fontsize=14)
        fit_text = "V = {:.2f} km/s\nSTDV = {:.2f} km/s".format(fit_v, std_v)
        ax_right.text(1.2, max_time_s * 0.83, fit_text, fontsize=11)
        ax_left.set_title("SNR / group velocity", fontsize=12)
        right_title = "Period-corrected"
        if min_wavelengths > 0:
            right_title += f" / D>={min_wavelengths:g} wavelength"
        ax_right.set_title(right_title, fontsize=12)

        for axis in (ax_left, ax_right):
            axis.set_xlim(0.0, max_distance_km)
            axis.set_ylim(0.0, max_time_s)
            axis.set_xticks(np.arange(0, max_distance_km + 0.1, 5.0))
            axis.set_yticks(np.arange(0, max_time_s + 0.1, 1.0))
            axis.minorticks_on()
            axis.grid(True, which="major", color="#d7d7d7", linestyle="-", linewidth=0.7)
            axis.grid(True, which="minor", color="#e9e9e9", linestyle=":", linewidth=0.7)
            axis.tick_params(labelsize=11)
        ax_left.set_ylabel("Travel Time from CDisp (s)", fontsize=15)
    axes[-1, 0].set_xlabel("Distance (km)", fontsize=18)
    axes[-1, 1].set_xlabel("Distance (km)", fontsize=18)
    plt.tight_layout()
    fig.savefig(str(path), dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, metadata: Dict[str, object], summary_rows: List[Dict[str, object]], figure_relpath: str) -> None:
    ensure_dir(path.parent)
    summary_html = "\n".join(
        "<tr><td>{period_s:.1f}</td><td>{initial_count}</td><td>{screened_count}</td>"
        "<td>{fit_velocity_km_s:.3f}</td><td>{std_velocity_km_s:.3f}</td>"
        "<td>{median_velocity_km_s:.3f}</td><td>{wang_velocity_km_s:.3f}</td></tr>".format(**row)
        for row in summary_rows
    )
    metadata_html = "\n".join(f"<li><strong>{key}</strong>: {value}</li>" for key, value in metadata.items())
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>DisperPicker Wang Figure 4</title>
  <style>
    body {{ font-family: Arial, "PingFang SC", sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d7dce2; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f6; }}
    img {{ max-width: 100%; border: 1px solid #d7dce2; }}
    code {{ background: #eef2f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>DisperPicker picked phase velocity -> Wang-style Figure 4</h1>
  <p>本报告读取当前全量 <code>CDisp.*.txt</code> 相速度曲线，并用配套 <code>GDisp.*.txt</code> 做群速度筛选。左列为目标周期所有非零 DisperPicker 相速度点换算得到的 <code>t = D / c(T)</code>；右列只经过 SNR 与群速度上限筛选后做过原点线性拟合，不再施加 one-wavelength、相速度范围或最大距离筛选。</p>
  <ul>{metadata_html}</ul>
  <table>
    <thead><tr><th>Period (s)</th><th>All CDisp picks</th><th>Screened picks</th><th>Fit V</th><th>STDV</th><th>Median C</th><th>Wang V</th></tr></thead>
    <tbody>{summary_html}</tbody>
  </table>
  <img src="{figure_relpath}" alt="DisperPicker Wang Figure 4">
</body>
</html>
""",
        encoding="utf-8",
    )


def write_paper_report(path: Path, metadata: Dict[str, object], summary_rows: List[Dict[str, object]], figure_relpath: str) -> None:
    ensure_dir(path.parent)
    summary_html = "\n".join(
        "<tr><td>{period_s:.1f}</td><td>{input_count}</td><td>{left_screened_count}</td>"
        "<td>{corrected_count}</td><td>{reference_velocity_km_s:.3f}</td>"
        "<td>{fit_velocity_km_s:.3f}</td><td>{std_velocity_km_s:.3f}</td>"
        "<td>{median_velocity_km_s:.3f}</td><td>{wang_velocity_km_s:.3f}</td></tr>".format(**row)
        for row in summary_rows
    )
    metadata_html = "\n".join(f"<li><strong>{key}</strong>: {value}</li>" for key, value in metadata.items())
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>DisperPicker Wang Figure 4 Paper Standard</title>
  <style>
    body {{ font-family: Arial, "PingFang SC", sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d7dce2; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f6; }}
    img {{ max-width: 100%; border: 1px solid #d7dce2; }}
    code {{ background: #eef2f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>DisperPicker picked phase velocity -> Wang Figure 4 paper-standard screen</h1>
  <p>本报告直接读取已保存的数字测量表。左列按目标周期 <code>T</code> 从 DisperPicker 相速度曲线抽取 <code>C(T)</code>，只保留满足 <code>SNR</code> 与群速度条件的原始相速度走时 <code>D/C(T)</code>；右列从左列点集继续施加更严格条件，包括台间距 <code>D &gt;= V_ref*T</code> 和整数个 <code>N*T</code> 的周期校正后再拟合。这里台间距筛选发生在 <code>D</code> 轴，不是走时轴。</p>
  <ul>{metadata_html}</ul>
  <table>
    <thead><tr><th>Period (s)</th><th>Input CDisp</th><th>Left screened</th><th>Corrected</th><th>Vref</th><th>Fit V</th><th>STDV</th><th>Median C</th><th>Wang V</th></tr></thead>
    <tbody>{summary_html}</tbody>
  </table>
  <img src="{figure_relpath}" alt="DisperPicker Wang Figure 4 paper standard">
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curves-dir",
        type=Path,
        default=Path(
            "/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/"
            "wang_disperpicker_fig4_xmlcoords_allpairs_20260701/curves_all"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_figure4_disperpicker_phase_20260701"),
    )
    parser.add_argument("--periods", default="3.0,4.0,5.0")
    parser.add_argument("--measurements-csv", type=Path, default=None)
    parser.add_argument("--group-curves-dir", type=Path, default=None)
    parser.add_argument("--max-distance-km", type=float, default=0.0, help="0 means auto/no distance cutoff.")
    parser.add_argument("--max-time-s", type=float, default=0.0, help="0 means auto/no time cutoff.")
    parser.add_argument("--min-snr", type=float, default=4.0)
    parser.add_argument("--group-vmin", type=float, default=0.0)
    parser.add_argument("--min-wavelengths", type=float, default=0.0)
    parser.add_argument("--paper-standard", action="store_true")
    parser.add_argument("--disperpicker-final-ct", type=float, default=None)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args(argv)


def load_input_measurements(args: argparse.Namespace, target_periods: Tuple[float, ...]) -> List[Measurement]:
    if args.measurements_csv is not None:
        return load_measurements_csv(args.measurements_csv, target_periods)
    return load_measurements(
        args.curves_dir,
        target_periods,
        workers=max(1, int(args.workers)),
        group_curves_dir=args.group_curves_dir,
    )


def run_paper_standard(args: argparse.Namespace, target_periods: Tuple[float, ...]) -> int:
    ensure_dir(args.output_dir)
    figures_dir = args.output_dir / "figures"
    ensure_dir(figures_dir)
    min_wavelengths = max(0.0, float(args.min_wavelengths))

    measurements = load_input_measurements(args, target_periods)
    by_period: Dict[float, List[Measurement]] = {float(period): [] for period in target_periods}
    for row in measurements:
        if args.max_distance_km <= 0 or row.distance_km <= args.max_distance_km:
            by_period[float(row.period_s)].append(row)

    per_period: Dict[float, Dict[str, object]] = {}
    summary_rows: List[Dict[str, object]] = []
    left_csv: List[Dict[str, object]] = []
    corrected_csv: List[Dict[str, object]] = []
    for period in target_periods:
        input_rows = by_period[float(period)]
        quality_rows = screen_measurements(
            input_rows,
            min_snr=args.min_snr,
            group_vmin=args.group_vmin,
            min_wavelengths=0.0,
        )
        wavelength_reference_velocity = fit_velocity_through_origin(
            [row.distance_km for row in quality_rows],
            [row.travel_time_s for row in quality_rows],
        )
        strict_rows, wavelength_distance_cutoff_km = reference_wavelength_filter(
            quality_rows,
            period_s=float(period),
            min_wavelengths=min_wavelengths,
            reference_velocity_km_s=wavelength_reference_velocity,
        )
        reference_velocity = fit_velocity_through_origin(
            [row.distance_km for row in strict_rows],
            [row.travel_time_s for row in strict_rows],
        )
        right_rows = correct_period_branches(strict_rows, reference_velocity_km_s=reference_velocity)
        summary = summarize_paper_period(float(period), input_rows, quality_rows, right_rows, reference_velocity)
        summary["quality_screened_count"] = len(quality_rows)
        summary["strict_source_count"] = len(strict_rows)
        summary["wavelength_filter_reference_velocity_km_s"] = wavelength_reference_velocity
        summary["wavelength_distance_cutoff_km"] = wavelength_distance_cutoff_km
        summary["wavelength_filter_mode"] = "distance >= min_wavelengths * period * quality_screen_fit_velocity"
        per_period[float(period)] = {"left": quality_rows, "right": right_rows, "summary": summary}
        summary_rows.append(summary)
        reference_wavelength_km = wavelength_reference_velocity * float(period)
        for row in quality_rows:
            payload = row_dict(row)
            payload["distance_over_measured_wavelength"] = row.travel_time_s / row.period_s
            payload["wavelength_filter_reference_velocity_km_s"] = wavelength_reference_velocity
            payload["reference_wavelength_km"] = reference_wavelength_km
            payload["distance_over_reference_wavelength"] = (
                row.distance_km / reference_wavelength_km
                if math.isfinite(reference_wavelength_km) and reference_wavelength_km > 0
                else float("nan")
            )
            left_csv.append(payload)
        for row in right_rows:
            payload = corrected_row_dict(row)
            payload["distance_over_measured_wavelength"] = row.raw_travel_time_s / row.period_s
            payload["wavelength_filter_reference_velocity_km_s"] = wavelength_reference_velocity
            payload["reference_wavelength_km"] = reference_wavelength_km
            payload["distance_over_reference_wavelength"] = (
                row.distance_km / reference_wavelength_km
                if math.isfinite(reference_wavelength_km) and reference_wavelength_km > 0
                else float("nan")
            )
            corrected_csv.append(payload)

    all_visible = left_csv + corrected_csv
    if args.max_distance_km > 0:
        plot_max_distance = args.max_distance_km
    else:
        max_distance = max((finite_float(row.get("distance_km")) for row in all_visible), default=25.0)
        plot_max_distance = max(5.0, math.ceil(max_distance / 5.0) * 5.0)
    if args.max_time_s > 0:
        plot_max_time = args.max_time_s
    else:
        max_time = 0.0
        for row in left_csv:
            max_time = max(max_time, finite_float(row.get("travel_time_s"), 0.0))
        for row in corrected_csv:
            max_time = max(max_time, finite_float(row.get("corrected_travel_time_s"), 0.0))
        plot_max_time = max(5.0, math.ceil(max_time))

    figure_path = figures_dir / "wang_figure4_disperpicker_phase_paper_standard.png"
    plot_paper_figure(
        figure_path,
        per_period,
        target_periods=target_periods,
        max_distance_km=plot_max_distance,
        max_time_s=plot_max_time,
        min_wavelengths=min_wavelengths,
    )
    write_csv(args.output_dir / "measurements_paper_left.csv", left_csv)
    write_csv(args.output_dir / "measurements_period_corrected.csv", corrected_csv)
    write_csv(args.output_dir / "fit_summary.csv", summary_rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "curves_dir": str(args.curves_dir),
        "measurements_csv": str(args.measurements_csv) if args.measurements_csv is not None else None,
        "target_periods_s": list(target_periods),
        "max_distance_km": args.max_distance_km,
        "plot_max_distance_km": plot_max_distance,
        "max_time_s": args.max_time_s,
        "plot_max_time_s": plot_max_time,
        "min_snr": args.min_snr,
        "group_velocity_min_km_s": args.group_vmin,
        "group_velocity_max_rule_km_s": "3.0 for T<4.5s, 3.3 for T>=4.5s",
        "min_wavelengths": min_wavelengths,
        "wavelength_screen_enabled": min_wavelengths > 0,
        "wavelength_filter_mode": "distance_reference",
        "one_wavelength_rule": (
            "disabled"
            if min_wavelengths <= 0
            else (
                f"D >= {min_wavelengths:g} * V_ref(T) * T; "
                "V_ref(T) is fit from SNR/group-velocity screened rows before the distance cutoff"
            )
        ),
        "left_column_filter": "SNR and group velocity only",
        "right_column_filter": "left-column rows plus reference-wavelength distance cutoff and period correction",
        "period_correction": "branch_n = round((D / Vref - t_raw) / T); t_corrected = t_raw + branch_n * T",
        "reference_velocity_rule": "through-origin fit from left-screened rows for each period; Wang reference is comparison only",
        "disperpicker_final_ct": args.disperpicker_final_ct,
        "wang_reference_velocities_km_s": WANG_REFERENCE_VELOCITIES,
        "input_cdisp_measurements": len(measurements),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_paper_report(args.output_dir / "report.html", metadata, summary_rows, "figures/" + figure_path.name)
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary_rows}, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    target_periods = parse_periods(args.periods)
    if args.paper_standard:
        return run_paper_standard(args, target_periods)
    ensure_dir(args.output_dir)
    figures_dir = args.output_dir / "figures"
    ensure_dir(figures_dir)

    measurements = load_input_measurements(args, target_periods)
    by_period: Dict[float, List[Measurement]] = {float(period): [] for period in target_periods}
    for row in measurements:
        if args.max_distance_km <= 0 or row.distance_km <= args.max_distance_km:
            by_period[float(row.period_s)].append(row)

    per_period: Dict[float, Dict[str, object]] = {}
    summary_rows: List[Dict[str, object]] = []
    initial_csv: List[Dict[str, object]] = []
    screened_csv: List[Dict[str, object]] = []
    for period in target_periods:
        initial = by_period[float(period)]
        screened = screen_measurements(
            initial,
            min_snr=args.min_snr,
            group_vmin=args.group_vmin,
            min_wavelengths=args.min_wavelengths,
        )
        summary = summarize_period(float(period), initial, screened)
        per_period[float(period)] = {"initial": initial, "screened": screened, "summary": summary}
        summary_rows.append(summary)
        initial_csv.extend(row_dict(row) for row in initial)
        screened_csv.extend(row_dict(row) for row in screened)

    all_visible = initial_csv + screened_csv
    if args.max_distance_km > 0:
        plot_max_distance = args.max_distance_km
    else:
        max_distance = max((finite_float(row.get("distance_km")) for row in all_visible), default=25.0)
        plot_max_distance = max(5.0, math.ceil(max_distance / 5.0) * 5.0)
    if args.max_time_s > 0:
        plot_max_time = args.max_time_s
    else:
        max_time = max((finite_float(row.get("travel_time_s")) for row in all_visible), default=10.0)
        plot_max_time = max(5.0, math.ceil(max_time))
    figure_path = figures_dir / "wang_figure4_disperpicker_phase.png"
    plot_figure(
        figure_path,
        per_period,
        target_periods=target_periods,
        max_distance_km=plot_max_distance,
        max_time_s=plot_max_time,
    )
    write_csv(args.output_dir / "measurements_all_cdisp.csv", initial_csv)
    write_csv(args.output_dir / "measurements_screened.csv", screened_csv)
    write_csv(args.output_dir / "fit_summary.csv", summary_rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "curves_dir": str(args.curves_dir),
        "target_periods_s": list(target_periods),
        "max_distance_km": args.max_distance_km,
        "plot_max_distance_km": plot_max_distance,
        "max_time_s": args.max_time_s,
        "plot_max_time_s": plot_max_time,
        "min_snr": args.min_snr,
        "group_velocity_min_km_s": args.group_vmin,
        "group_velocity_max_rule_km_s": "3.0 for T<4.5s, 3.3 for T>=4.5s",
        "min_wavelengths": args.min_wavelengths,
        "disperpicker_final_ct": args.disperpicker_final_ct,
        "wang_reference_velocities_km_s": WANG_REFERENCE_VELOCITIES,
        "input_cdisp_measurements": len(measurements),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir / "report.html", metadata, summary_rows, "figures/" + figure_path.name)
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
