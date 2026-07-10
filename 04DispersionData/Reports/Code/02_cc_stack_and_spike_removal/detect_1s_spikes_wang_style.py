#!/usr/bin/env python3
"""Show Wang-style repeating 1 s spikes using distance-bin stacked 1D CCFs."""

from __future__ import annotations

import argparse
import csv
import html
import os
import platform
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_STACK_ROOT = Path("/mnt/external_usb_ext4/2014_St.Helens/data_1D_20140718_20140805/STACK")
DEFAULT_CCF_ROOT = Path("/mnt/external_usb_ext4/2014_St.Helens/data_1D_20140718_20140805/CCF")
DEFAULT_OUTPUT = Path("/mnt/data_hdd/MSH_ANT/parameter_tests/1d_1s_spike_test_20260619")
NS = {"s": "http://www.fdsn.org/xml/station/1"}


def distance_bin_index(distance_km: float, width_km: float = 0.5) -> int:
    return int(np.floor(float(distance_km) / width_km))


def load_station_coordinates(metadata_root: Path) -> dict[str, tuple[float, float]]:
    records: dict[str, tuple[float, float]] = {}
    for xml_path in sorted(Path(metadata_root).glob("1D.*.xml")):
        root = ET.parse(xml_path).getroot()
        station = root.find(".//s:Station", NS)
        if station is None:
            continue
        latitude = float(station.findtext("s:Latitude", namespaces=NS))
        longitude = float(station.findtext("s:Longitude", namespaces=NS))
        records[xml_path.stem] = (latitude, longitude)
    return records


def _normalize_station_code(value) -> str:
    text = str(value)
    return text if text.startswith("1D.") else f"1D.{text}"


def great_circle_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * radius_km * np.arcsin(np.sqrt(a)))


def dataset_distance_km(attrs, station_coords: dict[str, tuple[float, float]]) -> float:
    if "dist" in attrs:
        return float(attrs["dist"])
    source = _normalize_station_code(attrs["station_source"])
    receiver = _normalize_station_code(attrs["station_receiver"])
    if source not in station_coords or receiver not in station_coords:
        raise KeyError(f"Missing coordinates for {source} or {receiver}")
    lat1, lon1 = station_coords[source]
    lat2, lon2 = station_coords[receiver]
    return great_circle_distance_km(lat1, lon1, lat2, lon2)


def positive_lag(trace: np.ndarray, dt: float, maxlag: float, end_s: float = 16.0):
    values = np.asarray(trace, dtype=float)
    center = int(round(maxlag / dt))
    count = int(round(end_s / dt)) + 1
    return np.arange(count) * dt, values[center : center + count]


def normalize_for_record_section(trace):
    values = np.asarray(trace, dtype=float).copy()
    values -= np.nanmean(values)
    scale = np.nanmax(np.abs(values))
    if not np.isfinite(scale) or scale <= 0:
        return values * np.nan
    return values / scale


def local_trend_residual(trace, dt: float, window_s: float = 0.8):
    values = np.asarray(trace, dtype=float)
    half_window = max(1, int(round(window_s / dt / 2.0)))
    kernel = np.ones(2 * half_window + 1, dtype=float)
    kernel /= kernel.size
    padded = np.pad(values, half_window, mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    return values - smooth


def zoom_window(center_s: float, width_s: float, min_s: float, max_s: float):
    start = max(min_s, center_s - width_s / 2.0)
    end = min(max_s, start + width_s)
    start = max(min_s, end - width_s)
    return float(start), float(end)


def add_trace_to_distance_bin(sums, counts, trace, distance_km, bin_width=0.5, max_distance=25.0):
    if not (0 <= float(distance_km) < max_distance):
        return sums, counts
    normalized = normalize_for_record_section(trace)
    if not np.all(np.isfinite(normalized)):
        return sums, counts
    if sums is None:
        sums = np.zeros((counts.size, normalized.size), dtype=np.float64)
    index = distance_bin_index(distance_km, bin_width)
    sums[index] += normalized
    counts[index] += 1
    return sums, counts


def integer_spike_contrast(time, trace, start_second=1, end_second=15, peak_half_width=0.06, background_half_width=0.30):
    time = np.asarray(time, dtype=float); values = np.abs(np.asarray(trace, dtype=float))
    rows = []
    for second in range(start_second, end_second + 1):
        peak = np.abs(time - second) <= peak_half_width
        background = (np.abs(time - second) > peak_half_width) & (np.abs(time - second) <= background_half_width)
        peak_value = float(np.max(values[peak])) if np.any(peak) else np.nan
        background_value = float(np.median(values[background])) if np.any(background) else np.nan
        rows.append({"second": second, "peak_abs": peak_value, "local_background_abs": background_value, "contrast": peak_value / background_value if background_value > 0 else np.nan})
    return rows


def one_second_phase_profile(time, trace, start_second=1, end_second=15):
    """Fold a trace modulo 1 s and score every possible spike phase."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(trace, dtype=float)
    dt = float(np.median(np.diff(time)))
    phases = np.arange(int(round(1.0 / dt))) * dt
    rows = []
    for phase in phases:
        samples = []
        for second in range(start_second, end_second + 1):
            target = second + phase
            if target <= time[-1] + dt / 2:
                samples.append(values[np.argmin(np.abs(time - target))])
        samples = np.asarray(samples, dtype=float)
        rows.append({
            "phase_s": float(phase),
            "median_signed": float(np.median(samples)),
            "median_abs": float(np.median(np.abs(samples))),
            "rms": float(np.sqrt(np.mean(samples ** 2))),
            "sample_count": int(samples.size),
        })
    phase_median = float(np.median([row["median_abs"] for row in rows]))
    for row in rows:
        row["contrast_to_phase_median"] = row["median_abs"] / phase_median if phase_median > 0 else np.nan
    return rows


def build_repeating_spike_template(time, trace, phase_s, start_second=1, end_second=15, half_width_s=0.16):
    """Build a median 1 s repeating-spike template from one coherent trace."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(trace, dtype=float)
    dt = float(np.median(np.diff(time)))
    half_samples = int(round(half_width_s / dt))
    offsets = np.arange(-half_samples, half_samples + 1) * dt
    snippets = []
    for second in range(start_second, end_second + 1):
        center = second + phase_s
        center_index = int(np.argmin(np.abs(time - center)))
        start = center_index - half_samples
        stop = center_index + half_samples + 1
        if start < 0 or stop > values.size:
            continue
        snippets.append(values[start:stop])
    if not snippets:
        raise ValueError("No complete 1 s spike windows available for template")
    template = np.median(np.vstack(snippets), axis=0)
    edge_count = max(1, int(round(0.04 / dt)))
    baseline = np.median(np.r_[template[:edge_count], template[-edge_count:]])
    return offsets, template - baseline


def fit_repeating_spike_scale(time, trace, offsets, template, phase_s, start_second=1, end_second=15):
    """Fit one scalar amplitude for a repeating template against one trace."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(trace, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    template = np.asarray(template, dtype=float)
    observed = []
    predicted = []
    for second in range(start_second, end_second + 1):
        center = second + phase_s
        for offset, value in zip(offsets, template):
            target = center + offset
            if target < time[0] or target > time[-1]:
                continue
            observed.append(values[int(np.argmin(np.abs(time - target)))])
            predicted.append(value)
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.dot(predicted, predicted))
    if not np.isfinite(denominator) or denominator <= 0:
        return 0.0
    return float(np.dot(observed, predicted) / denominator)


def subtract_repeating_spike_template(time, trace, offsets, template, phase_s, start_second=1, end_second=15, scale=1.0):
    """Subtract the same fixed-amplitude template at a fixed phase in each second."""
    time = np.asarray(time, dtype=float)
    corrected = np.asarray(trace, dtype=float).copy()
    offsets = np.asarray(offsets, dtype=float)
    template = np.asarray(template, dtype=float)
    if offsets.size != template.size:
        raise ValueError("offsets and template must have the same length")
    for second in range(start_second, end_second + 1):
        center = second + phase_s
        for offset, value in zip(offsets, template):
            target = center + offset
            if target < time[0] or target > time[-1]:
                continue
            corrected[int(np.argmin(np.abs(time - target)))] -= scale * value
    return corrected


def iter_unique_paths(root: Path):
    for source_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("1D.")):
        for receiver_dir in sorted(path for path in source_dir.iterdir() if path.is_dir() and path.name.startswith("1D.")):
            if source_dir.name >= receiver_dir.name:
                continue
            matches = sorted(receiver_dir.glob("*.h5"))
            if matches:
                yield matches[0]


def reservoir_sample(paths, size: int, seed: int):
    rng = random.Random(seed); sample = []; seen = 0
    for path in paths:
        seen += 1
        if len(sample) < size:
            sample.append(path)
        else:
            index = rng.randrange(seen)
            if index < size:
                sample[index] = path
    return sorted(sample), seen


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def read_sample(paths, bin_width=0.5, max_distance=25.0, end_s=16.0, normalize_each_pair=False, metadata_root: Path | None = None):
    import h5py
    nbin = int(np.ceil(max_distance / bin_width)); sums = None; counts = np.zeros(nbin, dtype=int); examples = []
    used = 0; failures = 0
    station_coords = load_station_coordinates(metadata_root) if metadata_root is not None else {}
    for path in paths:
        try:
            with h5py.File(path, "r") as handle:
                dataset = handle["AuxiliaryData/Allstack_pws/ZZ"]
                distance = dataset_distance_km(dataset.attrs, station_coords)
                if not (0 <= distance < max_distance):
                    continue
                time, trace = positive_lag(dataset[:], float(dataset.attrs["dt"]), float(dataset.attrs["maxlag"]), end_s)
            if normalize_each_pair:
                trace = normalize_for_record_section(trace)
            if not np.all(np.isfinite(trace)):
                continue
            if sums is None:
                sums = np.zeros((nbin, trace.size), dtype=np.float64)
            index = distance_bin_index(distance, bin_width)
            sums[index] += trace; counts[index] += 1; used += 1
            if len(examples) < 6 and distance > 3:
                examples.append((path.parent.parent.name + "__" + path.parent.name, distance, trace.copy()))
        except Exception:
            failures += 1
    if sums is None or used == 0:
        raise RuntimeError("No usable CCFs")
    stacks = np.full_like(sums, np.nan)
    valid = counts > 0; stacks[valid] = sums[valid] / counts[valid, None]
    return time, stacks, counts, examples, used, failures


def station_pair_from_group(name):
    parts = name.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected station-pair group name: {name}")
    return tuple(sorted(parts))


def read_daily_ccf_bins(files, bin_width=0.5, max_distance=25.0, end_s=16.0, dataset_name="DPZ_DPZ"):
    import h5py
    nbin = int(np.ceil(max_distance / bin_width))
    sums = None
    counts = np.zeros(nbin, dtype=int)
    examples = []
    used = 0
    failures = 0
    pair_names = set()
    time = None
    for path in files:
        try:
            with h5py.File(path, "r") as handle:
                root = handle["AuxiliaryData"]
                for group_name in root:
                    try:
                        station_a, station_b = station_pair_from_group(group_name)
                        if station_a == station_b:
                            continue
                        group = root[group_name]
                        if dataset_name not in group:
                            continue
                        dataset = group[dataset_name]
                        distance = float(dataset.attrs["dist"])
                        if not (0 <= distance < max_distance):
                            continue
                        this_time, trace = positive_lag(
                            np.asarray(dataset[0], dtype=float),
                            float(dataset.attrs["dt"]),
                            float(dataset.attrs["maxlag"]),
                            end_s,
                        )
                        if time is None:
                            time = this_time
                        sums, counts = add_trace_to_distance_bin(sums, counts, trace, distance, bin_width, max_distance)
                        if sums is not None:
                            used += 1
                            pair_names.add((station_a, station_b))
                            if len(examples) < 6 and distance > 3:
                                examples.append((group_name, distance, trace.copy()))
                    except Exception:
                        failures += 1
        except Exception:
            failures += 1
    if sums is None or used == 0 or time is None:
        raise RuntimeError("No usable daily CCFs")
    stacks = np.full_like(sums, np.nan)
    valid = counts > 0
    stacks[valid] = sums[valid] / counts[valid, None]
    return time, stacks, counts, examples, used, failures, len(pair_names)


def make_wang_figure3a_plot(output, time, stacks, counts, bin_width, zoom_center_s=0.5, zoom_width_s=1.0):
    output.mkdir(parents=True, exist_ok=True)
    normalized = normalized_bins(stacks, counts)
    valid = np.flatnonzero(counts > 0)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, ax = plt.subplots(figsize=(4.1, 5.2), constrained_layout=True)
    for index in valid:
        distance = (index + 0.5) * bin_width
        ax.plot(time, distance + normalized[index] * bin_width * 0.55, color="black", lw=0.38)
    ax.set_xlim(0, 16)
    ax.set_ylim(25, 0)
    ax.set_xticks(np.arange(0, 17, 2))
    ax.set_yticks(np.arange(0, 26, 5))
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Distance (km)")
    ax.text(0.01, 1.02, "(a)", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.tick_params(top=True, right=True, length=3, width=0.8)
    fig.savefig(output / "wang_figure3a_style_before_spike_removal.png", dpi=600)
    fig.savefig(output / "wang_figure3a_style_before_spike_removal.pdf")
    plt.close(fig)

    zoom_start_s, zoom_end_s = zoom_window(
        center_s=zoom_center_s,
        width_s=zoom_width_s,
        min_s=float(time[0]),
        max_s=float(time[-1]),
    )
    fig, ax = plt.subplots(figsize=(4.1, 5.2), constrained_layout=True)
    for index in valid:
        distance = (index + 0.5) * bin_width
        ax.plot(time, distance + normalized[index] * bin_width * 0.55, color="black", lw=0.38)
    ax.set_xlim(zoom_start_s, zoom_end_s)
    ax.set_ylim(25, 0)
    ax.set_xticks(np.linspace(zoom_start_s, zoom_end_s, 6))
    ax.set_yticks(np.arange(0, 26, 5))
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Distance (km)")
    ax.text(0.01, 1.02, "(a-zoom)", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.axvline(zoom_center_s, color="#dc2626", lw=0.5, alpha=0.55)
    ax.tick_params(top=True, right=True, length=3, width=0.8)
    fig.savefig(output / "wang_figure3a_style_zoom.png", dpi=600)
    plt.close(fig)

    dt = float(np.median(np.diff(time)))
    diagnostic = diagnostic_bins(normalized, counts, dt=dt, window_s=0.8)
    fig, ax = plt.subplots(figsize=(4.1, 5.2), constrained_layout=True)
    for index in valid:
        if not np.all(np.isfinite(diagnostic[index])):
            continue
        distance = (index + 0.5) * bin_width
        ax.plot(time, distance + diagnostic[index] * bin_width * 0.55, color="black", lw=0.38)
    for second in range(0, 17):
        ax.axvline(second + 0.5, color="#dc2626", lw=0.25, alpha=0.25)
    ax.set_xlim(0, 16)
    ax.set_ylim(25, 0)
    ax.set_xticks(np.arange(0, 17, 2))
    ax.set_yticks(np.arange(0, 26, 5))
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Distance (km)")
    ax.text(0.01, 1.02, "(diag)", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.tick_params(top=True, right=True, length=3, width=0.8)
    fig.savefig(output / "wang_figure3a_style_diagnostic_residual.png", dpi=600)
    plt.close(fig)

    coherent = np.nanmean(normalized[valid], axis=0)
    coherent /= max(np.nanmax(np.abs(coherent)), 1e-12)
    rows = integer_spike_contrast(time, coherent)
    phase_rows = one_second_phase_profile(time, coherent)
    write_csv(output / "wang_figure3a_integer_spike_contrast.csv", rows)
    write_csv(output / "wang_figure3a_phase_profile.csv", phase_rows)
    return rows, phase_rows


def bandpass(data, dt, low=0.8, high=1.2):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [low, high], btype="bandpass", fs=1.0 / dt, output="sos")
    return sosfiltfilt(sos, data)


def normalized_bins(stacks, counts):
    result = np.full_like(stacks, np.nan); valid = counts > 0
    scales = np.nanmax(np.abs(stacks[valid]), axis=1)
    keep = scales > 0
    indices = np.flatnonzero(valid)[keep]
    result[indices] = stacks[indices] / scales[keep, None]
    return result


def diagnostic_bins(normalized, counts, dt: float, window_s: float = 0.8):
    diagnostic = np.full_like(normalized, np.nan)
    for index in np.flatnonzero(counts > 0):
        if not np.all(np.isfinite(normalized[index])):
            continue
        residual = local_trend_residual(normalized[index], dt=dt, window_s=window_s)
        scale = np.nanmax(np.abs(residual))
        if np.isfinite(scale) and scale > 0:
            diagnostic[index] = residual / scale
    return diagnostic


def derive_template_trace(time, normalized, counts, source: str = "coherent"):
    valid = np.flatnonzero(counts > 0)
    if valid.size == 0:
        raise ValueError("No valid distance bins available for template derivation")
    if source == "coherent":
        source_bins = normalized
    elif source == "diagnostic":
        dt = float(np.median(np.diff(time)))
        source_bins = diagnostic_bins(normalized, counts, dt=dt, window_s=0.8)
    else:
        raise ValueError(f"Unsupported template source: {source}")

    usable = valid[np.array([np.all(np.isfinite(source_bins[index])) for index in valid], dtype=bool)]
    if usable.size == 0:
        raise ValueError(f"No usable distance bins available for template source: {source}")

    template_trace = np.nanmean(source_bins[usable], axis=0)
    scale = np.nanmax(np.abs(template_trace))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Template source {source} produced zero-amplitude coherent trace")
    return template_trace / scale, source_bins


def make_plots(output, time, stacks, counts, examples, bin_width):
    output.mkdir(parents=True, exist_ok=True); normalized = normalized_bins(stacks, counts); valid = np.flatnonzero(counts > 0)
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    for index in valid:
        distance = (index + 0.5) * bin_width
        ax.plot(time, distance + normalized[index] * bin_width * 0.42, color="#111827", lw=0.45)
    for second in range(1, 16): ax.axvline(second, color="#dc2626", lw=0.35, alpha=0.25)
    ax.set(xlim=(0, 16), ylim=(25, 0), xlabel="Positive lag (s)", ylabel="Distance (km)", title="Wang-style 0.5 km distance-bin CCF stacks (no bandpass)")
    fig.savefig(output / "distance_bin_wiggle.png", dpi=180); plt.close(fig)

    coherent = np.nanmean(normalized[valid], axis=0); filtered = bandpass(coherent, time[1] - time[0])
    coherent /= max(np.max(np.abs(coherent)), 1e-12); filtered /= max(np.max(np.abs(filtered)), 1e-12)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time, coherent, color="#111827", lw=1); axes[0].set_title("Coherent stack across distance bins (unfiltered)")
    axes[1].plot(time, filtered, color="#dc2626", lw=1); axes[1].set_title("Same trace, 0.8-1.2 Hz bandpass")
    for ax in axes:
        for second in range(1, 16): ax.axvline(second, color="#2563eb", lw=0.6, alpha=0.35)
        ax.set_ylabel("Normalized amplitude")
    axes[1].set(xlabel="Positive lag (s)", xlim=(0, 16))
    fig.savefig(output / "coherent_integer_seconds.png", dpi=180); plt.close(fig)

    rows = integer_spike_contrast(time, coherent)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar([r["second"] for r in rows], [r["contrast"] for r in rows], color="#0f766e")
    ax.axhline(1, color="#64748b", ls="--"); ax.set(xlabel="Integer lag (s)", ylabel="Peak / local background", title="Integer-second spike contrast")
    fig.savefig(output / "integer_spike_contrast.png", dpi=180); plt.close(fig)

    phase_rows = one_second_phase_profile(time, coherent)
    phases = np.array([row["phase_s"] for row in phase_rows])
    signed = np.array([row["median_signed"] for row in phase_rows])
    median_abs = np.array([row["median_abs"] for row in phase_rows])
    contrast = np.array([row["contrast_to_phase_median"] for row in phase_rows])
    best = int(np.nanargmax(median_abs))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    axes[0].plot(phases, signed / max(np.max(np.abs(signed)), 1e-12), color="#334155", marker="o", ms=3)
    axes[0].plot(phases, median_abs / max(np.max(median_abs), 1e-12), color="#dc2626", marker="o", ms=3)
    axes[0].axvline(phases[best], color="#2563eb", ls="--")
    axes[0].legend(["Median signed amplitude", "Median absolute amplitude"])
    axes[0].set(ylabel="Normalized folded amplitude", title="CCF folded modulo 1 s (lags 1-15 s)")
    axes[1].bar(phases, contrast, width=(time[1] - time[0]) * 0.8, color="#0f766e")
    axes[1].axhline(1, color="#64748b", ls="--")
    axes[1].axvline(phases[best], color="#2563eb", ls="--")
    axes[1].set(xlabel="Phase within each second (s)", ylabel="Median |amplitude| / phase median", xlim=(-0.02, 0.98))
    fig.savefig(output / "one_second_phase_fold.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 1.5 * len(examples)), sharex=True, constrained_layout=True)
    for ax, (name, distance, trace) in zip(np.atleast_1d(axes), examples):
        scale = max(np.max(np.abs(trace)), 1e-12); ax.plot(time, trace / scale, color="#334155", lw=0.6); ax.set_ylabel(f"{distance:.1f} km"); ax.set_title(name, loc="left", fontsize=8)
    axes[-1].set(xlabel="Positive lag (s)", xlim=(0, 16)); fig.savefig(output / "single_pair_examples.png", dpi=180); plt.close(fig)
    return coherent, filtered, rows, phase_rows


def make_template_subtraction_plots(output, time, stacks, counts, examples, bin_width, template_source="coherent"):
    output.mkdir(parents=True, exist_ok=True)
    normalized = normalized_bins(stacks, counts)
    valid = np.flatnonzero(counts > 0)
    before = np.nanmean(normalized[valid], axis=0)
    before /= max(np.max(np.abs(before)), 1e-12)
    template_phase_trace, _ = derive_template_trace(time, normalized, counts, source=template_source)
    phase_rows_before = one_second_phase_profile(time, template_phase_trace)
    best_phase = max(phase_rows_before, key=lambda row: row["median_abs"])
    template_reference = before if template_source == "diagnostic" else template_phase_trace
    offsets, template = build_repeating_spike_template(time, template_reference, best_phase["phase_s"])

    corrected_bins = normalized.copy()
    scales = []
    for index in valid:
        scale = fit_repeating_spike_scale(time, normalized[index], offsets, template, best_phase["phase_s"])
        scales.append(scale)
        corrected_bins[index] = subtract_repeating_spike_template(
            time, normalized[index], offsets, template, best_phase["phase_s"], scale=scale
        )
    after = np.nanmean(corrected_bins[valid], axis=0)
    after /= max(np.max(np.abs(after)), 1e-12)
    phase_rows_after = one_second_phase_profile(time, after)
    rows_before = integer_spike_contrast(time, before)
    rows_after = integer_spike_contrast(time, after)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    for ax, data, title in zip(axes, [normalized, corrected_bins], ["Before template subtraction", "After template subtraction"]):
        for index in valid:
            distance = (index + 0.5) * bin_width
            ax.plot(time, distance + data[index] * bin_width * 0.42, color="#111827", lw=0.45)
        for second in range(1, 16):
            ax.axvline(second + best_phase["phase_s"], color="#dc2626", lw=0.35, alpha=0.25)
        ax.set(xlim=(0, 16), ylim=(25, 0), xlabel="Positive lag (s)", title=title)
    axes[0].set_ylabel("Distance (km)")
    fig.savefig(output / "distance_bin_wiggle_before_after.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(time, before, color="#111827", lw=1.1, label="Before")
    ax.plot(time, after, color="#dc2626", lw=1.0, label="After")
    for second in range(1, 16):
        ax.axvline(second + best_phase["phase_s"], color="#2563eb", lw=0.6, alpha=0.35)
    ax.legend()
    ax.set(xlim=(0, 16), xlabel="Positive lag (s)", ylabel="Normalized amplitude", title="Coherent distance-bin stack before/after template subtraction")
    fig.savefig(output / "coherent_before_after.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax.plot(offsets, template, color="#dc2626", marker="o", ms=3)
    ax.axhline(0, color="#64748b", lw=0.7)
    ax.set(xlabel="Offset from spike center (s)", ylabel="Amplitude", title=f"Median template at phase {best_phase['phase_s']:.2f} s")
    fig.savefig(output / "spike_template.png", dpi=180)
    plt.close(fig)

    phases_before = np.array([row["phase_s"] for row in phase_rows_before])
    contrast_before = np.array([row["contrast_to_phase_median"] for row in phase_rows_before])
    contrast_after = np.array([row["contrast_to_phase_median"] for row in phase_rows_after])
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    width = (time[1] - time[0]) * 0.38
    ax.bar(phases_before - width / 2, contrast_before, width=width, color="#111827", label="Before")
    ax.bar(phases_before + width / 2, contrast_after, width=width, color="#dc2626", label="After")
    ax.axhline(1, color="#64748b", ls="--")
    ax.axvline(best_phase["phase_s"], color="#2563eb", ls="--")
    ax.legend()
    ax.set(xlabel="Phase within each second (s)", ylabel="Median |amplitude| / phase median", xlim=(-0.02, 0.98), title="1 s fold contrast before/after")
    fig.savefig(output / "one_second_phase_fold_before_after.png", dpi=180)
    plt.close(fig)

    seconds = [row["second"] for row in rows_before]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(seconds, [row["contrast"] for row in rows_before], color="#111827", marker="o", label="Before")
    ax.plot(seconds, [row["contrast"] for row in rows_after], color="#dc2626", marker="o", label="After")
    ax.axhline(1, color="#64748b", ls="--")
    ax.legend()
    ax.set(xlabel="Integer lag (s)", ylabel="Peak / local background", title="Integer-second contrast before/after")
    fig.savefig(output / "integer_spike_contrast_before_after.png", dpi=180)
    plt.close(fig)

    return {
        "best_phase": best_phase,
        "template_offsets": offsets,
        "template": template,
        "template_source": template_source,
        "scale_median": float(np.median(scales)) if scales else np.nan,
        "scale_abs_median": float(np.median(np.abs(scales))) if scales else np.nan,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "phase_rows_before": phase_rows_before,
        "phase_rows_after": phase_rows_after,
    }


def write_report(output, sample_size, candidate_count, used, failures, rows, phase_rows, counts):
    median = float(np.nanmedian([r["contrast"] for r in rows])); maximum = max(rows, key=lambda r: r["contrast"])
    best_phase = max(phase_rows, key=lambda row: row["median_abs"])
    table = "".join(f"<tr><td>{r['second']}</td><td>{r['peak_abs']:.4f}</td><td>{r['local_background_abs']:.4f}</td><td>{r['contrast']:.2f}</td></tr>" for r in rows)
    text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>1D 数据 1 s 重复尖峰测验</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1250px;margin:24px auto;padding:0 18px;color:#172033}}img{{max-width:100%;border:1px solid #cbd5e1}}table{{border-collapse:collapse}}th,td{{border:1px solid #cbd5e1;padding:6px}}.note{{background:#fff7ed;border:1px solid #f59e0b;padding:12px}}</style></head><body><h1>1D 数据中的 1 s 重复尖峰测验</h1><p>计算主机：<code>{html.escape(platform.node())}</code>。按照 Wang et al. (2017) Figure 3a，将单台对 PWS 互相关按 0.5 km 台距分箱后叠加；本轮仅检测，不做模板减除。</p><div class='note'>候选唯一无序台站对 {candidate_count}，固定随机抽样 {sample_size}，实际使用 {used}，读取失败 {failures}。整数秒尖峰对邻近背景的中位增强为 {median:.2f} 倍，最大出现在 {maximum['second']} s（{maximum['contrast']:.2f} 倍）。1 s 折叠检验的最强相位是每秒内 {best_phase['phase_s']:.2f} s，强度为其余相位中位数的 {best_phase['contrast_to_phase_median']:.2f} 倍。上述统计判断固定周期成分，不能单独证明来源一定是 GPS。</div><h2>0.5 km 台距分箱剖面</h2><img src='distance_bin_wiggle.png'><h2>跨距离箱相干叠加</h2><img src='coherent_integer_seconds.png'><h2>1 s 折叠与错位对照</h2><p>把 1–15 s 分成 14 个一秒片段并重合。若存在每秒重复的窄尖峰，应在某个固定相位形成明显窄峰；所有其他相位就是同一数据的错位对照。</p><img src='one_second_phase_fold.png'><h2>整数秒增强</h2><img src='integer_spike_contrast.png'><table><tr><th>延迟/s</th><th>尖峰振幅</th><th>邻近背景</th><th>比值</th></tr>{table}</table><h2>单台对示例</h2><img src='single_pair_examples.png'></body></html>"""
    (output / "report.html").write_text(text, encoding="utf-8")
    write_csv(output / "integer_spike_contrast.csv", rows)
    write_csv(output / "one_second_phase_profile.csv", phase_rows)
    write_csv(output / "distance_bin_counts.csv", [{"bin_center_km": (i + 0.5) * 0.5, "pair_count": int(n)} for i, n in enumerate(counts)])


def write_template_subtraction_report(output, sample_size, candidate_count, used, failures, result, counts):
    rows_before = result["rows_before"]
    rows_after = result["rows_after"]
    phase_rows_before = result["phase_rows_before"]
    phase_rows_after = result["phase_rows_after"]
    best_phase = result["best_phase"]
    template_source = result["template_source"]
    scale_median = result["scale_median"]
    scale_abs_median = result["scale_abs_median"]
    before_median = float(np.nanmedian([row["contrast"] for row in rows_before]))
    after_median = float(np.nanmedian([row["contrast"] for row in rows_after]))
    before_best = max(phase_rows_before, key=lambda row: row["median_abs"])
    after_at_phase = min(phase_rows_after, key=lambda row: abs(row["phase_s"] - best_phase["phase_s"]))
    reduction = 1.0 - (after_at_phase["median_abs"] / before_best["median_abs"]) if before_best["median_abs"] else np.nan
    table = "".join(
        f"<tr><td>{before['second']}</td><td>{before['contrast']:.2f}</td><td>{after['contrast']:.2f}</td><td>{after['contrast'] - before['contrast']:.2f}</td></tr>"
        for before, after in zip(rows_before, rows_after)
    )
    text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>1D 数据 1 s 尖峰模板减除测验</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1250px;margin:24px auto;padding:0 18px;color:#172033}}img{{max-width:100%;border:1px solid #cbd5e1}}table{{border-collapse:collapse}}th,td{{border:1px solid #cbd5e1;padding:6px}}.note{{background:#eff6ff;border:1px solid #60a5fa;padding:12px}}.warn{{background:#fff7ed;border:1px solid #f59e0b;padding:12px}}</style></head><body><h1>1D 数据 1 s 尖峰模板减除测验</h1><p>计算主机：<code>{html.escape(platform.node())}</code>。本报告使用全部可读 1D PWS STACK 参与 0.5 km 台距分箱，再从 <b>{html.escape(template_source)}</b> 模板源提取固定相位模板，并在每个距离 bin 上先拟合一个标量幅度，再按同一固定相位减除；原始 H5 未被覆盖。</p><div class='note'>候选唯一无序台站对 {candidate_count}，本轮抽样上限 {sample_size}，实际使用 {used}，读取失败 {failures}。模板相位为每秒内 <b>{best_phase['phase_s']:.2f} s</b>。该相位折叠幅度从 {before_best['median_abs']:.4f} 降到 {after_at_phase['median_abs']:.4f}，降低约 {reduction * 100:.1f}%。整数秒附近增强的中位数从 {before_median:.2f} 倍变为 {after_median:.2f} 倍。各距离 bin 拟合系数的中位数为 {scale_median:.3f}，绝对值中位数为 {scale_abs_median:.3f}。</div><div class='warn'>这是 Wang 式模板减除的诊断性实现：用于判断固定 1 s 相位干扰是否能被模板压低。因为 Wang 文献没有公开具体代码和完整模板参数，这里不直接改写原始台站对 H5。若模板源为 <b>diagnostic</b>，表示模板先从局部背景去除后的残差图里识别，再用于实际扣除。</div><h2>0.5 km 台距分箱：减除前/后</h2><img src='distance_bin_wiggle_before_after.png'><h2>跨距离箱相干叠加：减除前/后</h2><img src='coherent_before_after.png'><h2>提取出的 1 s 尖峰模板</h2><img src='spike_template.png'><h2>1 s 折叠相位：减除前/后</h2><img src='one_second_phase_fold_before_after.png'><h2>整数秒增强：减除前/后</h2><img src='integer_spike_contrast_before_after.png'><table><tr><th>延迟/s</th><th>减除前比值</th><th>减除后比值</th><th>变化</th></tr>{table}</table></body></html>"""
    (output / "report.html").write_text(text, encoding="utf-8")
    write_csv(output / "integer_spike_contrast_before.csv", rows_before)
    write_csv(output / "integer_spike_contrast_after.csv", rows_after)
    write_csv(output / "one_second_phase_profile_before.csv", phase_rows_before)
    write_csv(output / "one_second_phase_profile_after.csv", phase_rows_after)
    write_csv(output / "spike_template.csv", [
        {"offset_s": float(offset), "amplitude": float(value)}
        for offset, value in zip(result["template_offsets"], result["template"])
    ])
    write_csv(output / "distance_bin_counts.csv", [{"bin_center_km": (i + 0.5) * 0.5, "pair_count": int(n)} for i, n in enumerate(counts)])


def write_wang_figure3a_report(output, sample_size, candidate_count, used, failures, rows, phase_rows, counts):
    median = float(np.nanmedian([r["contrast"] for r in rows]))
    maximum = max(rows, key=lambda r: r["contrast"])
    best_phase = max(phase_rows, key=lambda row: row["median_abs"])
    text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Wang Figure 3a 风格 1D 尖峰图</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:960px;margin:24px auto;padding:0 18px;color:#172033;line-height:1.65}}img{{max-width:100%;border:1px solid #cbd5e1}}.note{{background:#f8fafc;border:1px solid #cbd5e1;padding:12px}}.warn{{background:#fff7ed;border:1px solid #f59e0b;padding:12px}}</style></head><body><h1>Wang Figure 3a 风格：去除尖峰前</h1><div class='note'>计算主机：<code>{html.escape(platform.node())}</code>。候选唯一无序台站对 {candidate_count}，本轮使用 {used}，读取失败 {failures}。处理方式：每条台站对正延迟 0-16 s 先去均值并按自身最大振幅归一化，再按 0.5 km 台距分箱线性叠加，最后每个距离 bin 归一化绘制。整数秒增强中位数 {median:.2f} 倍，最大在 {maximum['second']} s（{maximum['contrast']:.2f} 倍）；最强 1 s 固定相位为 {best_phase['phase_s']:.2f} s。</div><h2>1. 严格按文献风格的 Figure 3a</h2><img src='wang_figure3a_style_before_spike_removal.png'><h2>2. 固定相位放大图</h2><p>这张图不改数据，只把横轴缩到最强固定相位附近，方便直接检查 <b>{best_phase['phase_s']:.2f} s</b> 附近是否存在跨距离 bin 一致的窄尖峰。</p><img src='wang_figure3a_style_zoom.png'><h2>3. 诊断增强图</h2><div class='warn'>这张图仅用于显示增强，不是文献原图。做法是对每个距离 bin 的波形减去局部慢变化背景，再重新按本 bin 最大振幅归一化。它的作用是压低宽缓面波背景，让可能存在的窄 1 s 尖峰更容易看见。</div><img src='wang_figure3a_style_diagnostic_residual.png'></body></html>"""
    (output / "report.html").write_text(text, encoding="utf-8")
    write_csv(output / "distance_bin_counts.csv", [{"bin_center_km": (i + 0.5) * 0.5, "pair_count": int(n)} for i, n in enumerate(counts)])


def write_daily_ccf_figure3a_report(output, day_count, unique_pair_count, used, failures, rows, phase_rows, counts):
    median = float(np.nanmedian([r["contrast"] for r in rows]))
    maximum = max(rows, key=lambda r: r["contrast"])
    best_phase = max(phase_rows, key=lambda row: row["median_abs"])
    text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Wang Figure 3a 风格：日 CCF 去尖峰前</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:960px;margin:24px auto;padding:0 18px;color:#172033;line-height:1.65}}img{{max-width:100%;border:1px solid #cbd5e1}}.note{{background:#f8fafc;border:1px solid #cbd5e1;padding:12px}}.warn{{background:#fff7ed;border:1px solid #f59e0b;padding:12px}}</style></head><body><h1>Wang Figure 3a 风格：PWS 前日 CCF 去尖峰前</h1><div class='note'>计算主机：<code>{html.escape(platform.node())}</code>。输入为 1D 每日 CCF H5，共 {day_count} 天；参与累加的日 CCF 记录 {used} 条，唯一非自相关台站对 {unique_pair_count} 对，读取/单记录失败 {failures}。处理方式：每条日 CCF 取正延迟 0-16 s，先去均值并按自身最大振幅归一化，再按 0.5 km 台距分箱线性叠加，最后每个距离 bin 归一化绘制。整数秒增强中位数 {median:.2f} 倍，最大在 {maximum['second']} s（{maximum['contrast']:.2f} 倍）；最强 1 s 固定相位为 <b>{best_phase['phase_s']:.2f} s</b>。</div><h2>1. 严格按文献风格的 Figure 3a</h2><img src='wang_figure3a_style_before_spike_removal.png'><h2>2. 固定相位放大图</h2><p>为了和旧报告公平比较，这次仍然使用 <b>daily CCF</b> 输入，但把放大窗改成围绕实际最强固定相位 <b>{best_phase['phase_s']:.2f} s</b>，而不是预设 0.5 s。</p><img src='wang_figure3a_style_zoom.png'><h2>3. 诊断增强图</h2><div class='warn'>这张图仅用于显示增强，不是文献原图。做法是对每个距离 bin 的波形减去局部慢变化背景，再重新按本 bin 最大振幅归一化。它的作用是压低宽缓面波背景，让可能存在的窄 1 s 尖峰更容易看见。</div><img src='wang_figure3a_style_diagnostic_residual.png'></body></html>"""
    (output / "report.html").write_text(text, encoding="utf-8")
    write_csv(output / "distance_bin_counts.csv", [{"bin_center_km": (i + 0.5) * 0.5, "pair_count": int(n)} for i, n in enumerate(counts)])


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT); parser.add_argument("--ccf-root", type=Path, default=DEFAULT_CCF_ROOT); parser.add_argument("--metadata-root", type=Path); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--sample-size", type=int, default=20000); parser.add_argument("--seed", type=int, default=20260619); parser.add_argument("--template-subtract", action="store_true"); parser.add_argument("--template-source", choices=["coherent", "diagnostic"], default="coherent"); parser.add_argument("--wang-figure3a", action="store_true"); parser.add_argument("--daily-ccf-figure3a", action="store_true"); args = parser.parse_args()
    if args.daily_ccf_figure3a:
        files = sorted(args.ccf_root.glob("*.h5"))
        time, stacks, counts, examples, used, failures, unique_pair_count = read_daily_ccf_bins(files)
        rows, phase_rows = make_wang_figure3a_plot(
            args.output,
            time,
            stacks,
            counts,
            0.5,
            zoom_center_s=0.04,
            zoom_width_s=0.20,
        )
        write_daily_ccf_figure3a_report(args.output, len(files), unique_pair_count, used, failures, rows, phase_rows, counts)
        print(args.output / "report.html")
        return
    sample, candidates = reservoir_sample(iter_unique_paths(args.stack_root), args.sample_size, args.seed)
    normalize_each_pair = args.wang_figure3a or args.template_subtract
    time, stacks, counts, examples, used, failures = read_sample(sample, normalize_each_pair=normalize_each_pair, metadata_root=args.metadata_root)
    if args.wang_figure3a:
        rows, phase_rows = make_wang_figure3a_plot(args.output, time, stacks, counts, 0.5)
        write_wang_figure3a_report(args.output, len(sample), candidates, used, failures, rows, phase_rows, counts)
    elif args.template_subtract:
        result = make_template_subtraction_plots(args.output, time, stacks, counts, examples, 0.5, template_source=args.template_source)
        write_template_subtraction_report(args.output, len(sample), candidates, used, failures, result, counts)
    else:
        _, _, rows, phase_rows = make_plots(args.output, time, stacks, counts, examples, 0.5)
        write_report(args.output, len(sample), candidates, used, failures, rows, phase_rows, counts)
    print(args.output / "report.html")


if __name__ == "__main__": main()
