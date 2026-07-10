#!/usr/bin/env python3
"""Build a moveout NPZ cache from an already-generated before/after result folder."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


MODULE_PATH = Path(__file__).resolve().with_name("apply_spike_removal_moveout_compare.py")
LEGACY_PREFIX = "/mnt/data_hdd/lgx/MSH_ANT/"
SYNC_PREFIX = "/mnt/data_hdd/MSH_ANT/"


def load_moveout_module():
    spec = importlib.util.spec_from_file_location("moveout_compare_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plotting module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def infer_orientation(path: Path, source_code: str) -> str:
    left = path.parent.parent.name
    right = path.parent.name
    if left == source_code:
        return "source_receiver"
    if right == source_code:
        return "receiver_source"
    raise RuntimeError(f"Could not infer orientation for {path} with source {source_code}")


def resolve_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    text = str(path)
    if text.startswith(LEGACY_PREFIX):
        fallback = Path(SYNC_PREFIX + text[len(LEGACY_PREFIX) :])
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Could not resolve existing path: {path_text}")


def load_windowed_row(
    module,
    *,
    h5_path: Path,
    source_code: str,
    receiver: str,
    distance_km: float,
    receiver_latitude: float,
    receiver_longitude: float,
    lag_window_s: float,
) -> dict[str, object]:
    orientation = infer_orientation(h5_path, source_code)
    with h5py.File(h5_path, "r") as handle:
        dataset = handle["AuxiliaryData/Allstack_pws/ZZ"]
        data = np.asarray(dataset[:], dtype=np.float64).squeeze()
        if orientation == "receiver_source":
            data = data[::-1].copy()
        dt = float(dataset.attrs["dt"])
        maxlag = float(dataset.attrs["maxlag"])
        time_s, window_trace = module.FIG2.extract_lag_window(
            data,
            dt=dt,
            maxlag=maxlag,
            window_s=lag_window_s,
        )
    return {
        "source": source_code,
        "receiver": receiver,
        "time_s": time_s,
        "window_trace": window_trace,
        "distance_km": distance_km,
        "receiver_latitude": receiver_latitude,
        "receiver_longitude": receiver_longitude,
        "dt": dt,
        "maxlag": maxlag,
        "path": str(h5_path),
        "pair_orientation": orientation,
    }


def load_rows_from_existing_result(
    module,
    *,
    existing_output: Path,
    source_code: str,
    lag_window_s: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    receivers_csv = existing_output / "receivers_used.csv"
    scales_csv = existing_output / "spike_scales.csv"
    if not receivers_csv.exists():
        raise FileNotFoundError(f"Missing receivers CSV: {receivers_csv}")
    if not scales_csv.exists():
        raise FileNotFoundError(f"Missing scales CSV: {scales_csv}")

    scale_by_receiver: dict[str, dict[str, str]] = {}
    with scales_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scale_by_receiver[str(row["receiver"])] = row

    rows_before: list[dict[str, object]] = []
    rows_after: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    with receivers_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            receiver = str(row["receiver"])
            scale_row = scale_by_receiver.get(receiver)
            if scale_row is None:
                continue
            distance_km = float(row["distance_km"])
            receiver_latitude = float(row["receiver_latitude"])
            receiver_longitude = float(row["receiver_longitude"])
            original_path = resolve_existing_path(str(scale_row["original_path"]))
            cleaned_path = resolve_existing_path(str(scale_row["cleaned_path"]))
            rows_before.append(
                load_windowed_row(
                    module,
                    h5_path=original_path,
                    source_code=source_code,
                    receiver=receiver,
                    distance_km=distance_km,
                    receiver_latitude=receiver_latitude,
                    receiver_longitude=receiver_longitude,
                    lag_window_s=lag_window_s,
                )
            )
            rows_after.append(
                load_windowed_row(
                    module,
                    h5_path=cleaned_path,
                    source_code=source_code,
                    receiver=receiver,
                    distance_km=distance_km,
                    receiver_latitude=receiver_latitude,
                    receiver_longitude=receiver_longitude,
                    lag_window_s=lag_window_s,
                )
            )
            scale_rows.append(
                {
                    "source": source_code,
                    "receiver": receiver,
                    "distance_km": f"{distance_km:.6f}",
                    "original_path": str(original_path),
                    "cleaned_path": str(cleaned_path),
                    "scale": str(scale_row["scale"]),
                }
            )
    rows_before.sort(key=lambda item: (float(item["distance_km"]), str(item["receiver"])))
    rows_after.sort(key=lambda item: (float(item["distance_km"]), str(item["receiver"])))
    scale_rows.sort(key=lambda item: (float(item["distance_km"]), str(item["receiver"])))
    return rows_before, rows_after, scale_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-output", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--source", default="1D.4529")
    parser.add_argument("--source-lat", type=float, default=46.141102)
    parser.add_argument("--source-lon", type=float, default=-122.329498)
    parser.add_argument("--minlat", type=float, default=46.1384)
    parser.add_argument("--maxlat", type=float, default=46.1595)
    parser.add_argument("--minlon", type=float, default=-122.3363)
    parser.add_argument("--maxlon", type=float, default=-122.0297)
    parser.add_argument("--lag-window", type=float, default=15.0)
    parser.add_argument("--template-source", default="diagnostic")
    parser.add_argument("--phase-s", type=float, default=0.0)
    parser.add_argument("--used", type=int, default=402309)
    parser.add_argument("--candidate-count", type=int, default=402423)
    parser.add_argument("--original-stack-root", type=Path, default=Path("/mnt/data_hdd/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK"))
    parser.add_argument("--cleaned-stack-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_moveout_module()
    cleaned_stack_root = args.cleaned_stack_root or (args.existing_output / "stack_cleaned_subset")
    rows_before, rows_after, scale_rows = load_rows_from_existing_result(
        module,
        existing_output=args.existing_output,
        source_code=args.source,
        lag_window_s=float(args.lag_window),
    )
    if not rows_before or not rows_after:
        raise RuntimeError("No rows were loaded from the existing result folder")
    module.save_moveout_cache(
        args.cache_file,
        source_code=args.source,
        source_coords={"latitude": float(args.source_lat), "longitude": float(args.source_lon)},
        bbox=module.FIG2.BBox(
            minlat=float(args.minlat),
            maxlat=float(args.maxlat),
            minlon=float(args.minlon),
            maxlon=float(args.maxlon),
        ),
        rows_before=rows_before,
        rows_after=rows_after,
        model={
            "template_source": str(args.template_source),
            "best_phase_s": float(args.phase_s),
            "used": int(args.used),
            "candidate_count": int(args.candidate_count),
            "offsets": np.zeros((0,), dtype=float),
            "template": np.zeros((0,), dtype=float),
        },
        scale_rows=scale_rows,
        original_stack_root=args.original_stack_root,
        cleaned_stack_root=cleaned_stack_root,
        lag_window_s=float(args.lag_window),
    )
    print(args.cache_file)


if __name__ == "__main__":
    main()
