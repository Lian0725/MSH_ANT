#!/usr/bin/env python3
"""Run the 2014 1D Wang/Lin hourly cross-correlation and direct PWS stack.

This entry point is designed for the ``work`` server.  It keeps all staging,
checkpoints, logs, and final STACK products under ``/mnt/data_hdd/lgx/MSH_ANT``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wang_1d_pws import (  # noqa: E402
    SECONDS_PER_DAY,
    StreamingPWSAccumulator,
    WangPwsConfig,
    build_day_blocks,
    build_input_manifest,
    canonical_pair,
    choose_worker_count,
    day_window_starts,
    expected_stack_points,
    input_manifest_digest,
    iter_dates,
    maxabs_normalize_rows,
    normalize_station_code,
    read_csv,
    read_source_checkpoint,
    station_day_files,
    station_tail,
    write_csv,
    write_json,
    write_source_checkpoint,
)


WORK_ROOT = Path("/mnt/data_hdd/lgx/MSH_ANT")
RUN_NAME = "1D_WANG_PWS_150s_20260620"
STAGING_ROOT = WORK_ROOT / "staging" / "2014_1D_new_preprocessed"
DEFAULT_DATA_ROOT = STAGING_ROOT / "mseed_25Hz_resp_vel_prefilt_0p005_0p01_10_12"
DEFAULT_XML_ROOT = STAGING_ROOT / "xml"
DEFAULT_OUTPUT_ROOT = WORK_ROOT / "stack" / "2014" / RUN_NAME
DEFAULT_STACK_ROOT = DEFAULT_OUTPUT_ROOT / "STACK"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifest" / "manifest.json"
DEFAULT_LOG_ROOT = DEFAULT_OUTPUT_ROOT / "logs"
DEFAULT_CHECKPOINT_ROOT = DEFAULT_OUTPUT_ROOT / "checkpoints"
DEFAULT_QC_ROOT = DEFAULT_OUTPUT_ROOT / "qc"


def configure_process_environment(fft_threads: int = 4) -> None:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("SCIPY_NUM_THREADS", str(max(1, int(fft_threads))))
    os.environ["MSH_ANT_FFT_THREADS"] = str(max(1, int(fft_threads)))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def next_fast_len(value: int) -> int:
    try:
        from scipy.fft import next_fast_len as scipy_next_fast_len

        return int(scipy_next_fast_len(int(value), real=True))
    except Exception:
        return 1 << (int(value) - 1).bit_length()


def fft_worker_count() -> int:
    try:
        return max(1, int(os.environ.get("MSH_ANT_FFT_THREADS", "1")))
    except ValueError:
        return 1


def rfft(values: np.ndarray, nfft: int) -> np.ndarray:
    try:
        from scipy.fft import rfft as scipy_rfft

        return scipy_rfft(values, n=nfft, workers=fft_worker_count())
    except Exception:
        return np.fft.rfft(values, n=nfft)


def irfft(values: np.ndarray, nfft: int) -> np.ndarray:
    try:
        from scipy.fft import irfft as scipy_irfft

        return scipy_irfft(values, n=nfft, workers=fft_worker_count())
    except Exception:
        return np.fft.irfft(values, n=nfft)


def moving_average(values: np.ndarray, half_width: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    width = max(1, int(half_width) * 2 + 1)
    if width <= 1:
        return data
    kernel = np.ones(width, dtype=np.float64) / float(width)
    padded = np.pad(data, (half_width, half_width), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def cosine_taper(npts: int, fraction: float = 0.05) -> np.ndarray:
    n_taper = int(round(npts * fraction))
    if n_taper <= 1:
        return np.ones(npts, dtype=np.float64)
    taper = np.ones(npts, dtype=np.float64)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_taper)))
    taper[:n_taper] = ramp
    taper[-n_taper:] = ramp[::-1]
    return taper


def prepare_window(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("window contains non-finite samples")
    data = data - float(np.mean(data))
    try:
        from scipy.signal import detrend

        data = detrend(data, type="linear")
    except Exception:
        x = np.arange(data.size, dtype=np.float64)
        slope, intercept = np.polyfit(x, data, 1)
        data = data - (slope * x + intercept)
    return data * cosine_taper(data.size, fraction=0.05)


def whiten_window(values: np.ndarray, config: WangPwsConfig, nfft: int) -> np.ndarray:
    data = prepare_window(values)
    spectrum = rfft(data, nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(config.sampling_rate))
    amp = np.abs(spectrum)
    smooth = moving_average(amp, int(config.smoothspect_N))
    out = np.zeros_like(spectrum, dtype=np.complex128)
    band = (freqs >= float(config.freqmin)) & (freqs <= float(config.freqmax)) & (smooth > 0.0)
    out[band] = spectrum[band] / smooth[band]
    return out.astype(np.complex64)


def read_station_trace(path: Path, config: WangPwsConfig):
    from obspy import read

    stream = read(str(path))
    selected = stream.select(network=config.network, channel=config.channel)
    if len(selected) == 0:
        selected = stream.select(channel=config.channel)
    if len(selected) == 0:
        raise ValueError(f"No {config.network}.*.{config.channel} trace in {path}")
    selected.merge(method=1, fill_value=np.nan)
    if len(selected) != 1:
        selected.merge(method=0, fill_value=np.nan)
    if len(selected) != 1:
        raise ValueError(f"Could not merge to one trace: {path}")
    trace = selected[0]
    if abs(float(trace.stats.sampling_rate) - float(config.sampling_rate)) > 1e-6:
        raise ValueError(
            f"{path} sampling_rate={trace.stats.sampling_rate}, expected {config.sampling_rate}"
        )
    return trace


def compute_station_day_spectra(path: Path, day: str, config: WangPwsConfig) -> tuple[np.ndarray, np.ndarray, int]:
    from obspy import UTCDateTime

    trace = read_station_trace(path, config)
    nwin = int(round(float(config.cc_len) * float(config.sampling_rate)))
    nfft = next_fast_len(2 * nwin - 1)
    starts = day_window_starts(config)
    spectra = np.zeros((len(starts), nfft // 2 + 1), dtype=np.complex64)
    complete = np.zeros(len(starts), dtype=bool)
    day_start = UTCDateTime(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc))
    dt = 1.0 / float(config.sampling_rate)

    for index, offset in enumerate(starts):
        start = day_start + int(offset)
        end = start + float(config.cc_len) - dt
        window = trace.copy().trim(starttime=start, endtime=end, pad=True, fill_value=np.nan)
        data = np.asarray(window.data, dtype=np.float64)
        if data.size != nwin or not np.all(np.isfinite(data)):
            continue
        spectra[index, :] = whiten_window(data, config, nfft)
        complete[index] = True
    return spectra, complete, nfft


def crosscorr_rows_from_spectra(
    source_spectra: np.ndarray,
    receiver_spectra: np.ndarray,
    complete: np.ndarray,
    nfft: int,
    config: WangPwsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    nlag = int(round(float(config.maxlag) * float(config.sampling_rate)))
    rows = []
    row_complete = []
    for index in np.flatnonzero(complete):
        cc = irfft(np.conj(source_spectra[index]) * receiver_spectra[index], nfft)
        cropped = np.concatenate((cc[-nlag:], cc[: nlag + 1]))
        rows.append(cropped.astype(np.float64, copy=False))
        row_complete.append(True)
    if not rows:
        return np.empty((0, 2 * nlag + 1), dtype=np.float64), np.empty(0, dtype=bool)
    return np.asarray(rows, dtype=np.float64), np.asarray(row_complete, dtype=bool)


def build_config_from_manifest(manifest: dict) -> WangPwsConfig:
    payload = dict(manifest.get("config", {}))
    allowed = set(WangPwsConfig.__dataclass_fields__.keys())
    return WangPwsConfig(**{key: value for key, value in payload.items() if key in allowed})


def select_station_codes(manifest: dict, station_limit: int | None = None) -> list[str]:
    codes = sorted(str(row["code"]) for row in manifest.get("stations", []))
    return codes if station_limit is None else codes[: int(station_limit)]


def day_file_map(data_root: Path, stations: list[str]) -> dict[str, dict[str, str]]:
    return {
        station: {day: str(path) for day, path in station_day_files(data_root, station).items()}
        for station in stations
    }


def load_source_state(checkpoint_path: Path, source: str, npts: int, config: WangPwsConfig) -> tuple[dict[str, StreamingPWSAccumulator], set[str]]:
    if not checkpoint_path.exists():
        return {}, set()
    restored = read_source_checkpoint(checkpoint_path)
    receivers = restored.get("receivers", {})
    attrs = restored.get("attrs", {})
    processed_days = set()
    raw_days = attrs.get("processed_days_json")
    if isinstance(raw_days, str) and raw_days:
        try:
            processed_days = set(json.loads(raw_days))
        except json.JSONDecodeError:
            processed_days = set()
    for receiver, accumulator in list(receivers.items()):
        if accumulator.npts != npts:
            raise ValueError(f"{checkpoint_path} receiver {receiver} has npts={accumulator.npts}, expected {npts}")
        accumulator.power = float(config.pws_power)
    return receivers, processed_days


def source_checkpoint_attrs(
    source: str,
    config: WangPwsConfig,
    manifest_hash: str,
    processed_days: set[str],
    failures: list[dict],
) -> dict:
    return {
        "source": source,
        "pass": config.pass_name,
        "manifest_hash": manifest_hash,
        "processed_days_json": json.dumps(sorted(processed_days), separators=(",", ":")),
        "failure_count": int(len(failures)),
        "updated_utc": utc_now(),
        "stack_npts": expected_stack_points(config),
        "pws_power": float(config.pws_power),
        "hourly_normalization": config.hourly_normalization,
    }


def process_source_worker(payload: dict) -> dict:
    configure_process_environment(int(payload.get("fft_threads", 4)))
    manifest_path = Path(payload["manifest_path"])
    checkpoint_root = Path(payload["checkpoint_root"])
    log_root = Path(payload["log_root"])
    data_root = Path(payload["data_root"])
    source = str(payload["source"])
    receivers = [str(item) for item in payload["receivers"]]
    selected_days = [str(item) for item in payload["days"]]
    resume = bool(payload.get("resume", True))
    overwrite = bool(payload.get("overwrite", False))

    manifest = load_json(manifest_path)
    config = build_config_from_manifest(manifest)
    manifest_hash = input_manifest_digest(manifest)
    npts = expected_stack_points(config)
    files = day_file_map(data_root, [source, *receivers])
    checkpoint_path = checkpoint_root / f"{source}.h5"
    failure_log = log_root / "failures" / f"{source}.jsonl"
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()
    accumulators, processed_days = load_source_state(checkpoint_path, source, npts, config) if resume else ({}, set())
    failures: list[dict] = []

    for receiver in receivers:
        accumulators.setdefault(receiver, StreamingPWSAccumulator(npts=npts, power=config.pws_power))

    for day in selected_days:
        if resume and day in processed_days:
            continue
        if day not in files.get(source, {}):
            processed_days.add(day)
            continue
        try:
            source_spectra, source_complete, source_nfft = compute_station_day_spectra(
                Path(files[source][day]), day, config
            )
        except Exception as exc:
            failures.append({"source": source, "receiver": "", "day": day, "error": f"{type(exc).__name__}: {exc}"})
            processed_days.add(day)
            continue

        for receiver in receivers:
            if day not in files.get(receiver, {}):
                continue
            try:
                receiver_spectra, receiver_complete, receiver_nfft = compute_station_day_spectra(
                    Path(files[receiver][day]), day, config
                )
                if receiver_nfft != source_nfft:
                    raise ValueError(f"nfft mismatch source={source_nfft} receiver={receiver_nfft}")
                complete = source_complete & receiver_complete
                rows, row_complete = crosscorr_rows_from_spectra(
                    source_spectra, receiver_spectra, complete, source_nfft, config
                )
                normalized, _, kept = maxabs_normalize_rows(rows, complete_mask=row_complete)
                if normalized.shape[0] > 0:
                    accumulators[receiver].add_rows(normalized)
                if rows.shape[0] > 0 and rows.shape[0] != int(np.count_nonzero(complete)):
                    raise ValueError("internal complete-window accounting mismatch")
                if normalized.shape[0] != int(np.count_nonzero(kept)):
                    raise ValueError("internal normalization accounting mismatch")
            except Exception as exc:
                failures.append(
                    {
                        "source": source,
                        "receiver": receiver,
                        "day": day,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        processed_days.add(day)
        write_source_checkpoint(
            checkpoint_path,
            source=source,
            receivers=accumulators,
            attrs=source_checkpoint_attrs(source, config, manifest_hash, processed_days, failures),
        )

    if failures:
        with failure_log.open("a", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps({**failure, "time": utc_now()}, ensure_ascii=True) + "\n")
    write_source_checkpoint(
        checkpoint_path,
        source=source,
        receivers=accumulators,
        attrs=source_checkpoint_attrs(source, config, manifest_hash, processed_days, failures),
    )
    ngood_total = int(sum(acc.ngood for acc in accumulators.values()))
    return {
        "source": source,
        "receiver_count": len(receivers),
        "processed_day_count": len(processed_days),
        "failure_count": len(failures),
        "ngood_total": ngood_total,
        "checkpoint": str(checkpoint_path),
    }


def bounded_map_sources(tasks: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        results = []
        for task in tasks:
            results.append(process_source_worker(task))
            print(json.dumps(results[-1], ensure_ascii=True), flush=True)
        return results

    results: list[dict] = []
    max_pending = max(workers * 2, workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()
        task_iter = iter(tasks)
        while True:
            while len(pending) < max_pending:
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(process_source_worker, task))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results.append(result)
                print(json.dumps(result, ensure_ascii=True), flush=True)
    return results


def command_manifest(args: argparse.Namespace) -> int:
    config = WangPwsConfig()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_input_manifest(
        args.data_root,
        args.xml_root,
        config,
        compute_sha256=not args.skip_sha256,
    )
    manifest["run_name"] = RUN_NAME
    manifest["output_root"] = str(args.output_root)
    manifest["checkpoint_root"] = str(args.checkpoint_root)
    manifest["stack_root"] = str(args.stack_root)
    manifest["input_hash"] = input_manifest_digest(manifest)
    write_json(args.manifest, manifest)
    write_csv(args.output_root / "manifest" / "stations.csv", manifest["stations"])
    write_csv(
        args.output_root / "manifest" / "pairs.csv",
        manifest["pairs"],
        fieldnames=["pair_name", "source", "receiver", "distance_km"],
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "station_count": manifest["station_count"],
                "pair_count": manifest["pair_count"],
                "file_count": manifest["file_count"],
                "total_size": manifest["total_size"],
                "input_hash": manifest["input_hash"],
            },
            ensure_ascii=True,
        )
    )
    return 0


def build_source_tasks(args: argparse.Namespace, manifest: dict) -> list[dict]:
    config = build_config_from_manifest(manifest)
    all_codes = select_station_codes(manifest, args.station_limit)
    selected_days = [day.strftime("%Y%m%d") for day in iter_dates(config)]
    if args.day_limit is not None:
        selected_days = selected_days[: int(args.day_limit)]
    source_codes = all_codes
    if args.source_station:
        requested = normalize_station_code(args.source_station)
        source_codes = [requested] if requested in all_codes else []
    if args.source_limit is not None:
        source_codes = source_codes[: int(args.source_limit)]

    tasks = []
    pair_budget = args.pair_limit
    for source in source_codes:
        source_index = all_codes.index(source)
        receivers = all_codes[source_index + 1 :]
        if pair_budget is not None:
            remaining = int(pair_budget) - sum(len(task["receivers"]) for task in tasks)
            if remaining <= 0:
                break
            receivers = receivers[:remaining]
        if not receivers:
            continue
        tasks.append(
            {
                "manifest_path": str(args.manifest),
                "checkpoint_root": str(args.checkpoint_root),
                "log_root": str(args.log_root),
                "data_root": str(args.data_root),
                "source": source,
                "receivers": receivers,
                "days": selected_days,
                "resume": bool(args.resume),
                "overwrite": bool(args.overwrite),
                "fft_threads": int(args.fft_threads),
            }
        )
    return tasks


def command_correlate(args: argparse.Namespace) -> int:
    configure_process_environment(args.fft_threads)
    manifest = load_json(args.manifest)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    tasks = build_source_tasks(args, manifest)
    if args.dry_run:
        print(json.dumps({"task_count": len(tasks), "tasks": tasks[:5]}, ensure_ascii=True, indent=2))
        return 0
    workers = int(args.workers or choose_worker_count(os.cpu_count()))
    summary = {
        "started_utc": utc_now(),
        "workers": workers,
        "task_count": len(tasks),
        "checkpoint_root": str(args.checkpoint_root),
        "manifest": str(args.manifest),
    }
    atomic_write_json(args.output_root / "correlate_status.json", summary)
    results = bounded_map_sources(tasks, workers)
    summary.update(
        {
            "finished_utc": utc_now(),
            "results": results,
            "failed_sources": [row for row in results if int(row.get("failure_count", 0)) > 0],
        }
    )
    atomic_write_json(args.output_root / "correlate_status.json", summary)
    return 1 if summary["failed_sources"] else 0


def write_stack_h5(path: Path, trace: np.ndarray, attrs: dict) -> None:
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w") as handle:
        dataset = handle.create_dataset("AuxiliaryData/Allstack_pws/ZZ", data=np.asarray(trace, dtype=np.float32))
        for key, value in attrs.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                dataset.attrs[key] = value
    tmp.replace(path)


def command_export(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    config = build_config_from_manifest(manifest)
    manifest_hash = input_manifest_digest(manifest)
    args.stack_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in sorted(args.checkpoint_root.glob("1D.*.h5")):
        restored = read_source_checkpoint(checkpoint)
        source = str(restored["source"])
        for receiver, accumulator in sorted(restored["receivers"].items()):
            if accumulator.ngood <= 0:
                continue
            trace = accumulator.finalize()
            out = args.stack_root / source / receiver / "stack_pws.h5"
            attrs = {
                **config.as_dict(),
                "station_source": station_tail(source),
                "station_receiver": station_tail(receiver),
                "network": config.network,
                "component": "ZZ",
                "dt": 1.0 / float(config.sampling_rate),
                "maxlag": float(config.maxlag),
                "ngood_hours": int(accumulator.ngood),
                "stack_method": "PWS",
                "pws_power": float(config.pws_power),
                "input_hash": manifest_hash,
                "exported_utc": utc_now(),
            }
            write_stack_h5(out, trace, attrs)
            rows.append(
                {
                    "source": source,
                    "receiver": receiver,
                    "ngood_hours": accumulator.ngood,
                    "stack_path": str(out),
                }
            )
    write_csv(args.output_root / "export_summary.csv", rows, fieldnames=["source", "receiver", "ngood_hours", "stack_path"])
    print(json.dumps({"exported_pairs": len(rows), "stack_root": str(args.stack_root)}, ensure_ascii=True))
    return 0


def audit_windows(config: WangPwsConfig) -> dict:
    starts = day_window_starts(config)
    blocks = build_day_blocks(iter_dates(config)[0], config)
    block_starts = [start for block in blocks for start in block.window_start_seconds]
    return {
        "complete_day_window_count": len(starts),
        "expected_complete_day_window_count": 47,
        "window_starts_unique": len(block_starts) == len(set(block_starts)),
        "block_window_counts": [len(block.window_start_seconds) for block in blocks],
        "first_window_start_s": starts[0] if starts else None,
        "last_window_start_s": starts[-1] if starts else None,
        "stack_points": expected_stack_points(config),
    }


def audit_stacks(stack_root: Path, config: WangPwsConfig) -> dict:
    import h5py

    expected = expected_stack_points(config)
    checked = 0
    bad = []
    for path in sorted(Path(stack_root).glob("1D.*/1D.*/stack_pws.h5")):
        checked += 1
        try:
            with h5py.File(path, "r") as handle:
                data = handle["AuxiliaryData/Allstack_pws/ZZ"]
                if int(data.shape[0]) != expected:
                    bad.append({"path": str(path), "npts": int(data.shape[0])})
        except Exception as exc:
            bad.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {"checked_stack_files": checked, "bad_stack_files": bad}


def command_audit(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    config = build_config_from_manifest(manifest)
    checkpoint_files = sorted(args.checkpoint_root.glob("1D.*.h5"))
    checkpoint_summary = []
    for path in checkpoint_files:
        restored = read_source_checkpoint(path)
        checkpoint_summary.append(
            {
                "source": restored["source"],
                "receiver_count": len(restored["receivers"]),
                "ngood_total": int(sum(acc.ngood for acc in restored["receivers"].values())),
                "path": str(path),
            }
        )
    audit = {
        "created_utc": utc_now(),
        "manifest": str(args.manifest),
        "input_hash": input_manifest_digest(manifest),
        "parameters": config.as_dict(),
        "windows": audit_windows(config),
        "manifest_station_count": manifest.get("station_count"),
        "manifest_pair_count": manifest.get("pair_count"),
        "checkpoint_count": len(checkpoint_files),
        "checkpoints": checkpoint_summary,
        "stacks": audit_stacks(args.stack_root, config) if args.stack_root.exists() else {"checked_stack_files": 0, "bad_stack_files": []},
    }
    args.qc_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.qc_root / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    bad = audit["stacks"]["bad_stack_files"]
    windows = audit["windows"]
    if windows["complete_day_window_count"] != 47 or not windows["window_starts_unique"] or bad:
        return 1
    return 0


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--xml-root", type=Path, default=DEFAULT_XML_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--stack-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--qc-root", type=Path)


def resolve_common_paths(args: argparse.Namespace) -> None:
    args.output_root = Path(args.output_root)
    derived = {
        "checkpoint_root": args.output_root / "checkpoints",
        "stack_root": args.output_root / "STACK",
        "manifest": args.output_root / "manifest" / "manifest.json",
        "log_root": args.output_root / "logs",
        "qc_root": args.output_root / "qc",
    }
    for key, default in derived.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="scan staged data and write manifest")
    add_common_paths(manifest)
    manifest.add_argument("--skip-sha256", action="store_true", help="for smoke tests only")
    manifest.set_defaults(func=command_manifest)

    correlate = subparsers.add_parser("correlate", help="run hourly CCFs and streaming PWS accumulation")
    add_common_paths(correlate)
    correlate.add_argument("--resume", action="store_true", default=False)
    correlate.add_argument("--overwrite", action="store_true")
    correlate.add_argument("--workers", type=positive_int, default=None)
    correlate.add_argument("--fft-threads", type=positive_int, default=4)
    correlate.add_argument("--station-limit", type=positive_int)
    correlate.add_argument("--source-limit", type=positive_int)
    correlate.add_argument("--source-station")
    correlate.add_argument("--pair-limit", type=positive_int)
    correlate.add_argument("--day-limit", type=positive_int)
    correlate.add_argument("--dry-run", action="store_true")
    correlate.set_defaults(func=command_correlate)

    export = subparsers.add_parser("export", help="export checkpoint accumulators to STACK HDF5")
    add_common_paths(export)
    export.set_defaults(func=command_export)

    audit = subparsers.add_parser("audit", help="audit parameters, windows, checkpoints, and stacks")
    add_common_paths(audit)
    audit.set_defaults(func=command_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_common_paths(args)
    for name in ("data_root", "xml_root", "output_root", "checkpoint_root", "stack_root", "manifest", "log_root", "qc_root"):
        if hasattr(args, name):
            setattr(args, name, Path(getattr(args, name)))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
