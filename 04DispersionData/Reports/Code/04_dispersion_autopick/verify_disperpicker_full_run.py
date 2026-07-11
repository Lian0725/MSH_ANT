"""Verify the file and status contract of a DisperPicker batch run."""

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np


REQUIRED_NPZ_KEYS = {
    "group_image",
    "phase_image",
    "periods",
    "velocities",
    "velocity_axis_km_s",
    "actual_velocity_axis_km_s",
    "snr",
    "distance_km",
}


def pair_from_curve_name(path: Path, prefix: str) -> str:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(".txt"):
        return ""
    return name[len(prefix) : -4]


def collect_pairs(
    dat_dir: Path,
    dat_glob: str,
    curves_dir: Path,
    pixels_dir: Path,
    logs_dir: Path,
    log_glob: str,
):
    return {
        "dat": {path.stem for path in dat_dir.glob(dat_glob)},
        "g": {
            pair_from_curve_name(path, "GDisp.")
            for path in curves_dir.glob("GDisp.*.txt")
        },
        "c": {
            pair_from_curve_name(path, "CDisp.")
            for path in curves_dir.glob("CDisp.*.txt")
        },
        "npz": {path.stem for path in pixels_dir.glob("*.npz")},
        "logs": sorted(logs_dir.glob(log_glob)),
    }


def count_log_patterns(logs, patterns):
    counts = {pattern: 0 for pattern in patterns}
    for path in logs:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in patterns:
            counts[pattern] += text.count(pattern)
    return counts


def validate_curve_file(path: Path, expected_lines: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        return str(exc)
    if len(lines) != expected_lines:
        return "line_count={} expected={}".format(len(lines), expected_lines)
    for index, line in enumerate(lines[2:], start=3):
        parts = line.split()
        if len(parts) != 4:
            return "line {} has {} columns".format(index, len(parts))
        try:
            [float(value) for value in parts]
        except ValueError:
            return "line {} contains non-numeric values".format(index)
    return ""


def npz_container_error(path: Path) -> str:
    """Return a structural/failure-marker error without loading image arrays."""
    try:
        with zipfile.ZipFile(str(path), "r") as archive:
            members = set(archive.namelist())
        missing = sorted(
            key for key in REQUIRED_NPZ_KEYS if "{}.npy".format(key) not in members
        )
        if missing:
            return "missing keys {}".format(",".join(missing))
        has_failure_reason = "failure_reason.npy" in members
        if not has_failure_reason:
            return ""
        with np.load(str(path), allow_pickle=False) as data:
            reason = str(np.asarray(data["failure_reason"]).item()).strip()
        return reason
    except Exception as exc:
        return "unreadable NPZ: {}".format(exc)


def validate_npz_file(path: Path) -> str:
    marker = npz_container_error(path)
    if marker:
        return "failure_reason={}".format(marker)
    try:
        with np.load(str(path), allow_pickle=False) as data:
            if data["group_image"].shape != (701, 49):
                return "group_image shape {}".format(data["group_image"].shape)
            if data["phase_image"].shape != (701, 49):
                return "phase_image shape {}".format(data["phase_image"].shape)
            if data["periods"].shape != (49,):
                return "periods shape {}".format(data["periods"].shape)
    except Exception as exc:
        return str(exc)
    return ""


def resolve_layout(args, parser):
    if args.output_dir is not None:
        dat_dir = args.dat_dir or args.output_dir / "dat_ge10"
        curves_dir = args.curves_dir or args.output_dir / "curves_ge10"
        pixels_dir = args.pixels_dir or args.output_dir / "full_pixel_data_ge10"
        logs_dir = args.logs_dir or args.output_dir / "logs"
        return dat_dir, curves_dir, pixels_dir, logs_dir

    explicit = (args.dat_dir, args.curves_dir, args.pixels_dir, args.logs_dir)
    if any(path is None for path in explicit):
        parser.error(
            "provide --output-dir or all of --dat-dir, --curves-dir, --pixels-dir, --logs-dir"
        )
    return explicit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify Wang DisperPicker batch outputs.")
    parser.add_argument("--output-dir", type=Path, help="Legacy run root.")
    parser.add_argument("--dat-dir", type=Path)
    parser.add_argument("--dat-glob", default="*.dat")
    parser.add_argument("--curves-dir", type=Path)
    parser.add_argument("--pixels-dir", type=Path)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--log-glob")
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--expected-curve-lines", type=int, default=51)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--require-done-lines", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    legacy_layout = args.output_dir is not None
    expected_shards = args.expected_shards
    if expected_shards is None:
        expected_shards = 24 if legacy_layout else 16
    log_glob = args.log_glob
    if log_glob is None:
        log_glob = "dispersion24_shard_*_of_24.log" if legacy_layout else "shard_*.log"
    if expected_shards < 1:
        parser.error("--expected-shards must be >= 1")
    dat_dir, curves_dir, pixels_dir, logs_dir = resolve_layout(args, parser)
    pairs = collect_pairs(
        dat_dir,
        args.dat_glob,
        curves_dir,
        pixels_dir,
        logs_dir,
        log_glob,
    )

    dat_pairs = pairs["dat"]
    missing_g = sorted(dat_pairs.difference(pairs["g"]))
    missing_c = sorted(dat_pairs.difference(pairs["c"]))
    missing_npz = sorted(dat_pairs.difference(pairs["npz"]))
    extra_g = sorted(pairs["g"].difference(dat_pairs))
    extra_c = sorted(pairs["c"].difference(dat_pairs))
    extra_npz = sorted(pairs["npz"].difference(dat_pairs))

    zero_byte = []
    for directory, pattern in (
        (dat_dir, "*.dat"),
        (curves_dir, "*.txt"),
        (pixels_dir, "*.npz"),
    ):
        for path in directory.glob(pattern):
            try:
                if path.stat().st_size == 0:
                    zero_byte.append(str(path))
            except OSError:
                zero_byte.append(str(path))

    invalid_npz = []
    invalid_npz_count = 0
    for path in pixels_dir.glob("*.npz"):
        error = npz_container_error(path)
        if error:
            invalid_npz_count += 1
            if len(invalid_npz) < 100:
                invalid_npz.append({"file": path.name, "error": error})

    curve_failures = []
    curve_failure_count = 0
    curve_groups = (
        ("GDisp.", curves_dir.glob("GDisp.*.txt")),
        ("CDisp.", curves_dir.glob("CDisp.*.txt")),
    )
    for prefix, paths in curve_groups:
        for path in paths:
            error = validate_curve_file(path, args.expected_curve_lines)
            if error:
                curve_failure_count += 1
                if len(curve_failures) < 50:
                    curve_failures.append(
                        {
                            "pair": pair_from_curve_name(path, prefix),
                            "file": path.name,
                            "error": error,
                        }
                    )

    sample_pairs = sorted(dat_pairs)
    rng = random.Random(20260630)
    if len(sample_pairs) > args.sample_size:
        sample_pairs = sorted(rng.sample(sample_pairs, args.sample_size))

    npz_failures = []
    for pair in sample_pairs:
        error = validate_npz_file(pixels_dir / "{}.npz".format(pair))
        if error:
            npz_failures.append({"pair": pair, "error": error})

    log_counts = count_log_patterns(pairs["logs"], ("处理失败", "Traceback", "处理完成"))
    log_error_count = log_counts["处理失败"]
    traceback_count = log_counts["Traceback"]
    done_count = log_counts["处理完成"]

    report = {
        "directories": {
            "dat": str(dat_dir),
            "dat_glob": args.dat_glob,
            "curves": str(curves_dir),
            "pixels": str(pixels_dir),
            "logs": str(logs_dir),
        },
        "counts": {
            "dat": len(pairs["dat"]),
            "g": len(pairs["g"]),
            "c": len(pairs["c"]),
            "npz": len(pairs["npz"]),
            "logs": len(pairs["logs"]),
            "expected_shards": expected_shards,
            "done_lines": done_count,
            "log_processing_failures": log_error_count,
            "tracebacks": traceback_count,
            "zero_byte_files": len(zero_byte),
            "invalid_npz": invalid_npz_count,
        },
        "missing": {
            "g": missing_g[:100],
            "c": missing_c[:100],
            "npz": missing_npz[:100],
            "g_count": len(missing_g),
            "c_count": len(missing_c),
            "npz_count": len(missing_npz),
        },
        "extra": {
            "g_count": len(extra_g),
            "c_count": len(extra_c),
            "npz_count": len(extra_npz),
        },
        "invalid_npz": invalid_npz,
        "sample": {
            "checked_pairs": len(sample_pairs),
            "curve_failures": curve_failures[:50],
            "npz_failures": npz_failures[:50],
            "curve_failure_count": curve_failure_count,
            "npz_failure_count": len(npz_failures),
        },
    }
    ok = (
        bool(dat_pairs)
        and not missing_g
        and not missing_c
        and not missing_npz
        and not extra_g
        and not extra_c
        and not extra_npz
        and not zero_byte
        and invalid_npz_count == 0
        and log_error_count == 0
        and traceback_count == 0
        and len(pairs["logs"]) == expected_shards
        and done_count == expected_shards
        and curve_failure_count == 0
        and not npz_failures
    )
    report["ok"] = bool(ok)

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
