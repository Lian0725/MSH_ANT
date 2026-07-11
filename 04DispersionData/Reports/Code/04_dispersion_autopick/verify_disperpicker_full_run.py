"""Verify the file and status contract of a DisperPicker batch run."""

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np


def pair_from_curve_name(path: Path, prefix: str) -> str:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(".txt"):
        return ""
    return name[len(prefix) : -4]


def collect_pairs(dat_dir: Path, curves_dir: Path, pixels_dir: Path, logs_dir: Path, log_glob: str):
    return {
        "dat": {path.stem for path in dat_dir.glob("*.dat")},
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


def count_log_pattern(logs, pattern: str) -> int:
    count = 0
    for path in logs:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        count += text.count(pattern)
    return count


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


def failure_reason_from_npz(path: Path) -> str:
    """Return a failure marker or an unreadable-file diagnostic."""
    try:
        with zipfile.ZipFile(str(path), "r") as archive:
            has_failure_reason = "failure_reason.npy" in archive.namelist()
        if not has_failure_reason:
            return ""
        with np.load(str(path), allow_pickle=False) as data:
            reason = str(np.asarray(data["failure_reason"]).item()).strip()
        return reason
    except Exception as exc:
        return "unreadable NPZ: {}".format(exc)


def validate_npz_file(path: Path) -> str:
    marker = failure_reason_from_npz(path)
    if marker:
        return "failure_reason={}".format(marker)
    try:
        with np.load(str(path), allow_pickle=False) as data:
            required = {
                "group_image",
                "phase_image",
                "periods",
                "velocities",
                "velocity_axis_km_s",
                "actual_velocity_axis_km_s",
                "snr",
                "distance_km",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                return "missing keys {}".format(",".join(missing))
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
    parser.add_argument("--curves-dir", type=Path)
    parser.add_argument("--pixels-dir", type=Path)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--log-glob", default="shard_*.log")
    parser.add_argument("--expected-shards", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--expected-curve-lines", type=int, default=51)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--require-done-lines", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.expected_shards < 1:
        parser.error("--expected-shards must be >= 1")
    dat_dir, curves_dir, pixels_dir, logs_dir = resolve_layout(args, parser)
    pairs = collect_pairs(dat_dir, curves_dir, pixels_dir, logs_dir, args.log_glob)

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

    failure_marked_npz = []
    for path in pixels_dir.glob("*.npz"):
        marker = failure_reason_from_npz(path)
        if marker:
            failure_marked_npz.append({"file": path.name, "error": marker})

    sample_pairs = sorted(dat_pairs)
    rng = random.Random(20260630)
    if len(sample_pairs) > args.sample_size:
        sample_pairs = sorted(rng.sample(sample_pairs, args.sample_size))

    curve_failures = []
    npz_failures = []
    for pair in sample_pairs:
        for prefix in ("GDisp.", "CDisp."):
            path = curves_dir / "{}{}.txt".format(prefix, pair)
            error = validate_curve_file(path, args.expected_curve_lines)
            if error:
                curve_failures.append({"pair": pair, "file": path.name, "error": error})
        error = validate_npz_file(pixels_dir / "{}.npz".format(pair))
        if error:
            npz_failures.append({"pair": pair, "error": error})

    log_error_count = count_log_pattern(pairs["logs"], "处理失败")
    traceback_count = count_log_pattern(pairs["logs"], "Traceback")
    done_count = count_log_pattern(pairs["logs"], "处理完成")

    report = {
        "directories": {
            "dat": str(dat_dir),
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
            "expected_shards": args.expected_shards,
            "done_lines": done_count,
            "log_processing_failures": log_error_count,
            "tracebacks": traceback_count,
            "zero_byte_files": len(zero_byte),
            "failure_marked_npz": len(failure_marked_npz),
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
        "failure_marked_npz": failure_marked_npz[:100],
        "sample": {
            "checked_pairs": len(sample_pairs),
            "curve_failures": curve_failures[:50],
            "npz_failures": npz_failures[:50],
            "curve_failure_count": len(curve_failures),
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
        and not failure_marked_npz
        and log_error_count == 0
        and traceback_count == 0
        and len(pairs["logs"]) == args.expected_shards
        and done_count == args.expected_shards
        and not curve_failures
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
