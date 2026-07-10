"""Verify a full DisperPicker run output directory.

This checks the file-level contract for a batch run:
  - every DAT pair has GDisp, CDisp, and NPZ outputs
  - run logs are free of processing failures/tracebacks
  - sampled curve and NPZ files are readable and have the expected shape
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np


def pair_from_curve_name(path: Path, prefix: str) -> str:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(".txt"):
        return ""
    return name[len(prefix) : -4]


def collect_pairs(output_dir: Path):
    dat_dir = output_dir / "dat_ge10"
    curves_dir = output_dir / "curves_ge10"
    pixels_dir = output_dir / "full_pixel_data_ge10"
    logs_dir = output_dir / "logs"
    return {
        "dat": {p.stem for p in dat_dir.glob("*.dat")},
        "g": {pair_from_curve_name(p, "GDisp.") for p in curves_dir.glob("GDisp.*.txt")},
        "c": {pair_from_curve_name(p, "CDisp.") for p in curves_dir.glob("CDisp.*.txt")},
        "npz": {p.stem for p in pixels_dir.glob("*.npz")},
        "logs": sorted(logs_dir.glob("dispersion24_shard_*_of_24.log")),
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


def validate_npz_file(path: Path) -> str:
    try:
        with np.load(str(path)) as data:
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
            if data["periods"].shape[0] != 49:
                return "periods length {}".format(data["periods"].shape[0])
    except Exception as exc:
        return str(exc)
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify Wang DisperPicker full-run outputs.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--expected-curve-lines", type=int, default=51)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--require-done-lines",
        action="store_true",
        help="Require all 24 shard logs to contain a final completion line.",
    )
    args = parser.parse_args(argv)

    pairs = collect_pairs(args.output_dir)
    dat_pairs = pairs["dat"]
    missing_g = sorted(dat_pairs.difference(pairs["g"]))
    missing_c = sorted(dat_pairs.difference(pairs["c"]))
    missing_npz = sorted(dat_pairs.difference(pairs["npz"]))
    extra_g = sorted(pairs["g"].difference(dat_pairs))
    extra_c = sorted(pairs["c"].difference(dat_pairs))
    extra_npz = sorted(pairs["npz"].difference(dat_pairs))

    zero_byte = []
    for subdir, pattern in [
        ("curves_ge10", "*.txt"),
        ("full_pixel_data_ge10", "*.npz"),
        ("dat_ge10", "*.dat"),
    ]:
        for path in (args.output_dir / subdir).glob(pattern):
            try:
                if path.stat().st_size == 0:
                    zero_byte.append(str(path))
            except OSError:
                zero_byte.append(str(path))

    sample_pairs = sorted(dat_pairs)
    rng = random.Random(20260630)
    if len(sample_pairs) > args.sample_size:
        sample_pairs = sorted(rng.sample(sample_pairs, args.sample_size))

    curve_failures = []
    npz_failures = []
    curves_dir = args.output_dir / "curves_ge10"
    pixels_dir = args.output_dir / "full_pixel_data_ge10"
    for pair in sample_pairs:
        for prefix in ["GDisp.", "CDisp."]:
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
        "output_dir": str(args.output_dir),
        "counts": {
            "dat": len(pairs["dat"]),
            "g": len(pairs["g"]),
            "c": len(pairs["c"]),
            "npz": len(pairs["npz"]),
            "logs": len(pairs["logs"]),
            "done_lines": done_count,
            "require_done_lines": args.require_done_lines,
            "log_processing_failures": log_error_count,
            "tracebacks": traceback_count,
            "zero_byte_files": len(zero_byte),
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
        "sample": {
            "checked_pairs": len(sample_pairs),
            "curve_failures": curve_failures[:50],
            "npz_failures": npz_failures[:50],
            "curve_failure_count": len(curve_failures),
            "npz_failure_count": len(npz_failures),
        },
    }
    ok = (
        not missing_g
        and not missing_c
        and not missing_npz
        and not extra_g
        and not extra_c
        and not extra_npz
        and not zero_byte
        and log_error_count == 0
        and traceback_count == 0
        and len(pairs["logs"]) == 24
        and (not args.require_done_lines or done_count == 24)
        and not curve_failures
        and not npz_failures
    )
    report["ok"] = bool(ok)

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
