#!/usr/bin/env python3
"""2014 1D raw waveform preprocessing for MSH_ANT_Final."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from obspy import read, read_inventory

RAW_DIR = Path(
    os.environ.get(
        "MSH_1D_RAW_DIR",
        "/mnt/data_hdd/MSH_ANT_Final/01RawData/2014/WaveData/1D",
    )
)
XML_DIR = Path(
    os.environ.get(
        "MSH_1D_XML_DIR",
        "/mnt/data_hdd/MSH_ANT_Final/01RawData/2014/MetaData/1D",
    )
)
OUT_DIR = Path(
    os.environ.get(
        "MSH_1D_OUT_DIR",
        "/mnt/data_hdd/MSH_ANT_Final/02PreprocessData/2014/WaveData/1D",
    )
)

PRE_FILT = (0.005, 0.01, 10.0, 12.0)
WATER_LEVEL = 60
TARGET_SR = 25.0
OUTPUT_UNIT = "VEL"
OUTPUT_ENCODING = "FLOAT32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess 2014 1D raw MiniSEED with merge -> 25 Hz -> "
            "demean -> detrend -> remove_response(VEL)."
        )
    )
    parser.add_argument(
        "station_file",
        nargs="?",
        type=Path,
        help="Optional text file with one station code per line.",
    )
    return parser.parse_args()


def read_station_list(station_file: Path) -> set[str]:
    stations: set[str] = set()
    for line in station_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stations.add(line)
    return stations


def collect_input_files(raw_dir: Path, station_filter: set[str] | None = None) -> list[Path]:
    files: list[Path] = []
    for mseed_file in sorted(raw_dir.glob("1D.*/*.mseed")):
        if station_filter and mseed_file.parent.name not in station_filter:
            continue
        files.append(mseed_file)
    return files


def downsample_to_25hz(stream):
    out = stream.copy()
    for trace in out:
        sr = float(trace.stats.sampling_rate)
        if abs(sr - TARGET_SR) < 1e-6:
            continue
        ratio = sr / TARGET_SR
        nearest = int(round(ratio))
        if nearest > 1 and abs(ratio - nearest) < 1e-6:
            trace.decimate(nearest, no_filter=False, strict_length=False)
        else:
            trace.resample(TARGET_SR, no_filter=False)
    return out


def process_one_file(mseed_file: Path) -> None:
    station = mseed_file.parent.name
    xml_file = XML_DIR / f"{station}.xml"
    out_file = OUT_DIR / station / mseed_file.name

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"SKIP  {out_file}")
        return

    if not xml_file.exists():
        print(f"MISS  {xml_file}")
        return

    stream = read(str(mseed_file))
    inventory = read_inventory(str(xml_file))

    for trace in stream:
        trace.data = np.asarray(trace.data, dtype=np.float64)

    stream.merge(method=1, fill_value="interpolate")
    stream = downsample_to_25hz(stream)
    stream.detrend("demean")
    stream.detrend("linear")
    stream.remove_response(
        inventory=inventory,
        output=OUTPUT_UNIT,
        pre_filt=PRE_FILT,
        water_level=WATER_LEVEL,
        plot=False,
    )

    for trace in stream:
        trace.data = np.asarray(trace.data, dtype=np.float32)
        trace.stats.mseed = trace.stats.get("mseed", {})
        trace.stats.mseed["encoding"] = OUTPUT_ENCODING

    out_file.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(out_file), format="MSEED")
    print(f"OK    {out_file}")


def main() -> None:
    args = parse_args()
    station_filter = read_station_list(args.station_file) if args.station_file else None

    files = collect_input_files(RAW_DIR, station_filter)
    print(f"Found {len(files)} files")
    print(f"Input raw dir: {RAW_DIR}")
    print(f"Input xml dir: {XML_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print("Pipeline: merge -> downsample(25 Hz) -> demean -> detrend -> remove_response(VEL)")
    print(f"pre_filt={PRE_FILT}")
    print(f"water_level={WATER_LEVEL}")
    print(f"output_encoding={OUTPUT_ENCODING}")
    if station_filter:
        print(f"Station filter: {len(station_filter)} stations")

    for index, mseed_file in enumerate(files, start=1):
        try:
            process_one_file(mseed_file)
        except Exception as exc:
            print(f"FAIL  {mseed_file}  {exc}")

        if index % 100 == 0 or index == len(files):
            print(f"Progress {index}/{len(files)}")


if __name__ == "__main__":
    main()
