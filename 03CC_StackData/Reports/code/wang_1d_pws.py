#!/usr/bin/env python3
"""Core helpers for the 2014 1D Wang/Lin hourly CCF -> PWS run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


STATION_NS = {"s": "http://www.fdsn.org/xml/station/1"}
SECONDS_PER_DAY = 24 * 3600


@dataclass(frozen=True)
class WangPwsConfig:
    network: str = "1D"
    channel: str = "DPZ"
    start_date: str = "2014-07-18"
    end_date: str = "2014-08-05"
    sampling_rate: float = 25.0
    cc_len: int = 3600
    step: int = 1800
    inc_hours: int = 6
    maxlag: int = 150
    freqmin: float = 0.2
    freqmax: float = 10.0
    freq_norm: str = "RMA"
    smoothspect_N: int = 40
    time_norm: str = "NO"
    smooth_N: int = 40
    cc_method: str = "XCORR"
    rm_resp: str = "NO"
    max_over_std: int = 90
    substack: bool = True
    substack_windows: int = 1
    pws_power: float = 2.0
    stack_method: str = "PWS"
    hourly_normalization: str = "post_cc_maxabs"
    pass_name: str = "uncorrected"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DayBlock:
    day_start: datetime
    core_start: datetime
    core_end: datetime
    read_start: datetime
    read_end: datetime
    window_start_seconds: tuple[int, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "day_start": self.day_start.isoformat(),
            "core_start": self.core_start.isoformat(),
            "core_end": self.core_end.isoformat(),
            "read_start": self.read_start.isoformat(),
            "read_end": self.read_end.isoformat(),
            "window_start_seconds": list(self.window_start_seconds),
        }


@dataclass(frozen=True)
class StationInfo:
    code: str
    lat: float
    lon: float
    day_count: int
    xml_path: str


def normalize_station_code(code: str, network: str = "1D") -> str:
    value = str(code).strip()
    return value if value.startswith(f"{network}.") else f"{network}.{value}"


def station_tail(code: str) -> str:
    return str(code).split(".")[-1]


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((normalize_station_code(left), normalize_station_code(right))))


def pair_name(left: str, right: str) -> str:
    source, receiver = canonical_pair(left, right)
    return f"{source}__{receiver}"


def expected_stack_points(config: WangPwsConfig) -> int:
    dt = 1.0 / float(config.sampling_rate)
    return 2 * int(round(float(config.maxlag) / dt)) + 1


def day_window_starts(config: WangPwsConfig) -> list[int]:
    last_start = SECONDS_PER_DAY - int(config.cc_len)
    if last_start < 0:
        return []
    return list(range(0, last_start + 1, int(config.step)))


def _ensure_utc_day(day_start: datetime) -> datetime:
    if day_start.tzinfo is None:
        return day_start.replace(tzinfo=timezone.utc)
    return day_start.astimezone(timezone.utc)


def build_day_blocks(day_start: datetime, config: WangPwsConfig) -> list[DayBlock]:
    day_start = _ensure_utc_day(day_start)
    day_end = day_start + timedelta(days=1)
    block_seconds = int(config.inc_hours) * 3600
    if block_seconds <= 0:
        raise ValueError("inc_hours must be positive")

    starts = day_window_starts(config)
    blocks: list[DayBlock] = []
    for core_offset in range(0, SECONDS_PER_DAY, block_seconds):
        core_start = day_start + timedelta(seconds=core_offset)
        core_end = min(day_end, core_start + timedelta(seconds=block_seconds))
        read_end = min(day_end, core_end + timedelta(seconds=int(config.step)))
        block_window_starts = tuple(
            offset
            for offset in starts
            if core_offset <= offset < core_offset + block_seconds
            and offset + int(config.cc_len) <= SECONDS_PER_DAY
        )
        blocks.append(
            DayBlock(
                day_start=day_start,
                core_start=core_start,
                core_end=core_end,
                read_start=core_start,
                read_end=read_end,
                window_start_seconds=block_window_starts,
            )
        )
    return blocks


def parse_yyyymmdd(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def iter_dates(config: WangPwsConfig) -> list[datetime]:
    start = parse_yyyymmdd(config.start_date)
    end = parse_yyyymmdd(config.end_date)
    days: list[datetime] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def analytic_signal(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    npts = data.size
    spectrum = np.fft.fft(data)
    weights = np.zeros(npts, dtype=np.float64)
    if npts % 2 == 0:
        weights[0] = 1.0
        weights[npts // 2] = 1.0
        weights[1 : npts // 2] = 2.0
    else:
        weights[0] = 1.0
        weights[1 : (npts + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * weights)


def phase_unit(values: np.ndarray) -> np.ndarray:
    analytic = analytic_signal(values)
    amp = np.abs(analytic)
    out = np.zeros_like(analytic, dtype=np.complex128)
    mask = np.isfinite(amp) & (amp > 0.0)
    out[mask] = analytic[mask] / amp[mask]
    return out


def pws_stack(rows: np.ndarray, power: float = 2.0) -> np.ndarray:
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("rows must be a non-empty 2D array")
    linear = np.mean(data, axis=0)
    phase = np.vstack([phase_unit(row) for row in data])
    weight = np.abs(np.mean(phase, axis=0)) ** float(power)
    return linear * weight


def maxabs_normalize_rows(
    rows: np.ndarray,
    complete_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("rows must be a 2D array")
    finite_rows = np.all(np.isfinite(values), axis=1)
    if complete_mask is None:
        complete = np.ones(values.shape[0], dtype=bool)
    else:
        complete = np.asarray(complete_mask, dtype=bool)
        if complete.shape != (values.shape[0],):
            raise ValueError("complete_mask must have one entry per row")
    safe = np.where(np.isfinite(values), values, 0.0)
    scales_all = np.max(np.abs(safe), axis=1)
    kept = complete & finite_rows & np.isfinite(scales_all) & (scales_all > 0.0)
    normalized = values[kept] / scales_all[kept, None]
    return normalized, scales_all[kept], kept


class StreamingPWSAccumulator:
    def __init__(self, npts: int, power: float = 2.0):
        if int(npts) <= 0:
            raise ValueError("npts must be positive")
        self.npts = int(npts)
        self.power = float(power)
        self.linear_sum = np.zeros(self.npts, dtype=np.float64)
        self.phase_sum = np.zeros(self.npts, dtype=np.complex128)
        self.ngood = 0

    def add_rows(self, rows: np.ndarray) -> None:
        data = np.asarray(rows, dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2 or data.shape[1] != self.npts:
            raise ValueError(f"rows must have shape (n, {self.npts})")
        for row in data:
            if not np.all(np.isfinite(row)):
                continue
            self.linear_sum += row
            self.phase_sum += phase_unit(row)
            self.ngood += 1

    def finalize(self) -> np.ndarray:
        if self.ngood <= 0:
            raise ValueError("No valid hourly CCF rows accumulated")
        linear = self.linear_sum / float(self.ngood)
        weight = np.abs(self.phase_sum / float(self.ngood)) ** self.power
        return linear * weight


def _h5_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_source_checkpoint(
    path: Path,
    source: str,
    receivers: dict[str, StreamingPWSAccumulator],
    attrs: dict,
) -> None:
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with h5py.File(tmp_path, "w") as handle:
            handle.attrs["source"] = str(source)
            for key, value in attrs.items():
                if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                    handle.attrs[str(key)] = value
            root = handle.create_group("receivers")
            for receiver, accumulator in sorted(receivers.items()):
                group = root.create_group(str(receiver))
                group.create_dataset("linear_sum", data=accumulator.linear_sum.astype(np.float64))
                group.create_dataset("phase_sum_real", data=accumulator.phase_sum.real.astype(np.float64))
                group.create_dataset("phase_sum_imag", data=accumulator.phase_sum.imag.astype(np.float64))
                group.attrs["ngood"] = int(accumulator.ngood)
                group.attrs["npts"] = int(accumulator.npts)
                group.attrs["power"] = float(accumulator.power)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_source_checkpoint(path: Path) -> dict:
    import h5py

    path = Path(path)
    receivers: dict[str, StreamingPWSAccumulator] = {}
    with h5py.File(path, "r") as handle:
        source = _h5_scalar(handle.attrs["source"])
        attrs = {key: _h5_scalar(value) for key, value in handle.attrs.items() if key != "source"}
        root = handle.get("receivers")
        if root is not None:
            for receiver in root.keys():
                group = root[receiver]
                npts = int(group.attrs["npts"])
                acc = StreamingPWSAccumulator(npts=npts, power=float(group.attrs["power"]))
                acc.linear_sum = np.asarray(group["linear_sum"][:], dtype=np.float64)
                phase_real = np.asarray(group["phase_sum_real"][:], dtype=np.float64)
                phase_imag = np.asarray(group["phase_sum_imag"][:], dtype=np.float64)
                acc.phase_sum = phase_real + 1j * phase_imag
                acc.ngood = int(group.attrs["ngood"])
                receivers[str(receiver)] = acc
    return {"source": source, "attrs": attrs, "receivers": receivers}


def choose_worker_count(cpu_count: int | None = None) -> int:
    count = int(cpu_count or (os.cpu_count() or 1))
    return max(1, min(24, count - 6))


def parse_station_xml(xml_file: Path, network: str = "1D") -> tuple[str, float, float]:
    root = ET.parse(xml_file).getroot()
    station = root.find(".//s:Station", STATION_NS)
    if station is None:
        raise ValueError(f"No Station element in {xml_file}")
    code = normalize_station_code(station.attrib["code"], network=network)
    lat = float(station.findtext("s:Latitude", namespaces=STATION_NS))
    lon = float(station.findtext("s:Longitude", namespaces=STATION_NS))
    return code, lat, lon


def station_day_files(data_root: Path, station: str) -> dict[str, Path]:
    station_dir = Path(data_root) / normalize_station_code(station)
    return {
        path.stem: path
        for path in sorted(station_dir.glob("*.mseed"))
        if not path.name.startswith("._") and len(path.stem) == 8 and path.stem.isdigit()
    }


def collect_station_infos(data_root: Path, xml_root: Path, config: WangPwsConfig) -> list[StationInfo]:
    infos: list[StationInfo] = []
    for xml_file in sorted(Path(xml_root).glob(f"{config.network}.*.xml")):
        code, lat, lon = parse_station_xml(xml_file, network=config.network)
        day_count = len(station_day_files(data_root, code))
        if (Path(data_root) / code).is_dir() and day_count > 0:
            infos.append(
                StationInfo(
                    code=code,
                    lat=lat,
                    lon=lon,
                    day_count=day_count,
                    xml_path=str(xml_file),
                )
            )
    return infos


def haversine_km(left: StationInfo, right: StationInfo) -> float:
    radius = 6371.0
    lat1 = math.radians(left.lat)
    lon1 = math.radians(left.lon)
    lat2 = math.radians(right.lat)
    lon2 = math.radians(right.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def build_pair_rows(stations: list[StationInfo]) -> list[dict]:
    rows: list[dict] = []
    for index, left in enumerate(stations):
        for right in stations[index + 1 :]:
            source, receiver = canonical_pair(left.code, right.code)
            rows.append(
                {
                    "pair_name": pair_name(source, receiver),
                    "source": source,
                    "receiver": receiver,
                    "distance_km": f"{haversine_km(left, right):.6f}",
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(materialized[0].keys()) if materialized else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_input_files(data_root: Path, xml_root: Path) -> list[Path]:
    data_files = [
        path
        for path in sorted(Path(data_root).glob("1D.*/*.mseed"))
        if not path.name.startswith("._")
    ]
    xml_files = sorted(Path(xml_root).glob("1D.*.xml"))
    return data_files + xml_files


def build_input_manifest(
    data_root: Path,
    xml_root: Path,
    config: WangPwsConfig,
    compute_sha256: bool = True,
) -> dict:
    files = []
    total_size = 0
    for path in iter_input_files(data_root, xml_root):
        stat = path.stat()
        total_size += int(stat.st_size)
        row = {
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        if compute_sha256:
            row["sha256"] = sha256_file(path)
        files.append(row)
    stations = collect_station_infos(data_root, xml_root, config)
    return {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "data_root": str(data_root),
        "xml_root": str(xml_root),
        "file_count": len(files),
        "total_size": total_size,
        "stations": [asdict(station) for station in stations],
        "station_count": len(stations),
        "pairs": build_pair_rows(stations),
        "pair_count": len(stations) * (len(stations) - 1) // 2,
        "config": config.as_dict(),
        "files": files,
    }


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def input_manifest_digest(manifest: dict) -> str:
    file_rows = manifest.get("files", [])
    payload = [
        {
            "path": row.get("path"),
            "size": row.get("size"),
            "sha256": row.get("sha256"),
        }
        for row in file_rows
    ]
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
