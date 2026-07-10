#!/usr/bin/env python3
"""Prepare 2014 1D+XD phase-velocity observations for AANT tomography."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


FDSN_NS = {"fdsn": "http://www.fdsn.org/xml/station/1"}
DEFAULT_PERIODS = tuple(round(value, 1) for value in (1.0 + 0.5 * idx for idx in range(9)))
MAX_AANT_RECEIVERS_PER_SOURCE = 400
AANT_GRID_DX = 0.02
AANT_GRID_DY = 0.02


def parse_curve_filename(path: Path, *, curve_prefix: str = "CDisp") -> tuple[str, str]:
    match = re.match(rf"^{re.escape(curve_prefix)}\.(.+)__(.+)\.txt$", path.name)
    if not match:
        raise ValueError(f"Unsupported curve filename: {path.name}")
    return match.group(1), match.group(2)


def parse_curve_file(
    curve_path: Path,
    *,
    min_period: float = 1.0,
    max_period: float = 5.0,
    curve_prefix: str = "CDisp",
) -> Dict[str, object]:
    lines = curve_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Curve file is missing coordinate headers: {curve_path}")

    source_code, receiver_code = parse_curve_filename(curve_path, curve_prefix=curve_prefix)
    source_lon, source_lat = _parse_lon_lat(lines[0], curve_path, "source")
    receiver_lon, receiver_lat = _parse_lon_lat(lines[1], curve_path, "receiver")

    period_velocity: Dict[float, float] = {}
    for raw_line in lines[2:]:
        parts = raw_line.split()
        if len(parts) < 2:
            continue
        period = _safe_float(parts[0])
        velocity = _safe_float(parts[1])
        if period is None or velocity is None:
            continue
        if not (float(min_period) <= period <= float(max_period)):
            continue
        if velocity <= 0.0 or not math.isfinite(velocity):
            continue
        period_velocity[round(period, 3)] = velocity

    return {
        "curve_path": str(curve_path.resolve()),
        "source_code": source_code,
        "receiver_code": receiver_code,
        "source_lon": source_lon,
        "source_lat": source_lat,
        "receiver_lon": receiver_lon,
        "receiver_lat": receiver_lat,
        "period_velocity": period_velocity,
    }


def discover_curve_files(
    curve_root: Path,
    *,
    source_codes: Optional[Set[str]] = None,
    curve_prefix: str = "CDisp",
) -> List[Path]:
    station_curve_dirs = sorted(
        path / "curves"
        for path in curve_root.glob("dispersion_*")
        if (path / "curves").is_dir()
    )
    search_roots = station_curve_dirs or [curve_root]
    files: List[Path] = []
    for search_root in search_roots:
        iterator = search_root.glob(f"{curve_prefix}*.txt") if station_curve_dirs else search_root.rglob(f"{curve_prefix}*.txt")
        for path in sorted(iterator):
            try:
                source_code, _ = parse_curve_filename(path, curve_prefix=curve_prefix)
            except ValueError:
                continue
            if source_codes and source_code not in source_codes:
                continue
            files.append(path)
    return files


def filter_curve_files_by_pair_ids(
    curve_files: Sequence[Path],
    allowed_pair_ids: Set[str],
    *,
    curve_prefix: str = "CDisp",
) -> List[Path]:
    filtered: List[Path] = []
    for path in curve_files:
        try:
            source_code, receiver_code = parse_curve_filename(path, curve_prefix=curve_prefix)
        except ValueError:
            continue
        if f"{source_code}__{receiver_code}" in allowed_pair_ids:
            filtered.append(path)
    return filtered


def load_allowed_pair_ids_from_qc_csv(
    path: Path,
    allowed_grades: Set[str],
) -> Set[str]:
    pair_ids: Set[str] = set()
    allowed = {grade.strip().upper() for grade in allowed_grades if grade.strip()}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return pair_ids
        pair_field = "pair" if "pair" in reader.fieldnames else "pair_name"
        if pair_field not in reader.fieldnames or "grade_v5" not in reader.fieldnames:
            raise ValueError(f"QC CSV must contain pair/pair_name and grade_v5 columns: {path}")
        for row in reader:
            grade = str(row.get("grade_v5", "")).strip().upper()
            pair = str(row.get(pair_field, "")).strip()
            if pair and grade in allowed:
                pair_ids.add(pair)
    return pair_ids


def prepare_workspace(
    *,
    curve_root: Path,
    metadata_dirs: Sequence[Path],
    output_root: Path,
    periods: Sequence[float] = DEFAULT_PERIODS,
    min_period: float = 1.0,
    max_period: float = 5.0,
    min_paths: int = 2,
    source_codes: Optional[Set[str]] = None,
    bbox: Optional[Mapping[str, float]] = None,
    focus_bbox: Optional[Mapping[str, float]] = None,
    outer_node_spacing_deg: float = 0.08,
    focus_node_spacing_deg: float = 0.02,
    focus_padding_deg: float = 0.04,
    pixel_grid_size: int = 80,
    pair_qc_csv: Optional[Path] = None,
    qc_grades: Optional[Set[str]] = None,
    curve_prefix: str = "CDisp",
    min_wavelengths: float = 0.0,
    aant_damping: float = 700.0,
    aant_initial_icoord: int = 2,
    aant_iteration_icoord: int = 1,
    aant_final_icoord: Optional[int] = None,
) -> Dict[str, object]:
    curve_files = discover_curve_files(
        curve_root,
        source_codes=source_codes,
        curve_prefix=curve_prefix,
    )
    unfiltered_curve_file_count = len(curve_files)
    allowed_pair_ids: Optional[Set[str]] = None
    if pair_qc_csv is not None:
        allowed_pair_ids = load_allowed_pair_ids_from_qc_csv(
            pair_qc_csv,
            qc_grades or {"A", "B"},
        )
        curve_files = filter_curve_files_by_pair_ids(
            curve_files,
            allowed_pair_ids,
            curve_prefix=curve_prefix,
        )
    station_catalog = load_station_catalog(metadata_dirs)
    curve_records = [
        parse_curve_file(
            path,
            min_period=min_period,
            max_period=max_period,
            curve_prefix=curve_prefix,
        )
        for path in curve_files
    ]
    curve_records = apply_station_catalog_coordinates(curve_records, station_catalog)

    output_root.mkdir(parents=True, exist_ok=True)
    stations = merge_station_records(curve_records, station_catalog)
    write_stations_csv(output_root / "stations.csv", stations)

    summary: Dict[str, object] = {
        "curve_root": str(curve_root.resolve()),
        "curve_prefix": curve_prefix,
        "curve_file_count": len(curve_records),
        "curve_file_count_unfiltered": unfiltered_curve_file_count,
        "periods_requested": [round(float(period), 1) for period in periods],
        "station_count": len(stations),
        "pixel_grid_size": int(pixel_grid_size),
        "periods": {},
    }
    if pair_qc_csv is not None:
        summary["pair_qc_filter"] = {
            "pair_qc_csv": str(pair_qc_csv.resolve()),
            "qc_grades": sorted(qc_grades or {"A", "B"}),
            "allowed_pair_count": len(allowed_pair_ids or set()),
        }
    if bbox is not None:
        summary["bbox"] = normalized_bbox(bbox)
        summary["pixel_cell_size_km"] = estimate_pixel_cell_size_km(bbox, pixel_grid_size)
    if focus_bbox is not None:
        summary["focus_bbox"] = normalized_bbox(focus_bbox)
        summary["node_spacing_deg"] = {
            "outer": float(outer_node_spacing_deg),
            "focus": float(focus_node_spacing_deg),
            "focus_padding": float(focus_padding_deg),
        }

    for period in periods:
        period_key = format_period(period)
        period_dir = output_root / f"period_{period_key}s"
        period_dir.mkdir(parents=True, exist_ok=True)

        all_rows = build_period_rows(curve_records, period=period)
        bbox_rows = filter_rows_to_bbox(all_rows, bbox) if bbox is not None else all_rows
        rows = filter_rows_by_min_wavelengths(
            bbox_rows,
            min_wavelengths=min_wavelengths,
        )
        write_paths_csv(period_dir / "paths.csv", rows)
        plot_path_coverage(period_dir / "coverage.png", rows, stations, period=period, bbox=bbox)
        plot_phase_velocity_pixel_map(
            period_dir / "phase_velocity_pixel_map.png",
            rows,
            period=period,
            bbox=bbox,
            grid_size=pixel_grid_size,
        )

        ready = len(rows) >= int(min_paths)
        status = "ready" if ready else "insufficient_paths"
        period_summary = {
            "status": status,
            "unfiltered_path_count": len(all_rows),
            "bbox_path_count": len(bbox_rows),
            "min_wavelengths": float(min_wavelengths),
            "wavelength_rejected_count": len(bbox_rows) - len(rows),
            "valid_path_count": len(rows),
            "ready_for_aant": ready,
            "output_dir": str(period_dir.resolve()),
            "paths_csv": str((period_dir / "paths.csv").resolve()),
            "coverage_png": str((period_dir / "coverage.png").resolve()),
            "phase_velocity_pixel_map_png": str(
                (period_dir / "phase_velocity_pixel_map.png").resolve()
            ),
        }

        if ready:
            ordered_rows = write_source_receiver_inputs(period_dir, rows)
            write_data_txt(period_dir / "data.txt", ordered_rows)
            write_data_op_coor(period_dir / "data_op_coor.txt", ordered_rows)
            bounds = build_study_area_files(
                period_dir,
                rows,
                focus_bbox=focus_bbox,
                outer_node_spacing_deg=outer_node_spacing_deg,
                focus_node_spacing_deg=focus_node_spacing_deg,
                focus_padding_deg=focus_padding_deg,
            )
            source_count = count_sources(rows)
            write_run_script(period_dir, period=period, bounds=bounds, source_count=source_count)
            write_aant_tomography_script(
                period_dir,
                period=period,
                bounds=bounds,
                source_count=source_count,
                data_count=len(ordered_rows),
                damping=aant_damping,
                initial_icoord=aant_initial_icoord,
                iteration_icoord=aant_iteration_icoord,
                final_icoord=(
                    aant_final_icoord if aant_final_icoord is not None else aant_iteration_icoord
                ),
            )
            period_summary.update(
                {
                    "data_txt": str((period_dir / "data.txt").resolve()),
                    "data_op_coor": str((period_dir / "data_op_coor.txt").resolve()),
                    "source_txt": str((period_dir / "source.txt").resolve()),
                    "receiver_txt": str((period_dir / "receiver.txt").resolve()),
                    "vel_txt": str((period_dir / "vel.txt").resolve()),
                    "tri_txt": str((period_dir / "tri.txt").resolve()),
                    "studyarea_vtx_txt": str((period_dir / "studyarea_vtx.txt").resolve()),
                    "node_count": bounds["node_count"],
                    "edge_count": bounds["edge_count"],
                    "focus_node_count": bounds["focus_node_count"],
                    "run_script": str((period_dir / "run_aant_period.sh").resolve()),
                    "tomography_script": str((period_dir / "run_aant_tomography.sh").resolve()),
                }
            )

        summary["periods"][period_key] = period_summary

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_period_rows(curve_records: Sequence[Mapping[str, object]], *, period: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    period_key = round(float(period), 3)
    seen_pairs: Set[tuple[str, str]] = set()
    for record in curve_records:
        period_velocity = record["period_velocity"]
        assert isinstance(period_velocity, Mapping)
        velocity = period_velocity.get(period_key)
        if velocity is None:
            continue
        source_code = str(record["source_code"])
        receiver_code = str(record["receiver_code"])
        pair_key = tuple(sorted((source_code, receiver_code)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        rows.append(
            {
                "source_code": source_code,
                "receiver_code": receiver_code,
                "source_lon": float(record["source_lon"]),
                "source_lat": float(record["source_lat"]),
                "receiver_lon": float(record["receiver_lon"]),
                "receiver_lat": float(record["receiver_lat"]),
                "period": period_key,
                "phase_velocity_km_s": float(velocity),
                "velocity_km_s": float(velocity),
            }
        )
    rows.sort(key=lambda row: (str(row["source_code"]), str(row["receiver_code"])))
    return rows


def filter_rows_to_bbox(
    rows: Sequence[Mapping[str, object]],
    bbox: Mapping[str, float],
) -> List[Dict[str, object]]:
    return [dict(row) for row in rows if row_intersects_bbox(row, bbox)]


def filter_rows_by_min_wavelengths(
    rows: Sequence[Mapping[str, object]],
    *,
    min_wavelengths: float,
) -> List[Dict[str, object]]:
    if min_wavelengths <= 0.0:
        return [dict(row) for row in rows]
    filtered: List[Dict[str, object]] = []
    for row in rows:
        distance = distance_km(
            float(row["source_lon"]),
            float(row["source_lat"]),
            float(row["receiver_lon"]),
            float(row["receiver_lat"]),
        )
        min_distance = float(min_wavelengths) * float(row["period"]) * row_velocity(row)
        if distance >= min_distance:
            filtered.append(dict(row))
    return filtered


def row_intersects_bbox(row: Mapping[str, object], bbox: Mapping[str, float]) -> bool:
    return (
        clip_segment_to_bbox(
            float(row["source_lon"]),
            float(row["source_lat"]),
            float(row["receiver_lon"]),
            float(row["receiver_lat"]),
            bbox,
        )
        is not None
    )


def point_in_bbox(lon: float, lat: float, bbox: Mapping[str, float]) -> bool:
    box = normalized_bbox(bbox)
    return box["minlon"] <= lon <= box["maxlon"] and box["minlat"] <= lat <= box["maxlat"]


def clip_segment_to_bbox(
    lon0: float,
    lat0: float,
    lon1: float,
    lat1: float,
    bbox: Mapping[str, float],
) -> Optional[tuple[float, float, float, float]]:
    box = normalized_bbox(bbox)
    dx = lon1 - lon0
    dy = lat1 - lat0
    lower = 0.0
    upper = 1.0
    checks = (
        (-dx, lon0 - box["minlon"]),
        (dx, box["maxlon"] - lon0),
        (-dy, lat0 - box["minlat"]),
        (dy, box["maxlat"] - lat0),
    )
    for direction, distance in checks:
        if direction == 0.0:
            if distance < 0.0:
                return None
            continue
        ratio = distance / direction
        if direction < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return (
        lon0 + lower * dx,
        lat0 + lower * dy,
        lon0 + upper * dx,
        lat0 + upper * dy,
    )


def load_station_catalog(metadata_dirs: Sequence[Path]) -> Dict[str, Dict[str, float]]:
    records: Dict[str, Dict[str, float]] = {}
    for metadata_dir in metadata_dirs:
        if not metadata_dir.exists():
            continue
        for xml_path in sorted(metadata_dir.rglob("*.xml")):
            code, lon, lat = parse_stationxml(xml_path)
            records[code] = {"longitude": lon, "latitude": lat}
    return records


def apply_station_catalog_coordinates(
    curve_records: Sequence[Mapping[str, object]],
    station_catalog: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, object]]:
    corrected_records: List[Dict[str, object]] = []
    for record in curve_records:
        corrected = dict(record)
        source_code = str(record["source_code"])
        receiver_code = str(record["receiver_code"])
        source_station = station_catalog.get(source_code)
        receiver_station = station_catalog.get(receiver_code)
        if source_station:
            corrected["source_lon"] = float(source_station["longitude"])
            corrected["source_lat"] = float(source_station["latitude"])
        if receiver_station:
            corrected["receiver_lon"] = float(receiver_station["longitude"])
            corrected["receiver_lat"] = float(receiver_station["latitude"])
        corrected_records.append(corrected)
    return corrected_records


def parse_stationxml(xml_path: Path) -> tuple[str, float, float]:
    root = ET.parse(xml_path).getroot()
    network = root.find("fdsn:Network", FDSN_NS)
    if network is None:
        raise ValueError(f"No Network node found in {xml_path}")
    station = network.find("fdsn:Station", FDSN_NS)
    if station is None:
        raise ValueError(f"No Station node found in {xml_path}")
    network_code = network.attrib.get("code", "").strip()
    station_code = station.attrib.get("code", "").strip()
    lon_text = _first_text(station, "Longitude")
    lat_text = _first_text(station, "Latitude")
    return f"{network_code}.{station_code}", float(lon_text or 0.0), float(lat_text or 0.0)


def merge_station_records(
    curve_records: Sequence[Mapping[str, object]],
    station_catalog: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {}
    for record in curve_records:
        source_code = str(record["source_code"])
        receiver_code = str(record["receiver_code"])
        merged[source_code] = {
            "longitude": float(record["source_lon"]),
            "latitude": float(record["source_lat"]),
        }
        merged[receiver_code] = {
            "longitude": float(record["receiver_lon"]),
            "latitude": float(record["receiver_lat"]),
        }
    for code, values in station_catalog.items():
        merged.setdefault(
            code,
            {"longitude": float(values["longitude"]), "latitude": float(values["latitude"])},
        )
    return dict(sorted(merged.items()))


def write_stations_csv(path: Path, stations: Mapping[str, Mapping[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["station_code", "longitude", "latitude"])
        writer.writeheader()
        for station_code, values in stations.items():
            writer.writerow(
                {
                    "station_code": station_code,
                    "longitude": f"{float(values['longitude']):.6f}",
                    "latitude": f"{float(values['latitude']):.6f}",
                }
            )


def write_paths_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "source_code",
        "receiver_code",
        "source_lon",
        "source_lat",
        "receiver_lon",
        "receiver_lat",
        "period",
        "phase_velocity_km_s",
        "velocity_km_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_velocity(row: Mapping[str, object]) -> float:
    if "velocity_km_s" in row:
        return float(row["velocity_km_s"])
    return float(row["phase_velocity_km_s"])


def write_data_txt(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            distance = distance_km(
                float(row["source_lon"]),
                float(row["source_lat"]),
                float(row["receiver_lon"]),
                float(row["receiver_lat"]),
            )
            handle.write(f"{distance / row_velocity(row):.8f}\n")


def write_data_op_coor(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            distance = distance_km(
                float(row["source_lon"]),
                float(row["source_lat"]),
                float(row["receiver_lon"]),
                float(row["receiver_lat"]),
            )
            handle.write(
                f"{float(row['source_lon']) + 360.0:.6f} "
                f"{float(row['source_lat']):.6f} "
                f"{float(row['receiver_lon']) + 360.0:.6f} "
                f"{float(row['receiver_lat']):.6f} "
                f"{distance / row_velocity(row):.8f} "
                f"{distance:.6f} "
                f"{row_velocity(row):.6f}\n"
            )


def write_source_receiver_inputs(
    period_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> List[Mapping[str, object]]:
    grouped: MutableMapping[str, List[Mapping[str, object]]] = defaultdict(list)
    source_lookup: Dict[str, tuple[float, float]] = {}
    for row in rows:
        source_code = str(row["source_code"])
        grouped[source_code].append(row)
        source_lookup[source_code] = (
            float(row["source_lon"]) + 360.0,
            float(row["source_lat"]),
        )

    source_blocks: List[tuple[str, int, List[Mapping[str, object]]]] = []
    for source_code in sorted(grouped):
        rows_for_source = sorted(grouped[source_code], key=lambda row: str(row["receiver_code"]))
        for chunk_index, chunk_rows in enumerate(
            chunk_rows_by_limit(rows_for_source, MAX_AANT_RECEIVERS_PER_SOURCE),
            start=1,
        ):
            source_blocks.append((source_code, chunk_index, chunk_rows))

    with (period_dir / "source.txt").open("w", encoding="utf-8") as handle:
        for source_code, _chunk_index, _chunk_rows in source_blocks:
            lon360, lat = source_lookup[source_code]
            handle.write(f"{lon360:.6f} {lat:.6f}\n")

    receiver_manifest_lines: List[str] = []
    chunk_manifest_rows: List[Dict[str, object]] = []
    ordered_rows: List[Mapping[str, object]] = []
    for index, (source_code, chunk_index, rows_for_source) in enumerate(source_blocks, start=1):
        receiver_path = period_dir / f"rfile{index:03d}.txt"
        receiver_manifest_lines.append(receiver_path.name)
        with receiver_path.open("w", encoding="utf-8") as handle:
            handle.write(f"{len(rows_for_source)}\n")
            for row in rows_for_source:
                handle.write(
                    f"{float(row['receiver_lon']) + 360.0:.6f} "
                    f"{float(row['receiver_lat']):.6f}\n"
                )
                ordered_rows.append(row)
        chunk_manifest_rows.append(
            {
                "chunk_id": index,
                "source_code": source_code,
                "chunk_index": chunk_index,
                "receiver_count": len(rows_for_source),
            }
        )

    (period_dir / "receiver.txt").write_text(
        "\n".join(receiver_manifest_lines) + ("\n" if receiver_manifest_lines else ""),
        encoding="utf-8",
    )
    write_chunk_manifest(period_dir / "source_chunks.csv", chunk_manifest_rows)
    return ordered_rows


def count_sources(rows: Sequence[Mapping[str, object]]) -> int:
    grouped_counts: MutableMapping[str, int] = defaultdict(int)
    for row in rows:
        grouped_counts[str(row["source_code"])] += 1
    return sum(
        math.ceil(count / MAX_AANT_RECEIVERS_PER_SOURCE)
        for count in grouped_counts.values()
    )


def chunk_rows_by_limit(
    rows: Sequence[Mapping[str, object]],
    limit: int,
) -> Iterable[List[Mapping[str, object]]]:
    if limit <= 0:
        raise ValueError("Chunk limit must be positive.")
    for start in range(0, len(rows), limit):
        yield list(rows[start : start + limit])


def write_chunk_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chunk_id", "source_code", "chunk_index", "receiver_count"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_study_area_files(
    period_dir: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    focus_bbox: Optional[Mapping[str, float]] = None,
    outer_node_spacing_deg: float = 0.08,
    focus_node_spacing_deg: float = 0.02,
    focus_padding_deg: float = 0.04,
) -> Dict[str, float]:
    all_lon360 = [float(row["source_lon"]) + 360.0 for row in rows] + [
        float(row["receiver_lon"]) + 360.0 for row in rows
    ]
    all_lat = [float(row["source_lat"]) for row in rows] + [
        float(row["receiver_lat"]) for row in rows
    ]
    margin = 0.3
    minx, maxx = snap_bounds_to_grid(
        min(all_lon360) - margin,
        max(all_lon360) + margin,
        AANT_GRID_DX,
    )
    miny, maxy = snap_bounds_to_grid(
        min(all_lat) - margin,
        max(all_lat) + margin,
        AANT_GRID_DY,
    )
    corners = [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
    ]

    with (period_dir / "studyarea_vtx.txt").open("w", encoding="utf-8") as handle:
        for xval, yval in corners:
            handle.write(f"{xval:.6f} {yval:.6f}\n")

    default_velocity = sum(row_velocity(row) for row in rows) / len(rows)
    nodes = build_initial_velocity_nodes(
        minx,
        maxx,
        miny,
        maxy,
        default_velocity,
        focus_bbox=focus_bbox,
        outer_spacing_deg=outer_node_spacing_deg,
        focus_spacing_deg=focus_node_spacing_deg,
        focus_padding_deg=focus_padding_deg,
    )
    edge_count = sum(1 for node in nodes if node[3])
    focus_node_count = count_nodes_inside_bbox(nodes, focus_bbox) if focus_bbox is not None else 0
    with (period_dir / "vel.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{len(nodes)}\n")
        handle.write(f"{edge_count}\n")
        for xval, yval, velocity, _is_edge in nodes:
            handle.write(f"{xval:.6f} {yval:.6f} {velocity:.6f}\n")

    triangles = Delaunay(np.array([(node[0], node[1]) for node in nodes], dtype=float)).simplices
    with (period_dir / "tri.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{len(triangles)}\n")
        for triangle in triangles:
            # AANT reads qdelaunay output and internally adds 1 to triangle indices.
            handle.write(" ".join(str(int(index)) for index in triangle) + "\n")

    return {
        "minx": minx,
        "maxx": maxx,
        "miny": miny,
        "maxy": maxy,
        "refine_margin_x": 0.05,
        "refine_margin_y": 0.05,
        "default_velocity": default_velocity,
        "node_count": len(nodes),
        "edge_count": edge_count,
        "focus_node_count": focus_node_count,
        "studyarea_vertex_count": len(corners),
    }


def snap_bounds_to_grid(min_value: float, max_value: float, spacing: float) -> tuple[float, float]:
    if spacing <= 0.0:
        raise ValueError("Grid spacing must be positive.")
    epsilon = spacing * 1e-6
    snapped_min = math.floor((min_value + epsilon) / spacing) * spacing
    snapped_max = math.ceil((max_value - epsilon) / spacing) * spacing
    return round(snapped_min, 6), round(snapped_max, 6)


def build_initial_velocity_nodes(
    minx: float,
    maxx: float,
    miny: float,
    maxy: float,
    velocity: float,
    *,
    long_axis_nodes: int = 18,
    focus_bbox: Optional[Mapping[str, float]] = None,
    outer_spacing_deg: float = 0.08,
    focus_spacing_deg: float = 0.02,
    focus_padding_deg: float = 0.04,
) -> List[tuple[float, float, float, bool]]:
    if focus_bbox is not None:
        return build_focused_velocity_nodes(
            minx,
            maxx,
            miny,
            maxy,
            velocity,
            focus_bbox=focus_bbox,
            outer_spacing_deg=outer_spacing_deg,
            focus_spacing_deg=focus_spacing_deg,
            focus_padding_deg=focus_padding_deg,
        )

    width = maxx - minx
    height = maxy - miny
    if width >= height:
        nx = long_axis_nodes
        ny = max(4, int(round(long_axis_nodes * height / width)))
    else:
        ny = long_axis_nodes
        nx = max(4, int(round(long_axis_nodes * width / height)))
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    edge_nodes: List[tuple[float, float, float, bool]] = []
    inner_nodes: List[tuple[float, float, float, bool]] = []
    for iy, yval in enumerate(ys):
        for ix, xval in enumerate(xs):
            is_edge = ix == 0 or iy == 0 or ix == nx - 1 or iy == ny - 1
            node = (float(xval), float(yval), float(velocity), is_edge)
            if is_edge:
                edge_nodes.append(node)
            else:
                inner_nodes.append(node)
    return edge_nodes + inner_nodes


def build_focused_velocity_nodes(
    minx: float,
    maxx: float,
    miny: float,
    maxy: float,
    velocity: float,
    *,
    focus_bbox: Mapping[str, float],
    outer_spacing_deg: float,
    focus_spacing_deg: float,
    focus_padding_deg: float,
) -> List[tuple[float, float, float, bool]]:
    if outer_spacing_deg <= 0.0 or focus_spacing_deg <= 0.0:
        raise ValueError("Velocity-node spacing must be positive.")
    nodes: Dict[tuple[float, float], tuple[float, float, float, bool]] = {}

    for xval in coordinate_values(minx, maxx, outer_spacing_deg):
        for yval in coordinate_values(miny, maxy, outer_spacing_deg):
            add_velocity_node(nodes, xval, yval, velocity, is_edge=is_boundary_node(xval, yval, minx, maxx, miny, maxy))

    box = normalized_bbox(focus_bbox)
    focus_minx = max(minx, box["minlon"] + 360.0 - focus_padding_deg)
    focus_maxx = min(maxx, box["maxlon"] + 360.0 + focus_padding_deg)
    focus_miny = max(miny, box["minlat"] - focus_padding_deg)
    focus_maxy = min(maxy, box["maxlat"] + focus_padding_deg)
    for xval in coordinate_values(focus_minx, focus_maxx, focus_spacing_deg):
        for yval in coordinate_values(focus_miny, focus_maxy, focus_spacing_deg):
            add_velocity_node(nodes, xval, yval, velocity, is_edge=is_boundary_node(xval, yval, minx, maxx, miny, maxy))

    edge_nodes: List[tuple[float, float, float, bool]] = []
    inner_nodes: List[tuple[float, float, float, bool]] = []
    for _key, node in sorted(nodes.items(), key=lambda item: (item[1][1], item[1][0])):
        if node[3]:
            edge_nodes.append(node)
        else:
            inner_nodes.append(node)
    return edge_nodes + inner_nodes


def add_velocity_node(
    nodes: Dict[tuple[float, float], tuple[float, float, float, bool]],
    xval: float,
    yval: float,
    velocity: float,
    *,
    is_edge: bool,
) -> None:
    key = (round(float(xval), 6), round(float(yval), 6))
    old = nodes.get(key)
    nodes[key] = (key[0], key[1], float(velocity), bool(is_edge or (old[3] if old else False)))


def coordinate_values(min_value: float, max_value: float, spacing: float) -> np.ndarray:
    start = math.ceil((min_value - spacing * 1e-6) / spacing) * spacing
    stop = math.floor((max_value + spacing * 1e-6) / spacing) * spacing
    values = np.arange(start, stop + spacing * 0.5, spacing, dtype=float)
    values = np.concatenate(([min_value], values, [max_value]))
    return np.array(sorted({round(float(value), 6) for value in values}))


def is_boundary_node(xval: float, yval: float, minx: float, maxx: float, miny: float, maxy: float) -> bool:
    return (
        math.isclose(xval, minx, abs_tol=1e-6)
        or math.isclose(xval, maxx, abs_tol=1e-6)
        or math.isclose(yval, miny, abs_tol=1e-6)
        or math.isclose(yval, maxy, abs_tol=1e-6)
    )


def count_nodes_inside_bbox(
    nodes: Sequence[tuple[float, float, float, bool]],
    bbox: Optional[Mapping[str, float]],
) -> int:
    if bbox is None:
        return 0
    box = normalized_bbox(bbox)
    return sum(
        1
        for xval, yval, _velocity, _is_edge in nodes
        if box["minlon"] <= xval - 360.0 <= box["maxlon"]
        and box["minlat"] <= yval <= box["maxlat"]
    )


def write_run_script(
    period_dir: Path,
    *,
    period: float,
    bounds: Mapping[str, float],
    source_count: int,
) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail

AANT_ROOT="${{AANT_ROOT:-/path/to/AANT/src_tomo}}"
VEL_FILE="${{VEL_FILE:-vel.txt}}"
TRI_FILE="${{TRI_FILE:-tri.txt}}"
VTX_FILE="${{VTX_FILE:-studyarea_vtx.txt}}"

cd "{period_dir}"
cat > fmm_adpt.inp <<EOF
{period:.1f}
source.txt
receiver.txt
$VEL_FILE
$TRI_FILE
{int(source_count)}
84
0.02 0.02
5.0 5.0
0.2 0.2
{bounds['minx']:.6f} {bounds['maxx']:.6f}
{bounds['miny']:.6f} {bounds['maxy']:.6f}
0.4 4 10
2
2
2
{bounds['default_velocity']:.6f}
{int(bounds['studyarea_vertex_count'])}
$VTX_FILE
1
EOF

if [[ -x "$AANT_ROOT/fmm_adpt" ]]; then
  "$AANT_ROOT/fmm_adpt"
else
  echo "Prepared {period_dir.name} for desktop AANT execution."
  echo "Compile AANT or set AANT_ROOT before running this script."
fi
"""
    path = period_dir / "run_aant_period.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_aant_tomography_script(
    period_dir: Path,
    *,
    period: float,
    bounds: Mapping[str, float],
    source_count: int,
    data_count: int,
    damping: float = 700.0,
    initial_icoord: int = 2,
    iteration_icoord: int = 1,
    final_icoord: int = 1,
) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail

AANT_ROOT="${{AANT_ROOT:-/path/to/AANT/src_tomo}}"
AANT_DAMPING="${{AANT_DAMPING:-{float(damping):.6g}}}"
INITIAL_ICOORD="${{INITIAL_ICOORD:-{int(initial_icoord)}}}"
ITERATION_ICOORD="${{ITERATION_ICOORD:-{int(iteration_icoord)}}}"
FINAL_ICOORD="${{FINAL_ICOORD:-{int(final_icoord)}}}"
cd "{period_dir}"

run_forward() {{
  local model_file="$1"
  local icoord="$2"
  cat > fmm_forward.inp <<EOF
source.txt
receiver.txt
$model_file
tri.txt
{int(source_count)}
0.02 0.02
5.0 5.0
0.2 0.2
{bounds['minx']:.6f} {bounds['maxx']:.6f}
{bounds['miny']:.6f} {bounds['maxy']:.6f}
2
$icoord
2
{bounds['default_velocity']:.6f}
{int(bounds['studyarea_vertex_count'])}
studyarea_vtx.txt
1
EOF
  "$AANT_ROOT/fmm_forward"
}}

run_inverse() {{
  cat > inverseprb.inp <<EOF
fd.txt
data.txt
outmodel.txt
velmodel.txt
vel.txt
{int(data_count)}
{bounds['minx']:.6f} {bounds['maxx']:.6f}
{bounds['miny']:.6f} {bounds['maxy']:.6f}
{bounds['default_velocity']:.6f}
$AANT_DAMPING
1 1
EOF
  "$AANT_ROOT/inverseprb"
}}

write_residuals() {{
  local output_file="$1"
  paste data.txt traveltimbyfd.txt | awk '{{print $2-$1}}' > "$output_file"
}}

run_forward vel.txt "$INITIAL_ICOORD"
write_residuals residual_iter0.txt
run_inverse
cp velresult.txt velresult0.txt
cp traveltime.txt traveltime_iter0.txt
cp traveltimbyfd.txt traveltimbyfd_iter0.txt

for iteration in 1 2 3; do
  run_forward velresult.txt "$ITERATION_ICOORD"
  write_residuals "residual_iter${{iteration}}.txt"
  cp traveltime.txt "traveltime_iter${{iteration}}.txt"
  cp traveltimbyfd.txt "traveltimbyfd_iter${{iteration}}.txt"
  run_inverse
  cp velresult.txt "velresult${{iteration}}.txt"
done

run_forward velresult.txt "$FINAL_ICOORD"
write_residuals residual_final.txt
cp traveltime.txt traveltime_final.txt
cp traveltimbyfd.txt traveltimbyfd_final.txt
"""
    path = period_dir / "run_aant_tomography.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def plot_path_coverage(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    stations: Mapping[str, Mapping[str, float]],
    *,
    period: float,
    bbox: Optional[Mapping[str, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for row in rows:
        ax.plot(
            [float(row["source_lon"]), float(row["receiver_lon"])],
            [float(row["source_lat"]), float(row["receiver_lat"])],
            color="#1f78b4",
            linewidth=0.8,
            alpha=0.55,
        )
    if stations:
        station_values = list(stations.values())
        if bbox is not None:
            station_values = [
                values
                for values in station_values
                if point_in_bbox(float(values["longitude"]), float(values["latitude"]), bbox)
            ]
    else:
        station_values = []
    if station_values:
        ax.scatter(
            [float(values["longitude"]) for values in station_values],
            [float(values["latitude"]) for values in station_values],
            s=18,
            color="#111111",
            zorder=3,
        )
    if bbox is not None:
        box = normalized_bbox(bbox)
        ax.set_xlim(box["minlon"], box["maxlon"])
        ax.set_ylim(box["minlat"], box["maxlat"])
    ax.set_title(f"Path Coverage at {period:.1f} s")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_phase_velocity_pixel_map(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    period: float,
    bbox: Optional[Mapping[str, float]] = None,
    grid_size: int = 80,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if rows and bbox is not None:
        grid, extent = build_phase_velocity_grid(rows, bbox, grid_size)
        masked_grid = np.ma.masked_invalid(grid)
        image = ax.imshow(
            masked_grid,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            cmap="RdBu",
            aspect="auto",
        )
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Phase Velocity (km/s)")
        box = normalized_bbox(bbox)
        ax.set_xlim(box["minlon"], box["maxlon"])
        ax.set_ylim(box["minlat"], box["maxlat"])
    elif rows:
        mid_lon = [
            (float(row["source_lon"]) + float(row["receiver_lon"])) / 2.0
            for row in rows
        ]
        mid_lat = [
            (float(row["source_lat"]) + float(row["receiver_lat"])) / 2.0
            for row in rows
        ]
        velocities = [row_velocity(row) for row in rows]
        hb = ax.hexbin(
            mid_lon,
            mid_lat,
            C=velocities,
            reduce_C_function=lambda values: sum(values) / len(values),
            gridsize=45,
            mincnt=1,
            cmap="RdBu",
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Phase Velocity (km/s)")
    ax.set_title(f"Phase Velocity Pixel Map at {period:.1f} s")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_phase_velocity_grid(
    rows: Sequence[Mapping[str, object]],
    bbox: Mapping[str, float],
    grid_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    box = normalized_bbox(bbox)
    nx, ny = grid_shape_for_bbox(box, grid_size)
    sum_grid = np.zeros((ny, nx), dtype=float)
    count_grid = np.zeros((ny, nx), dtype=float)
    lon_span = box["maxlon"] - box["minlon"]
    lat_span = box["maxlat"] - box["minlat"]

    for row in rows:
        clipped = clip_segment_to_bbox(
            float(row["source_lon"]),
            float(row["source_lat"]),
            float(row["receiver_lon"]),
            float(row["receiver_lat"]),
            box,
        )
        if clipped is None:
            continue
        lon0, lat0, lon1, lat1 = clipped
        samples = max(
            2,
            int(
                max(
                    abs(lon1 - lon0) / (lon_span / nx),
                    abs(lat1 - lat0) / (lat_span / ny),
                )
                * 2
            )
            + 1,
        )
        velocity = row_velocity(row)
        for fraction in np.linspace(0.0, 1.0, samples):
            lon = lon0 + (lon1 - lon0) * fraction
            lat = lat0 + (lat1 - lat0) * fraction
            ix = min(nx - 1, max(0, int((lon - box["minlon"]) / lon_span * nx)))
            iy = min(ny - 1, max(0, int((lat - box["minlat"]) / lat_span * ny)))
            sum_grid[iy, ix] += velocity
            count_grid[iy, ix] += 1.0

    grid = np.full((ny, nx), np.nan, dtype=float)
    filled = count_grid > 0.0
    grid[filled] = sum_grid[filled] / count_grid[filled]
    return grid, (box["minlon"], box["maxlon"], box["minlat"], box["maxlat"])


def grid_shape_for_bbox(bbox: Mapping[str, float], grid_size: int) -> tuple[int, int]:
    box = normalized_bbox(bbox)
    size = max(2, int(grid_size))
    width_km = lon_degree_width_km((box["minlat"] + box["maxlat"]) / 2.0) * (
        box["maxlon"] - box["minlon"]
    )
    height_km = lat_degree_height_km() * (box["maxlat"] - box["minlat"])
    if width_km >= height_km:
        nx = size
        ny = max(2, int(round(size * height_km / width_km)))
    else:
        ny = size
        nx = max(2, int(round(size * width_km / height_km)))
    return nx, ny


def estimate_pixel_cell_size_km(
    bbox: Mapping[str, float],
    grid_size: int,
) -> Dict[str, float]:
    box = normalized_bbox(bbox)
    nx, ny = grid_shape_for_bbox(box, grid_size)
    mid_lat = (box["minlat"] + box["maxlat"]) / 2.0
    return {
        "width_km": round(lon_degree_width_km(mid_lat) * (box["maxlon"] - box["minlon"]) / nx, 4),
        "height_km": round(lat_degree_height_km() * (box["maxlat"] - box["minlat"]) / ny, 4),
        "nx": nx,
        "ny": ny,
    }


def lon_degree_width_km(latitude: float) -> float:
    return 111.320 * math.cos(math.radians(latitude))


def lat_degree_height_km() -> float:
    return 110.574


def distance_km(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    radius_km = 6371.0
    lon0_rad = math.radians(lon0)
    lat0_rad = math.radians(lat0)
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    dlon = lon1_rad - lon0_rad
    dlat = lat1_rad - lat0_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat0_rad) * math.cos(lat1_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(a)))


def normalized_bbox(bbox: Mapping[str, float]) -> Dict[str, float]:
    minlon = float(bbox["minlon"])
    maxlon = float(bbox["maxlon"])
    minlat = float(bbox["minlat"])
    maxlat = float(bbox["maxlat"])
    if minlon >= maxlon or minlat >= maxlat:
        raise ValueError(f"Invalid bbox: {bbox}")
    return {
        "minlon": minlon,
        "minlat": minlat,
        "maxlon": maxlon,
        "maxlat": maxlat,
    }


def parse_bbox(value: str) -> Dict[str, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Use --bbox minlon,minlat,maxlon,maxlat")
    try:
        minlon, minlat, maxlon, maxlat = (float(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("BBox values must be numeric.") from exc
    try:
        return normalized_bbox(
            {"minlon": minlon, "minlat": minlat, "maxlon": maxlon, "maxlat": maxlat}
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_grades(value: str) -> Set[str]:
    grades = {item.strip().upper() for item in value.split(",") if item.strip()}
    if not grades:
        raise argparse.ArgumentTypeError("Use at least one QC grade, e.g. A,B")
    return grades


def format_period(period: float) -> str:
    return f"{float(period):.1f}"


def _parse_lon_lat(line: str, curve_path: Path, label: str) -> tuple[float, float]:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid {label} coordinate line in {curve_path}: {line!r}")
    return float(parts[0]), float(parts[1])


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def _first_text(parent: ET.Element, child_name: str) -> str:
    child = parent.find(f"fdsn:{child_name}", FDSN_NS)
    return child.text.strip() if child is not None and child.text else ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare 2014 phase-velocity AANT inputs.")
    parser.add_argument("curve_root", type=Path, help="Root directory containing CDisp phase curves.")
    parser.add_argument("output_root", type=Path, help="Output directory for per-period AANT inputs.")
    parser.add_argument(
        "--metadata-dir",
        action="append",
        dest="metadata_dirs",
        default=[],
        type=Path,
        help="Optional StationXML directory; repeat for multiple roots.",
    )
    parser.add_argument(
        "--periods",
        default=",".join(format_period(period) for period in DEFAULT_PERIODS),
        help="Comma-separated list of periods in seconds.",
    )
    parser.add_argument("--min-period", type=float, default=1.0)
    parser.add_argument("--max-period", type=float, default=5.0)
    parser.add_argument("--min-paths", type=int, default=2)
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        help="Optional plot/data focus region as minlon,minlat,maxlon,maxlat.",
    )
    parser.add_argument(
        "--focus-bbox",
        type=parse_bbox,
        help=(
            "Optional velocity-node focus region as minlon,minlat,maxlon,maxlat. "
            "Defaults to --bbox when --bbox is supplied."
        ),
    )
    parser.add_argument(
        "--outer-node-spacing-deg",
        type=float,
        default=0.08,
        help="Coarse velocity-node spacing outside --focus-bbox, in degrees.",
    )
    parser.add_argument(
        "--focus-node-spacing-deg",
        type=float,
        default=0.02,
        help="Dense velocity-node spacing inside --focus-bbox, in degrees.",
    )
    parser.add_argument(
        "--focus-padding-deg",
        type=float,
        default=0.04,
        help="Padding around --focus-bbox for dense velocity nodes, in degrees.",
    )
    parser.add_argument(
        "--pixel-grid-size",
        type=int,
        default=80,
        help="Long-axis pixel count for bbox phase-velocity maps.",
    )
    parser.add_argument(
        "--pair-qc-csv",
        type=Path,
        help="Optional pair-level QC CSV with pair/pair_name and grade_v5 columns.",
    )
    parser.add_argument(
        "--qc-grades",
        type=parse_grades,
        default=parse_grades("A,B"),
        help="Comma-separated grade_v5 values to keep when --pair-qc-csv is set.",
    )
    parser.add_argument(
        "--curve-prefix",
        choices=["CDisp", "GDisp"],
        default="CDisp",
        help="Velocity curve file prefix to use: CDisp for phase, GDisp for group.",
    )
    parser.add_argument(
        "--min-wavelengths",
        type=float,
        default=0.0,
        help=(
            "Reject paths shorter than this many wavelengths "
            "(distance_km < value * period_s * velocity_km_s)."
        ),
    )
    parser.add_argument(
        "--aant-damping",
        type=float,
        default=700.0,
        help="Damping value written into inverseprb.inp.",
    )
    parser.add_argument(
        "--aant-initial-icoord",
        type=int,
        choices=[1, 2],
        default=2,
        help="icoord value for the first fmm_forward call.",
    )
    parser.add_argument(
        "--aant-iteration-icoord",
        type=int,
        choices=[1, 2],
        default=1,
        help="icoord value for iterative fmm_forward calls; AANT runtomo.bash uses 1.",
    )
    parser.add_argument(
        "--aant-final-icoord",
        type=int,
        choices=[1, 2],
        help="icoord value for the final fmm_forward call; defaults to --aant-iteration-icoord.",
    )
    parser.add_argument(
        "--source-code",
        action="append",
        dest="source_codes",
        default=[],
        help="Optional source station filter, e.g. XD.MG08.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    periods = [round(float(item), 1) for item in args.periods.split(",") if item.strip()]
    summary = prepare_workspace(
        curve_root=args.curve_root,
        metadata_dirs=args.metadata_dirs,
        output_root=args.output_root,
        periods=periods,
        min_period=args.min_period,
        max_period=args.max_period,
        min_paths=args.min_paths,
        source_codes=set(args.source_codes) if args.source_codes else None,
        bbox=args.bbox,
        focus_bbox=args.focus_bbox or args.bbox,
        outer_node_spacing_deg=args.outer_node_spacing_deg,
        focus_node_spacing_deg=args.focus_node_spacing_deg,
        focus_padding_deg=args.focus_padding_deg,
        pixel_grid_size=args.pixel_grid_size,
        pair_qc_csv=args.pair_qc_csv,
        qc_grades=args.qc_grades,
        curve_prefix=args.curve_prefix,
        min_wavelengths=args.min_wavelengths,
        aant_damping=args.aant_damping,
        aant_initial_icoord=args.aant_initial_icoord,
        aant_iteration_icoord=args.aant_iteration_icoord,
        aant_final_icoord=args.aant_final_icoord,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
