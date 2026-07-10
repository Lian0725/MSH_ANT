"""Convert NoisePy 2014 1D-1D STACK files to EGFAnalysisPy DAT files.

Expected input layout:
    STACK/1D.4001/1D.4002/*.h5

Output DAT layout:
    out_dir/1D.4001__1D.4002.dat

The DAT format matches ``convert_stack_to_dat.py``:
    line 1: source lon lat
    line 2: receiver lon lat
    line 3+: time(s) positive_lag negative_lag_reversed
"""

import argparse
import fnmatch
import logging
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _stack_attrs_from_file(h5_path: Path, component: str):
    with h5py.File(h5_path, "r") as handle:
        ds = _find_stack_dataset(handle, component)
        if ds is None:
            return None
        return dict(ds.attrs)


def _distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0088
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    d_phi = math.radians(lat_b - lat_a)
    d_lambda = math.radians(lon_b - lon_a)
    hav = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(hav)))


def _find_stack_dataset(handle: h5py.File, component: str):
    aux = handle.get("AuxiliaryData")
    if aux is None:
        return None
    for key in aux.keys():
        try:
            if component in aux[key]:
                return aux[f"{key}/{component}"]
        except Exception:
            continue
    return None


def _xml_local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _children_by_local_name(element, local_name: str):
    for child in list(element):
        if _xml_local_name(child.tag) == local_name:
            yield child


def _child_text_by_local_name(element, local_name: str):
    for child in _children_by_local_name(element, local_name):
        if child.text is not None:
            return child.text.strip()
    return None


def load_stationxml_coordinates(stationxml_dir: Path):
    coords = {}  # type: Dict[str, Tuple[float, float]]
    for xml_path in sorted(stationxml_dir.glob("*.xml")):
        try:
            root = ET.parse(str(xml_path)).getroot()
        except Exception as exc:
            logger.warning("skip StationXML %s: %s", xml_path, exc)
            continue
        for network in root.iter():
            if _xml_local_name(network.tag) != "Network":
                continue
            network_code = network.attrib.get("code", "").strip()
            for station in _children_by_local_name(network, "Station"):
                station_code = station.attrib.get("code", "").strip()
                lat_text = _child_text_by_local_name(station, "Latitude")
                lon_text = _child_text_by_local_name(station, "Longitude")
                if not station_code or lat_text is None or lon_text is None:
                    continue
                try:
                    lat = float(lat_text)
                    lon = float(lon_text)
                except ValueError:
                    logger.warning("skip StationXML %s station %s: invalid coordinates", xml_path, station_code)
                    continue
                coords[station_code] = (lat, lon)
                if network_code:
                    coords[f"{network_code}.{station_code}"] = (lat, lon)
    return coords


def _attrs_coordinates(attrs):
    return (
        (float(attrs.get("latS", 0.0)), float(attrs.get("lonS", 0.0))),
        (float(attrs.get("latR", 0.0)), float(attrs.get("lonR", 0.0))),
    )


def convert_one_stack(
    h5_path: Path,
    out_dat_path: Path,
    component: str = "ZZ",
    min_distance_km: float = 0.0,
    max_distance_km: Optional[float] = None,
    allow_zero_ngood: bool = False,
    source_coords: Optional[Tuple[float, float]] = None,
    receiver_coords: Optional[Tuple[float, float]] = None,
):
    try:
        with h5py.File(h5_path, "r") as handle:
            ds = _find_stack_dataset(handle, component)
            if ds is None:
                logger.warning("skip %s: component %s not found", h5_path, component)
                return False
            attrs = dict(ds.attrs)

        dt = float(attrs.get("dt", 0.04))
        maxlag = float(attrs.get("maxlag", 30.0))
        attr_source_coords, attr_receiver_coords = _attrs_coordinates(attrs)
        if source_coords is None:
            source_coords = attr_source_coords
        if receiver_coords is None:
            receiver_coords = attr_receiver_coords
        lat_src, lon_src = source_coords
        lat_rec, lon_rec = receiver_coords
        # ngood 字段名兼容：新版 CC/stack 写入 ngood_hours，旧版写入 ngood
        ngood = int(attrs.get("ngood_hours", attrs.get("ngood", 0)))
        distance_km = _distance_km(lat_src, lon_src, lat_rec, lon_rec)

        if distance_km < min_distance_km:
            logger.debug(
                "skip %s: distance %.2f km < min %.2f km",
                h5_path,
                distance_km,
                min_distance_km,
            )
            return None
        if max_distance_km is not None and distance_km > max_distance_km:
            logger.debug(
                "skip %s: distance %.2f km > max %.2f km",
                h5_path,
                distance_km,
                max_distance_km,
            )
            return None

        if ngood < 1 and not allow_zero_ngood:
            logger.warning("skip %s: ngood=%d < 1", h5_path, ngood)
            return False
        if ngood < 1 and allow_zero_ngood:
            logger.warning("continue %s: ngood=%d < 1 but allow_zero_ngood=True", h5_path, ngood)

        with h5py.File(h5_path, "r") as handle:
            ds = _find_stack_dataset(handle, component)
            if ds is None:
                logger.warning("skip %s: component %s not found", h5_path, component)
                return False
            data = ds[:]
        if np.all(data == 0):
            logger.warning("skip %s: data are all zero", h5_path)
            return False

        nlag = round(maxlag / dt)
        expected = 2 * nlag + 1
        if len(data) != expected:
            logger.warning("skip %s: length %d != expected %d", h5_path, len(data), expected)
            return False

        green_ab = data[nlag + 1 :]
        green_ba = data[nlag - 1 :: -1]
        time_axis = np.arange(1, nlag + 1) * dt

        max_amp = max(float(np.max(np.abs(green_ab))), float(np.max(np.abs(green_ba))))
        if max_amp > 0:
            green_ab = green_ab / max_amp
            green_ba = green_ba / max_amp

        out_dat_path.parent.mkdir(parents=True, exist_ok=True)
        with out_dat_path.open("w", encoding="utf-8") as fout:
            fout.write(f"{lon_src:.6f}  {lat_src:.6f}\n")
            fout.write(f"{lon_rec:.6f}  {lat_rec:.6f}\n")
            for time_s, ab, ba in zip(time_axis, green_ab, green_ba):
                fout.write(f"{time_s:.4f}  {ab:.8e}  {ba:.8e}\n")
        return True
    except Exception as exc:
        logger.error("convert failed %s: %s", h5_path, exc)
        return False


def load_station_coordinates(stack_root: Path, component: str = "ZZ"):
    coords = {}  # type: Dict[str, Tuple[float, float]]
    for source_dir in sorted(path for path in stack_root.iterdir() if path.is_dir()):
        station = source_dir.name
        self_files = sorted((source_dir / station).glob("*.h5"))
        if not self_files:
            continue
        attrs = _stack_attrs_from_file(self_files[0], component)
        if attrs is None:
            continue
        coords[station] = (float(attrs.get("latS", 0.0)), float(attrs.get("lonS", 0.0)))
    return coords


def _passes_distance_filter(
    source: str,
    receiver: str,
    station_coords,
    min_distance_km: float,
    max_distance_km: Optional[float],
) -> bool:
    if not station_coords:
        return True
    if source not in station_coords or receiver not in station_coords:
        return True
    lat_src, lon_src = station_coords[source]
    lat_rec, lon_rec = station_coords[receiver]
    distance_km = _distance_km(lat_src, lon_src, lat_rec, lon_rec)
    if distance_km < min_distance_km:
        return False
    if max_distance_km is not None and distance_km > max_distance_km:
        return False
    return True


def iter_stack_pairs(
    stack_root: Path,
    source_glob: str,
    receiver_glob: str,
    include_self: bool,
    include_reverse_duplicates: bool,
    station_coords=None,
    min_distance_km: float = 0.0,
    max_distance_km: Optional[float] = None,
):
    seen = set()  # type: Set[Tuple[str, str]]
    for source_dir in sorted(path for path in stack_root.iterdir() if path.is_dir()):
        source = source_dir.name
        if not fnmatch.fnmatch(source, source_glob):
            continue
        for receiver_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            receiver = receiver_dir.name
            if not fnmatch.fnmatch(receiver, receiver_glob):
                continue
            if not include_self and source == receiver:
                continue
            pair_key = tuple(sorted((source, receiver)))
            if not include_reverse_duplicates and pair_key in seen:
                continue
            seen.add(pair_key)
            if not _passes_distance_filter(
                source,
                receiver,
                station_coords,
                min_distance_km,
                max_distance_km,
            ):
                continue
            h5_files = sorted(receiver_dir.glob("*.h5"))
            if h5_files:
                yield source, receiver, h5_files[0]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert 2014 1D-1D NoisePy STACK h5 files to DAT.")
    parser.add_argument("--stack-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--component", default="ZZ")
    parser.add_argument("--source-glob", default="1D.*")
    parser.add_argument("--receiver-glob", default="1D.*")
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--include-reverse-duplicates", action="store_true")
    parser.add_argument("--min-distance-km", type=float, default=0.0)
    parser.add_argument("--max-distance-km", type=float)
    parser.add_argument("--limit", type=int, help="Optional maximum number of converted pairs.")
    parser.add_argument(
        "--num-shards",
        "--num_shards",
        type=int,
        default=1,
        help="Total conversion shards; process pair index %% num_shards == shard_index.",
    )
    parser.add_argument(
        "--shard-index",
        "--shard_index",
        type=int,
        default=0,
        help="Current conversion shard index, 0 <= shard_index < num_shards.",
    )
    parser.add_argument(
        "--stationxml-dir",
        type=Path,
        help="Optional StationXML directory used to supply station coordinates.",
    )
    parser.add_argument(
        "--allow-zero-ngood",
        action="store_true",
        help="Convert nonzero stacks even when ngood metadata is 0.",
    )
    args = parser.parse_args(argv)

    if not args.stack_root.is_dir():
        logger.error("STACK root does not exist: %s", args.stack_root)
        return 1
    if args.num_shards < 1:
        logger.error("--num-shards must be >= 1")
        return 1
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        logger.error("--shard-index must satisfy 0 <= shard_index < num_shards")
        return 1

    logger.info("STACK root: %s", args.stack_root)
    logger.info("DAT output: %s", args.out_dir)
    logger.info("component: %s", args.component)
    logger.info("source_glob=%s receiver_glob=%s", args.source_glob, args.receiver_glob)
    logger.info("stationxml_dir=%s", args.stationxml_dir if args.stationxml_dir else "disabled")
    logger.info(
        "distance filter: min=%.2f km max=%s",
        args.min_distance_km,
        "none" if args.max_distance_km is None else f"{args.max_distance_km:.2f} km",
    )
    logger.info("conversion shard: %d/%d", args.shard_index, args.num_shards)
    logger.info("allow_zero_ngood=%s", args.allow_zero_ngood)

    success = 0
    failed = 0
    skipped = 0
    shard_skipped = 0
    station_coords = None
    if args.stationxml_dir is not None:
        if not args.stationxml_dir.is_dir():
            logger.error("StationXML directory does not exist: %s", args.stationxml_dir)
            return 1
        station_coords = load_stationxml_coordinates(args.stationxml_dir)
        logger.info("loaded %d station coordinate aliases from StationXML", len(station_coords))
    elif args.min_distance_km > 0.0 or args.max_distance_km is not None:
        station_coords = load_station_coordinates(args.stack_root, component=args.component)
        logger.info("loaded %d station coordinates for distance prefilter", len(station_coords))

    for pair_index, (source, receiver, h5_path) in enumerate(iter_stack_pairs(
        args.stack_root,
        source_glob=args.source_glob,
        receiver_glob=args.receiver_glob,
        include_self=args.include_self,
        include_reverse_duplicates=args.include_reverse_duplicates,
        station_coords=station_coords,
        min_distance_km=args.min_distance_km,
        max_distance_km=args.max_distance_km,
    )):
        if pair_index % args.num_shards != args.shard_index:
            shard_skipped += 1
            continue
        pair_name = f"{source}__{receiver}"
        out_dat_path = args.out_dir / f"{pair_name}.dat"
        result = convert_one_stack(
            h5_path,
            out_dat_path,
            component=args.component,
            min_distance_km=args.min_distance_km,
            max_distance_km=args.max_distance_km,
            allow_zero_ngood=args.allow_zero_ngood,
            source_coords=station_coords.get(source) if station_coords else None,
            receiver_coords=station_coords.get(receiver) if station_coords else None,
        )
        if result is True:
            success += 1
            if success % 1000 == 0:
                logger.info("converted %d pairs", success)
        elif result is None:
            skipped += 1
        else:
            failed += 1
        if args.limit is not None and success >= args.limit:
            break

    logger.info(
        "done: success=%d skipped=%d shard_skipped=%d failed=%d output=%s",
        success,
        skipped,
        shard_skipped,
        failed,
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
