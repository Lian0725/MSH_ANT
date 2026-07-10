#!/usr/bin/env python3
"""Reproduce Wang et al. (2017) Figure 2 using 1D.4529 as the virtual source."""

from __future__ import annotations

import argparse
import csv
import html
import math
import platform
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.signal import butter, sosfiltfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms


DEFAULT_STACK_ROOT = Path("/mnt/data_hdd/MSH_ANT/stack/2014/1D_STACK")
DEFAULT_METADATA_ROOT = Path("/mnt/data_hdd/MSH_ANT/data/metadata/2014/1D")
DEFAULT_OUTPUT = Path("/mnt/data_hdd/MSH_ANT/parameter_tests/1d_wang_figure2_1d4529_20260621")
DEFAULT_SOURCE = "1D.4529"
NS = {"s": "http://www.fdsn.org/xml/station/1"}


@dataclass(frozen=True)
class BBox:
    minlat: float
    maxlat: float
    minlon: float
    maxlon: float


@dataclass(frozen=True)
class PanelSpec:
    label: str
    title: str
    period_band_s: tuple[float, float]
    velocity_lines_km_s: tuple[float, ...]


@dataclass(frozen=True)
class PlotStyle:
    amplitude_km: float
    line_width: float
    line_alpha: float
    top_pad_km: float
    bottom_pad_km: float


DEFAULT_BBOX = BBox(
    minlat=46.1322,
    maxlat=46.1702,
    minlon=-122.3363,
    maxlon=-122.0297,
)
PAIR_WAVEFORM_FILENAME_RE = re.compile(r"^\d+_(1D\.\d+)__(1D\.\d+)_")


def default_panel_specs() -> list[PanelSpec]:
    return [
        PanelSpec("(a)", "1-10 s", (1.0, 10.0), (7.0, 3.0, 1.6)),
        PanelSpec("(b)", "1.5-2.5 s", (1.5, 2.5), (7.0, 3.0)),
        PanelSpec("(c)", "2.5-5 s", (2.5, 5.0), (3.0, 1.6)),
    ]


def default_plot_style() -> PlotStyle:
    return PlotStyle(
        amplitude_km=0.9,
        line_width=0.32,
        line_alpha=0.72,
        top_pad_km=1.0,
        bottom_pad_km=0.75,
    )


def bbox_from_args(args: argparse.Namespace) -> BBox:
    return BBox(
        minlat=float(args.minlat),
        maxlat=float(args.maxlat),
        minlon=float(args.minlon),
        maxlon=float(args.maxlon),
    )


def load_station_coordinates(metadata_root: Path) -> dict[str, dict[str, float]]:
    records: dict[str, dict[str, float]] = {}
    for xml_path in sorted(Path(metadata_root).glob("1D.*.xml")):
        root = ET.parse(xml_path).getroot()
        station = root.find(".//s:Station", NS)
        if station is None:
            continue
        latitude = float(station.findtext("s:Latitude", namespaces=NS))
        longitude = float(station.findtext("s:Longitude", namespaces=NS))
        records[xml_path.stem] = {"latitude": latitude, "longitude": longitude}
    return records


def select_receivers_in_bbox(
    station_coords: dict[str, dict[str, float]],
    bbox: BBox,
    source_code: str,
) -> list[dict[str, float | str]]:
    selected: list[dict[str, float | str]] = []
    for code, values in sorted(station_coords.items()):
        if code == source_code:
            continue
        lat = float(values["latitude"])
        lon = float(values["longitude"])
        if bbox.minlat <= lat <= bbox.maxlat and bbox.minlon <= lon <= bbox.maxlon:
            selected.append({"code": code, "latitude": lat, "longitude": lon})
    return selected


def period_band_to_frequency_band(period_band_s: tuple[float, float]) -> tuple[float, float]:
    low_period, high_period = period_band_s
    return 1.0 / high_period, 1.0 / low_period


def extract_lag_window(data: np.ndarray, dt: float, maxlag: float, window_s: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data, dtype=float)
    nlag = int(round(maxlag / dt))
    center = nlag
    if values.size != 2 * nlag + 1:
        nlag = (values.size - 1) // 2
        center = nlag
        maxlag = nlag * dt
    keep = int(round(min(window_s, maxlag) / dt))
    start = center - keep
    stop = center + keep + 1
    time = np.arange(-keep, keep + 1, dtype=float) * dt
    return time, values[start:stop]


def normalize_trace(values: np.ndarray, scale: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if scale is None:
        scale = float(np.nanmax(np.abs(values)))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(values)
    return values / scale


def load_source_receiver_stack(stack_root: Path, source_code: str, receiver_code: str) -> dict[str, object] | None:
    candidates = [
        (Path(stack_root) / source_code / receiver_code, "source_receiver"),
        (Path(stack_root) / receiver_code / source_code, "receiver_source"),
    ]
    for receiver_dir, orientation in candidates:
        h5_files = sorted(receiver_dir.glob("*.h5"))
        if not h5_files:
            continue
        with h5py.File(h5_files[0], "r") as handle:
            dataset = handle["AuxiliaryData/Allstack_pws/ZZ"]
            return {
                "source": source_code,
                "receiver": receiver_code,
                "data": np.asarray(dataset[:], dtype=np.float64).squeeze(),
                "dt": float(dataset.attrs["dt"]),
                "maxlag": float(dataset.attrs["maxlag"]),
                "distance_km": float(dataset.attrs["dist"]),
                "path": str(h5_files[0]),
                "pair_orientation": orientation,
            }
    return None


def bandpass_trace(values: np.ndarray, dt: float, period_band_s: tuple[float, float]) -> np.ndarray:
    freqmin, freqmax = period_band_to_frequency_band(period_band_s)
    nyquist = 0.5 / dt
    adjusted_freqmax = min(freqmax, nyquist * 0.999)
    if not (0.0 < freqmin < adjusted_freqmax < nyquist):
        return np.asarray(values, dtype=float)
    sos = butter(4, [freqmin, adjusted_freqmax], btype="bandpass", fs=1.0 / dt, output="sos")
    return sosfiltfilt(sos, np.asarray(values, dtype=float))


def build_figure2_records(
    stack_root: Path,
    metadata_root: Path,
    source_code: str,
    bbox: BBox,
    lag_window_s: float = 15.0,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    station_coords = load_station_coordinates(metadata_root)
    source_coords = station_coords[source_code]
    selected = select_receivers_in_bbox(station_coords, bbox, source_code)
    rows: list[dict[str, object]] = []
    for receiver in selected:
        record = load_source_receiver_stack(stack_root, source_code, str(receiver["code"]))
        if record is None:
            continue
        time, trace = extract_lag_window(
            np.asarray(record["data"], dtype=float),
            dt=float(record["dt"]),
            maxlag=float(record["maxlag"]),
            window_s=lag_window_s,
        )
        rows.append(
            {
                **record,
                "time_s": time,
                "window_trace": trace,
                "receiver_latitude": float(receiver["latitude"]),
                "receiver_longitude": float(receiver["longitude"]),
            }
        )
    rows.sort(key=lambda row: (float(row["distance_km"]), str(row["receiver"])))
    return source_coords, rows


def extract_receivers_from_pair_waveforms_dir(pair_waveforms_dir: Path) -> list[str]:
    receivers: list[str] = []
    for path in sorted(Path(pair_waveforms_dir).glob("*.png")):
        match = PAIR_WAVEFORM_FILENAME_RE.match(path.name)
        if match is None:
            continue
        receivers.append(match.group(2))
    return receivers


def filter_rows_by_receivers(rows: list[dict[str, object]], receivers: list[str]) -> list[dict[str, object]]:
    receiver_set = set(receivers)
    return [row for row in rows if str(row["receiver"]) in receiver_set]


def compute_distance_axis_limits(distance_max: float, style: PlotStyle) -> tuple[float, float]:
    return distance_max + style.bottom_pad_km, -style.top_pad_km


def plot_record_section_panel(
    ax,
    rows: list[dict[str, object]],
    panel: PanelSpec,
    style: PlotStyle,
    distance_max: float,
) -> None:
    for row in rows:
        time = np.asarray(row["time_s"], dtype=float)
        filtered = bandpass_trace(
            np.asarray(row["window_trace"], dtype=float),
            dt=float(row["dt"]),
            period_band_s=panel.period_band_s,
        )
        normalized = normalize_trace(filtered)
        distance = float(row["distance_km"])
        ax.plot(
            time,
            distance + normalized * style.amplitude_km,
            color="black",
            lw=style.line_width,
            alpha=style.line_alpha,
        )
    ax.set_xlim(-15, 15)
    ax.set_ylim(*compute_distance_axis_limits(distance_max, style))
    ax.set_xticks(np.arange(-15, 16, 5))
    ax.set_yticks(np.arange(0, distance_max + 0.1, 5))
    ax.set_xlabel("Time (sec)", labelpad=16)
    ax.text(0.02, 1.02, panel.label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.set_title(panel.title, fontsize=10, pad=8)
    ax.tick_params(top=True, right=True, length=3, width=0.8, direction="in")
    ref_y = np.array([0.0, distance_max], dtype=float)
    label_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    label_y_axes = -0.13
    for velocity in panel.velocity_lines_km_s:
        ref_t = ref_y / velocity
        ax.plot(ref_t, ref_y, color="#ef4444", lw=0.85)
        ax.text(
            ref_t[-1],
            label_y_axes,
            f"{velocity:g} km/s",
            transform=label_transform,
            color="#ef4444",
            fontsize=8,
            ha="center",
            va="top",
            clip_on=False,
        )


def render_figure2(output_dir: Path, source_code: str, source_coords: dict[str, float], rows: list[dict[str, object]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True, constrained_layout=True)
    distance_max = max(25.0, math.ceil(max(float(row["distance_km"]) for row in rows) / 5.0) * 5.0)
    style = default_plot_style()
    for index, panel in enumerate(default_panel_specs()):
        plot_record_section_panel(axes[index], rows, panel, style=style, distance_max=distance_max)
    axes[0].set_ylabel("Distance (km)")
    figure_path = output_dir / "wang_figure2_reproduced_1d4529.png"
    pdf_path = output_dir / "wang_figure2_reproduced_1d4529.pdf"
    fig.savefig(figure_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    return figure_path


def write_receiver_csv(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    path = output_dir / "receivers_used.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "receiver",
                "distance_km",
                "receiver_latitude",
                "receiver_longitude",
                "path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source": row["source"],
                    "receiver": row["receiver"],
                    "distance_km": f"{float(row['distance_km']):.6f}",
                    "receiver_latitude": f"{float(row['receiver_latitude']):.6f}",
                    "receiver_longitude": f"{float(row['receiver_longitude']):.6f}",
                    "path": row["path"],
                }
            )
    return path


def safe_filename_piece(value: object) -> str:
    text = str(value)
    for old in ("/", "\\", ":", " "):
        text = text.replace(old, "_")
    return text


def pair_waveform_filename(index: int, row: dict[str, object]) -> str:
    source = safe_filename_piece(row["source"])
    receiver = safe_filename_piece(row["receiver"])
    distance_tag = f"{float(row['distance_km']):06.2f}".replace(".", "p")
    return f"{index:04d}_{source}__{receiver}_{distance_tag}km.png"


def render_pair_waveform(output_dir: Path, row: dict[str, object], index: int) -> Path:
    pair_dir = output_dir / "pair_waveforms"
    pair_dir.mkdir(parents=True, exist_ok=True)
    figure_path = pair_dir / pair_waveform_filename(index, row)
    time = np.asarray(row["time_s"], dtype=float)
    trace = np.asarray(row["window_trace"], dtype=float)
    distance = float(row["distance_km"])
    panels = default_panel_specs()
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        filtered = bandpass_trace(trace, dt=float(row["dt"]), period_band_s=panel.period_band_s)
        normalized = normalize_trace(filtered)
        ax.plot(time, normalized, color="black", lw=0.8)
        ax.axhline(0.0, color="#94a3b8", lw=0.6)
        for velocity in panel.velocity_lines_km_s:
            arrival = distance / velocity
            if arrival <= 15.0:
                ax.axvline(arrival, color="#ef4444", lw=0.75, alpha=0.9)
                ax.axvline(-arrival, color="#ef4444", lw=0.55, alpha=0.35)
                ax.text(arrival, 0.92, f"{velocity:g}", color="#ef4444", fontsize=7, ha="center", va="top")
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel(panel.title)
        ax.tick_params(top=True, right=True, length=3, width=0.8, direction="in")
    axes[-1].set_xlim(-15, 15)
    axes[-1].set_xticks(np.arange(-15, 16, 5))
    axes[-1].set_xlabel("Time (sec)")
    fig.suptitle(
        f"{row['source']} -> {row['receiver']}   distance={distance:.3f} km   "
        f"lat={float(row['receiver_latitude']):.6f}, lon={float(row['receiver_longitude']):.6f}",
        fontsize=10,
    )
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    return figure_path


def render_pair_waveforms(output_dir: Path, rows: list[dict[str, object]]) -> list[Path]:
    paths: list[Path] = []
    for index, row in enumerate(rows, start=1):
        path = render_pair_waveform(output_dir, row, index)
        row["pair_waveform_path"] = path
        paths.append(path)
    write_pair_waveform_index(output_dir, rows)
    return paths


def write_pair_waveform_index(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    index_path = output_dir / "pair_waveforms_index.html"
    cards: list[str] = []
    for row in rows:
        image_path = Path(row["pair_waveform_path"])
        try:
            image_src = image_path.relative_to(output_dir).as_posix()
        except ValueError:
            image_src = image_path.as_posix()
        pair = f"{html.escape(str(row['source']))}__{html.escape(str(row['receiver']))}"
        cards.append(
            "<article>"
            f"<a href='{html.escape(image_src)}'><img src='{html.escape(image_src)}' loading='lazy'></a>"
            f"<h2>{pair}</h2>"
            f"<p>distance={float(row['distance_km']):.3f} km; "
            f"lat={float(row['receiver_latitude']):.6f}; "
            f"lon={float(row['receiver_longitude']):.6f}</p>"
            "</article>"
        )
    text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Pair Waveforms</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#172033;background:#f8fafc}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}}
article{{background:white;border:1px solid #cbd5e1;padding:10px}}
img{{width:100%;display:block;border:1px solid #e2e8f0}}
h1{{font-size:22px;margin:0 0 12px}}
h2{{font-size:14px;margin:8px 0 4px}}
p{{font-size:12px;margin:0;color:#475569}}
</style></head><body>
<h1>1D.4529 source-receiver pair waveforms ({len(rows)} pairs)</h1>
<div class='grid'>
{''.join(cards)}
</div>
</body></html>"""
    index_path.write_text(text, encoding="utf-8")
    return index_path


def write_report(
    output_dir: Path,
    source_code: str,
    source_coords: dict[str, float],
    rows: list[dict[str, object]],
    *,
    bbox: BBox = DEFAULT_BBOX,
    stack_root: Path = DEFAULT_STACK_ROOT,
    selected_pair_waveforms_dir: Path | None = None,
) -> Path:
    report = output_dir / "report.html"
    distances = [float(row["distance_km"]) for row in rows]
    text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Wang Figure 2 Reproduction</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1100px;margin:24px auto;padding:0 18px;color:#172033;line-height:1.65}}
img{{max-width:100%;border:1px solid #cbd5e1}}
code{{background:#f1f5f9;padding:2px 4px}}
.note{{background:#eff6ff;border:1px solid #60a5fa;padding:12px}}
</style></head><body>
<h1>Wang et al. (2017) Figure 2 复现</h1>
<div class='note'>计算主机：<code>{html.escape(platform.node())}</code>。虚拟源固定为 <code>{html.escape(source_code)}</code>（{source_coords['latitude']:.6f}, {source_coords['longitude']:.6f}）。STACK 根目录：<code>{html.escape(str(stack_root))}</code>。接收台按经纬度框 <code>minlat={bbox.minlat:.4f}</code>, <code>maxlat={bbox.maxlat:.4f}</code>, <code>minlon={bbox.minlon:.4f}</code>, <code>maxlon={bbox.maxlon:.4f}</code> 从 1D StationXML 中筛选。实际成功读取并作图的台站对数量：<b>{len(rows)}</b>；最大台距 <b>{max(distances):.2f} km</b>。</div>
<h2>Figure 2 复现图</h2>
<img src='wang_figure2_reproduced_1d4529.png'>
<h2>说明</h2>
<p>这张图使用 <code>Allstack_pws/ZZ</code> 单台对 STACK 结果，时间窗为 <code>-15~15 s</code>。三个面板的滤波周期带分别为 <code>1-10 s</code>、<code>1.5-2.5 s</code> 和 <code>2.5-5 s</code>，并画出论文中的红色速度参考线。</p>
<p>{f"本次在 bbox 候选台站中，进一步按保留下来的单台对图片目录 <code>{html.escape(str(selected_pair_waveforms_dir))}</code> 做二次筛选，只保留你人工挑过的接收台。" if selected_pair_waveforms_dir is not None else "本次直接使用 bbox 内所有可读接收台。"} </p>
<p>如果这版图看起来比论文更乱，主要原因通常有三点：一是当前经纬度框内实际叠加进来的接收台很多（本次为 <b>{len(rows)}</b> 个），道数更密；二是每条道都做了单独归一化，所以弱噪声也会被抬得和强信号一样显眼；三是 Wang 图中的“west-east zone”在绘图上可能比当前实现更接近一条狭窄测线，而不是把框内全部可用台站都直接画上去。本次新版绘图已额外给 <code>0 km</code> 上方和底部加了留白，并收小了单道振幅与线宽，便于看清近距离波形。</p>
<p>接收台清单见 <code>receivers_used.csv</code>。</p>
</body></html>"""
    report.write_text(text, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--lag-window", type=float, default=15.0)
    parser.add_argument("--minlat", type=float, default=DEFAULT_BBOX.minlat)
    parser.add_argument("--maxlat", type=float, default=DEFAULT_BBOX.maxlat)
    parser.add_argument("--minlon", type=float, default=DEFAULT_BBOX.minlon)
    parser.add_argument("--maxlon", type=float, default=DEFAULT_BBOX.maxlon)
    parser.add_argument("--export-pair-waveforms", action="store_true")
    parser.add_argument("--selected-pair-waveforms-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_coords, rows = build_figure2_records(
        stack_root=args.stack_root,
        metadata_root=args.metadata_root,
        source_code=args.source,
        bbox=bbox_from_args(args),
        lag_window_s=args.lag_window,
    )
    if args.selected_pair_waveforms_dir is not None:
        selected_receivers = extract_receivers_from_pair_waveforms_dir(args.selected_pair_waveforms_dir)
        rows = filter_rows_by_receivers(rows, selected_receivers)
    if not rows:
        raise RuntimeError("No usable source-receiver STACK rows found for Wang Figure 2 reproduction")
    args.output.mkdir(parents=True, exist_ok=True)
    render_figure2(args.output, args.source, source_coords, rows)
    write_receiver_csv(args.output, rows)
    if args.export_pair_waveforms:
        render_pair_waveforms(args.output, rows)
    report = write_report(
        args.output,
        args.source,
        source_coords,
        rows,
        bbox=bbox_from_args(args),
        stack_root=args.stack_root,
        selected_pair_waveforms_dir=args.selected_pair_waveforms_dir,
    )
    print(report)


if __name__ == "__main__":
    main()
