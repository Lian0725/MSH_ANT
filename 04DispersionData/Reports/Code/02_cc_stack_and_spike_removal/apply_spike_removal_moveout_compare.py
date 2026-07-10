#!/usr/bin/env python3
"""Apply 1 s spike removal to a moveout subset and compare before/after moveout plots."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import math
import platform
import shutil
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPIKE_MODULE_PATH = PROJECT_ROOT / "scripts" / "02_cc" / "detect_1s_spikes_wang_style.py"
FIG2_MODULE_PATH = PROJECT_ROOT / "scripts" / "06_plotting" / "reproduce_wang_figure2_1d4529.py"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SPIKE = load_module(SPIKE_MODULE_PATH, "spike_detect_module")
FIG2 = load_module(FIG2_MODULE_PATH, "wang_fig2_module")


DEFAULT_STACK_ROOT = Path("/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK")
DEFAULT_METADATA_ROOT = Path("/mnt/data_hdd/lgx/MSH_ANT/data/metadata/2014/1D")
DEFAULT_OUTPUT = Path("/mnt/data_hdd/MSH_ANT/parameter_tests/1d_moveout_before_after_spike_removal_20260625")
DEFAULT_SOURCE = "1D.4529"
DEFAULT_BBOX = FIG2.BBox(
    minlat=46.1384,
    maxlat=46.1595,
    minlon=-122.3363,
    maxlon=-122.0297,
)


def load_moveout_rows(
    stack_root: Path,
    metadata_root: Path,
    source_code: str,
    bbox: FIG2.BBox,
    lag_window_s: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    station_coords = FIG2.load_station_coordinates(metadata_root)
    source_coords = station_coords[source_code]
    spike_station_coords = {
        code: (float(values["latitude"]), float(values["longitude"]))
        for code, values in station_coords.items()
    }
    selected = FIG2.select_receivers_in_bbox(station_coords, bbox, source_code)
    rows: list[dict[str, object]] = []
    for receiver in selected:
        receiver_code = str(receiver["code"])
        candidates = [
            (Path(stack_root) / source_code / receiver_code, "source_receiver"),
            (Path(stack_root) / receiver_code / source_code, "receiver_source"),
        ]
        record = None
        for receiver_dir, orientation in candidates:
            h5_files = sorted(receiver_dir.glob("*.h5"))
            if not h5_files:
                continue
            with h5py.File(h5_files[0], "r") as handle:
                dataset = handle["AuxiliaryData/Allstack_pws/ZZ"]
                data = np.asarray(dataset[:], dtype=np.float64).squeeze()
                if orientation == "receiver_source":
                    data = data[::-1].copy()
                record = {
                    "source": source_code,
                    "receiver": receiver_code,
                    "data": data,
                    "dt": float(dataset.attrs["dt"]),
                    "maxlag": float(dataset.attrs["maxlag"]),
                    "distance_km": float(SPIKE.dataset_distance_km(dataset.attrs, spike_station_coords)),
                    "path": str(h5_files[0]),
                    "pair_orientation": orientation,
                }
            break
        if record is None:
            continue
        time, trace = FIG2.extract_lag_window(
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


def positive_window_length(dt: float, end_s: float = 16.0) -> int:
    return int(round(end_s / dt)) + 1


def subtract_template_from_full_trace(
    data: np.ndarray,
    dt: float,
    maxlag: float,
    offsets: np.ndarray,
    template: np.ndarray,
    phase_s: float,
    *,
    start_second: int = 1,
    end_second: int = 15,
) -> tuple[np.ndarray, float]:
    values = np.asarray(data, dtype=float).copy()
    nlag = int(round(maxlag / dt))
    if values.size != 2 * nlag + 1:
        nlag = (values.size - 1) // 2
    center = nlag
    pos_count = positive_window_length(dt, end_s=16.0)
    positive_time = np.arange(pos_count, dtype=float) * dt
    positive_trace = values[center : center + pos_count]
    scale = SPIKE.fit_repeating_spike_scale(
        positive_time,
        positive_trace,
        offsets,
        template,
        phase_s,
        start_second=start_second,
        end_second=end_second,
    )
    corrected = values.copy()
    for second in range(start_second, end_second + 1):
        center_time = second + phase_s
        for offset, amplitude in zip(offsets, template):
            shift = int(round((center_time + float(offset)) / dt))
            pos_index = center + shift
            neg_index = center - shift
            if 0 <= pos_index < corrected.size:
                corrected[pos_index] -= scale * float(amplitude)
            if 0 <= neg_index < corrected.size:
                corrected[neg_index] -= scale * float(amplitude)
    return corrected, float(scale)


def derive_spike_model(
    stack_root: Path,
    metadata_root: Path,
    *,
    template_source: str = "diagnostic",
    seed: int = 20260619,
) -> dict[str, object]:
    sample, candidate_count = SPIKE.reservoir_sample(SPIKE.iter_unique_paths(stack_root), size=500000, seed=seed)
    time, stacks, counts, _, used, failures = SPIKE.read_sample(
        sample,
        normalize_each_pair=True,
        metadata_root=metadata_root,
    )
    normalized = SPIKE.normalized_bins(stacks, counts)
    valid = np.flatnonzero(counts > 0)
    before = np.nanmean(normalized[valid], axis=0)
    before /= max(np.nanmax(np.abs(before)), 1e-12)
    template_phase_trace, _ = SPIKE.derive_template_trace(time, normalized, counts, source=template_source)
    phase_rows = SPIKE.one_second_phase_profile(time, template_phase_trace)
    best_phase = max(phase_rows, key=lambda row: row["median_abs"])
    template_reference = before if template_source == "diagnostic" else template_phase_trace
    offsets, template = SPIKE.build_repeating_spike_template(time, template_reference, float(best_phase["phase_s"]))
    return {
        "sample_size": len(sample),
        "candidate_count": candidate_count,
        "used": used,
        "failures": failures,
        "best_phase_s": float(best_phase["phase_s"]),
        "offsets": np.asarray(offsets, dtype=float),
        "template": np.asarray(template, dtype=float),
        "template_source": template_source,
    }


def clean_moveout_subset(
    rows_before: list[dict[str, object]],
    original_stack_root: Path,
    cleaned_stack_root: Path,
    model: dict[str, object],
) -> list[dict[str, object]]:
    cleaned_stack_root.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, object]] = []
    for row in rows_before:
        src_path = Path(str(row["path"]))
        relative = src_path.relative_to(original_stack_root)
        dst_path = cleaned_stack_root / relative
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        with h5py.File(dst_path, "r+") as handle:
            dataset = handle["AuxiliaryData/Allstack_pws/ZZ"]
            corrected, scale = subtract_template_from_full_trace(
                np.asarray(dataset[:], dtype=float),
                dt=float(dataset.attrs["dt"]),
                maxlag=float(dataset.attrs["maxlag"]),
                offsets=np.asarray(model["offsets"], dtype=float),
                template=np.asarray(model["template"], dtype=float),
                phase_s=float(model["best_phase_s"]),
            )
            dataset[...] = corrected.astype(dataset.dtype, copy=False)
            dataset.attrs["spike_removed"] = "YES"
            dataset.attrs["spike_phase_s"] = float(model["best_phase_s"])
            dataset.attrs["spike_template_source"] = str(model["template_source"])
            dataset.attrs["spike_scale"] = float(scale)
        csv_rows.append(
            {
                "source": row["source"],
                "receiver": row["receiver"],
                "distance_km": f"{float(row['distance_km']):.6f}",
                "original_path": str(src_path),
                "cleaned_path": str(dst_path),
                "scale": f"{float(scale):.6f}",
            }
        )
    return csv_rows


def render_moveout_compare(
    output_dir: Path,
    rows_before: list[dict[str, object]],
    rows_after: list[dict[str, object]],
    *,
    source_code: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
    })
    panels = FIG2.default_panel_specs()
    style = FIG2.default_plot_style()
    distance_max = max(
        25.0,
        math.ceil(
            max(
                max(float(row["distance_km"]) for row in rows_before),
                max(float(row["distance_km"]) for row in rows_after),
            )
            / 5.0
        )
        * 5.0,
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8), sharex=True, sharey=True, constrained_layout=True)
    for col, panel in enumerate(panels):
        FIG2.plot_record_section_panel(axes[0, col], rows_before, panel, style=style, distance_max=distance_max)
        FIG2.plot_record_section_panel(axes[1, col], rows_after, panel, style=style, distance_max=distance_max)
    axes[0, 0].set_ylabel("Distance (km)")
    axes[1, 0].set_ylabel("Distance (km)")
    for col in range(3):
        axes[0, col].text(
            0.98,
            1.02,
            "Before",
            transform=axes[0, col].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#7c2d12",
        )
        axes[1, col].text(
            0.98,
            1.02,
            "After",
            transform=axes[1, col].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#166534",
        )
    fig.suptitle(f"Moveout comparison for {source_code}: before vs after 1 s spike removal", fontsize=12)
    compare_path = output_dir / "moveout_before_after_compare.png"
    fig.savefig(compare_path, dpi=600)
    plt.close(fig)
    return compare_path


def render_single_moveout(output_dir: Path, source_code: str, source_coords: dict[str, float], rows: list[dict[str, object]], stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True, constrained_layout=True)
    distance_max = max(25.0, math.ceil(max(float(row["distance_km"]) for row in rows) / 5.0) * 5.0)
    style = FIG2.default_plot_style()
    for index, panel in enumerate(FIG2.default_panel_specs()):
        FIG2.plot_record_section_panel(axes[index], rows, panel, style=style, distance_max=distance_max)
    axes[0].set_ylabel("Distance (km)")
    figure_path = output_dir / f"{stem}.png"
    fig.savefig(figure_path, dpi=600)
    plt.close(fig)
    return figure_path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    output_dir: Path,
    *,
    original_stack_root: Path,
    cleaned_stack_root: Path,
    source_code: str,
    bbox: FIG2.BBox,
    rows_before: list[dict[str, object]],
    model: dict[str, object],
    scale_rows: list[dict[str, object]],
) -> Path:
    report_path = output_dir / "report.html"
    scales = [float(row["scale"]) for row in scale_rows]
    scale_median = float(np.median(scales)) if scales else float("nan")
    scale_abs_median = float(np.median(np.abs(scales))) if scales else float("nan")
    distances = [float(row["distance_km"]) for row in rows_before]
    text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Moveout 去尖峰前后对比</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1120px;margin:24px auto;padding:0 18px;color:#172033;line-height:1.65}}
img{{max-width:100%;border:1px solid #cbd5e1}}
code{{background:#f1f5f9;padding:2px 4px}}
.note{{background:#eff6ff;border:1px solid #60a5fa;padding:12px}}
.warn{{background:#fff7ed;border:1px solid #f59e0b;padding:12px}}
</style></head><body>
<h1>Moveout 去尖峰前后对比</h1>
<div class='note'>
计算主机：<code>{html.escape(platform.node())}</code>。原始 STACK 只读保留在 <code>{html.escape(str(original_stack_root))}</code>，去尖峰后的 moveout 子集副本写到 <code>{html.escape(str(cleaned_stack_root))}</code>。<br>
虚拟源：<code>{html.escape(source_code)}</code>。经纬度范围：<code>minlat={bbox.minlat:.4f}</code>, <code>maxlat={bbox.maxlat:.4f}</code>, <code>minlon={bbox.minlon:.4f}</code>, <code>maxlon={bbox.maxlon:.4f}</code>。<br>
参与 moveout 的接收台数量：<b>{len(rows_before)}</b>；最大台距：<b>{max(distances):.2f} km</b>。<br>
尖峰模板来源：<b>{html.escape(str(model['template_source']))}</b>；固定相位：<b>{float(model['best_phase_s']):.2f} s</b>；用于识别模板的 1D 台站对：<b>{int(model['used'])}</b>/<b>{int(model['candidate_count'])}</b>。
</div>
<div class='warn'>
这里为了稳妥，没有改写整套原始 STACK，而是只对当前 moveout 用到的台站对子集生成了一份去尖峰副本。这样既保留了去除前的数据，也能直接看 before/after moveout 的差异。
</div>
<h2>Before / After 并排对比</h2>
<img src='moveout_before_after_compare.png'>
<h2>单独查看</h2>
<p>去尖峰前：</p>
<img src='moveout_before.png'>
<p>去尖峰后：</p>
<img src='moveout_after.png'>
<h2>幅度拟合统计</h2>
<p>每个台站对不是用同一个固定振幅硬减，而是先拟合模板幅度再相减。拟合系数中位数 <b>{scale_median:.3f}</b>，绝对值中位数 <b>{scale_abs_median:.3f}</b>。详细见 <code>spike_scales.csv</code>。</p>
</body></html>"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--minlat", type=float, default=DEFAULT_BBOX.minlat)
    parser.add_argument("--maxlat", type=float, default=DEFAULT_BBOX.maxlat)
    parser.add_argument("--minlon", type=float, default=DEFAULT_BBOX.minlon)
    parser.add_argument("--maxlon", type=float, default=DEFAULT_BBOX.maxlon)
    parser.add_argument("--lag-window", type=float, default=15.0)
    parser.add_argument("--template-source", choices=["coherent", "diagnostic"], default="diagnostic")
    parser.add_argument("--seed", type=int, default=20260619)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bbox = FIG2.BBox(
        minlat=float(args.minlat),
        maxlat=float(args.maxlat),
        minlon=float(args.minlon),
        maxlon=float(args.maxlon),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    source_coords, rows_before = load_moveout_rows(
        stack_root=args.stack_root,
        metadata_root=args.metadata_root,
        source_code=args.source,
        bbox=bbox,
        lag_window_s=float(args.lag_window),
    )
    if not rows_before:
        raise RuntimeError("No moveout rows found before spike removal")

    model = derive_spike_model(
        stack_root=args.stack_root,
        metadata_root=args.metadata_root,
        template_source=args.template_source,
        seed=int(args.seed),
    )
    cleaned_stack_root = args.output / "stack_cleaned_subset"
    scale_rows = clean_moveout_subset(rows_before, args.stack_root, cleaned_stack_root, model)
    _, rows_after = load_moveout_rows(
        stack_root=cleaned_stack_root,
        metadata_root=args.metadata_root,
        source_code=args.source,
        bbox=bbox,
        lag_window_s=float(args.lag_window),
    )
    if len(rows_after) != len(rows_before):
        raise RuntimeError(f"Moveout row count mismatch after cleaning: before={len(rows_before)} after={len(rows_after)}")

    render_single_moveout(args.output, args.source, source_coords, rows_before, "moveout_before")
    render_single_moveout(args.output, args.source, source_coords, rows_after, "moveout_after")
    render_moveout_compare(args.output, rows_before, rows_after, source_code=args.source)
    FIG2.write_receiver_csv(args.output, rows_before)
    write_csv(args.output / "spike_scales.csv", scale_rows)
    report = write_report(
        args.output,
        original_stack_root=args.stack_root,
        cleaned_stack_root=cleaned_stack_root,
        source_code=args.source,
        bbox=bbox,
        rows_before=rows_before,
        model=model,
        scale_rows=scale_rows,
    )
    print(report)


if __name__ == "__main__":
    main()
