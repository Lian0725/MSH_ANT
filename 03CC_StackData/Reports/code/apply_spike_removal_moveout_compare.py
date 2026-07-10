#!/usr/bin/env python3
"""Apply 1 s spike removal to a moveout subset and compare before/after moveout plots."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import math
import os
import platform
import shutil
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MOVEOUT_CACHE_VERSION = 1


def resolve_project_root() -> Path:
    required = (
        Path("scripts") / "02_cc" / "detect_1s_spikes_wang_style.py",
        Path("scripts") / "06_plotting" / "reproduce_wang_figure2_1d4529.py",
    )
    candidates: list[Path] = []
    env_root = os.environ.get("MSH_ANT_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    file_path = Path(__file__).resolve()
    candidates.extend(list(file_path.parents[:6]))
    candidates.extend(
        [
            Path("/Users/lgx/Projects/MSH_ANT"),
            Path("/mnt/data_hdd/lgx/MSH_ANT/code/MSH_ANT"),
            Path("/home/lenovo/ai_projects/MSH_ANT"),
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / part).exists() for part in required):
            return candidate
    raise RuntimeError(
        "Could not locate MSH_ANT project root. Set MSH_ANT_PROJECT_ROOT to a checkout containing scripts/02_cc and scripts/06_plotting."
    )


PROJECT_ROOT = resolve_project_root()


def resolve_local_or_project_script(filename: str, relative_parts: tuple[str, ...]) -> Path:
    local = Path(__file__).resolve().with_name(filename)
    if local.exists():
        return local
    return PROJECT_ROOT.joinpath(*relative_parts)


SPIKE_MODULE_PATH = resolve_local_or_project_script(
    "detect_1s_spikes_wang_style.py",
    ("scripts", "02_cc", "detect_1s_spikes_wang_style.py"),
)
FIG2_MODULE_PATH = resolve_local_or_project_script(
    "reproduce_wang_figure2_1d4529.py",
    ("scripts", "06_plotting", "reproduce_wang_figure2_1d4529.py"),
)


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
DEFAULT_CACHE_NAME = "moveout_before_after_data.npz"
CODE_DIRNAME = "code"
IMAGE_DIRNAME = "images"


def code_dir(output_dir: Path) -> Path:
    return Path(output_dir) / CODE_DIRNAME


def image_dir(output_dir: Path) -> Path:
    return Path(output_dir) / IMAGE_DIRNAME


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


def rows_to_cache_arrays(rows: list[dict[str, object]], prefix: str) -> dict[str, np.ndarray]:
    if not rows:
        empty_float = np.zeros((0,), dtype=float)
        empty_text = np.zeros((0,), dtype="<U1")
        return {
            f"{prefix}_receiver": empty_text,
            f"{prefix}_time_s": np.zeros((0, 0), dtype=float),
            f"{prefix}_window_trace": np.zeros((0, 0), dtype=float),
            f"{prefix}_distance_km": empty_float,
            f"{prefix}_receiver_latitude": empty_float,
            f"{prefix}_receiver_longitude": empty_float,
            f"{prefix}_dt": empty_float,
            f"{prefix}_maxlag": empty_float,
            f"{prefix}_path": empty_text,
            f"{prefix}_pair_orientation": empty_text,
        }
    return {
        f"{prefix}_receiver": np.asarray([str(row["receiver"]) for row in rows]),
        f"{prefix}_time_s": np.stack([np.asarray(row["time_s"], dtype=float) for row in rows]),
        f"{prefix}_window_trace": np.stack([np.asarray(row["window_trace"], dtype=float) for row in rows]),
        f"{prefix}_distance_km": np.asarray([float(row["distance_km"]) for row in rows], dtype=float),
        f"{prefix}_receiver_latitude": np.asarray([float(row["receiver_latitude"]) for row in rows], dtype=float),
        f"{prefix}_receiver_longitude": np.asarray([float(row["receiver_longitude"]) for row in rows], dtype=float),
        f"{prefix}_dt": np.asarray([float(row["dt"]) for row in rows], dtype=float),
        f"{prefix}_maxlag": np.asarray([float(row["maxlag"]) for row in rows], dtype=float),
        f"{prefix}_path": np.asarray([str(row["path"]) for row in rows]),
        f"{prefix}_pair_orientation": np.asarray([str(row["pair_orientation"]) for row in rows]),
    }


def cache_arrays_to_rows(data: dict[str, np.ndarray], prefix: str, source_code: str) -> list[dict[str, object]]:
    receivers = data[f"{prefix}_receiver"]
    rows: list[dict[str, object]] = []
    for index, receiver in enumerate(receivers):
        rows.append(
            {
                "source": source_code,
                "receiver": str(receiver),
                "time_s": np.asarray(data[f"{prefix}_time_s"][index], dtype=float),
                "window_trace": np.asarray(data[f"{prefix}_window_trace"][index], dtype=float),
                "distance_km": float(data[f"{prefix}_distance_km"][index]),
                "receiver_latitude": float(data[f"{prefix}_receiver_latitude"][index]),
                "receiver_longitude": float(data[f"{prefix}_receiver_longitude"][index]),
                "dt": float(data[f"{prefix}_dt"][index]),
                "maxlag": float(data[f"{prefix}_maxlag"][index]),
                "path": str(data[f"{prefix}_path"][index]),
                "pair_orientation": str(data[f"{prefix}_pair_orientation"][index]),
            }
        )
    return rows


def save_moveout_cache(
    cache_path: Path,
    *,
    source_code: str,
    source_coords: dict[str, float],
    bbox: FIG2.BBox,
    rows_before: list[dict[str, object]],
    rows_after: list[dict[str, object]],
    model: dict[str, object],
    scale_rows: list[dict[str, object]],
    original_stack_root: Path,
    cleaned_stack_root: Path,
    lag_window_s: float,
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "cache_version": np.array(MOVEOUT_CACHE_VERSION, dtype=np.int64),
        "source_code": np.array(str(source_code)),
        "source_latitude": np.array(float(source_coords["latitude"]), dtype=float),
        "source_longitude": np.array(float(source_coords["longitude"]), dtype=float),
        "bbox": np.asarray([bbox.minlat, bbox.maxlat, bbox.minlon, bbox.maxlon], dtype=float),
        "lag_window_s": np.array(float(lag_window_s), dtype=float),
        "original_stack_root": np.array(str(original_stack_root)),
        "cleaned_stack_root": np.array(str(cleaned_stack_root)),
        "model_template_source": np.array(str(model["template_source"])),
        "model_best_phase_s": np.array(float(model["best_phase_s"]), dtype=float),
        "model_used": np.array(int(model.get("used", 0)), dtype=np.int64),
        "model_candidate_count": np.array(int(model.get("candidate_count", 0)), dtype=np.int64),
        "model_failures": np.array(int(model.get("failures", 0)), dtype=np.int64),
        "model_sample_size": np.array(int(model.get("sample_size", 0)), dtype=np.int64),
        "model_offsets": np.asarray(model["offsets"], dtype=float),
        "model_template": np.asarray(model["template"], dtype=float),
        "scale_source": np.asarray([str(row["source"]) for row in scale_rows]),
        "scale_receiver": np.asarray([str(row["receiver"]) for row in scale_rows]),
        "scale_distance_km": np.asarray([float(row["distance_km"]) for row in scale_rows], dtype=float),
        "scale_original_path": np.asarray([str(row["original_path"]) for row in scale_rows]),
        "scale_cleaned_path": np.asarray([str(row["cleaned_path"]) for row in scale_rows]),
        "scale_value": np.asarray([float(row["scale"]) for row in scale_rows], dtype=float),
    }
    payload.update(rows_to_cache_arrays(rows_before, "before"))
    payload.update(rows_to_cache_arrays(rows_after, "after"))
    np.savez_compressed(cache_path, **payload)
    return cache_path


def load_moveout_cache(cache_path: Path) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    source_code = str(arrays["source_code"])
    bbox_values = np.asarray(arrays["bbox"], dtype=float)
    cache = {
        "cache_path": str(cache_path),
        "cache_version": int(arrays["cache_version"]),
        "source_code": source_code,
        "source_coords": {
            "latitude": float(arrays["source_latitude"]),
            "longitude": float(arrays["source_longitude"]),
        },
        "bbox": FIG2.BBox(
            minlat=float(bbox_values[0]),
            maxlat=float(bbox_values[1]),
            minlon=float(bbox_values[2]),
            maxlon=float(bbox_values[3]),
        ),
        "lag_window_s": float(arrays["lag_window_s"]),
        "original_stack_root": str(arrays["original_stack_root"]),
        "cleaned_stack_root": str(arrays["cleaned_stack_root"]),
        "model": {
            "template_source": str(arrays["model_template_source"]),
            "best_phase_s": float(arrays["model_best_phase_s"]),
            "used": int(arrays["model_used"]),
            "candidate_count": int(arrays["model_candidate_count"]),
            "failures": int(arrays["model_failures"]),
            "sample_size": int(arrays["model_sample_size"]),
            "offsets": np.asarray(arrays["model_offsets"], dtype=float),
            "template": np.asarray(arrays["model_template"], dtype=float),
        },
        "rows_before": cache_arrays_to_rows(arrays, "before", source_code),
        "rows_after": cache_arrays_to_rows(arrays, "after", source_code),
        "scale_rows": [
            {
                "source": str(source),
                "receiver": str(receiver),
                "distance_km": f"{float(distance):.6f}",
                "original_path": str(original),
                "cleaned_path": str(cleaned),
                "scale": f"{float(scale):.6f}",
            }
            for source, receiver, distance, original, cleaned, scale in zip(
                arrays["scale_source"],
                arrays["scale_receiver"],
                arrays["scale_distance_km"],
                arrays["scale_original_path"],
                arrays["scale_cleaned_path"],
                arrays["scale_value"],
            )
        ],
    }
    return cache


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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def optional_image_section(output_dir: Path, relative_path: str, title: str, caption: str) -> str:
    path = Path(output_dir) / relative_path
    if not path.exists():
        return ""
    return (
        f"<h2>{html.escape(title)}</h2>"
        f"<p>{html.escape(caption)}</p>"
        f"<img src='{html.escape(relative_path)}'>"
    )


def optional_code_section(output_dir: Path, files: list[tuple[str, str]]) -> str:
    items = []
    for relative_path, description in files:
        path = Path(output_dir) / relative_path
        if not path.exists():
            continue
        filename = Path(relative_path).name
        items.append(
            "<li>"
            f"<a href='{html.escape(relative_path)}'><code>{html.escape(filename)}</code></a>"
            f"：{html.escape(description)}"
            "</li>"
        )
    if not items:
        return ""
    return (
        "<h2>去除尖峰的代码</h2>"
        "<p>下面这些代码文件已经放在当前目录，报告里直接给出入口，后续查实现或继续改图时不需要再回主项目里找。</p>"
        f"<ul>{''.join(items)}</ul>"
    )


def optional_link_list_section(output_dir: Path, title: str, intro: str, files: list[tuple[str, str]]) -> str:
    items = []
    for relative_path, description in files:
        path = Path(output_dir) / relative_path
        if not path.exists():
            continue
        filename = Path(relative_path).name
        items.append(
            "<li>"
            f"<a href='{html.escape(relative_path)}'><code>{html.escape(filename)}</code></a>"
            f"：{html.escape(description)}"
            "</li>"
        )
    if not items:
        return ""
    return f"<h2>{html.escape(title)}</h2><p>{html.escape(intro)}</p><ul>{''.join(items)}</ul>"


def optional_wang_figure3_uniform_section(output_dir: Path) -> str:
    image_subdir = f"{IMAGE_DIRNAME}/wang_figure3_uniform"
    code_subdir = f"{CODE_DIRNAME}/wang_figure3_uniform"
    combined = f"{image_subdir}/wang_figure3_four_panel_uniform.png"
    panel_names = ["panel_a.png", "panel_b.png", "panel_c.png", "panel_d.png"]
    panel_paths = [f"{image_subdir}/{name}" for name in panel_names]
    if not (Path(output_dir) / combined).exists():
        return ""

    single_panels = "".join(
        f"<p><code>{html.escape(Path(path).name)}</code></p><img src='{html.escape(path)}'>"
        for path in panel_paths
        if (Path(output_dir) / path).exists()
    )
    code_links = optional_link_list_section(
        output_dir,
        "Wang Figure 3 绘图代码",
        "这组四联图的重绘代码和拼图代码都保存在这里，后续如果只想改版式、字号或布局，可以直接改这些脚本。",
        [
            (
                f"{code_subdir}/render_wang_figure3_from_npz.py",
                "从 4 个 NPZ 数值文件直接重绘 panel A-D，并输出统一尺寸的四联图。",
            ),
            (
                f"{code_subdir}/compose_wang_figure3_panels.py",
                "把已经生成好的 4 张单图再次拼成统一 2×2 组合图。",
            ),
            (
                f"{code_subdir}/panel_data_index.md",
                "记录 panel A-D 对应的原始图片来源、NPZ 路径和数组键名。",
            ),
        ],
    )
    data_links = optional_link_list_section(
        output_dir,
        "Wang Figure 3 NPZ 数据",
        "下面 4 个 NPZ 是对应单图的数值缓存。后续改图时优先读这些文件，不需要重新从更上游结果再推一次。",
        [
            (
                f"{image_subdir}/wang_figure3a_panel_data.npz",
                "Panel A 的距离分箱诊断残差数据。",
            ),
            (
                f"{image_subdir}/subset_strict_data.npz",
                "Panel B 的 0-1 s 波形子集数据。",
            ),
            (
                f"{image_subdir}/distance_bin_wiggle_panel_data.npz",
                "Panel C 的去尖峰后距离分箱波形数据。",
            ),
            (
                f"{image_subdir}/wang_figure3d_bandpassed_fill_scaled_2p00_no_fill_data.npz",
                "Panel D 的 bandpassed distance-bin 显示数据。",
            ),
        ],
    )
    return (
        "<h2>Wang Figure 3 四联图</h2>"
        "<p>这里补充保存 Wang Figure 3 的统一四联图、4 张单图、绘图代码和 4 个对应 NPZ 数值文件，方便后续直接改图。</p>"
        f"<p><code>{html.escape(Path(combined).name)}</code></p><img src='{html.escape(combined)}'>"
        "<h2>Wang Figure 3 单图</h2>"
        f"{single_panels}"
        f"{code_links}"
        f"{data_links}"
    )


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
    cache_path: Path,
    loaded_from_cache: bool,
) -> Path:
    report_path = output_dir / "report.html"
    image_root = image_dir(output_dir)
    code_root = code_dir(output_dir)
    scales = [float(row["scale"]) for row in scale_rows]
    scale_median = float(np.median(scales)) if scales else float("nan")
    scale_abs_median = float(np.median(np.abs(scales))) if scales else float("nan")
    distances = [float(row["distance_km"]) for row in rows_before]
    spike_sections = "".join(
        [
            optional_image_section(
                output_dir,
                f"{IMAGE_DIRNAME}/integer_spike_contrast_before_after.png",
                "尖峰显影前后图",
                "这张图是尖峰最明显的一版 before/after 对比图。左侧为去尖峰前，右侧为去尖峰后，用来直观看 1 s 重复尖峰是否被压下去。",
            ),
            optional_image_section(
                output_dir,
                f"{IMAGE_DIRNAME}/coherent_before_after.png",
                "相干叠加前后图",
                "这张图保留原始 coherent 叠加前后对比，可辅助判断去尖峰是否同时改变了主波形整体形态。",
            ),
            optional_image_section(
                output_dir,
                f"{IMAGE_DIRNAME}/spike_template.png",
                "尖峰模板图",
                "这是用于减除 1 s 重复尖峰的模板图。实际处理时不是用固定振幅硬减，而是每条道先拟合尺度，再减去 scale × template。",
            ),
            optional_code_section(
                output_dir,
                [
                    (
                        f"{CODE_DIRNAME}/detect_1s_spikes_wang_style.py",
                        "负责 1 s 重复尖峰的诊断、模板构建和幅度拟合。",
                    ),
                    (
                        f"{CODE_DIRNAME}/apply_spike_removal_moveout_compare.py",
                        "负责把模板减除应用到 moveout 子集，并输出 before/after 对比图和报告。",
                    ),
                ],
            ),
        ]
    )
    figure3_sections = optional_wang_figure3_uniform_section(output_dir)
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
尖峰模板来源：<b>{html.escape(str(model['template_source']))}</b>；固定相位：<b>{float(model['best_phase_s']):.2f} s</b>；用于识别模板的 1D 台站对：<b>{int(model['used'])}</b>/<b>{int(model['candidate_count'])}</b>。<br>
缓存文件：<code>{html.escape(str(cache_path))}</code>；本次输出{'直接从 NPZ 重绘' if loaded_from_cache else '先写 NPZ 再出图'}。<br>
目录结构：图片与缓存数据放在 <code>{html.escape(str(image_root))}</code>，代码副本放在 <code>{html.escape(str(code_root))}</code>。
</div>
<div class='warn'>
这里为了稳妥，没有改写整套原始 STACK，而是只对当前 moveout 用到的台站对子集生成了一份去尖峰副本。这样既保留了去除前的数据，也能直接看 before/after moveout 的差异。以后如果只是调颜色、改参考线或改版式，可以直接读这个 NPZ，不需要重新回读全部叠加波形。
</div>
<h2>Before / After 并排对比</h2>
<img src='{IMAGE_DIRNAME}/moveout_before_after_compare.png'>
<h2>单独查看</h2>
<p>去尖峰前：</p>
<img src='{IMAGE_DIRNAME}/moveout_before.png'>
<p>去尖峰后：</p>
<img src='{IMAGE_DIRNAME}/moveout_after.png'>
<h2>幅度拟合统计</h2>
<p>每个台站对不是用同一个固定振幅硬减，而是先拟合模板幅度再相减。拟合系数中位数 <b>{scale_median:.3f}</b>，绝对值中位数 <b>{scale_abs_median:.3f}</b>。详细见 <code>{IMAGE_DIRNAME}/spike_scales.csv</code>。</p>
{spike_sections}
{figure3_sections}
</body></html>"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def render_products_from_cache(cache: dict[str, object], output_dir: Path) -> dict[str, Path]:
    image_output_dir = image_dir(output_dir)
    image_output_dir.mkdir(parents=True, exist_ok=True)
    before_path = render_single_moveout(
        image_output_dir,
        str(cache["source_code"]),
        dict(cache["source_coords"]),
        list(cache["rows_before"]),
        "moveout_before",
    )
    after_path = render_single_moveout(
        image_output_dir,
        str(cache["source_code"]),
        dict(cache["source_coords"]),
        list(cache["rows_after"]),
        "moveout_after",
    )
    compare_path = render_moveout_compare(
        image_output_dir,
        list(cache["rows_before"]),
        list(cache["rows_after"]),
        source_code=str(cache["source_code"]),
    )
    return {
        "moveout_before": before_path,
        "moveout_after": after_path,
        "moveout_before_after_compare": compare_path,
    }


def write_outputs_from_cache(
    cache: dict[str, object],
    output_dir: Path,
    *,
    cache_path: Path,
    loaded_from_cache: bool,
) -> Path:
    render_products_from_cache(cache, output_dir)
    image_output_dir = image_dir(output_dir)
    FIG2.write_receiver_csv(image_output_dir, list(cache["rows_before"]))
    write_csv(image_output_dir / "spike_scales.csv", list(cache["scale_rows"]))
    return write_report(
        output_dir,
        original_stack_root=Path(str(cache["original_stack_root"])),
        cleaned_stack_root=Path(str(cache["cleaned_stack_root"])),
        source_code=str(cache["source_code"]),
        bbox=cache["bbox"],
        rows_before=list(cache["rows_before"]),
        model=dict(cache["model"]),
        scale_rows=list(cache["scale_rows"]),
        cache_path=cache_path,
        loaded_from_cache=loaded_from_cache,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument("--from-cache", type=Path, default=None)
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
    if args.from_cache is not None:
        cache = load_moveout_cache(args.from_cache)
        report = write_outputs_from_cache(
            cache,
            args.output,
            cache_path=args.from_cache,
            loaded_from_cache=True,
        )
        print(report)
        return
    bbox = FIG2.BBox(
        minlat=float(args.minlat),
        maxlat=float(args.maxlat),
        minlon=float(args.minlon),
        maxlon=float(args.maxlon),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_file or (image_dir(args.output) / DEFAULT_CACHE_NAME)
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

    save_moveout_cache(
        cache_path,
        source_code=args.source,
        source_coords=source_coords,
        bbox=bbox,
        rows_before=rows_before,
        rows_after=rows_after,
        model=model,
        scale_rows=scale_rows,
        original_stack_root=args.stack_root,
        cleaned_stack_root=cleaned_stack_root,
        lag_window_s=float(args.lag_window),
    )
    cache = load_moveout_cache(cache_path)
    report = write_outputs_from_cache(
        cache,
        args.output,
        cache_path=cache_path,
        loaded_from_cache=False,
    )
    print(report)


if __name__ == "__main__":
    main()
