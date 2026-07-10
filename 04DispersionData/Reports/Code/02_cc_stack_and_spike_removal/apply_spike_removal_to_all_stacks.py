#!/usr/bin/env python3
"""Apply 1 s spike removal to every stacked 1D CCF H5 and write a sibling output tree."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPIKE_MODULE_PATH = PROJECT_ROOT / "scripts" / "02_cc" / "detect_1s_spikes_wang_style.py"
MOVEOUT_MODULE_PATH = PROJECT_ROOT / "scripts" / "02_cc" / "apply_spike_removal_moveout_compare.py"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SPIKE = load_module(SPIKE_MODULE_PATH, "spike_detect_module_for_all")
MOVEOUT = load_module(MOVEOUT_MODULE_PATH, "moveout_spike_compare_for_all")


DEFAULT_STACK_ROOT = Path("/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK")
DEFAULT_METADATA_ROOT = Path("/mnt/data_hdd/lgx/MSH_ANT/data/metadata/2014/1D")


def default_output_root(stack_root: Path, label: str = "SPIKE_REMOVED") -> Path:
    return Path(stack_root).parent / f"{Path(stack_root).name}_{label}"


def iter_stack_files(stack_root: Path):
    yield from sorted(Path(stack_root).rglob("*.h5"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_stack_tree(
    *,
    original_stack_root: Path,
    cleaned_stack_root: Path,
    model: dict[str, object],
    limit: int | None = None,
    progress_every: int = 1000,
) -> list[dict[str, object]]:
    cleaned_stack_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for index, src_path in enumerate(iter_stack_files(original_stack_root), start=1):
        if limit is not None and index > limit:
            break
        relative = src_path.relative_to(original_stack_root)
        dst_path = cleaned_stack_root / relative
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        with h5py.File(dst_path, "r+") as dst_handle:
            dataset = dst_handle["AuxiliaryData/Allstack_pws/ZZ"]
            corrected, scale = MOVEOUT.subtract_template_from_full_trace(
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
        manifest_rows.append(
            {
                "relative_path": str(relative),
                "original_path": str(src_path),
                "cleaned_path": str(dst_path),
                "scale": f"{float(scale):.6f}",
            }
        )
        if progress_every > 0 and index % progress_every == 0:
            print(f"[progress] cleaned {index} stack files", flush=True)
    return manifest_rows


def write_report(
    output_root: Path,
    *,
    original_stack_root: Path,
    cleaned_stack_root: Path,
    model: dict[str, object],
    manifest_rows: list[dict[str, object]],
) -> Path:
    report_path = output_root / "report.html"
    scales = np.array([float(row["scale"]) for row in manifest_rows], dtype=float) if manifest_rows else np.array([])
    scale_median = float(np.median(scales)) if scales.size else float("nan")
    scale_abs_median = float(np.median(np.abs(scales))) if scales.size else float("nan")
    report_path.write_text(
        f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>全量 STACK 去尖峰报告</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1080px;margin:24px auto;padding:0 18px;color:#172033;line-height:1.65}}
code{{background:#f1f5f9;padding:2px 4px}}
.note{{background:#eff6ff;border:1px solid #60a5fa;padding:12px}}
</style></head><body>
<h1>全量 STACK 去尖峰报告</h1>
<div class='note'>
原始 STACK 目录：<code>{html.escape(str(original_stack_root))}</code><br>
去尖峰后目录：<code>{html.escape(str(cleaned_stack_root))}</code><br>
处理文件数：<b>{len(manifest_rows)}</b><br>
模板来源：<b>{html.escape(str(model['template_source']))}</b>；固定相位：<b>{float(model['best_phase_s']):.2f} s</b><br>
模板识别使用台站对：<b>{int(model['used'])}</b>/<b>{int(model['candidate_count'])}</b><br>
拟合系数中位数：<b>{scale_median:.3f}</b>；绝对值中位数：<b>{scale_abs_median:.3f}</b>
</div>
<p>详细逐文件结果见 <code>spike_scales.csv</code>，模板形状见 <code>spike_template.csv</code>。</p>
</body></html>""",
        encoding="utf-8",
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--label", default="SPIKE_REMOVED")
    parser.add_argument("--template-source", choices=["coherent", "diagnostic"], default="diagnostic")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or default_output_root(args.stack_root, args.label)
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    model = MOVEOUT.derive_spike_model(
        stack_root=args.stack_root,
        metadata_root=args.metadata_root,
        template_source=args.template_source,
        seed=int(args.seed),
    )
    manifest_rows = clean_stack_tree(
        original_stack_root=args.stack_root,
        cleaned_stack_root=output_root,
        model=model,
        limit=args.limit,
        progress_every=int(args.progress_every),
    )
    if not manifest_rows:
        raise RuntimeError("No stack files were cleaned")

    write_csv(output_root / "spike_scales.csv", manifest_rows)
    write_csv(
        output_root / "spike_template.csv",
        [
            {"offset_s": float(offset), "amplitude": float(value)}
            for offset, value in zip(np.asarray(model["offsets"], dtype=float), np.asarray(model["template"], dtype=float))
        ],
    )
    report = write_report(
        output_root,
        original_stack_root=args.stack_root,
        cleaned_stack_root=output_root,
        model=model,
        manifest_rows=manifest_rows,
    )
    print(report)


if __name__ == "__main__":
    main()
