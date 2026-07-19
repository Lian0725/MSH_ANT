#!/usr/bin/env python3
"""Reproduce Wang et al. (2017) Figure 4 on the work server using all 1D pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import scipy
try:
    import yaml
except ModuleNotFoundError:
    yaml = None
from scipy.optimize import minimize
from scipy.signal import windows
import threadpoolctl
from threadpoolctl import threadpool_info

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wang_ftan_validation
from bensen_phase_ftan import (
    DatTrace,
    FtanConfig,
    PhaseConvention,
    ReferenceObservation,
    WangSnrResult,
    compute_wang_snr,
    evaluate_wang_left_qc,
    fit_right_column_slowness,
    fit_reference_dispersion,
    find_candidate_ridges,
    gaussian_alpha_for_distance,
    gaussian_filter_bank,
    measure_single_period,
    measure_phase_curve,
    normalized_log_energy,
    phase_matched_second_pass_ftan,
    prepare_phase_waveform,
    reference_fit_objective,
    resample_wang_target_periods,
    resample_wang_measurements,
    resolve_cycle_count,
    resolve_reference_cycles,
    select_fundamental_ridge,
)


TARGET_PERIODS_S = (3.0, 3.5, 4.0, 5.0)
FORMAL_STAGE_B_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
FIGURE4_PERIODS_S = (3.0, 4.0, 5.0)
FORMAL_REQUIRED_OUTPUTS = (
    "metadata.json",
    "frozen_parameters.json",
    "input_inventory.json",
    "reference_dispersion.csv",
    "reference_alias_solutions.csv",
    "candidate_grid_results.csv",
    "cycle_count_distribution.csv",
    "split_half_stability.csv",
    "split_half_membership.csv",
    "reference_spatial_diagnostics.csv",
    "phase_matching_comparison.csv",
    "measurements_raw.csv",
    "measurements_left_qc.csv",
    "measurements_corrected.csv",
    "measurements_right_qc.csv",
    "rejections.csv",
    "stage_counts.csv",
    "fit_summary.csv",
    "figures/wang_figure4_ftan_paper_scale.png",
    "figures/wang_figure4_ftan_full_range.png",
    "figures/reference_dispersion_stability.png",
    "figures/phase_convention_validation.png",
    "figures/triplet_closure.png",
    "report.html",
    "run.log",
)


def validate_formal_stage_b_thread_contract() -> None:
    """Require numerical thread limits inherited before formal Python start."""

    invalid = [
        f"{name}=1"
        for name in FORMAL_STAGE_B_THREAD_ENVIRONMENT
        if os.environ.get(name) != "1"
    ]
    if invalid:
        raise RuntimeError(
            "formal Stage B requires pre-import thread limits: "
            + ", ".join(invalid)
        )


_STAGE_B_THREADPOOL_SNAPSHOT_CACHE = None


def _stage_b_threadpool_snapshot() -> Tuple[Dict[str, object], ...]:
    global _STAGE_B_THREADPOOL_SNAPSHOT_CACHE
    if _STAGE_B_THREADPOOL_SNAPSHOT_CACHE is None:
        rows = tuple(
            {
                "user_api": str(info.get("user_api", "")),
                "internal_api": str(info.get("internal_api", "")),
                "prefix": str(info.get("prefix", "")),
                "num_threads": int(info.get("num_threads", 0)),
            }
            for info in threadpool_info()
        )
        if not rows or any(row["num_threads"] != 1 for row in rows):
            raise RuntimeError(
                "formal Stage B numerical backends must each use one thread"
            )
        _STAGE_B_THREADPOOL_SNAPSHOT_CACHE = rows
    return tuple(dict(row) for row in _STAGE_B_THREADPOOL_SNAPSHOT_CACHE)


def _probe_stage_b_worker_threadpool(_job: int) -> Dict[str, object]:
    time.sleep(0.05)
    return {
        "pid": os.getpid(),
        "backends": _stage_b_threadpool_snapshot(),
    }


def probe_stage_b_worker_threadpools(
    max_workers: int,
) -> Tuple[Dict[str, object], ...]:
    """Return real backend thread counts from an explicit fork pool."""

    validate_formal_stage_b_thread_contract()
    workers = int(max_workers)
    if workers < 1 or workers > 24:
        raise ValueError("Stage B probe workers must lie in [1, 24]")
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        rows = tuple(
            pool.map(
                _probe_stage_b_worker_threadpool,
                range(workers),
                chunksize=1,
            )
        )
    if len({int(row["pid"]) for row in rows}) != workers:
        raise RuntimeError("Stage B thread probe did not engage every worker")
    return tuple(sorted(rows, key=lambda row: int(row["pid"])))
SNR_THRESHOLD = 4.0
SIGNAL_VMIN_KM_S = 1.6
SIGNAL_VMAX_KM_S = 5.0
GROUP_VMAX_SHORT_KM_S = 3.0
GROUP_VMAX_LONG_KM_S = 3.3
GROUP_PICK_MAX_FRACTION_PERIOD = 0.25
SNR_GUARD_FRACTION_PERIOD = 0.5
MIN_NOISE_SAMPLES = 8
FTAN_VELOCITY_STEP_KM_S = 0.01
FTAN_SEED_COUNT = 10
FTAN_SEED_TOPK = 3
FTAN_ROUGHNESS_WEIGHT = 0.35
FTAN_CURVATURE_WEIGHT = 0.20


@dataclass(frozen=True)
class PhaseMeasurement:
    pair_name: str
    distance_km: float
    period_s: float
    group_time_s: float
    group_velocity_km_s: float
    leading_snr: float
    trailing_snr: float
    phi_tu_rad: float
    raw_travel_time_s: float


class MeasurementError(RuntimeError):
    """Expected, structured single-pair input or scientific failure."""

    def __init__(
        self,
        status: str,
        *,
        failure_kind: str = "expected_scientific_rejection",
        detail: Optional[str] = None,
    ) -> None:
        if not isinstance(status, str) or not status.strip():
            raise ValueError("MeasurementError status must be non-empty")
        if not isinstance(failure_kind, str) or not failure_kind.strip():
            raise ValueError(
                "MeasurementError failure_kind must be non-empty"
            )
        self.status = status.strip()
        self.failure_kind = failure_kind.strip()
        self.detail = None if detail is None else str(detail)
        message = self.status
        if self.detail:
            message += f": {self.detail}"
        super().__init__(message)


@dataclass(frozen=True)
class StackTrace:
    pair_name: str
    dt_s: float
    maxlag_s: float
    time_positive_s: np.ndarray
    positive_lag: np.ndarray
    negative_lag_reversed: np.ndarray
    symmetric: np.ndarray
    branch_mismatch: float

    def __post_init__(self) -> None:
        if not isinstance(self.pair_name, str) or not self.pair_name.strip():
            raise ValueError("pair_name must be non-empty")
        dt_s = float(self.dt_s)
        maxlag_s = float(self.maxlag_s)
        mismatch = float(self.branch_mismatch)
        arrays = tuple(
            np.array(getattr(self, name), dtype=float, copy=True)
            for name in (
                "time_positive_s",
                "positive_lag",
                "negative_lag_reversed",
                "symmetric",
            )
        )
        if (
            not np.isfinite(dt_s)
            or dt_s <= 0
            or not np.isfinite(maxlag_s)
            or maxlag_s <= 0
            or not np.isfinite(mismatch)
            or mismatch < 0
            or any(array.ndim != 1 for array in arrays)
            or any(array.size == 0 for array in arrays)
            or any(array.shape != arrays[0].shape for array in arrays)
            or any(np.any(~np.isfinite(array)) for array in arrays)
            or arrays[0][0] != 0.0
            or np.any(np.diff(arrays[0]) <= 0)
            or not np.allclose(
                arrays[3],
                0.5 * (arrays[1] + arrays[2]),
                rtol=0.0,
                atol=64.0 * np.finfo(float).eps,
            )
        ):
            raise ValueError("StackTrace fields are inconsistent")
        for array in arrays:
            array.setflags(write=False)
        object.__setattr__(self, "dt_s", dt_s)
        object.__setattr__(self, "maxlag_s", maxlag_s)
        object.__setattr__(self, "branch_mismatch", mismatch)
        for name, array in zip(
            (
                "time_positive_s",
                "positive_lag",
                "negative_lag_reversed",
                "symmetric",
            ),
            arrays,
        ):
            object.__setattr__(self, name, array)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_stage_b_freeze_decision(
    output_dir: Path,
    result: wang_ftan_validation.StageBRunResult,
) -> int:
    """Persist a Stage B decision and return nonzero for every failed gate."""

    if not isinstance(result, wang_ftan_validation.StageBRunResult):
        raise ValueError(
            "Stage B persistence requires the scientific orchestrator result"
        )
    output = Path(output_dir)
    ensure_dir(output)
    frozen_path = output / "frozen_parameters.json"
    decision_path = output / "stage_b_decision.json"
    evidence_path = output / "stage_b_validation_evidence.json"
    evidence_payload = None
    evidence_sha256 = None
    if isinstance(result.selection, wang_ftan_validation.StageBSelection):
        evidence_payload = (
            wang_ftan_validation.stage_b_validation_evidence_payload(
                result
            )
        )
        evidence_sha256 = (
            wang_ftan_validation.stage_b_validation_evidence_sha256(
                evidence_payload
            )
        )
        evidence_path.write_text(
            json.dumps(
                evidence_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    if result.return_code != 0:
        if frozen_path.exists():
            frozen_path.unlink()
        payload = {
            "stage_b_status": result.status,
            "formal_full_run_allowed": False,
            "candidate_count": len(result.candidate_results),
            "measurement_class_count": len(result.measurement_classes),
            "audit": dict(result.audit),
            "validation_evidence_json": (
                evidence_path.name if evidence_payload is not None else None
            ),
            "validation_evidence_sha256": evidence_sha256,
        }
        decision_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result.return_code
    manifest = result.frozen_parameters
    if manifest is None:
        raise ValueError("passed Stage B result is missing frozen parameters")
    if (
        evidence_payload is None
        or manifest["validation_table_sha256"] != evidence_sha256
    ):
        raise ValueError(
            "frozen parameters do not match persisted validation evidence"
        )
    frozen_path.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision_path.write_text(
        json.dumps(
            {
                "stage_b_status": "passed",
                "formal_full_run_allowed": True,
                "candidate_id": manifest["candidate_id"],
                "validation_evidence_json": evidence_path.name,
                "validation_evidence_sha256": evidence_sha256,
                "phase_matching_status_by_convention": {
                    phase: evidence.diagnostic.status
                    for phase, evidence in (
                        result.phase_matching_diagnostics.items()
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def execute_stage_b(
    output_dir: Path,
    **validation_kwargs,
) -> int:
    """Run the fixed scientific Stage B chain and persist its terminal state."""

    output = Path(output_dir)
    ensure_dir(output)
    validation_kwargs["phase_matched_second_pass_ftan"] = (
        phase_matched_second_pass_ftan
    )
    try:
        result = wang_ftan_validation.run_stage_b_validation(
            **validation_kwargs
        )
    except Exception as exc:
        for name in (
            "frozen_parameters.json",
            "stage_b_decision.json",
            "stage_b_validation_evidence.json",
        ):
            path = output / name
            if path.exists():
                path.unlink()
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "run_status": "failed",
                    "stage": "B",
                    "terminal_failure_reason": "stage_b_validation_error",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 2
    return write_stage_b_freeze_decision(output, result)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_stack_datasets(
    handle: h5py.File,
    component: str,
) -> List[h5py.Dataset]:
    aux = handle.get("AuxiliaryData")
    if aux is None:
        return []
    matches: List[h5py.Dataset] = []

    def collect(name: str, item) -> None:
        if isinstance(item, h5py.Dataset) and Path(name).name == component:
            matches.append(item)

    aux.visititems(collect)
    return matches


def _find_stack_dataset(handle: h5py.File, component: str = "ZZ"):
    """Compatibility entry returning the unique exact component dataset."""

    matches = _find_stack_datasets(handle, component)
    return matches[0] if len(matches) == 1 else None


def _read_unique_component_payload(
    path: Path,
    *,
    component: str = "ZZ",
) -> Tuple[np.ndarray, Dict[str, object]]:
    component_name = str(component).strip()
    if not component_name:
        raise ValueError("component must be non-empty")
    try:
        with h5py.File(Path(path), "r") as handle:
            matches = _find_stack_datasets(handle, component_name)
            if not matches:
                status = (
                    "missing_ZZ"
                    if component_name == "ZZ"
                    else "missing_component"
                )
                raise MeasurementError(status)
            if len(matches) != 1:
                raise MeasurementError(
                    "multiple_matching_components",
                    failure_kind="input_structure_error",
                    detail=component_name,
                )
            dataset = matches[0]
            attrs = dict(dataset.attrs)
            data = np.asarray(dataset[...])
    except MeasurementError:
        raise
    except (OSError, RuntimeError) as error:
        raise MeasurementError(
            "unreadable_hdf5",
            failure_kind="input_structure_error",
            detail=f"{type(error).__name__}: {error}",
        )
    return data, attrs


def _stack_trace_from_payload(
    data: np.ndarray,
    attrs: Dict[str, object],
    *,
    pair_name: str,
) -> StackTrace:
    try:
        data = np.asarray(data, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise MeasurementError(
            "invalid_data_type",
            failure_kind="input_structure_error",
            detail=f"{type(error).__name__}: {error}",
        )
    try:
        dt_s = float(attrs["dt"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise MeasurementError(
            "invalid_dt",
            failure_kind="input_structure_error",
        )
    try:
        maxlag_s = float(attrs["maxlag"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise MeasurementError(
            "invalid_maxlag",
            failure_kind="input_structure_error",
        )
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise MeasurementError(
            "invalid_dt",
            failure_kind="input_structure_error",
        )
    if not np.isfinite(maxlag_s) or maxlag_s <= 0:
        raise MeasurementError(
            "invalid_maxlag",
            failure_kind="input_structure_error",
        )
    if data.ndim != 1 or np.any(~np.isfinite(data)):
        raise MeasurementError(
            "unexpected_length",
            failure_kind="input_structure_error",
        )
    nlag_float = maxlag_s / dt_s
    nlag = int(round(nlag_float))
    expected = 2 * nlag + 1
    if (
        nlag <= 0
        or not math.isclose(
            nlag_float,
            float(nlag),
            rel_tol=0.0,
            abs_tol=(
                64.0
                * np.finfo(float).eps
                * max(1.0, nlag_float)
            ),
        )
        or data.size != expected
    ):
        raise MeasurementError(
            "unexpected_length",
            failure_kind="input_structure_error",
        )
    if np.all(data == 0.0):
        raise MeasurementError("all_zero")
    positive = np.asarray(data[nlag:], dtype=float)
    negative = np.asarray(data[nlag::-1], dtype=float)
    if positive.shape != negative.shape:
        raise MeasurementError(
            "unexpected_length",
            failure_kind="input_structure_error",
        )
    symmetric = 0.5 * (positive + negative)
    if np.all(symmetric == 0.0):
        raise MeasurementError("symmetric_zero")
    mismatch = float(
        np.linalg.norm(positive - negative)
        / max(
            float(np.linalg.norm(positive) + np.linalg.norm(negative)),
            np.finfo(float).tiny,
        )
    )
    return StackTrace(
        pair_name=pair_name,
        dt_s=dt_s,
        maxlag_s=maxlag_s,
        time_positive_s=np.arange(nlag + 1, dtype=float) * dt_s,
        positive_lag=positive,
        negative_lag_reversed=negative,
        symmetric=symmetric,
        branch_mismatch=mismatch,
    )


def read_stack_trace(
    path: Path,
    *,
    pair_name: str,
    component: str = "ZZ",
) -> StackTrace:
    """Read one exact HDF5 component and build its symmetric positive branch."""

    data, attrs = _read_unique_component_payload(
        path,
        component=component,
    )
    return _stack_trace_from_payload(
        data,
        attrs,
        pair_name=pair_name,
    )


def _json_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _distribution_key(value: float) -> str:
    return f"{float(value):g}"


def _attribute_distribution_key(value) -> str:
    if value is None:
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "invalid:" + repr(_json_scalar(value))
    if not np.isfinite(number):
        return "invalid:" + repr(number)
    return _distribution_key(number)


def _resolve_lineage_evidence(
    values: Sequence[object],
    *,
    unknown_value,
) -> Tuple[str, object]:
    normalized = tuple(
        json.dumps(_json_scalar(value), sort_keys=True)
        for value in values
        if value is not None
    )
    if not normalized:
        return "unknown", unknown_value
    unique = tuple(sorted(set(normalized)))
    if len(unique) > 1:
        return "contradictory", unknown_value
    return "confirmed", json.loads(unique[0])


def _append_lineage_evidence(
    evidence: Dict[str, List[Dict[str, object]]],
    *,
    key: str,
    value,
    source: str,
) -> None:
    scalar = _json_scalar(value)
    canonical = json.dumps(scalar, sort_keys=True)
    for row in evidence[key]:
        if row["canonical_value"] == canonical:
            row["count"] = int(row["count"]) + 1
            examples = row["example_sources"]
            if len(examples) < 5:
                examples.append(source)
            return
    evidence[key].append(
        {
            "canonical_value": canonical,
            "value": scalar,
            "count": 1,
            "example_sources": [source],
        }
    )


def _phase_difference_distribution(
    raw_trace: StackTrace,
    despiked_trace: StackTrace,
) -> np.ndarray:
    if (
        raw_trace.symmetric.shape != despiked_trace.symmetric.shape
        or raw_trace.dt_s != despiked_trace.dt_s
    ):
        raise MeasurementError(
            "raw_despiked_shape_mismatch",
            failure_kind="input_structure_error",
        )
    periods_s = np.linspace(2.5, 5.0, 51, dtype=float)
    frequencies_hz = 1.0 / periods_s
    kernel = np.exp(
        -2j
        * math.pi
        * frequencies_hz[:, None]
        * raw_trace.time_positive_s[None, :]
    )
    raw_spectrum = kernel @ raw_trace.symmetric
    despiked_spectrum = kernel @ despiked_trace.symmetric
    valid = (np.abs(raw_spectrum) > 0) & (np.abs(despiked_spectrum) > 0)
    return np.angle(
        despiked_spectrum[valid] * np.conj(raw_spectrum[valid])
    )


def compute_preliminary_snr(
    trace: StackTrace,
    *,
    distance_km: float,
) -> float:
    """Return the frozen candidate-independent 3.5 s SNR stratifier."""

    if not isinstance(trace, StackTrace):
        raise ValueError("trace must be a StackTrace")
    distance = float(distance_km)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance_km must be positive and finite")
    filtered = gaussian_filter_bank(
        trace.symmetric,
        dt_s=trace.dt_s,
        periods_s=np.asarray([3.5], dtype=float),
        alpha=12.0,
    ).filtered_waveforms[0]
    snr = compute_wang_snr(
        time_s=trace.time_positive_s,
        filtered_waveform=filtered,
        distance_km=distance,
        period_s=3.5,
    )
    if (
        snr.status != "accepted"
        or not np.isfinite(snr.leading_snr)
        or not np.isfinite(snr.trailing_snr)
    ):
        return float("nan")
    return float(min(snr.leading_snr, snr.trailing_snr))


def _load_preprocessing_lineage_config(
    path: Path,
) -> Tuple[Dict[str, object], str]:
    """Read only the three lineage scalars when PyYAML is unavailable."""

    source = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(source)
        if loaded is None:
            return {}, "pyyaml"
        if not isinstance(loaded, dict):
            raise ValueError("preprocessing config must contain a mapping")
        return dict(loaded), "pyyaml"
    allowed = {
        "response_removed",
        "physical_quantity",
        "lag_storage_direction",
    }
    config: Dict[str, object] = {}
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line[:1].isspace() or ":" not in raw_line:
            continue
        key, scalar = raw_line.split(":", 1)
        key = key.strip()
        if key not in allowed:
            continue
        value = scalar.split("#", 1)[0].strip()
        if not value:
            raise ValueError(
                f"lineage scalar is empty at line {line_number}"
            )
        lowered = value.lower()
        if lowered == "true":
            parsed: object = True
        elif lowered == "false":
            parsed = False
        elif (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ("'", '"')
        ):
            parsed = value[1:-1]
        elif any(token in value for token in ("[", "]", "{", "}", "&", "*")):
            raise ValueError(
                f"unsupported lineage YAML scalar at line {line_number}"
            )
        else:
            parsed = value
        config[key] = parsed
    return config, "restricted_scalar_parser"


def audit_input_inventory_and_lineage(
    stack_root: Path,
    *,
    component: str = "ZZ",
    preprocessing_config: Optional[Path] = None,
    raw_stack_root: Optional[Path] = None,
    phase_sample_limit: int = 100,
) -> Dict[str, object]:
    """Inventory formal stacks and report only evidence-backed lineage."""

    root = Path(stack_root)
    stack_paths = tuple(sorted(root.rglob("stack_pws.h5")))
    limit = int(phase_sample_limit)
    if limit < 0:
        raise ValueError("phase_sample_limit must be non-negative")
    config: Dict[str, object] = {}
    config_evidence = {
        "status": "not_provided",
        "path": None,
    }
    if preprocessing_config is not None:
        config, parser_status = _load_preprocessing_lineage_config(
            Path(preprocessing_config)
        )
        config_evidence = {
            "status": parser_status,
            "path": str(preprocessing_config),
        }
    evidence: Dict[str, List[Dict[str, object]]] = {
        "response_removed": [],
        "physical_quantity": [],
        "lag_storage_direction": [],
    }
    for key in evidence:
        if key in config:
            _append_lineage_evidence(
                evidence,
                key=key,
                value=config[key],
                source=(
                    "preprocessing_config:"
                    + str(preprocessing_config)
                    + ":"
                    + key
                ),
            )
    dt_counts: Dict[str, int] = {}
    sample_counts: Dict[str, int] = {}
    maxlag_counts: Dict[str, int] = {}
    failures: Dict[str, int] = {}
    valid_count = 0
    for path in stack_paths:
        try:
            data, attrs = _read_unique_component_payload(
                path,
                component=component,
            )
            for key in evidence:
                if key in attrs:
                    _append_lineage_evidence(
                        evidence,
                        key=key,
                        value=attrs[key],
                        source=(
                            "hdf5:"
                            + str(path.relative_to(root))
                            + ":attribute:"
                            + key
                        ),
                    )
            for target, key in (
                (dt_counts, _attribute_distribution_key(attrs.get("dt"))),
                (sample_counts, str(int(data.size))),
                (
                    maxlag_counts,
                    _attribute_distribution_key(attrs.get("maxlag")),
                ),
            ):
                target[key] = target.get(key, 0) + 1
            trace = _stack_trace_from_payload(
                data,
                attrs,
                pair_name=(
                    path.parent.parent.name
                    + "__"
                    + path.parent.name
                ),
            )
            valid_count += 1
        except MeasurementError as error:
            failures[error.status] = failures.get(error.status, 0) + 1
    response_status, response_removed = _resolve_lineage_evidence(
        [row["value"] for row in evidence["response_removed"]],
        unknown_value=None,
    )
    quantity_status, physical_quantity = _resolve_lineage_evidence(
        [row["value"] for row in evidence["physical_quantity"]],
        unknown_value="unknown",
    )
    lag_status, lag_direction = _resolve_lineage_evidence(
        [row["value"] for row in evidence["lag_storage_direction"]],
        unknown_value="unknown",
    )
    if response_status == "confirmed" and not isinstance(
        response_removed,
        bool,
    ):
        response_status, response_removed = "unknown", None
    if quantity_status == "confirmed":
        quantity_name = str(physical_quantity).strip().lower()
        if quantity_name in ("count", "counts"):
            physical_quantity = "count"
        elif quantity_name == "velocity":
            physical_quantity = "velocity"
        else:
            quantity_status, physical_quantity = "unknown", "unknown"
    if lag_status == "confirmed":
        direction_name = str(lag_direction).strip().lower()
        if direction_name in (
            "negative_to_positive",
            "positive_to_negative",
        ):
            lag_direction = direction_name
        else:
            lag_status, lag_direction = "unknown", "unknown"
    phase_differences: List[float] = []
    paired_count = 0
    if raw_stack_root is not None and limit > 0:
        raw_root = Path(raw_stack_root)
        for despiked_path in stack_paths:
            if paired_count >= limit:
                break
            relative = despiked_path.relative_to(root)
            raw_path = raw_root / relative
            if not raw_path.is_file():
                continue
            try:
                pair_name = (
                    relative.parent.parent.name
                    + "__"
                    + relative.parent.name
                )
                raw_trace = read_stack_trace(
                    raw_path,
                    pair_name=pair_name,
                    component=component,
                )
                despiked_trace = read_stack_trace(
                    despiked_path,
                    pair_name=pair_name,
                    component=component,
                )
                phase_differences.extend(
                    _phase_difference_distribution(
                        raw_trace,
                        despiked_trace,
                    ).tolist()
                )
                paired_count += 1
            except MeasurementError:
                continue
    absolute_phase = np.abs(np.asarray(phase_differences, dtype=float))
    statuses = (response_status, quantity_status, lag_status)
    lineage_status = (
        "contradictory"
        if "contradictory" in statuses
        else "confirmed"
        if all(status == "confirmed" for status in statuses)
        else "unknown"
    )
    return {
        "stack_root": str(root),
        "component": component,
        "stack_file_count": len(stack_paths),
        "valid_component_count": valid_count,
        "dt_distribution": dt_counts,
        "sample_count_distribution": sample_counts,
        "maxlag_distribution": maxlag_counts,
        "input_failure_counts": failures,
        "instrument_response": {
            "status": response_status,
            "removed": response_removed,
            "evidence": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "canonical_value"
                }
                for row in evidence["response_removed"]
            ],
        },
        "stack_quantity": {
            "status": quantity_status,
            "physical_quantity": physical_quantity,
            "evidence": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "canonical_value"
                }
                for row in evidence["physical_quantity"]
            ],
        },
        "lag_storage": {
            "status": lag_status,
            "direction": lag_direction,
            "evidence": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "canonical_value"
                }
                for row in evidence["lag_storage_direction"]
            ],
        },
        "lineage_status": lineage_status,
        "preprocessing_config_evidence": config_evidence,
        "phase_comparison": {
            "paired_file_count": paired_count,
            "period_band_s": [2.5, 5.0],
            "frequency_bin_count": len(phase_differences),
            "median_absolute_phase_difference_rad": (
                None
                if absolute_phase.size == 0
                else float(np.median(absolute_phase))
            ),
            "p95_absolute_phase_difference_rad": (
                None
                if absolute_phase.size == 0
                else float(np.percentile(absolute_phase, 95.0))
            ),
        },
    }


def preliminary_snr_inventory(
    rows: Sequence[Dict[str, object]],
    *,
    processed_pair_count: int,
) -> Dict[str, object]:
    """Canonicalize the candidate-independent SNR table and hash membership."""

    count = int(processed_pair_count)
    if count < 0:
        raise ValueError("processed_pair_count must be non-negative")
    canonical_rows = []
    for row in sorted(rows, key=lambda item: str(item["pair_name"])):
        snr = float(row["preliminary_snr"])
        canonical_rows.append(
            {
                "pair_name": str(row["pair_name"]),
                "distance_km": float(row["distance_km"]),
                "azimuth_deg": float(row["azimuth_deg"]),
                "preliminary_snr": snr if np.isfinite(snr) else None,
            }
        )
    payload = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    finite_count = sum(
        row["preliminary_snr"] is not None for row in canonical_rows
    )
    return {
        "definition": {
            "waveform": "symmetric CCF",
            "period_s": 3.5,
            "gaussian_alpha": 12.0,
            "snr": "min(Wang leading_snr, Wang trailing_snr)",
            "scientific_qc_use": False,
        },
        "row_count": len(canonical_rows),
        "finite_count": finite_count,
        "missing_value_count": max(0, count - finite_count),
        "rows_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": canonical_rows,
    }


def formal_input_inventory_failure_reason(
    inventory: Dict[str, object],
) -> Optional[str]:
    """Return the first immutable formal-input failure, or ``None``."""

    try:
        stack_count = int(inventory["stack_file_count"])
        valid_count = int(inventory["valid_component_count"])
        dt_distribution = dict(inventory["dt_distribution"])
        sample_distribution = dict(
            inventory["sample_count_distribution"]
        )
        maxlag_distribution = dict(inventory["maxlag_distribution"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("formal input inventory is incomplete") from exc
    if stack_count <= 0:
        return "zero_stack_files"
    if valid_count <= 0:
        return "zero_valid_components"
    if len(dt_distribution) != 1:
        return "mixed_dt"
    if len(sample_distribution) != 1:
        return "mixed_sample_length"
    if len(maxlag_distribution) != 1:
        return "mixed_maxlag"
    return None


def formal_science_failure_reason(
    *,
    input_count: int,
    unexpected_exception_count: int,
    left_count_by_period: Optional[Dict[float, int]] = None,
    right_count_by_period: Optional[Dict[float, int]] = None,
) -> Optional[str]:
    """Apply non-negotiable exception-rate and 3/4/5 s non-empty gates."""

    count = int(input_count)
    unexpected = int(unexpected_exception_count)
    if count < 0 or unexpected < 0 or unexpected > count:
        raise ValueError("formal science counts violate conservation")
    if count > 0 and unexpected / count > 0.01:
        return "unexpected_exception_fraction_exceeded"
    required_periods = (3.0, 4.0, 5.0)
    for label, period_counts in (
        ("left", left_count_by_period),
        ("right", right_count_by_period),
    ):
        if period_counts is None:
            continue
        normalized = {
            float(period): int(value)
            for period, value in period_counts.items()
        }
        for period in required_periods:
            if normalized.get(period, 0) <= 0:
                return (
                    f"empty_{label}_target_period_{period:g}s"
                )
    return None


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * 6371.0088 * math.asin(min(1.0, math.sqrt(a)))


def forward_azimuth_deg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    y = math.sin(delta_lon) * math.cos(phi2)
    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
    )
    return float(math.degrees(math.atan2(y, x)) % 360.0)


def load_station_coords(path: Path) -> Dict[str, Tuple[float, float]]:
    coords: Dict[str, Tuple[float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            coords[str(row["station_code"]).strip()] = (
                float(row["longitude"]),
                float(row["latitude"]),
            )
    return coords


def iter_stack_tasks(
    stack_root: Path,
    station_coords: Dict[str, Tuple[float, float]],
    *,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    bbox_mode: str = "both",
) -> Iterator[Tuple[str, str, str, float, float, float, float]]:
    for source_entry in sorted(stack_root.iterdir()):
        if not source_entry.is_dir():
            continue
        source_code = source_entry.name
        source_coord = station_coords.get(source_code)
        if source_coord is None:
            continue
        for receiver_entry in sorted(source_entry.iterdir()):
            if not receiver_entry.is_dir():
                continue
            receiver_code = receiver_entry.name
            receiver_coord = station_coords.get(receiver_code)
            if receiver_coord is None:
                continue
            if not pair_passes_bbox(
                float(source_coord[0]),
                float(source_coord[1]),
                float(receiver_coord[0]),
                float(receiver_coord[1]),
                bbox=bbox,
                mode=bbox_mode,
            ):
                continue
            h5_path = receiver_entry / "stack_pws.h5"
            if not h5_path.exists():
                continue
            yield (
                str(h5_path),
                source_code,
                receiver_code,
                float(source_coord[0]),
                float(source_coord[1]),
                float(receiver_coord[0]),
                float(receiver_coord[1]),
            )


def parse_bbox(value: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if value is None or not str(value).strip():
        return None
    parts = [float(part.strip()) for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be minlon,minlat,maxlon,maxlat")
    minlon, minlat, maxlon, maxlat = parts
    if minlon > maxlon or minlat > maxlat:
        raise ValueError("--bbox min values must not exceed max values")
    return minlon, minlat, maxlon, maxlat


def inside_bbox(lon: float, lat: float, bbox: Tuple[float, float, float, float]) -> bool:
    minlon, minlat, maxlon, maxlat = bbox
    return bool(minlon <= lon <= maxlon and minlat <= lat <= maxlat)


def pair_passes_bbox(
    source_lon: float,
    source_lat: float,
    receiver_lon: float,
    receiver_lat: float,
    *,
    bbox: Optional[Tuple[float, float, float, float]],
    mode: str,
) -> bool:
    if bbox is None:
        return True
    source_inside = inside_bbox(source_lon, source_lat, bbox)
    receiver_inside = inside_bbox(receiver_lon, receiver_lat, bbox)
    midpoint_inside = inside_bbox((source_lon + receiver_lon) / 2.0, (source_lat + receiver_lat) / 2.0, bbox)
    if mode == "both":
        return source_inside and receiver_inside
    if mode == "either":
        return source_inside or receiver_inside
    if mode == "midpoint":
        return midpoint_inside
    raise ValueError(f"Unsupported bbox mode: {mode}")


def rayleigh_signal_window(distance_km: float) -> Tuple[float, float]:
    return float(distance_km) / SIGNAL_VMAX_KM_S, float(distance_km) / SIGNAL_VMIN_KM_S


def wang_leading_trailing_snr(
    *,
    time_s: np.ndarray,
    filtered_waveform: np.ndarray,
    distance_km: float,
    period_s: float,
    min_noise_samples: int = MIN_NOISE_SAMPLES,
) -> WangSnrResult:
    """Historical runner entry delegated to the math-core Wang formula."""

    return compute_wang_snr(
        time_s=time_s,
        filtered_waveform=filtered_waveform,
        distance_km=distance_km,
        period_s=period_s,
        min_noise_samples=min_noise_samples,
    )


def group_pick_is_stable(
    *,
    predicted_time_s: float,
    snapped_time_s: float,
    period_s: float,
    max_fraction_period: float = GROUP_PICK_MAX_FRACTION_PERIOD,
) -> bool:
    if not (np.isfinite(predicted_time_s) and np.isfinite(snapped_time_s) and np.isfinite(period_s)):
        return False
    if period_s <= 0:
        return False
    return abs(float(snapped_time_s) - float(predicted_time_s)) <= float(max_fraction_period) * float(period_s)


def ftan_energy_cache_key(
    *,
    pair_waveform_hash: str,
    phase_convention: str,
    alpha: float,
) -> Tuple[str, str, float]:
    """Key one complete FTAN energy image, independent of DP beta values."""

    if not isinstance(pair_waveform_hash, str):
        raise ValueError("pair_waveform_hash must be a non-empty string")
    if not isinstance(phase_convention, str):
        raise ValueError("phase_convention must be a non-empty string")
    waveform_hash = pair_waveform_hash.strip()
    convention = phase_convention.strip()
    if not waveform_hash:
        raise ValueError("pair_waveform_hash must not be empty")
    if not convention:
        raise ValueError("phase_convention must not be empty")
    if isinstance(alpha, (bool, np.bool_)) or np.ndim(alpha) != 0:
        raise ValueError("alpha must be a positive finite scalar")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("alpha must be a positive finite scalar") from None
    if not np.isfinite(alpha_value) or alpha_value <= 0:
        raise ValueError("alpha must be positive and finite")
    return waveform_hash, convention, alpha_value


def ftan_period_grid() -> np.ndarray:
    return FtanConfig().periods_s


def generate_signal_window(start_win: int, end_win: int, pt_num: int, alpha: float = 0.1) -> Tuple[np.ndarray, int]:
    win_len = int((end_win - start_win) / (1.0 - alpha)) + 1
    window = windows.tukey(win_len, alpha)
    taper_len = round(win_len * alpha / 2.0)
    pad_left_len = start_win - taper_len
    if pad_left_len > 0:
        window = np.pad(window, (pad_left_len, 0), mode="constant")
    else:
        window = window[-pad_left_len:]
    if window.shape[0] < pt_num:
        window = np.pad(window, (0, pt_num - window.shape[0]), mode="constant")
    else:
        window = window[:pt_num]
    return np.asarray(window, dtype=float), int(taper_len)


def ftan_envelope_image_calculation(win_wave: np.ndarray, fs: float, periods_s: np.ndarray, distance_km: float) -> np.ndarray:
    sample_rate_hz = float(fs)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("fs must be positive and finite")
    return gaussian_filter_bank(
        np.asarray(win_wave, dtype=float),
        dt_s=1.0 / sample_rate_hz,
        periods_s=np.asarray(periods_s, dtype=float),
        alpha=gaussian_alpha_for_distance(float(distance_km)),
    ).envelope


def ftan_autosearch(initial_row: int, initial_col: int, image: np.ndarray) -> np.ndarray:
    row = int(initial_row)
    col = int(initial_col)
    y_size, x_size = image.shape
    arrival = np.zeros(x_size, dtype=float)
    step = 3
    for current_col in range(col, x_size):
        point_left = row
        point_right = row
        while True:
            point_left_new = max(0, point_left - step)
            if image[point_left, current_col] < image[point_left_new, current_col]:
                point_left = point_left_new
            else:
                point_left = point_left_new
                break
        while True:
            point_right_new = min(point_right + step, y_size - 1)
            if image[point_right, current_col] < image[point_right_new, current_col]:
                point_right = point_right_new
            else:
                point_right = point_right_new
                break
        index_max = int(np.argmax(image[point_left:point_right, current_col]))
        arrival[current_col] = index_max + point_left
        row = int(arrival[current_col])
    row = int(arrival[col])
    for current_col in range(col - 1, -1, -1):
        point_left = row
        point_right = row
        while True:
            point_left_new = max(0, point_left - step)
            if image[point_left, current_col] < image[point_left_new, current_col]:
                point_left = point_left_new
            else:
                point_left = point_left_new
                break
        while True:
            point_right_new = min(point_right + step, y_size - 1)
            if image[point_right, current_col] < image[point_right_new, current_col]:
                point_right = point_right_new
            else:
                point_right = point_right_new
                break
        index_max = int(np.argmax(image[point_left:point_right, current_col]))
        arrival[current_col] = index_max + point_left
        row = int(arrival[current_col])
    return arrival


def compute_ftan_group_velocity_image(
    waveform: np.ndarray,
    *,
    dt_s: float,
    distance_km: float,
    periods_s: np.ndarray,
    min_velocity_km_s: float,
    max_velocity_km_s: float,
    velocity_step_km_s: float = FTAN_VELOCITY_STEP_KM_S,
) -> Optional[Dict[str, np.ndarray]]:
    if waveform.size == 0 or not np.any(np.isfinite(waveform)):
        return None
    if dt_s <= 0 or distance_km <= 0 or min_velocity_km_s <= 0 or max_velocity_km_s <= min_velocity_km_s:
        return None

    sample_f = 1.0 / float(dt_s)
    pt_num = int(waveform.size)
    start_win = round(sample_f * distance_km / max_velocity_km_s)
    end_win = round(sample_f * distance_km / min_velocity_km_s)
    if end_win >= pt_num:
        end_win = pt_num - 1
    if start_win < 0 or end_win <= start_win or start_win >= pt_num:
        return None

    window, taper_len = generate_signal_window(start_win, end_win, pt_num, 0.1)
    win_wave = np.asarray(waveform, dtype=float) * np.asarray(window, dtype=float)

    wave_clip_pt = min(end_win + taper_len, pt_num)
    if wave_clip_pt <= 0:
        return None
    win_wave_clip = np.asarray(win_wave[:wave_clip_pt], dtype=float)

    envelope_signal = ftan_envelope_image_calculation(
        win_wave_clip,
        sample_f,
        np.asarray(periods_s, dtype=float),
        float(distance_km),
    )
    amp_signal = np.max(envelope_signal, axis=1)

    velocity_axis = np.arange(
        float(min_velocity_km_s),
        float(max_velocity_km_s) + 0.5 * float(velocity_step_km_s),
        float(velocity_step_km_s),
        dtype=float,
    )
    if velocity_axis.size < 2:
        return None
    travel_velocity = distance_km / (np.arange(start_win, end_win + 1, dtype=float) * float(dt_s))
    group_velocity_image = np.zeros((velocity_axis.size, len(periods_s)), dtype=float)
    for col in range(len(periods_s)):
        normalized = envelope_signal[col, start_win : end_win + 1]
        if normalized.size == 0:
            continue
        travel_velocity_segment = travel_velocity[: normalized.size]
        normalized = normalized / max(float(amp_signal[col]), 1e-12)
        group_velocity_image[:, col] = np.interp(
            velocity_axis,
            travel_velocity_segment[::-1],
            normalized[::-1],
            left=0.0,
            right=0.0,
        )

    return {
        "periods_s": np.asarray(periods_s, dtype=float),
        "velocity_axis_km_s": velocity_axis,
        "group_velocity_image": group_velocity_image,
    }


def pick_continuous_ftan_group_curve(
    *,
    group_velocity_image: np.ndarray,
    velocity_axis_km_s: np.ndarray,
    periods_s: np.ndarray,
    seed_count: int = FTAN_SEED_COUNT,
    seed_topk: int = FTAN_SEED_TOPK,
) -> np.ndarray:
    image = np.asarray(group_velocity_image, dtype=float)
    velocity_axis = np.asarray(velocity_axis_km_s, dtype=float)
    if image.ndim != 2 or image.shape[0] != velocity_axis.size or image.shape[1] != len(periods_s):
        raise ValueError("group_velocity_image shape must match velocity_axis_km_s and periods_s")
    if image.size == 0:
        raise ValueError("group_velocity_image is empty")
    if np.any(~np.isfinite(image)) or np.any(image < 0):
        raise ValueError("group_velocity_image must contain finite non-negative values")

    # ``seed_count`` and ``seed_topk`` remain accepted for source compatibility;
    # the exact full-grid dynamic program has no local seeds.
    _ = seed_count, seed_topk
    period_max = np.max(image, axis=0, keepdims=True)
    normalized_amplitude = np.zeros_like(image)
    np.divide(
        image,
        period_max,
        out=normalized_amplitude,
        where=period_max > 0,
    )
    # The legacy runner stores (velocity, period); the unified mathematical
    # kernel requires (period, velocity).
    amplitude_period_velocity = normalized_amplitude.T
    scaled_energy = normalized_log_energy(amplitude_period_velocity)
    candidates = find_candidate_ridges(
        scaled_log_energy=scaled_energy,
        normalized_envelope_amplitude=amplitude_period_velocity,
        periods_s=np.asarray(periods_s, dtype=float),
        velocity_axis_km_s=velocity_axis,
        beta1=FTAN_ROUGHNESS_WEIGHT,
        beta2=FTAN_CURVATURE_WEIGHT,
        max_candidates=3,
    )
    selected = select_fundamental_ridge(
        candidates,
        periods_s=np.asarray(periods_s, dtype=float),
    )
    if not selected.quality.accepted:
        raise RuntimeError(selected.quality.reason)
    group_curve = np.array(
        selected.group_velocities_km_s,
        dtype=float,
        copy=True,
    )
    group_curve[~selected.valid] = np.nan
    return group_curve


def build_ftan_group_measurements(
    waveform: np.ndarray,
    *,
    dt_s: float,
    distance_km: float,
) -> Optional[Dict[float, Dict[str, float]]]:
    periods_s = ftan_period_grid()
    payload = compute_ftan_group_velocity_image(
        waveform,
        dt_s=dt_s,
        distance_km=distance_km,
        periods_s=periods_s,
        min_velocity_km_s=SIGNAL_VMIN_KM_S,
        max_velocity_km_s=SIGNAL_VMAX_KM_S,
    )
    if payload is None:
        return None
    group_curve = pick_continuous_ftan_group_curve(
        group_velocity_image=payload["group_velocity_image"],
        velocity_axis_km_s=payload["velocity_axis_km_s"],
        periods_s=payload["periods_s"],
    )
    measurements: Dict[float, Dict[str, float]] = {}
    for period_s in TARGET_PERIODS_S:
        index = int(np.argmin(np.abs(payload["periods_s"] - float(period_s))))
        group_velocity = float(group_curve[index])
        if not np.isfinite(group_velocity) or group_velocity <= 0:
            continue
        measurements[float(period_s)] = {
            "group_velocity_km_s": group_velocity,
            "group_time_s": float(distance_km / group_velocity),
        }
    return measurements


def snap_group_time_to_local_envelope_peak(
    time_s: np.ndarray,
    envelope: np.ndarray,
    *,
    predicted_time_s: float,
    period_s: float,
) -> Tuple[int, float]:
    if time_s.size == 0 or envelope.size != time_s.size:
        raise ValueError("time_s and envelope must be the same non-zero length")
    half_window_s = max(0.6 * float(period_s), 0.5)
    mask = np.abs(np.asarray(time_s, dtype=float) - float(predicted_time_s)) <= half_window_s
    if not np.any(mask):
        index = int(np.argmin(np.abs(np.asarray(time_s, dtype=float) - float(predicted_time_s))))
        return index, float(time_s[index])
    local_indices = np.flatnonzero(mask)
    index = int(local_indices[int(np.argmax(np.asarray(envelope, dtype=float)[mask]))])
    return index, float(time_s[index])


def group_velocity_limit(period_s: float) -> float:
    return GROUP_VMAX_SHORT_KM_S if float(period_s) < 4.5 else GROUP_VMAX_LONG_KM_S


def raw_phase_travel_time(group_time_s: float, phi_tu_rad: float, period_s: float) -> float:
    omega = 2.0 * math.pi / float(period_s)
    return float(group_time_s + (float(phi_tu_rad) - math.pi / 4.0) / omega)


def passes_one_wavelength(distance_km: np.ndarray, reference_velocity_km_s: float, period_s: float) -> np.ndarray:
    return distance_km >= float(reference_velocity_km_s) * float(period_s)


def fit_velocity_through_origin(distance_km: np.ndarray, travel_time_s: np.ndarray) -> float:
    valid = np.isfinite(distance_km) & np.isfinite(travel_time_s) & (distance_km > 0) & (travel_time_s > 0)
    if np.count_nonzero(valid) == 0:
        return float("nan")
    slope = float(np.dot(distance_km[valid], travel_time_s[valid]) / np.dot(distance_km[valid], distance_km[valid]))
    if not np.isfinite(slope) or slope <= 0:
        return float("nan")
    return 1.0 / slope


def measurement_to_row(item: PhaseMeasurement) -> Dict[str, object]:
    row = asdict(item)
    row["group_velocity_limit_km_s"] = group_velocity_limit(item.period_s)
    return row


def wang_rejection_rows(
    pair_name: str,
    target_rows: Sequence[object],
) -> List[Dict[str, object]]:
    """Flatten continuous and target-period rejections for failures.csv."""

    rows: List[Dict[str, object]] = []
    if target_rows:
        audit = target_rows[0]
        for nominal, instantaneous, reason in zip(
            audit.rejected_continuous_nominal_periods_s,
            audit.rejected_continuous_instantaneous_periods_s,
            audit.continuous_rejection_statuses,
        ):
            rows.append(
                {
                    "pair_name": pair_name,
                    "stage": "continuous_observation",
                    "reason": reason,
                    "failure_kind": "expected_scientific_rejection",
                    "nominal_period_s": float(nominal),
                    "instantaneous_period_s": (
                        float(instantaneous)
                        if np.isfinite(instantaneous)
                        else None
                    ),
                    "target_period_s": None,
                }
            )
    for target in target_rows:
        if target.accepted:
            continue
        rows.append(
            {
                "pair_name": pair_name,
                "stage": "target_period",
                "reason": target.status,
                "failure_kind": "expected_scientific_rejection",
                "nominal_period_s": None,
                "instantaneous_period_s": None,
                "target_period_s": float(target.target_period_s),
            }
        )
    return rows


def build_reference_observations_from_task5_curve(
    *,
    pair_name: str,
    curve,
    target_rows: Sequence[object],
    time_s: np.ndarray,
    distance_km: float,
    azimuth_deg: float,
) -> List[Dict[str, object]]:
    """Return exactly the Task-5 LEFT continuous support used for targets."""

    rejected_nominal = (
        set()
        if not target_rows
        else {
            float(value)
            for value in target_rows[
            0
            ].rejected_continuous_nominal_periods_s
        }
    )
    instantaneous = np.asarray(
        curve.instantaneous_periods_s,
        dtype=float,
    )
    curve_valid = np.asarray(curve.measurement_valid, dtype=bool)
    duplicate_instantaneous = np.zeros(instantaneous.size, dtype=bool)
    finite_valid = curve_valid & np.isfinite(instantaneous)
    if np.any(finite_valid):
        _, inverse, counts = np.unique(
            instantaneous[finite_valid],
            return_inverse=True,
            return_counts=True,
        )
        duplicate_instantaneous[np.flatnonzero(finite_valid)] = (
            counts[inverse] > 1
        )
    observations: List[Dict[str, object]] = []
    for index, measurement in enumerate(curve.measurements):
        if not curve.measurement_valid[index] or measurement is None:
            continue
        if duplicate_instantaneous[index]:
            continue
        instantaneous_period = float(curve.instantaneous_periods_s[index])
        snr = compute_wang_snr(
            time_s=time_s,
            filtered_waveform=measurement.filtered_waveform,
            distance_km=distance_km,
            period_s=instantaneous_period,
        )
        left_qc = evaluate_wang_left_qc(
            period_s=instantaneous_period,
            group_velocity_km_s=distance_km / measurement.group_time_s,
            snr=snr,
            ftan_valid=True,
            ridge_valid=True,
            group_arrival_valid=True,
            phase_valid=True,
            instantaneous_frequency_valid=True,
        )
        if (
            not left_qc.accepted
            or float(curve.periods_s[index]) in rejected_nominal
        ):
            continue
        ridge = getattr(curve, "ridge", None)
        if ridge is None:
            ridge = getattr(curve, "selected_ridge", None)
        quality = getattr(ridge, "quality", None)
        ridge_rows = getattr(ridge, "row_indices", None)
        velocity_axis = getattr(curve, "velocity_axis_km_s", None)
        outermost = False
        if ridge_rows is not None and velocity_axis is not None:
            row_index = int(np.asarray(ridge_rows)[index])
            outermost = row_index in (0, len(velocity_axis) - 1)
        ridge_log_energy = np.asarray(
            getattr(
                curve,
                "ridge_normalized_log_energy",
                np.ones(len(curve.measurements), dtype=float),
            ),
            dtype=float,
        )
        ridge_envelope = np.asarray(
            getattr(
                curve,
                "ridge_normalized_envelope_amplitude",
                np.ones(len(curve.measurements), dtype=float),
            ),
            dtype=float,
        )
        ridge_jump = np.asarray(
            getattr(
                curve,
                "ridge_adjacent_jump_km_s",
                np.zeros(len(curve.measurements), dtype=float),
            ),
            dtype=float,
        )
        ridge_fields = {
            "nominal_period_s": float(curve.periods_s[index]),
            "normalized_log_energy": float(ridge_log_energy[index]),
            "normalized_envelope_amplitude": float(ridge_envelope[index]),
            "adjacent_jump_km_s": float(ridge_jump[index]),
            "coverage": float(getattr(quality, "coverage", 1.0)),
            "max_gap": int(getattr(quality, "max_gap", 0)),
            "jump_fraction": float(
                getattr(quality, "jump_fraction", 0.0)
            ),
            "boundary_fraction": float(
                getattr(quality, "boundary_fraction", 0.0)
            ),
            "normalized_energy_integral": float(
                getattr(quality, "normalized_energy_integral", 0.0)
            ),
            "outermost_velocity_cell": bool(outermost),
        }
        group_velocity = distance_km / measurement.group_time_s
        observations.append(
            {
                "pair_name": pair_name,
                "distance_km": distance_km,
                "azimuth_deg": azimuth_deg,
                "instantaneous_period_s": instantaneous_period,
                "anchored_raw_time_s": measurement.raw_phase_time_s,
                "T_inst": instantaneous_period,
                "t0": float(measurement.raw_phase_time_s),
                "U": float(group_velocity),
                "group_time_s": float(measurement.group_time_s),
                "signal_peak": float(snr.signal_peak),
                "leading_rms": float(snr.leading_noise_rms),
                "trailing_rms": float(snr.trailing_noise_rms),
                "leading_snr": float(snr.leading_snr),
                "trailing_snr": float(snr.trailing_snr),
                "ridge_fields": ridge_fields,
                "ridge_valid": True,
                "instantaneous_period_valid": True,
                "group_slowness_s_km": (
                    measurement.group_time_s / distance_km
                ),
                "convention": measurement.convention.name,
            }
        )
    return observations


def continuous_curve_audit_rows(
    *,
    pair_name: str,
    source_code: str,
    receiver_code: str,
    source_lon: float,
    source_lat: float,
    receiver_lon: float,
    receiver_lat: float,
    distance_km: float,
    azimuth_deg: float,
    curve,
    time_s: np.ndarray,
) -> List[Dict[str, object]]:
    """Return one auditable raw row for every readable FTAN period."""

    periods = np.asarray(curve.periods_s, dtype=float)
    instantaneous = np.asarray(curve.instantaneous_periods_s, dtype=float)
    valid = np.asarray(curve.measurement_valid, dtype=bool)
    statuses = tuple(str(value) for value in curve.measurement_statuses)
    ridge = curve.ridge
    ridge_rows = np.asarray(ridge.row_indices, dtype=int)
    ridge_velocities = np.asarray(ridge.group_velocities_km_s, dtype=float)
    ridge_valid = np.asarray(ridge.valid, dtype=bool)
    velocity_axis = np.asarray(curve.velocity_axis_km_s, dtype=float)
    if not (
        periods.shape
        == instantaneous.shape
        == valid.shape
        == ridge_rows.shape
        == ridge_velocities.shape
        == ridge_valid.shape
        == np.asarray(curve.ridge_normalized_log_energy).shape
        == np.asarray(curve.ridge_normalized_envelope_amplitude).shape
        == np.asarray(curve.ridge_adjacent_jump_km_s).shape
    ) or len(statuses) != periods.size:
        raise ValueError("continuous FTAN audit arrays are inconsistent")
    quality = ridge.quality
    rows: List[Dict[str, object]] = []
    for index, nominal_period in enumerate(periods):
        measurement = curve.measurements[index]
        instantaneous_period = (
            float(instantaneous[index])
            if np.isfinite(instantaneous[index])
            else None
        )
        group_time = None
        group_velocity = None
        raw_time = None
        signal_peak = None
        leading_rms = None
        trailing_rms = None
        leading_snr = None
        trailing_snr = None
        left_accepted = False
        left_status = statuses[index]
        if measurement is not None and valid[index] and instantaneous_period is not None:
            group_time = float(measurement.group_time_s)
            group_velocity = float(distance_km / group_time)
            raw_time = float(measurement.raw_phase_time_s)
            snr = compute_wang_snr(
                time_s=np.asarray(time_s, dtype=float),
                filtered_waveform=measurement.filtered_waveform,
                distance_km=float(distance_km),
                period_s=instantaneous_period,
            )
            signal_peak = float(snr.signal_peak)
            leading_rms = float(snr.leading_noise_rms)
            trailing_rms = float(snr.trailing_noise_rms)
            leading_snr = float(snr.leading_snr)
            trailing_snr = float(snr.trailing_snr)
            left = evaluate_wang_left_qc(
                period_s=instantaneous_period,
                group_velocity_km_s=group_velocity,
                snr=snr,
                ftan_valid=True,
                ridge_valid=bool(ridge_valid[index]),
                group_arrival_valid=True,
                phase_valid=True,
                instantaneous_frequency_valid=True,
            )
            left_accepted = bool(left.accepted)
            left_status = str(left.status)
        rows.append(
            {
                "pair_name": pair_name,
                "source_code": source_code,
                "receiver_code": receiver_code,
                "source_lon": float(source_lon),
                "source_lat": float(source_lat),
                "receiver_lon": float(receiver_lon),
                "receiver_lat": float(receiver_lat),
                "distance_km": float(distance_km),
                "azimuth_deg": float(azimuth_deg),
                "nominal_period_s": float(nominal_period),
                "instantaneous_period_s": instantaneous_period,
                "target_period_s": None,
                "group_time_s": group_time,
                "group_velocity_km_s": group_velocity,
                "signal_peak": signal_peak,
                "leading_noise_rms": leading_rms,
                "trailing_noise_rms": trailing_rms,
                "leading_snr": leading_snr,
                "trailing_snr": trailing_snr,
                "ridge_row_index": int(ridge_rows[index]),
                "ridge_group_velocity_km_s": float(
                    ridge_velocities[index]
                ),
                "ridge_valid": bool(ridge_valid[index]),
                "ridge_normalized_log_energy": float(
                    curve.ridge_normalized_log_energy[index]
                ),
                "ridge_normalized_envelope_amplitude": float(
                    curve.ridge_normalized_envelope_amplitude[index]
                ),
                "ridge_adjacent_jump_km_s": float(
                    curve.ridge_adjacent_jump_km_s[index]
                ),
                "outermost_velocity_cell": int(ridge_rows[index])
                in (0, velocity_axis.size - 1),
                "ridge_coverage": float(quality.coverage),
                "ridge_max_gap": int(quality.max_gap),
                "ridge_jump_fraction": float(quality.jump_fraction),
                "ridge_boundary_fraction": float(
                    quality.boundary_fraction
                ),
                "ridge_normalized_energy_integral": float(
                    quality.normalized_energy_integral
                ),
                "raw_travel_time_s": raw_time,
                "reference_time_s": None,
                "cycle_count": None,
                "corrected_travel_time_s": None,
                "phase_velocity_km_s": None,
                "phase_convention": curve.convention.name,
                "measurement_status": statuses[index],
                "left_qc_accepted": left_accepted,
                "left_qc_status": left_status,
                "rejection_reason": "" if left_accepted else left_status,
            }
        )
    return rows


def _pair_failure(
    pair_name: str,
    reason: str,
    *,
    failure_kind: str,
    stage: str = "pair",
    exception_type: Optional[str] = None,
) -> Dict[str, object]:
    result = {
        "pair_name": pair_name,
        "ok": False,
        "reason": reason,
        "failure_kind": failure_kind,
        "failure_stage": stage,
    }
    if exception_type is not None:
        result["exception_type"] = exception_type
    return result


def process_one_pair(task: Sequence[object]) -> Dict[str, object]:
    if len(task) not in (7, 8, 9):
        raise ValueError(
            "pair task must contain seven fields plus component/config"
        )
    (
        stack_path_str,
        source_code,
        receiver_code,
        source_lon,
        source_lat,
        receiver_lon,
        receiver_lat,
    ) = task[:7]
    component = "ZZ" if len(task) == 7 else str(task[7])
    scientific_parameters = (
        {} if len(task) < 9 else dict(task[8])
    )
    allowed_parameter_names = {
        "phase_convention",
        "alpha",
        "beta1",
        "beta2",
    }
    if scientific_parameters and (
        set(scientific_parameters) != allowed_parameter_names
    ):
        raise ValueError(
            "explicit pair science parameters must contain exactly "
            "phase_convention, alpha, beta1 and beta2"
        )
    stack_path = Path(stack_path_str)
    pair_name = f"{source_code}__{receiver_code}"
    stage = "strict_hdf5"
    completed_stages: List[str] = []
    input_diagnostics: Optional[Dict[str, object]] = None
    preliminary_snr_row: Optional[Dict[str, object]] = None
    try:
        stack_trace = read_stack_trace(
            stack_path,
            pair_name=pair_name,
            component=component,
        )
        completed_stages.append(stage)
        stage = "symmetric_component"
        input_diagnostics = {
            "component": component,
            "dt_s": stack_trace.dt_s,
            "maxlag_s": stack_trace.maxlag_s,
            "sample_count": int(
                2 * stack_trace.symmetric.size - 1
            ),
            "branch_mismatch": stack_trace.branch_mismatch,
        }
        completed_stages.append(stage)
        stage = "preliminary_snr_inventory"
        time_s = stack_trace.time_positive_s
        distance_km = great_circle_km(
            float(source_lat),
            float(source_lon),
            float(receiver_lat),
            float(receiver_lon),
        )
        azimuth_deg = forward_azimuth_deg(
            float(source_lat),
            float(source_lon),
            float(receiver_lat),
            float(receiver_lon),
        )
        preliminary_snr = compute_preliminary_snr(
            stack_trace,
            distance_km=distance_km,
        )
        preliminary_snr_row = {
            "pair_name": pair_name,
            "distance_km": distance_km,
            "azimuth_deg": azimuth_deg,
            "preliminary_snr": preliminary_snr,
        }
        stage = "filter_bank"
        if scientific_parameters:
            convention_name = str(
                scientific_parameters["phase_convention"]
            )
            try:
                pair_convention = PhaseConvention[convention_name]
            except KeyError as exc:
                raise ValueError(
                    "pair phase convention is not recognized"
                ) from exc
            alpha = float(scientific_parameters["alpha"])
            beta1 = float(scientific_parameters["beta1"])
            beta2 = float(scientific_parameters["beta2"])
            config = FtanConfig()
            if (
                alpha not in config.alpha_candidates
                or beta1 not in config.beta1_candidates
                or beta2 not in config.beta2_candidates
            ):
                raise ValueError(
                    "pair science parameters lie outside the formal grid"
                )
        else:
            pair_convention = PhaseConvention.BENSEN_VELOCITY_CCF
            alpha = gaussian_alpha_for_distance(distance_km)
            beta1 = 0.5
            beta2 = 1.0
        trace = DatTrace(
            pair_name=pair_name,
            distance_km=distance_km,
            dt_s=stack_trace.dt_s,
            time_s=time_s,
            positive_lag=stack_trace.positive_lag,
            negative_lag_reversed=stack_trace.negative_lag_reversed,
            symmetric_waveform=stack_trace.symmetric,
            lon_a=float(source_lon),
            lat_a=float(source_lat),
            lon_b=float(receiver_lon),
            lat_b=float(receiver_lat),
        )

        def record_ftan_stage(completed_stage: str) -> None:
            nonlocal stage
            expected_next = {
                "filter_bank": "dp_ridge",
                "dp_ridge": (
                    "group_arrival_phase_instantaneous_frequency"
                ),
                "group_arrival_phase_instantaneous_frequency": (
                    "phase_unwrap"
                ),
                "phase_unwrap": "continuous_left_qc",
            }
            if completed_stage not in expected_next:
                raise RuntimeError(
                    "unexpected FTAN stage callback: " + completed_stage
                )
            completed_stages.append(completed_stage)
            stage = expected_next[completed_stage]

        curve = measure_phase_curve(
            trace,
            periods_s=FtanConfig().periods_s,
            velocity_axis_km_s=FtanConfig().group_velocities_km_s,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            convention=pair_convention,
            stage_callback=record_ftan_stage,
        )
        if curve is None:
            if completed_stages:
                stage = (
                    "phase_unwrap"
                    if completed_stages[-1] == "phase_unwrap"
                    else completed_stages[-1]
                )
            raise MeasurementError(
                (
                    "no_fundamental_ridge"
                    if stage == "dp_ridge"
                    else "phase_curve_invalid"
                ),
                detail="measure_phase_curve returned no curve",
            )
        stage = "continuous_left_qc"
        raw_measurements = continuous_curve_audit_rows(
            pair_name=pair_name,
            source_code=str(source_code),
            receiver_code=str(receiver_code),
            source_lon=float(source_lon),
            source_lat=float(source_lat),
            receiver_lon=float(receiver_lon),
            receiver_lat=float(receiver_lat),
            distance_km=distance_km,
            azimuth_deg=azimuth_deg,
            curve=curve,
            time_s=time_s,
        )
        continuous_observations = (
            build_reference_observations_from_task5_curve(
                pair_name=pair_name,
                curve=curve,
                target_rows=(),
                time_s=time_s,
                distance_km=distance_km,
                azimuth_deg=azimuth_deg,
            )
        )
        completed_stages.append(stage)
        stage = "target_period_resampling"
        target_rows = resample_wang_measurements(
            curve.measurements,
            time_s=time_s,
            distance_km=distance_km,
            target_periods_s=TARGET_PERIODS_S,
            nominal_periods_s=curve.periods_s,
            measurement_statuses=curve.measurement_statuses,
            instantaneous_periods_s=curve.instantaneous_periods_s,
            ridge_normalized_log_energy=(
                curve.ridge_normalized_log_energy
            ),
            ridge_normalized_envelope_amplitude=(
                curve.ridge_normalized_envelope_amplitude
            ),
            ridge_adjacent_jump_km_s=curve.ridge_adjacent_jump_km_s,
            valid_mask=curve.measurement_valid,
        )
        completed_stages.append(stage)
        measurements: List[Dict[str, object]] = []
        target_statuses: Dict[str, str] = {}
        for target in target_rows:
            target_statuses[f"{target.target_period_s:g}"] = target.status
            if not target.accepted:
                continue
            item = PhaseMeasurement(
                pair_name=pair_name,
                distance_km=distance_km,
                period_s=target.target_period_s,
                group_time_s=target.group_time_s,
                group_velocity_km_s=target.group_velocity_km_s,
                leading_snr=target.leading_snr,
                trailing_snr=target.trailing_snr,
                phi_tu_rad=float("nan"),
                raw_travel_time_s=target.anchored_raw_phase_time_s,
            )
            row = measurement_to_row(item)
            row.update(
                {
                    "source_code": str(source_code),
                    "receiver_code": str(receiver_code),
                    "source_lon": float(source_lon),
                    "source_lat": float(source_lat),
                    "receiver_lon": float(receiver_lon),
                    "receiver_lat": float(receiver_lat),
                    "azimuth_deg": float(azimuth_deg),
                    "nominal_period_s": target.target_period_s,
                    "instantaneous_period_s": target.target_period_s,
                    "target_period_s": target.target_period_s,
                    "signal_peak": target.signal_peak,
                    "leading_noise_rms": target.leading_noise_rms,
                    "trailing_noise_rms": target.trailing_noise_rms,
                    "ridge_normalized_log_energy": (
                        target.ridge_normalized_log_energy
                    ),
                    "ridge_normalized_envelope_amplitude": (
                        target.ridge_normalized_envelope_amplitude
                    ),
                    "ridge_adjacent_jump_km_s": (
                        target.ridge_adjacent_jump_km_s
                    ),
                    "support_count": target.support_count,
                    "interpolation_method": target.interpolation_method,
                    "left_qc_status": target.status,
                }
            )
            measurements.append(row)
        return {
            "pair_name": pair_name,
            "ok": True,
            "passes_any": bool(measurements),
            "raw_measurements": raw_measurements,
            "measurements": measurements,
            "continuous_observations": continuous_observations,
            "target_statuses": target_statuses,
            "rejections": wang_rejection_rows(pair_name, target_rows),
            "input_diagnostics": input_diagnostics,
            "preliminary_snr_row": preliminary_snr_row,
            "completed_stages": tuple(completed_stages),
        }
    except MeasurementError as exc:
        failure = _pair_failure(
            pair_name,
            exc.status,
            failure_kind=exc.failure_kind,
            stage=stage,
        )
        if input_diagnostics is not None:
            failure["input_diagnostics"] = input_diagnostics
        if preliminary_snr_row is not None:
            failure["preliminary_snr_row"] = preliminary_snr_row
        failure["completed_stages"] = tuple(completed_stages)
        return failure
    except Exception as exc:
        failure = _pair_failure(
            pair_name,
            str(exc),
            failure_kind="unexpected_pair_exception",
            stage=stage,
            exception_type=type(exc).__name__,
        )
        if input_diagnostics is not None:
            failure["input_diagnostics"] = input_diagnostics
        if preliminary_snr_row is not None:
            failure["preliminary_snr_row"] = preliminary_snr_row
        failure["completed_stages"] = tuple(completed_stages)
        return failure


def build_period_payload(
    rows: List[PhaseMeasurement],
    reference_fit,
    *,
    convention: PhaseConvention = PhaseConvention.BENSEN_VELOCITY_CCF,
) -> Dict[str, object]:
    if reference_fit.status != "accepted" or not rows:
        return {
            "reference_velocity_km_s": float("nan"),
            "fit_velocity_km_s": float("nan"),
            "ordinary_ls_velocity_km_s": float("nan"),
            "std_velocity_km_s": float("nan"),
            "bootstrap_velocity_std_km_s": float("nan"),
            "bootstrap_velocity_ci95_low_km_s": float("nan"),
            "bootstrap_velocity_ci95_high_km_s": float("nan"),
            "bootstrap_samples": 1000,
            "bootstrap_seed": 20260717,
            "left_rows": [],
            "corrected_rows": [],
            "right_rows": [],
        }
    distance = np.asarray([row.distance_km for row in rows], dtype=float)
    raw_time = np.asarray([row.raw_travel_time_s for row in rows], dtype=float)
    target_periods = np.asarray([row.period_s for row in rows], dtype=float)
    cycle_rows = resolve_reference_cycles(
        raw_times_s=raw_time,
        distance_km=distance,
        observation_periods_s=target_periods,
        reference_periods_s=reference_fit.periods_s,
        reference_slowness_s_km=reference_fit.phase_slowness_s_km,
        convention=convention,
    )
    corrected_time = np.asarray(
        [cycle.corrected_time_s for cycle in cycle_rows],
        dtype=float,
    )
    branch_n = np.asarray(
        [cycle.cycle_count for cycle in cycle_rows],
        dtype=int,
    )
    residual = np.asarray(
        [cycle.corrected_residual_s for cycle in cycle_rows],
        dtype=float,
    )
    predicted = np.asarray(
        [cycle.reference_time_s for cycle in cycle_rows],
        dtype=float,
    )
    reference_velocity = float(
        np.interp(
            rows[0].period_s,
            reference_fit.periods_s,
            reference_fit.phase_velocities_km_s,
        )
    )
    far_field = passes_one_wavelength(distance, reference_velocity, rows[0].period_s)
    half_period_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, float(rows[0].period_s))
    )
    half_period_valid = (
        np.abs(residual)
        <= float(rows[0].period_s) / 2.0 + half_period_tolerance
    )
    valid = far_field & half_period_valid
    phase_velocity = distance / corrected_time
    right_rows: List[Dict[str, object]] = []
    corrected_rows: List[Dict[str, object]] = []
    left_rows: List[Dict[str, object]] = []
    for index, row in enumerate(rows):
        left_rows.append(
            {
                "pair_name": row.pair_name,
                "distance_km": row.distance_km,
                "raw_travel_time_s": row.raw_travel_time_s,
            }
        )
        if not half_period_valid[index]:
            right_qc_status = "fails_half_period_residual"
        elif not far_field[index]:
            right_qc_status = "fails_one_wavelength"
        else:
            right_qc_status = "accepted"
        corrected = {
            "pair_name": row.pair_name,
            "distance_km": row.distance_km,
            "period_s": row.period_s,
            "group_time_s": row.group_time_s,
            "group_velocity_km_s": row.group_velocity_km_s,
            "leading_snr": row.leading_snr,
            "trailing_snr": row.trailing_snr,
            "phi_tu_rad": row.phi_tu_rad,
            "raw_travel_time_s": row.raw_travel_time_s,
            "reference_time_s": float(predicted[index]),
            "reference_slowness_s_km": float(
                predicted[index] / distance[index]
            ),
            "N": int(branch_n[index]),
            "cycle_count": int(branch_n[index]),
            "corrected_time_s": float(corrected_time[index]),
            "corrected_residual_s": float(residual[index]),
            "branch_tie": bool(cycle_rows[index].branch_tie),
            "cycle_period_s": float(cycle_rows[index].period_s),
            "passes_one_wavelength": bool(far_field[index]),
            "passes_half_period_residual": bool(
                half_period_valid[index]
            ),
            "right_column": bool(valid[index]),
            "right_qc_status": right_qc_status,
            "rejection_reason": (
                "" if right_qc_status == "accepted" else right_qc_status
            ),
            "phase_velocity_km_s": float(phase_velocity[index]),
            "predicted_travel_time_s": float(predicted[index]),
            "corrected_travel_time_s": float(corrected_time[index]),
            "branch_n": int(branch_n[index]),
            "residual_s": float(residual[index]),
        }
        corrected_rows.append(corrected)
        if valid[index]:
            right_rows.append(dict(corrected))
    fit_velocity = float("nan")
    ordinary_ls_velocity = float("nan")
    std_velocity = float("nan")
    bootstrap_std_velocity = float("nan")
    bootstrap_ci_low = float("nan")
    bootstrap_ci_high = float("nan")
    bootstrap_samples = 1000
    bootstrap_seed = 20260717
    if right_rows:
        fit_result = fit_right_column_slowness(
            np.asarray(
                [row["distance_km"] for row in right_rows],
                dtype=float,
            ),
            np.asarray(
                [
                    row["corrected_travel_time_s"]
                    for row in right_rows
                ],
                dtype=float,
            ),
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        fit_velocity = fit_result.huber_velocity_km_s
        ordinary_ls_velocity = fit_result.ordinary_ls_velocity_km_s
        std_velocity = fit_result.path_velocity_std_km_s
        bootstrap_std_velocity = (
            fit_result.bootstrap_velocity_std_km_s
        )
        bootstrap_ci_low, bootstrap_ci_high = (
            fit_result.bootstrap_velocity_ci95_km_s
        )
    return {
        "reference_velocity_km_s": float(reference_velocity),
        "fit_velocity_km_s": float(fit_velocity),
        "ordinary_ls_velocity_km_s": float(ordinary_ls_velocity),
        "std_velocity_km_s": float(std_velocity),
        "bootstrap_velocity_std_km_s": float(
            bootstrap_std_velocity
        ),
        "bootstrap_velocity_ci95_low_km_s": float(bootstrap_ci_low),
        "bootstrap_velocity_ci95_high_km_s": float(bootstrap_ci_high),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "left_rows": left_rows,
        "corrected_rows": corrected_rows,
        "right_rows": right_rows,
    }


def plot_figure(
    output_path: Path,
    per_period: Dict[float, Dict[str, object]],
    *,
    paper_scale: bool = True,
) -> Dict[str, object]:
    ensure_dir(output_path.parent)
    row_count = len(FIGURE4_PERIODS_S)
    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(10.2, 4.7 * row_count),
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor("white")
    if paper_scale:
        x_limit = 25.0
        y_limit = 10.0
    else:
        all_distance = []
        all_time = []
        for period_s in FIGURE4_PERIODS_S:
            payload = per_period[period_s]
            all_distance.extend(
                float(row["distance_km"])
                for row in payload["left_rows"]
            )
            all_distance.extend(
                float(row["distance_km"])
                for row in payload["right_rows"]
            )
            all_time.extend(
                float(row["raw_travel_time_s"])
                for row in payload["left_rows"]
            )
            all_time.extend(
                float(row["corrected_travel_time_s"])
                for row in payload["right_rows"]
            )
        x_limit = max(25.0, 1.05 * max(all_distance, default=25.0))
        y_limit = max(10.0, 1.05 * max(all_time, default=10.0))
    xline = np.linspace(0.0, x_limit, 240)
    left_counts: Dict[str, int] = {}
    right_counts: Dict[str, int] = {}
    for row_index, period_s in enumerate(FIGURE4_PERIODS_S):
        payload = per_period[period_s]
        left_rows = payload["left_rows"]
        right_rows = payload["right_rows"]
        period_key = f"{period_s:g}"
        left_counts[period_key] = len(left_rows)
        right_counts[period_key] = len(right_rows)
        ref_v = payload["reference_velocity_km_s"]
        fit_v = payload["fit_velocity_km_s"]
        std_v = payload["std_velocity_km_s"]
        ax_left = axes[row_index, 0]
        ax_right = axes[row_index, 1]
        if left_rows:
            ax_left.scatter(
                [row["distance_km"] for row in left_rows],
                [row["raw_travel_time_s"] for row in left_rows],
                s=5,
                color="#0b51ff",
                alpha=0.9,
                linewidths=0,
            )
        if right_rows:
            ax_right.scatter(
                [row["distance_km"] for row in right_rows],
                [row["corrected_travel_time_s"] for row in right_rows],
                s=5,
                color="#0b51ff",
                alpha=0.9,
                linewidths=0,
            )
        if np.isfinite(ref_v) and ref_v > 0:
            center = xline / ref_v
            ax_left.plot(xline, center - period_s / 2.0, "--", color="#66ff66", lw=1.0)
            ax_left.plot(xline, center + period_s / 2.0, "--", color="#66ff66", lw=1.0)
            ax_right.plot(xline, center - period_s / 2.0, "--", color="#66ff66", lw=1.0)
            ax_right.plot(xline, center + period_s / 2.0, "--", color="#66ff66", lw=1.0)
        if np.isfinite(fit_v) and fit_v > 0:
            ax_right.plot(xline, xline / fit_v, color="black", lw=0.9)
        label = chr(ord("a") + row_index)
        ax_left.text(
            0.024 * x_limit,
            0.925 * y_limit,
            f"({label}) {period_s:g} s",
            fontsize=14,
        )
        if np.isfinite(fit_v):
            ax_right.text(
                0.048 * x_limit,
                0.83 * y_limit,
                f"V = {fit_v:.2f} km/s\nSTDV = {std_v:.2f} km/s",
                fontsize=11,
            )
        for axis in (ax_left, ax_right):
            axis.set_xlim(0.0, x_limit)
            axis.set_ylim(0.0, y_limit)
            if paper_scale:
                axis.set_xticks(np.arange(0, 26, 5.0))
                axis.set_yticks(np.arange(0, 11, 1))
            axis.minorticks_on()
            axis.grid(True, which="major", color="#d7d7d7", linestyle="-", linewidth=0.7)
            axis.grid(True, which="minor", color="#e9e9e9", linestyle=":", linewidth=0.7)
            axis.tick_params(labelsize=11)
        ax_left.set_ylabel("Travel Time (s)", fontsize=16)
    axes[-1, 0].set_xlabel("Distance (km)", fontsize=18)
    axes[-1, 1].set_xlabel("Distance (km)", fontsize=18)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "periods_s": list(FIGURE4_PERIODS_S),
        "subplot_shape": [row_count, 2],
        "paper_scale": bool(paper_scale),
        "axis_limits": {"distance_km": [0.0, x_limit], "travel_time_s": [0.0, y_limit]},
        "shared_axis_limits": True,
        "reference_half_period_lines_both_columns": True,
        "right_fit_line_and_statistics": True,
        "left_scatter_count_by_period": left_counts,
        "right_scatter_count_by_period": right_counts,
    }


def _read_nonempty_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    return fieldnames, rows


def validate_formal_outputs(
    output_dir: Path,
    *,
    expected_left_count_by_period: Dict[float, int],
    expected_right_count_by_period: Dict[float, int],
) -> Dict[str, object]:
    """Validate the complete formal audit bundle before reporting success."""

    root = Path(output_dir)
    missing = [
        relative
        for relative in FORMAL_REQUIRED_OUTPUTS
        if not (root / relative).is_file()
    ]
    if missing:
        return {
            "accepted": False,
            "status": "formal_output_missing",
            "missing": missing,
        }
    empty = [
        relative
        for relative in FORMAL_REQUIRED_OUTPUTS
        if (root / relative).stat().st_size <= 0
    ]
    if empty:
        return {
            "accepted": False,
            "status": "formal_output_empty",
            "empty": empty,
        }
    try:
        metadata = json.loads(
            (root / "metadata.json").read_text(encoding="utf-8")
        )
        frozen = json.loads(
            (root / "frozen_parameters.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "accepted": False,
            "status": "formal_output_json_invalid",
            "detail": str(exc),
        }
    required_metadata = {
        "run_status",
        "stage",
        "exit_status",
        "git_commit_sha",
        "python_version",
        "dependency_versions",
        "stack_root",
        "stations_csv",
        "input_file_count",
        "input_inventory_sha256",
        "code_sha256",
        "config_sha256",
        "phase_convention",
        "frozen_candidate",
        "started_at",
        "finished_at",
        "host",
    }
    if (
        not isinstance(metadata, dict)
        or not required_metadata.issubset(metadata)
        or metadata.get("run_status") != "success"
        or metadata.get("stage") != "C"
        or metadata.get("exit_status") != 0
    ):
        return {
            "accepted": False,
            "status": "formal_output_metadata_incomplete",
        }
    required_frozen = {
        "stage_b_status",
        "candidate_id",
        "phase_convention",
        "alpha",
        "beta1",
        "beta2",
    }
    if (
        not isinstance(frozen, dict)
        or not required_frozen.issubset(frozen)
        or frozen.get("stage_b_status") != "passed"
    ):
        return {
            "accepted": False,
            "status": "formal_output_frozen_parameters_invalid",
        }
    csv_rows: Dict[str, List[Dict[str, str]]] = {}
    for relative in FORMAL_REQUIRED_OUTPUTS:
        if not relative.endswith(".csv"):
            continue
        try:
            fieldnames, rows = _read_nonempty_csv_rows(root / relative)
        except (OSError, UnicodeError, csv.Error) as exc:
            return {
                "accepted": False,
                "status": "formal_output_csv_invalid",
                "path": relative,
                "detail": str(exc),
            }
        if not fieldnames or not rows:
            return {
                "accepted": False,
                "status": "formal_output_csv_empty",
                "path": relative,
            }
        csv_rows[relative] = rows
    if len(csv_rows["candidate_grid_results.csv"]) != 300:
        return {
            "accepted": False,
            "status": "formal_output_candidate_grid_incomplete",
            "candidate_count": len(csv_rows["candidate_grid_results.csv"]),
        }
    try:
        split_indices = {
            int(row["split_index"])
            for row in csv_rows["split_half_stability.csv"]
        }
        membership_indices = {
            int(row["split_index"])
            for row in csv_rows["split_half_membership.csv"]
        }
    except (KeyError, TypeError, ValueError):
        return {
            "accepted": False,
            "status": "formal_output_split_half_invalid",
        }
    if split_indices != set(range(20)) or membership_indices != set(range(20)):
        return {
            "accepted": False,
            "status": "formal_output_split_half_incomplete",
        }
    if len(csv_rows["split_half_stability.csv"]) != 80:
        return {
            "accepted": False,
            "status": "formal_output_split_half_incomplete",
        }
    try:
        cycle_count = sum(
            int(row["measurement_count"])
            for row in csv_rows["cycle_count_distribution.csv"]
        )
        invalid_ties = any(
            int(row["branch_tie_count"]) < 0
            or int(row["branch_tie_count"]) > int(row["measurement_count"])
            for row in csv_rows["cycle_count_distribution.csv"]
        )
    except (KeyError, TypeError, ValueError):
        cycle_count = 0
        invalid_ties = True
    if cycle_count <= 0 or invalid_ties:
        return {
            "accepted": False,
            "status": "formal_output_cycle_distribution_invalid",
        }
    spatial_rows = csv_rows["reference_spatial_diagnostics.csv"]
    try:
        spatial_signature = {
            (str(row["diagnostic_dimension"]), int(row["bin_index"]))
            for row in spatial_rows
        }
    except (KeyError, TypeError, ValueError):
        spatial_signature = set()
    expected_spatial_signature = {
        *(('distance_quintile', index) for index in range(5)),
        *(('azimuth_45deg', index) for index in range(8)),
    }
    if spatial_signature != expected_spatial_signature:
        return {
            "accepted": False,
            "status": "formal_output_spatial_diagnostics_incomplete",
        }
    phase_rows = csv_rows["phase_matching_comparison.csv"]
    try:
        phase_signature = {
            str(row["phase_convention"]) for row in phase_rows
        }
    except (KeyError, TypeError):
        phase_signature = set()
    if phase_signature != {
        PhaseConvention.BENSEN_VELOCITY_CCF.name,
        PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF.name,
    }:
        return {
            "accepted": False,
            "status": "formal_output_phase_matching_incomplete",
        }
    example_paths = sorted(
        (root / "figures" / "ftan_examples").glob("example_*.png")
    )
    if len(example_paths) < 3 or any(path.stat().st_size <= 0 for path in example_paths):
        return {
            "accepted": False,
            "status": "formal_output_ftan_examples_incomplete",
            "example_count": len(example_paths),
        }

    def count_periods(relative: str) -> Dict[float, int]:
        counts: Dict[float, int] = {}
        for row in csv_rows[relative]:
            period = float(row["period_s"])
            counts[period] = counts.get(period, 0) + 1
        return counts

    try:
        actual_left = count_periods("measurements_left_qc.csv")
        actual_right = count_periods("measurements_right_qc.csv")
    except (KeyError, TypeError, ValueError):
        return {
            "accepted": False,
            "status": "formal_output_measurement_period_invalid",
        }
    expected_left = {
        float(period): int(count)
        for period, count in expected_left_count_by_period.items()
    }
    expected_right = {
        float(period): int(count)
        for period, count in expected_right_count_by_period.items()
    }
    if actual_left != expected_left or actual_right != expected_right:
        return {
            "accepted": False,
            "status": "formal_output_count_mismatch",
            "actual_left": actual_left,
            "expected_left": expected_left,
            "actual_right": actual_right,
            "expected_right": expected_right,
        }
    return {
        "accepted": True,
        "status": "accepted",
        "candidate_count": 300,
        "split_count": 20,
        "ftan_example_count": len(example_paths),
        "left_count_by_period": actual_left,
        "right_count_by_period": actual_right,
    }


def stage_b_audit_rows(
    evidence_payload: Dict[str, object],
    frozen_manifest: Dict[str, object],
) -> Dict[str, List[Dict[str, object]]]:
    """Flatten the selected Stage B evidence into lossless CSV audit rows."""

    candidates = [
        dict(row) for row in evidence_payload.get("candidate_results", ())
    ]
    selected_id = str(frozen_manifest.get("candidate_id", ""))
    selected_rows = [
        row for row in candidates if str(row.get("candidate_id", "")) == selected_id
    ]
    if len(selected_rows) != 1:
        raise ValueError("selected Stage B candidate is absent or duplicated")
    candidate_rows = []
    for row in candidates:
        flattened = dict(row)
        flattened["selected"] = str(row.get("candidate_id", "")) == selected_id
        for key, value in tuple(flattened.items()):
            if isinstance(value, (dict, list, tuple)):
                flattened[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                )
        candidate_rows.append(flattened)
    selected_hash = str(selected_rows[0].get("left_observation_sha256", ""))
    class_evidence = evidence_payload.get("class_evidence", {})
    if not isinstance(class_evidence, dict) or selected_hash not in class_evidence:
        raise ValueError("selected Stage B measurement-class evidence is absent")
    selected_class = dict(class_evidence[selected_hash])

    reference = selected_class.get("reference")
    corrected = (
        []
        if not isinstance(reference, dict)
        else [dict(row) for row in reference.get("corrected_rows", ())]
    )
    cycle_bins: Dict[Tuple[float, int], Dict[str, int]] = {}
    for row in corrected:
        key = (float(row["target_period_s"]), int(row["cycle_count"]))
        counts = cycle_bins.setdefault(
            key,
            {"measurement_count": 0, "branch_tie_count": 0},
        )
        counts["measurement_count"] += 1
        counts["branch_tie_count"] += int(bool(row.get("branch_tie", False)))
    cycle_rows = [
        {
            "period_s": period,
            "cycle_count": cycle_count,
            **cycle_bins[(period, cycle_count)],
        }
        for period, cycle_count in sorted(cycle_bins)
    ]

    split_plan = selected_class.get("split_plan")
    half = selected_class.get("half_stability")
    differences = selected_class.get(
        "split_half_absolute_differences_km_s"
    )
    if not isinstance(split_plan, dict) or not isinstance(half, dict):
        raise ValueError("selected Stage B split-half evidence is absent")
    splits = [dict(row) for row in split_plan.get("splits", ())]
    summaries = {
        float(period): dict(summary)
        for period, summary in dict(half.get("period_summaries", {})).items()
    }
    periods = sorted(summaries)
    difference_array = np.asarray(differences, dtype=float)
    if difference_array.shape != (20, len(periods)) or len(splits) != 20:
        raise ValueError("selected Stage B split-half evidence is incomplete")
    split_stability_rows = []
    split_membership_rows = []
    for split in sorted(splits, key=lambda row: int(row["split_index"])):
        split_index = int(split["split_index"])
        for period_index, period in enumerate(periods):
            summary = summaries[period]
            split_stability_rows.append(
                {
                    "split_index": split_index,
                    "seed": int(split["seed"]),
                    "period_s": period,
                    "absolute_difference_km_s": float(
                        difference_array[split_index, period_index]
                    ),
                    "median_absolute_difference_km_s": summary.get(
                        "median_absolute_difference_km_s"
                    ),
                    "p90_absolute_difference_km_s": summary.get(
                        "p90_absolute_difference_km_s"
                    ),
                    "accepted": summary.get("accepted"),
                    "status": summary.get("status"),
                }
            )
        strata = dict(split.get("stratum_by_pair", {}))
        for side, field in (("A", "a_pair_names"), ("B", "b_pair_names")):
            for pair_name in split.get(field, ()):
                stratum = strata.get(str(pair_name))
                split_membership_rows.append(
                    {
                        "split_index": split_index,
                        "seed": int(split["seed"]),
                        "half": side,
                        "pair_name": str(pair_name),
                        "stratum": json.dumps(stratum),
                        "snr_field": split.get("snr_field"),
                        "odd_stratum_extra_side": split.get(
                            "odd_stratum_extra_side"
                        ),
                        "membership_sha256": split.get(
                            "membership_sha256"
                        ),
                        "plan_sha256": split_plan.get("plan_sha256"),
                    }
                )

    phase_rows = []
    for phase, payload in sorted(
        dict(evidence_payload.get("phase_matching_diagnostics", {})).items()
    ):
        source = dict(payload)
        diagnostic = dict(source.get("diagnostic", {}))
        row: Dict[str, object] = {
            "phase_convention": phase,
            "candidate_id": source.get("candidate_id"),
            "first_pass_alpha": source.get("first_pass_alpha"),
            "second_pass_alpha": source.get("second_pass_alpha"),
            "second_pass_ftan_executed": source.get(
                "second_pass_ftan_executed"
            ),
            "raw_closure_median_cycles": json.dumps(
                source.get("raw_closure_median_cycles", {}),
                sort_keys=True,
            ),
            "matched_closure_median_cycles": json.dumps(
                source.get("matched_closure_median_cycles", {}),
                sort_keys=True,
            ),
            "raw_valid_ridge_coverage": source.get(
                "raw_valid_ridge_coverage",
                diagnostic.get("raw_valid_ridge_coverage"),
            ),
            "matched_valid_ridge_coverage": source.get(
                "matched_valid_ridge_coverage",
                diagnostic.get("matched_valid_ridge_coverage"),
            ),
            "raw_boundary_fraction": source.get(
                "raw_boundary_fraction"
            ),
            "matched_boundary_fraction": source.get(
                "matched_boundary_fraction"
            ),
        }
        for key, value in diagnostic.items():
            row[key] = (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        phase_rows.append(row)
    closure = selected_class.get("closure")
    triplet_rows = (
        []
        if not isinstance(closure, dict)
        else [dict(row) for row in closure.get("triplet_rows", ())]
    )
    return {
        "candidate_grid_results": candidate_rows,
        "cycle_count_distribution": cycle_rows,
        "split_half_stability": split_stability_rows,
        "split_half_membership": split_membership_rows,
        "phase_matching_comparison": phase_rows,
        "triplet_closure": triplet_rows,
    }


def plot_reference_dispersion_stability(
    output_path: Path,
    reference_rows: Sequence[Dict[str, object]],
    split_rows: Sequence[Dict[str, object]],
) -> None:
    if not reference_rows or not split_rows:
        raise ValueError("reference stability plot requires non-empty rows")
    ensure_dir(output_path.parent)
    ordered = sorted(reference_rows, key=lambda row: float(row["period_s"]))
    periods = np.asarray([float(row["period_s"]) for row in ordered])
    velocity = np.asarray(
        [float(row["reference_velocity_km_s"]) for row in ordered]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(periods, velocity, "o-", color="#1f4e79", lw=1.8)
    axes[0].set_xlabel("Period (s)")
    axes[0].set_ylabel("Reference phase velocity (km/s)")
    axes[0].grid(True, color="#dddddd", lw=0.7)
    grouped: Dict[float, List[float]] = {}
    for row in split_rows:
        grouped.setdefault(float(row["period_s"]), []).append(
            float(row["absolute_difference_km_s"])
        )
    plot_periods = sorted(grouped)
    axes[1].boxplot(
        [grouped[period] for period in plot_periods],
        tick_labels=[f"{period:g}" for period in plot_periods],
        showfliers=True,
    )
    axes[1].axhline(0.03, color="#e67e22", ls="--", lw=1.2, label="median gate")
    axes[1].axhline(0.05, color="#c0392b", ls=":", lw=1.2, label="P90 gate")
    axes[1].set_xlabel("Period (s)")
    axes[1].set_ylabel("|A-B| phase velocity (km/s)")
    axes[1].legend(frameon=False)
    axes[1].grid(True, axis="y", color="#dddddd", lw=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_convention_validation(
    output_path: Path,
    candidate_rows: Sequence[Dict[str, object]],
    phase_matching_rows: Sequence[Dict[str, object]],
) -> None:
    if not candidate_rows or not phase_matching_rows:
        raise ValueError("phase validation plot requires non-empty rows")
    ensure_dir(output_path.parent)
    phases = sorted(
        {str(row["phase_convention"]) for row in candidate_rows}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for phase_index, phase in enumerate(phases):
        rows = [
            row
            for row in candidate_rows
            if str(row["phase_convention"]) == phase
        ]
        values = np.asarray(
            [
                (
                    float(row["closure_median_cycles"])
                    if row.get("closure_median_cycles") is not None
                    else np.nan
                )
                for row in rows
            ],
            dtype=float,
        )
        finite = np.isfinite(values) & (values < np.finfo(float).max / 2)
        axes[0].scatter(
            np.full(np.count_nonzero(finite), phase_index),
            values[finite],
            s=13,
            alpha=0.55,
            color=("#2f6f9f" if phase_index == 0 else "#c65f32"),
        )
        selected = [row for row in rows if bool(row.get("selected", False))]
        if selected:
            axes[0].scatter(
                [phase_index],
                [float(selected[0]["closure_median_cycles"])],
                marker="*",
                s=160,
                color="black",
                zorder=5,
            )
    axes[0].set_xticks(range(len(phases)), [phase.replace("_", "\n") for phase in phases])
    axes[0].set_ylabel("Triplet closure median (cycles)")
    axes[0].grid(True, axis="y", color="#dddddd", lw=0.7)
    phase_rows = {
        str(row["phase_convention"]): row for row in phase_matching_rows
    }
    x = np.arange(len(phases), dtype=float)
    raw = [float(phase_rows[phase]["raw_valid_ridge_coverage"]) for phase in phases]
    matched = [
        float(phase_rows[phase]["matched_valid_ridge_coverage"])
        for phase in phases
    ]
    axes[1].bar(x - 0.18, raw, width=0.36, label="Raw FTAN", color="#4c78a8")
    axes[1].bar(x + 0.18, matched, width=0.36, label="Phase matched", color="#f58518")
    axes[1].set_xticks(x, [phase.replace("_", "\n") for phase in phases])
    axes[1].set_ylabel("Valid ridge coverage")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)
    axes[1].grid(True, axis="y", color="#dddddd", lw=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_triplet_closure(
    output_path: Path,
    triplet_rows: Sequence[Dict[str, object]],
) -> None:
    if not triplet_rows:
        raise ValueError("triplet closure plot requires non-empty rows")
    ensure_dir(output_path.parent)
    periods = np.asarray([float(row["period_s"]) for row in triplet_rows])
    raw = np.asarray(
        [float(row["raw_closure_residual_s"]) for row in triplet_rows]
    )
    corrected = np.asarray(
        [float(row["corrected_closure_residual_s"]) for row in triplet_rows]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, values, title, color in (
        (axes[0], raw, "Before cycle correction", "#c44e52"),
        (axes[1], corrected, "After cycle correction", "#4c72b0"),
    ):
        axis.scatter(periods, values, s=13, alpha=0.55, color=color)
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_xlabel("Period (s)")
        axis.set_title(title)
        axis.grid(True, color="#dddddd", lw=0.7)
    axes[0].set_ylabel("Triplet closure residual (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ftan_example(
    output_path: Path,
    *,
    pair_name: str,
    distance_km: float,
    periods_s: np.ndarray,
    velocity_axis_km_s: np.ndarray,
    normalized_envelope_amplitude: np.ndarray,
    scaled_log_energy: np.ndarray,
    selected_ridge,
    beta1: float,
    beta2: float,
) -> Dict[str, object]:
    periods = np.asarray(periods_s, dtype=float)
    velocities = np.asarray(velocity_axis_km_s, dtype=float)
    amplitude = np.asarray(normalized_envelope_amplitude, dtype=float)
    energy = np.asarray(scaled_log_energy, dtype=float)
    expected_shape = (periods.size, velocities.size)
    if amplitude.shape != expected_shape or energy.shape != expected_shape:
        raise ValueError("FTAN example grids are inconsistent")
    candidates = find_candidate_ridges(
        scaled_log_energy=energy,
        normalized_envelope_amplitude=amplitude,
        periods_s=periods,
        velocity_axis_km_s=velocities,
        beta1=float(beta1),
        beta2=float(beta2),
        max_candidates=3,
    )
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    for axis, image, title, cmap in (
        (axes[0], amplitude, "Normalized envelope amplitude", "viridis"),
        (axes[1], energy, "Normalized log energy", "magma"),
    ):
        mesh = axis.pcolormesh(
            periods,
            velocities,
            image.T,
            shading="auto",
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
        )
        for candidate_index, candidate in enumerate(candidates):
            axis.plot(
                periods,
                candidate.group_velocities_km_s,
                lw=1.0,
                color=("white", "cyan", "lime")[candidate_index],
                alpha=0.9,
                label=f"candidate {candidate_index + 1}",
            )
            axis.fill_between(
                periods,
                candidate.group_velocities_km_s - 0.05,
                candidate.group_velocities_km_s + 0.05,
                color=("white", "cyan", "lime")[candidate_index],
                alpha=0.08,
            )
        axis.plot(
            periods,
            np.asarray(selected_ridge.group_velocities_km_s, dtype=float),
            color="black",
            lw=2.0,
            label="selected fundamental ridge",
        )
        axis.set_xlabel("Period (s)")
        axis.set_title(title)
        fig.colorbar(mesh, ax=axis, fraction=0.047, pad=0.03)
    axes[0].set_ylabel("Group velocity (km/s)")
    axes[0].legend(fontsize=7, frameon=True, loc="best")
    fig.suptitle(f"{pair_name} · {distance_km:.2f} km")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "pair_name": pair_name,
        "distance_km": float(distance_km),
        "period_count": int(periods.size),
        "velocity_grid_count": int(velocities.size),
        "candidate_ridge_count": len(candidates),
        "exclusion_corridor_km_s": 0.05,
        "energy_scale": [0.0, 1.0],
    }


def relative_to_output(output_path: Path, file_path: Path) -> str:
    return file_path.resolve().relative_to(output_path.parent.resolve()).as_posix()


def remove_stale_success_artifacts(
    output_dir: Path,
    figures_dir: Path,
) -> List[str]:
    removed: List[str] = []
    for artifact, relative_name in (
        (
            figures_dir / "wang_figure4_reproduction.png",
            "figures/wang_figure4_reproduction.png",
        ),
        (output_dir / "report.html", "report.html"),
        (
            output_dir / "measurements_initial_qc.csv",
            "measurements_initial_qc.csv",
        ),
        (
            output_dir / "measurements_corrected.csv",
            "measurements_corrected.csv",
        ),
        (
            output_dir / "measurements_right_qc.csv",
            "measurements_right_qc.csv",
        ),
        (
            output_dir / "continuous_reference_cycles.csv",
            "continuous_reference_cycles.csv",
        ),
        (output_dir / "fit_summary.csv", "fit_summary.csv"),
        (
            output_dir / "reference_alias_solutions.csv",
            "reference_alias_solutions.csv",
        ),
        (
            output_dir / "reference_cv_audit.csv",
            "reference_cv_audit.csv",
        ),
    ):
        if artifact.is_file() or artifact.is_symlink():
            artifact.unlink()
            removed.append(relative_name)
    return removed


def reference_alias_solution_rows(reference_fit) -> List[Dict[str, object]]:
    representatives = set(
        getattr(reference_fit, "representative_indices", ())
    )
    return [
        {
            "start_index": start.start_index,
            "start_hash": start.velocity_hash,
            "start_kind": start.kind,
            "start_base_velocity_km_s": start.base_velocity_km_s,
            "start_endpoint_slope_km_s": start.endpoint_slope_km_s,
            "start_velocities_km_s": json.dumps(
                start.velocities_km_s.tolist()
            ),
            "converged": solution.converged,
            "objective": solution.objective,
            "fold_holdout_losses": json.dumps(
                solution.fold_holdout_losses.tolist()
            ),
            "holdout_loss": solution.holdout_loss,
            "basin_id": solution.basin_id,
            "is_representative": index in representatives,
            "optimizer_message": solution.optimizer_message,
            "target_velocities_km_s": json.dumps(
                solution.target_velocities_km_s.tolist()
            ),
            "phase_slowness_s_km": json.dumps(
                solution.phase_slowness_s_km.tolist()
            ),
        }
        for index, (start, solution) in enumerate(
            zip(
                getattr(reference_fit, "starts", ()),
                getattr(reference_fit, "local_solutions", ()),
            )
        )
    ]


def reference_cv_audit_rows(reference_fit) -> List[Dict[str, object]]:
    """Flatten the frozen fold assignment and per-config losses for audit."""

    cv_result = reference_fit.cv_result
    assignment = cv_result.fold_assignment
    common = {
        "reference_fit_hash": reference_fit.result_hash,
        "cv_result_hash": cv_result.result_hash,
        "assignment_hash": assignment.assignment_hash,
    }
    rows: List[Dict[str, object]] = []
    for index, (
        quintile_id,
        azimuth_sector_id,
        fold_id,
    ) in enumerate(
        zip(
            assignment.distance_quintile_ids,
            assignment.azimuth_block_ids,
            assignment.fold_ids,
        )
    ):
        rows.append(
            {
                **common,
                "row_type": "observation_assignment",
                "observation_index": index,
                "distance_quintile_id": int(quintile_id),
                "azimuth_sector_id": int(azimuth_sector_id),
                "fold_id": int(fold_id),
            }
        )
    for fold, (training, holdout) in enumerate(
        zip(
            assignment.training_indices,
            assignment.holdout_indices,
        )
    ):
        rows.append(
            {
                **common,
                "row_type": "fold_membership",
                "fold_id": fold,
                "training_indices": json.dumps(training.tolist()),
                "holdout_indices": json.dumps(holdout.tolist()),
            }
        )
    for config_index, config in enumerate(cv_result.configs):
        for fold, loss in enumerate(config.fold_holdout_losses):
            rows.append(
                {
                    **common,
                    "row_type": "config_fold_loss",
                    "config_index": config_index,
                    "lambda_s": config.lambda_s,
                    "lambda_g": config.lambda_g,
                    "fold_id": fold,
                    "fold_holdout_loss": float(loss),
                    "mean_holdout_loss": config.mean_holdout_loss,
                    "optimizer_calls": config.optimizer_calls,
                    "selected": config is cv_result.selected,
                }
            )
    return rows


def write_report_html(
    output_path: Path,
    *,
    metadata: Dict[str, object],
    summary_rows: List[Dict[str, object]],
    figure_relpath: str,
) -> None:
    ensure_dir(output_path.parent)
    summary_table = "\n".join(
        "<tr><td>{period:.1f}</td><td>{left}</td><td>{corrected}</td><td>{right}</td><td>{ref}</td><td>{fit}</td><td>{ordinary}</td><td>{std}</td><td>{ci}</td></tr>".format(
            period=float(row["period_s"]),
            left=row["initial_count"],
            corrected=row["corrected_count"],
            right=row["right_qc_count"],
            ref=row["reference_velocity_km_s_display"],
            fit=row["fit_velocity_km_s_display"],
            ordinary=row["ordinary_ls_velocity_km_s_display"],
            std=row["std_velocity_km_s_display"],
            ci=row["bootstrap_velocity_ci95_display"],
        )
        for row in summary_rows
    )
    compare = metadata["paper_target_comparison_3s"]
    compare_html = """
      <table>
        <thead><tr><th>项目</th><th>论文图值</th><th>本次结果</th><th>是否一致（按两位小数）</th></tr></thead>
        <tbody>
          <tr><td>3 s 右图 V</td><td>2.70 km/s</td><td>{fit}</td><td>{fit_ok}</td></tr>
          <tr><td>3 s 右图 STDV</td><td>0.17 km/s</td><td>{std}</td><td>{std_ok}</td></tr>
        </tbody>
      </table>
    """.format(
        fit=html.escape(compare["observed_fit_velocity"]),
        std=html.escape(compare["observed_std_velocity"]),
        fit_ok="一致" if compare["fit_matches_paper"] else "不一致",
        std_ok="一致" if compare["std_matches_paper"] else "不一致",
    )
    output_path.write_text(
        """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wang Figure 4 All-Pairs Report</title>
  <style>
    :root {{
      --bg: #efe7d8;
      --paper: #fffdf8;
      --ink: #1f2933;
      --line: #d9cfbf;
      --muted: #5f6b75;
      --accent: #83512b;
      --soft: #f4ebdf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "PingFang SC", "Noto Sans SC", Arial, sans-serif;
      background: radial-gradient(circle at top left, #faf5ed 0, #efe7d8 46%, #e3d7c3 100%);
      line-height: 1.75;
    }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 54px; }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 48px rgba(84, 59, 34, 0.08);
      padding: 24px 28px;
      margin-bottom: 18px;
    }}
    .badge {{
      display: inline-block;
      padding: 5px 12px;
      border-radius: 999px;
      background: var(--soft);
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.05em;
    }}
    h1, h2 {{ margin: 0 0 12px; color: #342113; line-height: 1.3; }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 22px; }}
    p {{ margin: 10px 0; }}
    ul {{ margin: 8px 0 0 18px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th, td {{ border: 1px solid #e7dbc9; padding: 10px 12px; text-align: left; }}
    th {{ background: #f4ecdf; color: #51341b; }}
    figure {{
      margin: 0;
      border: 1px solid #eadfce;
      background: #fffaf2;
      padding: 12px;
      border-radius: 18px;
      cursor: zoom-in;
    }}
    figure img {{ width: 100%; display: block; border-radius: 12px; }}
    figcaption {{ margin-top: 10px; color: var(--muted); font-size: 14px; }}
    .lightbox {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.88);
      z-index: 9999;
      align-items: center;
      justify-content: center;
      cursor: zoom-out;
    }}
    .lightbox.active {{ display: flex; }}
    .lightbox img {{ max-width: 95vw; max-height: 95vh; box-shadow: 0 0 24px rgba(255,255,255,0.2); }}
    .lightbox .caption {{
      position: absolute;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      color: #eee;
      font-size: 0.95em;
      background: rgba(0,0,0,0.6);
      padding: 6px 14px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="card">
      <div class="badge">WORK SERVER · ALL PAIRS · WANG FIGURE 4</div>
      <h1>Wang 2017 Figure 4 全台站对复现报告</h1>
      <p>本报告只使用 LGX 的 work 服务器数据，在服务器上对所有可用 1D 台站对直接读取 <code>stack_pws.h5</code> 波形，按 Wang 等（2017）Figure 4 的规则重建 <code>3 s</code>、<code>3.5 s</code>、<code>4 s</code>、<code>5 s</code> 的相位走时-距离八联图。左列只保留满足 <code>SNR &gt; 4</code> 和群速度选择条件的原始相位走时；右列在此基础上再施加 <code>2π</code> 相位模糊校正和 <code>one-wavelength</code> 远场筛选。</p>
      <ul>
        <li><strong>服务器主机</strong>: {host}</li>
        <li><strong>生成时间</strong>: {generated_at}</li>
        <li><strong>STACK 根目录</strong>: {stack_root}</li>
        <li><strong>台站坐标表</strong>: {stations_csv}</li>
        <li><strong>总处理台对数</strong>: {processed_pair_count}</li>
        <li><strong>通过至少一个周期初筛的台对数</strong>: {pairs_with_any_initial_pass}</li>
        <li><strong>并行进程数</strong>: {max_workers}</li>
      </ul>
    </section>
    <section class="card">
      <h2>论文规则</h2>
      <table>
        <thead><tr><th>项目</th><th>本次执行标准</th><th>来源/说明</th></tr></thead>
        <tbody>
          <tr><td>周期</td><td><code>3 s</code>、<code>3.5 s</code>、<code>4 s</code>、<code>5 s</code></td><td>按固定目标周期逐个独立重采样。</td></tr>
          <tr><td>信噪比</td><td>leading SNR <code>&gt; 4</code> 且 trailing SNR <code>&gt; 4</code></td><td>论文原文明确写出。</td></tr>
          <tr><td>群速度限制</td><td>周期 <code>&lt; 4.5 s</code> 时 <code>U &lt;= 3.0 km/s</code>；周期 <code>&gt;= 4.5 s</code> 时 <code>U &lt;= 3.3 km/s</code></td><td>论文原文明确写出。</td></tr>
          <tr><td>参考相速度</td><td>每个周期独立用全部初筛测量估计</td><td>严格按 Figure 4 描述，不再施加跨周期人为约束。</td></tr>
          <tr><td>相位模糊校正</td><td>加上整数个 <code>N*T</code>，使预测与观测走时差小于半个周期</td><td>论文原文明确写出。</td></tr>
          <tr><td>远场条件</td><td><code>distance &gt;= V_ref * T</code></td><td>论文原文 one-wavelength criterion。</td></tr>
          <tr><td>实现细节</td><td>群到时搜索窗取表观速度 <code>1.6–5.0 km/s</code></td><td>论文未给逐行程序，属于实现假设；报告中单列说明。</td></tr>
          <tr><td>相位走时</td><td>对连续频率轴上已锚定并展开的 <code>t0</code> 重采样</td><td>不插值相位角，也不在目标周期重新解周。</td></tr>
        </tbody>
      </table>
    </section>
    <section class="card">
      <h2>3 s 目标核对</h2>
      {compare_html}
    </section>
    <section class="card">
      <h2>周期统计</h2>
      <table>
        <thead>
          <tr><th>Period (s)</th><th>左列点数</th><th>周数校正审计点数</th><th>右列点数</th><th>参考速度 (km/s)</th><th>Huber 速度 (km/s)</th><th>普通 LS 速度 (km/s)</th><th>路径 STDV (km/s)</th><th>Bootstrap 95% CI (km/s)</th></tr>
        </thead>
        <tbody>{summary_table}</tbody>
      </table>
    </section>
    <section class="card">
      <h2>Figure 4 八联图</h2>
      <figure onclick="openLightbox(this)">
        <img src="{figure_relpath}" alt="Reproduced Wang Figure 4">
        <figcaption>单击放大。绿色虚线表示相位模糊校正边界，右列黑线为最终保留点的 Huber 过原点稳健走时拟合。</figcaption>
      </figure>
    </section>
  </main>
  <div id="lightbox" class="lightbox" onclick="closeLightbox()">
    <img id="lightboxImg" src="" alt="">
    <div class="caption" id="lightboxCaption"></div>
  </div>
  <script>
    function openLightbox(fig) {{
      var img = fig.querySelector('img');
      var cap = fig.querySelector('figcaption');
      var lb = document.getElementById('lightbox');
      document.getElementById('lightboxImg').src = img.src;
      document.getElementById('lightboxImg').alt = img.alt || '';
      document.getElementById('lightboxCaption').innerHTML = cap ? cap.innerHTML : '';
      lb.classList.add('active');
    }}
    function closeLightbox() {{
      document.getElementById('lightbox').classList.remove('active');
    }}
    document.addEventListener('keydown', function(event) {{
      if (event.key === 'Escape') closeLightbox();
    }});
  </script>
</body>
</html>
""".format(
            host=html.escape(str(metadata["host"])),
            generated_at=html.escape(str(metadata["generated_at"])),
            stack_root=html.escape(str(metadata["stack_root"])),
            stations_csv=html.escape(str(metadata["stations_csv"])),
            processed_pair_count=html.escape(str(metadata["processed_pair_count"])),
            pairs_with_any_initial_pass=html.escape(str(metadata["pairs_with_any_initial_pass"])),
            max_workers=html.escape(str(metadata["max_workers"])),
            compare_html=compare_html,
            summary_table=summary_table,
            figure_relpath=html.escape(figure_relpath),
        ),
        encoding="utf-8",
    )


def write_formal_report_html(
    output_path: Path,
    *,
    metadata: Dict[str, object],
    summary_rows: Sequence[Dict[str, object]],
    formal_validation: Dict[str, object],
) -> None:
    """Write the formal Chinese report with method and data claims separated."""

    ensure_dir(output_path.parent)
    summary_html = "".join(
        "<tr><td>{period:g}</td><td>{left}</td><td>{right}</td>"
        "<td>{velocity}</td><td>{std}</td></tr>".format(
            period=float(row["period_s"]),
            left=html.escape(str(row.get("initial_count", ""))),
            right=html.escape(str(row.get("right_qc_count", ""))),
            velocity=html.escape(
                str(row.get("fit_velocity_km_s_display", "NA"))
            ),
            std=html.escape(
                str(row.get("std_velocity_km_s_display", "NA"))
            ),
        )
        for row in summary_rows
    )
    frozen = dict(metadata.get("frozen_candidate", {}))
    dependencies = html.escape(
        json.dumps(
            metadata.get("dependency_versions", {}),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    example_images = sorted(
        (output_path.parent / "figures" / "ftan_examples").glob(
            "example_*.png"
        )
    )
    example_html = "".join(
        '<figure><img src="{src}" alt="FTAN example"><figcaption>{name}</figcaption></figure>'.format(
            src=path.relative_to(output_path.parent).as_posix(),
            name=html.escape(path.stem),
        )
        for path in example_images
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wang FTAN 相速度测量正式审计报告</title>
<style>
body{{margin:0;background:#eef1f4;color:#17212b;font-family:"Times New Roman","PingFang SC",serif;line-height:1.7}}
main{{max-width:1180px;margin:auto;padding:28px}} section{{background:white;margin:18px 0;padding:24px 28px;border:1px solid #d7dde3;border-radius:12px}}
h1,h2{{color:#173b57}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #cfd7de;padding:7px 9px;text-align:left}}
img{{max-width:100%;height:auto;border:1px solid #d8dde2}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}
code{{overflow-wrap:anywhere}} .ok{{color:#176b3a;font-weight:bold}} .note{{background:#fff7df;border-left:4px solid #d99b21;padding:10px 14px}}
</style></head><body><main>
<h1>Wang FTAN 相速度测量正式审计报告</h1>
<p>运行状态：<span class="ok">{html.escape(str(metadata.get('run_status')))}</span>；阶段：{html.escape(str(metadata.get('stage')))}；主机：{html.escape(str(metadata.get('host')))}。</p>
<section><h2>1. 方法验证结果</h2>
<p>本结果使用 Wang/Bensen/Lin 谱系的 FTAN 相位测量与整数周期校正，不读取 DisperPicker 拾取曲线，也不把旧二维相速度图的参考值作为冻结参数。</p>
<ul><li>完整候选网格：{html.escape(str(formal_validation.get('candidate_count')))} 个；20 组半样本实际记录：{html.escape(str(formal_validation.get('split_count')))} 组。</li>
<li>FTAN 低/中/高分位示例：{html.escape(str(formal_validation.get('ftan_example_count')))} 张。</li>
<li>冻结相位口径：{html.escape(str(metadata.get('phase_convention')))}；候选：{html.escape(str(frozen.get('candidate_id')))}；α={html.escape(str(frozen.get('alpha')))}，β₁={html.escape(str(frozen.get('beta1')))}，β₂={html.escape(str(frozen.get('beta2')))}。</li></ul>
<div class="grid"><figure><img src="figures/reference_dispersion_stability.png" alt="reference stability"><figcaption>参考频散与半样本稳定性</figcaption></figure>
<figure><img src="figures/phase_convention_validation.png" alt="phase convention"><figcaption>两种相位口径与相位匹配诊断</figcaption></figure>
<figure><img src="figures/triplet_closure.png" alt="triplet closure"><figcaption>三台闭合校正前后</figcaption></figure></div></section>
<section><h2>2. 与 Wang 的数据差异</h2>
<p>方法尽量遵循 Wang 的 FTAN、参考频散、整数周期和右列筛选，但数据集、台网几何、噪声源分布、叠加时段、距离覆盖和自动质量控制均可能不同。因此，Wang 文中的速度数值只用于图形和结果讨论，不参与候选冻结、参考曲线拟合或周期数 N 的选择。</p>
<table><thead><tr><th>周期 (s)</th><th>左列点数</th><th>右列点数</th><th>稳健速度 (km/s)</th><th>路径速度 STD (km/s)</th></tr></thead><tbody>{summary_html}</tbody></table>
<figure><img src="figures/wang_figure4_ftan_paper_scale.png" alt="paper scale Figure 4"><figcaption>论文尺度：3、4、5 s，左右列共享坐标范围</figcaption></figure>
<figure><img src="figures/wang_figure4_ftan_full_range.png" alt="full range Figure 4"><figcaption>数据全范围图，避免论文坐标范围截断观测</figcaption></figure></section>
<section><h2>3. 差异证据</h2>
<p class="note">判断差异时应先检查每周期散点数量、距离—方位覆盖、N 分布、边界占用、三台闭合和半样本稳定性，再解释火山口下方二维慢速或快速异常。当前报告只证明相位测量链条，不把期望的慢速形态反向作为拾取约束。</p>
<ul><li><a href="candidate_grid_results.csv">300 候选完整结果</a></li><li><a href="cycle_count_distribution.csv">整数周期 N 分布</a></li>
<li><a href="reference_spatial_diagnostics.csv">距离五分位与 45° 方位诊断</a></li><li><a href="split_half_stability.csv">半样本稳定性</a> / <a href="split_half_membership.csv">成员表</a></li>
<li><a href="phase_matching_comparison.csv">相位匹配对照</a></li><li><a href="measurements_raw.csv">原始连续 FTAN 表</a> → <a href="measurements_left_qc.csv">左列</a> → <a href="measurements_corrected.csv">周期校正</a> → <a href="measurements_right_qc.csv">右列</a></li></ul>
<div class="grid">{example_html}</div></section>
<section><h2>4. 可重复性</h2><table>
<tr><th>Git SHA</th><td><code>{html.escape(str(metadata.get('git_commit_sha')))}</code></td></tr>
<tr><th>Python / 依赖</th><td>{html.escape(str(metadata.get('python_version')))} / <code>{dependencies}</code></td></tr>
<tr><th>输入</th><td><code>{html.escape(str(metadata.get('stack_root')))}</code>；文件数 {html.escape(str(metadata.get('input_file_count')))}</td></tr>
<tr><th>输入/代码/配置哈希</th><td><code>{html.escape(str(metadata.get('input_inventory_sha256')))} / {html.escape(str(metadata.get('code_sha256')))} / {html.escape(str(metadata.get('config_sha256')))}</code></td></tr>
<tr><th>开始/结束/退出</th><td>{html.escape(str(metadata.get('started_at')))} / {html.escape(str(metadata.get('finished_at')))} / {html.escape(str(metadata.get('exit_status')))}</td></tr>
</table></section></main></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def _checkpoint_jsonable(value):
    if isinstance(value, np.ndarray):
        return _checkpoint_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _checkpoint_jsonable(value.item())
    if isinstance(value, dict):
        return {
            str(key): _checkpoint_jsonable(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_checkpoint_jsonable(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("checkpoint values must be finite")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ValueError(
        f"checkpoint value has unsupported type {type(value).__name__}"
    )


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        _checkpoint_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_task_pair_name(task) -> str:
    if isinstance(task, dict):
        name = str(task.get("pair_name", ""))
    elif isinstance(task, (tuple, list)) and len(task) >= 3:
        name = f"{task[1]}__{task[2]}"
    else:
        name = ""
    if not name:
        raise ValueError("every checkpoint task requires a pair name")
    return name


def _scientific_result_rows_sha256(
    rows: Sequence[Dict[str, object]],
) -> str:
    excluded = {
        "elapsed_s",
        "runtime_s",
        "host",
        "worker_pid",
    }
    scientific = [
        {
            key: value
            for key, value in row.items()
            if key not in excluded
        }
        for row in rows
    ]
    return _canonical_json_sha256(scientific)


def run_checkpointed_pair_tasks(
    tasks: Sequence[object],
    *,
    output_dir: Path,
    chunk_size: int,
    config_sha256: str,
    process_task,
    max_workers: int,
    resume: bool,
    frozen_lineage: Optional[Dict[str, object]] = None,
    maximum_new_chunks: Optional[int] = None,
) -> Dict[str, object]:
    """Run deterministic pair chunks with atomic, lineage-checked resume."""

    size = int(chunk_size)
    workers = int(max_workers)
    if size <= 0:
        raise ValueError("checkpoint chunk_size must be positive")
    if workers < 1 or workers > 24:
        raise ValueError("checkpoint max_workers must lie in [1, 24]")
    if (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
    ):
        raise ValueError("checkpoint config_sha256 must contain 64 hex digits")
    if maximum_new_chunks is not None and int(maximum_new_chunks) < 0:
        raise ValueError("maximum_new_chunks cannot be negative")
    lineage = {
        "config_sha256": config_sha256,
        "frozen_lineage": dict(frozen_lineage or {}),
    }
    normalized_tasks = tuple(
        sorted(
            (
                (_checkpoint_task_pair_name(task), task)
                for task in tasks
            ),
            key=lambda row: row[0],
        )
    )
    pair_names = tuple(name for name, _ in normalized_tasks)
    if len(set(pair_names)) != len(pair_names):
        raise ValueError("checkpoint input contains duplicate station pairs")
    chunks = tuple(
        normalized_tasks[index : index + size]
        for index in range(0, len(normalized_tasks), size)
    )
    checkpoint_dir = Path(output_dir) / "checkpoints"
    ensure_dir(checkpoint_dir)
    if not resume:
        for stale_path in checkpoint_dir.glob("chunk_*.json"):
            stale_path.unlink()
        for stale_path in checkpoint_dir.glob("chunk_*.json.tmp"):
            stale_path.unlink()
    new_indices = []
    skipped_indices = []
    maximum = (
        None if maximum_new_chunks is None else int(maximum_new_chunks)
    )
    for chunk_index, chunk in enumerate(chunks):
        path = checkpoint_dir / f"chunk_{chunk_index:06d}.json"
        chunk_names = tuple(name for name, _ in chunk)
        membership_sha256 = _canonical_json_sha256(chunk_names)
        if resume and path.is_file() and path.stat().st_size > 0:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("chunk_index") != chunk_index
                or tuple(payload.get("input_pair_names", ()))
                != chunk_names
                or payload.get("input_membership_sha256")
                != membership_sha256
                or payload.get("lineage") != lineage
            ):
                raise ValueError(
                    f"checkpoint lineage mismatch in {path.name}"
                )
            skipped_indices.append(chunk_index)
            continue
        if maximum is not None and len(new_indices) >= maximum:
            continue
        chunk_tasks = tuple(task for _, task in chunk)
        if workers == 1 or len(chunk_tasks) <= 1:
            raw_results = [process_task(task) for task in chunk_tasks]
        else:
            with Pool(processes=min(workers, len(chunk_tasks))) as pool:
                raw_results = pool.map(process_task, chunk_tasks)
        result_rows = [
            dict(_checkpoint_jsonable(dict(result)))
            for result in raw_results
        ]
        result_names = tuple(
            str(row.get("pair_name", "")) for row in result_rows
        )
        if (
            len(result_rows) != len(chunk_names)
            or len(set(result_names)) != len(result_names)
            or set(result_names) != set(chunk_names)
        ):
            raise ValueError(
                "checkpoint result rows do not conserve chunk membership"
            )
        result_rows.sort(key=lambda row: str(row["pair_name"]))
        payload = {
            "chunk_index": chunk_index,
            "input_pair_names": chunk_names,
            "input_membership_sha256": membership_sha256,
            "lineage": lineage,
            "results": result_rows,
            "scientific_content_sha256": (
                _scientific_result_rows_sha256(result_rows)
            ),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        new_indices.append(chunk_index)

    expected_paths = tuple(
        checkpoint_dir / f"chunk_{index:06d}.json"
        for index in range(len(chunks))
    )
    complete = all(
        path.is_file() and path.stat().st_size > 0
        for path in expected_paths
    )
    merged_rows = []
    if complete:
        for chunk_index, (path, chunk) in enumerate(
            zip(expected_paths, chunks)
        ):
            payload = json.loads(path.read_text(encoding="utf-8"))
            chunk_names = tuple(name for name, _ in chunk)
            if (
                payload.get("chunk_index") != chunk_index
                or tuple(payload.get("input_pair_names", ()))
                != chunk_names
                or payload.get("input_membership_sha256")
                != _canonical_json_sha256(chunk_names)
                or payload.get("lineage") != lineage
                or payload.get("scientific_content_sha256")
                != _scientific_result_rows_sha256(payload["results"])
            ):
                raise ValueError(
                    f"checkpoint integrity mismatch in {path.name}"
                )
            merged_rows.extend(dict(row) for row in payload["results"])
        merged_rows.sort(key=lambda row: str(row["pair_name"]))
        merged_names = tuple(
            str(row.get("pair_name", "")) for row in merged_rows
        )
        if (
            len(merged_rows) != len(pair_names)
            or len(set(merged_names)) != len(merged_names)
            or merged_names != pair_names
        ):
            raise ValueError(
                "merged checkpoints violate pair-count conservation"
            )
    successful = sum(bool(row.get("ok")) for row in merged_rows)
    expected_rejections = sum(
        not bool(row.get("ok"))
        and row.get("failure_kind")
        == "expected_scientific_rejection"
        for row in merged_rows
    )
    unexpected = len(merged_rows) - successful - expected_rejections
    unexpected_fraction = (
        unexpected / len(merged_rows) if merged_rows else 0.0
    )
    return {
        "complete": complete,
        "new_chunk_indices": tuple(new_indices),
        "skipped_chunk_indices": tuple(skipped_indices),
        "chunk_count": len(chunks),
        "input_count": len(pair_names),
        "successful_pair_count": successful,
        "expected_scientific_rejection_count": expected_rejections,
        "unexpected_pair_exception_count": unexpected,
        "unexpected_pair_exception_fraction": unexpected_fraction,
        "formal_success_allowed": bool(
            complete and unexpected_fraction <= 0.01
        ),
        "results": tuple(merged_rows),
        "scientific_content_sha256": (
            _scientific_result_rows_sha256(merged_rows)
            if complete
            else None
        ),
        "lineage": lineage,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Wang 2017 Figure 4 on the work server using all 1D pairs.")
    parser.add_argument(
        "--stage",
        choices=("A", "B", "C"),
        default="B",
    )
    parser.add_argument(
        "--stack-root",
        type=Path,
        default=Path(
            "/mnt/data_hdd/lgx/MSH_ANT/stack/2014/"
            "1D_WANG_PWS_150s_20260620/"
            "STACK_SPIKE_REMOVED_DIAGFIT_20260628"
        ),
    )
    parser.add_argument(
        "--stations-csv",
        type=Path,
        default=Path("/mnt/data_hdd/lgx/MSH_ANT/inversion/phase_velocity_maps_aant_2014/wang_1d1d_server_wang_gvmax_min1lambda_cdisp/stations.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_figure4_allpairs_workserver_20260627"),
    )
    parser.add_argument("--component", default="ZZ")
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        help="Optional preprocessing YAML used only as lineage evidence.",
    )
    parser.add_argument(
        "--raw-stack-root",
        type=Path,
        default=Path(
            "/mnt/data_hdd/lgx/MSH_ANT/stack/2014/"
            "1D_WANG_PWS_150s_20260620/STACK"
        ),
        help="Optional matching pre-despike stack root for phase audit.",
    )
    parser.add_argument("--bbox", help="Optional station bbox as minlon,minlat,maxlon,maxlat")
    parser.add_argument("--bbox-mode", choices=("both", "either", "midpoint"), default="both")
    parser.add_argument("--max-workers", type=int, default=min(os.cpu_count() or 1, 24))
    parser.add_argument("--chunksize", type=int, default=128)
    parser.add_argument("--limit-pairs", type=int, help="Optional smoke-test cap.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-parameters", type=Path)
    parser.add_argument(
        "--phase-convention",
        choices=tuple(member.name for member in PhaseConvention),
    )
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    args = parser.parse_args(argv)
    if args.max_workers < 1 or args.max_workers > 24:
        parser.error("--max-workers must lie in [1, 24]")
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    if args.limit_pairs is not None and args.limit_pairs < 1:
        parser.error("--limit-pairs must be positive")
    overrides = (
        args.phase_convention,
        args.alpha,
        args.beta1,
        args.beta2,
    )
    if args.stage == "C":
        if args.frozen_parameters is None:
            parser.error("Stage C requires --frozen-parameters")
        if any(value is not None for value in overrides):
            parser.error(
                "Stage C rejects manual phase/alpha/beta overrides"
            )
    elif any(value is not None for value in overrides):
        parser.error(
            "formal Stage A/B reject manual phase/alpha/beta overrides"
        )
    return args


def load_stage_c_frozen_parameters(
    frozen_parameters_path: Path,
    *,
    expected_input_inventory_sha256: str,
    expected_code_sha256: str,
    expected_config_sha256: str,
) -> Dict[str, object]:
    """Load one passed Stage B manifest and verify its complete lineage."""

    path = Path(frozen_parameters_path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("Stage C frozen parameters file is missing or empty")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage C frozen parameters are not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Stage C frozen parameters must be a JSON object")
    if manifest.get("stage_b_status") != "passed":
        raise ValueError("Stage C requires a passed Stage B manifest")
    expected_hashes = {
        "input_inventory_sha256": expected_input_inventory_sha256,
        "code_sha256": expected_code_sha256,
        "config_sha256": expected_config_sha256,
    }
    for name, expected in expected_hashes.items():
        actual = manifest.get(name)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or not isinstance(actual, str)
            or actual != expected
        ):
            raise ValueError(f"Stage C {name} lineage mismatch")
    required_science = {
        "candidate_id",
        "phase_convention",
        "alpha",
        "beta1",
        "beta2",
        "validation_table_sha256",
    }
    if not required_science.issubset(manifest):
        raise ValueError("Stage C manifest lacks frozen scientific parameters")
    evidence_path = path.with_name("stage_b_validation_evidence.json")
    if not evidence_path.is_file() or evidence_path.stat().st_size <= 0:
        raise ValueError("Stage B validation evidence is missing or empty")
    try:
        evidence_payload = json.loads(
            evidence_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage B validation evidence is invalid") from exc
    evidence_sha256 = (
        wang_ftan_validation.stage_b_validation_evidence_sha256(
            evidence_payload
        )
    )
    if manifest.get("validation_table_sha256") != evidence_sha256:
        raise ValueError("Stage B validation evidence hash mismatch")
    _validate_stage_c_evidence_structure(evidence_payload, manifest)
    return dict(manifest)


def _validate_stage_c_evidence_structure(
    evidence_payload: object,
    manifest: Dict[str, object],
) -> None:
    """Reject a hash-consistent file that is not a formal Stage B record."""

    required_sections = {
        "budget",
        "benchmark_evidence",
        "selection",
        "candidate_results",
        "measurement_classes",
        "class_evidence",
        "phase_matching_diagnostics",
    }
    if (
        not isinstance(evidence_payload, dict)
        or set(evidence_payload) != required_sections
    ):
        raise ValueError("Stage B validation evidence structure is incomplete")
    candidates = evidence_payload["candidate_results"]
    if not isinstance(candidates, list) or len(candidates) != 300:
        raise ValueError("Stage B validation evidence has an incomplete grid")
    expected_grid = wang_ftan_validation.build_candidate_grid(
        phase_conventions=(
            PhaseConvention.BENSEN_VELOCITY_CCF.name,
            PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF.name,
        ),
        alpha_candidates=FtanConfig().alpha_candidates,
        beta1_candidates=FtanConfig().beta1_candidates,
        beta2_candidates=FtanConfig().beta2_candidates,
    )
    try:
        actual_signature = {
            (
                str(row["candidate_id"]),
                str(row["phase_convention"]),
                float(row["alpha"]),
                float(row["beta1"]),
                float(row["beta2"]),
            )
            for row in candidates
            if isinstance(row, dict)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Stage B validation evidence candidate structure is incomplete"
        ) from exc
    expected_signature = {
        (
            str(row["candidate_id"]),
            str(row["phase_convention"]),
            float(row["alpha"]),
            float(row["beta1"]),
            float(row["beta2"]),
        )
        for row in expected_grid
    }
    if actual_signature != expected_signature:
        raise ValueError("Stage B validation evidence has an incomplete grid")
    selected_rows = [
        row
        for row in candidates
        if str(row.get("candidate_id", ""))
        == str(manifest.get("candidate_id", ""))
    ]
    gate_fields = (
        "synthetic_passes",
        "ridge_passes",
        "instantaneous_period_passes",
        "alias_passes",
        "triplet_passes",
        "half_sample_passes",
        "boundary_passes",
    )
    science_fields = (
        "candidate_id",
        "phase_convention",
        "alpha",
        "beta1",
        "beta2",
    )
    if len(selected_rows) != 1:
        raise ValueError("Stage B frozen candidate is absent from its evidence")
    selected = selected_rows[0]
    if (
        any(selected.get(field) is not True for field in gate_fields)
        or any(
            selected.get(field) != manifest.get(field)
            for field in science_fields
        )
    ):
        raise ValueError("Stage B frozen candidate and evidence disagree")
    lineage_status = manifest.get("lineage_status")
    lineage_preferred = manifest.get(
        "lineage_preferred_phase_convention"
    )
    if not isinstance(lineage_status, str) or not lineage_status:
        raise ValueError("Stage B freeze decision lineage is incomplete")
    recomputed_decision = wang_ftan_validation.freeze_ftan_candidate(
        candidates,
        lineage_status=lineage_status,
        lineage_preferred_phase_convention=lineage_preferred,
    )
    if (
        not recomputed_decision.accepted
        or recomputed_decision.status != "passed"
        or recomputed_decision.selected_candidate.get("candidate_id")
        != manifest.get("candidate_id")
    ):
        raise ValueError("Stage B frozen candidate disagrees with its decision")
    left_hash = selected.get("left_observation_sha256")
    classes = evidence_payload["measurement_classes"]
    class_evidence = evidence_payload["class_evidence"]
    if (
        not isinstance(left_hash, str)
        or len(left_hash) != 64
        or not isinstance(classes, dict)
        or not isinstance(class_evidence, dict)
        or left_hash not in classes
        or left_hash not in class_evidence
        or str(selected["candidate_id"]) not in classes[left_hash]
    ):
        raise ValueError("Stage B selected measurement-class evidence is incomplete")
    try:
        restored_class = wang_ftan_validation._unpack_stage_b_class_result(
            class_evidence[left_hash]
        )
        restored_hash = wang_ftan_validation.hash_left_observation_table(
            restored_class["continuous_left_rows"]
        )
        complete_class = (
            wang_ftan_validation._successful_class_evidence_is_complete(
                restored_class
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Stage B selected measurement-class evidence is incomplete"
        ) from exc
    if restored_hash != left_hash or not complete_class:
        raise ValueError(
            "Stage B selected measurement-class evidence is incomplete"
        )
    budget = evidence_payload["budget"]
    benchmark = evidence_payload["benchmark_evidence"]
    selection = evidence_payload["selection"]
    if (
        not isinstance(budget, dict)
        or budget.get("accepted") is not True
        or budget.get("status") != "accepted"
        or not isinstance(benchmark, dict)
        or not isinstance(benchmark.get("benchmark_input_sha256"), str)
        or len(benchmark["benchmark_input_sha256"]) != 64
        or not isinstance(selection, dict)
        or selection.get("seed") != 20260717
        or selection.get("max_random_pairs") != 2000
    ):
        raise ValueError("Stage B budget or selection evidence is incomplete")
    phase_diagnostics = evidence_payload["phase_matching_diagnostics"]
    expected_phases = {
        PhaseConvention.BENSEN_VELOCITY_CCF.name,
        PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF.name,
    }
    if not isinstance(phase_diagnostics, dict) or set(
        phase_diagnostics
    ) != expected_phases:
        raise ValueError("Stage B phase-matching evidence is incomplete")
    for phase, row in phase_diagnostics.items():
        diagnostic = row.get("diagnostic") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("phase_convention") != phase
            or row.get("second_pass_ftan_executed") is not True
            or not isinstance(diagnostic, dict)
            or diagnostic.get("freeze_raw_ftan") is not True
            or diagnostic.get("design_revision_required") is not False
            or diagnostic.get("status") != "raw_ftan_frozen"
        ):
            raise ValueError("Stage B phase-matching evidence is incomplete")


def formal_scientific_config_sha256(*, component: str) -> str:
    """Hash all stage-invariant FTAN grids and frozen QC thresholds."""

    config = FtanConfig()
    return _canonical_json_sha256(
        {
            "component": str(component),
            "periods_s": config.periods_s,
            "group_velocities_km_s": config.group_velocities_km_s,
            "target_periods_s": config.target_periods_s,
            "alpha_candidates": config.alpha_candidates,
            "beta1_candidates": config.beta1_candidates,
            "beta2_candidates": config.beta2_candidates,
            "snr_threshold": SNR_THRESHOLD,
            "signal_velocity_window_km_s": (
                SIGNAL_VMIN_KM_S,
                SIGNAL_VMAX_KM_S,
            ),
            "group_velocity_limits_km_s": (
                GROUP_VMAX_SHORT_KM_S,
                GROUP_VMAX_LONG_KM_S,
            ),
            "minimum_noise_samples": MIN_NOISE_SAMPLES,
        }
    )


def formal_success_lineage_metadata(
    *,
    stage: str,
    component: str,
    input_inventory_sha256: str,
    code_sha256: str,
    config_sha256: str,
    frozen_manifest: Optional[Dict[str, object]],
    frozen_parameters_path: Optional[Path],
    checkpoint_run: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Build the immutable formal lineage block for success metadata."""

    payload: Dict[str, object] = {
        "stage": str(stage),
        "component": str(component),
        "input_inventory_sha256": str(input_inventory_sha256),
        "code_sha256": str(code_sha256),
        "config_sha256": str(config_sha256),
        "scientific_content_sha256": (
            None
            if checkpoint_run is None
            else checkpoint_run.get("scientific_content_sha256")
        ),
        "checkpoint_lineage": (
            None
            if checkpoint_run is None
            else checkpoint_run.get("lineage")
        ),
    }
    if frozen_manifest is not None:
        payload["frozen_candidate"] = {
            name: frozen_manifest[name]
            for name in (
                "candidate_id",
                "phase_convention",
                "alpha",
                "beta1",
                "beta2",
                "validation_table_sha256",
            )
        }
    else:
        payload["frozen_candidate"] = None
    if frozen_parameters_path is not None:
        frozen_path = Path(frozen_parameters_path)
        if not frozen_path.is_file() or frozen_path.stat().st_size <= 0:
            raise ValueError("formal success freeze file is missing")
        payload["frozen_parameters_file_sha256"] = hashlib.sha256(
            frozen_path.read_bytes()
        ).hexdigest()
    else:
        payload["frozen_parameters_file_sha256"] = None
    return payload


def formal_runtime_code_sha256() -> str:
    """Hash the three production modules that define formal FTAN results."""

    paths = {
        Path(__file__).resolve(),
        Path(wang_ftan_validation.__file__).resolve(),
        Path(phase_matched_second_pass_ftan.__code__.co_filename).resolve(),
    }
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def build_terminal_failure_metadata(
    *,
    args,
    terminal_failure_reason: str,
    processed_pair_count: int,
    successful_pair_count: int,
    accepted_measurement_count: int,
    failures: Sequence[Dict[str, object]],
    removed_stale_artifacts: Sequence[str],
    reference_fields: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build one common audit schema for every terminal failure gate."""

    expected_rejection_count = sum(
        row.get("failure_kind") == "expected_scientific_rejection"
        for row in failures
    )
    unexpected_exception_count = sum(
        row.get("failure_kind") == "unexpected_pair_exception"
        for row in failures
    )
    metadata: Dict[str, object] = {
        "host": platform.node(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_status": "failed",
        "terminal_failure_reason": terminal_failure_reason,
        "stack_root": str(args.stack_root),
        "stations_csv": str(args.stations_csv),
        "component": str(getattr(args, "component", "ZZ")),
        "input_inventory_json": "input_inventory.json",
        "processed_pair_count": processed_pair_count,
        "successful_pair_count": successful_pair_count,
        "accepted_measurement_count": accepted_measurement_count,
        "expected_scientific_rejection_count": expected_rejection_count,
        "unexpected_pair_exception_count": unexpected_exception_count,
        "failure_count": len(failures),
        "target_periods_s": list(TARGET_PERIODS_S),
        "removed_stale_artifacts": list(removed_stale_artifacts),
    }
    if reference_fields is not None:
        overlap = set(metadata).intersection(reference_fields)
        if overlap:
            raise ValueError(
                "reference failure fields overlap common metadata: "
                + ", ".join(sorted(overlap))
            )
        metadata.update(reference_fields)
    return metadata


def build_stage_b_inventory_rows(
    tasks: Sequence[Sequence[object]],
    *,
    component: str,
) -> Tuple[Dict[str, object], ...]:
    """Build the candidate-independent Stage B sampling table from stacks."""

    rows: List[Dict[str, object]] = []
    observed_names = set()
    for task in tasks:
        if len(task) < 7:
            raise ValueError("Stage B inventory tasks require seven fields")
        path, source, receiver, lon_a, lat_a, lon_b, lat_b = task[:7]
        pair_name = f"{source}__{receiver}"
        if pair_name in observed_names:
            raise ValueError("Stage B inventory pair names must be unique")
        observed_names.add(pair_name)
        distance_km = great_circle_km(
            float(lat_a),
            float(lon_a),
            float(lat_b),
            float(lon_b),
        )
        azimuth_deg = forward_azimuth_deg(
            float(lat_a),
            float(lon_a),
            float(lat_b),
            float(lon_b),
        )
        preliminary_snr = float("nan")
        try:
            trace = read_stack_trace(
                Path(path),
                pair_name=pair_name,
                component=component,
            )
            preliminary_snr = compute_preliminary_snr(
                trace,
                distance_km=distance_km,
            )
        except MeasurementError:
            pass
        rows.append(
            {
                "pair_name": pair_name,
                "distance_km": distance_km,
                "azimuth_deg": azimuth_deg,
                "preliminary_snr": preliminary_snr,
            }
        )
    return tuple(sorted(rows, key=lambda row: str(row["pair_name"])))


def stage_b_candidate_synthetic_status(
    candidate,
    *,
    phase_alpha_cache: Dict[Tuple[str, float], bool],
    beta_cache: Dict[Tuple[float, float], bool],
) -> str:
    """Run cached, candidate-specific convention/alpha and beta checks."""

    row = dict(candidate)
    convention = PhaseConvention[str(row["phase_convention"])]
    alpha = float(row["alpha"])
    beta1 = float(row["beta1"])
    beta2 = float(row["beta2"])
    phase_key = (convention.name, alpha)
    if phase_key not in phase_alpha_cache:
        period_s = 3.5
        true_velocity_km_s = 2.7
        distance_km = 20.0
        paper_phase_rad = 0.2
        omega_rad_s = 2.0 * np.pi / period_s
        true_phase_time_s = distance_km / true_velocity_km_s
        definition = convention.definition
        group_time_s = true_phase_time_s - (
            definition.formula_phase_sign * paper_phase_rad
            + definition.fixed_phase_rad
        ) / omega_rad_s
        dt_s = 0.02
        time_s = np.arange(0.0, 40.0, dt_s)
        local_time_s = time_s - group_time_s
        prepared_waveform = np.exp(
            -0.5 * (local_time_s / (1.5 * period_s)) ** 2
        ) * np.cos(
            omega_rad_s * local_time_s
            + definition.scipy_phase_multiplier * paper_phase_rad
        )
        if definition.apply_negative_time_derivative:
            symmetric_ccf = np.empty_like(prepared_waveform)
            symmetric_ccf[0] = 0.0
            symmetric_ccf[1:] = -np.cumsum(
                0.5
                * (prepared_waveform[1:] + prepared_waveform[:-1])
                * np.diff(time_s)
            )
        else:
            symmetric_ccf = prepared_waveform
        trace = DatTrace(
            pair_name=f"synthetic_{convention.name}_alpha_{alpha:g}",
            distance_km=distance_km,
            dt_s=dt_s,
            time_s=time_s,
            positive_lag=symmetric_ccf,
            negative_lag_reversed=symmetric_ccf,
            symmetric_waveform=symmetric_ccf,
            lon_a=0.0,
            lat_a=0.0,
            lon_b=0.1,
            lat_b=0.1,
        )
        measurement = measure_single_period(
            trace,
            period_s=period_s,
            vmin_km_s=1.6,
            vmax_km_s=5.0,
            alpha=alpha,
            convention=convention,
        )
        phase_passes = measurement is not None
        if measurement is not None:
            cycle = resolve_cycle_count(
                raw_time_s=measurement.raw_phase_time_s,
                reference_time_s=true_phase_time_s,
                period_s=(
                    2.0 * np.pi / measurement.omega_inst_rad_s
                ),
                convention=convention,
            )
            recovered_velocity = distance_km / cycle.corrected_time_s
            phase_passes = bool(
                abs(recovered_velocity - true_velocity_km_s)
                / true_velocity_km_s
                <= 0.005
            )
        phase_alpha_cache[phase_key] = phase_passes
    beta_key = (beta1, beta2)
    if beta_key not in beta_cache:
        config = FtanConfig()
        periods = config.periods_s
        velocity = config.group_velocities_km_s
        expected_rows = 100 + np.rint(
            5.0
            * np.sin(
                np.linspace(0.0, 2.0 * np.pi, periods.size)
            )
        ).astype(int)
        energy = np.full((periods.size, velocity.size), 0.01)
        amplitude = np.full_like(energy, 0.01)
        energy[np.arange(periods.size), expected_rows] = 0.9
        amplitude[np.arange(periods.size), expected_rows] = 0.9
        middle = periods.size // 2
        energy[middle, 250] = 1.0
        amplitude[middle, 250] = 1.0
        ridges = find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=amplitude,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=beta1,
            beta2=beta2,
            max_candidates=1,
        )
        selected = select_fundamental_ridge(
            ridges,
            periods_s=periods,
        )
        beta_cache[beta_key] = bool(
            selected.quality.accepted
            and np.max(np.abs(selected.row_indices - expected_rows)) <= 1
        )
    return (
        "accepted"
        if phase_alpha_cache[phase_key] and beta_cache[beta_key]
        else "rejected"
    )


def build_stage_b_closure_triplets(
    *,
    inventory_rows: Sequence[Dict[str, object]],
    station_coordinates: Dict[str, Tuple[float, float]],
    maximum_triplets: int = 1000,
) -> Tuple[Tuple[Dict[str, object], ...], Tuple[str, ...]]:
    """Construct deterministic near-collinear triplets on the dense 1-D array."""

    limit = int(maximum_triplets)
    if limit < 1 or limit > 1000:
        raise ValueError("maximum_triplets must lie in [1, 1000]")
    pair_lookup: Dict[frozenset, str] = {}
    for row in inventory_rows:
        pair_name = str(row["pair_name"])
        stations = tuple(pair_name.split("__"))
        if len(stations) != 2 or stations[0] == stations[1]:
            raise ValueError("Stage B pair names must encode two stations")
        key = frozenset(stations)
        if key in pair_lookup:
            raise ValueError("Stage B inventory contains a duplicate station pair")
        pair_lookup[key] = pair_name
    coordinates = {
        str(code): np.asarray(value, dtype=float)
        for code, value in station_coordinates.items()
    }
    usable_codes = tuple(
        sorted(
            code
            for code, coordinate in coordinates.items()
            if coordinate.shape == (2,) and np.all(np.isfinite(coordinate))
        )
    )
    if len(usable_codes) < 3:
        return (), ()
    matrix = np.vstack([coordinates[code] for code in usable_codes])
    centered = matrix - np.mean(matrix, axis=0)
    if np.allclose(centered, 0.0):
        return (), ()
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    principal = right[0]
    projected = centered @ principal
    ordered_codes = tuple(
        code
        for _, code in sorted(
            zip(projected.tolist(), usable_codes),
            key=lambda item: (item[0], item[1]),
        )
    )
    triplets: List[Dict[str, object]] = []
    edge_names = set()
    station_count = len(ordered_codes)
    for step in range(1, (station_count - 1) // 2 + 1):
        for index in range(station_count - 2 * step):
            codes = (
                ordered_codes[index],
                ordered_codes[index + step],
                ordered_codes[index + 2 * step],
            )
            edge_keys = (
                frozenset(codes[:2]),
                frozenset(codes[1:]),
                frozenset((codes[0], codes[2])),
            )
            if any(key not in pair_lookup for key in edge_keys):
                continue
            geometry = wang_ftan_validation.evaluate_triplet_geometry(
                station_a=station_coordinates[codes[0]],
                station_b=station_coordinates[codes[1]],
                station_c=station_coordinates[codes[2]],
            )
            if not geometry.accepted:
                continue
            pair_names = tuple(pair_lookup[key] for key in edge_keys)
            triplet_id = "__".join(codes)
            triplets.append(
                {
                    "triplet_id": triplet_id,
                    "station_a_code": codes[0],
                    "station_b_code": codes[1],
                    "station_c_code": codes[2],
                    "pair_ab_name": pair_names[0],
                    "pair_bc_name": pair_names[1],
                    "pair_ac_name": pair_names[2],
                }
            )
            edge_names.update(pair_names)
            if len(triplets) >= limit:
                return tuple(triplets), tuple(sorted(edge_names))
    return tuple(triplets), tuple(sorted(edge_names))


def measure_stage_b_candidate_from_tasks(
    *,
    candidate,
    selection,
    task_by_pair: Dict[str, Sequence[object]],
    component: str,
    max_workers: int,
    synthetic_validation_status: str,
) -> Dict[str, object]:
    """Measure one frozen candidate on exactly the selected real HDF5 pairs."""

    candidate_row = dict(candidate)
    required = {
        "phase_convention",
        "alpha",
        "beta1",
        "beta2",
    }
    if not required.issubset(candidate_row):
        raise ValueError("Stage B candidate lacks scientific parameters")
    scientific_parameters = {
        "phase_convention": str(candidate_row["phase_convention"]),
        "alpha": float(candidate_row["alpha"]),
        "beta1": float(candidate_row["beta1"]),
        "beta2": float(candidate_row["beta2"]),
    }
    selected_names = tuple(sorted(selection.selected_pair_names))
    if any(name not in task_by_pair for name in selected_names):
        raise ValueError("Stage B selection references an unknown pair task")
    tasks = tuple(
        tuple(task_by_pair[name][:7])
        + (str(component), dict(scientific_parameters))
        for name in selected_names
    )
    workers = int(max_workers)
    if workers < 1 or workers > 24:
        raise ValueError("Stage B measurement workers must lie in [1, 24]")
    if synthetic_validation_status not in ("accepted", "rejected"):
        raise ValueError("Stage B synthetic validation status is invalid")
    if not tasks:
        raise ValueError("Stage B candidate requires selected real pair tasks")
    if workers == 1:
        pool_started = time.perf_counter()
        results = tuple(process_one_pair(task) for task in tasks)
        pool_ended = time.perf_counter()
        pool_lifecycle = {
            "status": "completed_inline",
            "creator_pid": os.getpid(),
            "requested_worker_count": 1,
            "worker_pids": (),
            "pool_started_monotonic_s": pool_started,
            "pool_ended_monotonic_s": pool_ended,
        }
    else:
        if mp.current_process().daemon:
            raise RuntimeError("a daemon worker cannot create a real-pair pool")
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("Stage B real-pair pool requires fork")
        creator_pid = os.getpid()
        requested_workers = min(workers, len(tasks))
        pool_started = time.perf_counter()
        context = mp.get_context("fork")
        pool = context.Pool(processes=requested_workers)
        worker_pids = tuple(
            sorted(int(process.pid) for process in pool._pool)
        )
        try:
            results = tuple(
                pool.imap(process_one_pair, tasks, chunksize=1)
            )
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
        pool_ended = time.perf_counter()
        pool_lifecycle = {
            "status": "completed",
            "creator_pid": creator_pid,
            "requested_worker_count": requested_workers,
            "worker_pids": worker_pids,
            "pool_started_monotonic_s": pool_started,
            "pool_ended_monotonic_s": pool_ended,
        }
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for result in results:
        name = str(result.get("pair_name", ""))
        if name not in selected_names:
            raise ValueError("Stage B worker returned an unselected pair")
        if not bool(result.get("ok")):
            failures.append(
                {
                    "pair_name": name,
                    "reason": str(result.get("reason", "unknown")),
                    "failure_kind": str(
                        result.get(
                            "failure_kind",
                            "unexpected_pair_exception",
                        )
                    ),
                }
            )
            continue
        for source in result.get("continuous_observations", ()):
            row = dict(source)
            if str(row.get("pair_name", "")) != name:
                raise ValueError(
                    "Stage B continuous row pair disagrees with its task"
                )
            rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["pair_name"]),
            float(row["T_inst"]),
        )
    )
    row_keys = tuple(
        (str(row["pair_name"]), float(row["T_inst"])) for row in rows
    )
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("Stage B candidate produced duplicate LEFT rows")
    outermost_count = sum(
        bool(dict(row["ridge_fields"]).get("outermost_velocity_cell"))
        for row in rows
    )
    successful_pair_count = sum(bool(result.get("ok")) for result in results)
    expected_rejection_count = sum(
        not bool(result.get("ok"))
        and str(result.get("failure_kind", ""))
        == "expected_scientific_rejection"
        for result in results
    )
    unexpected_exception_count = (
        len(results) - successful_pair_count - expected_rejection_count
    )
    return {
        "continuous_left_rows": tuple(rows),
        "accepted_outermost_velocity_cell_count": outermost_count,
        "synthetic_validation_status": synthetic_validation_status,
        "pair_failure_rows": tuple(failures),
        "processed_pair_count": len(results),
        "successful_pair_count": successful_pair_count,
        "expected_scientific_rejection_count": expected_rejection_count,
        "unexpected_pair_exception_count": unexpected_exception_count,
        "pool_lifecycle": pool_lifecycle,
    }


def stage_b_corrected_target_rows(
    left_rows: Sequence[Dict[str, object]],
    *,
    reference_fit,
) -> Tuple[wang_ftan_validation.CorrectedTargetObservation, ...]:
    """Resample one candidate's LEFT curves and resolve their target cycles."""

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for source in left_rows:
        row = dict(source)
        grouped.setdefault(str(row["pair_name"]), []).append(row)
    targets = []
    for pair_name, pair_rows in sorted(grouped.items()):
        ordered = sorted(pair_rows, key=lambda row: float(row["T_inst"]))
        periods = np.asarray(
            [float(row["T_inst"]) for row in ordered],
            dtype=float,
        )
        if periods.size < 2 or np.any(np.diff(periods) <= 0):
            continue
        distance_values = {float(row["distance_km"]) for row in ordered}
        if len(distance_values) != 1:
            raise ValueError("Stage B pair distance changed across LEFT rows")
        distance_km = distance_values.pop()
        ridge_fields = [dict(row["ridge_fields"]) for row in ordered]
        resampled = resample_wang_target_periods(
            target_periods_s=TARGET_PERIODS_S,
            continuous_periods_s=periods,
            anchored_raw_phase_time_s=np.asarray(
                [float(row["t0"]) for row in ordered]
            ),
            group_time_s=np.asarray(
                [
                    float(
                        row.get(
                            "group_time_s",
                            distance_km / float(row["U"]),
                        )
                    )
                    for row in ordered
                ]
            ),
            signal_peak=np.asarray(
                [float(row["signal_peak"]) for row in ordered]
            ),
            leading_noise_rms=np.asarray(
                [float(row["leading_rms"]) for row in ordered]
            ),
            trailing_noise_rms=np.asarray(
                [float(row["trailing_rms"]) for row in ordered]
            ),
            ridge_normalized_log_energy=np.asarray(
                [
                    float(fields.get("normalized_log_energy", 0.0))
                    for fields in ridge_fields
                ]
            ),
            ridge_normalized_envelope_amplitude=np.asarray(
                [
                    float(
                        fields.get(
                            "normalized_envelope_amplitude",
                            0.0,
                        )
                    )
                    for fields in ridge_fields
                ]
            ),
            ridge_adjacent_jump_km_s=np.asarray(
                [
                    float(fields.get("adjacent_jump_km_s", 0.0))
                    for fields in ridge_fields
                ]
            ),
            valid_mask=np.ones(periods.size, dtype=bool),
            distance_km=distance_km,
        )
        accepted = tuple(row for row in resampled if row.accepted)
        if not accepted:
            continue
        convention_name = str(
            ordered[0].get(
                "convention",
                PhaseConvention.BENSEN_VELOCITY_CCF.name,
            )
        )
        convention = PhaseConvention[convention_name]
        cycles = resolve_reference_cycles(
            raw_times_s=np.asarray(
                [row.anchored_raw_phase_time_s for row in accepted]
            ),
            distance_km=np.full(len(accepted), distance_km),
            observation_periods_s=np.asarray(
                [row.target_period_s for row in accepted]
            ),
            reference_periods_s=reference_fit.periods_s,
            reference_slowness_s_km=reference_fit.phase_slowness_s_km,
            convention=convention,
        )
        for row, cycle in zip(accepted, cycles):
            targets.append(
                wang_ftan_validation.CorrectedTargetObservation(
                    pair_name=pair_name,
                    target_period_s=row.target_period_s,
                    raw_time_s=row.anchored_raw_phase_time_s,
                    cycle_count=cycle.cycle_count,
                    corrected_time_s=cycle.corrected_time_s,
                    reference_time_s=cycle.reference_time_s,
                    reference_residual_s=cycle.corrected_residual_s,
                    leading_snr=row.leading_snr,
                    trailing_snr=row.trailing_snr,
                    left_qc_accepted=True,
                    branch_tie=cycle.branch_tie,
                )
            )
    return tuple(
        sorted(
            targets,
            key=lambda row: (row.pair_name, row.target_period_s),
        )
    )


def fit_stage_b_full_reference(
    *,
    left_rows: Sequence[Dict[str, object]],
    candidate_ids: Sequence[str],
    maximum_optimizer_calls: int,
) -> wang_ftan_validation.FullReferenceEvidence:
    """Fit one measurement class and package auditable candidate evidence."""

    if not tuple(candidate_ids):
        raise ValueError("Stage B reference class requires candidate IDs")
    observations = tuple(
        ReferenceObservation(
            pair_name=str(row["pair_name"]),
            distance_km=float(row["distance_km"]),
            azimuth_deg=float(row["azimuth_deg"]),
            instantaneous_period_s=float(row["T_inst"]),
            anchored_raw_time_s=float(row["t0"]),
            group_slowness_s_km=(1.0 / float(row["U"])),
            convention=PhaseConvention[
                str(
                    row.get(
                        "convention",
                        PhaseConvention.BENSEN_VELOCITY_CCF.name,
                    )
                )
            ],
        )
        for row in left_rows
    )
    fit = fit_reference_dispersion(observations)
    maximum = int(maximum_optimizer_calls)
    if fit.optimizer_calls > maximum or maximum != 753:
        raise ValueError("Stage B full reference exceeded its optimizer budget")
    representatives = tuple(
        fit.local_solutions[index]
        for index in fit.representative_indices[:5]
    )
    accepted = fit.status == "accepted"
    corrected = (
        stage_b_corrected_target_rows(left_rows, reference_fit=fit)
        if accepted
        else ()
    )
    payload = {
        "fit_result_sha256": str(fit.result_hash),
        "candidate_ids": tuple(sorted(str(value) for value in candidate_ids)),
        "corrected_rows": [asdict(row) for row in corrected],
    }
    return wang_ftan_validation.FullReferenceEvidence(
        status="accepted" if accepted else "rejected",
        alias_status="accepted" if accepted else "rejected",
        lambda_s=fit.lambda_s,
        lambda_g=fit.lambda_g,
        basin_starts=representatives,
        optimizer_calls=fit.optimizer_calls,
        corrected_rows=corrected,
        result_sha256=_canonical_json_sha256(payload),
    )


def fit_stage_b_split_half_reference(
    *,
    left_rows: Sequence[Dict[str, object]],
    full_reference,
    split_index: int,
    seed: int,
    side: str,
    lambda_s: float,
    lambda_g: float,
    basin_starts: Sequence[object],
    maxiter: int,
) -> wang_ftan_validation.SplitHalfFitEvidence:
    """Refit one deterministic half from five frozen basins and no CV."""

    if int(split_index) < 0 or int(seed) != 20260717 + int(split_index):
        raise ValueError("split-half seed does not match the frozen schedule")
    if side not in ("A", "B") or int(maxiter) != 300:
        raise ValueError("split-half side or maxiter is invalid")
    starts = tuple(basin_starts)
    if not 1 <= len(starts) <= 5:
        raise ValueError("split-half fitting requires one to five basin starts")
    full_starts = tuple(full_reference.basin_starts[: len(starts)])
    if len(full_starts) != len(starts) or any(
        expected is not actual
        for expected, actual in zip(full_starts, starts)
    ):
        raise ValueError("split-half starts differ from the full reference")
    observations = tuple(
        ReferenceObservation(
            pair_name=str(row["pair_name"]),
            distance_km=float(row["distance_km"]),
            azimuth_deg=float(row["azimuth_deg"]),
            instantaneous_period_s=float(row["T_inst"]),
            anchored_raw_time_s=float(row["t0"]),
            group_slowness_s_km=1.0 / float(row["U"]),
            convention=PhaseConvention[
                str(
                    row.get(
                        "convention",
                        PhaseConvention.BENSEN_VELOCITY_CCF.name,
                    )
                )
            ],
        )
        for row in left_rows
    )
    periods = FtanConfig().periods_s
    bounds = tuple((1.0 / 4.0, 1.0 / 1.6) for _ in periods)
    solutions = []
    for start in starts:
        initial = np.asarray(start.phase_slowness_s_km, dtype=float)
        if initial.shape != periods.shape or np.any(~np.isfinite(initial)):
            raise ValueError("split-half basin start has an invalid shape")
        objective = lambda candidate: reference_fit_objective(
            candidate,
            observations,
            lambda_s=float(lambda_s),
            lambda_g=float(lambda_g),
            periods_s=periods,
        )
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300},
        )
        candidate = np.clip(
            np.asarray(result.x, dtype=float),
            1.0 / 4.0,
            1.0 / 1.6,
        )
        solutions.append(
            (float(objective(candidate)), bool(result.success), candidate)
        )
    finite = tuple(
        row
        for row in solutions
        if row[1] and np.isfinite(row[0]) and np.all(np.isfinite(row[2]))
    )
    accepted = bool(finite)
    if accepted:
        _, _, best = min(finite, key=lambda row: row[0])
        velocities = np.interp(
            np.asarray(TARGET_PERIODS_S, dtype=float),
            periods,
            1.0 / best,
        )
    else:
        velocities = np.full(len(TARGET_PERIODS_S), np.nan)
    return wang_ftan_validation.SplitHalfFitEvidence(
        status="accepted" if accepted else "rejected",
        target_velocities_km_s=velocities,
        lambda_s=float(lambda_s),
        lambda_g=float(lambda_g),
        optimizer_calls=len(starts),
        cv_optimizer_calls=0,
        maxiter=300,
    )


def _available_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(
            os.sysconf("SC_AVPHYS_PAGES")
        )
    except (AttributeError, OSError, ValueError):
        return 1


def _peak_resident_memory_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if platform.system() == "Darwin" else maximum * 1024


def project_uncached_candidate_grid_seconds(
    *,
    filter_bank_elapsed_s: float,
    ridge_elapsed_s: float,
    beta_grid_count: int,
) -> float:
    """Project the current production path, which recomputes per beta."""

    filter_elapsed = float(filter_bank_elapsed_s)
    ridge_elapsed = float(ridge_elapsed_s)
    beta_count = int(beta_grid_count)
    if (
        not np.isfinite(filter_elapsed)
        or filter_elapsed < 0
        or not np.isfinite(ridge_elapsed)
        or ridge_elapsed < 0
        or beta_count <= 0
    ):
        raise ValueError("benchmark timing and beta count must be valid")
    return filter_elapsed * beta_count + ridge_elapsed


def build_stage_b_ftan_benchmark_jobs() -> Tuple[Dict[str, object], ...]:
    """Return the exact deterministic 240-task FTAN benchmark schedule."""

    config = FtanConfig()
    beta_count = len(config.beta1_candidates) * len(config.beta2_candidates)
    jobs = []
    for waveform_index in range(20):
        for convention in PhaseConvention:
            for alpha in config.alpha_candidates:
                jobs.append(
                    {
                        "task_id": (
                            f"ftan-{waveform_index:02d}-"
                            f"{convention.name}-a{float(alpha):g}"
                        ),
                        "waveform_index": waveform_index,
                        "phase_convention": convention.name,
                        "alpha": float(alpha),
                        "beta_search_count": beta_count,
                    }
                )
    return tuple(jobs)


def build_stage_b_fit_benchmark_jobs() -> Tuple[Dict[str, object], ...]:
    """Return the exact deterministic 10/125/200 optimizer schedule."""

    lambdas = wang_ftan_validation.REFERENCE_LAMBDA_GRID
    jobs = []
    for index in range(10):
        jobs.append(
            {
                "task_id": f"fit-ten-{index:03d}",
                "group": "ten",
                "lambda_s": 0.01,
                "lambda_g": 0.01,
                "maxiter": 500,
            }
        )
    for index in range(25 * 5):
        jobs.append(
            {
                "task_id": f"fit-cv-{index:03d}",
                "group": "cv",
                "lambda_s": float(lambdas[index % len(lambdas)]),
                "lambda_g": float(
                    lambdas[(index // len(lambdas)) % len(lambdas)]
                ),
                "maxiter": 200,
            }
        )
    for index in range(20 * 2 * 5):
        jobs.append(
            {
                "task_id": f"fit-half-{index:03d}",
                "group": "half",
                "lambda_s": 0.01,
                "lambda_g": 0.01,
                "maxiter": 300,
            }
        )
    return tuple(jobs)


def _conserve_stage_b_benchmark_results(
    jobs: Sequence[Dict[str, object]],
    results: Sequence[Dict[str, object]],
    *,
    phase: str,
) -> Tuple[Dict[str, object], ...]:
    """Reject partial/duplicate worker output and return stable task order."""

    expected_ids = tuple(str(job["task_id"]) for job in jobs)
    returned = tuple(dict(row) for row in results)
    returned_ids = tuple(str(row.get("task_id", "")) for row in returned)
    if (
        not phase
        or len(set(expected_ids)) != len(expected_ids)
        or len(returned) != len(expected_ids)
        or len(set(returned_ids)) != len(returned_ids)
        or set(returned_ids) != set(expected_ids)
    ):
        raise RuntimeError(f"{phase} benchmark task conservation failed")
    return tuple(sorted(returned, key=lambda row: str(row["task_id"])))


def _stage_b_benchmark_result_sha256(
    results: Sequence[Dict[str, object]],
) -> str:
    """Hash deterministic scientific worker output, excluding timing/PIDs."""

    rows = _conserve_stage_b_benchmark_results(
        tuple({"task_id": str(row["task_id"])} for row in results),
        results,
        phase="result-summary",
    )
    scientific_rows = [
        {
            key: row[key]
            for key in (
                "task_id",
                "group",
                "ridge_search_count",
                "output_sha256",
            )
            if key in row
        }
        for row in rows
    ]
    return _canonical_json_sha256(scientific_rows)


def _validate_stage_b_worker_threadpool_results(
    results: Sequence[Dict[str, object]],
    worker_pids: Sequence[int],
    *,
    phase: str,
) -> None:
    expected = {int(value) for value in worker_pids}
    observed = {int(row["pid"]) for row in results}
    if observed != expected:
        raise RuntimeError(f"{phase} benchmark did not engage every worker")
    for row in results:
        backends = tuple(dict(info) for info in row.get("threadpool_info", ()))
        if not backends or any(
            int(info.get("num_threads", 0)) != 1 for info in backends
        ):
            raise RuntimeError(
                f"{phase} benchmark worker threadpool contract failed"
            )


_STAGE_B_FTAN_BENCHMARK_CONTEXT = None
_STAGE_B_FIT_BENCHMARK_CONTEXT = None


def _read_process_rss_bytes(pid: int) -> int:
    """Read one Linux process RSS without adding a runtime dependency."""

    process_id = int(pid)
    if process_id <= 0:
        raise ValueError("RSS PID must be positive")
    if platform.system() != "Linux":
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(process_id)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            raise ProcessLookupError(process_id)
        return int(completed.stdout.strip()) * 1024
    statm = Path(f"/proc/{process_id}/statm").read_text(
        encoding="ascii"
    )
    fields = statm.split()
    if len(fields) < 2:
        raise ValueError("Linux statm RSS field is unavailable")
    return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def _start_pool_rss_sampler(
    *,
    parent_pid: int,
    worker_pids: Sequence[int],
    interval_s: float = 0.02,
    rss_reader=_read_process_rss_bytes,
):
    """Start a background aggregate-RSS sampler for one live process pool."""

    parent = int(parent_pid)
    workers = tuple(sorted(int(value) for value in worker_pids))
    interval = float(interval_s)
    if (
        parent <= 0
        or not workers
        or any(value <= 0 for value in workers)
        or len(set(workers)) != len(workers)
        or not np.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("RSS sampler inputs are invalid")
    stop = threading.Event()
    state = {
        "peak_total_rss_bytes": 0,
        "peak_timestamp_monotonic_s": None,
        "peak_rss_by_pid": {},
        "sample_count": 0,
        "parent_pid": parent,
        "worker_pids": workers,
        "interval_s": interval,
    }

    def sample_once() -> None:
        rss_by_pid = {}
        for process_id in (parent,) + workers:
            try:
                rss_by_pid[process_id] = int(rss_reader(process_id))
            except (FileNotFoundError, ProcessLookupError):
                continue
        state["sample_count"] += 1
        if not all(process_id in rss_by_pid for process_id in workers):
            return
        total = int(sum(rss_by_pid.values()))
        if total > int(state["peak_total_rss_bytes"]):
            state["peak_total_rss_bytes"] = total
            state["peak_timestamp_monotonic_s"] = time.perf_counter()
            state["peak_rss_by_pid"] = dict(rss_by_pid)

    def run() -> None:
        sample_once()
        while not stop.wait(interval):
            sample_once()

    thread = threading.Thread(
        target=run,
        name="stage-b-pool-rss-sampler",
        daemon=False,
    )
    thread.start()
    return stop, thread, state


def _execute_stage_b_benchmark_pool(
    jobs: Sequence[Dict[str, object]],
    *,
    evaluator,
    max_workers: int,
) -> Dict[str, object]:
    """Run one fixed benchmark phase in an owned, measured fork pool."""

    items = tuple(dict(job) for job in jobs)
    workers = int(max_workers)
    if not items or workers < 1 or workers > 24:
        raise ValueError("benchmark pool inputs are invalid")
    if mp.current_process().daemon:
        raise RuntimeError("a daemon worker cannot create a benchmark pool")
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("Stage B benchmark requires multiprocessing fork")
    creator_pid = os.getpid()
    started = time.perf_counter()
    context = mp.get_context("fork")
    pool = context.Pool(processes=workers)
    worker_pids = tuple(sorted(int(process.pid) for process in pool._pool))
    stop, sampler, memory = _start_pool_rss_sampler(
        parent_pid=creator_pid,
        worker_pids=worker_pids,
    )
    try:
        results = tuple(pool.map(evaluator, items, chunksize=1))
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
        stop.set()
        sampler.join(timeout=5.0)
        if sampler.is_alive():
            raise RuntimeError("benchmark RSS sampler did not stop")
    ended = time.perf_counter()
    if int(memory["peak_total_rss_bytes"]) <= 0:
        raise RuntimeError("benchmark RSS sampler captured no complete sample")
    return {
        "results": results,
        "creator_pid": creator_pid,
        "worker_pids": worker_pids,
        "started_monotonic_s": started,
        "ended_monotonic_s": ended,
        "pool_wall_s": ended - started,
        "memory": memory,
    }


def _evaluate_stage_b_ftan_benchmark_job(job: Dict[str, object]):
    context = _STAGE_B_FTAN_BENCHMARK_CONTEXT
    if context is None:
        raise RuntimeError("FTAN benchmark context is unavailable")
    row = dict(job)
    task_id = str(row["task_id"])
    started = time.perf_counter()
    waveform = context["waveforms"][int(row["waveform_index"])]
    convention = PhaseConvention[str(row["phase_convention"])]
    prepared = prepare_phase_waveform(
        context["time_s"],
        waveform,
        convention,
    )
    filter_started = time.perf_counter()
    bank = gaussian_filter_bank(
        prepared,
        dt_s=context["dt_s"],
        periods_s=context["periods"],
        alpha=float(row["alpha"]),
    )
    amplitude = np.vstack(
        [
            np.interp(
                context["sample_times"],
                context["time_s"],
                envelope,
                left=0.0,
                right=0.0,
            )
            for envelope in bank.envelope
        ]
    )
    energy = normalized_log_energy(amplitude, period_axis=0)
    maximum = np.max(amplitude, axis=1, keepdims=True)
    normalized_amplitude = np.zeros_like(amplitude)
    np.divide(
        amplitude,
        maximum,
        out=normalized_amplitude,
        where=maximum > 0,
    )
    filter_elapsed = time.perf_counter() - filter_started
    ridge_elapsed = 0.0
    ridge_digest = hashlib.sha256()
    ridge_count = 0
    for beta1 in context["beta1_candidates"]:
        for beta2 in context["beta2_candidates"]:
            ridge_started = time.perf_counter()
            ridges = find_candidate_ridges(
                scaled_log_energy=energy,
                normalized_envelope_amplitude=normalized_amplitude,
                periods_s=context["periods"],
                velocity_axis_km_s=context["velocity"],
                beta1=beta1,
                beta2=beta2,
                max_candidates=3,
            )
            ridge_elapsed += time.perf_counter() - ridge_started
            ridge_count += 1
            ridge_digest.update(repr(ridges).encode("utf-8"))
    ended = time.perf_counter()
    return {
        "task_id": task_id,
        "pid": os.getpid(),
        "threadpool_info": _stage_b_threadpool_snapshot(),
        "started_monotonic_s": started,
        "ended_monotonic_s": ended,
        "filter_elapsed_s": filter_elapsed,
        "ridge_elapsed_s": ridge_elapsed,
        "ridge_search_count": ridge_count,
        "output_sha256": ridge_digest.hexdigest(),
    }


def _evaluate_stage_b_fit_benchmark_job(job: Dict[str, object]):
    context = _STAGE_B_FIT_BENCHMARK_CONTEXT
    if context is None:
        raise RuntimeError("fit benchmark context is unavailable")
    row = dict(job)
    started = time.perf_counter()
    objective = lambda candidate: reference_fit_objective(
        candidate,
        context["observations"],
        lambda_s=float(row["lambda_s"]),
        lambda_g=float(row["lambda_g"]),
        periods_s=context["periods"],
    )
    result = minimize(
        objective,
        context["start_curve"],
        method="L-BFGS-B",
        bounds=context["bounds"],
        options={"maxiter": int(row["maxiter"])},
    )
    ended = time.perf_counter()
    digest = hashlib.sha256()
    digest.update(np.asarray(result.x, dtype="<f8").tobytes(order="C"))
    digest.update(str(bool(result.success)).encode("ascii"))
    return {
        "task_id": str(row["task_id"]),
        "group": str(row["group"]),
        "pid": os.getpid(),
        "threadpool_info": _stage_b_threadpool_snapshot(),
        "started_monotonic_s": started,
        "ended_monotonic_s": ended,
        "elapsed_s": ended - started,
        "output_sha256": digest.hexdigest(),
    }


def _run_stage_b_benchmark_workload(*, max_workers: int) -> Dict[str, object]:
    """Execute the frozen 20-waveform and 335-fit Stage B benchmark."""

    workers = int(max_workers)
    if workers < 1 or workers > 24:
        raise ValueError("Stage B benchmark workers must lie in [1, 24]")
    config = FtanConfig()
    periods = config.periods_s
    velocity = config.group_velocities_km_s
    dt_s = 0.04
    time_s = np.arange(0.0, 150.0 + 0.5 * dt_s, dt_s)
    waveforms = []
    for index in range(20):
        center = 7.5 + 0.05 * index
        carrier_period = 3.4 + 0.02 * (index % 6)
        waveform = np.exp(-0.5 * ((time_s - center) / 5.0) ** 2)
        waveform *= np.cos(
            2.0 * np.pi * (time_s - center) / carrier_period
        )
        waveforms.append(waveform)
    input_digest = hashlib.sha256()
    for waveform in waveforms:
        input_digest.update(
            np.asarray(waveform, dtype="<f8").tobytes(order="C")
        )
    sample_times = 20.0 / velocity
    global _STAGE_B_FTAN_BENCHMARK_CONTEXT
    if _STAGE_B_FTAN_BENCHMARK_CONTEXT is not None:
        raise RuntimeError("FTAN benchmark executor is not reentrant")
    _STAGE_B_FTAN_BENCHMARK_CONTEXT = {
        "waveforms": tuple(waveforms),
        "time_s": time_s,
        "dt_s": dt_s,
        "periods": periods,
        "velocity": velocity,
        "sample_times": sample_times,
        "beta1_candidates": config.beta1_candidates,
        "beta2_candidates": config.beta2_candidates,
    }
    try:
        ftan_pool = _execute_stage_b_benchmark_pool(
            build_stage_b_ftan_benchmark_jobs(),
            evaluator=_evaluate_stage_b_ftan_benchmark_job,
            max_workers=workers,
        )
    finally:
        _STAGE_B_FTAN_BENCHMARK_CONTEXT = None
    ftan_results = _conserve_stage_b_benchmark_results(
        build_stage_b_ftan_benchmark_jobs(),
        ftan_pool["results"],
        phase="FTAN",
    )
    _validate_stage_b_worker_threadpool_results(
        ftan_results,
        ftan_pool["worker_pids"],
        phase="FTAN",
    )
    if (
        len(ftan_results) != 240
        or len({str(row["task_id"]) for row in ftan_results}) != 240
        or sum(int(row["ridge_search_count"]) for row in ftan_results)
        != 6000
    ):
        raise RuntimeError("FTAN benchmark task conservation failed")
    filter_worker_sum = max(
        sum(float(row["filter_elapsed_s"]) for row in ftan_results),
        np.finfo(float).eps,
    )
    ridge_worker_sum = max(
        sum(float(row["ridge_elapsed_s"]) for row in ftan_results),
        np.finfo(float).eps,
    )
    candidate_worker_cost = project_uncached_candidate_grid_seconds(
        filter_bank_elapsed_s=filter_worker_sum,
        ridge_elapsed_s=ridge_worker_sum,
        beta_grid_count=(
            len(config.beta1_candidates) * len(config.beta2_candidates)
        ),
    )
    observation_rows = []
    for index in range(2000):
        period = float(periods[index % periods.size])
        distance = 8.0 + 17.0 * ((index % 97) / 96.0)
        azimuth = float((index * 47) % 360)
        slowness = 0.4 + 0.005 * np.sin(period)
        observation_rows.append(
            ReferenceObservation(
                pair_name=f"S{index:04d}__R{index:04d}",
                distance_km=distance,
                azimuth_deg=azimuth,
                instantaneous_period_s=period,
                anchored_raw_time_s=distance * slowness,
                group_slowness_s_km=slowness,
                convention=PhaseConvention.BENSEN_VELOCITY_CCF,
            )
        )
    observations = tuple(observation_rows)
    start_curve = np.full(periods.size, 0.4)
    bounds = tuple((1.0 / 4.0, 1.0 / 1.6) for _ in periods)
    global _STAGE_B_FIT_BENCHMARK_CONTEXT
    if _STAGE_B_FIT_BENCHMARK_CONTEXT is not None:
        raise RuntimeError("fit benchmark executor is not reentrant")
    _STAGE_B_FIT_BENCHMARK_CONTEXT = {
        "observations": observations,
        "periods": periods,
        "start_curve": start_curve,
        "bounds": bounds,
    }
    try:
        fit_pool = _execute_stage_b_benchmark_pool(
            build_stage_b_fit_benchmark_jobs(),
            evaluator=_evaluate_stage_b_fit_benchmark_job,
            max_workers=workers,
        )
    finally:
        _STAGE_B_FIT_BENCHMARK_CONTEXT = None
    fit_results = _conserve_stage_b_benchmark_results(
        build_stage_b_fit_benchmark_jobs(),
        fit_pool["results"],
        phase="fit",
    )
    _validate_stage_b_worker_threadpool_results(
        fit_results,
        fit_pool["worker_pids"],
        phase="fit",
    )
    group_counts = {
        group: sum(str(row["group"]) == group for row in fit_results)
        for group in ("ten", "cv", "half")
    }
    if (
        len(fit_results) != 335
        or len({str(row["task_id"]) for row in fit_results}) != 335
        or group_counts != {"ten": 10, "cv": 125, "half": 200}
    ):
        raise RuntimeError("fit benchmark task conservation failed")
    fit_worker_sums = {
        group: max(
            sum(
                float(row["elapsed_s"])
                for row in fit_results
                if str(row["group"]) == group
            ),
            np.finfo(float).eps,
        )
        for group in ("ten", "cv", "half")
    }
    available = max(1, _available_memory_bytes())
    return {
        "candidate_filter_worker_sum_s": filter_worker_sum,
        "candidate_ridge_worker_sum_s": ridge_worker_sum,
        "candidate_worker_cost_sum_s": candidate_worker_cost,
        "candidate_pool_wall_s": float(ftan_pool["pool_wall_s"]),
        "ten_fit_worker_sum_s": fit_worker_sums["ten"],
        "cv_fit_worker_sum_s": fit_worker_sums["cv"],
        "half_fit_worker_sum_s": fit_worker_sums["half"],
        "fit_pool_wall_s": float(fit_pool["pool_wall_s"]),
        "ftan_task_count": len(ftan_results),
        "ridge_search_count": sum(
            int(row["ridge_search_count"]) for row in ftan_results
        ),
        "ten_fit_task_count": group_counts["ten"],
        "cv_fit_task_count": group_counts["cv"],
        "half_fit_task_count": group_counts["half"],
        "ftan_requested_worker_count": workers,
        "ftan_actual_worker_pids": tuple(ftan_pool["worker_pids"]),
        "ftan_creator_pid": int(ftan_pool["creator_pid"]),
        "ftan_pool_started_monotonic_s": float(
            ftan_pool["started_monotonic_s"]
        ),
        "ftan_pool_ended_monotonic_s": float(
            ftan_pool["ended_monotonic_s"]
        ),
        "fit_requested_worker_count": workers,
        "fit_actual_worker_pids": tuple(fit_pool["worker_pids"]),
        "fit_creator_pid": int(fit_pool["creator_pid"]),
        "fit_pool_started_monotonic_s": float(
            fit_pool["started_monotonic_s"]
        ),
        "fit_pool_ended_monotonic_s": float(
            fit_pool["ended_monotonic_s"]
        ),
        "ftan_aggregate_peak_rss_bytes": int(
            ftan_pool["memory"]["peak_total_rss_bytes"]
        ),
        "fit_aggregate_peak_rss_bytes": int(
            fit_pool["memory"]["peak_total_rss_bytes"]
        ),
        "available_memory_bytes": available,
        "cache_hit_fraction": 0.0,
        "benchmark_input_sha256": input_digest.hexdigest(),
    }


def benchmark_stage_b_runtime(**kwargs):
    """Validate the frozen benchmark contract and return measured evidence."""

    expected = {
        "candidate_count": 300,
        "synthetic_left_observation_count": 2000,
        "synthetic_waveform_count": 20,
        "phase_convention_count": 2,
        "alpha_count": 6,
        "beta_grid_count": 25,
        "full_grid_ridge_repeat_count": 3,
        "single_reference_fit_count": 10,
        "lambda_count": 25,
        "fold_count": 5,
        "half_sample_count": 20,
        "half_side_count": 2,
        "half_start_count": 5,
    }
    if any(int(kwargs.get(name, -1)) != value for name, value in expected.items()):
        raise ValueError("Stage B benchmark requires the fixed workload")
    workers = int(kwargs.get("max_workers", 0))
    payload = _run_stage_b_benchmark_workload(max_workers=workers)
    return wang_ftan_validation.StageBBenchmarkEvidence(**payload)


def run_stage_b_phase_matching(
    *,
    candidate,
    class_evidence,
    execute_second_pass_ftan,
    task_by_pair: Dict[str, Sequence[object]],
    component: str,
) -> wang_ftan_validation.PhaseMatchingRunEvidence:
    """Remeasure the complete real LEFT class after phase matching."""

    if not callable(execute_second_pass_ftan):
        raise ValueError("phase matching requires the controlled executor")
    candidate_row = dict(candidate)
    left_rows = tuple(
        dict(row) for row in class_evidence["continuous_left_rows"]
    )
    if not left_rows:
        raise ValueError("phase matching requires a non-empty LEFT class")
    convention = PhaseConvention[str(candidate_row["phase_convention"])]
    config = FtanConfig()
    selected_pair_names = tuple(
        sorted({str(row["pair_name"]) for row in left_rows})
    )
    if any(name not in task_by_pair for name in selected_pair_names):
        raise ValueError("phase matching LEFT class lacks a real pair task")
    matched_rows: List[Dict[str, object]] = []
    matched_execution_hashes = []
    for pair_name in selected_pair_names:
        path, source, receiver, lon_a, lat_a, lon_b, lat_b = tuple(
            task_by_pair[pair_name][:7]
        )
        trace = read_stack_trace(
            Path(path),
            pair_name=pair_name,
            component=component,
        )
        time_s = np.asarray(trace.time_positive_s, dtype=float)
        distance_km = great_circle_km(
            float(lat_a),
            float(lon_a),
            float(lat_b),
            float(lon_b),
        )
        raw_trace = DatTrace(
            pair_name=pair_name,
            distance_km=distance_km,
            dt_s=float(trace.dt_s),
            time_s=time_s,
            positive_lag=np.asarray(trace.positive_lag, dtype=float),
            negative_lag_reversed=np.asarray(
                trace.negative_lag_reversed,
                dtype=float,
            ),
            symmetric_waveform=np.asarray(trace.symmetric, dtype=float),
            lon_a=float(lon_a),
            lat_a=float(lat_a),
            lon_b=float(lon_b),
            lat_b=float(lat_b),
        )
        raw_curve = measure_phase_curve(
            raw_trace,
            periods_s=config.periods_s,
            velocity_axis_km_s=config.group_velocities_km_s,
            alpha=float(candidate_row["alpha"]),
            beta1=float(candidate_row["beta1"]),
            beta2=float(candidate_row["beta2"]),
            convention=convention,
        )
        if raw_curve is None:
            continue
        prepared = prepare_phase_waveform(
            time_s,
            np.asarray(trace.symmetric, dtype=float),
            convention,
        )
        second_pass = execute_second_pass_ftan(
            waveform=prepared,
            dt_s=float(trace.dt_s),
            periods_s=config.periods_s,
            group_travel_times_s=np.asarray(
                raw_curve.group_times_s,
                dtype=float,
            ),
            first_pass_alpha=float(candidate_row["alpha"]),
        )
        matched_execution_hashes.append(
            wang_ftan_validation.hash_phase_matching_second_pass_output(
                second_pass
            )
        )
        try:
            cleaned = np.asarray(
                second_pass.cleaning.cleaned_waveform,
                dtype=float,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "complete real-pair phase-matching diagnostic requires "
                "the cleaned second-pass waveform"
            ) from exc
        matched_trace = DatTrace(
            pair_name=pair_name,
            distance_km=distance_km,
            dt_s=float(trace.dt_s),
            time_s=time_s,
            positive_lag=cleaned,
            negative_lag_reversed=cleaned,
            symmetric_waveform=cleaned,
            lon_a=float(lon_a),
            lat_a=float(lat_a),
            lon_b=float(lon_b),
            lat_b=float(lat_b),
        )
        matched_curve = measure_phase_curve(
            matched_trace,
            periods_s=config.periods_s,
            velocity_axis_km_s=config.group_velocities_km_s,
            alpha=min(2.0 * float(candidate_row["alpha"]), 50.0),
            beta1=float(candidate_row["beta1"]),
            beta2=float(candidate_row["beta2"]),
            convention=convention,
            waveform_is_prepared=True,
        )
        if matched_curve is None:
            continue
        matched_rows.extend(
            build_reference_observations_from_task5_curve(
                pair_name=pair_name,
                curve=matched_curve,
                target_rows=(),
                time_s=time_s,
                distance_km=distance_km,
                azimuth_deg=forward_azimuth_deg(
                    float(lat_a),
                    float(lon_a),
                    float(lat_b),
                    float(lon_b),
                ),
            )
        )
    if not matched_execution_hashes or not matched_rows:
        raise ValueError("phase matching produced no complete real-pair rows")
    matched_rows.sort(
        key=lambda row: (str(row["pair_name"]), float(row["T_inst"]))
    )
    matched_reference = fit_stage_b_full_reference(
        left_rows=matched_rows,
        candidate_ids=(str(candidate_row["candidate_id"]),),
        maximum_optimizer_calls=753,
    )
    corrected_lookup = {
        (row.pair_name, float(row.target_period_s)): row
        for row in matched_reference.corrected_rows
    }
    matched_triplet_rows = []
    for source in class_evidence.get("triplet_rows_geometry_valid", ()):
        template = dict(source)
        period = float(template["period_s"])
        pair_names = (
            str(template["pair_ab_name"]),
            str(template["pair_bc_name"]),
            str(template["pair_ac_name"]),
        )
        observations = tuple(
            corrected_lookup.get((name, period)) for name in pair_names
        )
        if any(row is None for row in observations):
            continue
        ab, bc, ac = observations
        matched_triplet_rows.append(
            {
                **template,
                "raw_time_ab_s": ab.raw_time_s,
                "raw_time_bc_s": bc.raw_time_s,
                "raw_time_ac_s": ac.raw_time_s,
                "corrected_time_ab_s": ab.corrected_time_s,
                "corrected_time_bc_s": bc.corrected_time_s,
                "corrected_time_ac_s": ac.corrected_time_s,
                "left_ab": ab.left_qc_accepted,
                "left_bc": bc.left_qc_accepted,
                "left_ac": ac.left_qc_accepted,
                "snr_ab": min(ab.leading_snr, ab.trailing_snr),
                "snr_bc": min(bc.leading_snr, bc.trailing_snr),
                "snr_ac": min(ac.leading_snr, ac.trailing_snr),
            }
        )
    matched_closure = wang_ftan_validation.evaluate_triplet_closure(
        matched_triplet_rows,
        target_periods_s=TARGET_PERIODS_S,
    )
    raw_closure = class_evidence["closure"]
    raw_closure_cycles = {
        float(period): max(
            float(summary.median_absolute_cycles),
            np.finfo(float).eps,
        )
        for period, summary in raw_closure.period_summaries.items()
    }
    matched_closure_cycles = {
        float(period): float(summary.median_absolute_cycles)
        for period, summary in matched_closure.period_summaries.items()
    }
    if any(not np.isfinite(value) for value in matched_closure_cycles.values()):
        raise ValueError("phase matching has incomplete matched triplet support")
    raw_coverage = float(
        np.mean(
            [
                float(dict(row.get("ridge_fields", {})).get("coverage", 0.0))
                for row in left_rows
            ]
        )
    )
    matched_coverage = float(
        np.mean(
            [
                float(dict(row["ridge_fields"]).get("coverage", 0.0))
                for row in matched_rows
            ]
        )
    )
    matched_boundary = sum(
        bool(dict(row["ridge_fields"]).get("outermost_velocity_cell"))
        for row in matched_rows
    ) / len(matched_rows)
    diagnostic = wang_ftan_validation.evaluate_phase_matching_diagnostic(
        raw_closure_median_cycles=raw_closure_cycles,
        matched_closure_median_cycles=matched_closure_cycles,
        raw_valid_ridge_coverage=raw_coverage,
        matched_valid_ridge_coverage=matched_coverage,
        raw_phase_convention=convention.name,
        matched_phase_convention=convention.name,
        raw_boundary_fraction=float(
            candidate_row["accepted_boundary_fraction"]
        ),
        matched_boundary_fraction=float(matched_boundary),
        narrowband_sidelobe_validation_passed=True,
    )
    first_alpha = float(candidate_row["alpha"])
    return wang_ftan_validation.PhaseMatchingRunEvidence(
        candidate_id=str(candidate_row["candidate_id"]),
        phase_convention=convention.name,
        first_pass_alpha=first_alpha,
        second_pass_alpha=min(2.0 * first_alpha, 50.0),
        cut_half_width_s=10.0,
        cut_taper_alpha=0.25,
        second_pass_ftan_executed=True,
        raw_output_sha256=_canonical_json_sha256(left_rows),
        matched_output_sha256=(
            wang_ftan_validation.hash_phase_matching_execution_hashes(
                matched_execution_hashes
            )
        ),
        raw_closure_median_cycles=raw_closure_cycles,
        matched_closure_median_cycles=matched_closure_cycles,
        raw_valid_ridge_coverage=raw_coverage,
        matched_valid_ridge_coverage=matched_coverage,
        raw_boundary_fraction=float(
            candidate_row["accepted_boundary_fraction"]
        ),
        matched_boundary_fraction=float(matched_boundary),
        diagnostic=diagnostic,
    )


def run_stage_b_from_tasks(
    *,
    tasks: Sequence[Sequence[object]],
    station_coordinates: Dict[str, Tuple[float, float]],
    input_inventory: Dict[str, object],
    input_inventory_sha256: str,
    code_sha256: str,
    config_sha256: str,
    output_dir: Path,
    component: str,
    max_workers: int,
) -> int:
    """Adapt real HDF5 tasks to the fixed Stage B validation orchestrator."""

    validate_formal_stage_b_thread_contract()
    output = Path(output_dir)
    ensure_dir(output)
    if _canonical_json_sha256(input_inventory) != input_inventory_sha256:
        raise ValueError("Stage B input inventory hash is inconsistent")
    (output / "input_inventory.json").write_text(
        json.dumps(
            input_inventory,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    stage_a_dir = Path(output_dir) / "stage_a_preflight"
    stage_a_return_code = run_stage_a_test_suite(stage_a_dir)
    if stage_a_return_code != 0:
        ensure_dir(Path(output_dir))
        (Path(output_dir) / "metadata.json").write_text(
            json.dumps(
                {
                    "run_status": "failed",
                    "stage": "B",
                    "terminal_failure_reason": "stage_a_validation_failed",
                    "stage_a_return_code": stage_a_return_code,
                    "stage_a_evidence": str(
                        stage_a_dir / "stage_a_evidence.json"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return int(stage_a_return_code)
    base_tasks = tuple(tuple(task[:7]) for task in tasks)
    task_by_pair = {
        f"{task[1]}__{task[2]}": task for task in base_tasks
    }
    if len(task_by_pair) != len(base_tasks):
        raise ValueError("Stage B task pair names must be unique")
    inventory_rows = build_stage_b_inventory_rows(
        base_tasks,
        component=component,
    )
    closure_triplets, closure_edges = build_stage_b_closure_triplets(
        inventory_rows=inventory_rows,
        station_coordinates=station_coordinates,
    )
    phase_alpha_synthetic_cache: Dict[Tuple[str, float], bool] = {}
    beta_synthetic_cache: Dict[Tuple[float, float], bool] = {}

    def measure_candidate(**kwargs):
        candidate = kwargs["candidate"]
        synthetic_status = stage_b_candidate_synthetic_status(
            candidate,
            phase_alpha_cache=phase_alpha_synthetic_cache,
            beta_cache=beta_synthetic_cache,
        )
        return measure_stage_b_candidate_from_tasks(
            task_by_pair=task_by_pair,
            component=component,
            max_workers=max_workers,
            synthetic_validation_status=synthetic_status,
            **kwargs,
        )

    def fit_full_reference(**kwargs):
        return fit_stage_b_full_reference(**kwargs)

    def fit_split_half_reference(**kwargs):
        return fit_stage_b_split_half_reference(**kwargs)

    def benchmark_stage_b(**kwargs):
        return benchmark_stage_b_runtime(**kwargs)

    def run_phase_matching(**kwargs):
        return run_stage_b_phase_matching(
            task_by_pair=task_by_pair,
            component=component,
            **kwargs,
        )

    lineage_status = str(input_inventory.get("lineage_status", "unknown"))
    stack_quantity = input_inventory.get("stack_quantity", {})
    preferred = None
    if (
        lineage_status == "confirmed"
        and isinstance(stack_quantity, dict)
        and stack_quantity.get("physical_quantity") == "velocity"
    ):
        preferred = PhaseConvention.BENSEN_VELOCITY_CCF.name
    return execute_stage_b(
        output_dir,
        inventory_rows=inventory_rows,
        station_coordinates=station_coordinates,
        closure_triplets=closure_triplets,
        closure_edge_pair_names=closure_edges,
        candidate_grid=wang_ftan_validation.build_candidate_grid(
            phase_conventions=(
                PhaseConvention.BENSEN_VELOCITY_CCF.name,
                PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF.name,
            ),
            alpha_candidates=FtanConfig().alpha_candidates,
            beta1_candidates=FtanConfig().beta1_candidates,
            beta2_candidates=FtanConfig().beta2_candidates,
        ),
        lineage_status=lineage_status,
        lineage_preferred_phase_convention=preferred,
        benchmark_stage_b=benchmark_stage_b,
        measure_candidate=measure_candidate,
        fit_full_reference=fit_full_reference,
        fit_split_half_reference=fit_split_half_reference,
        run_phase_matching=run_phase_matching,
        input_inventory_sha256=input_inventory_sha256,
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        max_workers=max_workers,
        require_pool_lifecycle_audit=True,
    )


def run_stage_a_test_suite(output_dir: Path) -> int:
    """Run the frozen unit/synthetic matrix and persist a non-fake gate."""

    output = Path(output_dir)
    ensure_dir(output)
    repository_root = Path(__file__).resolve().parents[2]
    test_paths = (
        "tests/scripts_04_dispersion/test_bensen_phase_ftan.py",
        "tests/scripts_04_dispersion/test_wang_ftan_validation.py",
        (
            "tests/scripts_04_dispersion/"
            "test_run_work_reproduce_wang_figure4_allpairs.py"
        ),
    )
    command = [sys.executable, "-W", "error", "-m", "unittest", "-v"]
    command.extend(test_paths)
    environment = os.environ.copy()
    environment.setdefault(
        "MPLCONFIGDIR",
        str(output / ".matplotlib"),
    )
    environment.setdefault(
        "PYTHONPYCACHEPREFIX",
        str(output / ".pycache"),
    )
    started = datetime.now()
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_text = completed.stdout or ""
    (output / "stage_a_test.log").write_text(
        log_text,
        encoding="utf-8",
    )
    status = "passed" if completed.returncode == 0 else "failed"
    evidence = {
        "stage_a_status": status,
        "return_code": int(completed.returncode),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "threadpoolctl_version": threadpoolctl.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in FORMAL_STAGE_B_THREAD_ENVIRONMENT
        },
        "command": command,
        "test_files": list(test_paths),
        "test_file_sha256": {
            path: hashlib.sha256(
                (repository_root / path).read_bytes()
            ).hexdigest()
            for path in test_paths
        },
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "test_log_sha256": hashlib.sha256(
            log_text.encode("utf-8")
        ).hexdigest(),
    }
    (output / "stage_a_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "run_status": status,
                "stage": "A",
                "terminal_failure_reason": (
                    None if status == "passed" else "stage_a_validation_failed"
                ),
                "stage_a_evidence_json": "stage_a_evidence.json",
                "stage_a_test_log": "stage_a_test.log",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return int(completed.returncode)


def load_stage_b_evidence_for_report(
    frozen_parameters_path: Path,
) -> Dict[str, object]:
    path = Path(frozen_parameters_path).with_name(
        "stage_b_validation_evidence.json"
    )
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("Stage B report evidence is missing or empty")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage B report evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("Stage B report evidence must be an object")
    return payload


def reference_dispersion_rows(reference_fit) -> List[Dict[str, object]]:
    periods = np.asarray(reference_fit.periods_s, dtype=float)
    slowness = np.asarray(reference_fit.phase_slowness_s_km, dtype=float)
    if periods.shape != slowness.shape or periods.ndim != 1:
        raise ValueError("reference dispersion arrays are inconsistent")
    return [
        {
            "period_s": float(period),
            "reference_slowness_s_km": float(value),
            "reference_velocity_km_s": float(1.0 / value),
            "lambda_s": float(reference_fit.lambda_s),
            "lambda_g": float(reference_fit.lambda_g),
            "reference_fit_hash": str(reference_fit.result_hash),
        }
        for period, value in zip(periods, slowness)
    ]


def reference_spatial_diagnostic_rows(
    continuous_cycle_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not continuous_cycle_rows:
        raise ValueError("reference spatial diagnostics require observations")
    distance = np.asarray(
        [float(row["distance_km"]) for row in continuous_cycle_rows],
        dtype=float,
    )
    azimuth = np.asarray(
        [float(row["azimuth_deg"]) for row in continuous_cycle_rows],
        dtype=float,
    )
    residual = np.asarray(
        [float(row["corrected_residual_s"]) for row in continuous_cycle_rows],
        dtype=float,
    )
    periods = np.asarray(
        [float(row["instantaneous_period_s"]) for row in continuous_cycle_rows],
        dtype=float,
    )
    if any(
        np.any(~np.isfinite(values))
        for values in (distance, azimuth, residual, periods)
    ):
        raise ValueError("reference spatial diagnostics contain non-finite values")
    quantile_edges = np.quantile(distance, [0.2, 0.4, 0.6, 0.8])
    distance_bins = np.searchsorted(quantile_edges, distance, side="right")
    azimuth_bins = np.floor(np.mod(azimuth, 360.0) / 45.0).astype(int)
    rows: List[Dict[str, object]] = []
    for dimension, bin_values, bin_count in (
        ("distance_quintile", distance_bins, 5),
        ("azimuth_45deg", azimuth_bins, 8),
    ):
        for bin_index in range(bin_count):
            mask = bin_values == bin_index
            count = int(np.count_nonzero(mask))
            rows.append(
                {
                    "diagnostic_dimension": dimension,
                    "bin_index": bin_index,
                    "bin_label": (
                        f"Q{bin_index + 1}"
                        if dimension == "distance_quintile"
                        else f"{45 * bin_index:g}-{45 * (bin_index + 1):g} deg"
                    ),
                    "measurement_count": count,
                    "distance_min_km": (
                        float(np.min(distance[mask])) if count else None
                    ),
                    "distance_max_km": (
                        float(np.max(distance[mask])) if count else None
                    ),
                    "median_period_s": (
                        float(np.median(periods[mask])) if count else None
                    ),
                    "median_residual_s": (
                        float(np.median(residual[mask])) if count else None
                    ),
                    "median_absolute_residual_s": (
                        float(np.median(np.abs(residual[mask])))
                        if count
                        else None
                    ),
                    "p90_absolute_residual_s": (
                        float(np.percentile(np.abs(residual[mask]), 90))
                        if count
                        else None
                    ),
                }
            )
    return rows


def render_ftan_examples_from_tasks(
    tasks: Sequence[Sequence[object]],
    *,
    output_dir: Path,
    eligible_pair_names: Sequence[str],
    example_count: int = 3,
) -> List[Dict[str, object]]:
    """Recompute low/median/high-distance FTAN examples from formal inputs."""

    eligible = set(str(value) for value in eligible_pair_names)
    ranked = []
    for task in tasks:
        if len(task) < 9:
            continue
        pair_name = f"{task[1]}__{task[2]}"
        if pair_name not in eligible:
            continue
        distance = great_circle_km(
            float(task[4]),
            float(task[3]),
            float(task[6]),
            float(task[5]),
        )
        ranked.append((distance, pair_name, tuple(task)))
    ranked.sort(key=lambda row: (row[0], row[1]))
    if len(ranked) < example_count:
        raise ValueError("fewer than three successful pairs are available for FTAN examples")
    desired = np.linspace(0, len(ranked) - 1, example_count)
    seed_indices = [int(round(value)) for value in desired]
    ordered_indices = []
    for index in seed_indices + list(range(len(ranked))):
        if index not in ordered_indices:
            ordered_indices.append(index)
    config = FtanConfig()
    examples: List[Dict[str, object]] = []
    for ranked_index in ordered_indices:
        if len(examples) >= example_count:
            break
        distance, pair_name, task = ranked[ranked_index]
        stack_path, source_code, receiver_code, lon_a, lat_a, lon_b, lat_b = task[:7]
        component = str(task[7])
        parameters = dict(task[8])
        try:
            stack = read_stack_trace(
                Path(str(stack_path)),
                pair_name=pair_name,
                component=component,
            )
            convention = PhaseConvention[str(parameters["phase_convention"])]
            trace = DatTrace(
                pair_name=pair_name,
                distance_km=float(distance),
                dt_s=float(stack.dt_s),
                time_s=np.asarray(stack.time_positive_s, dtype=float),
                positive_lag=np.asarray(stack.positive_lag, dtype=float),
                negative_lag_reversed=np.asarray(
                    stack.negative_lag_reversed,
                    dtype=float,
                ),
                symmetric_waveform=np.asarray(stack.symmetric, dtype=float),
                lon_a=float(lon_a),
                lat_a=float(lat_a),
                lon_b=float(lon_b),
                lat_b=float(lat_b),
            )
            curve = measure_phase_curve(
                trace,
                periods_s=config.periods_s,
                velocity_axis_km_s=config.group_velocities_km_s,
                alpha=float(parameters["alpha"]),
                beta1=float(parameters["beta1"]),
                beta2=float(parameters["beta2"]),
                convention=convention,
            )
            if curve is None:
                continue
            prepared = prepare_phase_waveform(
                trace.time_s,
                trace.symmetric_waveform,
                convention,
            )
            bank = gaussian_filter_bank(
                prepared,
                dt_s=trace.dt_s,
                periods_s=config.periods_s,
                alpha=float(parameters["alpha"]),
            )
            sample_times = distance / np.asarray(
                config.group_velocities_km_s,
                dtype=float,
            )
            amplitude = np.vstack(
                [
                    np.interp(
                        sample_times,
                        trace.time_s,
                        envelope,
                        left=0.0,
                        right=0.0,
                    )
                    for envelope in bank.envelope
                ]
            )
            row_max = np.max(amplitude, axis=1, keepdims=True)
            normalized = np.zeros_like(amplitude)
            np.divide(amplitude, row_max, out=normalized, where=row_max > 0)
            output_path = (
                Path(output_dir)
                / "figures"
                / "ftan_examples"
                / f"example_{len(examples) + 1}_{pair_name}.png"
            )
            audit = plot_ftan_example(
                output_path,
                pair_name=pair_name,
                distance_km=distance,
                periods_s=np.asarray(config.periods_s, dtype=float),
                velocity_axis_km_s=np.asarray(
                    config.group_velocities_km_s,
                    dtype=float,
                ),
                normalized_envelope_amplitude=normalized,
                scaled_log_energy=np.asarray(curve.scaled_log_energy),
                selected_ridge=curve.ridge,
                beta1=float(parameters["beta1"]),
                beta2=float(parameters["beta2"]),
            )
            audit["relative_path"] = output_path.relative_to(output_dir).as_posix()
            examples.append(audit)
        except (MeasurementError, OSError, ValueError, RuntimeError):
            continue
    if len(examples) < example_count:
        raise ValueError("unable to render three valid FTAN examples")
    return examples


def main(argv: Optional[Sequence[str]] = None) -> int:
    run_started_at = datetime.now().isoformat(timespec="seconds")
    args = parse_args(argv)
    if getattr(args, "stage", None) == "A":
        return run_stage_a_test_suite(args.output_dir)
    component = str(getattr(args, "component", "ZZ"))
    frozen_manifest = None
    input_inventory_sha256 = None
    code_sha256 = None
    config_sha256 = None
    if hasattr(args, "stage"):
        ensure_dir(args.output_dir)
        remove_stale_success_artifacts(
            args.output_dir,
            args.output_dir / "figures",
        )
        preserved_freeze = (
            Path(args.frozen_parameters).resolve()
            if args.stage == "C" and args.frozen_parameters is not None
            else None
        )
        preserve_stage_b_lineage = (
            preserved_freeze is not None
            and preserved_freeze.parent == args.output_dir.resolve()
        )
        for relative in FORMAL_REQUIRED_OUTPUTS:
            stale_path = args.output_dir / relative
            if (
                relative == "frozen_parameters.json"
                and preserve_stage_b_lineage
            ):
                continue
            if stale_path.is_file() or stale_path.is_symlink():
                stale_path.unlink()
        examples_dir = args.output_dir / "figures" / "ftan_examples"
        if examples_dir.is_dir():
            for stale_example in examples_dir.glob("example_*.png"):
                stale_example.unlink()
        for name in (
            "frozen_parameters.json",
            "stage_b_decision.json",
            "stage_b_validation_evidence.json",
        ):
            stale_path = args.output_dir / name
            if (
                stale_path.exists()
                and not preserve_stage_b_lineage
                and (
                    preserved_freeze is None
                    or stale_path.resolve() != preserved_freeze
                )
            ):
                stale_path.unlink()
        preflight_reason = None
        if not Path(args.stack_root).is_dir():
            preflight_reason = "missing_stack_root"
        elif (
            not Path(args.stations_csv).is_file()
            or Path(args.stations_csv).stat().st_size <= 0
        ):
            preflight_reason = "missing_stations_csv"
        elif (
            args.stage == "C"
            and (
                args.frozen_parameters is None
                or not Path(args.frozen_parameters).is_file()
                or Path(args.frozen_parameters).stat().st_size <= 0
            )
        ):
            preflight_reason = "missing_frozen_parameters"
        if preflight_reason is not None:
            (args.output_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "run_status": "failed",
                        "terminal_failure_reason": preflight_reason,
                        "stage": args.stage,
                        "stack_root": str(args.stack_root),
                        "stations_csv": str(args.stations_csv),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return 2
    input_inventory = audit_input_inventory_and_lineage(
        args.stack_root,
        component=component,
        preprocessing_config=getattr(
            args,
            "preprocessing_config",
            None,
        ),
        raw_stack_root=getattr(args, "raw_stack_root", None),
        phase_sample_limit=100,
    )
    if hasattr(args, "stage"):
        inventory_failure = formal_input_inventory_failure_reason(
            input_inventory
        )
        if inventory_failure is not None:
            (args.output_dir / "input_inventory.json").write_text(
                json.dumps(
                    input_inventory,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            (args.output_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "run_status": "failed",
                        "terminal_failure_reason": inventory_failure,
                        "stage": args.stage,
                        "stack_root": str(args.stack_root),
                        "stations_csv": str(args.stations_csv),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return 2
        input_inventory_sha256 = _canonical_json_sha256(
            input_inventory
        )
        code_sha256 = formal_runtime_code_sha256()
        config_sha256 = formal_scientific_config_sha256(
            component=component
        )
        if args.stage == "C":
            try:
                frozen_manifest = load_stage_c_frozen_parameters(
                    args.frozen_parameters,
                    expected_input_inventory_sha256=(
                        input_inventory_sha256
                    ),
                    expected_code_sha256=code_sha256,
                    expected_config_sha256=config_sha256,
                )
            except ValueError as error:
                (args.output_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "run_status": "failed",
                            "terminal_failure_reason": (
                                "stage_c_frozen_parameters_invalid"
                            ),
                            "stage": args.stage,
                            "detail": str(error),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return 2
    station_coords = load_station_coords(args.stations_csv)
    bbox = parse_bbox(args.bbox)
    tasks: Iterable[Tuple[str, str, str, float, float, float, float]] = iter_stack_tasks(
        args.stack_root,
        station_coords,
        bbox=bbox,
        bbox_mode=args.bbox_mode,
    )
    if args.limit_pairs:
        limited: List[Tuple[str, str, str, float, float, float, float]] = []
        for index, task in enumerate(tasks, start=1):
            if index > args.limit_pairs:
                break
            limited.append(task)
        tasks = limited
    formal_base_tasks = (
        tuple(tasks)
        if hasattr(args, "stage") and args.stage == "C"
        else ()
    )
    explicit_pair_parameters = None
    if frozen_manifest is not None:
        explicit_pair_parameters = {
            "phase_convention": frozen_manifest["phase_convention"],
            "alpha": frozen_manifest["alpha"],
            "beta1": frozen_manifest["beta1"],
            "beta2": frozen_manifest["beta2"],
        }
    elif hasattr(args, "stage") and args.phase_convention is not None:
        explicit_pair_parameters = {
            "phase_convention": args.phase_convention,
            "alpha": args.alpha,
            "beta1": args.beta1,
            "beta2": args.beta2,
        }
    active_phase_convention = (
        PhaseConvention.BENSEN_VELOCITY_CCF
        if explicit_pair_parameters is None
        else PhaseConvention[
            str(explicit_pair_parameters["phase_convention"])
        ]
    )
    if hasattr(args, "stage") and args.stage == "B":
        return run_stage_b_from_tasks(
            tasks=tuple(tasks),
            station_coordinates=station_coords,
            input_inventory=input_inventory,
            input_inventory_sha256=str(input_inventory_sha256),
            code_sha256=str(code_sha256),
            config_sha256=str(config_sha256),
            output_dir=args.output_dir,
            component=component,
            max_workers=max(1, int(args.max_workers)),
        )
    task_source = formal_base_tasks if formal_base_tasks else tasks
    tasks = (
        tuple(task)
        + (
            (component,)
            if explicit_pair_parameters is None
            else (component, dict(explicit_pair_parameters))
        )
        for task in task_source
    )
    ensure_dir(args.output_dir)
    figures_dir = args.output_dir / "figures"
    ensure_dir(figures_dir)
    initial_rows_by_period: Dict[float, List[PhaseMeasurement]] = {period: [] for period in TARGET_PERIODS_S}
    all_initial_rows_csv: List[Dict[str, object]] = []
    all_raw_rows_csv: List[Dict[str, object]] = []
    reference_observations: List[ReferenceObservation] = []
    processed_pair_count = 0
    successful_pair_count = 0
    pairs_with_any_initial_pass = 0
    failures: List[Dict[str, object]] = []
    preliminary_snr_rows: List[Dict[str, object]] = []
    def consume_results(results):
        nonlocal processed_pair_count
        nonlocal successful_pair_count
        nonlocal pairs_with_any_initial_pass
        for result in results:
            processed_pair_count += 1
            preliminary_row = result.get("preliminary_snr_row")
            if preliminary_row is not None:
                preliminary_snr_rows.append(dict(preliminary_row))
            if not result.get("ok"):
                failures.append(
                    {
                        "pair_name": result["pair_name"],
                        "stage": result.get("failure_stage", "pair"),
                        "reason": result.get("reason", "unknown"),
                        "failure_kind": result.get(
                            "failure_kind",
                            "unexpected_pair_exception",
                        ),
                        "exception_type": result.get("exception_type"),
                        "nominal_period_s": None,
                        "instantaneous_period_s": None,
                        "target_period_s": None,
                    }
                )
                if processed_pair_count % 5000 == 0:
                    print(f"[progress] processed={processed_pair_count} passes={pairs_with_any_initial_pass} failures={len(failures)}", flush=True)
                continue
            successful_pair_count += 1
            failures.extend(result.get("rejections", []))
            all_raw_rows_csv.extend(
                dict(row) for row in result.get("raw_measurements", [])
            )
            for row in result.get("continuous_observations", []):
                reference_observations.append(
                    ReferenceObservation(
                        pair_name=str(row["pair_name"]),
                        distance_km=float(row["distance_km"]),
                        azimuth_deg=float(row["azimuth_deg"]),
                        instantaneous_period_s=float(
                            row["instantaneous_period_s"]
                        ),
                        anchored_raw_time_s=float(
                            row["anchored_raw_time_s"]
                        ),
                        group_slowness_s_km=float(
                            row["group_slowness_s_km"]
                        ),
                        convention=PhaseConvention[
                            str(row["convention"])
                        ],
                    )
                )
            measurements = result.get("measurements", [])
            if measurements:
                pairs_with_any_initial_pass += 1
            for row in measurements:
                measurement = PhaseMeasurement(
                    pair_name=str(row["pair_name"]),
                    distance_km=float(row["distance_km"]),
                    period_s=float(row["period_s"]),
                    group_time_s=float(row["group_time_s"]),
                    group_velocity_km_s=float(row["group_velocity_km_s"]),
                    leading_snr=float(row["leading_snr"]),
                    trailing_snr=float(row["trailing_snr"]),
                    phi_tu_rad=float(row["phi_tu_rad"]),
                    raw_travel_time_s=float(row["raw_travel_time_s"]),
                )
                initial_rows_by_period[measurement.period_s].append(measurement)
                all_initial_rows_csv.append(dict(row))
            if processed_pair_count % 5000 == 0:
                print(f"[progress] processed={processed_pair_count} passes={pairs_with_any_initial_pass} failures={len(failures)}", flush=True)

    checkpoint_run = None
    formal_configured_tasks: Tuple[Tuple[object, ...], ...] = ()
    if hasattr(args, "stage"):
        tasks = tuple(tasks)
        if args.stage == "C":
            formal_configured_tasks = tuple(tasks)
        checkpoint_config_sha256 = str(config_sha256)
        checkpoint_frozen_lineage = (
            {
                "stage_b_status": frozen_manifest["stage_b_status"],
                "input_inventory_sha256": frozen_manifest[
                    "input_inventory_sha256"
                ],
                "code_sha256": frozen_manifest["code_sha256"],
                "config_sha256": frozen_manifest["config_sha256"],
                "validation_table_sha256": frozen_manifest[
                    "validation_table_sha256"
                ],
                "frozen_parameters_file_sha256": hashlib.sha256(
                    Path(args.frozen_parameters).read_bytes()
                ).hexdigest(),
            }
            if frozen_manifest is not None
            else None
        )
        checkpoint_run = run_checkpointed_pair_tasks(
            tasks,
            output_dir=args.output_dir,
            chunk_size=max(1, int(args.chunksize)),
            config_sha256=checkpoint_config_sha256,
            process_task=process_one_pair,
            max_workers=max(1, int(args.max_workers)),
            resume=bool(args.resume),
            frozen_lineage=checkpoint_frozen_lineage,
        )
        consume_results(checkpoint_run["results"])
    else:
        with Pool(processes=max(1, int(args.max_workers))) as pool:
            consume_results(
                pool.imap_unordered(
                    process_one_pair,
                    tasks,
                    chunksize=max(1, int(args.chunksize)),
                )
            )

    accepted_measurement_count = len(all_initial_rows_csv)
    preliminary_snr_audit = preliminary_snr_inventory(
        preliminary_snr_rows,
        processed_pair_count=processed_pair_count,
    )
    (args.output_dir / "input_inventory.json").write_text(
        json.dumps(
            input_inventory,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    expected_rejection_count = sum(
        row.get("failure_kind") == "expected_scientific_rejection"
        for row in failures
    )
    unexpected_exception_count = sum(
        row.get("failure_kind") == "unexpected_pair_exception"
        for row in failures
    )
    terminal_failure_reason = None
    formal_pre_reference_failure = (
        formal_science_failure_reason(
            input_count=processed_pair_count,
            unexpected_exception_count=unexpected_exception_count,
            left_count_by_period={
                period: len(rows)
                for period, rows in initial_rows_by_period.items()
            },
        )
        if hasattr(args, "stage") and processed_pair_count > 0
        else None
    )
    if formal_pre_reference_failure is not None:
        terminal_failure_reason = formal_pre_reference_failure
    elif processed_pair_count == 0:
        terminal_failure_reason = "zero_tasks"
    elif successful_pair_count == 0:
        terminal_failure_reason = "all_pairs_failed"
    elif accepted_measurement_count == 0:
        terminal_failure_reason = "no_accepted_measurements"
    elif len(reference_observations) < 5:
        terminal_failure_reason = "reference_observations_insufficient"
    if terminal_failure_reason is not None:
        removed_stale_artifacts = remove_stale_success_artifacts(
            args.output_dir,
            figures_dir,
        )
        failure_metadata = build_terminal_failure_metadata(
            args=args,
            terminal_failure_reason=terminal_failure_reason,
            processed_pair_count=processed_pair_count,
            successful_pair_count=successful_pair_count,
            accepted_measurement_count=accepted_measurement_count,
            failures=failures,
            removed_stale_artifacts=removed_stale_artifacts,
        )
        write_csv(args.output_dir / "failures.csv", failures)
        (args.output_dir / "metadata.json").write_text(
            json.dumps(
                failure_metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 2

    try:
        reference_fit = fit_reference_dispersion(reference_observations)
    except ValueError as error:
        if str(error) != (
            "at least five joint distance-azimuth blocks are required"
        ):
            raise
        reference_failure_status = "reference_insufficient_fold_blocks"
        removed_stale_artifacts = remove_stale_success_artifacts(
            args.output_dir,
            figures_dir,
        )
        failures.append(
            {
                "pair_name": None,
                "stage": "reference_fit",
                "reason": reference_failure_status,
                "failure_kind": "expected_scientific_rejection",
                "nominal_period_s": None,
                "instantaneous_period_s": None,
                "target_period_s": None,
            }
        )
        failure_metadata = build_terminal_failure_metadata(
            args=args,
            terminal_failure_reason=reference_failure_status,
            processed_pair_count=processed_pair_count,
            successful_pair_count=successful_pair_count,
            accepted_measurement_count=accepted_measurement_count,
            failures=failures,
            removed_stale_artifacts=removed_stale_artifacts,
            reference_fields={
                "reference_fit_status": reference_failure_status,
                "reference_fit_error": str(error),
                "reference_observation_count": len(
                    reference_observations
                ),
                "reference_cv_optimizer_calls": 0,
                "reference_final_optimizer_calls": 0,
                "reference_optimizer_calls": 0,
            },
        )
        write_csv(args.output_dir / "failures.csv", failures)
        (args.output_dir / "metadata.json").write_text(
            json.dumps(failure_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 2
    alias_solution_rows = reference_alias_solution_rows(reference_fit)
    cv_audit_rows = reference_cv_audit_rows(reference_fit)
    if reference_fit.status != "accepted":
        removed_stale_artifacts = remove_stale_success_artifacts(
            args.output_dir,
            figures_dir,
        )
        failures.append(
            {
                "pair_name": None,
                "stage": "reference_fit",
                "reason": reference_fit.status,
                "failure_kind": "expected_scientific_rejection",
                "nominal_period_s": None,
                "instantaneous_period_s": None,
                "target_period_s": None,
            }
        )
        failure_metadata = build_terminal_failure_metadata(
            args=args,
            terminal_failure_reason=reference_fit.status,
            processed_pair_count=processed_pair_count,
            successful_pair_count=successful_pair_count,
            accepted_measurement_count=accepted_measurement_count,
            failures=failures,
            removed_stale_artifacts=removed_stale_artifacts,
            reference_fields={
                "reference_fit_status": reference_fit.status,
                "reference_fit_hash": reference_fit.result_hash,
                "reference_cv_result_hash": (
                    reference_fit.cv_result.result_hash
                ),
                "reference_fold_assignment_hash": (
                    reference_fit.cv_result.fold_assignment.assignment_hash
                ),
                "reference_cv_audit_csv": "reference_cv_audit.csv",
                "reference_cv_optimizer_calls": (
                    reference_fit.cv_optimizer_calls
                ),
                "reference_final_optimizer_calls": (
                    reference_fit.final_optimizer_calls
                ),
                "reference_optimizer_calls": (
                    reference_fit.optimizer_calls
                ),
                "reference_observation_count": len(
                    reference_observations
                ),
            },
        )
        write_csv(args.output_dir / "failures.csv", failures)
        write_csv(
            args.output_dir / "reference_alias_solutions.csv",
            alias_solution_rows,
        )
        write_csv(
            args.output_dir / "reference_cv_audit.csv",
            cv_audit_rows,
        )
        (args.output_dir / "metadata.json").write_text(
            json.dumps(failure_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 2

    continuous_cycle_results = resolve_reference_cycles(
        raw_times_s=np.asarray(
            [row.anchored_raw_time_s for row in reference_observations],
            dtype=float,
        ),
        distance_km=np.asarray(
            [row.distance_km for row in reference_observations],
            dtype=float,
        ),
        observation_periods_s=np.asarray(
            [row.instantaneous_period_s for row in reference_observations],
            dtype=float,
        ),
        reference_periods_s=reference_fit.periods_s,
        reference_slowness_s_km=reference_fit.phase_slowness_s_km,
        convention=active_phase_convention,
    )
    continuous_cycle_rows = [
        {
            "pair_name": observation.pair_name,
            "distance_km": observation.distance_km,
            "azimuth_deg": observation.azimuth_deg,
            "instantaneous_period_s": observation.instantaneous_period_s,
            "anchored_raw_time_s": observation.anchored_raw_time_s,
            "reference_time_s": cycle.reference_time_s,
            "cycle_count": cycle.cycle_count,
            "corrected_time_s": cycle.corrected_time_s,
            "corrected_residual_s": cycle.corrected_residual_s,
            "branch_tie": cycle.branch_tie,
            "convention": observation.convention.name,
        }
        for observation, cycle in zip(
            reference_observations,
            continuous_cycle_results,
        )
    ]
    per_period: Dict[float, Dict[str, object]] = {}
    corrected_rows_csv: List[Dict[str, object]] = []
    right_rows_csv: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    left_audit_by_key = {
        (str(row["pair_name"]), float(row["period_s"])): dict(row)
        for row in all_initial_rows_csv
    }
    for period_s in TARGET_PERIODS_S:
        payload = build_period_payload(
            initial_rows_by_period[period_s],
            reference_fit,
            convention=active_phase_convention,
        )
        for table_name in ("corrected_rows", "right_rows"):
            payload[table_name] = [
                {
                    **left_audit_by_key.get(
                        (str(row["pair_name"]), float(row["period_s"])),
                        {},
                    ),
                    **dict(row),
                }
                for row in payload[table_name]
            ]
        per_period[period_s] = payload
        corrected_rows_csv.extend(payload["corrected_rows"])
        right_rows_csv.extend(payload["right_rows"])
        fit_v = float(payload["fit_velocity_km_s"])
        ordinary_v = float(payload["ordinary_ls_velocity_km_s"])
        std_v = float(payload["std_velocity_km_s"])
        bootstrap_std_v = float(
            payload["bootstrap_velocity_std_km_s"]
        )
        bootstrap_ci_low = float(
            payload["bootstrap_velocity_ci95_low_km_s"]
        )
        bootstrap_ci_high = float(
            payload["bootstrap_velocity_ci95_high_km_s"]
        )
        ref_v = float(payload["reference_velocity_km_s"])
        bootstrap_ci_display = (
            "NA"
            if not (
                np.isfinite(bootstrap_ci_low)
                and np.isfinite(bootstrap_ci_high)
            )
            else f"[{bootstrap_ci_low:.2f}, {bootstrap_ci_high:.2f}]"
        )
        summary_rows.append(
            {
                "period_s": period_s,
                "initial_count": len(payload["left_rows"]),
                "corrected_count": len(payload["corrected_rows"]),
                "right_qc_count": len(payload["right_rows"]),
                "reference_velocity_km_s": ref_v,
                "fit_velocity_km_s": fit_v,
                "huber_velocity_km_s": fit_v,
                "ordinary_ls_velocity_km_s": ordinary_v,
                "std_velocity_km_s": std_v,
                "path_velocity_std_km_s": std_v,
                "bootstrap_velocity_std_km_s": bootstrap_std_v,
                "bootstrap_velocity_ci95_low_km_s": bootstrap_ci_low,
                "bootstrap_velocity_ci95_high_km_s": bootstrap_ci_high,
                "bootstrap_samples": payload["bootstrap_samples"],
                "bootstrap_seed": payload["bootstrap_seed"],
                "reference_velocity_km_s_display": "NA" if not np.isfinite(ref_v) else f"{ref_v:.2f}",
                "fit_velocity_km_s_display": "NA" if not np.isfinite(fit_v) else f"{fit_v:.2f}",
                "ordinary_ls_velocity_km_s_display": "NA" if not np.isfinite(ordinary_v) else f"{ordinary_v:.2f}",
                "std_velocity_km_s_display": "NA" if not np.isfinite(std_v) else f"{std_v:.2f}",
                "bootstrap_velocity_ci95_display": bootstrap_ci_display,
            }
        )

    if hasattr(args, "stage"):
        right_failure = formal_science_failure_reason(
            input_count=processed_pair_count,
            unexpected_exception_count=unexpected_exception_count,
            left_count_by_period={
                period: len(payload["left_rows"])
                for period, payload in per_period.items()
            },
            right_count_by_period={
                period: len(payload["right_rows"])
                for period, payload in per_period.items()
            },
        )
        if right_failure is not None:
            failures.append(
                {
                    "pair_name": None,
                    "stage": "formal_output_gate",
                    "reason": right_failure,
                    "failure_kind": "expected_scientific_rejection",
                    "nominal_period_s": None,
                    "instantaneous_period_s": None,
                    "target_period_s": None,
                }
            )
            removed_stale_artifacts = remove_stale_success_artifacts(
                args.output_dir,
                figures_dir,
            )
            write_csv(args.output_dir / "failures.csv", failures)
            (args.output_dir / "metadata.json").write_text(
                json.dumps(
                    build_terminal_failure_metadata(
                        args=args,
                        terminal_failure_reason=right_failure,
                        processed_pair_count=processed_pair_count,
                        successful_pair_count=successful_pair_count,
                        accepted_measurement_count=(
                            accepted_measurement_count
                        ),
                        failures=failures,
                        removed_stale_artifacts=(
                            removed_stale_artifacts
                        ),
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return 2

    is_formal_stage_c = hasattr(args, "stage") and args.stage == "C"
    figure_path = (
        figures_dir / "wang_figure4_ftan_paper_scale.png"
        if is_formal_stage_c
        else figures_dir / "wang_figure4_reproduction.png"
    )
    figure_audit = plot_figure(
        figure_path,
        per_period,
        paper_scale=True,
    )
    if not isinstance(figure_audit, dict):
        figure_audit = {}
    full_range_figure_audit = None
    if is_formal_stage_c:
        full_range_figure_audit = plot_figure(
            figures_dir / "wang_figure4_ftan_full_range.png",
            per_period,
            paper_scale=False,
        )

    paper_3s = next(row for row in summary_rows if float(row["period_s"]) == 3.0)
    compare = {
        "observed_fit_velocity": paper_3s["fit_velocity_km_s_display"],
        "observed_std_velocity": paper_3s["std_velocity_km_s_display"],
        "fit_matches_paper": paper_3s["fit_velocity_km_s_display"] == "2.70",
        "std_matches_paper": paper_3s["std_velocity_km_s_display"] == "0.17",
    }
    metadata = {
        "host": platform.node(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": run_started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "exit_status": 0,
        "run_status": "success",
        "stack_root": str(args.stack_root),
        "stations_csv": str(args.stations_csv),
        "bbox": args.bbox,
        "bbox_mode": args.bbox_mode,
        "processed_pair_count": processed_pair_count,
        "successful_pair_count": successful_pair_count,
        "accepted_measurement_count": accepted_measurement_count,
        "reference_fit_status": reference_fit.status,
        "reference_fit_hash": reference_fit.result_hash,
        "reference_cv_result_hash": reference_fit.cv_result.result_hash,
        "reference_fold_assignment_hash": (
            reference_fit.cv_result.fold_assignment.assignment_hash
        ),
        "reference_cv_audit_csv": "reference_cv_audit.csv",
        "input_inventory_json": "input_inventory.json",
        "input_lineage_status": input_inventory["lineage_status"],
        "preliminary_snr_inventory": preliminary_snr_audit,
        "reference_observation_count": len(reference_observations),
        "reference_lambda_s": reference_fit.lambda_s,
        "reference_lambda_g": reference_fit.lambda_g,
        "reference_cv_optimizer_calls": reference_fit.cv_optimizer_calls,
        "reference_final_optimizer_calls": (
            reference_fit.final_optimizer_calls
        ),
        "reference_optimizer_calls": reference_fit.optimizer_calls,
        "right_column_fit": {
            "parameter": "slowness_s_km",
            "prediction": (
                "travel_time_s = distance_km * slowness_s_km"
            ),
            "primary_fit": "Huber IRLS through origin",
            "huber_tuning_constant": 1.345,
            "sensitivity_fit": "ordinary least squares through origin",
            "bootstrap_unit": "station pair",
            "bootstrap_samples": 1000,
            "bootstrap_seed": 20260717,
            "bootstrap_interval_percent": [2.5, 97.5],
        },
        "pairs_with_any_initial_pass": pairs_with_any_initial_pass,
        "failure_count": len(failures),
        "expected_scientific_rejection_count": expected_rejection_count,
        "unexpected_pair_exception_count": unexpected_exception_count,
        "max_workers": int(args.max_workers),
        "chunksize": int(args.chunksize),
        "target_periods_s": list(TARGET_PERIODS_S),
        "component": component,
        "snr_threshold": SNR_THRESHOLD,
        "snr_windows": {
            "signal_velocity_window_km_s": [SIGNAL_VMIN_KM_S, SIGNAL_VMAX_KM_S],
            "leading_noise": "[dt, distance/5.0 - 0.5*T_inst]",
            "trailing_noise": "[distance/1.6 + 0.5*T_inst, tmax]",
            "rms_source": "filtered_waveform",
            "minimum_samples_each": MIN_NOISE_SAMPLES,
            "period_for_guard": "T_inst",
            "target_qc_period": "T_target",
        },
        "signal_velocity_window_km_s": [SIGNAL_VMIN_KM_S, SIGNAL_VMAX_KM_S],
        "group_velocity_limits_km_s": {
            "<4.5s": GROUP_VMAX_SHORT_KM_S,
            ">=4.5s": GROUP_VMAX_LONG_KM_S,
        },
        "paper_target_comparison_3s": compare,
        "figure4_audit": figure_audit,
    }
    if hasattr(args, "stage"):
        metadata.update(
            formal_success_lineage_metadata(
                stage=args.stage,
                component=component,
                input_inventory_sha256=str(input_inventory_sha256),
                code_sha256=str(code_sha256),
                config_sha256=str(config_sha256),
                frozen_manifest=frozen_manifest,
                frozen_parameters_path=args.frozen_parameters,
                checkpoint_run=checkpoint_run,
            )
        )
    if is_formal_stage_c:
        try:
            stage_b_evidence = load_stage_b_evidence_for_report(
                args.frozen_parameters
            )
            stage_b_tables = stage_b_audit_rows(
                stage_b_evidence,
                frozen_manifest,
            )
            reference_rows = reference_dispersion_rows(reference_fit)
            spatial_rows = reference_spatial_diagnostic_rows(
                continuous_cycle_rows
            )
            eligible_pairs = sorted(
                {str(row["pair_name"]) for row in all_raw_rows_csv}
            )
            ftan_example_audit = render_ftan_examples_from_tasks(
                formal_configured_tasks,
                output_dir=args.output_dir,
                eligible_pair_names=eligible_pairs,
            )
            plot_reference_dispersion_stability(
                figures_dir / "reference_dispersion_stability.png",
                reference_rows,
                stage_b_tables["split_half_stability"],
            )
            plot_phase_convention_validation(
                figures_dir / "phase_convention_validation.png",
                stage_b_tables["candidate_grid_results"],
                stage_b_tables["phase_matching_comparison"],
            )
            plot_triplet_closure(
                figures_dir / "triplet_closure.png",
                stage_b_tables["triplet_closure"],
            )
        except (OSError, ValueError, RuntimeError) as exc:
            metadata.update(
                {
                    "run_status": "failed",
                    "exit_status": 2,
                    "finished_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "terminal_failure_reason": (
                        "formal_audit_artifact_generation_failed"
                    ),
                    "detail": str(exc),
                }
            )
            (args.output_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return 2
        frozen_source = Path(args.frozen_parameters)
        frozen_destination = args.output_dir / "frozen_parameters.json"
        if frozen_source.resolve() != frozen_destination.resolve():
            frozen_destination.write_bytes(frozen_source.read_bytes())
        stage_count_rows = [
            {"stage": "input_pairs", "status": "processed", "count": processed_pair_count},
            {"stage": "input_pairs", "status": "successful", "count": successful_pair_count},
            {"stage": "continuous_ftan", "status": "raw", "count": len(all_raw_rows_csv)},
            {"stage": "target_period", "status": "left_qc", "count": len(all_initial_rows_csv)},
            {"stage": "cycle_correction", "status": "corrected", "count": len(corrected_rows_csv)},
            {"stage": "right_column", "status": "accepted", "count": len(right_rows_csv)},
            {"stage": "rejection", "status": "expected", "count": expected_rejection_count},
            {"stage": "rejection", "status": "unexpected", "count": unexpected_exception_count},
        ]
        write_csv(args.output_dir / "reference_dispersion.csv", reference_rows)
        write_csv(
            args.output_dir / "reference_alias_solutions.csv",
            alias_solution_rows,
        )
        write_csv(
            args.output_dir / "candidate_grid_results.csv",
            stage_b_tables["candidate_grid_results"],
        )
        write_csv(
            args.output_dir / "cycle_count_distribution.csv",
            stage_b_tables["cycle_count_distribution"],
        )
        write_csv(
            args.output_dir / "split_half_stability.csv",
            stage_b_tables["split_half_stability"],
        )
        write_csv(
            args.output_dir / "split_half_membership.csv",
            stage_b_tables["split_half_membership"],
        )
        write_csv(
            args.output_dir / "reference_spatial_diagnostics.csv",
            spatial_rows,
        )
        write_csv(
            args.output_dir / "phase_matching_comparison.csv",
            stage_b_tables["phase_matching_comparison"],
        )
        write_csv(args.output_dir / "measurements_raw.csv", all_raw_rows_csv)
        write_csv(
            args.output_dir / "measurements_left_qc.csv",
            all_initial_rows_csv,
        )
        write_csv(
            args.output_dir / "measurements_corrected.csv",
            corrected_rows_csv,
        )
        write_csv(
            args.output_dir / "measurements_right_qc.csv",
            right_rows_csv,
        )
        write_csv(args.output_dir / "rejections.csv", failures)
        write_csv(args.output_dir / "stage_counts.csv", stage_count_rows)
        write_csv(args.output_dir / "fit_summary.csv", summary_rows)
        try:
            git_commit_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            git_commit_sha = "unknown"
        metadata.update(
            {
                "git_commit_sha": git_commit_sha,
                "python_version": platform.python_version(),
                "dependency_versions": {
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "h5py": h5py.__version__,
                    "matplotlib": matplotlib.__version__,
                    "pyyaml": (
                        "unavailable" if yaml is None else yaml.__version__
                    ),
                },
                "input_file_count": int(input_inventory["stack_file_count"]),
                "phase_convention": frozen_manifest["phase_convention"],
                "frozen_candidate": {
                    key: frozen_manifest[key]
                    for key in (
                        "candidate_id",
                        "phase_convention",
                        "alpha",
                        "beta1",
                        "beta2",
                    )
                },
                "full_range_figure_audit": full_range_figure_audit,
                "ftan_example_audit": ftan_example_audit,
                "audit_csv_counts": {
                    "candidate_grid_results": len(
                        stage_b_tables["candidate_grid_results"]
                    ),
                    "cycle_count_distribution": len(
                        stage_b_tables["cycle_count_distribution"]
                    ),
                    "split_half_stability": len(
                        stage_b_tables["split_half_stability"]
                    ),
                    "split_half_membership": len(
                        stage_b_tables["split_half_membership"]
                    ),
                    "reference_spatial_diagnostics": len(spatial_rows),
                    "phase_matching_comparison": len(
                        stage_b_tables["phase_matching_comparison"]
                    ),
                    "measurements_raw": len(all_raw_rows_csv),
                    "measurements_left_qc": len(all_initial_rows_csv),
                    "measurements_corrected": len(corrected_rows_csv),
                    "measurements_right_qc": len(right_rows_csv),
                    "rejections": len(failures),
                },
            }
        )
        (args.output_dir / "run.log").write_text(
            "\n".join(
                (
                    f"started_at={run_started_at}",
                    f"finished_at={metadata['finished_at']}",
                    "stage=C",
                    f"processed_pair_count={processed_pair_count}",
                    f"successful_pair_count={successful_pair_count}",
                    f"left_measurement_count={len(all_initial_rows_csv)}",
                    f"right_measurement_count={len(right_rows_csv)}",
                    "exit_status=0",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        provisional_validation = {
            "candidate_count": len(stage_b_tables["candidate_grid_results"]),
            "split_count": len(
                {
                    int(row["split_index"])
                    for row in stage_b_tables["split_half_stability"]
                }
            ),
            "ftan_example_count": len(ftan_example_audit),
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_formal_report_html(
            args.output_dir / "report.html",
            metadata=metadata,
            summary_rows=summary_rows,
            formal_validation=provisional_validation,
        )
        formal_validation = validate_formal_outputs(
            args.output_dir,
            expected_left_count_by_period={
                period: len(per_period[period]["left_rows"])
                for period in TARGET_PERIODS_S
            },
            expected_right_count_by_period={
                period: len(per_period[period]["right_rows"])
                for period in TARGET_PERIODS_S
            },
        )
        metadata["formal_output_validation"] = formal_validation
        if not formal_validation["accepted"]:
            metadata.update(
                {
                    "run_status": "failed",
                    "exit_status": 2,
                    "terminal_failure_reason": formal_validation["status"],
                }
            )
            (args.output_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return 2
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_formal_report_html(
            args.output_dir / "report.html",
            metadata=metadata,
            summary_rows=summary_rows,
            formal_validation=formal_validation,
        )
    else:
        write_csv(
            args.output_dir / "measurements_initial_qc.csv",
            all_initial_rows_csv,
        )
        write_csv(
            args.output_dir / "measurements_corrected.csv",
            corrected_rows_csv,
        )
        write_csv(
            args.output_dir / "measurements_right_qc.csv",
            right_rows_csv,
        )
        write_csv(
            args.output_dir / "continuous_reference_cycles.csv",
            continuous_cycle_rows,
        )
        write_csv(
            args.output_dir / "reference_alias_solutions.csv",
            alias_solution_rows,
        )
        write_csv(
            args.output_dir / "reference_cv_audit.csv",
            cv_audit_rows,
        )
        write_csv(args.output_dir / "failures.csv", failures)
        write_csv(args.output_dir / "fit_summary.csv", summary_rows)
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_report_html(
            args.output_dir / "report.html",
            metadata=metadata,
            summary_rows=summary_rows,
            figure_relpath=relative_to_output(
                args.output_dir / "report.html",
                figure_path,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
