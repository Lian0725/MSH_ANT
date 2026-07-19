#!/usr/bin/env python3
"""Deterministic Stage A/B validation helpers for the Wang-style FTAN flow."""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import multiprocessing as mp
import os
from types import MappingProxyType
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


DISTANCE_QUANTILE_PROBABILITIES = (0.2, 0.4, 0.6, 0.8)
SNR_QUANTILE_PROBABILITIES = (1.0 / 3.0, 2.0 / 3.0)
AZIMUTH_SECTOR_WIDTH_DEG = 45.0
EARTH_KM_PER_DEGREE = 111.195
WANG_TARGET_PERIODS_S = (3.0, 3.5, 4.0, 5.0)
STAGE_B_OPTIMIZER_CALLS_PER_CLASS = 953
STAGE_B_MAXIMUM_WALL_SECONDS = 24.0 * 60.0 * 60.0
STAGE_B_MAXIMUM_MEMORY_FRACTION = 0.70
STAGE_B_BENCHMARK_WAVEFORM_COUNT = 20
STAGE_B_BENCHMARK_OPTIMIZER_CALLS = 335
REFERENCE_LAMBDA_GRID = (0.0, 0.001, 0.01, 0.1, 1.0)
FORMAL_PHASE_CONVENTIONS = (
    "BENSEN_VELOCITY_CCF",
    "LIN_NEGATIVE_DERIVATIVE_EGF",
)
FORMAL_ALPHA_CANDIDATES = (5.0, 8.0, 12.0, 16.0, 20.0, 25.0)
FORMAL_BETA1_CANDIDATES = (0.0, 0.5, 1.0, 2.0, 4.0)
FORMAL_BETA2_CANDIDATES = (0.0, 1.0, 2.0, 4.0, 8.0)
STAGE_B_RANDOM_PAIR_LIMIT = 2000
TRIPLET_MINIMUM_SUPPORT = 100
TRIPLET_MAXIMUM_MEDIAN_ABSOLUTE_CYCLES = 0.15
TRIPLET_MAXIMUM_ABSOLUTE_BIAS_CYCLES = 0.05
HALF_SAMPLE_MAXIMUM_MEDIAN_DIFFERENCE_KM_S = 0.03
HALF_SAMPLE_MAXIMUM_P90_DIFFERENCE_KM_S = 0.05
CANDIDATE_MAXIMUM_BOUNDARY_FRACTION = 0.05
CANDIDATE_RELATIVE_TIE_FRACTION = 0.05
PHASE_MATCHING_MINIMUM_CLOSURE_REDUCTION_FRACTION = 0.10


_MEASUREMENT_CLASS_PROCESS_EVALUATOR = None


def _evaluate_measurement_class_process_job(job: object) -> object:
    evaluator = _MEASUREMENT_CLASS_PROCESS_EVALUATOR
    if evaluator is None:
        raise RuntimeError("measurement-class process evaluator is unavailable")
    return evaluator(job)


def execute_measurement_class_processes(
    jobs: Sequence[object],
    *,
    evaluator: Callable[[object], object],
    max_workers: int,
) -> Tuple[object, ...]:
    """Evaluate independent class jobs in a bounded fork process pool."""

    workers = int(max_workers)
    if workers < 1 or workers > 24:
        raise ValueError("max_workers must lie in [1, 24]")
    items = tuple(jobs)
    if not items:
        return ()
    if workers == 1:
        return tuple(evaluator(job) for job in items)
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError(
            "formal Stage B class parallelism requires multiprocessing fork"
        )
    global _MEASUREMENT_CLASS_PROCESS_EVALUATOR
    if _MEASUREMENT_CLASS_PROCESS_EVALUATOR is not None:
        raise RuntimeError("measurement-class process executor is not reentrant")
    _MEASUREMENT_CLASS_PROCESS_EVALUATOR = evaluator
    try:
        context = mp.get_context("fork")
        with context.Pool(processes=min(workers, len(items))) as pool:
            results = pool.map(
                _evaluate_measurement_class_process_job,
                items,
                chunksize=1,
            )
    finally:
        _MEASUREMENT_CLASS_PROCESS_EVALUATOR = None
    return tuple(results)


@dataclass(frozen=True)
class StageBSelection:
    random_pair_names: Tuple[str, ...]
    closure_edge_pair_names: Tuple[str, ...]
    selected_pair_names: Tuple[str, ...]
    distance_quintile_edges_km: np.ndarray
    snr_tertile_edges: np.ndarray
    distance_quantile_probabilities: Tuple[float, ...]
    snr_quantile_probabilities: Tuple[float, ...]
    azimuth_sector_width_deg: float
    stratum_by_pair: Mapping[str, Tuple[int, int, int]]
    stratum_random_counts: Mapping[Tuple[int, int, int], int]
    seed: int
    max_random_pairs: int
    membership_sha256: str

    def __post_init__(self) -> None:
        random_names = tuple(self.random_pair_names)
        closure_names = tuple(self.closure_edge_pair_names)
        selected_names = tuple(self.selected_pair_names)
        distance_edges = np.array(
            self.distance_quintile_edges_km,
            dtype=float,
            copy=True,
        )
        snr_edges = np.array(
            self.snr_tertile_edges,
            dtype=float,
            copy=True,
        )
        seed = int(self.seed)
        maximum = int(self.max_random_pairs)
        if (
            len(random_names) > maximum
            or maximum <= 0
            or seed < 0
            or len(set(random_names)) != len(random_names)
            or len(set(closure_names)) != len(closure_names)
            or len(set(selected_names)) != len(selected_names)
            or tuple(sorted(random_names)) != random_names
            or tuple(sorted(closure_names)) != closure_names
            or tuple(sorted(selected_names)) != selected_names
            or set(selected_names) != set(random_names).union(closure_names)
            or distance_edges.shape != (4,)
            or snr_edges.shape != (2,)
            or np.any(~np.isfinite(distance_edges))
            or np.any(~np.isfinite(snr_edges))
            or np.any(np.diff(distance_edges) < 0)
            or np.any(np.diff(snr_edges) < 0)
            or not isinstance(self.membership_sha256, str)
            or len(self.membership_sha256) != 64
        ):
            raise ValueError("StageBSelection fields are inconsistent")
        strata = {
            str(name): tuple(int(value) for value in values)
            for name, values in self.stratum_by_pair.items()
        }
        counts = {
            tuple(int(value) for value in key): int(value)
            for key, value in self.stratum_random_counts.items()
        }
        if (
            set(strata) != set(selected_names)
            or any(len(values) != 3 for values in strata.values())
            or any(value < 0 for value in counts.values())
            or sum(counts.values()) != len(random_names)
        ):
            raise ValueError("Stage B stratum audit is inconsistent")
        distance_edges.setflags(write=False)
        snr_edges.setflags(write=False)
        object.__setattr__(self, "random_pair_names", random_names)
        object.__setattr__(self, "closure_edge_pair_names", closure_names)
        object.__setattr__(self, "selected_pair_names", selected_names)
        object.__setattr__(
            self,
            "distance_quintile_edges_km",
            distance_edges,
        )
        object.__setattr__(self, "snr_tertile_edges", snr_edges)
        object.__setattr__(
            self,
            "stratum_by_pair",
            MappingProxyType(strata),
        )
        object.__setattr__(
            self,
            "stratum_random_counts",
            MappingProxyType(counts),
        )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "max_random_pairs", maximum)


@dataclass(frozen=True)
class TripletGeometryResult:
    accepted: bool
    status: str
    projection_fraction: float
    cross_track_km: float
    distance_ab_km: float
    distance_bc_km: float
    distance_ac_km: float
    distance_closure_error_km: float


@dataclass(frozen=True)
class TripletPeriodSummary:
    period_s: float
    support_count: int
    median_absolute_cycles: float
    absolute_bias_cycles: float
    accepted: bool
    status: str


@dataclass(frozen=True)
class TripletClosureResult:
    accepted: bool
    status: str
    period_summaries: Mapping[float, TripletPeriodSummary]
    triplet_rows: Tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        summaries = {
            float(period): summary
            for period, summary in sorted(self.period_summaries.items())
        }
        rows = tuple(MappingProxyType(dict(row)) for row in self.triplet_rows)
        if (
            not summaries
            or any(
                not isinstance(summary, TripletPeriodSummary)
                for summary in summaries.values()
            )
            or not isinstance(self.accepted, (bool, np.bool_))
            or not isinstance(self.status, str)
            or not self.status
        ):
            raise ValueError("TripletClosureResult fields are inconsistent")
        object.__setattr__(
            self,
            "period_summaries",
            MappingProxyType(summaries),
        )
        object.__setattr__(self, "triplet_rows", rows)
        object.__setattr__(self, "accepted", bool(self.accepted))


@dataclass(frozen=True)
class HalfSampleSplit:
    split_index: int
    seed: int
    a_pair_names: Tuple[str, ...]
    b_pair_names: Tuple[str, ...]
    stratum_half_counts: Mapping[Tuple[int, int, int], Tuple[int, int]]
    stratum_by_pair: Mapping[str, Tuple[int, int, int]]
    snr_field: str
    odd_stratum_extra_side: str
    membership_sha256: str

    def __post_init__(self) -> None:
        a_names = tuple(self.a_pair_names)
        b_names = tuple(self.b_pair_names)
        counts = {
            tuple(int(component) for component in key): (
                int(half_count[0]),
                int(half_count[1]),
            )
            for key, half_count in self.stratum_half_counts.items()
        }
        strata = {
            str(name): tuple(int(component) for component in key)
            for name, key in self.stratum_by_pair.items()
        }
        if (
            self.split_index < 0
            or self.seed < 0
            or tuple(sorted(a_names)) != a_names
            or tuple(sorted(b_names)) != b_names
            or set(a_names).intersection(b_names)
            or set(strata) != set(a_names).union(b_names)
            or any(len(key) != 3 for key in strata.values())
            or self.snr_field != "candidate_left_snr"
            or self.odd_stratum_extra_side not in ("A", "B")
            or any(
                min(value) < 0 or abs(value[0] - value[1]) > 1
                for value in counts.values()
            )
            or len(self.membership_sha256) != 64
        ):
            raise ValueError("HalfSampleSplit fields are inconsistent")
        object.__setattr__(self, "a_pair_names", a_names)
        object.__setattr__(self, "b_pair_names", b_names)
        object.__setattr__(
            self,
            "stratum_half_counts",
            MappingProxyType(counts),
        )
        object.__setattr__(
            self,
            "stratum_by_pair",
            MappingProxyType(strata),
        )


@dataclass(frozen=True)
class HalfSamplePlan:
    splits: Tuple[HalfSampleSplit, ...]
    base_seed: int
    plan_sha256: str

    def __post_init__(self) -> None:
        splits = tuple(self.splits)
        if (
            not splits
            or any(not isinstance(row, HalfSampleSplit) for row in splits)
            or len(self.plan_sha256) != 64
        ):
            raise ValueError("HalfSamplePlan fields are inconsistent")
        object.__setattr__(self, "splits", splits)


@dataclass(frozen=True)
class HalfSamplePeriodSummary:
    period_s: float
    median_absolute_difference_km_s: float
    p90_absolute_difference_km_s: float
    accepted: bool
    status: str


@dataclass(frozen=True)
class HalfSampleStabilityResult:
    accepted: bool
    status: str
    period_summaries: Mapping[float, HalfSamplePeriodSummary]

    def __post_init__(self) -> None:
        summaries = {
            float(period): summary
            for period, summary in sorted(self.period_summaries.items())
        }
        if not summaries or any(
            not isinstance(summary, HalfSamplePeriodSummary)
            for summary in summaries.values()
        ):
            raise ValueError(
                "HalfSampleStabilityResult fields are inconsistent"
            )
        object.__setattr__(
            self,
            "period_summaries",
            MappingProxyType(summaries),
        )


@dataclass(frozen=True)
class CandidateBoundaryResult:
    accepted_measurement_count: int
    accepted_outermost_velocity_cell_count: int
    accepted_boundary_fraction: float
    maximum_boundary_fraction: float
    accepted: bool
    status: str

    def __post_init__(self) -> None:
        count = int(self.accepted_measurement_count)
        boundary_count = int(self.accepted_outermost_velocity_cell_count)
        fraction = float(self.accepted_boundary_fraction)
        maximum = float(self.maximum_boundary_fraction)
        if (
            count <= 0
            or boundary_count < 0
            or boundary_count > count
            or not np.isfinite(fraction)
            or not np.isfinite(maximum)
            or maximum < 0
            or maximum > 1
            or not np.isclose(fraction, boundary_count / count)
            or not isinstance(self.status, str)
            or not self.status
        ):
            raise ValueError("CandidateBoundaryResult fields are inconsistent")
        object.__setattr__(self, "accepted_measurement_count", count)
        object.__setattr__(
            self,
            "accepted_outermost_velocity_cell_count",
            boundary_count,
        )
        object.__setattr__(self, "accepted_boundary_fraction", fraction)
        object.__setattr__(self, "maximum_boundary_fraction", maximum)
        object.__setattr__(self, "accepted", bool(self.accepted))


@dataclass(frozen=True)
class PhaseMatchingDiagnosticResult:
    design_revision_required: bool
    freeze_raw_ftan: bool
    status: str
    closure_reduction_fraction_by_period: Mapping[float, float]
    period_reduction_passes: Mapping[float, bool]
    valid_ridge_coverage_increased: bool
    phase_convention_unchanged: bool
    boundary_fraction_not_worse: bool
    narrowband_sidelobe_validation_passed: bool

    def __post_init__(self) -> None:
        reductions = {
            float(period): float(value)
            for period, value in sorted(
                self.closure_reduction_fraction_by_period.items()
            )
        }
        passes = {
            float(period): bool(value)
            for period, value in sorted(self.period_reduction_passes.items())
        }
        if (
            not reductions
            or set(reductions) != set(passes)
            or tuple(sorted(reductions)) != WANG_TARGET_PERIODS_S
            or any(not np.isfinite(value) for value in reductions.values())
            or any(
                passes[period]
                != (
                    reductions[period]
                    >= PHASE_MATCHING_MINIMUM_CLOSURE_REDUCTION_FRACTION
                )
                for period in reductions
            )
            or bool(self.design_revision_required)
            != (
                all(passes.values())
                and bool(self.valid_ridge_coverage_increased)
                and bool(self.phase_convention_unchanged)
                and bool(self.boundary_fraction_not_worse)
                and bool(self.narrowband_sidelobe_validation_passed)
            )
            or bool(self.freeze_raw_ftan)
            == bool(self.design_revision_required)
            or self.status
            != (
                "phase_matching_design_revision_required"
                if bool(self.design_revision_required)
                else "raw_ftan_frozen"
            )
        ):
            raise ValueError(
                "PhaseMatchingDiagnosticResult fields are inconsistent"
            )
        object.__setattr__(
            self,
            "closure_reduction_fraction_by_period",
            MappingProxyType(reductions),
        )
        object.__setattr__(
            self,
            "period_reduction_passes",
            MappingProxyType(passes),
        )
        for name in (
            "design_revision_required",
            "freeze_raw_ftan",
            "valid_ridge_coverage_increased",
            "phase_convention_unchanged",
            "boundary_fraction_not_worse",
            "narrowband_sidelobe_validation_passed",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))


@dataclass(frozen=True)
class PhaseMatchingRunEvidence:
    candidate_id: str
    phase_convention: str
    first_pass_alpha: float
    second_pass_alpha: float
    cut_half_width_s: float
    cut_taper_alpha: float
    second_pass_ftan_executed: bool
    raw_output_sha256: str
    matched_output_sha256: str
    raw_closure_median_cycles: Mapping[float, float]
    matched_closure_median_cycles: Mapping[float, float]
    raw_valid_ridge_coverage: float
    matched_valid_ridge_coverage: float
    raw_boundary_fraction: float
    matched_boundary_fraction: float
    diagnostic: PhaseMatchingDiagnosticResult

    def __post_init__(self) -> None:
        first = float(self.first_pass_alpha)
        second = float(self.second_pass_alpha)
        half_width = float(self.cut_half_width_s)
        taper = float(self.cut_taper_alpha)
        hashes = (self.raw_output_sha256, self.matched_output_sha256)
        raw_closure = {
            float(period): float(value)
            for period, value in self.raw_closure_median_cycles.items()
        }
        matched_closure = {
            float(period): float(value)
            for period, value in self.matched_closure_median_cycles.items()
        }
        raw_coverage = float(self.raw_valid_ridge_coverage)
        matched_coverage = float(self.matched_valid_ridge_coverage)
        raw_boundary = float(self.raw_boundary_fraction)
        matched_boundary = float(self.matched_boundary_fraction)
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or self.phase_convention not in FORMAL_PHASE_CONVENTIONS
            or first not in FORMAL_ALPHA_CANDIDATES
            or second != min(2.0 * first, 50.0)
            or half_width != 10.0
            or taper != 0.25
            or self.second_pass_ftan_executed is not True
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in hashes
            )
            or not isinstance(
                self.diagnostic,
                PhaseMatchingDiagnosticResult,
            )
            or set(raw_closure) != set(WANG_TARGET_PERIODS_S)
            or set(matched_closure) != set(WANG_TARGET_PERIODS_S)
            or any(
                not np.isfinite(value) or value < 0
                for value in tuple(raw_closure.values())
                + tuple(matched_closure.values())
            )
            or any(
                not np.isfinite(value) or value < 0 or value > 1
                for value in (
                    raw_coverage,
                    matched_coverage,
                    raw_boundary,
                    matched_boundary,
                )
            )
        ):
            raise ValueError(
                "PhaseMatchingRunEvidence fields are inconsistent"
            )
        object.__setattr__(self, "first_pass_alpha", first)
        object.__setattr__(self, "second_pass_alpha", second)
        object.__setattr__(self, "cut_half_width_s", half_width)
        object.__setattr__(self, "cut_taper_alpha", taper)
        object.__setattr__(
            self,
            "raw_closure_median_cycles",
            MappingProxyType(raw_closure),
        )
        object.__setattr__(
            self,
            "matched_closure_median_cycles",
            MappingProxyType(matched_closure),
        )
        object.__setattr__(self, "raw_valid_ridge_coverage", raw_coverage)
        object.__setattr__(
            self,
            "matched_valid_ridge_coverage",
            matched_coverage,
        )
        object.__setattr__(self, "raw_boundary_fraction", raw_boundary)
        object.__setattr__(self, "matched_boundary_fraction", matched_boundary)


@dataclass(frozen=True)
class CandidateFreezeDecision:
    accepted: bool
    status: str
    selected_candidate: Mapping[str, object]
    eligible_candidate_ids: Tuple[str, ...]
    best_candidate_by_phase_convention: Mapping[str, str]
    lineage_status: str
    lineage_preferred_phase_convention: object

    def __post_init__(self) -> None:
        selected = MappingProxyType(dict(self.selected_candidate))
        eligible = tuple(sorted(str(value) for value in self.eligible_candidate_ids))
        best = MappingProxyType(
            {
                str(key): str(value)
                for key, value in sorted(
                    self.best_candidate_by_phase_convention.items()
                )
            }
        )
        if (
            not isinstance(self.status, str)
            or not self.status
            or not isinstance(self.lineage_status, str)
            or not self.lineage_status
            or len(set(eligible)) != len(eligible)
            or (bool(self.accepted) and not selected)
            or (not bool(self.accepted) and selected)
        ):
            raise ValueError("CandidateFreezeDecision fields are inconsistent")
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "selected_candidate", selected)
        object.__setattr__(self, "eligible_candidate_ids", eligible)
        object.__setattr__(
            self,
            "best_candidate_by_phase_convention",
            best,
        )


@dataclass(frozen=True)
class StageBBudgetResult:
    accepted: bool
    status: str
    projected_candidate_seconds: float
    projected_reference_seconds: float
    projected_total_seconds: float
    memory_fraction: float
    optimizer_calls_per_class: int
    candidate_projection_formula: str
    reference_projection_formula: str

    def __post_init__(self) -> None:
        candidate = float(self.projected_candidate_seconds)
        reference = float(self.projected_reference_seconds)
        total = float(self.projected_total_seconds)
        memory = float(self.memory_fraction)
        calls = int(self.optimizer_calls_per_class)
        accepted = bool(self.accepted)
        expected_acceptance = (
            total <= STAGE_B_MAXIMUM_WALL_SECONDS
            and memory <= STAGE_B_MAXIMUM_MEMORY_FRACTION
        )
        if (
            any(
                not np.isfinite(value) or value < 0
                for value in (candidate, reference, total, memory)
            )
            or not np.isclose(total, candidate + reference)
            or calls != STAGE_B_OPTIMIZER_CALLS_PER_CLASS
            or accepted != expected_acceptance
            or self.status
            != ("accepted" if accepted else "stage_b_budget_exceeded")
            or not isinstance(self.candidate_projection_formula, str)
            or not self.candidate_projection_formula
            or not isinstance(self.reference_projection_formula, str)
            or "953" not in self.reference_projection_formula
        ):
            raise ValueError("StageBBudgetResult fields are inconsistent")
        object.__setattr__(self, "projected_candidate_seconds", candidate)
        object.__setattr__(self, "projected_reference_seconds", reference)
        object.__setattr__(self, "projected_total_seconds", total)
        object.__setattr__(self, "memory_fraction", memory)
        object.__setattr__(self, "optimizer_calls_per_class", calls)
        object.__setattr__(self, "accepted", accepted)


@dataclass(frozen=True)
class StageBBenchmarkEvidence:
    candidate_grid_elapsed_s: float
    ten_single_reference_fits_elapsed_s: float
    lambda_cv_elapsed_s: float
    twenty_half_samples_elapsed_s: float
    measured_peak_memory_bytes: int
    available_memory_bytes: int
    cache_hit_fraction: float
    benchmark_input_sha256: str

    def __post_init__(self) -> None:
        elapsed_names = (
            "candidate_grid_elapsed_s",
            "ten_single_reference_fits_elapsed_s",
            "lambda_cv_elapsed_s",
            "twenty_half_samples_elapsed_s",
        )
        elapsed = {
            name: float(getattr(self, name)) for name in elapsed_names
        }
        peak = int(self.measured_peak_memory_bytes)
        available = int(self.available_memory_bytes)
        cache = float(self.cache_hit_fraction)
        if (
            any(
                not np.isfinite(value) or value <= 0
                for value in elapsed.values()
            )
            or peak <= 0
            or available <= 0
            or peak > available
            or not np.isfinite(cache)
            or cache < 0
            or cache > 1
            or not isinstance(self.benchmark_input_sha256, str)
            or len(self.benchmark_input_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.benchmark_input_sha256
            )
        ):
            raise ValueError("StageBBenchmarkEvidence fields are inconsistent")
        for name, value in elapsed.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "measured_peak_memory_bytes", peak)
        object.__setattr__(self, "available_memory_bytes", available)
        object.__setattr__(self, "cache_hit_fraction", cache)


@dataclass(frozen=True)
class CorrectedTargetObservation:
    pair_name: str
    target_period_s: float
    raw_time_s: float
    cycle_count: int
    corrected_time_s: float
    reference_time_s: float
    reference_residual_s: float
    leading_snr: float
    trailing_snr: float
    left_qc_accepted: bool
    branch_tie: bool = False

    def __post_init__(self) -> None:
        period = float(self.target_period_s)
        raw = float(self.raw_time_s)
        corrected = float(self.corrected_time_s)
        reference = float(self.reference_time_s)
        residual = float(self.reference_residual_s)
        count = int(self.cycle_count)
        leading = float(self.leading_snr)
        trailing = float(self.trailing_snr)
        tolerance = 128.0 * np.finfo(float).eps
        if (
            not isinstance(self.pair_name, str)
            or not self.pair_name
            or period not in WANG_TARGET_PERIODS_S
            or isinstance(self.cycle_count, (bool, np.bool_))
            or not isinstance(self.cycle_count, (int, np.integer))
            or any(
                not np.isfinite(value)
                for value in (
                    raw,
                    corrected,
                    reference,
                    residual,
                    leading,
                    trailing,
                )
            )
            or not isinstance(self.left_qc_accepted, (bool, np.bool_))
            or not isinstance(self.branch_tie, (bool, np.bool_))
            or not np.isclose(
                corrected,
                raw + count * period,
                rtol=tolerance,
                atol=tolerance,
            )
            or not np.isclose(
                residual,
                corrected - reference,
                rtol=tolerance,
                atol=tolerance,
            )
            or abs(residual) > 0.5 * period
        ):
            raise ValueError(
                "CorrectedTargetObservation fields are inconsistent"
            )
        object.__setattr__(self, "target_period_s", period)
        object.__setattr__(self, "raw_time_s", raw)
        object.__setattr__(self, "cycle_count", count)
        object.__setattr__(self, "corrected_time_s", corrected)
        object.__setattr__(self, "reference_time_s", reference)
        object.__setattr__(self, "reference_residual_s", residual)
        object.__setattr__(self, "leading_snr", leading)
        object.__setattr__(self, "trailing_snr", trailing)
        object.__setattr__(
            self,
            "left_qc_accepted",
            bool(self.left_qc_accepted),
        )
        object.__setattr__(self, "branch_tie", bool(self.branch_tie))


@dataclass(frozen=True)
class FullReferenceEvidence:
    status: str
    alias_status: str
    lambda_s: float
    lambda_g: float
    basin_starts: Tuple[object, ...]
    optimizer_calls: int
    corrected_rows: Tuple[CorrectedTargetObservation, ...]
    result_sha256: str

    def __post_init__(self) -> None:
        lambda_s = float(self.lambda_s)
        lambda_g = float(self.lambda_g)
        starts = tuple(self.basin_starts)
        calls = int(self.optimizer_calls)
        corrected = tuple(self.corrected_rows)
        corrected_keys = tuple(
            (row.pair_name, row.target_period_s)
            for row in corrected
            if isinstance(row, CorrectedTargetObservation)
        )
        if (
            self.status not in ("accepted", "rejected")
            or self.alias_status not in ("accepted", "rejected")
            or lambda_s not in REFERENCE_LAMBDA_GRID
            or lambda_g not in REFERENCE_LAMBDA_GRID
            or calls <= 0
            or calls > 753
            or (
                self.status == "accepted"
                and (
                    len(starts) != 5
                    or len(
                        {
                            repr(
                                getattr(start, "basin_id", start)
                            )
                            for start in starts
                        }
                    )
                    != 5
                )
            )
            or not isinstance(self.result_sha256, str)
            or len(self.result_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.result_sha256
            )
            or any(
                not isinstance(row, CorrectedTargetObservation)
                for row in corrected
            )
            or len(set(corrected_keys)) != len(corrected_keys)
            or (
                self.status == "accepted"
                and any(
                    not row.left_qc_accepted
                    or row.leading_snr <= 4.0
                    or row.trailing_snr <= 4.0
                    for row in corrected
                )
            )
        ):
            raise ValueError("FullReferenceEvidence fields are inconsistent")
        object.__setattr__(self, "lambda_s", lambda_s)
        object.__setattr__(self, "lambda_g", lambda_g)
        object.__setattr__(self, "basin_starts", starts)
        object.__setattr__(self, "optimizer_calls", calls)
        object.__setattr__(self, "corrected_rows", corrected)


@dataclass(frozen=True)
class SplitHalfFitEvidence:
    status: str
    target_velocities_km_s: np.ndarray
    lambda_s: float
    lambda_g: float
    optimizer_calls: int
    cv_optimizer_calls: int
    maxiter: int

    def __post_init__(self) -> None:
        velocities = np.array(
            self.target_velocities_km_s,
            dtype=float,
            copy=True,
        )
        lambda_s = float(self.lambda_s)
        lambda_g = float(self.lambda_g)
        calls = int(self.optimizer_calls)
        cv_calls = int(self.cv_optimizer_calls)
        maxiter = int(self.maxiter)
        if (
            self.status not in ("accepted", "rejected")
            or velocities.shape != (len(WANG_TARGET_PERIODS_S),)
            or (
                self.status == "accepted"
                and np.any(~np.isfinite(velocities))
            )
            or lambda_s not in REFERENCE_LAMBDA_GRID
            or lambda_g not in REFERENCE_LAMBDA_GRID
            or calls <= 0
            or calls > 5
            or cv_calls != 0
            or maxiter != 300
        ):
            raise ValueError("SplitHalfFitEvidence fields are inconsistent")
        velocities.setflags(write=False)
        object.__setattr__(self, "target_velocities_km_s", velocities)
        object.__setattr__(self, "lambda_s", lambda_s)
        object.__setattr__(self, "lambda_g", lambda_g)
        object.__setattr__(self, "optimizer_calls", calls)
        object.__setattr__(self, "cv_optimizer_calls", cv_calls)
        object.__setattr__(self, "maxiter", maxiter)


@dataclass(frozen=True)
class StageBRunResult:
    status: str
    return_code: int
    budget: StageBBudgetResult
    benchmark_evidence: StageBBenchmarkEvidence
    selection: object
    candidate_results: Tuple[Mapping[str, object], ...]
    measurement_classes: Mapping[str, Tuple[str, ...]]
    class_evidence: Mapping[str, Mapping[str, object]]
    decision: object
    phase_matching_diagnostics: Mapping[str, PhaseMatchingRunEvidence]
    frozen_parameters: object
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        code = int(self.return_code)
        candidates = tuple(
            MappingProxyType(dict(row)) for row in self.candidate_results
        )
        classes = MappingProxyType(
            {
                str(key): tuple(str(value) for value in values)
                for key, values in sorted(self.measurement_classes.items())
            }
        )
        class_evidence = MappingProxyType(
            {
                str(key): MappingProxyType(dict(value))
                for key, value in sorted(self.class_evidence.items())
            }
        )
        phase_matching = MappingProxyType(
            {
                str(key): value
                for key, value in sorted(
                    self.phase_matching_diagnostics.items()
                )
            }
        )
        audit = MappingProxyType(dict(self.audit))
        if (
            not isinstance(self.status, str)
            or not self.status
            or code not in (0, 2)
            or (code == 0) != (self.status == "passed")
            or not isinstance(self.budget, StageBBudgetResult)
            or not isinstance(
                self.benchmark_evidence,
                StageBBenchmarkEvidence,
            )
            or (code == 0 and self.frozen_parameters is None)
            or (code != 0 and self.frozen_parameters is not None)
        ):
            raise ValueError("StageBRunResult fields are inconsistent")
        if code == 0 and (
            not self.budget.accepted
            or not isinstance(self.selection, StageBSelection)
            or len(candidates) != 300
            or len(
                {
                    str(row.get("candidate_id", ""))
                    for row in candidates
                }
            )
            != 300
            or not isinstance(self.decision, CandidateFreezeDecision)
            or not self.decision.accepted
            or set(phase_matching) != set(FORMAL_PHASE_CONVENTIONS)
            or any(
                not isinstance(value, PhaseMatchingRunEvidence)
                or value.phase_convention != phase
                or not value.diagnostic.freeze_raw_ftan
                for phase, value in phase_matching.items()
            )
            or dict(self.frozen_parameters).get("stage_b_status")
            != "passed"
        ):
            raise ValueError(
                "passed StageBRunResult lacks complete scientific evidence"
            )
        if code == 0:
            candidate_ids = {
                str(row["candidate_id"]) for row in candidates
            }
            formal_signature = {
                (
                    str(row["candidate_id"]),
                    str(row["phase_convention"]),
                    float(row["alpha"]),
                    float(row["beta1"]),
                    float(row["beta2"]),
                )
                for row in build_candidate_grid(
                    phase_conventions=FORMAL_PHASE_CONVENTIONS,
                    alpha_candidates=FORMAL_ALPHA_CANDIDATES,
                    beta1_candidates=FORMAL_BETA1_CANDIDATES,
                    beta2_candidates=FORMAL_BETA2_CANDIDATES,
                )
            }
            candidate_signature = {
                (
                    str(row["candidate_id"]),
                    str(row["phase_convention"]),
                    float(row["alpha"]),
                    float(row["beta1"]),
                    float(row["beta2"]),
                )
                for row in candidates
            }
            expected_classes: Dict[str, list] = {}
            for row in candidates:
                expected_classes.setdefault(
                    str(row["left_observation_sha256"]),
                    [],
                ).append(str(row["candidate_id"]))
            expected_classes = {
                key: tuple(sorted(values))
                for key, values in expected_classes.items()
            }
            class_members = {
                candidate_id
                for members in classes.values()
                for candidate_id in members
            }
            frozen = dict(self.frozen_parameters)
            selected = dict(self.decision.selected_candidate)
            required_hashes = (
                "input_inventory_sha256",
                "code_sha256",
                "config_sha256",
                "validation_table_sha256",
            )
            recomputed = freeze_ftan_candidate(
                candidates,
                lineage_status=self.decision.lineage_status,
                lineage_preferred_phase_convention=(
                    self.decision.lineage_preferred_phase_convention
                ),
            )
            recomputed_evidence_hash = (
                stage_b_validation_evidence_sha256(
                    _stage_b_evidence_components(
                        budget=self.budget,
                        benchmark_evidence=self.benchmark_evidence,
                        selection=self.selection,
                        candidate_results=candidates,
                        measurement_classes=classes,
                        class_evidence=class_evidence,
                        phase_matching_diagnostics=phase_matching,
                    )
                )
            )
            eligible_class_hashes = {
                str(row["left_observation_sha256"])
                for row in candidates
                if all(bool(row.get(name, False)) for name in _CANDIDATE_GATE_FIELDS)
            }
            if (
                not classes
                or candidate_signature != formal_signature
                or set(class_evidence) != set(classes)
                or dict(classes) != expected_classes
                or any(
                    len(key) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in key
                    )
                    for key in classes
                )
                or any(
                    evidence.get("continuous_left_rows_sha256") != key
                    or hash_left_observation_table(
                        evidence.get("continuous_left_rows", ())
                    )
                    != key
                    for key, evidence in class_evidence.items()
                )
                or any(
                    not _successful_class_evidence_is_complete(
                        class_evidence[key]
                    )
                    for key in eligible_class_hashes
                )
                or any(
                    not _eligible_candidate_matches_class_evidence(
                        row,
                        class_evidence[
                            str(row["left_observation_sha256"])
                        ],
                    )
                    for row in candidates
                    if all(
                        bool(row.get(name, False))
                        for name in _CANDIDATE_GATE_FIELDS
                    )
                )
                or not class_members.issubset(candidate_ids)
                or selected["candidate_id"] not in class_members
                or any(
                    evidence.candidate_id
                    != self.decision.best_candidate_by_phase_convention[
                        phase
                    ]
                    for phase, evidence in phase_matching.items()
                )
                or recomputed.status != self.decision.status
                or dict(recomputed.selected_candidate) != selected
                or frozen.get("candidate_id") != selected["candidate_id"]
                or frozen.get("method") != "raw_ftan"
                or frozen.get("validation_table_sha256")
                != recomputed_evidence_hash
                or any(
                    frozen.get(name) != selected[name]
                    for name in (
                        "phase_convention",
                        "alpha",
                        "beta1",
                        "beta2",
                        "closure_median_cycles",
                    )
                )
                or any(
                    not isinstance(frozen.get(name), str)
                    or len(frozen[name]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in frozen[name]
                    )
                    for name in required_hashes
                )
            ):
                raise ValueError(
                    "passed StageBRunResult evidence is not cross-consistent"
                )
        object.__setattr__(self, "return_code", code)
        object.__setattr__(self, "candidate_results", candidates)
        object.__setattr__(self, "measurement_classes", classes)
        object.__setattr__(self, "class_evidence", class_evidence)
        object.__setattr__(
            self,
            "phase_matching_diagnostics",
            phase_matching,
        )
        object.__setattr__(self, "audit", audit)


def _stage_b_jsonable(value: object) -> object:
    if is_dataclass(value):
        return {
            field.name: _stage_b_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return _stage_b_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _stage_b_jsonable(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _stage_b_jsonable(item)
            for key, item in sorted(
                value.items(),
                key=lambda row: str(row[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_stage_b_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _pack_full_reference_evidence(
    evidence: object,
) -> object:
    if evidence is None:
        return None
    if not isinstance(evidence, FullReferenceEvidence):
        raise ValueError("class reference evidence has the wrong type")
    return {
        "status": evidence.status,
        "alias_status": evidence.alias_status,
        "lambda_s": evidence.lambda_s,
        "lambda_g": evidence.lambda_g,
        "basin_starts": tuple(
            _stage_b_jsonable(value) for value in evidence.basin_starts
        ),
        "optimizer_calls": evidence.optimizer_calls,
        "corrected_rows": tuple(
            {
                field.name: getattr(row, field.name)
                for field in fields(row)
            }
            for row in evidence.corrected_rows
        ),
        "result_sha256": evidence.result_sha256,
    }


def _unpack_full_reference_evidence(payload: object) -> object:
    if payload is None:
        return None
    data = dict(payload)
    return FullReferenceEvidence(
        status=data["status"],
        alias_status=data["alias_status"],
        lambda_s=data["lambda_s"],
        lambda_g=data["lambda_g"],
        basin_starts=tuple(data["basin_starts"]),
        optimizer_calls=data["optimizer_calls"],
        corrected_rows=tuple(
            CorrectedTargetObservation(**dict(row))
            for row in data["corrected_rows"]
        ),
        result_sha256=data["result_sha256"],
    )


def _pack_triplet_closure_result(result: object) -> object:
    if result is None:
        return None
    if not isinstance(result, TripletClosureResult):
        raise ValueError("class triplet evidence has the wrong type")
    return {
        "accepted": result.accepted,
        "status": result.status,
        "period_summaries": {
            period: {
                field.name: getattr(summary, field.name)
                for field in fields(summary)
            }
            for period, summary in result.period_summaries.items()
        },
        "triplet_rows": tuple(dict(row) for row in result.triplet_rows),
    }


def _unpack_triplet_closure_result(payload: object) -> object:
    if payload is None:
        return None
    data = dict(payload)
    return TripletClosureResult(
        accepted=data["accepted"],
        status=data["status"],
        period_summaries={
            float(period): TripletPeriodSummary(**dict(summary))
            for period, summary in data["period_summaries"].items()
        },
        triplet_rows=tuple(dict(row) for row in data["triplet_rows"]),
    )


def _pack_half_sample_plan(plan: object) -> object:
    if plan is None:
        return None
    if not isinstance(plan, HalfSamplePlan):
        raise ValueError("class split plan has the wrong type")
    return {
        "base_seed": plan.base_seed,
        "plan_sha256": plan.plan_sha256,
        "splits": tuple(
            {
                "split_index": split.split_index,
                "seed": split.seed,
                "a_pair_names": split.a_pair_names,
                "b_pair_names": split.b_pair_names,
                "stratum_half_counts": dict(
                    split.stratum_half_counts
                ),
                "stratum_by_pair": dict(split.stratum_by_pair),
                "snr_field": split.snr_field,
                "odd_stratum_extra_side": (
                    split.odd_stratum_extra_side
                ),
                "membership_sha256": split.membership_sha256,
            }
            for split in plan.splits
        ),
    }


def _unpack_half_sample_plan(payload: object) -> object:
    if payload is None:
        return None
    data = dict(payload)
    restored_splits = []
    for split in data["splits"]:
        restored = dict(split)
        restored_counts = {}
        for key, value in restored["stratum_half_counts"].items():
            decoded = ast.literal_eval(key) if isinstance(key, str) else key
            decoded_tuple = tuple(int(component) for component in decoded)
            if len(decoded_tuple) != 3:
                raise ValueError("persisted half-sample stratum key is invalid")
            restored_counts[decoded_tuple] = tuple(value)
        restored["stratum_half_counts"] = restored_counts
        restored_splits.append(HalfSampleSplit(**restored))
    return HalfSamplePlan(
        splits=tuple(restored_splits),
        base_seed=data["base_seed"],
        plan_sha256=data["plan_sha256"],
    )


def _pack_split_half_fit_evidence(
    evidence: SplitHalfFitEvidence,
) -> Mapping[str, object]:
    if not isinstance(evidence, SplitHalfFitEvidence):
        raise ValueError("split-half evidence has the wrong type")
    return {
        "status": evidence.status,
        "target_velocities_km_s": np.array(
            evidence.target_velocities_km_s,
            dtype=float,
            copy=True,
        ),
        "lambda_s": evidence.lambda_s,
        "lambda_g": evidence.lambda_g,
        "optimizer_calls": evidence.optimizer_calls,
        "cv_optimizer_calls": evidence.cv_optimizer_calls,
        "maxiter": evidence.maxiter,
    }


def _unpack_split_half_fit_evidence(
    payload: Mapping[str, object],
) -> SplitHalfFitEvidence:
    return SplitHalfFitEvidence(**dict(payload))


def _pack_half_sample_stability(result: object) -> object:
    if result is None:
        return None
    if not isinstance(result, HalfSampleStabilityResult):
        raise ValueError("class half-sample result has the wrong type")
    return {
        "accepted": result.accepted,
        "status": result.status,
        "period_summaries": {
            period: {
                field.name: getattr(summary, field.name)
                for field in fields(summary)
            }
            for period, summary in result.period_summaries.items()
        },
    }


def _unpack_half_sample_stability(payload: object) -> object:
    if payload is None:
        return None
    data = dict(payload)
    return HalfSampleStabilityResult(
        accepted=data["accepted"],
        status=data["status"],
        period_summaries={
            float(period): HalfSamplePeriodSummary(**dict(summary))
            for period, summary in data["period_summaries"].items()
        },
    )


def _pack_stage_b_class_result(
    result: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        "candidate_ids": tuple(result["candidate_ids"]),
        "continuous_left_rows": tuple(
            dict(row) for row in result["continuous_left_rows"]
        ),
        "continuous_left_rows_sha256": result[
            "continuous_left_rows_sha256"
        ],
        "reference": _pack_full_reference_evidence(
            result.get("reference")
        ),
        "reference_passes": bool(result["reference_passes"]),
        "alias_passes": bool(result["alias_passes"]),
        "closure": _pack_triplet_closure_result(
            result.get("closure")
        ),
        "triplet_rows_raw_input": tuple(
            dict(row) for row in result["triplet_rows_raw_input"]
        ),
        "triplet_rows_geometry_valid": tuple(
            dict(row) for row in result["triplet_rows_geometry_valid"]
        ),
        "split_plan": _pack_half_sample_plan(
            result.get("split_plan")
        ),
        "split_half_fit_audit": tuple(
            {
                "split_index": row["split_index"],
                "seed": row["seed"],
                "side": row["side"],
                "fit": _pack_split_half_fit_evidence(row["fit"]),
            }
            for row in result["split_half_fit_audit"]
        ),
        "split_half_absolute_differences_km_s": (
            None
            if result["split_half_absolute_differences_km_s"] is None
            else np.array(
                result["split_half_absolute_differences_km_s"],
                dtype=float,
                copy=True,
            )
        ),
        "half_stability": _pack_half_sample_stability(
            result.get("half_stability")
        ),
    }


def _unpack_stage_b_class_result(
    payload: Mapping[str, object],
) -> Dict[str, object]:
    result = dict(payload)
    return {
        "candidate_ids": tuple(result["candidate_ids"]),
        "continuous_left_rows": tuple(
            MappingProxyType(dict(row))
            for row in result["continuous_left_rows"]
        ),
        "continuous_left_rows_sha256": result[
            "continuous_left_rows_sha256"
        ],
        "reference": _unpack_full_reference_evidence(
            result["reference"]
        ),
        "reference_passes": bool(result["reference_passes"]),
        "alias_passes": bool(result["alias_passes"]),
        "closure": _unpack_triplet_closure_result(result["closure"]),
        "triplet_rows_raw_input": tuple(
            MappingProxyType(dict(row))
            for row in result["triplet_rows_raw_input"]
        ),
        "triplet_rows_geometry_valid": tuple(
            MappingProxyType(dict(row))
            for row in result["triplet_rows_geometry_valid"]
        ),
        "split_plan": _unpack_half_sample_plan(result["split_plan"]),
        "split_half_fit_audit": tuple(
            MappingProxyType(
                {
                    "split_index": row["split_index"],
                    "seed": row["seed"],
                    "side": row["side"],
                    "fit": _unpack_split_half_fit_evidence(
                        row["fit"]
                    ),
                }
            )
            for row in result["split_half_fit_audit"]
        ),
        "split_half_absolute_differences_km_s": (
            None
            if result["split_half_absolute_differences_km_s"] is None
            else np.array(
                result["split_half_absolute_differences_km_s"],
                dtype=float,
                copy=True,
            )
        ),
        "half_stability": _unpack_half_sample_stability(
            result["half_stability"]
        ),
    }


def _stage_b_evidence_components(
    *,
    budget: StageBBudgetResult,
    benchmark_evidence: StageBBenchmarkEvidence,
    selection: StageBSelection,
    candidate_results: Sequence[Mapping[str, object]],
    measurement_classes: Mapping[str, Sequence[str]],
    class_evidence: Mapping[str, Mapping[str, object]],
    phase_matching_diagnostics: Mapping[str, PhaseMatchingRunEvidence],
) -> Mapping[str, object]:
    return {
        "budget": _stage_b_jsonable(budget),
        "benchmark_evidence": _stage_b_jsonable(benchmark_evidence),
        "selection": _stage_b_jsonable(selection),
        "candidate_results": _stage_b_jsonable(candidate_results),
        "measurement_classes": _stage_b_jsonable(measurement_classes),
        "class_evidence": _stage_b_jsonable(class_evidence),
        "phase_matching_diagnostics": _stage_b_jsonable(
            phase_matching_diagnostics
        ),
    }


def stage_b_validation_evidence_payload(
    result: StageBRunResult,
) -> Mapping[str, object]:
    if not isinstance(result, StageBRunResult):
        raise ValueError("StageBRunResult is required")
    if not isinstance(result.selection, StageBSelection):
        raise ValueError("Stage B selection evidence is unavailable")
    return _stage_b_evidence_components(
        budget=result.budget,
        benchmark_evidence=result.benchmark_evidence,
        selection=result.selection,
        candidate_results=result.candidate_results,
        measurement_classes=result.measurement_classes,
        class_evidence=result.class_evidence,
        phase_matching_diagnostics=result.phase_matching_diagnostics,
    )


def stage_b_validation_evidence_sha256(
    payload: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def hash_phase_matching_second_pass_output(result: object) -> str:
    """Hash the actual cleaned waveform and second FTAN filter-bank arrays."""

    try:
        cleaning = result.cleaning
        filter_bank = result.second_pass_filter_bank
        arrays = (
            cleaning.compressed_waveform,
            cleaning.compressed_envelope,
            cleaning.cleaning_window,
            cleaning.cleaned_compressed_waveform,
            cleaning.cleaned_waveform,
            filter_bank.filtered_waveforms,
            filter_bank.analytic_signals,
            filter_bank.envelope,
        )
        metadata = (
            int(cleaning.cut_center_index),
            float(cleaning.cut_center_time_s),
            float(cleaning.cut_half_width_s),
            float(cleaning.cut_taper_alpha),
            float(cleaning.first_pass_alpha),
            float(cleaning.second_pass_alpha),
            float(cleaning.reference_group_time_s),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "actual PhaseMatchedSecondPassResult output is required"
        ) from exc
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        if array.size == 0 or np.any(~np.isfinite(array)):
            raise ValueError(
                "phase-matched second-pass arrays must be finite and non-empty"
            )
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(
            np.asarray(array.shape, dtype=np.int64).tobytes(order="C")
        )
        digest.update(array.tobytes(order="C"))
    digest.update(
        np.asarray(metadata, dtype=np.float64).tobytes(order="C")
    )
    return digest.hexdigest()


def hash_phase_matching_execution_hashes(
    execution_hashes: Sequence[str],
) -> str:
    """Hash an ordered, non-empty set of actual second-pass executions."""

    hashes = tuple(str(value) for value in execution_hashes)
    if not hashes or any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise ValueError("phase-matching execution hashes must be SHA-256 hex")
    return hashlib.sha256(
        json.dumps(
            hashes,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


_CANDIDATE_GATE_FIELDS = (
    "synthetic_passes",
    "ridge_passes",
    "instantaneous_period_passes",
    "alias_passes",
    "triplet_passes",
    "half_sample_passes",
    "boundary_passes",
)


def _successful_class_evidence_is_complete(
    evidence: Mapping[str, object],
) -> bool:
    """Return whether one eligible measurement class has all frozen gates."""

    reference = evidence.get("reference")
    closure = evidence.get("closure")
    split_plan = evidence.get("split_plan")
    half_stability = evidence.get("half_stability")
    audit = tuple(evidence.get("split_half_fit_audit", ()))
    differences = evidence.get(
        "split_half_absolute_differences_km_s"
    )
    if (
        not isinstance(reference, FullReferenceEvidence)
        or reference.status != "accepted"
        or reference.alias_status != "accepted"
        or evidence.get("reference_passes") is not True
        or evidence.get("alias_passes") is not True
        or not isinstance(closure, TripletClosureResult)
        or not closure.accepted
        or not isinstance(split_plan, HalfSamplePlan)
        or len(split_plan.splits) != 20
        or not isinstance(half_stability, HalfSampleStabilityResult)
        or not half_stability.accepted
        or len(audit) != 40
    ):
        return False
    expected_fit_keys = {
        (split.split_index, side)
        for split in split_plan.splits
        for side in ("A", "B")
    }
    actual_fit_keys = set()
    for row in audit:
        if not isinstance(row, Mapping):
            return False
        fit = row.get("fit")
        key = (row.get("split_index"), row.get("side"))
        if (
            key in actual_fit_keys
            or key not in expected_fit_keys
            or row.get("seed")
            != split_plan.splits[int(key[0])].seed
            or not isinstance(fit, SplitHalfFitEvidence)
            or fit.status != "accepted"
            or fit.lambda_s != reference.lambda_s
            or fit.lambda_g != reference.lambda_g
        ):
            return False
        actual_fit_keys.add(key)
    try:
        difference_array = np.asarray(differences, dtype=float)
    except (TypeError, ValueError):
        return False
    if (
        actual_fit_keys != expected_fit_keys
        or difference_array.shape
        != (20, len(WANG_TARGET_PERIODS_S))
        or np.any(~np.isfinite(difference_array))
    ):
        return False
    try:
        recomputed_closure = evaluate_triplet_closure(
            evidence.get("triplet_rows_geometry_valid", ()),
            target_periods_s=WANG_TARGET_PERIODS_S,
        )
        recomputed_half = evaluate_half_sample_stability(
            difference_array,
            target_periods_s=WANG_TARGET_PERIODS_S,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        _stage_b_jsonable(recomputed_closure)
        == _stage_b_jsonable(closure)
        and _stage_b_jsonable(recomputed_half)
        == _stage_b_jsonable(half_stability)
    )


def _eligible_candidate_matches_class_evidence(
    candidate: Mapping[str, object],
    evidence: Mapping[str, object],
) -> bool:
    """Bind candidate gate booleans and score to recomputed class evidence."""

    reference = evidence.get("reference")
    closure = evidence.get("closure")
    half = evidence.get("half_stability")
    if (
        not isinstance(reference, FullReferenceEvidence)
        or not isinstance(closure, TripletClosureResult)
        or not isinstance(half, HalfSampleStabilityResult)
    ):
        return False
    closure_values = tuple(
        summary.median_absolute_cycles
        for summary in closure.period_summaries.values()
    )
    if len(closure_values) != len(WANG_TARGET_PERIODS_S):
        return False
    try:
        score = float(candidate["closure_median_cycles"])
        boundary = evaluate_candidate_boundary_fraction(
            accepted_measurement_count=int(
                candidate["accepted_measurement_count"]
            ),
            accepted_outermost_velocity_cell_count=int(
                candidate[
                    "accepted_outermost_velocity_cell_count"
                ]
            ),
        )
    except (KeyError, TypeError, ValueError):
        return False
    expected_score = float(np.median(closure_values))
    return bool(
        reference.status == "accepted"
        and reference.alias_status == "accepted"
        and closure.accepted
        and half.accepted
        and np.isclose(
            score,
            expected_score,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        )
        and bool(candidate.get("alias_passes")) == (
            reference.alias_status == "accepted"
        )
        and bool(candidate.get("triplet_passes")) == closure.accepted
        and bool(candidate.get("half_sample_passes")) == half.accepted
        and bool(candidate.get("boundary_passes")) == boundary.accepted
        and str(candidate.get("boundary_status")) == boundary.status
        and np.isclose(
            float(candidate.get("accepted_boundary_fraction", np.nan)),
            boundary.accepted_boundary_fraction,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        )
    )


def evaluate_stage_b_budget(
    *,
    candidate_benchmark_elapsed_s: float,
    candidate_benchmark_work_units: float,
    stage_b_candidate_work_units: float,
    reference_benchmark_elapsed_s: float,
    reference_benchmark_optimizer_calls: int,
    distinct_measurement_class_count: int,
    worker_count: int,
    measured_peak_memory_bytes: int,
    available_memory_bytes: int,
) -> StageBBudgetResult:
    """Project the complete Stage B workload without reducing its science grid."""

    scalar_values = {
        "candidate_benchmark_elapsed_s": candidate_benchmark_elapsed_s,
        "candidate_benchmark_work_units": candidate_benchmark_work_units,
        "stage_b_candidate_work_units": stage_b_candidate_work_units,
        "reference_benchmark_elapsed_s": reference_benchmark_elapsed_s,
    }
    normalized = {name: float(value) for name, value in scalar_values.items()}
    if any(
        not np.isfinite(value) or value <= 0
        for value in normalized.values()
    ):
        raise ValueError("Stage B budget scalar inputs must be positive finite")
    integer_values = {
        "reference_benchmark_optimizer_calls": (
            reference_benchmark_optimizer_calls
        ),
        "distinct_measurement_class_count": distinct_measurement_class_count,
        "worker_count": worker_count,
        "measured_peak_memory_bytes": measured_peak_memory_bytes,
        "available_memory_bytes": available_memory_bytes,
    }
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in integer_values.values()
    ):
        raise ValueError("Stage B budget count inputs must be positive integers")
    classes = int(distinct_measurement_class_count)
    workers = int(worker_count)
    if classes > 300:
        raise ValueError("distinct measurement classes cannot exceed 300")
    if workers > 24:
        raise ValueError("Stage B worker count cannot exceed 24")
    calls_per_class = STAGE_B_OPTIMIZER_CALLS_PER_CLASS
    candidate_seconds = (
        normalized["candidate_benchmark_elapsed_s"]
        * normalized["stage_b_candidate_work_units"]
        / normalized["candidate_benchmark_work_units"]
    )
    reference_process_waves = (classes + workers - 1) // workers
    reference_seconds = (
        normalized["reference_benchmark_elapsed_s"]
        / int(reference_benchmark_optimizer_calls)
        * calls_per_class
        * reference_process_waves
    )
    total_seconds = candidate_seconds + reference_seconds
    memory_fraction = (
        int(measured_peak_memory_bytes) / int(available_memory_bytes)
    )
    accepted = (
        total_seconds <= STAGE_B_MAXIMUM_WALL_SECONDS
        and memory_fraction <= STAGE_B_MAXIMUM_MEMORY_FRACTION
    )
    return StageBBudgetResult(
        accepted=accepted,
        status="accepted" if accepted else "stage_b_budget_exceeded",
        projected_candidate_seconds=candidate_seconds,
        projected_reference_seconds=reference_seconds,
        projected_total_seconds=total_seconds,
        memory_fraction=memory_fraction,
        optimizer_calls_per_class=calls_per_class,
        candidate_projection_formula=(
            "candidate_benchmark_elapsed_s * "
            "stage_b_candidate_work_units / candidate_benchmark_work_units"
        ),
        reference_projection_formula=(
            "reference_benchmark_elapsed_s / "
            "reference_benchmark_optimizer_calls * "
            f"{calls_per_class} * "
            "ceil(distinct_measurement_class_count / worker_count)"
        ),
    )


def project_stage_b_budget_from_fixed_benchmark(
    evidence: StageBBenchmarkEvidence,
    *,
    selected_pair_count: int,
    maximum_measurement_class_count: int,
    worker_count: int,
) -> StageBBudgetResult:
    """Project Stage B only from the fixed 20-waveform/335-fit benchmark."""

    if not isinstance(evidence, StageBBenchmarkEvidence):
        raise ValueError("fixed Stage B benchmark evidence is required")
    reference_elapsed = (
        evidence.ten_single_reference_fits_elapsed_s
        + evidence.lambda_cv_elapsed_s
        + evidence.twenty_half_samples_elapsed_s
    )
    return evaluate_stage_b_budget(
        candidate_benchmark_elapsed_s=evidence.candidate_grid_elapsed_s,
        candidate_benchmark_work_units=(
            STAGE_B_BENCHMARK_WAVEFORM_COUNT
        ),
        stage_b_candidate_work_units=selected_pair_count,
        reference_benchmark_elapsed_s=reference_elapsed,
        reference_benchmark_optimizer_calls=(
            STAGE_B_BENCHMARK_OPTIMIZER_CALLS
        ),
        distinct_measurement_class_count=(
            maximum_measurement_class_count
        ),
        worker_count=worker_count,
        measured_peak_memory_bytes=(
            evidence.measured_peak_memory_bytes
        ),
        available_memory_bytes=evidence.available_memory_bytes,
    )


def build_candidate_grid(
    *,
    phase_conventions: Sequence[str],
    alpha_candidates: Sequence[float],
    beta1_candidates: Sequence[float],
    beta2_candidates: Sequence[float],
) -> Tuple[Mapping[str, object], ...]:
    """Return the complete deterministic FTAN candidate Cartesian product."""

    phases = tuple(str(value) for value in phase_conventions)
    alpha = tuple(float(value) for value in alpha_candidates)
    beta1 = tuple(float(value) for value in beta1_candidates)
    beta2 = tuple(float(value) for value in beta2_candidates)
    if (
        not phases
        or len(set(phases)) != len(phases)
        or any(not value for value in phases)
        or not alpha
        or not beta1
        or not beta2
        or any(
            not np.isfinite(value) or value < 0
            for values in (alpha, beta1, beta2)
            for value in values
        )
        or len(set(alpha)) != len(alpha)
        or len(set(beta1)) != len(beta1)
        or len(set(beta2)) != len(beta2)
    ):
        raise ValueError("candidate grid axes are invalid")
    candidate_count = len(phases) * len(alpha) * len(beta1) * len(beta2)
    if candidate_count > 300:
        raise ValueError("candidate grid cannot exceed 300 configurations")
    rows = []
    for phase in phases:
        for alpha_value in alpha:
            for beta1_value in beta1:
                for beta2_value in beta2:
                    candidate_id = (
                        f"{phase}__alpha_{alpha_value:g}"
                        f"__beta1_{beta1_value:g}"
                        f"__beta2_{beta2_value:g}"
                    )
                    rows.append(
                        MappingProxyType(
                            {
                                "candidate_id": candidate_id,
                                "phase_convention": phase,
                                "alpha": alpha_value,
                                "beta1": beta1_value,
                                "beta2": beta2_value,
                            }
                        )
                    )
    return tuple(rows)


def _normalize_candidate_row(
    source: Mapping[str, object],
) -> Dict[str, object]:
    row = dict(source)
    required = (
        "candidate_id",
        "phase_convention",
        "alpha",
        "beta1",
        "beta2",
        "closure_median_cycles",
        *_CANDIDATE_GATE_FIELDS,
    )
    if any(name not in row for name in required):
        raise ValueError("candidate row is missing a frozen decision field")
    row["candidate_id"] = str(row["candidate_id"])
    row["phase_convention"] = str(row["phase_convention"])
    for name in ("alpha", "beta1", "beta2", "closure_median_cycles"):
        row[name] = float(row[name])
    if (
        not row["candidate_id"]
        or not row["phase_convention"]
        or any(
            not np.isfinite(row[name]) or row[name] < 0
            for name in ("alpha", "beta1", "beta2")
        )
        or not np.isfinite(row["closure_median_cycles"])
        or row["closure_median_cycles"] < 0
        or any(
            not isinstance(row[name], (bool, np.bool_))
            for name in _CANDIDATE_GATE_FIELDS
        )
    ):
        raise ValueError("candidate row contains invalid decision values")
    for name in _CANDIDATE_GATE_FIELDS:
        row[name] = bool(row[name])
    return row


def _strictly_within_relative_tie(
    value: float,
    best_value: float,
    relative_tie_fraction: float,
) -> bool:
    if best_value == 0:
        return value == 0
    relative_difference = (value - best_value) / best_value
    tolerance = 64.0 * np.finfo(float).eps
    return (
        relative_difference < relative_tie_fraction
        and not np.isclose(
            relative_difference,
            relative_tie_fraction,
            rtol=tolerance,
            atol=tolerance,
        )
    )


def freeze_ftan_candidate(
    candidate_rows: Iterable[Mapping[str, object]],
    *,
    lineage_status: str,
    lineage_preferred_phase_convention: object,
    relative_tie_fraction: float = CANDIDATE_RELATIVE_TIE_FRACTION,
) -> CandidateFreezeDecision:
    """Apply hard gates, simplicity ties, and physical-lineage phase choice."""

    rows = [_normalize_candidate_row(source) for source in candidate_rows]
    if (
        not rows
        or len(rows) > 300
        or len({row["candidate_id"] for row in rows}) != len(rows)
    ):
        raise ValueError("candidate rows must be non-empty with unique IDs")
    lineage = str(lineage_status)
    preferred = (
        None
        if lineage_preferred_phase_convention is None
        else str(lineage_preferred_phase_convention)
    )
    tie_fraction = float(relative_tie_fraction)
    if (
        not lineage
        or not np.isfinite(tie_fraction)
        or tie_fraction < 0
        or tie_fraction > 1
    ):
        raise ValueError("candidate freeze metadata is invalid")
    eligible = [
        row
        for row in rows
        if all(row[name] for name in _CANDIDATE_GATE_FIELDS)
    ]
    eligible_ids = tuple(sorted(row["candidate_id"] for row in eligible))
    if not eligible:
        status = (
            "insufficient_triplet_support"
            if all(not row["triplet_passes"] for row in rows)
            else "no_candidate_passed"
        )
        return CandidateFreezeDecision(
            accepted=False,
            status=status,
            selected_candidate={},
            eligible_candidate_ids=(),
            best_candidate_by_phase_convention={},
            lineage_status=lineage,
            lineage_preferred_phase_convention=preferred,
        )

    by_phase: Dict[str, list] = {}
    for row in eligible:
        by_phase.setdefault(row["phase_convention"], []).append(row)
    phase_winners = {}
    for phase, phase_rows in sorted(by_phase.items()):
        best_score = min(row["closure_median_cycles"] for row in phase_rows)
        within_tie = [
            row
            for row in phase_rows
            if row["closure_median_cycles"] == best_score
            or _strictly_within_relative_tie(
                row["closure_median_cycles"],
                best_score,
                tie_fraction,
            )
        ]
        phase_winners[phase] = min(
            within_tie,
            key=lambda row: (
                row["beta2"],
                row["beta1"],
                row["alpha"],
                row["candidate_id"],
            ),
        )
    best_ids = {
        phase: row["candidate_id"]
        for phase, row in phase_winners.items()
    }
    if len(phase_winners) == 1:
        selected = next(iter(phase_winners.values()))
    else:
        ordered = sorted(
            phase_winners.values(),
            key=lambda row: (
                row["closure_median_cycles"],
                row["phase_convention"],
            ),
        )
        lowest = ordered[0]["closure_median_cycles"]
        tied_phases = [
            row
            for row in ordered
            if row["closure_median_cycles"] == lowest
            or _strictly_within_relative_tie(
                row["closure_median_cycles"],
                lowest,
                tie_fraction,
            )
        ]
        if len(tied_phases) == 1:
            selected = tied_phases[0]
        elif (
            lineage.lower() in ("confirmed", "known")
            and preferred in phase_winners
        ):
            selected = phase_winners[preferred]
        else:
            return CandidateFreezeDecision(
                accepted=False,
                status="phase_convention_unidentifiable",
                selected_candidate={},
                eligible_candidate_ids=eligible_ids,
                best_candidate_by_phase_convention=best_ids,
                lineage_status=lineage,
                lineage_preferred_phase_convention=preferred,
            )
    return CandidateFreezeDecision(
        accepted=True,
        status="passed",
        selected_candidate=selected,
        eligible_candidate_ids=eligible_ids,
        best_candidate_by_phase_convention=best_ids,
        lineage_status=lineage,
        lineage_preferred_phase_convention=preferred,
    )


def build_frozen_parameters_manifest(
    decision: CandidateFreezeDecision,
    *,
    input_inventory_sha256: str,
    code_sha256: str,
    config_sha256: str,
    validation_table_sha256: str,
) -> Mapping[str, object]:
    """Build the Stage B success manifest without writing a false success."""

    if not isinstance(decision, CandidateFreezeDecision) or not decision.accepted:
        raise ValueError("frozen parameters require a passed decision")
    hashes = {
        "input_inventory_sha256": input_inventory_sha256,
        "code_sha256": code_sha256,
        "config_sha256": config_sha256,
        "validation_table_sha256": validation_table_sha256,
    }
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes.values()
    ):
        raise ValueError("frozen parameter lineage hashes must be SHA-256 hex")
    selected = decision.selected_candidate
    manifest = {
        "stage_b_status": "passed",
        "method": "raw_ftan",
        "candidate_id": selected["candidate_id"],
        "phase_convention": selected["phase_convention"],
        "alpha": selected["alpha"],
        "beta1": selected["beta1"],
        "beta2": selected["beta2"],
        "closure_median_cycles": selected["closure_median_cycles"],
        "lineage_status": decision.lineage_status,
        "lineage_preferred_phase_convention": (
            decision.lineage_preferred_phase_convention
        ),
        **hashes,
    }
    return MappingProxyType(manifest)


def evaluate_candidate_boundary_fraction(
    *,
    accepted_measurement_count: int,
    accepted_outermost_velocity_cell_count: int,
    maximum_boundary_fraction: float = CANDIDATE_MAXIMUM_BOUNDARY_FRACTION,
) -> CandidateBoundaryResult:
    """Apply the Stage B aggregate boundary-cell hard gate."""

    if (
        isinstance(accepted_measurement_count, (bool, np.bool_))
        or not isinstance(accepted_measurement_count, (int, np.integer))
        or int(accepted_measurement_count) <= 0
    ):
        raise ValueError("accepted measurement count must be a positive integer")
    if (
        isinstance(
            accepted_outermost_velocity_cell_count,
            (bool, np.bool_),
        )
        or not isinstance(
            accepted_outermost_velocity_cell_count,
            (int, np.integer),
        )
        or int(accepted_outermost_velocity_cell_count) < 0
        or int(accepted_outermost_velocity_cell_count)
        > int(accepted_measurement_count)
    ):
        raise ValueError(
            "accepted outermost velocity cell count must lie within "
            "the accepted measurement count"
        )
    maximum = float(maximum_boundary_fraction)
    if not np.isfinite(maximum) or maximum < 0 or maximum > 1:
        raise ValueError("maximum boundary fraction must lie in [0, 1]")
    count = int(accepted_measurement_count)
    boundary_count = int(accepted_outermost_velocity_cell_count)
    fraction = boundary_count / count
    accepted = fraction <= maximum
    return CandidateBoundaryResult(
        accepted_measurement_count=count,
        accepted_outermost_velocity_cell_count=boundary_count,
        accepted_boundary_fraction=fraction,
        maximum_boundary_fraction=maximum,
        accepted=accepted,
        status=(
            "accepted"
            if accepted
            else "candidate_boundary_fraction_exceeded"
        ),
    )


def stage_b_candidate_input_integrity_passes(
    measurement: Mapping[str, object],
    *,
    selected_pair_count: int,
) -> bool:
    """Check per-candidate pair conservation and the 1% exception gate."""

    selected = int(selected_pair_count)
    names = (
        "processed_pair_count",
        "successful_pair_count",
        "expected_scientific_rejection_count",
        "unexpected_pair_exception_count",
    )
    try:
        counts = {
            name: int(measurement[name])
            for name in names
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Stage B candidate input conservation counts are incomplete"
        ) from exc
    if (
        selected <= 0
        or any(value < 0 for value in counts.values())
        or counts["processed_pair_count"] != selected
        or counts["processed_pair_count"]
        != counts["successful_pair_count"]
        + counts["expected_scientific_rejection_count"]
        + counts["unexpected_pair_exception_count"]
    ):
        raise ValueError("Stage B candidate pair-count conservation failed")
    return bool(
        counts["unexpected_pair_exception_count"] / selected <= 0.01
    )


def evaluate_phase_matching_diagnostic(
    *,
    raw_closure_median_cycles: Mapping[float, float],
    matched_closure_median_cycles: Mapping[float, float],
    raw_valid_ridge_coverage: float,
    matched_valid_ridge_coverage: float,
    raw_phase_convention: str,
    matched_phase_convention: str,
    raw_boundary_fraction: float,
    matched_boundary_fraction: float,
    narrowband_sidelobe_validation_passed: bool,
    minimum_closure_reduction_fraction: float = (
        PHASE_MATCHING_MINIMUM_CLOSURE_REDUCTION_FRACTION
    ),
) -> PhaseMatchingDiagnosticResult:
    """Decide whether phase matching merits a separately reviewed redesign.

    A positive diagnostic deliberately stops Stage B for a design revision;
    it never promotes phase matching inside the same validation run.
    """

    raw = {
        float(period): float(value)
        for period, value in raw_closure_median_cycles.items()
    }
    matched = {
        float(period): float(value)
        for period, value in matched_closure_median_cycles.items()
    }
    if (
        not raw
        or set(raw) != set(matched)
        or tuple(sorted(raw)) != WANG_TARGET_PERIODS_S
        or any(
            not np.isfinite(value) or value <= 0
            for value in raw.values()
        )
        or any(
            not np.isfinite(value) or value < 0
            for value in matched.values()
        )
    ):
        raise ValueError(
            "phase matching requires exactly the four target periods "
            "3.0, 3.5, 4.0 and 5.0 s"
        )
    coverage_raw = float(raw_valid_ridge_coverage)
    coverage_matched = float(matched_valid_ridge_coverage)
    boundary_raw = float(raw_boundary_fraction)
    boundary_matched = float(matched_boundary_fraction)
    minimum_reduction = float(minimum_closure_reduction_fraction)
    if (
        any(
            not np.isfinite(value) or value < 0 or value > 1
            for value in (
                coverage_raw,
                coverage_matched,
                boundary_raw,
                boundary_matched,
                minimum_reduction,
            )
        )
        or not isinstance(raw_phase_convention, str)
        or not raw_phase_convention
        or not isinstance(matched_phase_convention, str)
        or not matched_phase_convention
        or not isinstance(
            narrowband_sidelobe_validation_passed,
            (bool, np.bool_),
        )
    ):
        raise ValueError("phase-matching diagnostic metadata is invalid")
    reductions = {
        period: (raw[period] - matched[period]) / raw[period]
        for period in sorted(raw)
    }
    reduction_passes = {
        period: value >= minimum_reduction
        for period, value in reductions.items()
    }
    coverage_increased = coverage_matched > coverage_raw
    convention_unchanged = (
        matched_phase_convention == raw_phase_convention
    )
    boundary_not_worse = boundary_matched <= boundary_raw
    revision_required = (
        all(reduction_passes.values())
        and coverage_increased
        and convention_unchanged
        and boundary_not_worse
        and bool(narrowband_sidelobe_validation_passed)
    )
    return PhaseMatchingDiagnosticResult(
        design_revision_required=revision_required,
        freeze_raw_ftan=not revision_required,
        status=(
            "phase_matching_design_revision_required"
            if revision_required
            else "raw_ftan_frozen"
        ),
        closure_reduction_fraction_by_period=reductions,
        period_reduction_passes=reduction_passes,
        valid_ridge_coverage_increased=coverage_increased,
        phase_convention_unchanged=convention_unchanged,
        boundary_fraction_not_worse=boundary_not_worse,
        narrowband_sidelobe_validation_passed=(
            narrowband_sidelobe_validation_passed
        ),
    )


def _quantile(values: np.ndarray, probabilities: Sequence[float]) -> np.ndarray:
    return np.asarray(
        np.quantile(
            values,
            np.asarray(probabilities, dtype=float),
            method="linear",
        ),
        dtype=float,
    )


def _station_xy_km(
    station: Sequence[float],
    *,
    reference_lon: float,
    reference_lat: float,
) -> np.ndarray:
    if len(station) != 2:
        raise ValueError("station coordinates must be (longitude, latitude)")
    longitude, latitude = (float(value) for value in station)
    if not (np.isfinite(longitude) and np.isfinite(latitude)):
        raise ValueError("station coordinates must be finite")
    return np.asarray(
        [
            (longitude - reference_lon)
            * EARTH_KM_PER_DEGREE
            * np.cos(np.deg2rad(reference_lat)),
            (latitude - reference_lat) * EARTH_KM_PER_DEGREE,
        ],
        dtype=float,
    )


def evaluate_triplet_geometry(
    *,
    station_a: Sequence[float],
    station_b: Sequence[float],
    station_c: Sequence[float],
    maximum_cross_track_km: float = 0.5,
    maximum_distance_closure_error_km: float = 0.5,
) -> TripletGeometryResult:
    """Evaluate whether B lies sufficiently close to the A-C segment."""

    reference_lon = float(station_a[0])
    reference_lat = float(station_a[1])
    a = _station_xy_km(
        station_a,
        reference_lon=reference_lon,
        reference_lat=reference_lat,
    )
    b = _station_xy_km(
        station_b,
        reference_lon=reference_lon,
        reference_lat=reference_lat,
    )
    c = _station_xy_km(
        station_c,
        reference_lon=reference_lon,
        reference_lat=reference_lat,
    )
    ac = c - a
    ac_length = float(np.linalg.norm(ac))
    if ac_length <= 0:
        raise ValueError("stations A and C must be distinct")
    projection = float(np.dot(b - a, ac) / np.dot(ac, ac))
    ab = b - a
    cross_track = float(
        abs(ac[0] * ab[1] - ac[1] * ab[0]) / ac_length
    )
    distance_ab = float(np.linalg.norm(b - a))
    distance_bc = float(np.linalg.norm(c - b))
    distance_error = abs(distance_ab + distance_bc - ac_length)
    if projection < 0.0 or projection > 1.0:
        status = "station_b_outside_ac"
    elif cross_track > float(maximum_cross_track_km):
        status = "cross_track_too_large"
    elif distance_error > float(maximum_distance_closure_error_km):
        status = "distance_closure_too_large"
    else:
        status = "accepted"
    return TripletGeometryResult(
        accepted=status == "accepted",
        status=status,
        projection_fraction=projection,
        cross_track_km=cross_track,
        distance_ab_km=distance_ab,
        distance_bc_km=distance_bc,
        distance_ac_km=ac_length,
        distance_closure_error_km=distance_error,
    )


def _closure_residual(
    *,
    distance_ab_km: float,
    distance_bc_km: float,
    distance_ac_km: float,
    time_ab_s: float,
    time_bc_s: float,
    time_ac_s: float,
) -> float:
    denominator = distance_ab_km + distance_bc_km
    if denominator <= 0:
        raise ValueError("triplet AB+BC distance must be positive")
    return float(
        distance_ac_km * (time_ab_s + time_bc_s) / denominator
        - time_ac_s
    )


def evaluate_triplet_closure(
    triplet_rows: Iterable[Mapping[str, object]],
    *,
    target_periods_s: Iterable[float],
    minimum_support: int = TRIPLET_MINIMUM_SUPPORT,
    maximum_median_absolute_cycles: float = (
        TRIPLET_MAXIMUM_MEDIAN_ABSOLUTE_CYCLES
    ),
    maximum_absolute_bias_cycles: float = (
        TRIPLET_MAXIMUM_ABSOLUTE_BIAS_CYCLES
    ),
) -> TripletClosureResult:
    """Compute raw diagnostics and corrected candidate-freezing closure gates."""

    targets = tuple(sorted(float(value) for value in target_periods_s))
    support_gate = int(minimum_support)
    median_gate = float(maximum_median_absolute_cycles)
    bias_gate = float(maximum_absolute_bias_cycles)
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(not np.isfinite(value) or value <= 0 for value in targets)
        or support_gate <= 0
        or not np.isfinite(median_gate)
        or median_gate < 0
        or not np.isfinite(bias_gate)
        or bias_gate < 0
    ):
        raise ValueError("triplet closure configuration is invalid")
    accepted_rows = []
    observed_triplet_periods = set()
    for source in triplet_rows:
        row = dict(source)
        period = float(row["period_s"])
        if not np.isfinite(period) or period <= 0:
            raise ValueError("triplet period must be positive finite")
        if period not in targets:
            continue
        triplet_id = str(row["triplet_id"])
        if not triplet_id:
            raise ValueError("triplet_id must be non-empty")
        triplet_period_key = (triplet_id, period)
        if triplet_period_key in observed_triplet_periods:
            raise ValueError(
                f"duplicate triplet-period row: {triplet_id}, {period:g} s"
            )
        observed_triplet_periods.add(triplet_period_key)
        left_values = tuple(
            row[name] for name in ("left_ab", "left_bc", "left_ac")
        )
        if any(
            not isinstance(value, (bool, np.bool_))
            for value in left_values
        ):
            raise ValueError("triplet LEFT flags must be boolean")
        if not all(left_values):
            continue
        snr_values = tuple(
            float(row[name]) for name in ("snr_ab", "snr_bc", "snr_ac")
        )
        if any(not np.isfinite(value) for value in snr_values):
            continue
        if min(snr_values) <= 8.0:
            continue
        distances = tuple(
            float(row[name])
            for name in (
                "distance_ab_km",
                "distance_bc_km",
                "distance_ac_km",
            )
        )
        if (
            any(not np.isfinite(value) or value <= 0 for value in distances)
            or abs(distances[0] + distances[1] - distances[2]) > 0.5
        ):
            continue
        raw_times = tuple(
            float(row[name])
            for name in ("raw_time_ab_s", "raw_time_bc_s", "raw_time_ac_s")
        )
        corrected_times = tuple(
            float(row[name])
            for name in (
                "corrected_time_ab_s",
                "corrected_time_bc_s",
                "corrected_time_ac_s",
            )
        )
        if any(
            not np.isfinite(value)
            for value in raw_times + corrected_times
        ):
            continue
        raw_residual = _closure_residual(
            distance_ab_km=distances[0],
            distance_bc_km=distances[1],
            distance_ac_km=distances[2],
            time_ab_s=raw_times[0],
            time_bc_s=raw_times[1],
            time_ac_s=raw_times[2],
        )
        corrected_residual = _closure_residual(
            distance_ab_km=distances[0],
            distance_bc_km=distances[1],
            distance_ac_km=distances[2],
            time_ab_s=corrected_times[0],
            time_bc_s=corrected_times[1],
            time_ac_s=corrected_times[2],
        )
        accepted_rows.append(
            {
                **row,
                "raw_closure_residual_s": raw_residual,
                "corrected_closure_residual_s": corrected_residual,
                "corrected_closure_residual_cycles": (
                    corrected_residual / period
                ),
            }
        )
    summaries = {}
    for period in targets:
        residuals = np.asarray(
            [
                row["corrected_closure_residual_s"]
                for row in accepted_rows
                if float(row["period_s"]) == period
            ],
            dtype=float,
        )
        support = int(residuals.size)
        median_absolute = (
            float(np.median(np.abs(residuals)) / period)
            if support
            else float("nan")
        )
        absolute_bias = (
            float(abs(np.median(residuals)) / period)
            if support
            else float("nan")
        )
        if support < support_gate:
            status = "insufficient_triplet_support"
        elif median_absolute > median_gate:
            status = "triplet_median_too_large"
        elif absolute_bias > bias_gate:
            status = "triplet_bias_too_large"
        else:
            status = "accepted"
        summaries[period] = TripletPeriodSummary(
            period_s=period,
            support_count=support,
            median_absolute_cycles=median_absolute,
            absolute_bias_cycles=absolute_bias,
            accepted=status == "accepted",
            status=status,
        )
    accepted = all(summary.accepted for summary in summaries.values())
    return TripletClosureResult(
        accepted=accepted,
        status="accepted" if accepted else "triplet_closure_failed",
        period_summaries=summaries,
        triplet_rows=tuple(accepted_rows),
    )


def _stratify_pair_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    snr_field: str,
) -> Tuple[
    Mapping[str, Tuple[int, int, int]],
    np.ndarray,
    np.ndarray,
]:
    ordered = tuple(
        sorted((dict(row) for row in rows), key=lambda row: str(row["pair_name"]))
    )
    names = tuple(str(row["pair_name"]) for row in ordered)
    if (
        not ordered
        or any(not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("stratification requires unique non-empty pair names")
    distance = np.asarray(
        [float(row["distance_km"]) for row in ordered],
        dtype=float,
    )
    azimuth = np.asarray(
        [float(row["azimuth_deg"]) for row in ordered],
        dtype=float,
    )
    snr = np.asarray([float(row[snr_field]) for row in ordered], dtype=float)
    if (
        np.any(~np.isfinite(distance))
        or np.any(distance <= 0)
        or np.any(~np.isfinite(azimuth))
        or np.any(np.isinf(snr))
    ):
        raise ValueError("pair stratification values are invalid")
    distance_edges = _quantile(distance, DISTANCE_QUANTILE_PROBABILITIES)
    finite_snr = snr[np.isfinite(snr)]
    snr_edges = (
        _quantile(finite_snr, SNR_QUANTILE_PROBABILITIES)
        if finite_snr.size
        else np.zeros(2, dtype=float)
    )
    distance_bins = np.searchsorted(
        distance_edges,
        distance,
        side="right",
    )
    azimuth_bins = np.floor(
        np.mod(azimuth, 360.0) / AZIMUTH_SECTOR_WIDTH_DEG
    ).astype(int)
    snr_bins = np.full(snr.size, -1, dtype=int)
    snr_bins[np.isfinite(snr)] = np.searchsorted(
        snr_edges,
        snr[np.isfinite(snr)],
        side="right",
    )
    return (
        {
            name: (
                int(distance_bins[index]),
                int(azimuth_bins[index]),
                int(snr_bins[index]),
            )
            for index, name in enumerate(names)
        },
        distance_edges,
        snr_edges,
    )


def build_half_sample_splits(
    pair_rows: Iterable[Mapping[str, object]],
    *,
    split_count: int = 20,
    base_seed: int = 20260717,
) -> HalfSamplePlan:
    """Build deterministic A/B halves within every frozen Stage B stratum."""

    rows = tuple(dict(row) for row in pair_rows)
    number_of_splits = int(split_count)
    first_seed = int(base_seed)
    if (
        not rows
        or number_of_splits != 20
        or first_seed != 20260717
    ):
        raise ValueError("half-sample split inputs are invalid")
    stratum_by_pair, distance_edges, snr_edges = _stratify_pair_rows(
        rows,
        snr_field="candidate_left_snr",
    )
    members_by_stratum: Dict[Tuple[int, int, int], list] = {}
    for name, key in stratum_by_pair.items():
        members_by_stratum.setdefault(key, []).append(name)
    splits = []
    for split_index in range(number_of_splits):
        seed = first_seed + split_index
        generator = np.random.default_rng(seed)
        extra_side = "A" if split_index % 2 == 0 else "B"
        a_names = []
        b_names = []
        half_counts = {}
        for key in sorted(members_by_stratum):
            members = np.asarray(
                sorted(members_by_stratum[key]),
                dtype=object,
            )
            shuffled = members[generator.permutation(members.size)]
            if members.size % 2 == 0:
                a_count = members.size // 2
            elif extra_side == "A":
                a_count = members.size // 2 + 1
            else:
                a_count = members.size // 2
            a_part = tuple(str(value) for value in shuffled[:a_count])
            b_part = tuple(str(value) for value in shuffled[a_count:])
            a_names.extend(a_part)
            b_names.extend(b_part)
            half_counts[key] = (len(a_part), len(b_part))
        a_names = tuple(sorted(a_names))
        b_names = tuple(sorted(b_names))
        membership = {
            "split_index": split_index,
            "seed": seed,
            "odd_stratum_extra_side": extra_side,
            "a_pair_names": a_names,
            "b_pair_names": b_names,
            "stratum_half_counts": sorted(half_counts.items()),
            "stratum_by_pair": sorted(stratum_by_pair.items()),
            "snr_field": "candidate_left_snr",
        }
        membership_hash = hashlib.sha256(
            json.dumps(
                membership,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        splits.append(
            HalfSampleSplit(
                split_index=split_index,
                seed=seed,
                a_pair_names=a_names,
                b_pair_names=b_names,
                stratum_half_counts=half_counts,
                stratum_by_pair=stratum_by_pair,
                snr_field="candidate_left_snr",
                odd_stratum_extra_side=extra_side,
                membership_sha256=membership_hash,
            )
        )
    plan_hash = hashlib.sha256(
        json.dumps(
            {
                "base_seed": first_seed,
                "split_hashes": [
                    split.membership_sha256 for split in splits
                ],
                "stratum_membership_sha256": (
                    hashlib.sha256(
                        json.dumps(
                            {
                                "snr_field": "candidate_left_snr",
                                "distance_quintile_edges_km": (
                                    distance_edges.tolist()
                                ),
                                "snr_tertile_edges": snr_edges.tolist(),
                                "stratum_by_pair": sorted(
                                    stratum_by_pair.items()
                                ),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return HalfSamplePlan(
        splits=tuple(splits),
        base_seed=first_seed,
        plan_sha256=plan_hash,
    )


def evaluate_half_sample_stability(
    absolute_velocity_differences_km_s,
    *,
    target_periods_s: Iterable[float],
    maximum_median_difference_km_s: float = (
        HALF_SAMPLE_MAXIMUM_MEDIAN_DIFFERENCE_KM_S
    ),
    maximum_p90_difference_km_s: float = (
        HALF_SAMPLE_MAXIMUM_P90_DIFFERENCE_KM_S
    ),
) -> HalfSampleStabilityResult:
    """Apply the frozen median and linear-90th-percentile stability gates."""

    differences = np.asarray(
        absolute_velocity_differences_km_s,
        dtype=float,
    )
    periods = tuple(float(value) for value in target_periods_s)
    median_gate = float(maximum_median_difference_km_s)
    p90_gate = float(maximum_p90_difference_km_s)
    if (
        differences.ndim != 2
        or differences.shape != (20, len(periods))
        or np.any(~np.isfinite(differences))
        or np.any(differences < 0)
        or tuple(sorted(periods)) != WANG_TARGET_PERIODS_S
        or not np.isfinite(median_gate)
        or median_gate < 0
        or not np.isfinite(p90_gate)
        or p90_gate < 0
    ):
        raise ValueError(
            "half-sample differences must have shape (20, target_count)"
        )
    summaries = {}
    for index, period in enumerate(periods):
        median = float(np.median(differences[:, index]))
        p90 = float(
            np.percentile(
                differences[:, index],
                90.0,
                method="linear",
            )
        )
        if median > median_gate:
            status = "half_sample_median_too_large"
        elif p90 > p90_gate:
            status = "half_sample_p90_too_large"
        else:
            status = "accepted"
        summaries[period] = HalfSamplePeriodSummary(
            period_s=period,
            median_absolute_difference_km_s=median,
            p90_absolute_difference_km_s=p90,
            accepted=status == "accepted",
            status=status,
        )
    accepted = all(summary.accepted for summary in summaries.values())
    return HalfSampleStabilityResult(
        accepted=accepted,
        status="accepted" if accepted else "half_sample_stability_failed",
        period_summaries=summaries,
    )


def _canonical_bitwise_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise ValueError("object arrays cannot enter the LEFT table hash")
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "bytes": contiguous.tobytes(order="C").hex(),
        }
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return {
            "__numpy_scalar__": True,
            "dtype": scalar.dtype.str,
            "bytes": scalar.tobytes().hex(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_bitwise_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda row: str(row[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_bitwise_value(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite floats cannot enter the LEFT table hash")
        return {"__python_float_hex__": value.hex()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ValueError(
        "unsupported value type in LEFT table hash: "
        f"{type(value).__name__}"
    )


def hash_left_observation_table(
    rows: Iterable[Mapping[str, object]],
) -> str:
    """Hash only the frozen candidate-equivalence LEFT observation fields."""

    fields = (
        "pair_name",
        "T_inst",
        "t0",
        "U",
        "signal_peak",
        "leading_rms",
        "trailing_rms",
        "ridge_fields",
    )
    normalized = []
    for source in rows:
        row = dict(source)
        if any(field not in row for field in fields):
            raise ValueError("LEFT observation row is missing a hash field")
        normalized.append(
            _canonical_bitwise_value(
                {field: row[field] for field in fields}
            )
        )
    normalized.sort(
        key=lambda row: json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _largest_remainder_quotas(
    stratum_sizes: Mapping[Tuple[int, int, int], int],
    *,
    sample_size: int,
) -> Dict[Tuple[int, int, int], int]:
    total = sum(stratum_sizes.values())
    if sample_size >= total:
        return dict(stratum_sizes)
    exact = {
        key: sample_size * size / total
        for key, size in stratum_sizes.items()
    }
    quotas = {
        key: min(size, int(np.floor(exact[key])))
        for key, size in stratum_sizes.items()
    }
    remaining = sample_size - sum(quotas.values())
    order = sorted(
        stratum_sizes,
        key=lambda key: (
            -(exact[key] - np.floor(exact[key])),
            key,
        ),
    )
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] >= stratum_sizes[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("could not allocate Stage B stratum quotas")
    return quotas


def select_stage_b_pairs(
    pair_rows: Iterable[Mapping[str, object]],
    *,
    closure_edge_pair_names: Iterable[str] = (),
    max_random_pairs: int = STAGE_B_RANDOM_PAIR_LIMIT,
    seed: int = 20260717,
) -> StageBSelection:
    """Select a deterministic distance/azimuth/preliminary-SNR Stage B set."""

    rows = sorted(
        (dict(row) for row in pair_rows),
        key=lambda row: str(row["pair_name"]),
    )
    maximum = int(max_random_pairs)
    random_seed = int(seed)
    if not rows or maximum <= 0 or random_seed < 0:
        raise ValueError("Stage B selection inputs are invalid")
    names = tuple(str(row["pair_name"]) for row in rows)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("pair_name values must be non-empty and unique")
    distance = np.asarray(
        [float(row["distance_km"]) for row in rows],
        dtype=float,
    )
    azimuth = np.asarray(
        [float(row["azimuth_deg"]) for row in rows],
        dtype=float,
    )
    preliminary_snr = np.asarray(
        [float(row["preliminary_snr"]) for row in rows],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(distance))
        or np.any(distance <= 0)
        or np.any(~np.isfinite(azimuth))
        or np.any(np.isinf(preliminary_snr))
    ):
        raise ValueError("Stage B stratification values are invalid")
    distance_edges = _quantile(
        distance,
        DISTANCE_QUANTILE_PROBABILITIES,
    )
    finite_snr = preliminary_snr[np.isfinite(preliminary_snr)]
    snr_edges = (
        _quantile(finite_snr, SNR_QUANTILE_PROBABILITIES)
        if finite_snr.size
        else np.zeros(2, dtype=float)
    )
    distance_bins = np.searchsorted(
        distance_edges,
        distance,
        side="right",
    )
    azimuth_bins = np.floor(
        np.mod(azimuth, 360.0) / AZIMUTH_SECTOR_WIDTH_DEG
    ).astype(int)
    snr_bins = np.full(preliminary_snr.size, -1, dtype=int)
    snr_bins[np.isfinite(preliminary_snr)] = np.searchsorted(
        snr_edges,
        preliminary_snr[np.isfinite(preliminary_snr)],
        side="right",
    )
    strata: Dict[Tuple[int, int, int], list] = {}
    stratum_by_all_pair: Dict[str, Tuple[int, int, int]] = {}
    for index, name in enumerate(names):
        key = (
            int(distance_bins[index]),
            int(azimuth_bins[index]),
            int(snr_bins[index]),
        )
        strata.setdefault(key, []).append(name)
        stratum_by_all_pair[name] = key
    quotas = _largest_remainder_quotas(
        {key: len(value) for key, value in strata.items()},
        sample_size=min(maximum, len(rows)),
    )
    generator = np.random.default_rng(random_seed)
    random_names = []
    random_counts = {}
    for key in sorted(strata):
        members = np.asarray(sorted(strata[key]), dtype=object)
        order = generator.permutation(members.size)
        chosen = members[order[: quotas[key]]].tolist()
        random_names.extend(str(value) for value in chosen)
        random_counts[key] = len(chosen)
    random_names = tuple(sorted(random_names))
    closure_names = tuple(
        sorted(set(str(value) for value in closure_edge_pair_names))
    )
    unknown_closure = set(closure_names).difference(names)
    if unknown_closure:
        raise ValueError(
            "closure edges are absent from pair_rows: "
            + ", ".join(sorted(unknown_closure))
        )
    selected_names = tuple(sorted(set(random_names).union(closure_names)))
    selected_strata = {
        name: stratum_by_all_pair[name] for name in selected_names
    }
    audit = {
        "seed": random_seed,
        "max_random_pairs": maximum,
        "distance_quantile_probabilities": (
            DISTANCE_QUANTILE_PROBABILITIES
        ),
        "distance_quintile_edges_km": distance_edges.tolist(),
        "azimuth_sector_width_deg": AZIMUTH_SECTOR_WIDTH_DEG,
        "snr_quantile_probabilities": SNR_QUANTILE_PROBABILITIES,
        "snr_tertile_edges": snr_edges.tolist(),
        "random_pair_names": random_names,
        "closure_edge_pair_names": closure_names,
        "selected_pair_names": selected_names,
        "stratum_by_pair": sorted(selected_strata.items()),
        "stratum_random_counts": sorted(random_counts.items()),
    }
    membership_hash = hashlib.sha256(
        json.dumps(
            audit,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return StageBSelection(
        random_pair_names=random_names,
        closure_edge_pair_names=closure_names,
        selected_pair_names=selected_names,
        distance_quintile_edges_km=distance_edges,
        snr_tertile_edges=snr_edges,
        distance_quantile_probabilities=(
            DISTANCE_QUANTILE_PROBABILITIES
        ),
        snr_quantile_probabilities=SNR_QUANTILE_PROBABILITIES,
        azimuth_sector_width_deg=AZIMUTH_SECTOR_WIDTH_DEG,
        stratum_by_pair=selected_strata,
        stratum_random_counts=random_counts,
        seed=random_seed,
        max_random_pairs=maximum,
        membership_sha256=membership_hash,
    )


def run_stage_b_validation(
    *,
    inventory_rows: Iterable[Mapping[str, object]],
    station_coordinates: Mapping[str, Sequence[float]],
    closure_triplets: Iterable[Mapping[str, object]],
    closure_edge_pair_names: Iterable[str],
    candidate_grid: Sequence[Mapping[str, object]],
    lineage_status: str,
    lineage_preferred_phase_convention: object,
    benchmark_stage_b: Callable[..., StageBBenchmarkEvidence],
    measure_candidate: Callable[..., Mapping[str, object]],
    fit_full_reference: Callable[..., Mapping[str, object]],
    fit_split_half_reference: Callable[..., Mapping[str, object]],
    run_phase_matching: Callable[..., PhaseMatchingRunEvidence],
    input_inventory_sha256: str,
    code_sha256: str,
    config_sha256: str,
    max_workers: int = 24,
    phase_matched_second_pass_ftan: object = None,
) -> StageBRunResult:
    """Execute the fixed Stage B scientific gate sequence.

    Callbacks adapt the validation kernels to the runner's HDF5 and Task 6
    reference-fit objects.  They receive only upstream data, which prevents
    closure results from leaking backward into reference or cycle fitting.
    """

    def left_row_passes_fixed_qc(row: Mapping[str, object]) -> bool:
        try:
            period = float(row["T_inst"])
            velocity = float(row["U"])
            signal_peak = float(row["signal_peak"])
            leading_rms = float(row["leading_rms"])
            trailing_rms = float(row["trailing_rms"])
            leading_snr = float(row["leading_snr"])
            trailing_snr = float(row["trailing_snr"])
        except (KeyError, TypeError, ValueError):
            return False
        values = (
            period,
            velocity,
            signal_peak,
            leading_rms,
            trailing_rms,
            leading_snr,
            trailing_snr,
        )
        if (
            any(not np.isfinite(value) for value in values)
            or period <= 0
            or signal_peak < 0
            or leading_rms <= 0
            or trailing_rms <= 0
            or leading_snr <= 4.0
            or trailing_snr <= 4.0
        ):
            return False
        upper_velocity = 3.0 if period < 4.5 else 3.3
        tolerance = 64.0 * np.finfo(float).eps
        return bool(
            1.6 <= velocity <= upper_velocity
            and np.isclose(
                leading_snr,
                signal_peak / leading_rms,
                rtol=tolerance,
                atol=tolerance,
            )
            and np.isclose(
                trailing_snr,
                signal_peak / trailing_rms,
                rtol=tolerance,
                atol=tolerance,
            )
        )

    inventory = tuple(dict(row) for row in inventory_rows)
    inventory_by_name = {
        str(row["pair_name"]): row for row in inventory
    }
    if len(inventory_by_name) != len(inventory):
        raise ValueError("formal inventory pair names must be unique")
    coordinates = {
        str(name): tuple(float(value) for value in coordinate)
        for name, coordinate in station_coordinates.items()
    }
    if not coordinates or any(
        len(coordinate) != 2
        or any(not np.isfinite(value) for value in coordinate)
        for coordinate in coordinates.values()
    ):
        raise ValueError("formal Stage B requires finite station coordinates")
    triplet_definitions = tuple(dict(row) for row in closure_triplets)
    if len(triplet_definitions) > 1000:
        raise ValueError("Stage B cannot exceed 1000 closure triplets")
    triplet_ids = tuple(
        str(row.get("triplet_id", "")) for row in triplet_definitions
    )
    if (
        any(not value for value in triplet_ids)
        or len(set(triplet_ids)) != len(triplet_ids)
    ):
        raise ValueError("closure triplet IDs must be non-empty and unique")
    derived_closure_edges = {
        str(row[name])
        for row in triplet_definitions
        for name in ("pair_ab_name", "pair_bc_name", "pair_ac_name")
    }
    declared_closure_edges = {
        str(value) for value in closure_edge_pair_names
    }
    if derived_closure_edges != declared_closure_edges:
        raise ValueError(
            "closure edge membership disagrees with triplet definitions"
        )
    grid = tuple(dict(row) for row in candidate_grid)
    expected_grid = build_candidate_grid(
        phase_conventions=FORMAL_PHASE_CONVENTIONS,
        alpha_candidates=FORMAL_ALPHA_CANDIDATES,
        beta1_candidates=FORMAL_BETA1_CANDIDATES,
        beta2_candidates=FORMAL_BETA2_CANDIDATES,
    )
    grid_signature = {
        (
            str(row.get("candidate_id", "")),
            str(row.get("phase_convention", "")),
            float(row.get("alpha", np.nan)),
            float(row.get("beta1", np.nan)),
            float(row.get("beta2", np.nan)),
        )
        for row in grid
    }
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
    if len(grid) != 300 or grid_signature != expected_signature:
        raise ValueError("formal Stage B requires the complete 300-candidate grid")
    workers = int(max_workers)
    if workers <= 0 or workers > 24:
        raise ValueError("formal Stage B max_workers must lie in [1, 24]")
    selection = select_stage_b_pairs(
        inventory,
        closure_edge_pair_names=declared_closure_edges,
        max_random_pairs=STAGE_B_RANDOM_PAIR_LIMIT,
        seed=20260717,
    )
    benchmark_evidence = benchmark_stage_b(
        candidate_count=300,
        synthetic_left_observation_count=2000,
        synthetic_waveform_count=STAGE_B_BENCHMARK_WAVEFORM_COUNT,
        phase_convention_count=2,
        alpha_count=6,
        beta_grid_count=25,
        full_grid_ridge_repeat_count=3,
        single_reference_fit_count=10,
        lambda_count=25,
        fold_count=5,
        half_sample_count=20,
        half_side_count=2,
        half_start_count=5,
        max_workers=workers,
    )
    if not isinstance(benchmark_evidence, StageBBenchmarkEvidence):
        raise ValueError(
            "benchmark_stage_b must return fixed StageBBenchmarkEvidence"
        )
    budget = project_stage_b_budget_from_fixed_benchmark(
        benchmark_evidence,
        selected_pair_count=len(selection.selected_pair_names),
        maximum_measurement_class_count=300,
        worker_count=workers,
    )
    if (
        budget.optimizer_calls_per_class
        != STAGE_B_OPTIMIZER_CALLS_PER_CLASS
    ):
        raise ValueError("Stage B benchmark did not use the frozen 953-call load")
    if not budget.accepted:
        return StageBRunResult(
            status="stage_b_budget_exceeded",
            return_code=2,
            budget=budget,
            benchmark_evidence=benchmark_evidence,
            selection=selection,
            candidate_results=(),
            measurement_classes={},
            class_evidence={},
            decision=None,
            phase_matching_diagnostics={},
            frozen_parameters=None,
            audit={
                "candidate_count": 300,
                "maximum_measurement_class_count": 300,
                "optimizer_calls_per_class": (
                    STAGE_B_OPTIMIZER_CALLS_PER_CLASS
                ),
                "benchmark_input_sha256": (
                    benchmark_evidence.benchmark_input_sha256
                ),
                "benchmark_cache_hit_fraction": (
                    benchmark_evidence.cache_hit_fraction
                ),
                "reference_projection_worker_count": workers,
            },
        )
    selected_names = set(selection.selected_pair_names)
    measurement_by_candidate: Dict[str, Dict[str, object]] = {}
    classes: Dict[str, list] = {}
    all_classes: Dict[str, list] = {}
    for candidate in grid:
        candidate_id = str(candidate["candidate_id"])
        measured = dict(
            measure_candidate(
                candidate=MappingProxyType(dict(candidate)),
                selection=selection,
            )
        )
        count_fields = (
            "processed_pair_count",
            "successful_pair_count",
            "expected_scientific_rejection_count",
            "unexpected_pair_exception_count",
        )
        if not any(name in measured for name in count_fields):
            measured.update(
                {
                    "processed_pair_count": len(selection.selected_pair_names),
                    "successful_pair_count": len(selection.selected_pair_names),
                    "expected_scientific_rejection_count": 0,
                    "unexpected_pair_exception_count": 0,
                }
            )
        input_integrity_passes = stage_b_candidate_input_integrity_passes(
            measured,
            selected_pair_count=len(selection.selected_pair_names),
        )
        left_rows = tuple(
            dict(row)
            for row in measured.get("continuous_left_rows", ())
        )
        for row in left_rows:
            name = str(row.get("pair_name", ""))
            if name in inventory_by_name:
                row["distance_km"] = float(
                    inventory_by_name[name]["distance_km"]
                )
                row["azimuth_deg"] = float(
                    inventory_by_name[name]["azimuth_deg"]
                )
        if any(
            str(row.get("pair_name", "")) not in selected_names
            for row in left_rows
        ):
            raise ValueError("candidate LEFT rows contain an unselected pair")
        left_hash = (
            hash_left_observation_table(left_rows)
            if left_rows
            else hashlib.sha256(b"[]").hexdigest()
        )
        left_qc_invariants_pass = bool(left_rows) and all(
            left_row_passes_fixed_qc(row) for row in left_rows
        )
        ridge_passes = bool(left_rows) and all(
            isinstance(row.get("ridge_valid"), (bool, np.bool_))
            and bool(row["ridge_valid"])
            for row in left_rows
        )
        instantaneous_passes = bool(left_rows) and all(
            isinstance(
                row.get("instantaneous_period_valid"),
                (bool, np.bool_),
            )
            and bool(row["instantaneous_period_valid"])
            for row in left_rows
        )
        boundary = evaluate_candidate_boundary_fraction(
            accepted_measurement_count=max(1, len(left_rows)),
            accepted_outermost_velocity_cell_count=int(
                measured.get(
                    "accepted_outermost_velocity_cell_count",
                    0,
                )
            ),
        )
        if not left_rows:
            boundary = CandidateBoundaryResult(
                accepted_measurement_count=1,
                accepted_outermost_velocity_cell_count=1,
                accepted_boundary_fraction=1.0,
                maximum_boundary_fraction=(
                    CANDIDATE_MAXIMUM_BOUNDARY_FRACTION
                ),
                accepted=False,
                status="no_accepted_measurements",
            )
        synthetic_passes = (
            str(measured.get("synthetic_validation_status", ""))
            == "accepted"
        )
        candidate_independent_passes = (
            synthetic_passes
            and input_integrity_passes
            and left_qc_invariants_pass
            and ridge_passes
            and instantaneous_passes
            and boundary.accepted
        )
        measurement_by_candidate[candidate_id] = {
            "candidate": candidate,
            "left_rows": left_rows,
            "left_observation_sha256": left_hash,
            "synthetic_passes": synthetic_passes,
            "input_integrity_passes": input_integrity_passes,
            "ridge_passes": ridge_passes,
            "left_qc_invariants_pass": left_qc_invariants_pass,
            "instantaneous_period_passes": instantaneous_passes,
            "boundary": boundary,
            "candidate_independent_passes": (
                candidate_independent_passes
            ),
        }
        all_classes.setdefault(left_hash, []).append(candidate_id)
        if candidate_independent_passes:
            classes.setdefault(left_hash, []).append(candidate_id)

    def evaluate_class(job):
        left_hash, candidate_ids = job
        representative = measurement_by_candidate[candidate_ids[0]]
        left_rows = representative["left_rows"]
        reference = fit_full_reference(
            left_rows=left_rows,
            candidate_ids=tuple(sorted(candidate_ids)),
            maximum_optimizer_calls=753,
        )
        if not isinstance(reference, FullReferenceEvidence):
            raise ValueError(
                "fit_full_reference must return FullReferenceEvidence"
            )
        reference_passes = reference.status == "accepted"
        alias_passes = (
            reference_passes
            and reference.alias_status == "accepted"
        )
        closure = None
        split_plan = None
        half_stability = None
        triplet_rows = ()
        geometry_valid_rows = []
        half_fit_audit = []
        differences = None
        if alias_passes:
            continuous_pair_names = {
                str(row["pair_name"]) for row in left_rows
            }
            corrected_by_pair_period = {}
            for corrected in reference.corrected_rows:
                key = (
                    corrected.pair_name,
                    corrected.target_period_s,
                )
                if (
                    corrected.pair_name not in continuous_pair_names
                    or key in corrected_by_pair_period
                ):
                    raise ValueError(
                        "corrected target rows violate LEFT membership "
                        "or pair-period uniqueness"
                    )
                corrected_by_pair_period[key] = corrected
            for definition in triplet_definitions:
                triplet_id = str(definition["triplet_id"])
                station_codes = tuple(
                    str(definition.get(name, ""))
                    for name in (
                        "station_a_code",
                        "station_b_code",
                        "station_c_code",
                    )
                )
                if (
                    not triplet_id
                    or any(code not in coordinates for code in station_codes)
                    or len(set(station_codes)) != 3
                ):
                    raise ValueError(
                        "triplet geometry references an unknown station"
                    )
                pair_names = tuple(
                    str(definition.get(name, ""))
                    for name in (
                        "pair_ab_name",
                        "pair_bc_name",
                        "pair_ac_name",
                    )
                )
                expected_pair_station_sets = (
                    frozenset(station_codes[:2]),
                    frozenset(station_codes[1:]),
                    frozenset((station_codes[0], station_codes[2])),
                )
                actual_pair_station_sets = tuple(
                    frozenset(name.split("__")) for name in pair_names
                )
                if actual_pair_station_sets != expected_pair_station_sets:
                    raise ValueError(
                        "triplet edge pair identities disagree with stations"
                    )
                geometry = evaluate_triplet_geometry(
                    station_a=coordinates[station_codes[0]],
                    station_b=coordinates[station_codes[1]],
                    station_c=coordinates[station_codes[2]],
                )
                if not geometry.accepted:
                    continue
                for period in WANG_TARGET_PERIODS_S:
                    observations = tuple(
                        corrected_by_pair_period.get((name, period))
                        for name in pair_names
                    )
                    if any(value is None for value in observations):
                        continue
                    ab, bc, ac = observations
                    row = {
                            "triplet_id": triplet_id,
                            "period_s": period,
                            "station_a_code": station_codes[0],
                            "station_b_code": station_codes[1],
                            "station_c_code": station_codes[2],
                            "pair_ab_name": pair_names[0],
                            "pair_bc_name": pair_names[1],
                            "pair_ac_name": pair_names[2],
                            "distance_ab_km": geometry.distance_ab_km,
                            "distance_bc_km": geometry.distance_bc_km,
                            "distance_ac_km": geometry.distance_ac_km,
                            "raw_time_ab_s": ab.raw_time_s,
                            "raw_time_bc_s": bc.raw_time_s,
                            "raw_time_ac_s": ac.raw_time_s,
                            "corrected_time_ab_s": ab.corrected_time_s,
                            "corrected_time_bc_s": bc.corrected_time_s,
                            "corrected_time_ac_s": ac.corrected_time_s,
                            "left_ab": ab.left_qc_accepted,
                            "left_bc": bc.left_qc_accepted,
                            "left_ac": ac.left_qc_accepted,
                            "snr_ab": min(
                                ab.leading_snr,
                                ab.trailing_snr,
                            ),
                            "snr_bc": min(
                                bc.leading_snr,
                                bc.trailing_snr,
                            ),
                            "snr_ac": min(
                                ac.leading_snr,
                                ac.trailing_snr,
                            ),
                        }
                    geometry_valid_rows.append(row)
            triplet_rows = tuple(geometry_valid_rows)
            closure = evaluate_triplet_closure(
                geometry_valid_rows,
                target_periods_s=WANG_TARGET_PERIODS_S,
            )
            if closure.accepted:
                pair_rows_by_name: Dict[str, Dict[str, object]] = {}
                for row in left_rows:
                    name = str(row["pair_name"])
                    snr = min(
                        float(row["signal_peak"])
                        / float(row["leading_rms"]),
                        float(row["signal_peak"])
                        / float(row["trailing_rms"]),
                    )
                    if not np.isfinite(snr):
                        continue
                    if name not in pair_rows_by_name:
                        inventory_row = inventory_by_name[name]
                        pair_rows_by_name[name] = {
                            "pair_name": name,
                            "distance_km": float(
                                inventory_row["distance_km"]
                            ),
                            "azimuth_deg": float(
                                inventory_row["azimuth_deg"]
                            ),
                            "candidate_left_snr": snr,
                        }
                    else:
                        pair_rows_by_name[name][
                            "candidate_left_snr"
                        ] = min(
                            pair_rows_by_name[name][
                                "candidate_left_snr"
                            ],
                            snr,
                        )
                split_plan = build_half_sample_splits(
                    tuple(pair_rows_by_name.values()),
                )
                differences = np.empty(
                    (20, len(WANG_TARGET_PERIODS_S)),
                    dtype=float,
                )
                for split in split_plan.splits:
                    velocities = {}
                    for side, names in (
                        ("A", split.a_pair_names),
                        ("B", split.b_pair_names),
                    ):
                        name_set = set(names)
                        subset = tuple(
                            row
                            for row in left_rows
                            if str(row["pair_name"]) in name_set
                        )
                        half_fit = fit_split_half_reference(
                            left_rows=subset,
                            full_reference=reference,
                            split_index=split.split_index,
                            seed=split.seed,
                            side=side,
                            lambda_s=reference.lambda_s,
                            lambda_g=reference.lambda_g,
                            basin_starts=reference.basin_starts[:5],
                            maxiter=300,
                        )
                        if not isinstance(
                            half_fit,
                            SplitHalfFitEvidence,
                        ):
                            raise ValueError(
                                "fit_split_half_reference must return "
                                "SplitHalfFitEvidence"
                            )
                        if (
                            half_fit.lambda_s != reference.lambda_s
                            or half_fit.lambda_g != reference.lambda_g
                        ):
                            raise ValueError(
                                "split-half fit changed the frozen lambdas"
                            )
                        if half_fit.status != "accepted":
                            half_fit_audit.append(
                                {
                                    "split_index": split.split_index,
                                    "seed": split.seed,
                                    "side": side,
                                    "fit": half_fit,
                                }
                            )
                            velocities[side] = None
                            continue
                        half_fit_audit.append(
                            {
                                "split_index": split.split_index,
                                "seed": split.seed,
                                "side": side,
                                "fit": half_fit,
                            }
                        )
                        velocities[side] = (
                            half_fit.target_velocities_km_s
                        )
                    if velocities["A"] is None or velocities["B"] is None:
                        differences[split.split_index, :] = np.inf
                    else:
                        differences[split.split_index, :] = np.abs(
                            velocities["A"] - velocities["B"]
                        )
                if np.all(np.isfinite(differences)):
                    half_stability = evaluate_half_sample_stability(
                        differences,
                        target_periods_s=WANG_TARGET_PERIODS_S,
                    )
        class_result = {
            "candidate_ids": tuple(sorted(candidate_ids)),
            "continuous_left_rows": tuple(
                MappingProxyType(dict(row)) for row in left_rows
            ),
            "continuous_left_rows_sha256": left_hash,
            "reference": reference,
            "reference_passes": reference_passes,
            "alias_passes": alias_passes,
            "closure": closure,
            "triplet_rows_raw_input": tuple(
                MappingProxyType(dict(row)) for row in triplet_rows
            ),
            "triplet_rows_geometry_valid": tuple(
                MappingProxyType(dict(row))
                for row in geometry_valid_rows
            ),
            "split_plan": split_plan,
            "split_half_fit_audit": tuple(
                MappingProxyType(dict(row)) for row in half_fit_audit
            ),
            "split_half_absolute_differences_km_s": (
                None
                if differences is None
                else np.array(differences, dtype=float, copy=True)
            ),
            "half_stability": half_stability,
        }
        return (
            left_hash,
            _pack_stage_b_class_result(class_result),
            os.getpid(),
        )

    class_jobs = tuple(sorted(classes.items()))
    class_outputs = execute_measurement_class_processes(
        class_jobs,
        evaluator=evaluate_class,
        max_workers=workers,
    )
    class_results: Dict[str, Dict[str, object]] = {
        left_hash: _unpack_stage_b_class_result(payload)
        for left_hash, payload, _worker_pid in class_outputs
    }
    class_worker_pids = tuple(
        sorted({int(worker_pid) for _, _, worker_pid in class_outputs})
    )
    for left_hash, candidate_ids in sorted(all_classes.items()):
        if left_hash in class_results:
            continue
        representative = measurement_by_candidate[candidate_ids[0]]
        class_results[left_hash] = {
            "candidate_ids": tuple(sorted(candidate_ids)),
            "continuous_left_rows": tuple(
                MappingProxyType(dict(row))
                for row in representative["left_rows"]
            ),
            "continuous_left_rows_sha256": left_hash,
            "reference": None,
            "reference_passes": False,
            "alias_passes": False,
            "closure": None,
            "triplet_rows_raw_input": (),
            "triplet_rows_geometry_valid": (),
            "split_plan": None,
            "split_half_fit_audit": (),
            "split_half_absolute_differences_km_s": None,
            "half_stability": None,
        }
    for left_hash, candidate_ids in all_classes.items():
        class_results[left_hash]["candidate_ids"] = tuple(
            sorted(candidate_ids)
        )

    candidate_results = []
    for candidate in grid:
        candidate_id = str(candidate["candidate_id"])
        measurement = measurement_by_candidate[candidate_id]
        class_result = class_results.get(
            measurement["left_observation_sha256"]
        )
        closure = None if class_result is None else class_result["closure"]
        half = (
            None if class_result is None else class_result["half_stability"]
        )
        alias_passes = bool(
            class_result is not None and class_result["alias_passes"]
        )
        closure_values = (
            [
                summary.median_absolute_cycles
                for summary in closure.period_summaries.values()
            ]
            if closure is not None and closure.accepted
            else []
        )
        closure_score = (
            float(np.median(closure_values))
            if closure_values
            else float(np.finfo(float).max)
        )
        candidate_results.append(
            {
                **candidate,
                "closure_median_cycles": closure_score,
                "synthetic_passes": measurement["synthetic_passes"],
                "input_integrity_passes": measurement[
                    "input_integrity_passes"
                ],
                "ridge_passes": measurement["ridge_passes"],
                "instantaneous_period_passes": (
                    measurement["instantaneous_period_passes"]
                ),
                "alias_passes": alias_passes,
                "triplet_passes": bool(
                    closure is not None and closure.accepted
                ),
                "half_sample_passes": bool(
                    half is not None and half.accepted
                ),
                "boundary_passes": measurement["boundary"].accepted,
                "left_qc_invariants_pass": measurement[
                    "left_qc_invariants_pass"
                ],
                "boundary_status": measurement["boundary"].status,
                "accepted_measurement_count": (
                    measurement["boundary"].accepted_measurement_count
                ),
                "accepted_outermost_velocity_cell_count": (
                    measurement[
                        "boundary"
                    ].accepted_outermost_velocity_cell_count
                ),
                "accepted_boundary_fraction": (
                    measurement["boundary"].accepted_boundary_fraction
                ),
                "left_observation_sha256": measurement[
                    "left_observation_sha256"
                ],
            }
        )
    decision = freeze_ftan_candidate(
        candidate_results,
        lineage_status=lineage_status,
        lineage_preferred_phase_convention=(
            lineage_preferred_phase_convention
        ),
    )
    class_members = {
        left_hash: tuple(sorted(result["candidate_ids"]))
        for left_hash, result in class_results.items()
    }
    common_audit = {
        "candidate_count": len(candidate_results),
        "selection_sha256": selection.membership_sha256,
        "measurement_class_count": len(class_members),
        "optimizer_calls_per_class": STAGE_B_OPTIMIZER_CALLS_PER_CLASS,
        "benchmark_input_sha256": (
            benchmark_evidence.benchmark_input_sha256
        ),
        "benchmark_cache_hit_fraction": (
            benchmark_evidence.cache_hit_fraction
        ),
        "benchmark_fixed_counts": {
            "waveforms": STAGE_B_BENCHMARK_WAVEFORM_COUNT,
            "optimizer_calls": STAGE_B_BENCHMARK_OPTIMIZER_CALLS,
        },
        "reference_projection_worker_count": min(
            workers,
            max(1, len(class_jobs)),
        ),
        "reference_worker_process_count": len(class_worker_pids),
        "reference_worker_pids": class_worker_pids,
        "fixed_thresholds": {
            "triplet_minimum_support": TRIPLET_MINIMUM_SUPPORT,
            "triplet_maximum_median_absolute_cycles": (
                TRIPLET_MAXIMUM_MEDIAN_ABSOLUTE_CYCLES
            ),
            "triplet_maximum_absolute_bias_cycles": (
                TRIPLET_MAXIMUM_ABSOLUTE_BIAS_CYCLES
            ),
            "half_sample_maximum_median_difference_km_s": (
                HALF_SAMPLE_MAXIMUM_MEDIAN_DIFFERENCE_KM_S
            ),
            "half_sample_maximum_p90_difference_km_s": (
                HALF_SAMPLE_MAXIMUM_P90_DIFFERENCE_KM_S
            ),
            "candidate_maximum_boundary_fraction": (
                CANDIDATE_MAXIMUM_BOUNDARY_FRACTION
            ),
            "candidate_relative_tie_fraction": (
                CANDIDATE_RELATIVE_TIE_FRACTION
            ),
        },
    }
    if not decision.accepted:
        return StageBRunResult(
            status=decision.status,
            return_code=2,
            budget=budget,
            benchmark_evidence=benchmark_evidence,
            selection=selection,
            candidate_results=tuple(candidate_results),
            measurement_classes=class_members,
            class_evidence=class_results,
            decision=decision,
            phase_matching_diagnostics={},
            frozen_parameters=None,
            audit=common_audit,
        )
    phase_matching = {}
    phase_matching_execution_hashes = {}
    candidate_results_by_id = {
        str(row["candidate_id"]): row for row in candidate_results
    }
    for phase in FORMAL_PHASE_CONVENTIONS:
        candidate_id = decision.best_candidate_by_phase_convention[phase]
        candidate = candidate_results_by_id[candidate_id]
        execution_hashes = []

        def execute_second_pass_ftan(**kwargs):
            if not callable(phase_matched_second_pass_ftan):
                raise ValueError(
                    "formal phase matching requires the actual second-pass "
                    "FTAN core"
                )
            if float(kwargs.get("first_pass_alpha", np.nan)) != float(
                candidate["alpha"]
            ):
                raise ValueError(
                    "phase-matching execution changed first-pass alpha"
                )
            result = phase_matched_second_pass_ftan(**kwargs)
            execution_hashes.append(
                hash_phase_matching_second_pass_output(result)
            )
            return result

        evidence = run_phase_matching(
            candidate=candidate,
            class_evidence=class_results[
                candidate[
                    "left_observation_sha256"
                ]
            ],
            execute_second_pass_ftan=execute_second_pass_ftan,
        )
        aggregate_execution_hash = hash_phase_matching_execution_hashes(
            execution_hashes
        )
        if (
            not isinstance(evidence, PhaseMatchingRunEvidence)
            or evidence.phase_convention != phase
            or evidence.candidate_id != candidate_id
            or evidence.matched_output_sha256 != aggregate_execution_hash
        ):
            raise ValueError(
                "run_phase_matching lacks matching actual second-pass "
                "FTAN executions"
            )
        phase_matching[phase] = evidence
        phase_matching_execution_hashes[phase] = tuple(execution_hashes)
    common_audit["phase_matching_second_pass_execution_count"] = sum(
        len(values) for values in phase_matching_execution_hashes.values()
    )
    common_audit["phase_matching_second_pass_output_sha256"] = dict(
        sorted(
            (
                phase,
                hash_phase_matching_execution_hashes(values),
            )
            for phase, values in phase_matching_execution_hashes.items()
        )
    )
    if any(
        evidence.diagnostic.design_revision_required
        for evidence in phase_matching.values()
    ):
        phase_matching_payload = _stage_b_evidence_components(
            budget=budget,
            benchmark_evidence=benchmark_evidence,
            selection=selection,
            candidate_results=candidate_results,
            measurement_classes=class_members,
            class_evidence=class_results,
            phase_matching_diagnostics=phase_matching,
        )
        common_audit["validation_evidence_sha256"] = (
            stage_b_validation_evidence_sha256(
                phase_matching_payload
            )
        )
        return StageBRunResult(
            status="phase_matching_design_revision_required",
            return_code=2,
            budget=budget,
            benchmark_evidence=benchmark_evidence,
            selection=selection,
            candidate_results=tuple(candidate_results),
            measurement_classes=class_members,
            class_evidence=class_results,
            decision=decision,
            phase_matching_diagnostics=phase_matching,
            frozen_parameters=None,
            audit=common_audit,
        )
    evidence_payload = _stage_b_evidence_components(
        budget=budget,
        benchmark_evidence=benchmark_evidence,
        selection=selection,
        candidate_results=candidate_results,
        measurement_classes=class_members,
        class_evidence=class_results,
        phase_matching_diagnostics=phase_matching,
    )
    validation_evidence_sha256 = stage_b_validation_evidence_sha256(
        evidence_payload
    )
    common_audit["validation_evidence_sha256"] = (
        validation_evidence_sha256
    )
    frozen = build_frozen_parameters_manifest(
        decision,
        input_inventory_sha256=input_inventory_sha256,
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        validation_table_sha256=validation_evidence_sha256,
    )
    return StageBRunResult(
        status="passed",
        return_code=0,
        budget=budget,
        benchmark_evidence=benchmark_evidence,
        selection=selection,
        candidate_results=tuple(candidate_results),
        measurement_classes=class_members,
        class_evidence=class_results,
        decision=decision,
        phase_matching_diagnostics=phase_matching,
        frozen_parameters=frozen,
        audit=common_audit,
    )
