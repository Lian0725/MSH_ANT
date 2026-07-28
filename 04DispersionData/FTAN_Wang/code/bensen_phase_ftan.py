#!/usr/bin/env python3
"""Bensen-style FTAN helpers for phase-velocity extraction.

This module focuses on the mathematical core that we can unit test locally:

1. Build deterministic FTAN grids and a Gaussian filter bank.
2. Convert observed phase at group arrival into candidate phase velocities
   across integer 2π branches.
3. Select a smooth branch sequence across periods with a Bensen-like
   reference/smoothness constraint.

The higher-level batch runner can then reuse these utilities on real DAT files.
"""

from dataclasses import dataclass, fields, replace
from enum import Enum
import hashlib
import math
from numbers import Real
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, minimize
from scipy.signal import hilbert


LOG_ENERGY_FLOOR = 1e-12
RIDGE_COST_TIE_ULPS = 32.0
PHASE_UNWRAP_PREDICTION_FRACTION = 0.25
PHASE_UNWRAP_MAX_ANOMALY_FRACTION = 0.05
PHASE_UNWRAP_MAX_CONSECUTIVE_ANOMALIES = 1
PHASE_UNWRAP_MAX_CYCLE_STEP = 1
PHASE_MATCHING_MAXIMUM_PERIOD_S = 5.0
PHASE_MATCHING_CUT_TAPER_ALPHA = 0.25
_VELOCITY_CCF_FIXED_PHASE_RAD = -math.pi / 4.0


class _FrozenMetadata(dict):
    """A JSON-serializable mapping whose contents cannot be changed."""

    @staticmethod
    def _immutable(*args, **kwargs):
        raise TypeError("phase convention metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


@dataclass(frozen=True)
class PhaseConventionDefinition:
    hilbert_phase_sign: int
    scipy_phase_multiplier: int
    phase_time_sign: int
    formula_phase_sign: int
    fixed_phase_rad: float
    cycle_phase_sign: int
    apply_negative_time_derivative: bool
    cycle_count_meaning: str
    formula: str
    description: str

    def __post_init__(self) -> None:
        sign_fields = (
            "hilbert_phase_sign",
            "scipy_phase_multiplier",
            "phase_time_sign",
            "formula_phase_sign",
            "cycle_phase_sign",
        )
        if any(
            isinstance(getattr(self, name), (bool, np.bool_))
            or not isinstance(getattr(self, name), (int, np.integer))
            or int(getattr(self, name)) not in (-1, 1)
            for name in sign_fields
        ):
            raise ValueError("phase convention signs must be integer +1 or -1")
        if self.hilbert_phase_sign != self.scipy_phase_multiplier:
            raise ValueError("Hilbert and SciPy phase signs must agree")
        if self.phase_time_sign != (
            self.formula_phase_sign * self.scipy_phase_multiplier
        ):
            raise ValueError("phase_time_sign is inconsistent with phase signs")


class PhaseConvention(Enum):
    BENSEN_VELOCITY_CCF = PhaseConventionDefinition(
        hilbert_phase_sign=-1,
        scipy_phase_multiplier=-1,
        phase_time_sign=-1,
        formula_phase_sign=1,
        fixed_phase_rad=_VELOCITY_CCF_FIXED_PHASE_RAD,
        cycle_phase_sign=1,
        apply_negative_time_derivative=False,
        cycle_count_meaning=(
            "N adds one positive period to phase travel time: t=t0+N*T"
        ),
        formula="t = tu + (phi - pi/4)/omega + N*T",
        description=(
            "Bensen velocity CCF convention using the symmetric correlation "
            "without a time derivative"
        ),
    )
    LIN_NEGATIVE_DERIVATIVE_EGF = PhaseConventionDefinition(
        hilbert_phase_sign=-1,
        scipy_phase_multiplier=-1,
        phase_time_sign=-1,
        formula_phase_sign=1,
        fixed_phase_rad=_VELOCITY_CCF_FIXED_PHASE_RAD,
        cycle_phase_sign=-1,
        apply_negative_time_derivative=True,
        cycle_count_meaning=(
            "N subtracts one positive period from phase travel time: t=t0-N*T"
        ),
        formula=(
            "t = tu + (phi - pi/4)/omega - N*T; "
            "G_AB = -d[(C_AB(t)+C_AB(-t))/2]/dt"
        ),
        description=(
            "Lin empirical Green function convention using the negative time "
            "derivative of the symmetric correlation"
        ),
    )

    @property
    def definition(self) -> PhaseConventionDefinition:
        return self.value

    @property
    def metadata(self) -> _FrozenMetadata:
        definition = self.definition
        return _FrozenMetadata(
            name=self.name,
            hilbert_phase_sign=definition.hilbert_phase_sign,
            scipy_phase_multiplier=definition.scipy_phase_multiplier,
            phase_time_sign=definition.phase_time_sign,
            formula_phase_sign=definition.formula_phase_sign,
            fixed_phase_rad=definition.fixed_phase_rad,
            cycle_phase_sign=definition.cycle_phase_sign,
            apply_negative_time_derivative=(
                definition.apply_negative_time_derivative
            ),
            cycle_count_meaning=definition.cycle_count_meaning,
            formula=definition.formula,
            description=definition.description,
        )


def _require_phase_convention(convention: PhaseConvention) -> PhaseConvention:
    if not isinstance(convention, PhaseConvention):
        raise ValueError("convention must be a PhaseConvention member")
    return convention


def _finite_scalar(
    value: float,
    name: str,
    *,
    positive: bool = False,
) -> float:
    qualifier = "positive finite" if positive else "finite"
    if (
        isinstance(value, (bool, np.bool_, np.ndarray))
        or not isinstance(value, Real)
        or np.ndim(value) != 0
    ):
        raise ValueError(f"{name} must be a {qualifier} scalar")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be a {qualifier} scalar")
    return result


def _integer_scalar(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or np.ndim(value) != 0
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer scalar")
    return int(value)


def _canonical_principal_phase_rad(phase_rad: float) -> float:
    """Return the unique principal representative in ``(-pi, pi]``."""

    phase = _finite_scalar(phase_rad, "phase_rad")
    principal = float(math.remainder(phase, 2.0 * math.pi))
    if principal == -math.pi:
        return float(math.pi)
    return principal


def _real_numeric_array(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} must be a real numeric array, not boolean or complex"
        )
    return np.asarray(raw, dtype=float)


def _complex_numeric_array(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind != "c":
        raise ValueError(f"{name} must be a complex numeric array")
    return np.asarray(raw, dtype=complex)


def _real_numeric_iterable(values, name: str) -> np.ndarray:
    source = values if isinstance(values, np.ndarray) else list(values)
    return _real_numeric_array(source, name)


def _strict_finite_real(values, name: str):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric")
    result = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite real numeric")
    return result


def wrap_periodic(x, period):
    """Wrap finite real values to ``[-period/2, period/2)``."""

    values = _strict_finite_real(x, "x")
    period_value = _finite_scalar(period, "period", positive=True)
    wrapped = (
        (values + period_value / 2.0) % period_value
        - period_value / 2.0
    )
    if values.ndim == 0:
        return float(wrapped)
    return np.asarray(wrapped, dtype=float)


def huber_loss(residual, delta: float = 0.10):
    """Return elementwise deterministic Huber loss."""

    values = _strict_finite_real(residual, "residual")
    delta_value = _finite_scalar(delta, "delta", positive=True)
    magnitude = np.abs(values)
    result = np.where(
        magnitude <= delta_value,
        0.5 * values * values,
        delta_value * (magnitude - 0.5 * delta_value),
    )
    if values.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=float)


@dataclass(frozen=True)
class RightColumnFitResult:
    observation_count: int
    huber_slowness_s_km: float
    huber_velocity_km_s: float
    ordinary_ls_slowness_s_km: float
    ordinary_ls_velocity_km_s: float
    path_velocity_std_km_s: float
    bootstrap_velocity_ci95_km_s: np.ndarray
    bootstrap_velocity_std_km_s: float
    bootstrap_samples: int
    seed: int

    def __post_init__(self) -> None:
        count = _integer_scalar(self.observation_count, "observation_count")
        samples = _integer_scalar(self.bootstrap_samples, "bootstrap_samples")
        seed = _integer_scalar(self.seed, "seed")
        positive_values = tuple(
            _finite_scalar(getattr(self, name), name, positive=True)
            for name in (
                "huber_slowness_s_km",
                "huber_velocity_km_s",
                "ordinary_ls_slowness_s_km",
                "ordinary_ls_velocity_km_s",
            )
        )
        path_std = _finite_scalar(
            self.path_velocity_std_km_s,
            "path_velocity_std_km_s",
        )
        bootstrap_std = _finite_scalar(
            self.bootstrap_velocity_std_km_s,
            "bootstrap_velocity_std_km_s",
        )
        ci = np.array(
            _strict_finite_real(
                self.bootstrap_velocity_ci95_km_s,
                "bootstrap_velocity_ci95_km_s",
            ),
            dtype=float,
            copy=True,
        )
        if (
            count <= 0
            or samples <= 0
            or seed < 0
            or path_std < 0
            or bootstrap_std < 0
            or ci.shape != (2,)
            or np.any(ci <= 0)
            or ci[0] > ci[1]
        ):
            raise ValueError("right-column fit result is inconsistent")
        (
            huber_slowness,
            huber_velocity,
            ordinary_slowness,
            ordinary_velocity,
        ) = positive_values
        tolerance = 64.0 * np.finfo(float).eps
        if (
            not math.isclose(
                huber_velocity,
                1.0 / huber_slowness,
                rel_tol=tolerance,
                abs_tol=0.0,
            )
            or not math.isclose(
                ordinary_velocity,
                1.0 / ordinary_slowness,
                rel_tol=tolerance,
                abs_tol=0.0,
            )
        ):
            raise ValueError("right-column slowness and velocity disagree")
        ci.setflags(write=False)
        object.__setattr__(self, "observation_count", count)
        object.__setattr__(self, "huber_slowness_s_km", huber_slowness)
        object.__setattr__(self, "huber_velocity_km_s", huber_velocity)
        object.__setattr__(
            self,
            "ordinary_ls_slowness_s_km",
            ordinary_slowness,
        )
        object.__setattr__(
            self,
            "ordinary_ls_velocity_km_s",
            ordinary_velocity,
        )
        object.__setattr__(self, "path_velocity_std_km_s", path_std)
        object.__setattr__(
            self,
            "bootstrap_velocity_ci95_km_s",
            ci,
        )
        object.__setattr__(
            self,
            "bootstrap_velocity_std_km_s",
            bootstrap_std,
        )
        object.__setattr__(self, "bootstrap_samples", samples)
        object.__setattr__(self, "seed", seed)


def _fit_huber_slowness_through_origin(
    distance_km: np.ndarray,
    travel_time_s: np.ndarray,
    *,
    tuning_constant: float,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> float:
    """Fit ``travel_time = distance * slowness`` by deterministic Huber IRLS."""

    tuning = _finite_scalar(
        tuning_constant,
        "tuning_constant",
        positive=True,
    )
    iterations = _integer_scalar(max_iterations, "max_iterations")
    convergence = _finite_scalar(tolerance, "tolerance", positive=True)
    if iterations <= 0:
        raise ValueError("max_iterations must be positive")
    denominator = float(np.dot(distance_km, distance_km))
    slowness = float(np.dot(distance_km, travel_time_s) / denominator)
    scale_floor = (
        np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(travel_time_s))))
    )
    for _ in range(iterations):
        residual = travel_time_s - distance_km * slowness
        residual_center = float(np.median(residual))
        scale = float(
            1.4826
            * np.median(np.abs(residual - residual_center))
        )
        if scale <= scale_floor:
            scale = max(float(np.median(np.abs(residual))), scale_floor)
        cutoff = tuning * scale
        magnitude = np.abs(residual)
        weights = np.ones_like(magnitude)
        outlier = magnitude > cutoff
        weights[outlier] = cutoff / magnitude[outlier]
        weighted_denominator = float(
            np.dot(weights * distance_km, distance_km)
        )
        if weighted_denominator <= 0:
            raise ValueError("Huber fit has zero weighted distance")
        updated = float(
            np.dot(weights * distance_km, travel_time_s)
            / weighted_denominator
        )
        if updated <= 0 or not np.isfinite(updated):
            raise ValueError("Huber fit produced invalid slowness")
        if abs(updated - slowness) <= convergence * max(
            1.0,
            abs(slowness),
        ):
            return updated
        slowness = updated
    return slowness


def fit_right_column_slowness(
    distance_km,
    travel_time_s,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 20260717,
    huber_tuning_constant: float = 1.345,
) -> RightColumnFitResult:
    """Fit Wang right-column slowness and pair-bootstrap its robust velocity."""

    distance = _strict_finite_real(distance_km, "distance_km")
    travel_time = _strict_finite_real(travel_time_s, "travel_time_s")
    samples = _integer_scalar(bootstrap_samples, "bootstrap_samples")
    random_seed = _integer_scalar(seed, "seed")
    if (
        distance.ndim != 1
        or travel_time.ndim != 1
        or distance.shape != travel_time.shape
        or distance.size == 0
        or np.any(distance <= 0)
        or np.any(travel_time <= 0)
        or samples <= 0
        or random_seed < 0
    ):
        raise ValueError("right-column fit inputs are invalid")
    ordinary_slowness = float(
        np.dot(distance, travel_time) / np.dot(distance, distance)
    )
    if ordinary_slowness <= 0 or not np.isfinite(ordinary_slowness):
        raise ValueError("ordinary fit produced invalid slowness")
    huber_slowness = _fit_huber_slowness_through_origin(
        distance,
        travel_time,
        tuning_constant=huber_tuning_constant,
    )
    generator = np.random.default_rng(random_seed)
    bootstrap_velocities = np.empty(samples, dtype=float)
    for index in range(samples):
        draw = generator.integers(0, distance.size, size=distance.size)
        bootstrap_slowness = _fit_huber_slowness_through_origin(
            distance[draw],
            travel_time[draw],
            tuning_constant=huber_tuning_constant,
        )
        bootstrap_velocities[index] = 1.0 / bootstrap_slowness
    ci = np.percentile(
        bootstrap_velocities,
        np.asarray([2.5, 97.5], dtype=float),
        method="linear",
    )
    return RightColumnFitResult(
        observation_count=int(distance.size),
        huber_slowness_s_km=huber_slowness,
        huber_velocity_km_s=1.0 / huber_slowness,
        ordinary_ls_slowness_s_km=ordinary_slowness,
        ordinary_ls_velocity_km_s=1.0 / ordinary_slowness,
        path_velocity_std_km_s=float(np.std(distance / travel_time)),
        bootstrap_velocity_ci95_km_s=ci,
        bootstrap_velocity_std_km_s=float(
            np.std(bootstrap_velocities)
        ),
        bootstrap_samples=samples,
        seed=random_seed,
    )


def phase_slowness_to_group_slowness(
    periods_s,
    phase_slowness_s_km,
) -> np.ndarray:
    """Derive group slowness as ``d(omega*s_phase)/d omega``."""

    periods = _strict_finite_real(periods_s, "periods_s")
    slowness = _strict_finite_real(
        phase_slowness_s_km,
        "phase_slowness_s_km",
    )
    if (
        periods.ndim != 1
        or slowness.ndim != 1
        or periods.shape != slowness.shape
        or periods.size < 3
        or np.any(periods <= 0)
        or np.any(np.diff(periods) <= 0)
        or np.any(slowness <= 0)
    ):
        raise ValueError(
            "periods and phase slowness must be matching positive "
            "one-dimensional arrays with at least three increasing periods"
        )
    omega = 2.0 * math.pi / periods
    derivative = np.gradient(slowness, omega, edge_order=2)
    return np.asarray(slowness + omega * derivative, dtype=float)


def raw_phase_travel_time(
    *,
    convention: PhaseConvention,
    group_time_s: float,
    phase_rad: float,
    omega_rad_s: float,
) -> float:
    convention = _require_phase_convention(convention)
    group_time = _finite_scalar(
        group_time_s,
        "group_time_s",
        positive=True,
    )
    phase = _finite_scalar(phase_rad, "phase_rad")
    omega = _finite_scalar(
        omega_rad_s,
        "omega_rad_s",
        positive=True,
    )
    definition = convention.definition
    return float(
        group_time
        + (
            definition.formula_phase_sign * phase
            + definition.fixed_phase_rad
        )
        / omega
    )


def apply_cycle_count(
    raw_time_s: float,
    cycle_count: int,
    period_s: float,
    convention: PhaseConvention = PhaseConvention.BENSEN_VELOCITY_CCF,
) -> float:
    """Apply a scalar cycle branch; the default is legacy Bensen compatibility."""

    convention = _require_phase_convention(convention)
    raw_time = _finite_scalar(raw_time_s, "raw_time_s")
    cycle = _integer_scalar(cycle_count, "cycle_count")
    period = _finite_scalar(period_s, "period_s", positive=True)
    corrected_time = float(
        raw_time
        + convention.definition.cycle_phase_sign
        * cycle
        * period
    )
    if not np.isfinite(corrected_time) or corrected_time <= 0:
        raise ValueError("cycle-corrected phase travel time must be positive")
    return corrected_time


@dataclass(frozen=True)
class CycleResolution:
    convention: PhaseConvention
    raw_time_s: float
    reference_time_s: float
    period_s: float
    cycle_count: int
    corrected_time_s: float
    corrected_residual_s: float
    branch_tie: bool

    def __post_init__(self) -> None:
        convention = _require_phase_convention(self.convention)
        raw_time = _finite_scalar(self.raw_time_s, "raw_time_s")
        reference_time = _finite_scalar(
            self.reference_time_s,
            "reference_time_s",
            positive=True,
        )
        period = _finite_scalar(self.period_s, "period_s", positive=True)
        cycle = _integer_scalar(self.cycle_count, "cycle_count")
        corrected = _finite_scalar(
            self.corrected_time_s,
            "corrected_time_s",
            positive=True,
        )
        residual = _finite_scalar(
            self.corrected_residual_s,
            "corrected_residual_s",
        )
        if not isinstance(self.branch_tie, (bool, np.bool_)):
            raise ValueError("branch_tie must be boolean")
        expected = (
            raw_time
            + convention.definition.cycle_phase_sign * cycle * period
        )
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0,
            abs(expected),
            abs(reference_time),
        )
        if not math.isclose(corrected, expected, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("corrected_time_s is inconsistent with cycle_count")
        if not math.isclose(
            residual,
            corrected - reference_time,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("corrected_residual_s is inconsistent")
        if abs(residual) > period / 2.0 + tolerance:
            raise ValueError("corrected residual exceeds half a period")
        object.__setattr__(self, "raw_time_s", raw_time)
        object.__setattr__(self, "reference_time_s", reference_time)
        object.__setattr__(self, "period_s", period)
        object.__setattr__(self, "cycle_count", cycle)
        object.__setattr__(self, "corrected_time_s", corrected)
        object.__setattr__(self, "corrected_residual_s", residual)
        object.__setattr__(self, "branch_tie", bool(self.branch_tie))


def resolve_cycle_count(
    *,
    raw_time_s: float,
    reference_time_s: float,
    period_s: float,
    convention: PhaseConvention = PhaseConvention.BENSEN_VELOCITY_CCF,
) -> CycleResolution:
    """Resolve the closest integer branch using the canonical tie rule."""

    convention = _require_phase_convention(convention)
    raw_time = _finite_scalar(raw_time_s, "raw_time_s")
    reference_time = _finite_scalar(
        reference_time_s,
        "reference_time_s",
        positive=True,
    )
    period = _finite_scalar(period_s, "period_s", positive=True)
    sign = convention.definition.cycle_phase_sign
    ideal = (reference_time - raw_time) / (sign * period)
    candidates = tuple(sorted({math.floor(ideal), math.ceil(ideal)}))
    residual_by_cycle = {
        cycle: raw_time + sign * cycle * period - reference_time
        for cycle in candidates
    }
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(raw_time),
        abs(reference_time),
        period,
    )
    branch_tie = (
        len(candidates) == 2
        and math.isclose(
            abs(residual_by_cycle[candidates[0]]),
            abs(residual_by_cycle[candidates[1]]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    )
    best = min(
        candidates,
        key=lambda cycle: (
            abs(residual_by_cycle[cycle]),
            abs(cycle),
            cycle,
        ),
    )
    if branch_tie:
        best = min(candidates, key=lambda cycle: (abs(cycle), cycle))
    corrected = raw_time + sign * best * period
    return CycleResolution(
        convention=convention,
        raw_time_s=raw_time,
        reference_time_s=reference_time,
        period_s=period,
        cycle_count=best,
        corrected_time_s=corrected,
        corrected_residual_s=corrected - reference_time,
        branch_tie=branch_tie,
    )


@dataclass(frozen=True)
class ReferenceFoldAssignment:
    fold_ids: np.ndarray
    distance_quintile_ids: np.ndarray
    azimuth_block_ids: np.ndarray
    training_indices: Tuple[np.ndarray, ...]
    holdout_indices: Tuple[np.ndarray, ...]
    fold_count: int
    assignment_hash: str

    def __post_init__(self) -> None:
        raw = np.asarray(self.fold_ids)
        if raw.dtype.kind not in "iu" or raw.ndim != 1 or raw.size == 0:
            raise ValueError("fold_ids must be a non-empty integer array")
        folds = np.array(raw, dtype=int, copy=True)
        distance_quintiles = np.array(
            self.distance_quintile_ids,
            dtype=int,
            copy=True,
        )
        azimuth_blocks = np.array(
            self.azimuth_block_ids,
            dtype=int,
            copy=True,
        )
        fold_count = _integer_scalar(self.fold_count, "fold_count")
        if (
            fold_count != 5
            or np.any(folds < 0)
            or np.any(folds >= fold_count)
            or set(folds.tolist()) != set(range(fold_count))
        ):
            raise ValueError("fold_ids must assign every one of five folds")
        if (
            distance_quintiles.shape != folds.shape
            or azimuth_blocks.shape != folds.shape
            or np.any(distance_quintiles < 0)
            or np.any(distance_quintiles > 4)
            or np.any(azimuth_blocks < 0)
            or np.any(azimuth_blocks > 7)
        ):
            raise ValueError("fold stratification metadata is invalid")
        training = tuple(
            np.array(value, dtype=int, copy=True)
            for value in self.training_indices
        )
        holdout = tuple(
            np.array(value, dtype=int, copy=True)
            for value in self.holdout_indices
        )
        all_indices = set(range(folds.size))
        if len(training) != 5 or len(holdout) != 5:
            raise ValueError("fold train/holdout audit must contain five folds")
        for fold in range(5):
            expected_holdout = set(np.flatnonzero(folds == fold).tolist())
            if (
                holdout[fold].ndim != 1
                or training[fold].ndim != 1
                or set(holdout[fold].tolist()) != expected_holdout
                or set(training[fold].tolist())
                != all_indices - expected_holdout
            ):
                raise ValueError("fold train/holdout audit is inconsistent")
        digest = hashlib.sha256()
        for array in (folds, distance_quintiles, azimuth_blocks):
            digest.update(np.asarray(array, dtype="<i8").tobytes())
        expected_hash = digest.hexdigest()
        if self.assignment_hash != expected_hash:
            raise ValueError("assignment_hash is inconsistent with fold_ids")
        for array in (
            folds,
            distance_quintiles,
            azimuth_blocks,
            *training,
            *holdout,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "fold_ids", folds)
        object.__setattr__(
            self,
            "distance_quintile_ids",
            distance_quintiles,
        )
        object.__setattr__(self, "azimuth_block_ids", azimuth_blocks)
        object.__setattr__(self, "training_indices", training)
        object.__setattr__(self, "holdout_indices", holdout)
        object.__setattr__(self, "fold_count", fold_count)


def assign_reference_folds(
    distance_km,
    azimuth_deg,
) -> ReferenceFoldAssignment:
    """Assign five folds jointly by distance quintile and 45-degree azimuth block."""

    distance = _strict_finite_real(distance_km, "distance_km")
    azimuth = _strict_finite_real(azimuth_deg, "azimuth_deg")
    if (
        distance.ndim != 1
        or azimuth.ndim != 1
        or distance.shape != azimuth.shape
        or distance.size < 5
        or np.any(distance <= 0)
    ):
        raise ValueError(
            "distance_km and azimuth_deg must be matching one-dimensional "
            "arrays with at least five positive-distance observations"
        )
    thresholds = np.quantile(
        distance,
        (0.2, 0.4, 0.6, 0.8),
        method="linear",
    )
    distance_quintile = np.searchsorted(
        thresholds,
        distance,
        side="left",
    ).astype(int)
    azimuth_block = np.floor(np.mod(azimuth, 360.0) / 45.0).astype(int)
    joint_block = 8 * distance_quintile + azimuth_block
    unique_blocks = np.unique(joint_block)
    if unique_blocks.size < 5:
        raise ValueError("at least five joint distance-azimuth blocks are required")
    fold_ids = np.empty(distance.size, dtype=int)
    for block_order, block_id in enumerate(unique_blocks):
        fold_ids[joint_block == block_id] = block_order % 5
    training = tuple(
        np.flatnonzero(fold_ids != fold) for fold in range(5)
    )
    holdout = tuple(
        np.flatnonzero(fold_ids == fold) for fold in range(5)
    )
    digest = hashlib.sha256()
    for array in (fold_ids, distance_quintile, azimuth_block):
        digest.update(np.asarray(array, dtype="<i8").tobytes())
    assignment_hash = digest.hexdigest()
    return ReferenceFoldAssignment(
        fold_ids=fold_ids,
        distance_quintile_ids=distance_quintile,
        azimuth_block_ids=azimuth_block,
        training_indices=training,
        holdout_indices=holdout,
        fold_count=5,
        assignment_hash=assignment_hash,
    )


@dataclass(frozen=True)
class ReferenceCvConfig:
    lambda_s: float
    lambda_g: float
    fold_holdout_losses: np.ndarray
    mean_holdout_loss: float
    optimizer_calls: int

    def __post_init__(self) -> None:
        allowed = (0.0, 0.001, 0.01, 0.1, 1.0)
        lambda_s = _finite_scalar(self.lambda_s, "lambda_s")
        lambda_g = _finite_scalar(self.lambda_g, "lambda_g")
        if lambda_s not in allowed or lambda_g not in allowed:
            raise ValueError("reference lambdas must use the fixed search grid")
        losses = np.array(
            _strict_finite_real(
                self.fold_holdout_losses,
                "fold_holdout_losses",
            ),
            dtype=float,
            copy=True,
        )
        if losses.shape != (5,) or np.any(losses < 0):
            raise ValueError(
                "fold_holdout_losses must contain five non-negative losses"
            )
        mean_loss = _finite_scalar(
            self.mean_holdout_loss,
            "mean_holdout_loss",
        )
        if mean_loss < 0 or not math.isclose(
            mean_loss,
            float(np.mean(losses)),
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(float).eps * max(1.0, mean_loss),
        ):
            raise ValueError("mean_holdout_loss must equal the fold mean")
        optimizer_calls = _integer_scalar(
            self.optimizer_calls,
            "optimizer_calls",
        )
        if optimizer_calls < 0 or optimizer_calls > 25:
            raise ValueError("optimizer_calls must lie within the 25-call budget")
        losses.setflags(write=False)
        object.__setattr__(self, "lambda_s", lambda_s)
        object.__setattr__(self, "lambda_g", lambda_g)
        object.__setattr__(self, "fold_holdout_losses", losses)
        object.__setattr__(self, "mean_holdout_loss", mean_loss)
        object.__setattr__(self, "optimizer_calls", optimizer_calls)


def select_reference_cv_config(
    configs: Iterable[ReferenceCvConfig],
) -> ReferenceCvConfig:
    """Select within one percent of best, then smaller lambda_g and lambda_s."""

    candidates = tuple(configs)
    if not candidates or any(
        not isinstance(item, ReferenceCvConfig) for item in candidates
    ):
        raise ValueError("configs must contain ReferenceCvConfig values")
    best_loss = min(item.mean_holdout_loss for item in candidates)
    eligible = tuple(
        item
        for item in candidates
        if (
            item.mean_holdout_loss == best_loss
            or (
                best_loss > 0.0
                and (item.mean_holdout_loss - best_loss) / best_loss < 0.01
            )
        )
    )
    return min(
        eligible,
        key=lambda item: (
            item.lambda_g,
            item.lambda_s,
            item.mean_holdout_loss,
        ),
    )


@dataclass(frozen=True)
class ReferenceStart:
    start_index: int
    kind: str
    base_velocity_km_s: float
    endpoint_slope_km_s: float
    velocities_km_s: np.ndarray
    velocity_hash: str

    def __post_init__(self) -> None:
        index = _integer_scalar(self.start_index, "start_index")
        if index < 0:
            raise ValueError("start_index must be non-negative")
        if self.kind not in {"endpoint_linear", "sine_perturbation"}:
            raise ValueError("invalid reference start kind")
        base = _finite_scalar(
            self.base_velocity_km_s,
            "base_velocity_km_s",
            positive=True,
        )
        slope = _finite_scalar(
            self.endpoint_slope_km_s,
            "endpoint_slope_km_s",
        )
        velocity = np.array(
            _strict_finite_real(self.velocities_km_s, "velocities_km_s"),
            dtype=float,
            copy=True,
        )
        if (
            velocity.ndim != 1
            or velocity.size < 3
            or np.any(velocity < 1.6)
            or np.any(velocity > 4.0)
        ):
            raise ValueError("start velocities must lie within 1.6--4.0 km/s")
        expected_hash = hashlib.sha256(
            np.asarray(velocity, dtype="<f8").tobytes()
        ).hexdigest()
        if self.velocity_hash != expected_hash:
            raise ValueError("velocity_hash is inconsistent")
        velocity.setflags(write=False)
        object.__setattr__(self, "start_index", index)
        object.__setattr__(self, "base_velocity_km_s", base)
        object.__setattr__(self, "endpoint_slope_km_s", slope)
        object.__setattr__(self, "velocities_km_s", velocity)


def generate_reference_starts(
    periods_s,
    *,
    max_starts: int = 71,
    seed: int = 20260717,
) -> Tuple[ReferenceStart, ...]:
    """Generate the fixed structured starts followed by seeded sine starts."""

    periods = _strict_finite_real(periods_s, "periods_s")
    if (
        periods.ndim != 1
        or periods.size < 3
        or np.any(periods <= 0)
        or np.any(np.diff(periods) <= 0)
    ):
        raise ValueError("periods_s must be a positive increasing grid")
    maximum = _integer_scalar(max_starts, "max_starts")
    seed_value = _integer_scalar(seed, "seed")
    if maximum < 39 or maximum > 128:
        raise ValueError("max_starts must lie between 39 and 128")
    normalized = np.linspace(-0.5, 0.5, periods.size)
    starts = []
    hashes = set()

    def append(kind, base, slope, velocity):
        clipped = np.clip(np.asarray(velocity, dtype=float), 1.6, 4.0)
        digest = hashlib.sha256(
            np.asarray(clipped, dtype="<f8").tobytes()
        ).hexdigest()
        if digest in hashes:
            return
        hashes.add(digest)
        starts.append(
            ReferenceStart(
                start_index=len(starts),
                kind=kind,
                base_velocity_km_s=float(base),
                endpoint_slope_km_s=float(slope),
                velocities_km_s=clipped,
                velocity_hash=digest,
            )
        )

    for base in np.linspace(1.6, 4.0, 13):
        for slope in (-0.20, 0.0, 0.20):
            append(
                "endpoint_linear",
                base,
                slope,
                base + slope * normalized,
            )
    rng = np.random.default_rng(seed_value)
    coordinate = np.linspace(0.0, 1.0, periods.size)
    while len(starts) < maximum:
        base = float(rng.uniform(1.6, 4.0))
        coefficients = rng.normal(size=3)
        perturbation = sum(
            coefficients[mode - 1] * np.sin(mode * math.pi * coordinate)
            for mode in (1, 2, 3)
        )
        max_amplitude = float(np.max(np.abs(perturbation)))
        requested_amplitude = float(rng.uniform(0.02, 0.15))
        if max_amplitude > 0:
            perturbation *= requested_amplitude / max_amplitude
        append(
            "sine_perturbation",
            base,
            0.0,
            base + perturbation,
        )
    return tuple(starts)


@dataclass(frozen=True)
class ReferenceBasinClustering:
    basin_ids: np.ndarray
    representative_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        ids = np.array(self.basin_ids, dtype=int, copy=True)
        representatives = tuple(
            _integer_scalar(value, "representative_index")
            for value in self.representative_indices
        )
        if (
            ids.ndim != 1
            or ids.size == 0
            or np.any(ids < 0)
            or set(ids.tolist()) != set(range(len(representatives)))
            or any(index < 0 or index >= ids.size for index in representatives)
        ):
            raise ValueError("basin metadata is inconsistent")
        ids.setflags(write=False)
        object.__setattr__(self, "basin_ids", ids)
        object.__setattr__(self, "representative_indices", representatives)


def cluster_reference_solutions(solutions) -> ReferenceBasinClustering:
    """Cluster local solutions using both strict velocity and slowness thresholds."""

    rows = tuple(solutions)
    if not rows:
        raise ValueError("solutions must not be empty")
    basin_ids = np.full(len(rows), -1, dtype=int)
    basin_members = []
    for index, row in enumerate(rows):
        velocity = _strict_finite_real(
            row.target_velocities_km_s,
            "target_velocities_km_s",
        )
        slowness = _strict_finite_real(
            row.phase_slowness_s_km,
            "phase_slowness_s_km",
        )
        if velocity.shape != (4,) or slowness.ndim != 1:
            raise ValueError("solution arrays have invalid shapes")
        assigned = None
        for basin_index, member_indices in enumerate(basin_members):
            compatible_with_every_member = all(
                float(
                    np.max(
                        np.abs(
                            velocity
                            - np.asarray(
                                rows[member_index].target_velocities_km_s,
                                dtype=float,
                            )
                        )
                    )
                )
                < 0.02
                and float(
                    np.sqrt(
                        np.mean(
                            (
                                slowness
                                - np.asarray(
                                    rows[member_index].phase_slowness_s_km,
                                    dtype=float,
                                )
                            )
                            ** 2
                        )
                    )
                )
                < 0.002
                for member_index in member_indices
            )
            if compatible_with_every_member:
                assigned = basin_index
                break
        if assigned is None:
            assigned = len(basin_members)
            basin_members.append([])
        basin_members[assigned].append(index)
        basin_ids[index] = assigned
    representatives = tuple(
        min(
            members,
            key=lambda index: (
                _finite_scalar(rows[index].objective, "objective"),
                index,
            ),
        )
        for members in basin_members
    )
    return ReferenceBasinClustering(
        basin_ids=basin_ids,
        representative_indices=representatives,
    )


@dataclass(frozen=True)
class ReferenceObservation:
    pair_name: str
    distance_km: float
    azimuth_deg: float
    instantaneous_period_s: float
    anchored_raw_time_s: float
    group_slowness_s_km: float
    convention: PhaseConvention

    def __post_init__(self) -> None:
        if not isinstance(self.pair_name, str) or not self.pair_name:
            raise ValueError("pair_name must be a non-empty string")
        for name in (
            "distance_km",
            "instantaneous_period_s",
            "anchored_raw_time_s",
            "group_slowness_s_km",
        ):
            object.__setattr__(
                self,
                name,
                _finite_scalar(getattr(self, name), name, positive=True),
            )
        object.__setattr__(
            self,
            "azimuth_deg",
            _finite_scalar(self.azimuth_deg, "azimuth_deg"),
        )
        _require_phase_convention(self.convention)


def _reference_grid(periods_s) -> np.ndarray:
    periods = np.array(
        _strict_finite_real(periods_s, "periods_s"),
        dtype=float,
        copy=True,
    )
    if (
        periods.ndim != 1
        or periods.size < 3
        or np.any(periods <= 0)
        or np.any(np.diff(periods) <= 0)
    ):
        raise ValueError("periods_s must be a positive increasing grid")
    return periods


def _immutable_numeric_array(values, dtype) -> np.ndarray:
    """Copy an array onto immutable bytes-backed storage."""

    contiguous = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    immutable = np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)
    immutable.setflags(write=False)
    return immutable


_PREPARED_REFERENCE_ARRAY_DTYPES = (
    ("periods_s", float),
    ("interpolation_left_indices", np.intp),
    ("interpolation_right_indices", np.intp),
    ("interpolation_left_weights", float),
    ("interpolation_right_weights", float),
    ("observation_periods_s", float),
    ("distance_km", float),
    ("anchored_raw_time_s", float),
    ("distance_over_period", float),
    ("raw_phase_cycles", float),
    ("group_grid_indices", np.intp),
    ("group_observed_medians", float),
    ("second_difference_matrix", float),
    ("group_slowness_operator", float),
)


def _prepared_reference_operators(
    periods: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    size = periods.size
    second_difference = np.zeros((size - 2, size), dtype=float)
    rows = np.arange(size - 2)
    second_difference[rows, rows] = 1.0
    second_difference[rows, rows + 1] = -2.0
    second_difference[rows, rows + 2] = 1.0
    omega = 2.0 * math.pi / periods
    derivative = np.gradient(
        np.eye(size, dtype=float),
        omega,
        axis=0,
        edge_order=2,
    )
    group_slowness = (
        np.eye(size, dtype=float)
        + omega[:, np.newaxis] * derivative
    )
    return second_difference, group_slowness


def _prepared_reference_semantic_sha256(arrays, observation_count: int) -> str:
    digest = hashlib.sha256(b"PreparedReferenceObjective|semantic-v1")
    for name, _ in _PREPARED_REFERENCE_ARRAY_DTYPES:
        array = np.asarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    digest.update(b"observation_count|int|")
    digest.update(str(observation_count).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedReferenceObjective:
    """Auditable, immutable arrays reused by reference-objective calls.

    Preparation depends only on the observations and reference-period grid.
    Candidate slowness and regularization weights deliberately remain inputs to
    :func:`prepared_reference_objective_value`.
    """

    periods_s: np.ndarray
    interpolation_left_indices: np.ndarray
    interpolation_right_indices: np.ndarray
    interpolation_left_weights: np.ndarray
    interpolation_right_weights: np.ndarray
    observation_periods_s: np.ndarray
    distance_km: np.ndarray
    anchored_raw_time_s: np.ndarray
    distance_over_period: np.ndarray
    raw_phase_cycles: np.ndarray
    group_grid_indices: np.ndarray
    group_observed_medians: np.ndarray
    second_difference_matrix: np.ndarray
    group_slowness_operator: np.ndarray
    observation_count: int
    semantic_sha256: str

    def __reduce__(self):
        return (
            _reconstruct_prepared_reference_objective,
            tuple(getattr(self, field.name) for field in fields(self)),
        )

    def __post_init__(self) -> None:
        count = _integer_scalar(self.observation_count, "observation_count")
        arrays = {
            name: _immutable_numeric_array(getattr(self, name), dtype)
            for name, dtype in _PREPARED_REFERENCE_ARRAY_DTYPES
        }
        periods = arrays["periods_s"]
        left_indices = arrays["interpolation_left_indices"]
        right_indices = arrays["interpolation_right_indices"]
        left_weights = arrays["interpolation_left_weights"]
        right_weights = arrays["interpolation_right_weights"]
        observation_periods = arrays["observation_periods_s"]
        distance = arrays["distance_km"]
        raw_time = arrays["anchored_raw_time_s"]
        distance_over_period = arrays["distance_over_period"]
        raw_phase_cycles = arrays["raw_phase_cycles"]
        group_indices = arrays["group_grid_indices"]
        group_medians = arrays["group_observed_medians"]
        second_difference = arrays["second_difference_matrix"]
        group_operator = arrays["group_slowness_operator"]
        size = periods.size
        observation_shape = (count,)
        if (
            count <= 0
            or periods.ndim != 1
            or size < 3
            or np.any(~np.isfinite(periods))
            or np.any(periods <= 0)
            or np.any(np.diff(periods) <= 0)
            or any(
                array.shape != observation_shape
                for array in (
                    left_indices,
                    right_indices,
                    left_weights,
                    right_weights,
                    observation_periods,
                    distance,
                    raw_time,
                    distance_over_period,
                    raw_phase_cycles,
                )
            )
            or group_indices.ndim != 1
            or group_indices.size == 0
            or group_medians.shape != group_indices.shape
            or second_difference.shape != (size - 2, size)
            or group_operator.shape != (size, size)
            or np.any(left_indices < 0)
            or np.any(left_indices >= size)
            or np.any(right_indices < 0)
            or np.any(right_indices >= size)
            or np.any(left_indices > right_indices)
            or np.any(group_indices < 0)
            or np.any(group_indices >= size)
            or np.any(np.diff(group_indices) <= 0)
            or np.any(~np.isfinite(left_weights))
            or np.any(~np.isfinite(right_weights))
            or np.any(left_weights < 0)
            or np.any(right_weights < 0)
            or np.any(left_weights > 1)
            or np.any(right_weights > 1)
            or np.any(left_weights + right_weights != 1.0)
            or np.any(~np.isfinite(observation_periods))
            or np.any(observation_periods < periods[0])
            or np.any(observation_periods > periods[-1])
            or np.any(~np.isfinite(distance))
            or np.any(distance <= 0)
            or np.any(~np.isfinite(raw_time))
            or np.any(raw_time <= 0)
            or np.any(~np.isfinite(distance_over_period))
            or np.any(distance_over_period <= 0)
            or np.any(~np.isfinite(raw_phase_cycles))
            or np.any(raw_phase_cycles <= 0)
            or np.any(~np.isfinite(group_medians))
            or np.any(group_medians <= 0)
            or np.any(~np.isfinite(second_difference))
            or np.any(~np.isfinite(group_operator))
        ):
            raise ValueError("prepared reference objective arrays are inconsistent")
        exact_interpolation = left_indices == right_indices
        interpolated = ~exact_interpolation
        expected_right_indices = np.searchsorted(
            periods,
            observation_periods,
            side="left",
        )
        expected_right_indices = np.minimum(
            expected_right_indices,
            periods.size - 1,
        )
        expected_exact = (
            periods[expected_right_indices] == observation_periods
        )
        expected_left_indices = np.where(
            expected_exact,
            expected_right_indices,
            expected_right_indices - 1,
        )
        expected_right_weights = np.zeros(count, dtype=float)
        expected_interpolated = ~expected_exact
        expected_right_weights[expected_interpolated] = (
            observation_periods[expected_interpolated]
            - periods[expected_left_indices[expected_interpolated]]
        ) / (
            periods[expected_right_indices[expected_interpolated]]
            - periods[expected_left_indices[expected_interpolated]]
        )
        expected_left_weights = 1.0 - expected_right_weights
        nearest_grid_indices = np.argmin(
            np.abs(periods[:, np.newaxis] - observation_periods),
            axis=0,
        )
        expected_second_difference, expected_group_operator = (
            _prepared_reference_operators(periods)
        )
        if (
            np.any(right_indices[interpolated] != left_indices[interpolated] + 1)
            or not np.array_equal(left_indices, expected_left_indices)
            or not np.array_equal(right_indices, expected_right_indices)
            or not np.array_equal(left_weights, expected_left_weights)
            or not np.array_equal(right_weights, expected_right_weights)
            or np.any(left_weights[exact_interpolation] != 1.0)
            or np.any(right_weights[exact_interpolation] != 0.0)
            or np.any(left_weights[interpolated] <= 0.0)
            or np.any(right_weights[interpolated] <= 0.0)
            or not np.array_equal(
                distance_over_period,
                distance / observation_periods,
            )
            or not np.array_equal(
                raw_phase_cycles,
                raw_time / observation_periods,
            )
            or not np.array_equal(
                np.unique(nearest_grid_indices),
                group_indices,
            )
            or not np.array_equal(
                second_difference,
                expected_second_difference,
            )
            or not np.array_equal(group_operator, expected_group_operator)
        ):
            raise ValueError(
                "prepared reference objective semantics are inconsistent"
            )
        semantic_sha256 = self.semantic_sha256
        if (
            not isinstance(semantic_sha256, str)
            or len(semantic_sha256) != 64
            or any(character not in "0123456789abcdef" for character in semantic_sha256)
            or semantic_sha256
            != _prepared_reference_semantic_sha256(arrays, count)
        ):
            raise ValueError(
                "prepared reference objective semantic hash is inconsistent"
            )
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "observation_count", count)


def _reconstruct_prepared_reference_objective(
    *values,
) -> PreparedReferenceObjective:
    return PreparedReferenceObjective(*values)


def _make_prepared_reference_objective(
    *,
    observation_count: int,
    **values,
) -> PreparedReferenceObjective:
    count = _integer_scalar(observation_count, "observation_count")
    arrays = {
        name: _immutable_numeric_array(values[name], dtype)
        for name, dtype in _PREPARED_REFERENCE_ARRAY_DTYPES
    }
    return PreparedReferenceObjective(
        **arrays,
        observation_count=count,
        semantic_sha256=_prepared_reference_semantic_sha256(arrays, count),
    )


def prepare_reference_objective(
    observations: Iterable[ReferenceObservation],
    *,
    periods_s=None,
) -> PreparedReferenceObjective:
    """Precompute observation-only arrays for repeated objective values."""

    periods = _reference_grid(
        _deterministic_inclusive_grid(2.5, 5.0, 0.05)
        if periods_s is None
        else periods_s
    )
    rows = tuple(observations)
    if not rows or any(not isinstance(row, ReferenceObservation) for row in rows):
        raise ValueError("reference observations are invalid")
    observation_periods = np.asarray(
        [row.instantaneous_period_s for row in rows],
        dtype=float,
    )
    if np.any(observation_periods < periods[0]) or np.any(
        observation_periods > periods[-1]
    ):
        raise ValueError("observation periods must lie on the reference grid span")

    right_indices = np.searchsorted(
        periods,
        observation_periods,
        side="left",
    )
    right_indices = np.minimum(right_indices, periods.size - 1)
    exact = periods[right_indices] == observation_periods
    left_indices = np.where(exact, right_indices, right_indices - 1)
    right_weights = np.zeros(observation_periods.size, dtype=float)
    interpolated = ~exact
    right_weights[interpolated] = (
        observation_periods[interpolated] - periods[left_indices[interpolated]]
    ) / (
        periods[right_indices[interpolated]]
        - periods[left_indices[interpolated]]
    )
    left_weights = 1.0 - right_weights

    distance = np.asarray([row.distance_km for row in rows], dtype=float)
    raw_time = np.asarray(
        [row.anchored_raw_time_s for row in rows],
        dtype=float,
    )
    observation_grid_indices = np.asarray(
        [
            int(np.argmin(np.abs(periods - row.instantaneous_period_s)))
            for row in rows
        ],
        dtype=np.intp,
    )
    group_grid_indices = np.unique(observation_grid_indices)
    group_observed_medians = np.asarray(
        [
            np.median(
                [
                    row.group_slowness_s_km
                    for row, index in zip(rows, observation_grid_indices)
                    if index == grid_index
                ]
            )
            for grid_index in group_grid_indices
        ],
        dtype=float,
    )

    second_difference_matrix, group_slowness_operator = (
        _prepared_reference_operators(periods)
    )
    return _make_prepared_reference_objective(
        periods_s=periods,
        interpolation_left_indices=left_indices,
        interpolation_right_indices=right_indices,
        interpolation_left_weights=left_weights,
        interpolation_right_weights=right_weights,
        observation_periods_s=observation_periods,
        distance_km=distance,
        anchored_raw_time_s=raw_time,
        distance_over_period=distance / observation_periods,
        raw_phase_cycles=raw_time / observation_periods,
        group_grid_indices=group_grid_indices,
        group_observed_medians=group_observed_medians,
        second_difference_matrix=second_difference_matrix,
        group_slowness_operator=group_slowness_operator,
        observation_count=len(rows),
    )


def prepared_reference_objective_value(
    phase_slowness_s_km,
    prepared: PreparedReferenceObjective,
    *,
    lambda_s: float = 0.01,
    lambda_g: float = 0.1,
) -> float:
    """Evaluate a prepared reference objective without rebuilding arrays."""

    if not isinstance(prepared, PreparedReferenceObjective):
        raise ValueError("prepared must be a PreparedReferenceObjective")
    current_semantic_sha256 = _prepared_reference_semantic_sha256(
        {
            name: getattr(prepared, name)
            for name, _ in _PREPARED_REFERENCE_ARRAY_DTYPES
        },
        prepared.observation_count,
    )
    if (
        not isinstance(prepared.semantic_sha256, str)
        or prepared.semantic_sha256 != current_semantic_sha256
    ):
        raise ValueError("prepared semantic hash is inconsistent")
    slowness = _strict_finite_real(
        phase_slowness_s_km,
        "phase_slowness_s_km",
    )
    if (
        slowness.shape != prepared.periods_s.shape
        or np.any(slowness < 1.0 / 4.0)
        or np.any(slowness > 1.0 / 1.6)
    ):
        raise ValueError("reference slowness is invalid")
    lambda_s_value = _finite_scalar(lambda_s, "lambda_s")
    lambda_g_value = _finite_scalar(lambda_g, "lambda_g")
    if lambda_s_value < 0 or lambda_g_value < 0:
        raise ValueError("reference lambdas must be non-negative")

    predicted_phase_slowness = (
        prepared.interpolation_left_weights
        * slowness[prepared.interpolation_left_indices]
        + prepared.interpolation_right_weights
        * slowness[prepared.interpolation_right_indices]
    )
    circular_cycles = (
        (
            prepared.raw_phase_cycles
            - prepared.distance_over_period * predicted_phase_slowness
            + 0.5
        )
        % 1.0
        - 0.5
    )
    circular_cycles[
        np.abs(circular_cycles) <= 64.0 * np.finfo(float).eps
    ] = 0.0
    circular_magnitude = np.abs(circular_cycles)
    data_loss = float(
        np.mean(
            np.where(
                circular_magnitude <= 0.10,
                0.5 * circular_cycles * circular_cycles,
                0.10 * (circular_magnitude - 0.05),
            )
        )
    )
    second_difference = prepared.second_difference_matrix @ slowness
    curvature_loss = float(np.mean((second_difference / 0.01) ** 2))
    group_slowness = prepared.group_slowness_operator @ slowness
    group_residual = (
        group_slowness[prepared.group_grid_indices]
        - prepared.group_observed_medians
    ) / 0.02
    group_magnitude = np.abs(group_residual)
    group_loss = float(
        np.mean(
            np.where(
                group_magnitude <= 1.0,
                0.5 * group_residual * group_residual,
                group_magnitude - 0.5,
            )
        )
    )
    return float(
        data_loss
        + lambda_s_value * curvature_loss
        + lambda_g_value * group_loss
    )


def _prepared_reference_legacy_compatible_value_unchecked(
    slowness: np.ndarray,
    prepared: PreparedReferenceObjective,
    lambda_s_value: float,
    lambda_g_value: float,
) -> float:
    """Reproduce the frozen scalar objective's operation order exactly."""

    predicted_phase_slowness = np.interp(
        prepared.observation_periods_s,
        prepared.periods_s,
        slowness,
    )
    circular_cycles = wrap_periodic(
        (
            prepared.anchored_raw_time_s
            - prepared.distance_km * predicted_phase_slowness
        )
        / prepared.observation_periods_s,
        1.0,
    )
    circular_cycles = np.asarray(circular_cycles, dtype=float)
    circular_cycles[
        np.abs(circular_cycles) <= 64.0 * np.finfo(float).eps
    ] = 0.0
    data_loss = float(
        np.mean(huber_loss(circular_cycles, delta=0.10))
    )
    curvature_loss = float(
        np.mean((np.diff(slowness, n=2) / 0.01) ** 2)
    )
    full_group_slowness = phase_slowness_to_group_slowness(
        prepared.periods_s,
        slowness,
    )
    group_residuals = (
        full_group_slowness[prepared.group_grid_indices]
        - prepared.group_observed_medians
    ) / 0.02
    group_loss = float(
        np.mean(huber_loss(group_residuals, delta=1.0))
    )
    return float(
        data_loss
        + lambda_s_value * curvature_loss
        + lambda_g_value * group_loss
    )


def _prepared_reference_legacy_compatible_values_unchecked(
    phase_slowness_rows: np.ndarray,
    prepared: PreparedReferenceObjective,
    lambda_s_value: float,
    lambda_g_value: float,
) -> np.ndarray:
    """Evaluate legacy-compatible values for a small candidate batch."""

    candidates = np.asarray(phase_slowness_rows, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1:] != prepared.periods_s.shape:
        raise ValueError("reference candidate batch is invalid")
    predicted_phase_slowness = np.asarray(
        [
            np.interp(
                prepared.observation_periods_s,
                prepared.periods_s,
                candidate,
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    circular_cycles = wrap_periodic(
        (
            prepared.anchored_raw_time_s[np.newaxis, :]
            - prepared.distance_km[np.newaxis, :] * predicted_phase_slowness
        )
        / prepared.observation_periods_s[np.newaxis, :],
        1.0,
    )
    circular_cycles = np.asarray(circular_cycles, dtype=np.float64)
    circular_cycles[
        np.abs(circular_cycles) <= 64.0 * np.finfo(float).eps
    ] = 0.0
    data_loss = np.mean(
        huber_loss(circular_cycles, delta=0.10),
        axis=1,
    )
    curvature_loss = np.mean(
        (np.diff(candidates, n=2, axis=1) / 0.01) ** 2,
        axis=1,
    )
    omega = 2.0 * math.pi / prepared.periods_s
    group_slowness = candidates + omega[np.newaxis, :] * np.gradient(
        candidates,
        omega,
        axis=1,
        edge_order=2,
    )
    group_residuals = (
        group_slowness[:, prepared.group_grid_indices]
        - prepared.group_observed_medians[np.newaxis, :]
    ) / 0.02
    group_huber = huber_loss(group_residuals, delta=1.0)
    group_loss = np.asarray(
        [float(np.mean(row)) for row in group_huber],
        dtype=np.float64,
    )
    return np.asarray(
        data_loss
        + lambda_s_value * curvature_loss
        + lambda_g_value * group_loss,
        dtype=np.float64,
    )


def _validated_legacy_compatible_inputs(
    phase_slowness_s_km,
    prepared: PreparedReferenceObjective,
    lambda_s,
    lambda_g,
) -> Tuple[np.ndarray, float, float]:
    if not isinstance(prepared, PreparedReferenceObjective):
        raise ValueError("prepared must be a PreparedReferenceObjective")
    current_semantic_sha256 = _prepared_reference_semantic_sha256(
        {
            name: getattr(prepared, name)
            for name, _ in _PREPARED_REFERENCE_ARRAY_DTYPES
        },
        prepared.observation_count,
    )
    if (
        not isinstance(prepared.semantic_sha256, str)
        or prepared.semantic_sha256 != current_semantic_sha256
    ):
        raise ValueError("prepared semantic hash is inconsistent")
    slowness = _strict_finite_real(
        phase_slowness_s_km,
        "phase_slowness_s_km",
    )
    if (
        slowness.shape != prepared.periods_s.shape
        or np.any(slowness < 1.0 / 4.0)
        or np.any(slowness > 1.0 / 1.6)
    ):
        raise ValueError("reference slowness is invalid")
    lambda_s_value = _finite_scalar(lambda_s, "lambda_s")
    lambda_g_value = _finite_scalar(lambda_g, "lambda_g")
    if lambda_s_value < 0 or lambda_g_value < 0:
        raise ValueError("reference lambdas must be non-negative")
    return slowness, lambda_s_value, lambda_g_value


def prepared_reference_legacy_compatible_value(
    phase_slowness_s_km,
    prepared: PreparedReferenceObjective,
    *,
    lambda_s: float = 0.01,
    lambda_g: float = 0.1,
) -> float:
    """Return the prepared value with frozen legacy scalar bit semantics."""

    slowness, lambda_s_value, lambda_g_value = (
        _validated_legacy_compatible_inputs(
            phase_slowness_s_km,
            prepared,
            lambda_s,
            lambda_g,
        )
    )
    return _prepared_reference_legacy_compatible_value_unchecked(
        slowness,
        prepared,
        lambda_s_value,
        lambda_g_value,
    )


def prepared_reference_legacy_compatible_value_and_gradient(
    phase_slowness_s_km,
    prepared: PreparedReferenceObjective,
    lambda_s: float = 0.01,
    lambda_g: float = 0.1,
) -> Tuple[float, np.ndarray]:
    """Return the legacy value and its frozen SciPy two-point Jacobian.

    This compatibility Jacobian deliberately reproduces the historical
    ``L-BFGS-B`` ``eps=1e-8`` forward-difference trajectory.  It is distinct
    from :func:`prepared_reference_objective_value_and_gradient`, which
    returns the mathematical analytic subgradient.
    """

    slowness, lambda_s_value, lambda_g_value = (
        _validated_legacy_compatible_inputs(
            phase_slowness_s_km,
            prepared,
            lambda_s,
            lambda_g,
        )
    )
    value = _prepared_reference_legacy_compatible_value_unchecked(
        slowness,
        prepared,
        lambda_s_value,
        lambda_g_value,
    )
    step = np.full(slowness.shape, 1e-8, dtype=np.float64)
    delta = (slowness + step) - slowness
    zero_step = delta == 0.0
    if np.any(zero_step):
        sign = (slowness >= 0).astype(float) * 2.0 - 1.0
        step[zero_step] = (
            math.sqrt(np.finfo(np.float64).eps)
            * sign[zero_step]
            * np.maximum(1.0, np.abs(slowness[zero_step]))
        )
    lower_distance = slowness - 1.0 / 4.0
    upper_distance = 1.0 / 1.6 - slowness
    proposed = slowness + step
    violated = (proposed < 1.0 / 4.0) | (proposed > 1.0 / 1.6)
    fitting = np.abs(step) <= np.maximum(lower_distance, upper_distance)
    step[violated & fitting] *= -1.0
    forward = (upper_distance >= lower_distance) & ~fitting
    step[forward] = upper_distance[forward]
    backward = (upper_distance < lower_distance) & ~fitting
    step[backward] = -lower_distance[backward]

    shifted_candidates = np.repeat(
        slowness[np.newaxis, :],
        slowness.size,
        axis=0,
    )
    for index in range(slowness.size):
        shifted_candidates[index, index] += step[index]
    shifted_values = _prepared_reference_legacy_compatible_values_unchecked(
        shifted_candidates,
        prepared,
        lambda_s_value,
        lambda_g_value,
    )
    actual_step = np.diag(
        shifted_candidates - slowness[np.newaxis, :]
    )
    gradient = (shifted_values - value) / actual_step
    return value, gradient


def prepared_reference_objective_value_and_gradient(
    phase_slowness_s_km,
    prepared: PreparedReferenceObjective,
    lambda_s: float = 0.01,
    lambda_g: float = 0.1,
) -> Tuple[float, np.ndarray]:
    """Evaluate the prepared objective and its frozen analytic subgradient."""

    if not isinstance(prepared, PreparedReferenceObjective):
        raise ValueError("prepared must be a PreparedReferenceObjective")
    current_semantic_sha256 = _prepared_reference_semantic_sha256(
        {
            name: getattr(prepared, name)
            for name, _ in _PREPARED_REFERENCE_ARRAY_DTYPES
        },
        prepared.observation_count,
    )
    if (
        not isinstance(prepared.semantic_sha256, str)
        or prepared.semantic_sha256 != current_semantic_sha256
    ):
        raise ValueError("prepared semantic hash is inconsistent")
    slowness = _strict_finite_real(
        phase_slowness_s_km,
        "phase_slowness_s_km",
    )
    if (
        slowness.shape != prepared.periods_s.shape
        or np.any(slowness < 1.0 / 4.0)
        or np.any(slowness > 1.0 / 1.6)
    ):
        raise ValueError("reference slowness is invalid")
    lambda_s_value = _finite_scalar(lambda_s, "lambda_s")
    lambda_g_value = _finite_scalar(lambda_g, "lambda_g")
    if lambda_s_value < 0 or lambda_g_value < 0:
        raise ValueError("reference lambdas must be non-negative")

    predicted_phase_slowness = (
        prepared.interpolation_left_weights
        * slowness[prepared.interpolation_left_indices]
        + prepared.interpolation_right_weights
        * slowness[prepared.interpolation_right_indices]
    )
    circular_cycles = (
        (
            prepared.raw_phase_cycles
            - prepared.distance_over_period * predicted_phase_slowness
            + 0.5
        )
        % 1.0
        - 0.5
    )
    snap_tolerance = 64.0 * np.finfo(np.float64).eps
    circular_cycles[np.abs(circular_cycles) <= snap_tolerance] = 0.0
    circular_magnitude = np.abs(circular_cycles)
    data_loss = float(
        np.mean(
            np.where(
                circular_magnitude <= 0.10,
                0.5 * circular_cycles * circular_cycles,
                0.10 * (circular_magnitude - 0.05),
            )
        )
    )
    data_influence = np.where(
        circular_magnitude <= 0.10,
        circular_cycles,
        0.10 * np.sign(circular_cycles),
    )
    wrap_discontinuity = (
        np.abs(circular_magnitude - 0.5) <= snap_tolerance
    )
    data_influence[wrap_discontinuity] = 0.0
    predicted_gradient = (
        -prepared.distance_over_period
        * data_influence
        / float(prepared.observation_count)
    )
    gradient = np.zeros(slowness.shape, dtype=np.float64)
    np.add.at(
        gradient,
        prepared.interpolation_left_indices,
        prepared.interpolation_left_weights * predicted_gradient,
    )
    np.add.at(
        gradient,
        prepared.interpolation_right_indices,
        prepared.interpolation_right_weights * predicted_gradient,
    )

    second_difference = prepared.second_difference_matrix @ slowness
    curvature_loss = float(np.mean((second_difference / 0.01) ** 2))
    if lambda_s_value:
        gradient += (
            lambda_s_value
            * (prepared.second_difference_matrix.T @ second_difference)
            * (
                2.0
                / (
                    float(second_difference.size)
                    * 0.01
                    * 0.01
                )
            )
        )

    group_slowness = prepared.group_slowness_operator @ slowness
    group_residual = (
        group_slowness[prepared.group_grid_indices]
        - prepared.group_observed_medians
    ) / 0.02
    group_magnitude = np.abs(group_residual)
    group_loss = float(
        np.mean(
            np.where(
                group_magnitude <= 1.0,
                0.5 * group_residual * group_residual,
                group_magnitude - 0.5,
            )
        )
    )
    if lambda_g_value:
        group_influence = np.where(
            group_magnitude <= 1.0,
            group_residual,
            np.sign(group_residual),
        )
        group_grid_gradient = np.zeros(slowness.shape, dtype=np.float64)
        group_grid_gradient[prepared.group_grid_indices] = (
            group_influence / (float(group_residual.size) * 0.02)
        )
        gradient += (
            lambda_g_value
            * (prepared.group_slowness_operator.T @ group_grid_gradient)
        )

    value = float(
        data_loss
        + lambda_s_value * curvature_loss
        + lambda_g_value * group_loss
    )
    return value, np.asarray(gradient, dtype=np.float64)


def reference_fit_objective(
    phase_slowness_s_km,
    observations: Iterable[ReferenceObservation],
    *,
    lambda_s: float,
    lambda_g: float,
    periods_s=None,
) -> float:
    """Circular phase-data objective with smoothness and derived-group terms."""

    periods = _reference_grid(
        _deterministic_inclusive_grid(2.5, 5.0, 0.05)
        if periods_s is None
        else periods_s
    )
    slowness = _strict_finite_real(
        phase_slowness_s_km,
        "phase_slowness_s_km",
    )
    rows = tuple(observations)
    if (
        slowness.shape != periods.shape
        or np.any(slowness < 1.0 / 4.0)
        or np.any(slowness > 1.0 / 1.6)
        or not rows
        or any(not isinstance(row, ReferenceObservation) for row in rows)
    ):
        raise ValueError("reference slowness or observations are invalid")
    lambda_s_value = _finite_scalar(lambda_s, "lambda_s")
    lambda_g_value = _finite_scalar(lambda_g, "lambda_g")
    if lambda_s_value < 0 or lambda_g_value < 0:
        raise ValueError("reference lambdas must be non-negative")
    observation_periods = np.asarray(
        [row.instantaneous_period_s for row in rows],
        dtype=float,
    )
    if np.any(observation_periods < periods[0]) or np.any(
        observation_periods > periods[-1]
    ):
        raise ValueError("observation periods must lie on the reference grid span")
    data_loss = reference_phase_holdout_loss(
        slowness,
        rows,
        periods_s=periods,
    )
    curvature_loss = float(
        np.mean((np.diff(slowness, n=2) / 0.01) ** 2)
    )
    full_group_slowness = phase_slowness_to_group_slowness(
        periods,
        slowness,
    )
    grid_indices = np.asarray(
        [
            int(np.argmin(np.abs(periods - row.instantaneous_period_s)))
            for row in rows
        ],
        dtype=int,
    )
    group_residuals = []
    for grid_index in np.unique(grid_indices):
        observed_median = float(
            np.median(
                [
                    row.group_slowness_s_km
                    for row, index in zip(rows, grid_indices)
                    if index == grid_index
                ]
            )
        )
        group_residuals.append(
            (full_group_slowness[grid_index] - observed_median) / 0.02
        )
    group_loss = float(
        np.mean(huber_loss(np.asarray(group_residuals), delta=1.0))
    )
    return float(
        data_loss
        + lambda_s_value * curvature_loss
        + lambda_g_value * group_loss
    )


def reference_phase_holdout_loss(
    phase_slowness_s_km,
    observations: Iterable[ReferenceObservation],
    *,
    periods_s=None,
) -> float:
    """Return phase-only circular Huber loss for held-out observations."""

    periods = _reference_grid(
        _deterministic_inclusive_grid(2.5, 5.0, 0.05)
        if periods_s is None
        else periods_s
    )
    slowness = _strict_finite_real(
        phase_slowness_s_km,
        "phase_slowness_s_km",
    )
    rows = tuple(observations)
    if slowness.shape != periods.shape or not rows:
        raise ValueError("phase holdout inputs are invalid")
    observation_periods = np.asarray(
        [row.instantaneous_period_s for row in rows],
        dtype=float,
    )
    predicted_phase_slowness = np.interp(
        observation_periods,
        periods,
        slowness,
    )
    distance = np.asarray([row.distance_km for row in rows], dtype=float)
    raw_time = np.asarray([row.anchored_raw_time_s for row in rows], dtype=float)
    circular_cycles = wrap_periodic(
        (raw_time - distance * predicted_phase_slowness)
        / observation_periods,
        1.0,
    )
    circular_cycles = np.asarray(circular_cycles, dtype=float)
    circular_cycles[
        np.abs(circular_cycles) <= 64.0 * np.finfo(float).eps
    ] = 0.0
    return float(np.mean(huber_loss(circular_cycles, delta=0.10)))


def reference_final_fold_holdout_losses(
    phase_slowness_s_km,
    observations: Iterable[ReferenceObservation],
    fold_assignment: ReferenceFoldAssignment,
    *,
    periods_s,
) -> Tuple[np.ndarray, float]:
    """Evaluate one final solution on each frozen fold without refitting."""

    rows = tuple(observations)
    if (
        not isinstance(fold_assignment, ReferenceFoldAssignment)
        or len(rows) != fold_assignment.fold_ids.size
    ):
        raise ValueError("final fold-loss inputs are inconsistent")
    losses = np.asarray(
        [
            reference_phase_holdout_loss(
                phase_slowness_s_km,
                tuple(rows[index] for index in holdout),
                periods_s=periods_s,
            )
            for holdout in fold_assignment.holdout_indices
        ],
        dtype=float,
    )
    losses.setflags(write=False)
    return losses, float(np.mean(losses))


def _default_reference_optimizer(objective, x0, bounds, maxiter):
    candidate = np.asarray(x0, dtype=float)
    # SciPy's scalar two-point path charges one value plus one value per
    # coordinate to maxfun.  Charge the tuple/Jacobian path the same 52-call
    # budget so convergence status remains part of the frozen science result.
    legacy_compatible_maxfun = 15000 // (candidate.size + 1)
    return minimize(
        objective,
        candidate,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": int(maxiter),
            "maxfun": legacy_compatible_maxfun,
        },
    )


@dataclass(frozen=True)
class ReferenceCvResult:
    fold_assignment: ReferenceFoldAssignment
    configs: Tuple[ReferenceCvConfig, ...]
    selected: ReferenceCvConfig
    optimizer_calls: int
    result_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.fold_assignment, ReferenceFoldAssignment):
            raise ValueError("fold_assignment is invalid")
        configs = tuple(self.configs)
        if not configs or any(
            not isinstance(item, ReferenceCvConfig) for item in configs
        ):
            raise ValueError("configs must not be empty")
        if self.selected not in configs:
            raise ValueError("selected config must belong to configs")
        if self.selected != select_reference_cv_config(configs):
            raise ValueError("selected config is inconsistent with CV losses")
        calls = _integer_scalar(self.optimizer_calls, "optimizer_calls")
        if calls != sum(item.optimizer_calls for item in configs) or calls > 625:
            raise ValueError("optimizer_calls exceeds or mismatches the CV budget")
        digest_source = "|".join(
            (
                self.fold_assignment.assignment_hash,
                *(
                    f"{item.lambda_s:g},{item.lambda_g:g},"
                    f"{item.mean_holdout_loss:.17g},{item.optimizer_calls},"
                    + ",".join(
                        f"{loss:.17g}"
                        for loss in item.fold_holdout_losses
                    )
                    for item in configs
                ),
            )
        )
        expected_hash = hashlib.sha256(digest_source.encode("ascii")).hexdigest()
        if self.result_hash != expected_hash:
            raise ValueError("result_hash is inconsistent")
        object.__setattr__(self, "configs", configs)
        object.__setattr__(self, "optimizer_calls", calls)


def cross_validate_reference_fit(
    observations: Iterable[ReferenceObservation],
    *,
    lambda_values=(0.0, 0.001, 0.01, 0.1, 1.0),
    optimizer=None,
) -> ReferenceCvResult:
    """Run fixed five-fold/five-start CV within the 625-call budget."""

    rows = tuple(observations)
    if len(rows) < 5 or any(
        not isinstance(row, ReferenceObservation) for row in rows
    ):
        raise ValueError("at least five reference observations are required")
    lambdas = tuple(
        _finite_scalar(value, "lambda_value") for value in lambda_values
    )
    allowed = (0.0, 0.001, 0.01, 0.1, 1.0)
    if (
        not lambdas
        or len(set(lambdas)) != len(lambdas)
        or any(value not in allowed for value in lambdas)
    ):
        raise ValueError("lambda_values must be unique values from the fixed grid")
    use_legacy_compatibility = optimizer is None
    optimize = _default_reference_optimizer if optimizer is None else optimizer
    periods = _deterministic_inclusive_grid(2.5, 5.0, 0.05)
    fold_assignment = assign_reference_folds(
        [row.distance_km for row in rows],
        [row.azimuth_deg for row in rows],
    )
    training_by_fold = tuple(
        tuple(
            row
            for index, row in enumerate(rows)
            if fold_assignment.fold_ids[index] != fold
        )
        for fold in range(5)
    )
    holdout_by_fold = tuple(
        tuple(
            row
            for index, row in enumerate(rows)
            if fold_assignment.fold_ids[index] == fold
        )
        for fold in range(5)
    )
    prepared_by_fold = (
        tuple(
            prepare_reference_objective(training, periods_s=periods)
            for training in training_by_fold
        )
        if use_legacy_compatibility
        else ()
    )
    fixed_velocities = (1.7, 2.25, 2.8, 3.35, 3.9)
    bounds = tuple((1.0 / 4.0, 1.0 / 1.6) for _ in periods)
    configs = []
    total_calls = 0
    for lambda_s_value in lambdas:
        for lambda_g_value in lambdas:
            fold_losses = []
            for fold in range(5):
                training = training_by_fold[fold]
                holdout = holdout_by_fold[fold]
                candidates = []
                for velocity in fixed_velocities:
                    x0 = np.full(periods.size, 1.0 / velocity, dtype=float)
                    if use_legacy_compatibility:
                        prepared = prepared_by_fold[fold]
                        objective = lambda candidate, prepared=prepared: (
                            prepared_reference_legacy_compatible_value(
                                candidate,
                                prepared,
                                lambda_s=lambda_s_value,
                                lambda_g=lambda_g_value,
                            )
                        )
                        optimizer_objective = (
                            lambda candidate, prepared=prepared: (
                                prepared_reference_legacy_compatible_value_and_gradient(
                                    candidate,
                                    prepared,
                                    lambda_s=lambda_s_value,
                                    lambda_g=lambda_g_value,
                                )
                            )
                        )
                    else:
                        objective = lambda candidate, training=training: (
                            reference_fit_objective(
                                candidate,
                                training,
                                lambda_s=lambda_s_value,
                                lambda_g=lambda_g_value,
                                periods_s=periods,
                            )
                        )
                        optimizer_objective = objective
                    result = optimize(
                        optimizer_objective,
                        x0,
                        bounds,
                        200,
                    )
                    total_calls += 1
                    candidate_x = np.asarray(result.x, dtype=float)
                    candidate_loss = float(objective(candidate_x))
                    candidates.append((candidate_loss, candidate_x))
                _, winner = min(candidates, key=lambda item: item[0])
                fold_losses.append(
                    reference_phase_holdout_loss(
                        winner,
                        holdout,
                        periods_s=periods,
                    )
                )
            configs.append(
                ReferenceCvConfig(
                    lambda_s=lambda_s_value,
                    lambda_g=lambda_g_value,
                    fold_holdout_losses=np.asarray(fold_losses, dtype=float),
                    mean_holdout_loss=float(np.mean(fold_losses)),
                    optimizer_calls=25,
                )
            )
    selected = select_reference_cv_config(configs)
    digest_source = "|".join(
        (
            fold_assignment.assignment_hash,
            *(
                f"{item.lambda_s:g},{item.lambda_g:g},"
                f"{item.mean_holdout_loss:.17g},{item.optimizer_calls},"
                + ",".join(
                    f"{loss:.17g}"
                    for loss in item.fold_holdout_losses
                )
                for item in configs
            ),
        )
    )
    return ReferenceCvResult(
        fold_assignment=fold_assignment,
        configs=tuple(configs),
        selected=selected,
        optimizer_calls=total_calls,
        result_hash=hashlib.sha256(digest_source.encode("ascii")).hexdigest(),
    )


@dataclass(frozen=True)
class AliasCandidate:
    objective: float
    target_velocities_km_s: np.ndarray

    def __post_init__(self) -> None:
        objective = _finite_scalar(self.objective, "objective")
        velocity = np.array(
            _strict_finite_real(
                self.target_velocities_km_s,
                "target_velocities_km_s",
            ),
            dtype=float,
            copy=True,
        )
        if objective < 0 or velocity.shape != (4,) or np.any(velocity <= 0):
            raise ValueError("alias candidate is invalid")
        velocity.setflags(write=False)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "target_velocities_km_s", velocity)


def reference_alias_status(representatives) -> str:
    """Apply the five-minimum sufficiency and near-equal target alias rules."""

    rows = tuple(representatives)
    if len(rows) < 5:
        return "reference_search_insufficient_minima"
    ordered = sorted(
        rows,
        key=lambda row: _finite_scalar(
            getattr(row, "holdout_loss", row.objective),
            "holdout_loss",
        ),
    )
    best, second = ordered[:2]
    best_objective = float(getattr(best, "holdout_loss", best.objective))
    second_objective = float(
        getattr(second, "holdout_loss", second.objective)
    )
    near_equal = second_objective == best_objective or (
        best_objective > 0.0
        and (second_objective - best_objective) / best_objective < 0.01
    )
    target_difference = float(
        np.max(
            np.abs(
                np.asarray(best.target_velocities_km_s, dtype=float)
                - np.asarray(second.target_velocities_km_s, dtype=float)
            )
        )
    )
    if near_equal and target_difference > 0.10:
        return "reference_alias_unresolved"
    return "accepted"


@dataclass(frozen=True)
class LocalReferenceSolution:
    start_index: int
    start_hash: str
    converged: bool
    objective: float
    fold_holdout_losses: np.ndarray
    holdout_loss: float
    phase_slowness_s_km: np.ndarray
    velocities_km_s: np.ndarray
    target_velocities_km_s: np.ndarray
    optimizer_message: str
    basin_id: int = -1

    def __post_init__(self) -> None:
        index = _integer_scalar(self.start_index, "start_index")
        if index < 0:
            raise ValueError("start_index must be non-negative")
        if (
            not isinstance(self.start_hash, str)
            or len(self.start_hash) != 64
        ):
            raise ValueError("start_hash must be a SHA-256 digest")
        if not isinstance(self.converged, (bool, np.bool_)):
            raise ValueError("converged must be boolean")
        objective = _finite_scalar(self.objective, "objective")
        fold_losses = np.array(
            _strict_finite_real(
                self.fold_holdout_losses,
                "fold_holdout_losses",
            ),
            dtype=float,
            copy=True,
        )
        holdout_loss = _finite_scalar(self.holdout_loss, "holdout_loss")
        if (
            objective < 0
            or fold_losses.shape != (5,)
            or np.any(fold_losses < 0)
            or holdout_loss < 0
            or not math.isclose(
                holdout_loss,
                float(np.mean(fold_losses)),
                rel_tol=0.0,
                abs_tol=64.0 * np.finfo(float).eps * max(1.0, holdout_loss),
            )
        ):
            raise ValueError("solution losses must be non-negative")
        slowness = np.array(
            _strict_finite_real(
                self.phase_slowness_s_km,
                "phase_slowness_s_km",
            ),
            dtype=float,
            copy=True,
        )
        velocity = np.array(
            _strict_finite_real(self.velocities_km_s, "velocities_km_s"),
            dtype=float,
            copy=True,
        )
        targets = np.array(
            _strict_finite_real(
                self.target_velocities_km_s,
                "target_velocities_km_s",
            ),
            dtype=float,
            copy=True,
        )
        if (
            slowness.shape != (51,)
            or velocity.shape != (51,)
            or targets.shape != (4,)
            or np.any(velocity < 1.6)
            or np.any(velocity > 4.0)
            or not np.allclose(
                slowness,
                1.0 / velocity,
                rtol=0.0,
                atol=64.0 * np.finfo(float).eps,
            )
        ):
            raise ValueError("local solution arrays are inconsistent")
        if (
            not isinstance(self.optimizer_message, str)
            or not self.optimizer_message
        ):
            raise ValueError("optimizer_message must be non-empty")
        basin_id = _integer_scalar(self.basin_id, "basin_id")
        if basin_id < -1:
            raise ValueError("basin_id must be -1 or non-negative")
        for array in (fold_losses, slowness, velocity, targets):
            array.setflags(write=False)
        object.__setattr__(self, "start_index", index)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "fold_holdout_losses", fold_losses)
        object.__setattr__(self, "holdout_loss", holdout_loss)
        object.__setattr__(self, "phase_slowness_s_km", slowness)
        object.__setattr__(self, "velocities_km_s", velocity)
        object.__setattr__(self, "target_velocities_km_s", targets)
        object.__setattr__(self, "basin_id", basin_id)


@dataclass(frozen=True)
class ReferenceFitResult:
    status: str
    periods_s: np.ndarray
    phase_slowness_s_km: np.ndarray
    phase_velocities_km_s: np.ndarray
    group_slowness_s_km: np.ndarray
    target_periods_s: np.ndarray
    target_velocities_km_s: np.ndarray
    lambda_s: float
    lambda_g: float
    cv_result: ReferenceCvResult
    starts: Tuple[ReferenceStart, ...]
    local_solutions: Tuple[LocalReferenceSolution, ...]
    basin_ids: np.ndarray
    representative_indices: Tuple[int, ...]
    cv_optimizer_calls: int
    final_optimizer_calls: int
    optimizer_calls: int
    result_hash: str

    def __post_init__(self) -> None:
        allowed_statuses = {
            "accepted",
            "reference_alias_unresolved",
            "reference_search_insufficient_minima",
        }
        if self.status not in allowed_statuses:
            raise ValueError("invalid reference fit status")
        arrays = {}
        for name, shape in (
            ("periods_s", (51,)),
            ("phase_slowness_s_km", (51,)),
            ("phase_velocities_km_s", (51,)),
            ("group_slowness_s_km", (51,)),
            ("target_periods_s", (4,)),
            ("target_velocities_km_s", (4,)),
        ):
            array = np.array(
                _strict_finite_real(getattr(self, name), name),
                dtype=float,
                copy=True,
            )
            if array.shape != shape:
                raise ValueError(f"{name} has the wrong shape")
            array.setflags(write=False)
            arrays[name] = array
            object.__setattr__(self, name, array)
        expected_periods = _deterministic_inclusive_grid(2.5, 5.0, 0.05)
        if not np.allclose(
            arrays["periods_s"],
            expected_periods,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("reference fit must use the full 2.5--5.0 grid")
        if not np.array_equal(
            arrays["target_periods_s"],
            np.array([3.0, 3.5, 4.0, 5.0]),
        ):
            raise ValueError("target periods are invalid")
        if (
            np.any(arrays["phase_velocities_km_s"] < 1.6)
            or np.any(arrays["phase_velocities_km_s"] > 4.0)
            or not np.allclose(
                arrays["phase_slowness_s_km"],
                1.0 / arrays["phase_velocities_km_s"],
                rtol=0.0,
                atol=64.0 * np.finfo(float).eps,
            )
        ):
            raise ValueError("reference phase velocity/slowness is inconsistent")
        expected_group_slowness = phase_slowness_to_group_slowness(
            arrays["periods_s"],
            arrays["phase_slowness_s_km"],
        )
        if not np.allclose(
            arrays["group_slowness_s_km"],
            expected_group_slowness,
            rtol=0.0,
            atol=128.0 * np.finfo(float).eps,
        ):
            raise ValueError(
                "group_slowness must be derived from phase slowness"
            )
        expected_targets = np.interp(
            arrays["target_periods_s"],
            arrays["periods_s"],
            arrays["phase_velocities_km_s"],
        )
        if not np.allclose(
            arrays["target_velocities_km_s"],
            expected_targets,
            rtol=0.0,
            atol=128.0 * np.finfo(float).eps,
        ):
            raise ValueError(
                "target_velocities must derive from the reference curve"
            )
        if not isinstance(self.cv_result, ReferenceCvResult):
            raise ValueError("cv_result is invalid")
        if (
            self.lambda_s != self.cv_result.selected.lambda_s
            or self.lambda_g != self.cv_result.selected.lambda_g
        ):
            raise ValueError("lambda values must match selected CV config")
        starts = tuple(self.starts)
        solutions = tuple(self.local_solutions)
        if (
            len(starts) not in (71, 128)
            or len(solutions) != len(starts)
            or any(
                solution.start_index != start.start_index
                or solution.start_hash != start.velocity_hash
                for start, solution in zip(starts, solutions)
            )
        ):
            raise ValueError("starts and local_solutions are inconsistent")
        basin_ids = np.array(self.basin_ids, dtype=int, copy=True)
        if basin_ids.shape != (len(solutions),) or any(
            solution.basin_id != int(basin_ids[index])
            for index, solution in enumerate(solutions)
        ):
            raise ValueError("basin_ids are inconsistent")
        representatives = tuple(
            _integer_scalar(value, "representative_index")
            for value in self.representative_indices
        )
        if any(index < 0 or index >= len(solutions) for index in representatives):
            raise ValueError("representative index is invalid")
        expected_basin_ids, expected_representatives = (
            _cluster_converged_reference_solutions(solutions)
        )
        if not np.array_equal(basin_ids, expected_basin_ids) or (
            representatives != expected_representatives
        ):
            raise ValueError("basin representatives are inconsistent")
        if self.status == "accepted" and len(representatives) < 5:
            raise ValueError("accepted result must contain at least five basins")
        representative_rows = tuple(
            solutions[index] for index in representatives
        )
        expected_status = reference_alias_status(representative_rows)
        if self.status != expected_status:
            raise ValueError("status is inconsistent with alias solutions")
        best_index = (
            min(
                representatives,
                key=lambda index: (solutions[index].objective, index),
            )
            if representatives
            else min(
                range(len(solutions)),
                key=lambda index: (solutions[index].objective, index),
            )
        )
        best = solutions[best_index]
        if (
            not np.array_equal(
                arrays["phase_slowness_s_km"],
                best.phase_slowness_s_km,
            )
            or not np.array_equal(
                arrays["phase_velocities_km_s"],
                best.velocities_km_s,
            )
        ):
            raise ValueError(
                "returned reference curve must match the best local solution"
            )
        cv_calls = _integer_scalar(
            self.cv_optimizer_calls,
            "cv_optimizer_calls",
        )
        final_calls = _integer_scalar(
            self.final_optimizer_calls,
            "final_optimizer_calls",
        )
        total_calls = _integer_scalar(self.optimizer_calls, "optimizer_calls")
        if (
            cv_calls != self.cv_result.optimizer_calls
            or final_calls != len(starts)
            or total_calls != cv_calls + final_calls
            or total_calls > 753
        ):
            raise ValueError("optimizer call accounting is inconsistent")
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.status}|{self.cv_result.result_hash}|"
                f"{self.lambda_s:.17g}|{self.lambda_g:.17g}"
            ).encode("ascii")
        )
        for name in sorted(arrays):
            digest.update(name.encode("ascii"))
            digest.update(np.asarray(arrays[name], dtype="<f8").tobytes())
        for start in starts:
            digest.update(start.velocity_hash.encode("ascii"))
        digest.update(b"basin_ids")
        digest.update(np.asarray(basin_ids, dtype="<i8").tobytes())
        digest.update(b"representative_indices")
        digest.update(np.asarray(representatives, dtype="<i8").tobytes())
        digest.update(f"best_index|{best_index}".encode("ascii"))
        for solution in solutions:
            digest.update(
                (
                    f"{solution.objective:.17g}|"
                    f"{solution.holdout_loss:.17g}|{solution.basin_id}|"
                    f"{int(solution.converged)}"
                ).encode("ascii")
            )
            digest.update(
                np.asarray(
                    solution.phase_slowness_s_km,
                    dtype="<f8",
                ).tobytes()
            )
            digest.update(
                np.asarray(
                    solution.fold_holdout_losses,
                    dtype="<f8",
                ).tobytes()
            )
        expected_hash = digest.hexdigest()
        if self.result_hash != expected_hash:
            raise ValueError("result_hash is inconsistent")
        basin_ids.setflags(write=False)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "local_solutions", solutions)
        object.__setattr__(self, "basin_ids", basin_ids)
        object.__setattr__(self, "representative_indices", representatives)
        object.__setattr__(self, "cv_optimizer_calls", cv_calls)
        object.__setattr__(self, "final_optimizer_calls", final_calls)
        object.__setattr__(self, "optimizer_calls", total_calls)


def _cluster_converged_reference_solutions(solutions):
    converged_indices = tuple(
        index for index, row in enumerate(solutions) if row.converged
    )
    if not converged_indices:
        return np.full(len(solutions), -1, dtype=int), ()
    clustering = cluster_reference_solutions(
        solutions[index] for index in converged_indices
    )
    basin_ids = np.full(len(solutions), -1, dtype=int)
    for local_index, solution_index in enumerate(converged_indices):
        basin_ids[solution_index] = clustering.basin_ids[local_index]
    representatives = tuple(
        converged_indices[index]
        for index in clustering.representative_indices
    )
    return basin_ids, representatives


def fit_reference_dispersion(
    observations: Iterable[ReferenceObservation],
    *,
    optimizer=None,
) -> ReferenceFitResult:
    """Fit, cross-validate, and deterministically search phase-cycle aliases."""

    rows = tuple(observations)
    use_legacy_compatibility = optimizer is None
    optimize = _default_reference_optimizer if optimizer is None else optimizer
    cv_result = cross_validate_reference_fit(rows, optimizer=optimizer)
    periods = _deterministic_inclusive_grid(2.5, 5.0, 0.05)
    target_periods = np.array([3.0, 3.5, 4.0, 5.0])
    bounds = tuple((1.0 / 4.0, 1.0 / 1.6) for _ in periods)
    starts = list(generate_reference_starts(periods, max_starts=71))
    solutions = []
    prepared = (
        prepare_reference_objective(rows, periods_s=periods)
        if use_legacy_compatibility
        else None
    )

    def run_start(start):
        x0 = 1.0 / start.velocities_km_s
        if use_legacy_compatibility:
            objective = lambda candidate: prepared_reference_legacy_compatible_value(
                candidate,
                prepared,
                lambda_s=cv_result.selected.lambda_s,
                lambda_g=cv_result.selected.lambda_g,
            )
            optimizer_objective = (
                lambda candidate: prepared_reference_legacy_compatible_value_and_gradient(
                    candidate,
                    prepared,
                    lambda_s=cv_result.selected.lambda_s,
                    lambda_g=cv_result.selected.lambda_g,
                )
            )
        else:
            objective = lambda candidate: reference_fit_objective(
                candidate,
                rows,
                lambda_s=cv_result.selected.lambda_s,
                lambda_g=cv_result.selected.lambda_g,
                periods_s=periods,
            )
            optimizer_objective = objective
        result = optimize(optimizer_objective, x0, bounds, 500)
        candidate = np.clip(
            np.asarray(result.x, dtype=float),
            1.0 / 4.0,
            1.0 / 1.6,
        )
        candidate_velocity = 1.0 / candidate
        target_velocity = np.interp(
            target_periods,
            periods,
            candidate_velocity,
        )
        fold_losses, mean_fold_loss = reference_final_fold_holdout_losses(
            candidate,
            rows,
            cv_result.fold_assignment,
            periods_s=periods,
        )
        return LocalReferenceSolution(
            start_index=start.start_index,
            start_hash=start.velocity_hash,
            converged=bool(result.success),
            objective=float(objective(candidate)),
            fold_holdout_losses=fold_losses,
            holdout_loss=mean_fold_loss,
            phase_slowness_s_km=candidate,
            velocities_km_s=candidate_velocity,
            target_velocities_km_s=target_velocity,
            optimizer_message=str(result.message) or "no optimizer message",
        )

    solutions.extend(run_start(start) for start in starts)
    basin_ids, representatives = _cluster_converged_reference_solutions(
        solutions
    )
    if len(representatives) < 5:
        all_starts = generate_reference_starts(periods, max_starts=128)
        additional = all_starts[len(starts) :]
        starts.extend(additional)
        solutions.extend(run_start(start) for start in additional)
        basin_ids, representatives = _cluster_converged_reference_solutions(
            solutions
        )
    solutions = [
        replace(solution, basin_id=int(basin_ids[index]))
        for index, solution in enumerate(solutions)
    ]
    representative_rows = tuple(solutions[index] for index in representatives)
    status = reference_alias_status(representative_rows)
    if representative_rows:
        best_index = min(
            representatives,
            key=lambda index: (solutions[index].objective, index),
        )
    else:
        best_index = min(
            range(len(solutions)),
            key=lambda index: (solutions[index].objective, index),
        )
    best = solutions[best_index]
    group_slowness = phase_slowness_to_group_slowness(
        periods,
        best.phase_slowness_s_km,
    )
    digest = hashlib.sha256()
    digest.update(
        (
            f"{status}|{cv_result.result_hash}|"
            f"{cv_result.selected.lambda_s:.17g}|"
            f"{cv_result.selected.lambda_g:.17g}"
        ).encode("ascii")
    )
    result_arrays = {
        "periods_s": periods,
        "phase_slowness_s_km": best.phase_slowness_s_km,
        "phase_velocities_km_s": best.velocities_km_s,
        "group_slowness_s_km": group_slowness,
        "target_periods_s": target_periods,
        "target_velocities_km_s": best.target_velocities_km_s,
    }
    for name in sorted(result_arrays):
        digest.update(name.encode("ascii"))
        digest.update(
            np.asarray(result_arrays[name], dtype="<f8").tobytes()
        )
    for start in starts:
        digest.update(start.velocity_hash.encode("ascii"))
    digest.update(b"basin_ids")
    digest.update(np.asarray(basin_ids, dtype="<i8").tobytes())
    digest.update(b"representative_indices")
    digest.update(np.asarray(representatives, dtype="<i8").tobytes())
    digest.update(f"best_index|{best_index}".encode("ascii"))
    for solution in solutions:
        digest.update(
            (
                f"{solution.objective:.17g}|"
                f"{solution.holdout_loss:.17g}|{solution.basin_id}|"
                f"{int(solution.converged)}"
            ).encode("ascii")
        )
        digest.update(
            np.asarray(
                solution.phase_slowness_s_km,
                dtype="<f8",
            ).tobytes()
        )
        digest.update(
            np.asarray(
                solution.fold_holdout_losses,
                dtype="<f8",
            ).tobytes()
        )
    return ReferenceFitResult(
        status=status,
        periods_s=periods,
        phase_slowness_s_km=best.phase_slowness_s_km,
        phase_velocities_km_s=best.velocities_km_s,
        group_slowness_s_km=group_slowness,
        target_periods_s=target_periods,
        target_velocities_km_s=best.target_velocities_km_s,
        lambda_s=cv_result.selected.lambda_s,
        lambda_g=cv_result.selected.lambda_g,
        cv_result=cv_result,
        starts=tuple(starts),
        local_solutions=tuple(solutions),
        basin_ids=basin_ids,
        representative_indices=representatives,
        cv_optimizer_calls=cv_result.optimizer_calls,
        final_optimizer_calls=len(starts),
        optimizer_calls=cv_result.optimizer_calls + len(starts),
        result_hash=digest.hexdigest(),
    )


def resolve_reference_cycles(
    *,
    raw_times_s,
    distance_km,
    observation_periods_s,
    reference_periods_s,
    reference_slowness_s_km,
    convention: PhaseConvention = PhaseConvention.BENSEN_VELOCITY_CCF,
) -> Tuple[CycleResolution, ...]:
    """Resolve cycles at each supplied instantaneous or exact target period."""

    convention = _require_phase_convention(convention)
    raw_times = _strict_finite_real(raw_times_s, "raw_times_s")
    distances = _strict_finite_real(distance_km, "distance_km")
    observation_periods = _strict_finite_real(
        observation_periods_s,
        "observation_periods_s",
    )
    reference_periods = _reference_grid(reference_periods_s)
    reference_slowness = _strict_finite_real(
        reference_slowness_s_km,
        "reference_slowness_s_km",
    )
    endpoint_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, abs(float(reference_periods[-1])))
    )
    if (
        raw_times.ndim != 1
        or raw_times.shape != distances.shape
        or raw_times.shape != observation_periods.shape
        or raw_times.size == 0
        or np.any(distances <= 0)
        or np.any(observation_periods <= 0)
        or reference_slowness.shape != reference_periods.shape
        or np.any(reference_slowness <= 0)
        or np.any(
            observation_periods < reference_periods[0] - endpoint_tolerance
        )
        or np.any(
            observation_periods > reference_periods[-1] + endpoint_tolerance
        )
    ):
        raise ValueError("cycle-resolution arrays are inconsistent")
    predicted_slowness = np.interp(
        np.clip(
            observation_periods,
            reference_periods[0],
            reference_periods[-1],
        ),
        reference_periods,
        reference_slowness,
    )
    return tuple(
        resolve_cycle_count(
            raw_time_s=float(raw_time),
            reference_time_s=float(distance * slowness),
            period_s=float(period),
            convention=convention,
        )
        for raw_time, distance, period, slowness in zip(
            raw_times,
            distances,
            observation_periods,
            predicted_slowness,
        )
    )


def prepare_phase_waveform(
    time_s: np.ndarray,
    symmetric_ccf: np.ndarray,
    convention: PhaseConvention,
) -> np.ndarray:
    convention = _require_phase_convention(convention)
    time = _real_numeric_array(time_s, "time_s")
    waveform = _real_numeric_array(symmetric_ccf, "symmetric_ccf")
    if time.ndim != 1 or waveform.ndim != 1:
        raise ValueError("time_s and symmetric_ccf must be one-dimensional")
    if time.size < 2 or time.shape != waveform.shape:
        raise ValueError(
            "time_s and symmetric_ccf must have the same shape with at least two samples"
        )
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(waveform)):
        raise ValueError("time_s and symmetric_ccf must contain only finite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if convention.definition.apply_negative_time_derivative:
        prepared = -np.gradient(waveform, time)
    else:
        prepared = np.array(waveform, dtype=float, copy=True)
    prepared = np.asarray(prepared, dtype=float)
    prepared.setflags(write=False)
    return prepared


def validate_phase_convention(
    *,
    convention: PhaseConvention,
    true_phase_velocities_km_s: np.ndarray,
    recovered_phase_velocities_km_s: np.ndarray,
    noise_free_mask: np.ndarray,
    valid_mask=None,  # type: Optional[np.ndarray]
    noise_free_max_tolerance: float = 0.005,
    noisy_median_tolerance: float = 0.02,
) -> "PhaseConventionValidationSummary":
    """Summarize an independently executed convention-validation matrix."""

    convention = _require_phase_convention(convention)
    true_velocity = _real_numeric_array(
        true_phase_velocities_km_s,
        "true_phase_velocities_km_s",
    )
    recovered_velocity = _real_numeric_array(
        recovered_phase_velocities_km_s,
        "recovered_phase_velocities_km_s",
    )
    noise_free = np.asarray(noise_free_mask)
    if (
        true_velocity.ndim != 1
        or recovered_velocity.ndim != 1
        or noise_free.ndim != 1
    ):
        raise ValueError("validation inputs must be one-dimensional")
    if true_velocity.size == 0 or not (
        true_velocity.shape == recovered_velocity.shape == noise_free.shape
    ):
        raise ValueError(
            "validation inputs must be non-empty with the same shape"
        )
    if np.any(~np.isfinite(true_velocity)) or np.any(true_velocity <= 0):
        raise ValueError(
            "true_phase_velocities_km_s must contain positive finite values"
        )
    if noise_free.dtype != np.dtype(bool):
        raise ValueError("noise_free_mask must contain boolean values")
    if valid_mask is None:
        requested_valid = np.ones(true_velocity.size, dtype=bool)
    else:
        requested_valid = np.asarray(valid_mask)
        if requested_valid.ndim != 1 or requested_valid.shape != true_velocity.shape:
            raise ValueError("valid_mask must have the same one-dimensional shape")
        if requested_valid.dtype != np.dtype(bool):
            raise ValueError("valid_mask must contain boolean values")
    noise_free_limit = _finite_scalar(
        noise_free_max_tolerance,
        "noise_free_max_tolerance",
        positive=True,
    )
    noisy_limit = _finite_scalar(
        noisy_median_tolerance,
        "noisy_median_tolerance",
        positive=True,
    )

    valid = (
        requested_valid
        & np.isfinite(recovered_velocity)
        & (recovered_velocity > 0)
    )
    relative_error = np.full(true_velocity.size, np.nan, dtype=float)
    relative_error[valid] = np.abs(
        recovered_velocity[valid] / true_velocity[valid] - 1.0
    )
    noise_free_valid = valid & noise_free
    noisy_valid = valid & ~noise_free
    noise_free_max = (
        float(np.max(relative_error[noise_free_valid]))
        if np.any(noise_free_valid)
        else float("nan")
    )
    noisy_median = (
        float(np.median(relative_error[noisy_valid]))
        if np.any(noisy_valid)
        else float("nan")
    )
    total_count = int(true_velocity.size)
    valid_count = int(np.count_nonzero(valid))
    failure_count = total_count - valid_count
    if failure_count:
        status = "invalid_measurements"
    elif not np.any(noise_free_valid) or not np.any(noisy_valid):
        status = "insufficient_validation_coverage"
    elif noise_free_max > noise_free_limit:
        status = "noise_free_error_exceeds_limit"
    elif noisy_median > noisy_limit:
        status = "noisy_median_error_exceeds_limit"
    else:
        status = "thresholds_passed"
    return PhaseConventionValidationSummary(
        convention=convention,
        total_count=total_count,
        valid_count=valid_count,
        failure_count=failure_count,
        noise_free_max_relative_error=noise_free_max,
        noisy_median_relative_error=noisy_median,
        status=status,
    )


def _deterministic_inclusive_grid(
    start: float,
    stop: float,
    step: float,
) -> np.ndarray:
    if (
        not np.isfinite(start)
        or not np.isfinite(stop)
        or not np.isfinite(step)
        or start <= 0
        or stop < start
        or step <= 0
    ):
        raise ValueError("grid bounds and step must define a positive increasing grid")
    interval_count = int(round((float(stop) - float(start)) / float(step)))
    if not math.isclose(
        float(start) + interval_count * float(step),
        float(stop),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("grid range must contain an integer number of steps")
    grid = np.round(
        np.linspace(float(start), float(stop), interval_count + 1, dtype=float),
        decimals=12,
    )
    grid.setflags(write=False)
    return grid


@dataclass(frozen=True)
class FtanConfig:
    period_min_s: float = 2.5
    period_max_s: float = 5.0
    period_step_s: float = 0.05
    group_velocity_min_km_s: float = 1.6
    group_velocity_max_km_s: float = 5.0
    group_velocity_step_km_s: float = 0.01
    target_periods_s: Tuple[float, ...] = (3.0, 3.5, 4.0, 5.0)
    alpha_candidates: Tuple[float, ...] = (5.0, 8.0, 12.0, 16.0, 20.0, 25.0)
    beta1_candidates: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    beta2_candidates: Tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)

    @property
    def periods_s(self) -> np.ndarray:
        return _deterministic_inclusive_grid(
            self.period_min_s,
            self.period_max_s,
            self.period_step_s,
        )

    @property
    def group_velocities_km_s(self) -> np.ndarray:
        return _deterministic_inclusive_grid(
            self.group_velocity_min_km_s,
            self.group_velocity_max_km_s,
            self.group_velocity_step_km_s,
        )


@dataclass(frozen=True)
class GaussianFilterBankResult:
    filtered_waveforms: np.ndarray
    analytic_signals: np.ndarray
    envelope: np.ndarray


@dataclass(frozen=True)
class PhaseMatchedFilterResult:
    compressed_waveform: np.ndarray
    compressed_envelope: np.ndarray
    cleaning_window: np.ndarray
    cleaned_compressed_waveform: np.ndarray
    cleaned_waveform: np.ndarray
    cut_center_index: int
    cut_center_time_s: float
    cut_half_width_s: float
    cut_taper_alpha: float
    first_pass_alpha: float
    second_pass_alpha: float
    reference_group_time_s: float

    def __post_init__(self) -> None:
        arrays = {}
        for name in (
            "compressed_waveform",
            "compressed_envelope",
            "cleaning_window",
            "cleaned_compressed_waveform",
            "cleaned_waveform",
        ):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if array.ndim != 1 or array.size == 0 or np.any(~np.isfinite(array)):
                raise ValueError(
                    "phase-matched filter arrays must be finite and "
                    "one-dimensional"
                )
            array.setflags(write=False)
            arrays[name] = array
        shape = arrays["compressed_waveform"].shape
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("phase-matched filter arrays must share one shape")
        center = _integer_scalar(self.cut_center_index, "cut_center_index")
        if center < 0 or center >= shape[0]:
            raise ValueError("cut_center_index lies outside the waveform")
        scalar_names = (
            "cut_center_time_s",
            "cut_half_width_s",
            "cut_taper_alpha",
            "first_pass_alpha",
            "second_pass_alpha",
            "reference_group_time_s",
        )
        scalars = {
            name: _finite_scalar(
                getattr(self, name),
                name,
                positive=name
                in (
                    "cut_half_width_s",
                    "first_pass_alpha",
                    "second_pass_alpha",
                    "reference_group_time_s",
                ),
            )
            for name in scalar_names
        }
        if (
            scalars["cut_center_time_s"] < 0
            or not 0 <= scalars["cut_taper_alpha"] <= 1
            or np.any(arrays["compressed_envelope"] < 0)
            or np.any(arrays["cleaning_window"] < 0)
            or np.any(arrays["cleaning_window"] > 1)
            or arrays["cleaning_window"][center] != 1.0
        ):
            raise ValueError("phase-matched filter metadata is inconsistent")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "cut_center_index", center)
        for name, value in scalars.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PhaseMatchedSecondPassResult:
    cleaning: PhaseMatchedFilterResult
    second_pass_filter_bank: GaussianFilterBankResult

    def __post_init__(self) -> None:
        if not isinstance(self.cleaning, PhaseMatchedFilterResult):
            raise ValueError("cleaning must be PhaseMatchedFilterResult")
        if not isinstance(
            self.second_pass_filter_bank,
            GaussianFilterBankResult,
        ):
            raise ValueError(
                "second_pass_filter_bank must be GaussianFilterBankResult"
            )


@dataclass(frozen=True)
class RidgeQuality:
    accepted: bool
    reason: str
    coverage: float
    max_gap: int
    jump_fraction: float
    boundary_fraction: float
    normalized_energy_integral: float


@dataclass(frozen=True)
class RidgeResult:
    row_indices: np.ndarray
    group_velocities_km_s: np.ndarray
    valid: np.ndarray
    wang_group_limit_pass_count: int
    quality: RidgeQuality

    def __post_init__(self) -> None:
        rows = np.array(self.row_indices, dtype=int, copy=True)
        velocities = np.array(self.group_velocities_km_s, dtype=float, copy=True)
        valid = np.array(self.valid, dtype=bool, copy=True)
        if rows.ndim != 1 or velocities.ndim != 1 or valid.ndim != 1:
            raise ValueError("ridge result arrays must be one-dimensional")
        if not (rows.shape == velocities.shape == valid.shape):
            raise ValueError("ridge result arrays must have identical shapes")
        rows.setflags(write=False)
        velocities.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "group_velocities_km_s", velocities)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class GroupArrivalRefinement:
    peak_index: int
    grid_time_s: float
    group_time_s: float
    vertex_offset_samples: float
    refinement_used: bool
    status: str


@dataclass(frozen=True)
class InstantaneousFrequencyResult:
    fitted_phase_intercept_rad: float
    fitted_phase_slope_rad_s: float
    omega_inst_rad_s: float
    instantaneous_period_s: float
    period_ratio: float
    window_sample_count: int
    status: str


@dataclass(frozen=True)
class WangSnrResult:
    signal_peak: float
    leading_noise_rms: float
    trailing_noise_rms: float
    leading_snr: float
    trailing_snr: float
    signal_sample_count: int
    leading_noise_sample_count: int
    trailing_noise_sample_count: int
    signal_window_start_s: float
    signal_window_end_s: float
    leading_noise_start_s: float
    leading_noise_end_s: float
    trailing_noise_start_s: float
    trailing_noise_end_s: float
    leading_status: str
    trailing_status: str
    status: str


@dataclass(frozen=True)
class WangLeftQcResult:
    accepted: bool
    status: str
    group_velocity_min_km_s: float
    group_velocity_max_km_s: float
    snr_threshold: float


@dataclass(frozen=True)
class WangTargetPeriodResult:
    target_period_s: float
    anchored_raw_phase_time_s: float
    group_time_s: float
    group_velocity_km_s: float
    signal_peak: float
    leading_noise_rms: float
    trailing_noise_rms: float
    leading_snr: float
    trailing_snr: float
    ridge_normalized_log_energy: float
    ridge_normalized_envelope_amplitude: float
    ridge_adjacent_jump_km_s: float
    support_periods_s: np.ndarray
    support_count: int
    interpolation_method: str
    accepted: bool
    status: str
    rejected_continuous_nominal_periods_s: Optional[np.ndarray] = None
    rejected_continuous_instantaneous_periods_s: Optional[np.ndarray] = None
    continuous_rejection_statuses: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        target = _finite_scalar(
            self.target_period_s,
            "target_period_s",
            positive=True,
        )
        object.__setattr__(self, "target_period_s", target)
        for name, upper_bound in (
            ("ridge_normalized_log_energy", 1.0),
            ("ridge_normalized_envelope_amplitude", 1.0),
            ("ridge_adjacent_jump_km_s", None),
        ):
            value = float(getattr(self, name))
            if np.isinf(value) or (np.isfinite(value) and value < 0):
                raise ValueError(
                    f"{name} must be non-negative and finite or NaN"
                )
            if upper_bound is not None and np.isfinite(value):
                tolerance = (
                    64.0 * np.finfo(float).eps * max(1.0, abs(value))
                )
                if value > upper_bound + tolerance:
                    raise ValueError(
                        f"{name} must not exceed {upper_bound:g}"
                    )
            if self.accepted and not np.isfinite(value):
                raise ValueError(
                    f"accepted target must have a finite {name}"
                )
            object.__setattr__(self, name, value)
        support = np.array(
            _real_numeric_array(self.support_periods_s, "support_periods_s"),
            dtype=float,
            copy=True,
        )
        if support.ndim != 1 or np.any(~np.isfinite(support)):
            raise ValueError(
                "support_periods_s must be a finite one-dimensional array"
            )
        if (
            np.any(support <= 0)
            or (support.size > 1 and np.any(np.diff(support) <= 0))
        ):
            raise ValueError(
                "support_periods_s must contain positive increasing values"
            )
        if (
            isinstance(self.support_count, (bool, np.bool_))
            or not isinstance(self.support_count, (int, np.integer))
            or int(self.support_count) != support.size
        ):
            raise ValueError(
                "support_count must equal the number of support periods"
            )
        support_count = int(self.support_count)
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise ValueError("accepted must be boolean")
        accepted = bool(self.accepted)
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a non-empty string")
        if accepted != (self.status == "accepted"):
            raise ValueError("accepted must be true exactly for accepted status")
        method = self.interpolation_method
        if method not in {"none", "linear", "pchip"}:
            raise ValueError("interpolation_method is invalid")
        if method == "linear" and support_count != 2:
            raise ValueError("linear interpolation requires two supports")
        if method == "pchip" and support_count < 3:
            raise ValueError("pchip interpolation requires at least three supports")
        if self.status == "target_period_not_bracketed":
            if method != "none":
                raise ValueError(
                    "unbracketed target must use no interpolation"
                )
        elif method == "none":
            raise ValueError(
                "only an unbracketed target may use no interpolation"
            )
        if accepted and (
            (support_count == 2 and method != "linear")
            or (support_count >= 3 and method != "pchip")
            or support_count < 2
        ):
            raise ValueError(
                "accepted target interpolation must match support count"
            )
        if method != "none":
            tolerance = (
                64.0 * np.finfo(float).eps * max(1.0, abs(target))
            )
            lower = support[support < target]
            upper = support[support > target]
            if lower.size == 0 or upper.size == 0:
                raise ValueError(
                    "interpolated target supports must strictly bracket target"
                )
            lower_period = float(lower[-1])
            upper_period = float(upper[0])
            if (
                np.any(np.abs(support - target) > 0.10 + tolerance)
                or target - lower_period > 0.10 + tolerance
                or upper_period - target > 0.10 + tolerance
                or upper_period - lower_period > 0.20 + tolerance
            ):
                raise ValueError(
                    "interpolated target supports exceed Wang bracket limits"
                )
        support.setflags(write=False)
        object.__setattr__(self, "support_periods_s", support)
        object.__setattr__(self, "support_count", support_count)
        object.__setattr__(self, "accepted", accepted)
        rejected_nominal = (
            np.empty(0, dtype=float)
            if self.rejected_continuous_nominal_periods_s is None
            else np.array(
                _real_numeric_array(
                    self.rejected_continuous_nominal_periods_s,
                    "rejected_continuous_nominal_periods_s",
                ),
                dtype=float,
                copy=True,
            )
        )
        if (
            rejected_nominal.ndim != 1
            or np.any(~np.isfinite(rejected_nominal))
            or np.any(rejected_nominal <= 0)
        ):
            raise ValueError(
                "rejected continuous nominal periods must be positive, finite, "
                "and one-dimensional"
            )
        rejected_instantaneous = (
            np.empty(0, dtype=float)
            if self.rejected_continuous_instantaneous_periods_s is None
            else np.array(
                _real_numeric_array(
                    self.rejected_continuous_instantaneous_periods_s,
                    "rejected_continuous_instantaneous_periods_s",
                ),
                dtype=float,
                copy=True,
            )
        )
        if (
            rejected_instantaneous.ndim != 1
            or rejected_instantaneous.shape != rejected_nominal.shape
            or np.any(
                np.isfinite(rejected_instantaneous)
                & (rejected_instantaneous <= 0)
            )
            or np.any(np.isinf(rejected_instantaneous))
        ):
            raise ValueError(
                "rejected continuous instantaneous periods must match nominal "
                "periods and contain positive finite values or NaN"
            )
        rejection_statuses = tuple(self.continuous_rejection_statuses)
        if (
            len(rejection_statuses) != rejected_nominal.size
            or any(
                not isinstance(value, str) or not value
                for value in rejection_statuses
            )
        ):
            raise ValueError(
                "continuous_rejection_statuses must match rejected periods"
            )
        rejected_nominal.setflags(write=False)
        rejected_instantaneous.setflags(write=False)
        object.__setattr__(
            self,
            "rejected_continuous_nominal_periods_s",
            rejected_nominal,
        )
        object.__setattr__(
            self,
            "rejected_continuous_instantaneous_periods_s",
            rejected_instantaneous,
        )
        object.__setattr__(
            self,
            "continuous_rejection_statuses",
            rejection_statuses,
        )


@dataclass(frozen=True)
class PhaseUnwrapResult:
    unwrapped_phase_rad: np.ndarray
    cycle_counts: np.ndarray
    raw_phase_time_s: np.ndarray
    prediction_error_s: np.ndarray
    sort_order: np.ndarray
    valid_mask: np.ndarray
    anomaly_fraction: float
    max_consecutive_anomalies: int
    anchor_index: int
    status: str

    def __post_init__(self) -> None:
        unwrapped = np.array(self.unwrapped_phase_rad, dtype=float, copy=True)
        cycles = np.array(self.cycle_counts, dtype=int, copy=True)
        raw_time = np.array(self.raw_phase_time_s, dtype=float, copy=True)
        errors = np.array(self.prediction_error_s, dtype=float, copy=True)
        sort_order = np.array(self.sort_order, dtype=int, copy=True)
        valid_mask = np.array(self.valid_mask, dtype=bool, copy=True)
        if any(
            array.ndim != 1
            for array in (
                unwrapped,
                cycles,
                raw_time,
                errors,
                sort_order,
                valid_mask,
            )
        ):
            raise ValueError("phase unwrap result arrays must be one-dimensional")
        if not (
            unwrapped.shape
            == cycles.shape
            == raw_time.shape
            == errors.shape
            == sort_order.shape
            == valid_mask.shape
        ):
            raise ValueError("phase unwrap result arrays must have identical shapes")
        for array in (
            unwrapped,
            cycles,
            raw_time,
            errors,
            sort_order,
            valid_mask,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "unwrapped_phase_rad", unwrapped)
        object.__setattr__(self, "cycle_counts", cycles)
        object.__setattr__(self, "raw_phase_time_s", raw_time)
        object.__setattr__(self, "prediction_error_s", errors)
        object.__setattr__(self, "sort_order", sort_order)
        object.__setattr__(self, "valid_mask", valid_mask)


@dataclass(frozen=True)
class PhaseConventionValidationSummary:
    convention: PhaseConvention
    total_count: int
    valid_count: int
    failure_count: int
    noise_free_max_relative_error: float
    noisy_median_relative_error: float
    status: str

    @property
    def metadata(self) -> _FrozenMetadata:
        return _FrozenMetadata(
            convention=self.convention.name,
            total_count=self.total_count,
            valid_count=self.valid_count,
            failure_count=self.failure_count,
            noise_free_max_relative_error=(
                self.noise_free_max_relative_error
                if np.isfinite(self.noise_free_max_relative_error)
                else None
            ),
            noisy_median_relative_error=(
                self.noisy_median_relative_error
                if np.isfinite(self.noisy_median_relative_error)
                else None
            ),
            status=self.status,
        )


@dataclass(frozen=True)
class PhaseCandidate:
    branch: int
    phase_velocity_km_s: float
    phase_slowness_s_km: float


@dataclass(frozen=True)
class BranchSelection:
    branches: List[int]
    phase_velocities_km_s: np.ndarray
    reference_velocities_km_s: np.ndarray


@dataclass(frozen=True)
class DatTrace:
    pair_name: str
    distance_km: float
    dt_s: float
    time_s: np.ndarray
    positive_lag: np.ndarray
    negative_lag_reversed: np.ndarray
    symmetric_waveform: np.ndarray
    lon_a: float
    lat_a: float
    lon_b: float
    lat_b: float


@dataclass(frozen=True)
class PeriodMeasurement:
    convention: PhaseConvention
    period_s: float
    omega_inst_rad_s: float
    principal_paper_phase_rad: float
    unwrapped_paper_phase_rad: float
    raw_phase_time_s: float
    paper_phase_cycle_offset: int
    group_time_s: float
    group_velocity_km_s: float
    snr: float
    signal_window_start_s: float
    signal_window_end_s: float
    filtered_waveform: np.ndarray
    envelope: np.ndarray

    def __post_init__(self) -> None:
        _require_phase_convention(self.convention)
        positive_scalars = (
            "period_s",
            "omega_inst_rad_s",
            "group_time_s",
            "group_velocity_km_s",
        )
        for name in positive_scalars:
            object.__setattr__(
                self,
                name,
                _finite_scalar(getattr(self, name), name, positive=True),
            )
        for name in (
            "principal_paper_phase_rad",
            "unwrapped_paper_phase_rad",
            "raw_phase_time_s",
            "snr",
            "signal_window_start_s",
            "signal_window_end_s",
        ):
            object.__setattr__(
                self,
                name,
                _finite_scalar(getattr(self, name), name),
            )
        cycle_offset = _integer_scalar(
            self.paper_phase_cycle_offset,
            "paper_phase_cycle_offset",
        )
        object.__setattr__(
            self,
            "paper_phase_cycle_offset",
            cycle_offset,
        )
        if not (-math.pi < self.principal_paper_phase_rad <= math.pi):
            raise ValueError(
                "principal_paper_phase_rad must lie in the principal range (-pi, pi]"
            )
        if self.snr < 0:
            raise ValueError("snr must be non-negative")
        if self.signal_window_start_s > self.signal_window_end_s:
            raise ValueError(
                "signal_window_start_s must not exceed signal_window_end_s"
            )
        expected_unwrapped = (
            self.principal_paper_phase_rad
            + 2.0 * math.pi * cycle_offset
        )
        phase_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(expected_unwrapped))
        )
        if not math.isclose(
            self.unwrapped_paper_phase_rad,
            expected_unwrapped,
            rel_tol=0.0,
            abs_tol=phase_tolerance,
        ):
            raise ValueError(
                "unwrapped_paper_phase_rad must equal "
                "principal_paper_phase_rad + 2*pi*paper_phase_cycle_offset"
            )
        expected_raw_time = raw_phase_travel_time(
            convention=self.convention,
            group_time_s=self.group_time_s,
            phase_rad=self.unwrapped_paper_phase_rad,
            omega_rad_s=self.omega_inst_rad_s,
        )
        time_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(expected_raw_time))
        )
        if not math.isclose(
            self.raw_phase_time_s,
            expected_raw_time,
            rel_tol=0.0,
            abs_tol=time_tolerance,
        ):
            raise ValueError(
                "raw_phase_time_s is inconsistent with the unwrapped paper phase"
            )
        for name in ("filtered_waveform", "envelope"):
            array = np.array(
                _real_numeric_array(getattr(self, name), name),
                dtype=float,
                copy=True,
            )
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            if name == "envelope" and np.any(array < 0):
                raise ValueError("envelope must be non-negative")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.filtered_waveform.shape != self.envelope.shape:
            raise ValueError(
                "filtered_waveform and envelope must have identical shapes"
            )

    @property
    def phi_tu_rad(self) -> float:
        """Principal paper phase retained as a diagnostic compatibility alias."""

        return self.principal_paper_phase_rad


@dataclass(frozen=True)
class PhaseCurveMeasurement:
    convention: PhaseConvention
    periods_s: np.ndarray
    velocity_axis_km_s: np.ndarray
    group_times_s: np.ndarray
    scaled_log_energy: np.ndarray
    measurements: Tuple[Optional[PeriodMeasurement], ...]
    ridge: RidgeResult
    phase_unwrap: PhaseUnwrapResult
    instantaneous_periods_s: np.ndarray
    ridge_normalized_log_energy: np.ndarray
    ridge_normalized_envelope_amplitude: np.ndarray
    ridge_adjacent_jump_km_s: np.ndarray
    status: str
    measurement_valid: Optional[np.ndarray] = None
    measurement_statuses: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        _require_phase_convention(self.convention)
        for name in ("periods_s", "velocity_axis_km_s", "group_times_s"):
            array = np.array(
                _real_numeric_array(getattr(self, name), name),
                dtype=float,
                copy=True,
            )
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if (
            self.periods_s.size < 3
            or np.any(self.periods_s <= 0)
            or np.any(np.diff(self.periods_s) <= 0)
        ):
            raise ValueError(
                "periods_s must contain at least three positive increasing values"
            )
        if (
            self.velocity_axis_km_s.size < 2
            or np.any(self.velocity_axis_km_s <= 0)
            or np.any(np.diff(self.velocity_axis_km_s) <= 0)
        ):
            raise ValueError(
                "velocity_axis_km_s must contain positive increasing values"
            )
        if np.any(self.group_times_s <= 0):
            raise ValueError("group_times_s must contain positive values")
        energy = np.array(
            _real_numeric_array(
                self.scaled_log_energy,
                "scaled_log_energy",
            ),
            dtype=float,
            copy=True,
        )
        if energy.shape != (self.periods_s.size, self.velocity_axis_km_s.size):
            raise ValueError("scaled_log_energy must match period and velocity axes")
        if np.any(~np.isfinite(energy)):
            raise ValueError("scaled_log_energy must contain only finite values")
        energy.setflags(write=False)
        object.__setattr__(self, "scaled_log_energy", energy)
        for name in (
            "ridge_normalized_log_energy",
            "ridge_normalized_envelope_amplitude",
            "ridge_adjacent_jump_km_s",
        ):
            array = np.array(
                _real_numeric_array(getattr(self, name), name),
                dtype=float,
                copy=True,
            )
            if (
                array.ndim != 1
                or array.shape != self.periods_s.shape
                or np.any(~np.isfinite(array))
                or np.any(array < 0)
                or (
                    name != "ridge_adjacent_jump_km_s"
                    and np.any(array > 1)
                )
            ):
                raise ValueError(
                    f"{name} must be finite, match periods_s, and lie in "
                    + (
                        "[0, infinity)"
                        if name == "ridge_adjacent_jump_km_s"
                        else "[0, 1]"
                    )
                )
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        rows = tuple(self.measurements)
        if len(rows) != self.periods_s.size:
            raise ValueError("measurements must match periods_s")
        if any(
            row is not None and not isinstance(row, PeriodMeasurement)
            for row in rows
        ):
            raise ValueError(
                "every measurement must be a PeriodMeasurement or None"
            )
        if self.measurement_valid is None:
            measurement_valid = np.asarray(
                [row is not None for row in rows],
                dtype=bool,
            )
        else:
            raw_valid = np.asarray(self.measurement_valid)
            if (
                raw_valid.dtype.kind != "b"
                or raw_valid.ndim != 1
                or raw_valid.shape != self.periods_s.shape
            ):
                raise ValueError(
                    "measurement_valid must be a boolean array matching periods_s"
                )
            measurement_valid = np.array(raw_valid, dtype=bool, copy=True)
        if any(
            (row is None) == bool(measurement_valid[index])
            for index, row in enumerate(rows)
        ):
            raise ValueError(
                "measurement_valid must be true exactly where measurements exist"
            )
        if any(
            row is not None and row.convention is not self.convention
            for row in rows
        ):
            raise ValueError("every measurement convention must match the curve")
        if self.measurement_statuses is None:
            statuses = tuple(
                "accepted" if valid else "invalid_measurement"
                for valid in measurement_valid
            )
        else:
            statuses = tuple(self.measurement_statuses)
            if (
                len(statuses) != self.periods_s.size
                or any(not isinstance(value, str) or not value for value in statuses)
            ):
                raise ValueError(
                    "measurement_statuses must contain one non-empty string per period"
                )
        for valid, row_status in zip(measurement_valid, statuses):
            if bool(valid) != (row_status == "accepted"):
                raise ValueError(
                    "measurement status must be accepted exactly for valid rows"
                )
        expected_curve_status = (
            "accepted"
            if np.all(measurement_valid)
            else "partial_phase_curve"
        )
        if self.status != expected_curve_status:
            raise ValueError(
                "curve status must match complete or partial measurement validity"
            )
        measurement_valid.setflags(write=False)
        object.__setattr__(self, "measurement_valid", measurement_valid)
        object.__setattr__(self, "measurement_statuses", statuses)
        instantaneous_periods = np.array(
            _real_numeric_array(
                self.instantaneous_periods_s,
                "instantaneous_periods_s",
            ),
            dtype=float,
            copy=True,
        )
        if (
            instantaneous_periods.ndim != 1
            or instantaneous_periods.shape != self.periods_s.shape
            or np.any(np.isinf(instantaneous_periods))
            or np.any(
                np.isfinite(instantaneous_periods)
                & (instantaneous_periods <= 0)
            )
        ):
            raise ValueError(
                "instantaneous_periods_s must match periods_s and contain "
                "positive finite values or NaN"
            )
        post_frequency_statuses = {
            "accepted",
            "duplicate_instantaneous_period",
            "phase_unwrap_discontinuous",
            "outside_anchored_phase_segment",
        }
        for index, row_status in enumerate(statuses):
            known = np.isfinite(instantaneous_periods[index])
            if known != (row_status in post_frequency_statuses):
                raise ValueError(
                    "instantaneous_periods_s must be finite exactly for rows "
                    "that passed instantaneous-frequency estimation"
                )
        instantaneous_periods.setflags(write=False)
        object.__setattr__(
            self,
            "instantaneous_periods_s",
            instantaneous_periods,
        )
        for index, row in enumerate(rows):
            if row is None:
                continue
            scalar_tolerance = (
                64.0
                * np.finfo(float).eps
                * max(
                    1.0,
                    abs(float(self.periods_s[index])),
                    abs(float(self.group_times_s[index])),
                )
            )
            if not math.isclose(
                row.period_s,
                float(self.periods_s[index]),
                rel_tol=0.0,
                abs_tol=scalar_tolerance,
            ):
                raise ValueError(
                    "measurement periods must match periods_s"
                )
            if not math.isclose(
                row.group_time_s,
                float(self.group_times_s[index]),
                rel_tol=0.0,
                abs_tol=scalar_tolerance,
            ):
                raise ValueError(
                    "measurement group times must match group_times_s"
                )
            expected_instantaneous_period = (
                2.0 * math.pi / row.omega_inst_rad_s
            )
            if not math.isclose(
                expected_instantaneous_period,
                float(instantaneous_periods[index]),
                rel_tol=0.0,
                abs_tol=64.0
                * np.finfo(float).eps
                * max(1.0, abs(expected_instantaneous_period)),
            ):
                raise ValueError(
                    "valid measurement instantaneous period must match "
                    "omega_inst_rad_s"
                )
        if not isinstance(self.ridge, RidgeResult):
            raise ValueError("ridge must be a RidgeResult")
        if self.ridge.row_indices.shape != self.periods_s.shape:
            raise ValueError("ridge arrays must match periods_s")
        if np.any(self.ridge.row_indices < 0) or np.any(
            self.ridge.row_indices >= self.velocity_axis_km_s.size
        ):
            raise ValueError("ridge row indices must lie on the velocity axis")
        expected_ridge_velocity = self.velocity_axis_km_s[
            self.ridge.row_indices
        ]
        if not np.allclose(
            self.ridge.group_velocities_km_s,
            expected_ridge_velocity,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        ):
            raise ValueError(
                "ridge velocities must match its velocity-axis indices"
            )
        if not isinstance(self.phase_unwrap, PhaseUnwrapResult):
            raise ValueError("phase_unwrap must be a PhaseUnwrapResult")
        if self.phase_unwrap.unwrapped_phase_rad.shape != self.periods_s.shape:
            raise ValueError("phase unwrap arrays must match periods_s")
        if not np.array_equal(
            self.phase_unwrap.valid_mask,
            measurement_valid,
        ):
            raise ValueError(
                "phase_unwrap valid_mask must match measurement_valid"
            )
        valid_indices = np.flatnonzero(measurement_valid)
        if valid_indices.size == 0:
            if not isinstance(self.status, str) or not self.status:
                raise ValueError("status must be a non-empty string")
            object.__setattr__(self, "measurements", rows)
            return
        multiplier = self.convention.definition.scipy_phase_multiplier
        expected_paper_phase = (
            multiplier
            * self.phase_unwrap.unwrapped_phase_rad[valid_indices]
        )
        actual_paper_phase = np.asarray(
            [
                rows[index].unwrapped_paper_phase_rad
                for index in valid_indices
            ],
            dtype=float,
        )
        actual_raw_time = np.asarray(
            [rows[index].raw_phase_time_s for index in valid_indices],
            dtype=float,
        )
        actual_principal_phase = np.asarray(
            [
                rows[index].principal_paper_phase_rad
                for index in valid_indices
            ],
            dtype=float,
        )
        actual_cycle_offsets = np.asarray(
            [
                rows[index].paper_phase_cycle_offset
                for index in valid_indices
            ],
            dtype=int,
        )
        scipy_principal_phase = (
            self.phase_unwrap.unwrapped_phase_rad[valid_indices]
            - 2.0
            * math.pi
            * self.phase_unwrap.cycle_counts[valid_indices]
        )
        if np.any(scipy_principal_phase <= -math.pi) or np.any(
            scipy_principal_phase > math.pi
        ):
            raise ValueError(
                "phase_unwrap cycles are inconsistent with principal phases"
            )
        expected_principal_phase = np.asarray(
            [
                _canonical_principal_phase_rad(multiplier * phase)
                for phase in scipy_principal_phase
            ],
            dtype=float,
        )
        expected_cycle_offsets = np.rint(
            (
                expected_paper_phase
                - expected_principal_phase
            )
            / (2.0 * math.pi)
        ).astype(int)
        reconstructed_paper_phase = (
            expected_principal_phase
            + 2.0 * math.pi * expected_cycle_offsets
        )
        cross_tolerance = (
            128.0
            * np.finfo(float).eps
            * max(
                1.0,
                float(np.max(np.abs(expected_paper_phase))),
                float(
                    np.max(
                        np.abs(
                            self.phase_unwrap.raw_phase_time_s[valid_indices]
                        )
                    )
                ),
            )
        )
        if not np.allclose(
            actual_paper_phase,
            expected_paper_phase,
            rtol=0.0,
            atol=cross_tolerance,
        ):
            raise ValueError(
                "measurement unwrapped phases must match phase_unwrap"
            )
        if not np.allclose(
            reconstructed_paper_phase,
            expected_paper_phase,
            rtol=0.0,
            atol=cross_tolerance,
        ):
            raise ValueError(
                "phase_unwrap does not define integer paper phase cycles"
            )
        if not np.allclose(
            actual_principal_phase,
            expected_principal_phase,
            rtol=0.0,
            atol=cross_tolerance,
        ):
            raise ValueError(
                "measurement principal paper phases must match phase_unwrap"
            )
        if not np.allclose(
            actual_raw_time,
            self.phase_unwrap.raw_phase_time_s[valid_indices],
            rtol=0.0,
            atol=cross_tolerance,
        ):
            raise ValueError(
                "measurement raw phase times must match phase_unwrap"
            )
        if not np.array_equal(
            actual_cycle_offsets,
            expected_cycle_offsets,
        ):
            raise ValueError(
                "measurement paper phase cycle offsets must match phase_unwrap cycles"
            )
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a non-empty string")
        object.__setattr__(self, "measurements", rows)


@dataclass(frozen=True)
class BensenExtraction:
    trace: DatTrace
    periods_s: np.ndarray
    measurements: List[Optional[PeriodMeasurement]]
    group_velocities_km_s: np.ndarray
    phase_velocities_km_s: np.ndarray
    branches: List[Optional[int]]
    snr: np.ndarray
    reference_velocities_km_s: np.ndarray


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


def read_dat_trace(path: str) -> DatTrace:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lon_a, lat_a = map(float, handle.readline().split()[:2])
        lon_b, lat_b = map(float, handle.readline().split()[:2])
    table = np.loadtxt(path, skiprows=2, ndmin=2)
    time_s = np.asarray(table[:, 0], dtype=float)
    positive = np.asarray(table[:, 1], dtype=float)
    negative = np.asarray(table[:, 2], dtype=float)
    symmetric = 0.5 * (positive + negative)
    dt_s = float(time_s[1] - time_s[0])
    return DatTrace(
        pair_name=path.rsplit("/", 1)[-1].rsplit(".", 1)[0],
        distance_km=_distance_km(lat_a, lon_a, lat_b, lon_b),
        dt_s=dt_s,
        time_s=time_s,
        positive_lag=positive,
        negative_lag_reversed=negative,
        symmetric_waveform=symmetric,
        lon_a=lon_a,
        lat_a=lat_a,
        lon_b=lon_b,
        lat_b=lat_b,
    )


def gaussian_alpha_for_distance(distance_km: float) -> float:
    distance = _finite_scalar(distance_km, "distance_km")
    return float(
        np.interp(
            distance,
            [0, 100, 250, 500, 1000, 2000, 4000, 20000],
            [5, 8, 12, 20, 25, 35, 50, 75],
        )
    )


def gaussian_narrowband_waveform(
    waveform: np.ndarray,
    *,
    dt_s: float,
    period_s: float,
    alpha: float,
) -> np.ndarray:
    """Compatibility wrapper around :func:`gaussian_filter_bank`."""

    period = _finite_scalar(period_s, "period_s", positive=True)
    result = gaussian_filter_bank(
        waveform,
        dt_s=dt_s,
        periods_s=np.asarray([period], dtype=float),
        alpha=alpha,
    )
    return result.filtered_waveforms[0]


def _deterministic_complex_magnitude(
    values: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return complex magnitude independent of input alignment and row stride."""

    complex_values = np.asarray(values)
    if not np.iscomplexobj(complex_values):
        raise ValueError("values must be a complex array")
    return np.hypot(complex_values.real, complex_values.imag, out=out)


def gaussian_filter_bank(
    waveform: np.ndarray,
    *,
    dt_s: float,
    periods_s: np.ndarray,
    alpha: float,
) -> GaussianFilterBankResult:
    """Filter one real waveform at multiple periods.

    Every returned array has shape ``(n_periods, n_samples)``: rows are filter
    periods and columns are input time samples.
    """

    waveform = _real_numeric_array(waveform, "waveform")
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(waveform)):
        raise ValueError("waveform must contain only finite values")
    dt_s = _finite_scalar(dt_s, "dt_s", positive=True)
    periods = _real_numeric_array(periods_s, "periods_s")
    if periods.ndim != 1 or periods.size == 0:
        raise ValueError("periods_s must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError("periods_s must contain only positive finite values")
    alpha = _finite_scalar(alpha, "alpha", positive=True)

    npts = waveform.size
    nfft = int(2 ** np.ceil(np.log2(max(npts, 1024))))
    spectrum = fft(waveform, nfft)
    freq = np.fft.fftfreq(nfft, d=dt_s)[np.newaxis, :]
    center_hz = (1.0 / periods)[:, np.newaxis]
    taper = np.exp(-alpha * ((np.abs(freq) - center_hz) / center_hz) ** 2)
    filtered = np.real(ifft(spectrum[np.newaxis, :] * taper, axis=1))[:, :npts]
    filtered = np.asarray(filtered, dtype=float)
    analytic = np.asarray(hilbert(filtered, axis=1), dtype=complex)
    envelope = _deterministic_complex_magnitude(analytic)
    return GaussianFilterBankResult(
        filtered_waveforms=filtered,
        analytic_signals=analytic,
        envelope=envelope,
    )


def phase_matched_filter(
    waveform: np.ndarray,
    *,
    dt_s: float,
    periods_s: np.ndarray,
    group_travel_times_s: np.ndarray,
    first_pass_alpha: float,
    maximum_period_s: float = PHASE_MATCHING_MAXIMUM_PERIOD_S,
    cut_taper_alpha: float = PHASE_MATCHING_CUT_TAPER_ALPHA,
) -> PhaseMatchedFilterResult:
    """Run Bensen's diagnostic de-dispersion, cleaning and re-dispersion.

    The group-delay curve is integrated in angular frequency to construct a
    phase-only correction.  Its median delay is retained, so the compressed
    wave packet remains inside the original time axis.  The cleaned compressed
    signal is then re-dispersed with the exact inverse phase operator.
    """

    signal = _real_numeric_array(waveform, "waveform")
    if signal.ndim != 1 or signal.size < 3 or np.any(~np.isfinite(signal)):
        raise ValueError("waveform must be a finite one-dimensional array")
    dt = _finite_scalar(dt_s, "dt_s", positive=True)
    periods = _real_numeric_array(periods_s, "periods_s")
    group_times = _real_numeric_array(
        group_travel_times_s,
        "group_travel_times_s",
    )
    if periods.ndim != 1 or group_times.ndim != 1:
        raise ValueError("periods and group travel times must be one-dimensional")
    if periods.shape != group_times.shape:
        raise ValueError("periods and group travel times must have the same shape")
    if periods.size < 2:
        raise ValueError("phase matching needs at least two group-delay samples")
    if (
        np.any(~np.isfinite(periods))
        or np.any(periods <= 0)
        or np.any(~np.isfinite(group_times))
        or np.any(group_times <= 0)
        or np.unique(periods).size != periods.size
    ):
        raise ValueError("phase-matching dispersion curve is invalid")
    first_alpha = _finite_scalar(
        first_pass_alpha,
        "first_pass_alpha",
        positive=True,
    )
    maximum_period = _finite_scalar(
        maximum_period_s,
        "maximum_period_s",
        positive=True,
    )
    taper_alpha = _finite_scalar(cut_taper_alpha, "cut_taper_alpha")
    if not 0 <= taper_alpha <= 1:
        raise ValueError("cut_taper_alpha must lie in [0, 1]")
    if (
        maximum_period != PHASE_MATCHING_MAXIMUM_PERIOD_S
        or taper_alpha != PHASE_MATCHING_CUT_TAPER_ALPHA
    ):
        raise ValueError(
            "phase-matching validation requires frozen T_max=5.0 s "
            "and Tukey alpha=0.25"
        )

    curve_frequency_hz = 1.0 / periods
    order = np.argsort(curve_frequency_hz, kind="stable")
    curve_frequency_hz = curve_frequency_hz[order]
    curve_group_times_s = group_times[order]
    frequencies_hz = np.fft.rfftfreq(signal.size, d=dt)
    interpolated_group_times_s = np.interp(
        frequencies_hz,
        curve_frequency_hz,
        curve_group_times_s,
    )
    reference_group_time_s = float(np.median(curve_group_times_s))
    differential_delay_s = (
        interpolated_group_times_s - reference_group_time_s
    )
    in_band = (
        (frequencies_hz >= curve_frequency_hz[0])
        & (frequencies_hz <= curve_frequency_hz[-1])
    )
    differential_delay_s[~in_band] = 0.0
    angular_frequency = 2.0 * np.pi * frequencies_hz
    phase_integral = np.zeros_like(angular_frequency)
    if phase_integral.size > 1:
        phase_integral[1:] = np.cumsum(
            0.5
            * (differential_delay_s[1:] + differential_delay_s[:-1])
            * np.diff(angular_frequency)
        )
    correction = np.exp(1j * phase_integral)
    spectrum = np.fft.rfft(signal)
    compressed = np.fft.irfft(spectrum * correction, n=signal.size)
    compressed = np.asarray(compressed, dtype=float)
    compressed_envelope = _deterministic_complex_magnitude(
        hilbert(compressed)
    )
    cut_center_index = int(np.argmax(compressed_envelope))
    cut_half_width_s = 2.0 * maximum_period
    relative_time_s = (
        np.arange(signal.size, dtype=float) - cut_center_index
    ) * dt
    normalized_radius = np.abs(relative_time_s) / cut_half_width_s
    cleaning_window = np.ones(signal.size, dtype=float)
    cleaning_window[normalized_radius >= 1.0] = 0.0
    if taper_alpha > 0:
        taper_start = 1.0 - taper_alpha
        taper = (
            (normalized_radius > taper_start)
            & (normalized_radius < 1.0)
        )
        cleaning_window[taper] = 0.5 * (
            1.0
            + np.cos(
                np.pi
                * (normalized_radius[taper] - taper_start)
                / taper_alpha
            )
        )
    cleaned_compressed = compressed * cleaning_window
    cleaned_spectrum = np.fft.rfft(cleaned_compressed)
    cleaned_waveform = np.fft.irfft(
        cleaned_spectrum * np.conjugate(correction),
        n=signal.size,
    )
    return PhaseMatchedFilterResult(
        compressed_waveform=compressed,
        compressed_envelope=compressed_envelope,
        cleaning_window=cleaning_window,
        cleaned_compressed_waveform=cleaned_compressed,
        cleaned_waveform=np.asarray(cleaned_waveform, dtype=float),
        cut_center_index=cut_center_index,
        cut_center_time_s=cut_center_index * dt,
        cut_half_width_s=cut_half_width_s,
        cut_taper_alpha=taper_alpha,
        first_pass_alpha=first_alpha,
        second_pass_alpha=min(2.0 * first_alpha, 50.0),
        reference_group_time_s=reference_group_time_s,
    )


def phase_matched_second_pass_ftan(
    waveform: np.ndarray,
    *,
    dt_s: float,
    periods_s: np.ndarray,
    group_travel_times_s: np.ndarray,
    first_pass_alpha: float,
) -> PhaseMatchedSecondPassResult:
    """Clean a dispersed packet and execute the frozen second Gaussian FTAN."""

    cleaning = phase_matched_filter(
        waveform,
        dt_s=dt_s,
        periods_s=periods_s,
        group_travel_times_s=group_travel_times_s,
        first_pass_alpha=first_pass_alpha,
    )
    second_pass = gaussian_filter_bank(
        cleaning.cleaned_waveform,
        dt_s=dt_s,
        periods_s=periods_s,
        alpha=cleaning.second_pass_alpha,
    )
    return PhaseMatchedSecondPassResult(
        cleaning=cleaning,
        second_pass_filter_bank=second_pass,
    )


def normalized_log_energy(
    envelope: np.ndarray,
    *,
    period_axis: int = 0,
) -> np.ndarray:
    """Return independently normalized log energy for every FTAN period.

    ``period_axis`` identifies the period dimension of the two-dimensional
    input. Each period slice is divided by its own maximum envelope before
    squared energy is floored by ``LOG_ENERGY_FLOOR`` and logged. The result is
    then scaled to ``[0, 1]`` along that period's sample or velocity axis.
    All-zero and constant period slices are returned as zero.
    """

    envelope = _real_numeric_array(envelope, "envelope")
    if envelope.ndim != 2 or 0 in envelope.shape:
        raise ValueError("envelope must be a non-empty two-dimensional array")
    if np.any(~np.isfinite(envelope)):
        raise ValueError("envelope must contain only finite values")
    if np.any(envelope < 0):
        raise ValueError("envelope must contain only non-negative values")
    period_axis = _integer_scalar(period_axis, "period_axis")
    if period_axis not in (0, 1):
        raise ValueError("period_axis must be 0 or 1")

    sample_axis = 1 - period_axis
    period_max = np.max(envelope, axis=sample_axis, keepdims=True)
    relative_envelope = np.zeros_like(envelope)
    np.divide(
        envelope,
        period_max,
        out=relative_envelope,
        where=period_max > 0,
    )
    energy = np.maximum(relative_envelope**2, LOG_ENERGY_FLOOR)
    log_energy = np.log(energy)
    period_min = np.min(log_energy, axis=sample_axis, keepdims=True)
    period_span = (
        np.max(log_energy, axis=sample_axis, keepdims=True) - period_min
    )
    normalized = np.zeros_like(log_energy)
    np.divide(
        log_energy - period_min,
        period_span,
        out=normalized,
        where=period_span > 0,
    )
    return np.clip(normalized, 0.0, 1.0)


def refine_group_arrival(
    time_s: np.ndarray,
    envelope: np.ndarray,
    peak_index: int,
) -> GroupArrivalRefinement:
    """Refine one ridge-grid arrival with only its two adjacent samples.

    Invalid local quadratics fall back to the supplied grid peak. The function
    deliberately does not search a wider window, because that would turn a
    ridge refinement into a second and potentially inconsistent peak picker.
    """

    time = _real_numeric_array(time_s, "time_s")
    amplitude = _real_numeric_array(envelope, "envelope")
    if time.ndim != 1 or amplitude.ndim != 1:
        raise ValueError("time_s and envelope must be one-dimensional")
    if time.size == 0 or time.shape != amplitude.shape:
        raise ValueError("time_s and envelope must be non-empty with the same shape")
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(amplitude)):
        raise ValueError("time_s and envelope must contain only finite values")
    if time.size > 1:
        time_steps = np.diff(time)
        if np.any(time_steps <= 0):
            raise ValueError("time_s must be strictly increasing")
    try:
        integer_peak = _integer_scalar(peak_index, "peak_index")
    except ValueError:
        raise ValueError("peak_index must be an integer within the input") from None
    if not 0 <= integer_peak < time.size:
        raise ValueError("peak_index must be an integer within the input")

    grid_time = float(time[integer_peak])

    def fallback(status: str) -> GroupArrivalRefinement:
        return GroupArrivalRefinement(
            peak_index=integer_peak,
            grid_time_s=grid_time,
            group_time_s=grid_time,
            vertex_offset_samples=0.0,
            refinement_used=False,
            status=status,
        )

    if integer_peak == 0 or integer_peak == time.size - 1:
        return fallback("boundary_peak")

    left = float(amplitude[integer_peak - 1])
    center = float(amplitude[integer_peak])
    right = float(amplitude[integer_peak + 1])
    if center < left or center < right:
        return fallback("not_local_peak")
    local_time = time[integer_peak - 1 : integer_peak + 2] - grid_time
    local_amplitude = amplitude[integer_peak - 1 : integer_peak + 2]
    time_scale = max(abs(float(local_time[0])), abs(float(local_time[2])))
    normalized_time = local_time / time_scale
    centered_amplitude = local_amplitude - center
    amplitude_scale = float(np.max(np.abs(centered_amplitude)))
    if amplitude_scale == 0.0:
        return fallback("degenerate_quadratic")
    normalized_amplitude = centered_amplitude / amplitude_scale
    try:
        quadratic, linear, _ = np.linalg.solve(
            np.column_stack(
                (
                    normalized_time**2,
                    normalized_time,
                    np.ones(3, dtype=float),
                )
            ),
            normalized_amplitude,
        )
    except np.linalg.LinAlgError:
        return fallback("quadratic_fit_singular")
    curvature_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, abs(float(quadratic)), abs(float(linear)))
    )
    if abs(quadratic) <= curvature_tolerance:
        return fallback("degenerate_quadratic")
    if quadratic > 0.0:
        return fallback("nonnegative_curvature")

    vertex_normalized_time = -linear / (2.0 * quadratic)
    if (
        not np.isfinite(vertex_normalized_time)
        or vertex_normalized_time < normalized_time[0]
        or vertex_normalized_time > normalized_time[2]
    ):
        return fallback("vertex_outside_neighbors")

    vertex_local_time = vertex_normalized_time * time_scale
    refined_time = grid_time + vertex_local_time
    if (
        not np.isfinite(refined_time)
        or refined_time < time[integer_peak - 1]
        or refined_time > time[integer_peak + 1]
    ):
        return fallback("vertex_outside_neighbors")
    representative_dt = 0.5 * float(
        time[integer_peak + 1] - time[integer_peak - 1]
    )
    vertex_offset = vertex_local_time / representative_dt
    return GroupArrivalRefinement(
        peak_index=integer_peak,
        grid_time_s=grid_time,
        group_time_s=float(refined_time),
        vertex_offset_samples=float(vertex_offset),
        refinement_used=True,
        status="refined",
    )


def interpolate_analytic_phase_at_arrival(
    time_s: np.ndarray,
    analytic_signal: np.ndarray,
    group_time_s: float,
    *,
    convention=None,  # type: Optional[PhaseConvention]
) -> float:
    """Interpolate analytic components before taking a principal phase.

    With no convention this preserves the legacy SciPy-Hilbert phase. Passing
    a convention converts that increasing-time phase to the forward-transform
    phase used by the Bensen/Lin formulas.
    """

    time = _real_numeric_array(time_s, "time_s")
    analytic = _complex_numeric_array(analytic_signal, "analytic_signal")
    if time.ndim != 1 or analytic.ndim != 1:
        raise ValueError("time_s and analytic_signal must be one-dimensional")
    if time.size == 0 or time.shape != analytic.shape:
        raise ValueError(
            "time_s and analytic_signal must be non-empty with the same shape"
        )
    if (
        np.any(~np.isfinite(time))
        or np.any(~np.isfinite(analytic.real))
        or np.any(~np.isfinite(analytic.imag))
    ):
        raise ValueError("time_s and analytic_signal must contain only finite values")
    if time.size < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing")
    arrival = _finite_scalar(group_time_s, "group_time_s")
    if arrival < time[0] or arrival > time[-1]:
        raise ValueError("group_time_s must lie within the time_s range")
    real = float(np.interp(arrival, time, analytic.real))
    imaginary = float(np.interp(arrival, time, analytic.imag))
    if real == 0.0 and imaginary == 0.0:
        raise ValueError("interpolated analytic signal has zero amplitude")
    phase = _canonical_principal_phase_rad(
        math.atan2(imaginary, real)
    )
    if convention is not None:
        selected = _require_phase_convention(convention)
        phase = _canonical_principal_phase_rad(
            selected.definition.scipy_phase_multiplier * phase
        )
    return phase


def _invalid_instantaneous_frequency(
    *,
    window_sample_count: int,
) -> InstantaneousFrequencyResult:
    return InstantaneousFrequencyResult(
        fitted_phase_intercept_rad=float("nan"),
        fitted_phase_slope_rad_s=float("nan"),
        omega_inst_rad_s=float("nan"),
        instantaneous_period_s=float("nan"),
        period_ratio=float("nan"),
        window_sample_count=int(window_sample_count),
        status="invalid_instantaneous_frequency",
    )


def estimate_instantaneous_frequency(
    time_s: np.ndarray,
    phase_rad: np.ndarray,
    *,
    group_time_s: float,
    nominal_period_s: float,
) -> InstantaneousFrequencyResult:
    """Fit signed local phase slope in one nominal-period Huber window."""

    time = _real_numeric_array(time_s, "time_s")
    phase = _real_numeric_array(phase_rad, "phase_rad")
    if time.ndim != 1 or phase.ndim != 1:
        raise ValueError("time_s and phase_rad must be one-dimensional")
    if time.size == 0 or time.shape != phase.shape:
        raise ValueError("time_s and phase_rad must be non-empty with the same shape")
    if np.any(~np.isfinite(time)):
        raise ValueError("time_s must contain only finite values")
    if time.size < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing")
    group_time = _finite_scalar(group_time_s, "group_time_s")
    nominal_period = _finite_scalar(
        nominal_period_s,
        "nominal_period_s",
        positive=True,
    )
    half_window = 0.5 * nominal_period
    window_start = group_time - half_window
    window_end = group_time + half_window
    time_tolerance = (
        32.0
        * np.finfo(float).eps
        * max(1.0, abs(window_start), abs(window_end))
    )
    if (
        window_start < float(time[0]) - time_tolerance
        or window_end > float(time[-1]) + time_tolerance
    ):
        return _invalid_instantaneous_frequency(window_sample_count=0)

    window = (
        (time >= window_start - time_tolerance)
        & (time <= window_end + time_tolerance)
    )
    sample_count = int(np.count_nonzero(window))
    if sample_count < 5 or np.any(~np.isfinite(phase)):
        return _invalid_instantaneous_frequency(
            window_sample_count=sample_count,
        )

    local_time = time[window] - group_time
    local_phase = np.unwrap(phase[window])
    if np.any(~np.isfinite(local_phase)):
        return _invalid_instantaneous_frequency(
            window_sample_count=sample_count,
        )
    design = np.column_stack((np.ones(sample_count), local_time))
    initial, _, _, _ = np.linalg.lstsq(design, local_phase, rcond=None)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return parameters[0] + parameters[1] * local_time - local_phase

    fit = least_squares(
        residual,
        initial,
        loss="huber",
        f_scale=1.0,
    )
    if not fit.success or fit.x.shape != (2,) or np.any(~np.isfinite(fit.x)):
        return _invalid_instantaneous_frequency(
            window_sample_count=sample_count,
        )
    intercept = float(fit.x[0])
    signed_slope = float(fit.x[1])
    omega = abs(signed_slope)
    if not np.isfinite(omega) or omega <= np.finfo(float).eps:
        return _invalid_instantaneous_frequency(
            window_sample_count=sample_count,
        )
    instantaneous_period = 2.0 * math.pi / omega
    ratio = instantaneous_period / nominal_period
    ratio_tolerance = 64.0 * np.finfo(float).eps
    if (
        not np.isfinite(instantaneous_period)
        or not np.isfinite(ratio)
        or ratio < 0.90 - ratio_tolerance
        or ratio > 1.10 + ratio_tolerance
    ):
        return InstantaneousFrequencyResult(
            fitted_phase_intercept_rad=intercept,
            fitted_phase_slope_rad_s=signed_slope,
            omega_inst_rad_s=omega,
            instantaneous_period_s=float(instantaneous_period),
            period_ratio=float(ratio),
            window_sample_count=sample_count,
            status="invalid_instantaneous_frequency",
        )
    return InstantaneousFrequencyResult(
        fitted_phase_intercept_rad=intercept,
        fitted_phase_slope_rad_s=signed_slope,
        omega_inst_rad_s=omega,
        instantaneous_period_s=float(instantaneous_period),
        period_ratio=float(ratio),
        window_sample_count=sample_count,
        status="accepted",
    )


def _choose_cycle_count(
    *,
    previous_unwrapped_phase_rad: float,
    principal_phase_rad: float,
) -> int:
    """Choose the deterministic nearest 2π branch for one adjacent column."""

    approximate = (
        float(previous_unwrapped_phase_rad) - float(principal_phase_rad)
    ) / (2.0 * math.pi)
    lower = math.floor(approximate)
    candidates = range(lower - 1, lower + 3)
    candidate_rows = [
        (
            int(k),
            abs(
                float(principal_phase_rad)
                + 2.0 * math.pi * int(k)
                - float(previous_unwrapped_phase_rad)
            ),
        )
        for k in candidates
    ]
    minimum_error = min(error for _, error in candidate_rows)
    tie_tolerance = (
        32.0
        * np.finfo(float).eps
        * max(1.0, abs(float(previous_unwrapped_phase_rad)))
    )
    tied = [
        k
        for k, error in candidate_rows
        if error <= minimum_error + tie_tolerance
    ]
    return min(tied, key=lambda k: (abs(k), k))


def _has_excessive_cycle_step(cycle_counts: np.ndarray) -> bool:
    cycles = np.asarray(cycle_counts)
    if cycles.ndim != 1:
        raise ValueError("cycle_counts must be one-dimensional")
    if cycles.size < 2:
        return False
    return bool(
        np.any(np.abs(np.diff(cycles.astype(np.int64))) > PHASE_UNWRAP_MAX_CYCLE_STEP)
    )


def _maximum_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _phase_time_prediction_errors(
    sorted_periods_s: np.ndarray,
    sorted_raw_phase_time_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return internal linear-prediction errors on a sorted nonuniform grid."""

    periods = np.asarray(sorted_periods_s, dtype=float)
    raw_time = np.asarray(sorted_raw_phase_time_s, dtype=float)
    if periods.ndim != 1 or raw_time.ndim != 1:
        raise ValueError("periods and raw phase time must be one-dimensional")
    if periods.shape != raw_time.shape or periods.size < 3:
        raise ValueError(
            "periods and raw phase time must have the same shape with at least three points"
        )
    if (
        np.any(~np.isfinite(periods))
        or np.any(periods <= 0)
        or np.any(np.diff(periods) <= 0)
    ):
        raise ValueError("periods must be positive, finite, unique, and increasing")
    if np.any(~np.isfinite(raw_time)):
        raise ValueError("raw phase time must contain only finite values")

    errors = np.full(periods.size, np.nan, dtype=float)
    anomalies = np.zeros(periods.size - 2, dtype=bool)
    for index in range(1, periods.size - 1):
        interpolation_fraction = (
            (periods[index] - periods[index - 1])
            / (periods[index + 1] - periods[index - 1])
        )
        predicted_time = (
            raw_time[index - 1]
            + interpolation_fraction
            * (raw_time[index + 1] - raw_time[index - 1])
        )
        error = abs(raw_time[index] - predicted_time)
        errors[index] = error
        threshold = PHASE_UNWRAP_PREDICTION_FRACTION * periods[index]
        comparison_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, error, threshold)
        )
        anomalies[index - 1] = error > threshold + comparison_tolerance
    return errors, anomalies


def unwrap_phase_along_frequency(
    instantaneous_periods_s: np.ndarray,
    principal_phase_rad: np.ndarray,
    group_times_s: np.ndarray,
    *,
    phase_time_sign=None,  # type: Optional[int]
    convention=None,  # type: Optional[PhaseConvention]
    anchor_period_s: float = 3.5,
) -> PhaseUnwrapResult:
    """Unwrap principal phase by adjacent frequency continuity.

    The continuity diagnostic uses the complete raw phase time,
    ``group_time + sign * phase * T_inst / (2π)``. Both group arrival and the
    convention-specific sign are mandatory: omitting either can hide or invent
    discontinuities. When ``convention`` is supplied, ``principal_phase_rad``
    must be the raw SciPy-Hilbert principal phase; the convention's
    ``phase_time_sign`` performs the Fourier-sign conversion for this
    diagnostic. Phases already converted to the paper convention must instead
    use an explicit ``phase_time_sign=+1``.
    """

    periods = _real_numeric_array(
        instantaneous_periods_s,
        "instantaneous_periods_s",
    )
    phase = _real_numeric_array(principal_phase_rad, "principal_phase_rad")
    group_times = _real_numeric_array(group_times_s, "group_times_s")
    if any(array.ndim != 1 for array in (periods, phase, group_times)):
        raise ValueError(
            "instantaneous_periods_s, principal_phase_rad, and group_times_s "
            "must be one-dimensional"
        )
    if periods.size == 0 or not (
        periods.shape == phase.shape == group_times.shape
    ):
        raise ValueError(
            "instantaneous_periods_s, principal_phase_rad, and group_times_s "
            "must be non-empty with the same shape"
        )
    if periods.size < 3:
        raise ValueError("phase unwrapping requires at least three valid points")
    if np.any(~np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError(
            "instantaneous_periods_s must contain positive finite values"
        )
    if np.unique(periods).size != periods.size:
        raise ValueError("instantaneous_periods_s values must be unique")
    if np.any(~np.isfinite(phase)):
        raise ValueError("principal_phase_rad must contain only finite values")
    if np.any(phase < -math.pi) or np.any(phase > math.pi):
        raise ValueError("principal_phase_rad must lie in the principal range (-pi, pi]")
    phase = np.array(phase, dtype=float, copy=True)
    phase[phase == -math.pi] = math.pi
    if np.any(~np.isfinite(group_times)):
        raise ValueError("group_times_s must contain only finite values")
    if convention is not None:
        selected_convention = _require_phase_convention(convention)
        convention_sign = selected_convention.definition.phase_time_sign
        if phase_time_sign is None:
            phase_time_sign = convention_sign
        elif phase_time_sign != convention_sign:
            raise ValueError(
                "phase_time_sign conflicts with the selected PhaseConvention"
            )
    elif phase_time_sign is None:
        raise TypeError(
            "phase_time_sign or convention is required"
        )
    if (
        isinstance(phase_time_sign, (bool, np.bool_))
        or not isinstance(phase_time_sign, (int, np.integer))
        or int(phase_time_sign) not in (-1, 1)
    ):
        raise ValueError("phase_time_sign must be the integer +1 or -1")
    anchor_period = _finite_scalar(
        anchor_period_s,
        "anchor_period_s",
        positive=True,
    )

    order = np.argsort(periods, kind="stable")
    sorted_periods = periods[order]
    sorted_phase = phase[order]
    sorted_group_times = group_times[order]
    anchor_distance = np.abs(sorted_periods - anchor_period)
    anchor_sorted = int(np.flatnonzero(anchor_distance == np.min(anchor_distance))[0])

    sorted_unwrapped = np.empty_like(sorted_phase)
    sorted_cycles = np.empty(sorted_phase.size, dtype=int)
    sorted_unwrapped[anchor_sorted] = sorted_phase[anchor_sorted]
    sorted_cycles[anchor_sorted] = 0
    for index in range(anchor_sorted + 1, sorted_phase.size):
        cycle = _choose_cycle_count(
            previous_unwrapped_phase_rad=sorted_unwrapped[index - 1],
            principal_phase_rad=sorted_phase[index],
        )
        sorted_cycles[index] = cycle
        sorted_unwrapped[index] = sorted_phase[index] + 2.0 * math.pi * cycle
    for index in range(anchor_sorted - 1, -1, -1):
        cycle = _choose_cycle_count(
            previous_unwrapped_phase_rad=sorted_unwrapped[index + 1],
            principal_phase_rad=sorted_phase[index],
        )
        sorted_cycles[index] = cycle
        sorted_unwrapped[index] = sorted_phase[index] + 2.0 * math.pi * cycle

    sorted_raw_time = (
        sorted_group_times
        + int(phase_time_sign)
        * sorted_unwrapped
        * sorted_periods
        / (2.0 * math.pi)
    )
    if convention is not None:
        sorted_raw_time = (
            sorted_raw_time
            + selected_convention.definition.fixed_phase_rad
            * sorted_periods
            / (2.0 * math.pi)
        )
    sorted_error, anomalies = _phase_time_prediction_errors(
        sorted_periods,
        sorted_raw_time,
    )

    internal_count = int(anomalies.size)
    anomaly_fraction = (
        float(np.count_nonzero(anomalies) / internal_count)
        if internal_count
        else 0.0
    )
    maximum_run = _maximum_true_run(anomalies)
    excessive_cycle_step = _has_excessive_cycle_step(sorted_cycles)
    discontinuous = (
        excessive_cycle_step
        or anomaly_fraction > PHASE_UNWRAP_MAX_ANOMALY_FRACTION
        or maximum_run > PHASE_UNWRAP_MAX_CONSECUTIVE_ANOMALIES
    )

    unwrapped = np.empty_like(sorted_unwrapped)
    cycles = np.empty_like(sorted_cycles)
    raw_time = np.empty_like(sorted_raw_time)
    errors = np.empty_like(sorted_error)
    unwrapped[order] = sorted_unwrapped
    cycles[order] = sorted_cycles
    raw_time[order] = sorted_raw_time
    errors[order] = sorted_error
    return PhaseUnwrapResult(
        unwrapped_phase_rad=unwrapped,
        cycle_counts=cycles,
        raw_phase_time_s=raw_time,
        prediction_error_s=errors,
        sort_order=order,
        valid_mask=np.ones(periods.size, dtype=bool),
        anomaly_fraction=anomaly_fraction,
        max_consecutive_anomalies=maximum_run,
        anchor_index=int(order[anchor_sorted]),
        status=(
            "phase_unwrap_discontinuous"
            if discontinuous
            else "accepted"
        ),
    )


def _validate_ridge_grid(
    scaled_log_energy: np.ndarray,
    velocity_axis_km_s: np.ndarray,
    *,
    blocked: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    energy = _real_numeric_array(scaled_log_energy, "scaled_log_energy")
    velocity = _real_numeric_array(
        velocity_axis_km_s,
        "velocity_axis_km_s",
    )
    if energy.ndim != 2 or 0 in energy.shape:
        raise ValueError(
            "scaled_log_energy must have non-empty (period, velocity) shape"
        )
    if velocity.ndim != 1 or velocity.size != energy.shape[1]:
        raise ValueError(
            "velocity_axis_km_s length must match the velocity dimension"
        )
    if velocity.size < 2:
        raise ValueError("velocity_axis_km_s must contain at least two points")
    if np.any(~np.isfinite(energy)):
        raise ValueError("scaled_log_energy must contain only finite values")
    if np.any(~np.isfinite(velocity)) or np.any(velocity <= 0):
        raise ValueError("velocity_axis_km_s must contain positive finite values")
    differences = np.diff(velocity)
    if (
        np.any(differences <= 0)
        or not np.allclose(
            differences,
            differences[0],
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise ValueError("velocity_axis_km_s must be a uniform increasing grid")
    if blocked is None:
        blocked_array = np.zeros_like(energy, dtype=bool)
    else:
        blocked_array = np.asarray(blocked, dtype=bool)
        if blocked_array.shape != energy.shape:
            raise ValueError("blocked must match scaled_log_energy shape")
    return energy, velocity, blocked_array, float(differences[0])


def _validate_nonnegative_penalty(value: float, name: str) -> float:
    penalty = _finite_scalar(value, name)
    if penalty < 0:
        raise ValueError(f"{name} must be a non-negative finite scalar")
    return penalty


def _l1_minplus_columns(
    values: np.ndarray,
    *,
    step_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact L1 min-plus transform down the first axis of ``values``.

    The second axis is independent. The returned predecessor indices choose
    the smallest index on exact ties, which keeps ridge tracing deterministic.
    """

    count = values.shape[0]
    forward = np.array(values, dtype=float, copy=True)
    forward_arg = np.broadcast_to(
        np.arange(count, dtype=np.int16)[:, np.newaxis],
        values.shape,
    ).copy()
    for row in range(1, count):
        candidate = forward[row - 1] + step_penalty
        candidate_arg = forward_arg[row - 1]
        improve = candidate <= forward[row]
        forward[row, improve] = candidate[improve]
        forward_arg[row, improve] = candidate_arg[improve]

    backward = np.array(values, dtype=float, copy=True)
    backward_arg = np.broadcast_to(
        np.arange(count, dtype=np.int16)[:, np.newaxis],
        values.shape,
    ).copy()
    for row in range(count - 2, -1, -1):
        candidate = backward[row + 1] + step_penalty
        improve = candidate < backward[row]
        backward[row, improve] = candidate[improve]
        backward_arg[row, improve] = backward_arg[row + 1, improve]

    improve = backward < forward
    transformed = np.where(improve, backward, forward)
    predecessor = np.where(improve, backward_arg, forward_arg).astype(
        np.int16,
        copy=False,
    )
    return transformed, predecessor


def _ridge_cost_tolerance(
    *,
    energy: np.ndarray,
    velocity_count: int,
    dv: float,
    beta1: float,
    beta2: float,
) -> float:
    """Return one machine-scale tolerance bounding this complete DP problem."""

    period_count = energy.shape[0]
    max_index_step = velocity_count - 1
    energy_bound = period_count * float(np.max(np.abs(energy)))
    first_order_bound = (
        beta1 * dv * max(period_count - 1, 0) * max_index_step
    )
    second_order_bound = (
        beta2
        * dv
        * max(period_count - 2, 0)
        * 2
        * max_index_step
    )
    cost_scale = max(
        1.0,
        energy_bound + first_order_bound + second_order_bound,
    )
    return RIDGE_COST_TIE_ULPS * np.finfo(float).eps * cost_scale


def _trace_optimal_ridge(
    scaled_log_energy: np.ndarray,
    velocity_axis_km_s: np.ndarray,
    *,
    beta1: float,
    beta2: float,
    blocked: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """Trace the exact full-grid minimum-cost second-order FTAN ridge.

    This searches every velocity-grid point. Its recurrence uses an exact
    forward/backward L1 min-plus distance transform, reducing the second-order
    transition from O(n_velocity**3) to O(n_velocity**2) per period without
    changing the objective or search space.
    """

    energy, velocity, blocked_array, dv = _validate_ridge_grid(
        scaled_log_energy,
        velocity_axis_km_s,
        blocked=blocked,
    )
    beta1 = _validate_nonnegative_penalty(beta1, "beta1")
    beta2 = _validate_nonnegative_penalty(beta2, "beta2")
    period_count, velocity_count = energy.shape
    tie_tolerance = _ridge_cost_tolerance(
        energy=energy,
        velocity_count=velocity_count,
        dv=dv,
        beta1=beta1,
        beta2=beta2,
    )
    unary = -energy.copy()
    unary[blocked_array] = np.inf

    # Every transition below is a strict minimization. The roundoff allowance
    # is consumed exactly once, when choosing among complete terminal paths.
    if period_count == 1:
        terminal_cost = float(np.min(unary[0]))
        if not np.isfinite(terminal_cost):
            return None, float("inf")
        terminal = int(
            np.flatnonzero(unary[0] <= terminal_cost + tie_tolerance)[0]
        )
        rows = np.asarray([terminal], dtype=int)
    else:
        row_index = np.arange(velocity_count, dtype=int)
        first_rows = row_index[:, np.newaxis]
        second_rows = row_index[np.newaxis, :]
        first_order_penalty = beta1 * dv * np.abs(
            second_rows - first_rows
        )
        previous = (
            unary[0, :, np.newaxis]
            + unary[1, np.newaxis, :]
            + first_order_penalty
        )
        backpointers = []
        second_order_step = beta2 * dv
        query_index = (
            2 * row_index[:, np.newaxis] - row_index[np.newaxis, :]
        )
        clipped_query = np.clip(query_index, 0, velocity_count - 1)
        previous_row_grid = np.broadcast_to(
            row_index[:, np.newaxis],
            query_index.shape,
        )
        outside_steps = np.where(
            query_index < 0,
            -query_index,
            np.where(
                query_index >= velocity_count,
                query_index - (velocity_count - 1),
                0,
            ),
        )

        for period_index in range(2, period_count):
            transformed, predecessor = _l1_minplus_columns(
                previous,
                step_penalty=second_order_step,
            )
            transition = transformed[clipped_query, previous_row_grid]
            predecessor_query = predecessor[
                clipped_query,
                previous_row_grid,
            ]
            if second_order_step:
                transition = transition + outside_steps * second_order_step
            current = (
                transition
                + unary[period_index, np.newaxis, :]
                + first_order_penalty
            )
            backpointers.append(
                np.asarray(predecessor_query, dtype=np.int16)
            )
            previous = current

        terminal_cost = float(np.min(previous))
        if not np.isfinite(terminal_cost):
            return None, float("inf")
        terminal_flat = int(
            np.flatnonzero(
                previous.ravel() <= terminal_cost + tie_tolerance
            )[0]
        )
        penultimate, terminal = np.unravel_index(
            terminal_flat,
            previous.shape,
        )
        rows = np.empty(period_count, dtype=int)
        rows[-2] = int(penultimate)
        rows[-1] = int(terminal)
        for period_index in range(period_count - 1, 1, -1):
            rows[period_index - 2] = int(
                backpointers[period_index - 2][
                    rows[period_index - 1],
                    rows[period_index],
                ]
            )

    total_cost = -float(
        np.sum(energy[np.arange(period_count, dtype=int), rows])
    )
    if period_count > 1:
        total_cost += beta1 * dv * float(
            np.sum(np.abs(np.diff(rows)))
        )
    if period_count > 2:
        total_cost += beta2 * dv * float(
            np.sum(np.abs(np.diff(rows, n=2)))
        )
    return rows, total_cost


def _longest_false_run(valid: np.ndarray) -> int:
    longest = 0
    current = 0
    for item in valid:
        if item:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def _wang_group_velocity_limits(periods_s: np.ndarray) -> np.ndarray:
    periods = np.asarray(periods_s, dtype=float)
    return np.where(periods < 4.5, 3.0, 3.3)


def _ridge_quality(
    *,
    rows: np.ndarray,
    velocities_km_s: np.ndarray,
    valid: np.ndarray,
    scaled_log_energy: np.ndarray,
    periods_s: np.ndarray,
    velocity_count: int,
) -> RidgeQuality:
    valid_count = int(np.count_nonzero(valid))
    coverage = float(valid_count / valid.size)
    max_gap = _longest_false_run(valid)

    adjacent_valid = valid[:-1] & valid[1:]
    adjacent_count = int(np.count_nonzero(adjacent_valid))
    if adjacent_count:
        jump_fraction = float(
            np.count_nonzero(
                np.abs(np.diff(velocities_km_s))[adjacent_valid] > 0.20
            )
            / adjacent_count
        )
    else:
        jump_fraction = 0.0

    boundary_fraction = float(
        np.count_nonzero((rows == 0) | (rows == velocity_count - 1))
        / rows.size
    )

    selected_energy = np.zeros(rows.size, dtype=float)
    selected_energy[valid] = scaled_log_energy[
        np.arange(rows.size, dtype=int)[valid],
        rows[valid],
    ]
    normalized_energy_integral = float(np.sum(selected_energy))

    reason = "accepted"
    if coverage < 0.80:
        reason = "insufficient_coverage"
    elif max_gap > 2:
        reason = "ridge_discontinuous"
    elif jump_fraction > 0.05:
        reason = "ridge_jump"
    elif boundary_fraction > 0.10:
        reason = "ridge_boundary"
    return RidgeQuality(
        accepted=reason == "accepted",
        reason=reason,
        coverage=coverage,
        max_gap=max_gap,
        jump_fraction=jump_fraction,
        boundary_fraction=boundary_fraction,
        normalized_energy_integral=normalized_energy_integral,
    )


def find_candidate_ridges(
    *,
    scaled_log_energy: np.ndarray,
    normalized_envelope_amplitude: np.ndarray,
    periods_s: np.ndarray,
    velocity_axis_km_s: np.ndarray,
    beta1: float,
    beta2: float,
    max_candidates: int = 3,
) -> List[RidgeResult]:
    """Return up to three deterministic, physically distinct FTAN ridges."""

    energy, velocity, _, _ = _validate_ridge_grid(
        scaled_log_energy,
        velocity_axis_km_s,
    )
    amplitude = _real_numeric_array(
        normalized_envelope_amplitude,
        "normalized_envelope_amplitude",
    )
    periods = _real_numeric_array(periods_s, "periods_s")
    if amplitude.shape != energy.shape:
        raise ValueError(
            "normalized_envelope_amplitude must match scaled_log_energy shape"
        )
    if np.any(~np.isfinite(amplitude)) or np.any(amplitude < 0):
        raise ValueError(
            "normalized_envelope_amplitude must contain finite non-negative values"
        )
    if periods.ndim != 1 or periods.size != energy.shape[0]:
        raise ValueError("periods_s length must match the period dimension")
    if (
        np.any(~np.isfinite(periods))
        or np.any(periods <= 0)
        or np.any(np.diff(periods) <= 0)
    ):
        raise ValueError("periods_s must be positive, finite, and increasing")
    try:
        candidate_count = _integer_scalar(max_candidates, "max_candidates")
    except ValueError:
        raise ValueError("max_candidates must be an integer from 1 through 3") from None
    if not 1 <= candidate_count <= 3:
        raise ValueError("max_candidates must be an integer from 1 through 3")
    beta1 = _validate_nonnegative_penalty(beta1, "beta1")
    beta2 = _validate_nonnegative_penalty(beta2, "beta2")

    blocked = np.zeros_like(energy, dtype=bool)
    candidates = []  # type: List[RidgeResult]
    period_index = np.arange(periods.size, dtype=int)
    limits = _wang_group_velocity_limits(periods)
    corridor_tolerance = 1e-12

    for _ in range(candidate_count):
        rows, _ = _trace_optimal_ridge(
            energy,
            velocity,
            beta1=beta1,
            beta2=beta2,
            blocked=blocked,
        )
        if rows is None:
            break
        velocities = velocity[rows]
        valid = amplitude[period_index, rows] >= 0.15
        quality = _ridge_quality(
            rows=rows,
            velocities_km_s=velocities,
            valid=valid,
            scaled_log_energy=energy,
            periods_s=periods,
            velocity_count=velocity.size,
        )
        result = RidgeResult(
            row_indices=rows,
            group_velocities_km_s=velocities,
            valid=valid,
            wang_group_limit_pass_count=int(
                np.count_nonzero(valid & (velocities <= limits))
            ),
            quality=quality,
        )

        duplicate = False
        for previous in candidates:
            common_valid = previous.valid & result.valid
            common_count = int(np.count_nonzero(common_valid))
            if common_count == 0:
                duplicate = True
                break
            distinct_fraction = float(
                np.count_nonzero(
                    np.abs(
                        previous.group_velocities_km_s[common_valid]
                        - result.group_velocities_km_s[common_valid]
                    )
                    >= 0.05 - corridor_tolerance
                )
                / common_count
            )
            if distinct_fraction < 0.50:
                duplicate = True
                break
        if duplicate:
            break
        candidates.append(result)

        for selected in candidates:
            blocked |= (
                np.abs(
                    velocity[np.newaxis, :]
                    - selected.group_velocities_km_s[:, np.newaxis]
                )
                <= 0.05 + corridor_tolerance
            )

    return candidates


def _ridge_roughness(result: RidgeResult) -> Tuple[float, float]:
    adjacent_valid = result.valid[:-1] & result.valid[1:]
    if np.any(adjacent_valid):
        first_order = float(
            np.mean(
                np.abs(np.diff(result.group_velocities_km_s))[adjacent_valid]
            )
        )
    else:
        first_order = 0.0
    triple_valid = result.valid[:-2] & result.valid[1:-1] & result.valid[2:]
    if np.any(triple_valid):
        second_order = float(
            np.mean(
                np.abs(np.diff(result.group_velocities_km_s, n=2))[
                    triple_valid
                ]
            )
        )
    else:
        second_order = 0.0
    return first_order, second_order


def _no_fundamental_ridge() -> RidgeResult:
    return RidgeResult(
        row_indices=np.asarray([], dtype=int),
        group_velocities_km_s=np.asarray([], dtype=float),
        valid=np.asarray([], dtype=bool),
        wang_group_limit_pass_count=0,
        quality=RidgeQuality(
            accepted=False,
            reason="no_fundamental_ridge",
            coverage=0.0,
            max_gap=0,
            jump_fraction=0.0,
            boundary_fraction=0.0,
            normalized_energy_integral=0.0,
        ),
    )


def select_fundamental_ridge(
    candidates: Iterable[RidgeResult],
    *,
    periods_s: np.ndarray,
) -> RidgeResult:
    """Select the fundamental ridge using Wang-limit coverage before energy."""

    candidate_list = list(candidates)
    periods = _real_numeric_array(periods_s, "periods_s")
    if periods.ndim != 1 or np.any(~np.isfinite(periods)):
        raise ValueError("periods_s must be a finite one-dimensional array")
    if not candidate_list:
        return _no_fundamental_ridge()
    for result in candidate_list:
        if result.row_indices.size != periods.size:
            raise ValueError("every candidate length must match periods_s")

    accepted = [row for row in candidate_list if row.quality.accepted]
    if not accepted:
        return _no_fundamental_ridge()
    ranked = accepted
    limits = _wang_group_velocity_limits(periods)

    def ranking_key(result: RidgeResult):
        wang_pass_count = int(
            np.count_nonzero(
                result.valid & (result.group_velocities_km_s <= limits)
            )
        )
        first_roughness, second_roughness = _ridge_roughness(result)
        return (
            wang_pass_count,
            result.quality.coverage,
            result.quality.normalized_energy_integral,
            -first_roughness,
            -second_roughness,
        )

    return max(ranked, key=ranking_key)


def compute_wang_snr(
    *,
    time_s: np.ndarray,
    filtered_waveform: np.ndarray,
    distance_km: float,
    period_s: float,
    min_noise_samples: int = 8,
) -> WangSnrResult:
    """Compute Wang leading/trailing waveform-RMS SNR for one observation."""

    time = _real_numeric_array(time_s, "time_s")
    waveform = _real_numeric_array(filtered_waveform, "filtered_waveform")
    if time.ndim != 1 or waveform.ndim != 1:
        raise ValueError("time_s and filtered_waveform must be one-dimensional")
    if time.shape != waveform.shape or time.size < 2:
        raise ValueError(
            "time_s and filtered_waveform must have the same length of at least two"
        )
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(waveform)):
        raise ValueError(
            "time_s and filtered_waveform must contain only finite values"
        )
    steps = np.diff(time)
    if np.any(steps <= 0):
        raise ValueError("time_s must be strictly increasing and uniform")
    dt_s = float(steps[0])
    spacing_tolerance = (
        64.0 * np.finfo(float).eps * max(1.0, abs(dt_s))
    )
    if not np.allclose(
        steps,
        dt_s,
        rtol=1e-10,
        atol=spacing_tolerance,
    ):
        raise ValueError("time_s must be strictly increasing and uniform")
    distance = _finite_scalar(distance_km, "distance_km", positive=True)
    period = _finite_scalar(period_s, "period_s", positive=True)
    minimum_samples = _integer_scalar(min_noise_samples, "min_noise_samples")
    if minimum_samples <= 0:
        raise ValueError("min_noise_samples must be a positive integer")

    signal_start = distance / 5.0
    signal_end = distance / 1.6
    leading_start = dt_s
    leading_end = signal_start - 0.5 * period
    trailing_start = signal_end + 0.5 * period
    trailing_end = float(time[-1])
    signal_mask = (time >= signal_start) & (time <= signal_end)
    leading_mask = (time >= leading_start) & (time <= leading_end)
    trailing_mask = (time >= trailing_start) & (time <= trailing_end)
    signal_count = int(np.count_nonzero(signal_mask))
    leading_count = int(np.count_nonzero(leading_mask))
    trailing_count = int(np.count_nonzero(trailing_mask))

    signal_peak = (
        float(np.max(np.abs(waveform[signal_mask])))
        if signal_count
        else float("nan")
    )

    def noise_values(mask, count, side):
        if count < minimum_samples:
            return (
                float("nan"),
                float("nan"),
                f"insufficient_{side}_noise",
            )
        rms_value = float(np.sqrt(np.mean(np.square(waveform[mask]))))
        if rms_value <= 0:
            return float(rms_value), float("nan"), f"zero_{side}_noise_rms"
        if signal_count == 0:
            return rms_value, float("nan"), "signal_window_empty"
        return rms_value, float(signal_peak / rms_value), "accepted"

    leading_rms, leading_snr, leading_status = noise_values(
        leading_mask,
        leading_count,
        "leading",
    )
    trailing_rms, trailing_snr, trailing_status = noise_values(
        trailing_mask,
        trailing_count,
        "trailing",
    )
    if signal_count == 0:
        status = "signal_window_empty"
    elif leading_status != "accepted" and trailing_status != "accepted":
        status = f"{leading_status};{trailing_status}"
    elif leading_status != "accepted":
        status = leading_status
    elif trailing_status != "accepted":
        status = trailing_status
    else:
        status = "accepted"
    return WangSnrResult(
        signal_peak=signal_peak,
        leading_noise_rms=leading_rms,
        trailing_noise_rms=trailing_rms,
        leading_snr=leading_snr,
        trailing_snr=trailing_snr,
        signal_sample_count=signal_count,
        leading_noise_sample_count=leading_count,
        trailing_noise_sample_count=trailing_count,
        signal_window_start_s=float(signal_start),
        signal_window_end_s=float(signal_end),
        leading_noise_start_s=float(leading_start),
        leading_noise_end_s=float(leading_end),
        trailing_noise_start_s=float(trailing_start),
        trailing_noise_end_s=float(trailing_end),
        leading_status=leading_status,
        trailing_status=trailing_status,
        status=status,
    )


def evaluate_wang_left_qc(
    *,
    period_s: float,
    group_velocity_km_s: float,
    snr: WangSnrResult,
    ftan_valid: bool = True,
    ridge_valid: bool = True,
    group_arrival_valid: bool = True,
    phase_valid: bool = True,
    instantaneous_frequency_valid: bool = True,
) -> WangLeftQcResult:
    """Apply only the Wang Figure-4 left-column observation checks."""

    period = _finite_scalar(period_s, "period_s", positive=True)
    group_velocity = _finite_scalar(
        group_velocity_km_s,
        "group_velocity_km_s",
        positive=True,
    )
    if not isinstance(snr, WangSnrResult):
        raise ValueError("snr must be a WangSnrResult")
    validity = (
        ("ftan_valid", ftan_valid),
        ("ridge_valid", ridge_valid),
        ("group_arrival_valid", group_arrival_valid),
        ("phase_valid", phase_valid),
        ("instantaneous_frequency_valid", instantaneous_frequency_valid),
    )
    for name, value in validity:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    maximum_velocity = 3.0 if period < 4.5 else 3.3
    status = "accepted"
    for name, value in validity:
        if not bool(value):
            status = f"{name}_failed"
            break
    else:
        if (
            snr.status != "accepted"
            or not np.isfinite(snr.leading_snr)
            or not np.isfinite(snr.trailing_snr)
            or snr.leading_snr <= 4.0
            or snr.trailing_snr <= 4.0
        ):
            status = "snr_threshold_failed"
        elif not 1.6 <= group_velocity <= maximum_velocity:
            status = "group_velocity_out_of_range"
    return WangLeftQcResult(
        accepted=status == "accepted",
        status=status,
        group_velocity_min_km_s=1.6,
        group_velocity_max_km_s=maximum_velocity,
        snr_threshold=4.0,
    )


def resample_wang_target_period(
    *,
    continuous_periods_s: np.ndarray,
    anchored_raw_phase_time_s: np.ndarray,
    group_time_s: np.ndarray,
    signal_peak: np.ndarray,
    leading_noise_rms: np.ndarray,
    trailing_noise_rms: np.ndarray,
    ridge_normalized_log_energy: np.ndarray,
    ridge_normalized_envelope_amplitude: np.ndarray,
    ridge_adjacent_jump_km_s: np.ndarray,
    valid_mask: np.ndarray,
    distance_km: float,
    target_period_s: float,
) -> WangTargetPeriodResult:
    """Interpolate valid continuous observations to one exact Wang target."""

    periods = _real_numeric_array(
        continuous_periods_s,
        "continuous_periods_s",
    )
    if (
        periods.ndim != 1
        or periods.size < 2
        or np.any(~np.isfinite(periods))
        or np.any(periods <= 0)
        or np.any(np.diff(periods) <= 0)
    ):
        raise ValueError(
            "continuous_periods_s must contain increasing positive finite values"
        )
    values = {}
    for name, raw in (
        ("anchored_raw_phase_time_s", anchored_raw_phase_time_s),
        ("group_time_s", group_time_s),
        ("signal_peak", signal_peak),
        ("leading_noise_rms", leading_noise_rms),
        ("trailing_noise_rms", trailing_noise_rms),
        ("ridge_normalized_log_energy", ridge_normalized_log_energy),
        (
            "ridge_normalized_envelope_amplitude",
            ridge_normalized_envelope_amplitude,
        ),
        ("ridge_adjacent_jump_km_s", ridge_adjacent_jump_km_s),
    ):
        array = _real_numeric_array(raw, name)
        if (
            array.ndim != 1
            or array.shape != periods.shape
            or np.any(~np.isfinite(array))
        ):
            raise ValueError(
                f"{name} must be a finite one-dimensional array matching periods"
            )
        if name in (
            "ridge_normalized_log_energy",
            "ridge_normalized_envelope_amplitude",
        ) and (np.any(array < 0) or np.any(array > 1)):
            raise ValueError(f"{name} must lie in [0, 1]")
        if name == "ridge_adjacent_jump_km_s" and np.any(array < 0):
            raise ValueError(
                "ridge_adjacent_jump_km_s must be non-negative"
            )
        values[name] = array
    valid = np.asarray(valid_mask)
    if valid.dtype.kind != "b" or valid.ndim != 1 or valid.shape != periods.shape:
        raise ValueError("valid_mask must be a boolean array matching periods")
    distance = _finite_scalar(distance_km, "distance_km", positive=True)
    target = _finite_scalar(target_period_s, "target_period_s", positive=True)
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(target))
    local = valid & (np.abs(periods - target) <= 0.10 + tolerance)
    local_indices = np.flatnonzero(local)
    lower = local_indices[periods[local_indices] < target - tolerance]
    upper = local_indices[periods[local_indices] > target + tolerance]
    bracketed = bool(lower.size and upper.size)
    if bracketed:
        lower_period = float(periods[lower[-1]])
        upper_period = float(periods[upper[0]])
        bracketed = (
            target - lower_period <= 0.10 + tolerance
            and upper_period - target <= 0.10 + tolerance
            and upper_period - lower_period <= 0.20 + tolerance
        )
    support_periods = periods[local_indices]

    def failure() -> WangTargetPeriodResult:
        nan = float("nan")
        return WangTargetPeriodResult(
            target_period_s=target,
            anchored_raw_phase_time_s=nan,
            group_time_s=nan,
            group_velocity_km_s=nan,
            signal_peak=nan,
            leading_noise_rms=nan,
            trailing_noise_rms=nan,
            leading_snr=nan,
            trailing_snr=nan,
            ridge_normalized_log_energy=nan,
            ridge_normalized_envelope_amplitude=nan,
            ridge_adjacent_jump_km_s=nan,
            support_periods_s=support_periods,
            support_count=int(support_periods.size),
            interpolation_method="none",
            accepted=False,
            status="target_period_not_bracketed",
        )

    if not bracketed or support_periods.size < 2:
        return failure()
    method = "linear" if support_periods.size == 2 else "pchip"

    def interpolate(name):
        support_values = values[name][local_indices]
        if method == "linear":
            return float(
                np.interp(target, support_periods, support_values)
            )
        return float(PchipInterpolator(support_periods, support_values)(target))

    target_values = {name: interpolate(name) for name in values}
    group_time = target_values["group_time_s"]
    signal = target_values["signal_peak"]
    leading_rms = target_values["leading_noise_rms"]
    trailing_rms = target_values["trailing_noise_rms"]
    if (
        group_time <= 0
        or signal < 0
        or leading_rms <= 0
        or trailing_rms <= 0
    ):
        return failure()
    leading_snr = signal / leading_rms
    trailing_snr = signal / trailing_rms
    component_snr = WangSnrResult(
        signal_peak=signal,
        leading_noise_rms=leading_rms,
        trailing_noise_rms=trailing_rms,
        leading_snr=leading_snr,
        trailing_snr=trailing_snr,
        signal_sample_count=0,
        leading_noise_sample_count=0,
        trailing_noise_sample_count=0,
        signal_window_start_s=float("nan"),
        signal_window_end_s=float("nan"),
        leading_noise_start_s=float("nan"),
        leading_noise_end_s=float("nan"),
        trailing_noise_start_s=float("nan"),
        trailing_noise_end_s=float("nan"),
        leading_status="accepted",
        trailing_status="accepted",
        status="accepted",
    )
    group_velocity = distance / group_time
    qc = evaluate_wang_left_qc(
        period_s=target,
        group_velocity_km_s=group_velocity,
        snr=component_snr,
    )
    return WangTargetPeriodResult(
        target_period_s=target,
        anchored_raw_phase_time_s=target_values[
            "anchored_raw_phase_time_s"
        ],
        group_time_s=group_time,
        group_velocity_km_s=float(group_velocity),
        signal_peak=signal,
        leading_noise_rms=leading_rms,
        trailing_noise_rms=trailing_rms,
        leading_snr=float(leading_snr),
        trailing_snr=float(trailing_snr),
        ridge_normalized_log_energy=target_values[
            "ridge_normalized_log_energy"
        ],
        ridge_normalized_envelope_amplitude=target_values[
            "ridge_normalized_envelope_amplitude"
        ],
        ridge_adjacent_jump_km_s=target_values[
            "ridge_adjacent_jump_km_s"
        ],
        support_periods_s=support_periods,
        support_count=int(support_periods.size),
        interpolation_method=method,
        accepted=qc.accepted,
        status=qc.status,
    )


def resample_wang_target_periods(
    *,
    target_periods_s: Iterable[float],
    **continuous_fields,
) -> Tuple[WangTargetPeriodResult, ...]:
    """Resample targets independently so one failed period cannot drop another."""

    return tuple(
        resample_wang_target_period(
            target_period_s=target,
            **continuous_fields,
        )
        for target in target_periods_s
    )


def _exact_duplicate_mask(
    values: np.ndarray,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    """Mark every eligible row belonging to an exact duplicate group."""

    array = np.asarray(values, dtype=float)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if array.ndim != 1 or eligible.ndim != 1 or array.shape != eligible.shape:
        raise ValueError("duplicate detection arrays must be aligned")
    groups = {}
    for index in np.flatnonzero(eligible):
        groups.setdefault(float(array[index]), []).append(int(index))
    duplicate = np.zeros(array.size, dtype=bool)
    for indices in groups.values():
        if len(indices) > 1:
            duplicate[np.asarray(indices, dtype=int)] = True
    return duplicate


def resample_wang_measurements(
    measurements: Iterable[Optional[PeriodMeasurement]],
    *,
    time_s: np.ndarray,
    distance_km: float,
    target_periods_s: Iterable[float],
    nominal_periods_s: np.ndarray,
    measurement_statuses: Iterable[str],
    instantaneous_periods_s: np.ndarray,
    ridge_normalized_log_energy: np.ndarray,
    ridge_normalized_envelope_amplitude: np.ndarray,
    ridge_adjacent_jump_km_s: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[WangTargetPeriodResult, ...]:
    """Resample anchored continuous rows using T_inst for observation SNR."""

    rows = tuple(measurements)
    if any(
        row is not None and not isinstance(row, PeriodMeasurement)
        for row in rows
    ):
        raise ValueError(
            "measurements must contain only PeriodMeasurement rows or None"
        )
    valid = np.asarray(valid_mask)
    if valid.dtype.kind != "b" or valid.ndim != 1 or valid.size != len(rows):
        raise ValueError("valid_mask must be a boolean array matching measurements")
    nominal = _real_numeric_array(nominal_periods_s, "nominal_periods_s")
    if (
        nominal.ndim != 1
        or nominal.size != len(rows)
        or np.any(~np.isfinite(nominal))
        or np.any(nominal <= 0)
    ):
        raise ValueError(
            "nominal_periods_s must contain one positive finite value per row"
        )
    statuses = tuple(measurement_statuses)
    if (
        len(statuses) != len(rows)
        or any(not isinstance(value, str) or not value for value in statuses)
    ):
        raise ValueError(
            "measurement_statuses must contain one non-empty string per row"
        )
    instantaneous = _real_numeric_array(
        instantaneous_periods_s,
        "instantaneous_periods_s",
    )
    if (
        instantaneous.ndim != 1
        or instantaneous.shape != nominal.shape
        or np.any(np.isinf(instantaneous))
        or np.any(np.isfinite(instantaneous) & (instantaneous <= 0))
    ):
        raise ValueError(
            "instantaneous_periods_s must match rows and contain positive "
            "finite values or NaN"
        )
    for index, (row, row_valid, row_status) in enumerate(
        zip(rows, valid, statuses)
    ):
        if bool(row_valid) != (row is not None):
            raise ValueError(
                "valid_mask must be true exactly where measurements exist"
            )
        if bool(row_valid) != (row_status == "accepted"):
            raise ValueError(
                "measurement status must be accepted exactly for valid rows"
            )
        if row is not None and not math.isclose(
            row.period_s,
            float(nominal[index]),
            rel_tol=0.0,
            abs_tol=64.0
            * np.finfo(float).eps
            * max(1.0, abs(float(nominal[index]))),
        ):
            raise ValueError(
                "nominal_periods_s must match measurement periods"
            )
        if row is not None:
            expected_instantaneous = 2.0 * math.pi / row.omega_inst_rad_s
            if (
                not np.isfinite(instantaneous[index])
                or not math.isclose(
                    expected_instantaneous,
                    float(instantaneous[index]),
                    rel_tol=0.0,
                    abs_tol=64.0
                    * np.finfo(float).eps
                    * max(1.0, abs(expected_instantaneous)),
                )
            ):
                raise ValueError(
                    "valid measurement instantaneous period must match "
                    "omega_inst_rad_s"
                )
        elif (
            np.isfinite(instantaneous[index])
            != (
                row_status
                in {
                    "duplicate_instantaneous_period",
                    "phase_unwrap_discontinuous",
                    "outside_anchored_phase_segment",
                }
            )
        ):
            raise ValueError(
                "invalid-row instantaneous period must be finite exactly "
                "after successful instantaneous-frequency estimation"
            )
    diagnostics = {}
    for name, raw in (
        ("ridge_normalized_log_energy", ridge_normalized_log_energy),
        (
            "ridge_normalized_envelope_amplitude",
            ridge_normalized_envelope_amplitude,
        ),
        ("ridge_adjacent_jump_km_s", ridge_adjacent_jump_km_s),
    ):
        array = _real_numeric_array(raw, name)
        if (
            array.ndim != 1
            or array.shape != nominal.shape
            or np.any(~np.isfinite(array))
            or np.any(array < 0)
            or (
                name != "ridge_adjacent_jump_km_s"
                and np.any(array > 1)
            )
        ):
            raise ValueError(
                f"{name} must be finite, match rows, and lie in "
                + (
                    "[0, infinity)"
                    if name == "ridge_adjacent_jump_km_s"
                    else "[0, 1]"
                )
            )
        diagnostics[name] = array
    distance = _finite_scalar(distance_km, "distance_km", positive=True)
    targets = tuple(
        _finite_scalar(target, "target_period_s", positive=True)
        for target in target_periods_s
    )
    accepted_rows = []
    accepted_indices = []
    accepted_periods = []
    accepted_snr = []
    rejected_nominal_periods = []
    rejected_instantaneous_periods = []
    rejection_statuses = []
    duplicate = _exact_duplicate_mask(instantaneous, valid)
    for index, (row, row_valid, row_status) in enumerate(
        zip(rows, valid, statuses)
    ):
        if not row_valid:
            rejected_nominal_periods.append(float(nominal[index]))
            rejected_instantaneous_periods.append(
                float(instantaneous[index])
            )
            rejection_statuses.append(row_status)
            continue
        instantaneous_period = float(instantaneous[index])
        if duplicate[index]:
            rejected_nominal_periods.append(float(nominal[index]))
            rejected_instantaneous_periods.append(instantaneous_period)
            rejection_statuses.append("duplicate_instantaneous_period")
            continue
        snr = compute_wang_snr(
            time_s=time_s,
            filtered_waveform=row.filtered_waveform,
            distance_km=distance,
            period_s=instantaneous_period,
        )
        continuous_qc = evaluate_wang_left_qc(
            period_s=instantaneous_period,
            group_velocity_km_s=distance / row.group_time_s,
            snr=snr,
        )
        if not continuous_qc.accepted:
            rejected_nominal_periods.append(float(nominal[index]))
            rejected_instantaneous_periods.append(instantaneous_period)
            rejection_statuses.append(continuous_qc.status)
            continue
        accepted_rows.append(row)
        accepted_indices.append(index)
        accepted_periods.append(instantaneous_period)
        accepted_snr.append(snr)
    order = np.argsort(np.asarray(accepted_periods, dtype=float))
    if len(accepted_rows) < 2:
        nan = float("nan")
        return tuple(
            WangTargetPeriodResult(
                target_period_s=target,
                anchored_raw_phase_time_s=nan,
                group_time_s=nan,
                group_velocity_km_s=nan,
                signal_peak=nan,
                leading_noise_rms=nan,
                trailing_noise_rms=nan,
                leading_snr=nan,
                trailing_snr=nan,
                ridge_normalized_log_energy=nan,
                ridge_normalized_envelope_amplitude=nan,
                ridge_adjacent_jump_km_s=nan,
                support_periods_s=np.empty(0, dtype=float),
                support_count=0,
                interpolation_method="none",
                accepted=False,
                status="target_period_not_bracketed",
                rejected_continuous_nominal_periods_s=np.asarray(
                    rejected_nominal_periods,
                    dtype=float,
                ),
                rejected_continuous_instantaneous_periods_s=np.asarray(
                    rejected_instantaneous_periods,
                    dtype=float,
                ),
                continuous_rejection_statuses=tuple(rejection_statuses),
            )
            for target in targets
        )
    sorted_rows = [accepted_rows[index] for index in order]
    sorted_snr = [accepted_snr[index] for index in order]
    sorted_indices = np.asarray(accepted_indices, dtype=int)[order]
    periods = np.asarray(accepted_periods, dtype=float)[order]
    target_rows = resample_wang_target_periods(
        continuous_periods_s=periods,
        anchored_raw_phase_time_s=np.asarray(
            [row.raw_phase_time_s for row in sorted_rows],
            dtype=float,
        ),
        group_time_s=np.asarray(
            [row.group_time_s for row in sorted_rows],
            dtype=float,
        ),
        signal_peak=np.asarray(
            [snr.signal_peak for snr in sorted_snr],
            dtype=float,
        ),
        leading_noise_rms=np.asarray(
            [snr.leading_noise_rms for snr in sorted_snr],
            dtype=float,
        ),
        trailing_noise_rms=np.asarray(
            [snr.trailing_noise_rms for snr in sorted_snr],
            dtype=float,
        ),
        ridge_normalized_log_energy=diagnostics[
            "ridge_normalized_log_energy"
        ][sorted_indices],
        ridge_normalized_envelope_amplitude=diagnostics[
            "ridge_normalized_envelope_amplitude"
        ][sorted_indices],
        ridge_adjacent_jump_km_s=diagnostics[
            "ridge_adjacent_jump_km_s"
        ][sorted_indices],
        valid_mask=np.ones(periods.size, dtype=bool),
        distance_km=distance,
        target_periods_s=targets,
    )
    rejected_nominal = np.asarray(rejected_nominal_periods, dtype=float)
    rejected_instantaneous = np.asarray(
        rejected_instantaneous_periods,
        dtype=float,
    )
    rejection_status_tuple = tuple(rejection_statuses)
    return tuple(
        replace(
            row,
            rejected_continuous_nominal_periods_s=rejected_nominal,
            rejected_continuous_instantaneous_periods_s=(
                rejected_instantaneous
            ),
            continuous_rejection_statuses=rejection_status_tuple,
        )
        for row in target_rows
    )


def _phase_measurement_snr(
    filtered_waveform: np.ndarray,
    time_s: np.ndarray,
    *,
    distance_km: float,
    period_s: float,
) -> float:
    """Compatibility scalar uniquely delegated to exact Wang waveform SNR."""

    result = compute_wang_snr(
        time_s=time_s,
        filtered_waveform=filtered_waveform,
        distance_km=distance_km,
        period_s=period_s,
    )
    if result.status != "accepted":
        return float("nan")
    return float(min(result.leading_snr, result.trailing_snr))


def measure_phase_curve(
    trace: DatTrace,
    *,
    periods_s: np.ndarray,
    velocity_axis_km_s: np.ndarray,
    alpha=None,  # type: Optional[float]
    beta1: float = 0.5,
    beta2: float = 1.0,
    snr_gap_s: float = 0.5,
    convention: PhaseConvention,
    waveform_is_prepared: bool = False,
    stage_callback=None,
) -> Optional[PhaseCurveMeasurement]:
    """Measure one convention-aware FTAN phase curve through the formal kernels."""

    if stage_callback is not None and not callable(stage_callback):
        raise ValueError("stage_callback must be callable or None")

    def mark_stage(stage: str) -> None:
        if stage_callback is not None:
            stage_callback(stage)

    convention = _require_phase_convention(convention)
    if not isinstance(waveform_is_prepared, (bool, np.bool_)):
        raise ValueError("waveform_is_prepared must be boolean")
    periods = _real_numeric_array(periods_s, "periods_s")
    velocity = _real_numeric_array(
        velocity_axis_km_s,
        "velocity_axis_km_s",
    )
    if (
        periods.ndim != 1
        or periods.size < 3
        or np.any(~np.isfinite(periods))
        or np.any(periods <= 0)
        or np.any(np.diff(periods) <= 0)
    ):
        raise ValueError("periods_s must contain at least three increasing values")
    if (
        velocity.ndim != 1
        or velocity.size < 2
        or np.any(~np.isfinite(velocity))
        or np.any(velocity <= 0)
        or np.any(np.diff(velocity) <= 0)
    ):
        raise ValueError("velocity_axis_km_s must be positive and increasing")
    snr_gap = _finite_scalar(snr_gap_s, "snr_gap_s", positive=True)
    first_penalty = _finite_scalar(beta1, "beta1")
    second_penalty = _finite_scalar(beta2, "beta2")
    if first_penalty < 0 or second_penalty < 0:
        raise ValueError("beta1 and beta2 must be non-negative")
    filter_alpha = (
        None if alpha is None else _finite_scalar(alpha, "alpha", positive=True)
    )
    if trace.distance_km <= 0 or trace.dt_s <= 0:
        return None
    if filter_alpha is None:
        filter_alpha = _finite_scalar(
            gaussian_alpha_for_distance(trace.distance_km),
            "alpha",
            positive=True,
        )
    time = _real_numeric_array(trace.time_s, "trace.time_s")
    if time.ndim != 1 or time.size < 2 or np.any(~np.isfinite(time)):
        raise ValueError("trace.time_s must be a finite one-dimensional array")
    steps = np.diff(time)
    if np.any(steps <= 0):
        raise ValueError("trace.time_s must be strictly increasing and uniform")
    spacing_tolerance = (
        64.0 * np.finfo(float).eps * max(1.0, abs(trace.dt_s))
    )
    if not np.allclose(
        steps,
        trace.dt_s,
        rtol=1e-10,
        atol=spacing_tolerance,
    ):
        raise ValueError("trace.time_s must be uniform and match trace.dt_s")

    prepared = (
        np.array(trace.symmetric_waveform, dtype=float, copy=True)
        if bool(waveform_is_prepared)
        else prepare_phase_waveform(
            time,
            trace.symmetric_waveform,
            convention,
        )
    )
    bank = gaussian_filter_bank(
        prepared,
        dt_s=trace.dt_s,
        periods_s=periods,
        alpha=filter_alpha,
    )
    mark_stage("filter_bank")
    sample_times = trace.distance_km / velocity
    amplitude_image = np.vstack(
        [
            np.interp(sample_times, time, row, left=0.0, right=0.0)
            for row in bank.envelope
        ]
    )
    row_max = np.max(amplitude_image, axis=1, keepdims=True)
    normalized_amplitude = np.zeros_like(amplitude_image)
    np.divide(
        amplitude_image,
        row_max,
        out=normalized_amplitude,
        where=row_max > 0,
    )
    scaled_energy = normalized_log_energy(amplitude_image)
    candidates = find_candidate_ridges(
        scaled_log_energy=scaled_energy,
        normalized_envelope_amplitude=normalized_amplitude,
        periods_s=periods,
        velocity_axis_km_s=velocity,
        beta1=first_penalty,
        beta2=second_penalty,
        max_candidates=1,
    )
    ridge = select_fundamental_ridge(candidates, periods_s=periods)
    mark_stage("dp_ridge")
    if not ridge.quality.accepted:
        return None

    ridge_columns = ridge.row_indices
    ridge_normalized_log_energy = scaled_energy[
        np.arange(periods.size),
        ridge_columns,
    ]
    ridge_normalized_envelope_amplitude = normalized_amplitude[
        np.arange(periods.size),
        ridge_columns,
    ]
    ridge_adjacent_jump_km_s = np.zeros(periods.size, dtype=float)
    ridge_adjacent_jump_km_s[1:] = np.abs(
        np.diff(ridge.group_velocities_km_s)
    )
    group_times = trace.distance_km / ridge.group_velocities_km_s
    scipy_phases = np.full(periods.size, np.nan, dtype=float)
    instantaneous = [None] * periods.size
    measurement_statuses = np.full(
        periods.size,
        "ridge_invalid",
        dtype=object,
    )
    for index, period in enumerate(periods):
        if not ridge.valid[index]:
            continue
        ridge_time = trace.distance_km / ridge.group_velocities_km_s[index]
        nearest = int(np.argmin(np.abs(time - ridge_time)))
        local_start = max(0, nearest - 1)
        local_stop = min(time.size, nearest + 2)
        local_peak = local_start + int(
            np.argmax(bank.envelope[index, local_start:local_stop])
        )
        refined = refine_group_arrival(
            time,
            bank.envelope[index],
            local_peak,
        )
        group_times[index] = refined.group_time_s
        scipy_phases[index] = interpolate_analytic_phase_at_arrival(
            time,
            bank.analytic_signals[index],
            refined.group_time_s,
        )
        phase_series = (
            convention.definition.scipy_phase_multiplier
            * np.angle(bank.analytic_signals[index])
        )
        frequency = estimate_instantaneous_frequency(
            time,
            phase_series,
            group_time_s=refined.group_time_s,
            nominal_period_s=period,
        )
        if frequency.status != "accepted":
            measurement_statuses[index] = frequency.status
            continue
        instantaneous[index] = frequency
        measurement_statuses[index] = "ready_for_phase_unwrap"
    mark_stage("group_arrival_phase_instantaneous_frequency")

    frequency_valid = np.asarray(
        [row is not None for row in instantaneous],
        dtype=bool,
    )
    instantaneous_periods = np.full(periods.size, np.nan, dtype=float)
    for index in np.flatnonzero(frequency_valid):
        instantaneous_periods[index] = instantaneous[
            index
        ].instantaneous_period_s
    duplicate_instantaneous = _exact_duplicate_mask(
        instantaneous_periods,
        frequency_valid,
    )
    measurement_statuses[
        duplicate_instantaneous
    ] = "duplicate_instantaneous_period"

    def anchored_component(valid):
        component = np.zeros(periods.size, dtype=bool)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            return component
        valid_periods = np.asarray(
            [
                instantaneous[index].instantaneous_period_s
                for index in valid_indices
            ],
            dtype=float,
        )
        anchor_index = int(
            valid_indices[
                int(np.argmin(np.abs(valid_periods - 3.5)))
            ]
        )
        start = anchor_index
        stop = anchor_index + 1
        while start > 0 and valid[start - 1]:
            start -= 1
        while stop < periods.size and valid[stop]:
            stop += 1
        component[start:stop] = True
        return component

    working_valid = frequency_valid & ~duplicate_instantaneous
    phase_segment = anchored_component(working_valid)
    segment_unwrap = None
    while np.count_nonzero(phase_segment) >= 3:
        segment_indices = np.flatnonzero(phase_segment)
        measured_periods = np.asarray(
            [
                instantaneous[index].instantaneous_period_s
                for index in segment_indices
            ],
            dtype=float,
        )
        candidate_unwrap = unwrap_phase_along_frequency(
            measured_periods,
            scipy_phases[segment_indices],
            group_times[segment_indices],
            convention=convention,
            anchor_period_s=3.5,
        )
        if candidate_unwrap.status == "accepted":
            segment_unwrap = candidate_unwrap
            break
        prediction_bad = (
            np.isfinite(candidate_unwrap.prediction_error_s)
            & (
                candidate_unwrap.prediction_error_s
                > PHASE_UNWRAP_PREDICTION_FRACTION * measured_periods
            )
        )
        cycle_jumps = (
            np.abs(np.diff(candidate_unwrap.cycle_counts))
            > PHASE_UNWRAP_MAX_CYCLE_STEP
        )
        local_bad = np.flatnonzero(prediction_bad)
        if np.any(cycle_jumps):
            local_bad = np.unique(
                np.concatenate(
                    (local_bad, np.flatnonzero(cycle_jumps) + 1)
                )
            )
        if local_bad.size == 0:
            break
        rejected_indices = segment_indices[local_bad]
        working_valid[rejected_indices] = False
        measurement_statuses[
            rejected_indices
        ] = "phase_unwrap_discontinuous"
        phase_segment = anchored_component(working_valid)

    measurement_valid = np.zeros(periods.size, dtype=bool)
    full_unwrapped = np.full(periods.size, np.nan, dtype=float)
    full_cycles = np.zeros(periods.size, dtype=int)
    full_raw_time = np.full(periods.size, np.nan, dtype=float)
    full_errors = np.full(periods.size, np.nan, dtype=float)
    full_sort_order = np.arange(periods.size, dtype=int)
    full_anchor_index = -1
    unwrap_anomaly_fraction = 0.0
    unwrap_maximum_run = 0
    if segment_unwrap is not None:
        segment_indices = np.flatnonzero(phase_segment)
        measurement_valid[segment_indices] = True
        full_unwrapped[segment_indices] = segment_unwrap.unwrapped_phase_rad
        full_cycles[segment_indices] = segment_unwrap.cycle_counts
        full_raw_time[segment_indices] = segment_unwrap.raw_phase_time_s
        full_errors[segment_indices] = segment_unwrap.prediction_error_s
        ordered_segment = segment_indices[segment_unwrap.sort_order]
        outside_segment = np.flatnonzero(~phase_segment)
        full_sort_order = np.concatenate(
            (ordered_segment, outside_segment)
        )
        full_anchor_index = int(
            segment_indices[segment_unwrap.anchor_index]
        )
        unwrap_anomaly_fraction = segment_unwrap.anomaly_fraction
        unwrap_maximum_run = segment_unwrap.max_consecutive_anomalies
        measurement_statuses[segment_indices] = "accepted"
    unresolved = frequency_valid & ~measurement_valid
    measurement_statuses[
        unresolved & (measurement_statuses == "ready_for_phase_unwrap")
    ] = "outside_anchored_phase_segment"
    unwrap = PhaseUnwrapResult(
        unwrapped_phase_rad=full_unwrapped,
        cycle_counts=full_cycles,
        raw_phase_time_s=full_raw_time,
        prediction_error_s=full_errors,
        sort_order=full_sort_order,
        valid_mask=measurement_valid,
        anomaly_fraction=unwrap_anomaly_fraction,
        max_consecutive_anomalies=unwrap_maximum_run,
        anchor_index=full_anchor_index,
        status=(
            "accepted"
            if np.all(measurement_valid)
            else "partial_phase_unwrap"
        ),
    )
    mark_stage("phase_unwrap")
    signal_start = trace.distance_km / 5.0
    signal_end = trace.distance_km / 1.6
    measurements = []
    for index, period in enumerate(periods):
        if not measurement_valid[index]:
            measurements.append(None)
            continue
        frequency = instantaneous[index]
        snr = _phase_measurement_snr(
            bank.filtered_waveforms[index],
            time,
            distance_km=trace.distance_km,
            period_s=frequency.instantaneous_period_s,
        )
        if not np.isfinite(snr) or snr < 0:
            snr = 0.0
        unwrapped_paper_phase = float(
            convention.definition.scipy_phase_multiplier
            * unwrap.unwrapped_phase_rad[index]
        )
        scipy_principal_phase = _canonical_principal_phase_rad(
            scipy_phases[index]
        )
        principal_paper_phase = _canonical_principal_phase_rad(
            convention.definition.scipy_phase_multiplier
            * scipy_principal_phase
        )
        paper_cycle_position = (
            unwrapped_paper_phase - principal_paper_phase
        ) / (2.0 * math.pi)
        paper_cycle_offset = int(round(paper_cycle_position))
        cycle_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(paper_cycle_position))
        )
        if not math.isclose(
            paper_cycle_position,
            paper_cycle_offset,
            rel_tol=0.0,
            abs_tol=cycle_tolerance,
        ):
            return None
        measurements.append(
            PeriodMeasurement(
                convention=convention,
                period_s=float(period),
                omega_inst_rad_s=frequency.omega_inst_rad_s,
                principal_paper_phase_rad=principal_paper_phase,
                unwrapped_paper_phase_rad=unwrapped_paper_phase,
                raw_phase_time_s=float(unwrap.raw_phase_time_s[index]),
                paper_phase_cycle_offset=paper_cycle_offset,
                group_time_s=float(group_times[index]),
                group_velocity_km_s=float(
                    trace.distance_km / group_times[index]
                ),
                snr=float(snr),
                signal_window_start_s=float(signal_start),
                signal_window_end_s=float(signal_end),
                filtered_waveform=bank.filtered_waveforms[index],
                envelope=bank.envelope[index],
            )
        )
    return PhaseCurveMeasurement(
        convention=convention,
        periods_s=periods,
        velocity_axis_km_s=velocity,
        group_times_s=group_times,
        scaled_log_energy=scaled_energy,
        measurements=tuple(measurements),
        ridge=ridge,
        phase_unwrap=unwrap,
        instantaneous_periods_s=instantaneous_periods,
        ridge_normalized_log_energy=ridge_normalized_log_energy,
        ridge_normalized_envelope_amplitude=(
            ridge_normalized_envelope_amplitude
        ),
        ridge_adjacent_jump_km_s=ridge_adjacent_jump_km_s,
        status=(
            "accepted"
            if np.all(measurement_valid)
            else "partial_phase_curve"
        ),
        measurement_valid=measurement_valid,
        measurement_statuses=tuple(measurement_statuses),
    )


def measure_single_period(
    trace: DatTrace,
    *,
    period_s: float,
    convention: PhaseConvention,
    vmin_km_s: float = 1.0,
    vmax_km_s: float = 4.0,
    alpha=None,  # type: Optional[float]
    snr_gap_s: float = 0.5,
):  # type: (...) -> Optional[PeriodMeasurement]
    """Thin target-row adapter over :func:`measure_phase_curve`."""

    convention = _require_phase_convention(convention)
    period = _finite_scalar(period_s, "period_s", positive=True)
    minimum_velocity = _finite_scalar(
        vmin_km_s,
        "vmin_km_s",
        positive=True,
    )
    maximum_velocity = _finite_scalar(
        vmax_km_s,
        "vmax_km_s",
        positive=True,
    )
    if minimum_velocity >= maximum_velocity:
        raise ValueError("vmin_km_s must be less than vmax_km_s")
    snr_gap = _finite_scalar(snr_gap_s, "snr_gap_s", positive=True)
    velocity = _deterministic_inclusive_grid(
        minimum_velocity,
        maximum_velocity,
        0.01,
    )
    curve = measure_phase_curve(
        trace,
        periods_s=period * np.array([0.96, 1.0, 1.04]),
        velocity_axis_km_s=velocity,
        alpha=alpha,
        beta1=0.5,
        beta2=1.0,
        snr_gap_s=snr_gap,
        convention=convention,
    )
    return None if curve is None else curve.measurements[1]


def compute_phase_speed_candidates(
    *,
    phi_tu: float,
    omega: float,
    distance_km: float,
    group_velocity_km_s: float,
    convention: PhaseConvention,
    branch_min: int = -8,
    branch_max: int = 8,
    phase_shift_rad=None,  # type: Optional[float]
    velocity_bounds=(1.0, 4.0),  # type: Tuple[float, float]
):  # type: (...) -> List[PhaseCandidate]
    """Return all physically plausible phase-speed candidates for integer branches.

    For the explicitly selected Bensen convention this implements:

        s_c = s_u + [phi(t_u) + 2*pi*N - pi/4] / (omega * distance)

    Passing the Lin convention applies its explicit ``-N*T`` branch meaning.
    The legacy ``phase_shift_rad`` parameter is accepted only when it equals
    the fixed term of the selected convention; it cannot define a third,
    implicit convention.
    """

    convention = _require_phase_convention(convention)
    phase = _finite_scalar(phi_tu, "phi_tu")
    angular_frequency = _finite_scalar(omega, "omega", positive=True)
    distance = _finite_scalar(distance_km, "distance_km", positive=True)
    group_velocity = _finite_scalar(
        group_velocity_km_s,
        "group_velocity_km_s",
        positive=True,
    )
    minimum_branch = _integer_scalar(branch_min, "branch_min")
    maximum_branch = _integer_scalar(branch_max, "branch_max")
    if minimum_branch > maximum_branch:
        raise ValueError("branch_min must not exceed branch_max")
    min_velocity, max_velocity = _validate_velocity_bounds(velocity_bounds)
    fixed_phase = convention.definition.fixed_phase_rad
    if phase_shift_rad is not None:
        compatibility_shift = _finite_scalar(
            phase_shift_rad,
            "phase_shift_rad",
        )
        compatibility_tolerance = (
            8.0 * np.finfo(float).eps * max(1.0, abs(fixed_phase))
        )
        if not math.isclose(
            compatibility_shift,
            fixed_phase,
            rel_tol=0.0,
            abs_tol=compatibility_tolerance,
        ):
            raise ValueError(
                "phase_shift_rad must equal the selected PhaseConvention fixed phase"
            )

    group_time = distance / group_velocity
    raw_time = raw_phase_travel_time(
        convention=convention,
        group_time_s=group_time,
        phase_rad=phase,
        omega_rad_s=angular_frequency,
    )
    period = 2.0 * math.pi / angular_frequency
    return _phase_candidates_from_raw_time(
        raw_time_s=raw_time,
        period_s=period,
        distance_km=distance,
        convention=convention,
        branch_min=minimum_branch,
        branch_max=maximum_branch,
        velocity_bounds=(min_velocity, max_velocity),
    )


def _validate_velocity_bounds(
    velocity_bounds,
) -> Tuple[float, float]:
    if isinstance(velocity_bounds, (str, bytes)):
        raise ValueError("velocity_bounds must be a positive increasing interval")
    try:
        raw_velocity_bounds = tuple(velocity_bounds)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "velocity_bounds must be a positive increasing interval"
        ) from None
    if len(raw_velocity_bounds) != 2:
        raise ValueError("velocity_bounds must be a positive increasing interval")
    try:
        min_velocity = _finite_scalar(
            raw_velocity_bounds[0],
            "velocity_bounds",
            positive=True,
        )
        max_velocity = _finite_scalar(
            raw_velocity_bounds[1],
            "velocity_bounds",
            positive=True,
        )
    except ValueError:
        raise ValueError(
            "velocity_bounds must be a positive increasing interval"
        ) from None
    if min_velocity >= max_velocity:
        raise ValueError("velocity_bounds must be a positive increasing interval")
    return min_velocity, max_velocity


def _phase_candidates_from_raw_time(
    *,
    raw_time_s: float,
    period_s: float,
    distance_km: float,
    convention: PhaseConvention,
    branch_min: int,
    branch_max: int,
    velocity_bounds,
) -> List[PhaseCandidate]:
    convention = _require_phase_convention(convention)
    raw_time = _finite_scalar(raw_time_s, "raw_time_s")
    period = _finite_scalar(period_s, "period_s", positive=True)
    distance = _finite_scalar(distance_km, "distance_km", positive=True)
    minimum_branch = _integer_scalar(branch_min, "branch_min")
    maximum_branch = _integer_scalar(branch_max, "branch_max")
    if minimum_branch > maximum_branch:
        raise ValueError("branch_min must not exceed branch_max")
    min_velocity, max_velocity = _validate_velocity_bounds(
        velocity_bounds
    )
    candidates = []  # type: List[PhaseCandidate]
    for branch in range(minimum_branch, maximum_branch + 1):
        try:
            phase_time = apply_cycle_count(
                raw_time,
                branch,
                period,
                convention=convention,
            )
        except ValueError:
            continue
        phase_slowness = phase_time / distance
        phase_velocity = distance / phase_time
        if not np.isfinite(phase_velocity):
            continue
        if phase_velocity < min_velocity or phase_velocity > max_velocity:
            continue
        candidates.append(
            PhaseCandidate(
                branch=branch,
                phase_velocity_km_s=float(phase_velocity),
                phase_slowness_s_km=float(phase_slowness),
            )
        )
    return candidates


def phase_speed_candidates_from_measurement(
    measurement: PeriodMeasurement,
    *,
    distance_km: float,
    branch_min: int = -8,
    branch_max: int = 8,
    velocity_bounds=(1.0, 4.0),
    convention=None,  # type: Optional[PhaseConvention]
) -> List[PhaseCandidate]:
    """Generate candidates using the convention carried by a formal measurement."""

    if not isinstance(measurement, PeriodMeasurement):
        raise ValueError("measurement must be a PeriodMeasurement")
    selected = measurement.convention
    if convention is not None:
        requested = _require_phase_convention(convention)
        if requested is not selected:
            raise ValueError("convention conflicts with measurement convention")
    return _phase_candidates_from_raw_time(
        raw_time_s=measurement.raw_phase_time_s,
        period_s=2.0 * math.pi / measurement.omega_inst_rad_s,
        distance_km=distance_km,
        convention=selected,
        branch_min=branch_min,
        branch_max=branch_max,
        velocity_bounds=velocity_bounds,
    )


def _smooth_reference_curve(
    periods_s: np.ndarray,
    velocities_km_s: np.ndarray,
    *,
    fallback: np.ndarray,
):  # type: (...) -> np.ndarray
    valid = np.isfinite(periods_s) & np.isfinite(velocities_km_s) & (velocities_km_s > 0)
    if np.count_nonzero(valid) < 2:
        return np.asarray(fallback, dtype=float)

    log_period = np.log(periods_s[valid])
    vel = velocities_km_s[valid]
    degree = min(3, len(vel) - 1)
    if degree <= 0:
        return np.asarray(fallback, dtype=float)
    coeff = np.polyfit(log_period, vel, deg=degree)
    ref = np.polyval(coeff, np.log(periods_s))
    if np.any(~np.isfinite(ref)):
        return np.asarray(fallback, dtype=float)
    return np.asarray(ref, dtype=float)


def select_branch_sequence(
    *,
    periods_s,  # type: Iterable[float]
    group_velocities_km_s,  # type: Iterable[float]
    candidate_lists,  # type: Iterable[List[PhaseCandidate]]
    iterations: int = 3,
):  # type: (...) -> BranchSelection
    """Pick one branch per period using a smooth-reference adaptation of Bensen.

    The original paper uses an external long-period reference plus a smoothness
    constraint. For our short-period, short-path local use we generally lack the
    long-period reference, so we adapt the same idea by:

    1. Starting from the group-velocity curve as a provisional reference.
    2. Selecting the candidate closest to that reference period by period.
    3. Fitting a smooth low-order curve in log-period.
    4. Repeating the candidate selection a few times.
    """

    periods = _real_numeric_iterable(periods_s, "periods_s")
    group_velocities = _real_numeric_iterable(
        group_velocities_km_s,
        "group_velocities_km_s",
    )
    candidate_lists = list(candidate_lists)
    iteration_count = _integer_scalar(iterations, "iterations")
    if periods.shape != group_velocities.shape:
        raise ValueError("periods_s and group_velocities_km_s must have the same shape")
    if len(candidate_lists) != len(periods):
        raise ValueError("candidate_lists length must match period count")

    reference = np.asarray(group_velocities, dtype=float)
    selected_branches = []  # type: List[int]
    selected_velocities = np.full(len(periods), np.nan, dtype=float)

    for _ in range(max(1, iteration_count)):
        current_branches = []  # type: List[int]
        current_velocities = np.full(len(periods), np.nan, dtype=float)
        for index, candidates in enumerate(candidate_lists):
            if not candidates:
                current_branches.append(0)
                continue
            best = min(
                candidates,
                key=lambda row: (
                    abs(row.phase_velocity_km_s - reference[index]),
                    abs(row.phase_velocity_km_s - group_velocities[index]),
                    abs(row.branch),
                ),
            )
            current_branches.append(best.branch)
            current_velocities[index] = best.phase_velocity_km_s

        reference = _smooth_reference_curve(
            periods,
            current_velocities,
            fallback=reference,
        )
        selected_branches = current_branches
        selected_velocities = current_velocities

    return BranchSelection(
        branches=selected_branches,
        phase_velocities_km_s=selected_velocities,
        reference_velocities_km_s=np.asarray(reference, dtype=float),
    )


def extract_bensen_phase_curve(
    dat_path: str,
    *,
    periods_s=None,  # type: Optional[np.ndarray]
    vmin_km_s: float = 1.0,
    vmax_km_s: float = 4.0,
    min_snr: float = 5.0,
    branch_min: int = -8,
    branch_max: int = 8,
):  # type: (...) -> BensenExtraction
    minimum_velocity = _finite_scalar(
        vmin_km_s,
        "vmin_km_s",
        positive=True,
    )
    maximum_velocity = _finite_scalar(
        vmax_km_s,
        "vmax_km_s",
        positive=True,
    )
    if minimum_velocity >= maximum_velocity:
        raise ValueError("vmin_km_s must be less than vmax_km_s")
    minimum_snr = _finite_scalar(min_snr, "min_snr")
    minimum_branch = _integer_scalar(branch_min, "branch_min")
    maximum_branch = _integer_scalar(branch_max, "branch_max")
    if minimum_branch > maximum_branch:
        raise ValueError("branch_min must not exceed branch_max")
    if periods_s is None:
        periods_s = np.round(np.arange(0.2, 5.0 + 1e-9, 0.1), 2)
    periods = _real_numeric_array(periods_s, "periods_s")
    if (
        periods.ndim != 1
        or periods.size == 0
        or np.any(~np.isfinite(periods))
        or np.any(periods <= 0)
        or np.any(np.diff(periods) <= 0)
    ):
        raise ValueError("periods_s must be positive, finite, and increasing")
    trace = read_dat_trace(dat_path)
    measurements = []  # type: List[Optional[PeriodMeasurement]]
    candidate_lists = []  # type: List[List[PhaseCandidate]]
    group_velocities = np.full(periods.size, np.nan, dtype=float)
    snr = np.full(periods.size, np.nan, dtype=float)
    anchor_period_s = 3.5
    grid_start = min(float(periods[0]), anchor_period_s)
    grid_stop = max(float(periods[-1]), anchor_period_s)
    if grid_start == grid_stop:
        grid_start = max(0.05, grid_start - 0.05)
        grid_stop += 0.05
    interval_count = max(
        2,
        int(math.ceil((grid_stop - grid_start) / 0.05)),
    )
    continuous_periods = np.linspace(
        grid_start,
        grid_stop,
        interval_count + 1,
        dtype=float,
    )
    measurement_periods = np.unique(
        np.concatenate(
            (
                continuous_periods,
                periods,
                np.array([anchor_period_s]),
            )
        )
    )
    velocity_axis = _deterministic_inclusive_grid(
        minimum_velocity,
        maximum_velocity,
        0.01,
    )
    curve = measure_phase_curve(
        trace,
        periods_s=measurement_periods,
        velocity_axis_km_s=velocity_axis,
        beta1=0.5,
        beta2=1.0,
        convention=PhaseConvention.BENSEN_VELOCITY_CCF,
    )

    for index, period_s in enumerate(periods):
        measurement = None
        if curve is not None:
            curve_index = int(
                np.argmin(np.abs(curve.periods_s - period_s))
            )
            match_tolerance = (
                32.0
                * np.finfo(float).eps
                * max(1.0, abs(float(period_s)))
            )
            if math.isclose(
                float(curve.periods_s[curve_index]),
                float(period_s),
                rel_tol=0.0,
                abs_tol=match_tolerance,
            ):
                measurement = curve.measurements[curve_index]
        measurements.append(measurement)
        if measurement is None:
            candidate_lists.append([])
            continue
        group_velocities[index] = measurement.group_velocity_km_s
        snr[index] = measurement.snr
        if np.isfinite(measurement.snr) and measurement.snr < minimum_snr:
            candidate_lists.append([])
            continue
        candidates = phase_speed_candidates_from_measurement(
            measurement,
            distance_km=trace.distance_km,
            branch_min=minimum_branch,
            branch_max=maximum_branch,
            velocity_bounds=(minimum_velocity, maximum_velocity),
        )
        candidate_lists.append(candidates)

    finite_group = group_velocities[np.isfinite(group_velocities)]
    if finite_group.size == 0:
        safe_group = np.full(
            periods.size,
            0.5 * (minimum_velocity + maximum_velocity),
            dtype=float,
        )
    else:
        group_median = float(np.median(finite_group))
        safe_group = np.where(np.isfinite(group_velocities), group_velocities, group_median)

    selection = select_branch_sequence(
        periods_s=periods,
        group_velocities_km_s=safe_group,
        candidate_lists=candidate_lists,
    )
    branches = []  # type: List[Optional[int]]
    phase_velocities = np.full(periods.size, np.nan, dtype=float)
    for index, candidates in enumerate(candidate_lists):
        if not candidates or not np.isfinite(selection.phase_velocities_km_s[index]):
            branches.append(None)
            continue
        branches.append(selection.branches[index])
        phase_velocities[index] = selection.phase_velocities_km_s[index]

    return BensenExtraction(
        trace=trace,
        periods_s=periods,
        measurements=measurements,
        group_velocities_km_s=group_velocities,
        phase_velocities_km_s=phase_velocities,
        branches=branches,
        snr=snr,
        reference_velocities_km_s=selection.reference_velocities_km_s,
    )
