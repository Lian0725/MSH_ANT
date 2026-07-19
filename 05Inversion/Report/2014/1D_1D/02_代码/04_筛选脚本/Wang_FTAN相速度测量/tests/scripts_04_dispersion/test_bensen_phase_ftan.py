import importlib.util
import inspect
import itertools
import json
import math
import contextlib
import sys
import time
import types
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.interpolate import PchipInterpolator


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "04_dispersion"
    / "bensen_phase_ftan.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("bensen_phase_ftan", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BensenPhaseFtanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def require_module_attribute(self, name):
        value = getattr(self.mod, name, None)
        self.assertIsNotNone(value, f"bensen_phase_ftan must define {name}")
        return value

    def test_ftan_config_defaults_are_immutable_deterministic_grids(self):
        config_class = self.require_module_attribute("FtanConfig")
        config = config_class()

        self.assertEqual(config.target_periods_s, (3.0, 3.5, 4.0, 5.0))
        approved_fields = (
            "alpha_candidates",
            "beta1_candidates",
            "beta2_candidates",
        )
        self.assertTrue(
            all(hasattr(config, name) for name in approved_fields),
            "FtanConfig must expose the approved candidate field names",
        )
        self.assertEqual(
            config.alpha_candidates,
            (5.0, 8.0, 12.0, 16.0, 20.0, 25.0),
        )
        self.assertEqual(config.beta1_candidates, (0.0, 0.5, 1.0, 2.0, 4.0))
        self.assertEqual(config.beta2_candidates, (0.0, 1.0, 2.0, 4.0, 8.0))
        self.assertFalse(
            any(
                hasattr(config, name)
                for name in ("alpha_values", "beta1_values", "beta2_values")
            )
        )

        expected_periods = np.arange(250, 501, 5, dtype=float) / 100.0
        expected_group_velocities = np.arange(160, 501, dtype=float) / 100.0
        np.testing.assert_array_equal(config.periods_s, expected_periods)
        np.testing.assert_array_equal(
            config.group_velocities_km_s,
            expected_group_velocities,
        )
        np.testing.assert_array_equal(config.periods_s, config.periods_s)
        np.testing.assert_array_equal(
            config.group_velocities_km_s,
            config.group_velocities_km_s,
        )
        self.assertFalse(config.periods_s.flags.writeable)
        self.assertFalse(config.group_velocities_km_s.flags.writeable)

        with self.assertRaises(FrozenInstanceError):
            config.period_min_s = 3.0
        with self.assertRaises(ValueError):
            config.periods_s[0] = 3.0

    def test_gaussian_filter_bank_rejects_nonpositive_dt(self):
        gaussian_filter_bank = self.require_module_attribute("gaussian_filter_bank")
        waveform = np.ones(16, dtype=float)

        for dt_s in (0.0, -0.01, True, np.array([0.01])):
            with self.subTest(dt_s=dt_s):
                with self.assertRaisesRegex(ValueError, "dt_s"):
                    gaussian_filter_bank(
                        waveform,
                        dt_s=dt_s,
                        periods_s=np.array([1.0]),
                        alpha=20.0,
                    )

    def test_gaussian_filter_bank_rejects_nonpositive_periods(self):
        gaussian_filter_bank = self.require_module_attribute("gaussian_filter_bank")
        waveform = np.ones(16, dtype=float)

        for periods_s in (
            np.array([0.0]),
            np.array([1.0, -2.0]),
            np.array([True]),
        ):
            with self.subTest(periods_s=periods_s):
                with self.assertRaisesRegex(ValueError, "periods_s"):
                    gaussian_filter_bank(
                        waveform,
                        dt_s=0.01,
                        periods_s=periods_s,
                        alpha=20.0,
                    )

    def test_gaussian_filter_bank_rejects_nonpositive_and_nonfinite_alpha(self):
        gaussian_filter_bank = self.require_module_attribute("gaussian_filter_bank")
        waveform = np.ones(16, dtype=float)

        for alpha in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    gaussian_filter_bank(
                        waveform,
                        dt_s=0.01,
                        periods_s=np.array([1.0]),
                        alpha=alpha,
                    )

    def test_gaussian_filter_bank_rejects_bool_and_nonscalar_alpha(self):
        gaussian_filter_bank = self.require_module_attribute("gaussian_filter_bank")
        waveform = np.ones(16, dtype=float)

        for alpha in (False, True, np.array([12.0])):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    gaussian_filter_bank(
                        waveform,
                        dt_s=0.01,
                        periods_s=np.array([1.0]),
                        alpha=alpha,
                    )

    def test_gaussian_filter_bank_rejects_nonfinite_waveform(self):
        gaussian_filter_bank = self.require_module_attribute("gaussian_filter_bank")
        for bad_value in (np.nan, np.inf, -np.inf):
            waveform = np.ones(16, dtype=float)
            waveform[5] = bad_value
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "waveform"):
                    gaussian_filter_bank(
                        waveform,
                        dt_s=0.01,
                        periods_s=np.array([1.0]),
                        alpha=20.0,
                    )

    def test_gaussian_filter_bank_is_shaped_selective_and_bitwise_deterministic(self):
        gaussian_filter_bank = self.require_module_attribute("gaussian_filter_bank")
        dt_s = 1.0 / 64.0
        time_s = np.arange(1024, dtype=float) * dt_s
        waveform = np.sin(2.0 * np.pi * time_s)
        periods_s = np.array([1.0, 4.0], dtype=float)

        first = gaussian_filter_bank(
            waveform,
            dt_s=dt_s,
            periods_s=periods_s,
            alpha=20.0,
        )
        second = gaussian_filter_bank(
            waveform,
            dt_s=dt_s,
            periods_s=periods_s,
            alpha=20.0,
        )

        expected_shape = (len(periods_s), len(waveform))
        self.assertEqual(first.filtered_waveforms.shape, expected_shape)
        self.assertEqual(first.analytic_signals.shape, expected_shape)
        self.assertEqual(first.envelope.shape, expected_shape)
        self.assertTrue(np.iscomplexobj(first.analytic_signals))
        np.testing.assert_allclose(
            first.envelope,
            np.abs(first.analytic_signals),
            rtol=np.finfo(float).eps,
            atol=np.finfo(float).eps,
        )
        np.testing.assert_array_equal(
            first.filtered_waveforms,
            second.filtered_waveforms,
        )
        np.testing.assert_array_equal(
            first.analytic_signals,
            second.analytic_signals,
        )
        np.testing.assert_array_equal(first.envelope, second.envelope)

        target_response = float(np.mean(first.envelope[0]))
        far_response = float(np.mean(first.envelope[1]))
        self.assertGreater(target_response, 100.0 * max(far_response, 1e-15))

    @staticmethod
    def _aligned_strided_array(
        *,
        shape,
        dtype,
        row_stride_bytes,
        offset_mod_4096,
    ):
        itemsize = np.dtype(dtype).itemsize
        required = row_stride_bytes * (shape[0] - 1) + itemsize * shape[1]
        raw = np.empty(required + 8192, dtype=np.uint8)
        offset = (
            (-raw.ctypes.data) % 4096 + int(offset_mod_4096)
        )
        view = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=raw,
            offset=offset,
            strides=(row_stride_bytes, itemsize),
        )
        return raw, view

    def test_deterministic_complex_magnitude_ignores_alignment_and_strides(self):
        magnitude = self.require_module_attribute(
            "_deterministic_complex_magnitude"
        )
        rng = np.random.default_rng(4)
        values = (
            rng.normal(size=(2, 1024))
            + 1j * rng.normal(size=(2, 1024))
        )
        layouts = (
            (16384, 0, 8192, 0),
            (16400, 8, 8200, 4),
        )
        outputs = []
        buffers = []
        for input_stride, input_offset, output_stride, output_offset in layouts:
            input_raw, input_view = self._aligned_strided_array(
                shape=values.shape,
                dtype=np.complex128,
                row_stride_bytes=input_stride,
                offset_mod_4096=input_offset,
            )
            output_raw, output_view = self._aligned_strided_array(
                shape=values.shape,
                dtype=np.float64,
                row_stride_bytes=output_stride,
                offset_mod_4096=output_offset,
            )
            self.assertEqual(input_view.strides[0], input_stride)
            self.assertEqual(input_view.ctypes.data % 4096, input_offset)
            self.assertEqual(output_view.strides[0], output_stride)
            self.assertEqual(output_view.ctypes.data % 4096, output_offset)
            self.assertEqual(output_view.flags.aligned, output_offset == 0)
            input_view[...] = values

            returned = magnitude(input_view, out=output_view)

            self.assertIs(returned, output_view)
            np.testing.assert_allclose(
                output_view,
                np.abs(values),
                rtol=np.finfo(float).eps,
                atol=np.finfo(float).eps,
            )
            outputs.append(output_view.copy())
            buffers.extend((input_raw, output_raw))

        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertTrue(all(buffer.size > 0 for buffer in buffers))

    def test_filter_bank_envelope_is_bitwise_stable_across_repeated_allocations(self):
        gaussian_filter_bank = self.require_module_attribute(
            "gaussian_filter_bank"
        )
        dt_s = 1.0 / 64.0
        time_s = np.arange(1024, dtype=float) * dt_s
        waveform = (
            np.sin(2.0 * np.pi * time_s)
            + 0.3 * np.cos(2.0 * np.pi * time_s / 4.0)
        )
        periods_s = np.array([1.0, 4.0], dtype=float)
        reference = gaussian_filter_bank(
            waveform,
            dt_s=dt_s,
            periods_s=periods_s,
            alpha=20.0,
        )

        for allocation_offset in range(32):
            _allocator_perturbation = np.empty(
                17 * allocation_offset + 1,
                dtype=np.uint8,
            )
            repeated = gaussian_filter_bank(
                waveform,
                dt_s=dt_s,
                periods_s=periods_s,
                alpha=20.0,
            )
            np.testing.assert_array_equal(
                repeated.filtered_waveforms,
                reference.filtered_waveforms,
            )
            np.testing.assert_array_equal(
                repeated.analytic_signals,
                reference.analytic_signals,
            )
            np.testing.assert_array_equal(
                repeated.envelope,
                reference.envelope,
            )
            self.assertEqual(_allocator_perturbation.size, 17 * allocation_offset + 1)

        source = inspect.getsource(gaussian_filter_bank)
        self.assertIn("_deterministic_complex_magnitude", source)
        self.assertNotIn("np.abs(analytic)", source)
        self.assertNotIn(
            "envelope = np.abs(analytic)",
            MODULE_PATH.read_text(encoding="utf-8"),
        )

    def test_legacy_narrowband_wrapper_matches_filter_bank(self):
        gaussian_filter_bank = self.require_module_attribute("gaussian_filter_bank")
        dt_s = 0.02
        time_s = np.arange(256, dtype=float) * dt_s
        waveform = np.cos(2.0 * np.pi * time_s / 0.8)

        bank = gaussian_filter_bank(
            waveform,
            dt_s=dt_s,
            periods_s=np.array([0.8]),
            alpha=12.0,
        )
        legacy = self.mod.gaussian_narrowband_waveform(
            waveform,
            dt_s=dt_s,
            period_s=0.8,
            alpha=12.0,
        )

        np.testing.assert_array_equal(legacy, bank.filtered_waveforms[0])

        for bad_period in (True, 0.0, np.array([0.8])):
            with self.subTest(bad_period=bad_period):
                with self.assertRaisesRegex(ValueError, "period_s"):
                    self.mod.gaussian_narrowband_waveform(
                        waveform,
                        dt_s=dt_s,
                        period_s=bad_period,
                        alpha=12.0,
                    )

    def test_normalized_log_energy_scales_within_each_period_row(self):
        normalized_log_energy = self.require_module_attribute("normalized_log_energy")
        envelope = np.array(
            [
                [0.0, 1e-3, 1.0],
                [0.0, 1.0, 1000.0],
            ],
            dtype=float,
        )

        normalized = normalized_log_energy(envelope)

        # Rows are periods and columns are time samples. Each period is first
        # divided by its own maximum, then scaled along its own sample axis.
        expected = np.array(
            [
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(normalized, expected, rtol=0.0, atol=1e-15)
        self.assertTrue(np.all(np.isfinite(normalized)))
        self.assertTrue(np.all((normalized >= 0.0) & (normalized <= 1.0)))

    def test_normalized_log_energy_is_invariant_to_positive_period_scaling(self):
        normalized_log_energy = self.require_module_attribute("normalized_log_energy")
        envelope = np.array(
            [
                [0.0, 0.01, 0.1, 1.0],
                [4.0, 0.4, 0.04, 0.0],
            ],
            dtype=float,
        )
        scaled = envelope * np.array([[100.0], [0.01]])

        baseline = normalized_log_energy(envelope)
        rescaled = normalized_log_energy(scaled)

        np.testing.assert_allclose(rescaled, baseline, rtol=0.0, atol=1e-15)

    def test_normalized_log_energy_zeroes_all_zero_and_constant_periods(self):
        normalized_log_energy = self.require_module_attribute("normalized_log_energy")
        envelope = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [3.0, 3.0, 3.0, 3.0],
                [0.0, 0.1, 1.0, 10.0],
            ],
            dtype=float,
        )

        normalized = normalized_log_energy(envelope)

        np.testing.assert_array_equal(normalized[:2], np.zeros((2, 4)))
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_normalized_log_energy_rejects_negative_envelope(self):
        normalized_log_energy = self.require_module_attribute("normalized_log_energy")
        envelope = np.array([[0.0, -0.1, 1.0]], dtype=float)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalized_log_energy(envelope)

    def test_normalized_log_energy_supports_period_axis_one(self):
        normalized_log_energy = self.require_module_attribute("normalized_log_energy")
        self.assertIn(
            "period_axis",
            inspect.signature(normalized_log_energy).parameters,
        )
        period_rows = np.array(
            [
                [0.0, 1e-3, 1.0],
                [0.0, 1.0, 1000.0],
            ],
            dtype=float,
        )

        normalized = normalized_log_energy(period_rows.T, period_axis=1)

        expected_period_rows = np.array(
            [
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            normalized,
            expected_period_rows.T,
            rtol=0.0,
            atol=1e-15,
        )

    def test_compute_phase_speed_candidates_recovers_true_branch(self):
        distance_km = 50.0
        period_s = 1.0
        omega = 2.0 * math.pi / period_s
        group_velocity = 2.1
        true_phase_velocity = 2.45
        true_branch = 1

        su = 1.0 / group_velocity
        sc = 1.0 / true_phase_velocity
        phi_tu = omega * distance_km * (sc - su) - 2.0 * math.pi * true_branch + math.pi / 4.0

        candidates = self.mod.compute_phase_speed_candidates(
            phi_tu=phi_tu,
            omega=omega,
            distance_km=distance_km,
            group_velocity_km_s=group_velocity,
            convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            branch_min=-3,
            branch_max=3,
            velocity_bounds=(1.0, 4.0),
        )

        by_branch = {row.branch: row.phase_velocity_km_s for row in candidates}
        self.assertIn(true_branch, by_branch)
        self.assertAlmostEqual(by_branch[true_branch], true_phase_velocity, places=6)

    def test_bensen_phase_formula_recovers_velocity_and_rejects_wrong_fixed_sign(self):
        phase_convention = self.require_module_attribute("PhaseConvention")
        raw_phase_travel_time = self.require_module_attribute(
            "raw_phase_travel_time"
        )
        apply_cycle_count = self.require_module_attribute("apply_cycle_count")
        distance_km = 18.75
        group_velocity_km_s = 2.47
        true_phase_velocity_km_s = 3.18
        period_s = 3.5
        omega_rad_s = 2.0 * np.pi / period_s
        cycle_count = -2
        group_time_s = distance_km / group_velocity_km_s
        true_phase_time_s = distance_km / true_phase_velocity_km_s
        phi_tu_rad = (
            omega_rad_s * (true_phase_time_s - group_time_s)
            - 2.0 * np.pi * cycle_count
            + np.pi / 4.0
        )

        raw_time_s = raw_phase_travel_time(
            convention=phase_convention.BENSEN_VELOCITY_CCF,
            group_time_s=group_time_s,
            phase_rad=phi_tu_rad,
            omega_rad_s=omega_rad_s,
        )
        corrected_time_s = apply_cycle_count(
            raw_time_s,
            cycle_count,
            2.0 * np.pi / omega_rad_s,
        )

        recovered_velocity = distance_km / corrected_time_s
        self.assertAlmostEqual(
            recovered_velocity,
            true_phase_velocity_km_s,
            places=10,
        )

        wrong_raw_time_s = (
            group_time_s + (phi_tu_rad + np.pi / 4.0) / omega_rad_s
        )
        wrong_corrected_time_s = (
            wrong_raw_time_s + cycle_count * 2.0 * np.pi / omega_rad_s
        )
        wrong_velocity = distance_km / wrong_corrected_time_s
        self.assertGreater(
            abs(wrong_velocity / true_phase_velocity_km_s - 1.0),
            1e-2,
        )

    def test_phase_convention_definitions_are_immutable_and_json_serializable(self):
        phase_convention = self.require_module_attribute("PhaseConvention")
        definition_type = self.require_module_attribute(
            "PhaseConventionDefinition"
        )
        expected = {
            phase_convention.BENSEN_VELOCITY_CCF: {
                "phase_time_sign": -1,
                "formula_phase_sign": 1,
                "cycle_phase_sign": 1,
                "apply_negative_time_derivative": False,
            },
            phase_convention.LIN_NEGATIVE_DERIVATIVE_EGF: {
                "phase_time_sign": -1,
                "formula_phase_sign": 1,
                "cycle_phase_sign": -1,
                "apply_negative_time_derivative": True,
            },
        }

        self.assertEqual(
            set(phase_convention.__members__),
            {
                "BENSEN_VELOCITY_CCF",
                "LIN_NEGATIVE_DERIVATIVE_EGF",
            },
        )
        for convention, expected_fields in expected.items():
            with self.subTest(convention=convention.name):
                definition = convention.definition
                self.assertIsInstance(definition, definition_type)
                self.assertEqual(definition.hilbert_phase_sign, -1)
                self.assertEqual(definition.scipy_phase_multiplier, -1)
                self.assertAlmostEqual(
                    definition.fixed_phase_rad,
                    -np.pi / 4.0,
                )
                for name, value in expected_fields.items():
                    self.assertEqual(getattr(definition, name), value)
                self.assertIn("phi", definition.formula)
                self.assertTrue(definition.cycle_count_meaning)

                metadata = convention.metadata
                serialized = json.dumps(metadata, sort_keys=True)
                decoded = json.loads(serialized)
                self.assertEqual(decoded["name"], convention.name)
                self.assertEqual(
                    decoded["phase_time_sign"],
                    expected_fields["phase_time_sign"],
                )
                self.assertEqual(
                    decoded["formula_phase_sign"],
                    expected_fields["formula_phase_sign"],
                )
                self.assertEqual(decoded["scipy_phase_multiplier"], -1)
                self.assertEqual(
                    decoded["cycle_phase_sign"],
                    expected_fields["cycle_phase_sign"],
                )
                self.assertEqual(
                    decoded["apply_negative_time_derivative"],
                    expected_fields["apply_negative_time_derivative"],
                )
                with self.assertRaises(TypeError):
                    metadata["phase_time_sign"] = -1
                with self.assertRaises(FrozenInstanceError):
                    definition.phase_time_sign = -1

        template = dict(
            hilbert_phase_sign=-1,
            scipy_phase_multiplier=-1,
            phase_time_sign=-1,
            formula_phase_sign=1,
            fixed_phase_rad=-np.pi / 4.0,
            cycle_phase_sign=1,
            apply_negative_time_derivative=False,
            cycle_count_meaning="test",
            formula="test",
            description="test",
        )
        for field, value in (
            ("hilbert_phase_sign", 0),
            ("scipy_phase_multiplier", 2),
            ("formula_phase_sign", 0),
            ("cycle_phase_sign", -2),
            ("phase_time_sign", 1),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    definition_type(**{**template, field: value})

    def test_lin_negative_derivative_pipeline_recovers_velocity_and_raw_ccf_fails(self):
        phase_convention = self.require_module_attribute("PhaseConvention")
        prepare_phase_waveform = self.require_module_attribute(
            "prepare_phase_waveform"
        )
        raw_phase_travel_time = self.require_module_attribute(
            "raw_phase_travel_time"
        )
        apply_cycle_count = self.require_module_attribute("apply_cycle_count")
        convention = phase_convention.LIN_NEGATIVE_DERIVATIVE_EGF
        period_s = 2.5
        omega_rad_s = 2.0 * np.pi / period_s
        distance_km = 25.0
        group_velocity_km_s = 2.6
        true_phase_velocity_km_s = 3.3
        cycle_count = 1
        group_time_s = distance_km / group_velocity_km_s
        true_phase_time_s = distance_km / true_phase_velocity_km_s
        phi_tu_rad = (
            omega_rad_s * (true_phase_time_s - group_time_s)
            + np.pi / 4.0
            + 2.0 * np.pi * cycle_count
        )
        phi_tu_rad = float(np.angle(np.exp(1j * phi_tu_rad)))
        scipy_phi_tu_rad = -phi_tu_rad
        dt_s = 0.01
        time_s = np.arange(0.0, 40.0, dt_s)
        symmetric_ccf = (
            -np.sin(
                omega_rad_s * (time_s - group_time_s) + scipy_phi_tu_rad
            )
            / omega_rad_s
        )
        original_ccf = symmetric_ccf.copy()

        lin_egf = prepare_phase_waveform(
            time_s,
            symmetric_ccf,
            convention,
        )

        np.testing.assert_array_equal(
            lin_egf,
            -np.gradient(symmetric_ccf, time_s),
        )
        np.testing.assert_array_equal(symmetric_ccf, original_ccf)
        self.assertFalse(lin_egf.flags.writeable)
        analytic_egf = self.mod.hilbert(lin_egf)
        measured_phi_rad = self.mod.interpolate_analytic_phase_at_arrival(
            time_s,
            analytic_egf,
            group_time_s,
            convention=convention,
        )
        raw_time_s = raw_phase_travel_time(
            convention=convention,
            group_time_s=group_time_s,
            phase_rad=measured_phi_rad,
            omega_rad_s=omega_rad_s,
        )
        corrected_time_s = apply_cycle_count(
            raw_time_s,
            cycle_count,
            period_s,
            convention=convention,
        )
        recovered_velocity = distance_km / corrected_time_s
        self.assertLess(
            abs(recovered_velocity / true_phase_velocity_km_s - 1.0),
            1e-4,
        )

        wrong_analytic = self.mod.hilbert(symmetric_ccf)
        wrong_phi_rad = self.mod.interpolate_analytic_phase_at_arrival(
            time_s,
            wrong_analytic,
            group_time_s,
            convention=convention,
        )
        wrong_raw_time_s = raw_phase_travel_time(
            convention=convention,
            group_time_s=group_time_s,
            phase_rad=wrong_phi_rad,
            omega_rad_s=omega_rad_s,
        )
        wrong_corrected_time_s = apply_cycle_count(
            wrong_raw_time_s,
            cycle_count,
            period_s,
            convention=convention,
        )
        wrong_velocity = distance_km / wrong_corrected_time_s
        self.assertGreater(
            abs(wrong_velocity / true_phase_velocity_km_s - 1.0),
            5e-2,
        )

    def test_single_period_measurement_preprocesses_each_convention_independently(self):
        measure = self.require_module_attribute("measure_single_period")
        period_s = 3.0
        omega_rad_s = 2.0 * np.pi / period_s
        paper_phi_rad = 0.6
        scipy_phi_rad = -paper_phi_rad
        distance_km = 20.0
        group_time_s = 8.0
        dt_s = 0.02
        time_s = np.arange(-30.0, 50.0, dt_s)
        local_time_s = time_s - group_time_s
        desired_egf = (
            np.exp(-0.5 * (local_time_s / (1.5 * period_s)) ** 2)
            * np.cos(omega_rad_s * local_time_s + scipy_phi_rad)
        )
        lin_ccf = np.empty_like(desired_egf)
        lin_ccf[0] = 0.0
        lin_ccf[1:] = -np.cumsum(
            0.5 * (desired_egf[1:] + desired_egf[:-1]) * np.diff(time_s)
        )

        def trace_for(waveform, name):
            return self.mod.DatTrace(
                pair_name=name,
                distance_km=distance_km,
                dt_s=dt_s,
                time_s=time_s,
                positive_lag=waveform.copy(),
                negative_lag_reversed=waveform.copy(),
                symmetric_waveform=waveform.copy(),
                lon_a=0.0,
                lat_a=0.0,
                lon_b=0.1,
                lat_b=0.1,
            )

        bensen_convention = (
            self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        )
        lin_convention = (
            self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF
        )
        bensen_trace = trace_for(desired_egf, "bensen")
        lin_trace = trace_for(lin_ccf, "lin")
        with mock.patch.object(
            self.mod,
            "prepare_phase_waveform",
            wraps=self.mod.prepare_phase_waveform,
        ) as prepare:
            bensen_measurement = measure(
                bensen_trace,
                period_s=period_s,
                vmin_km_s=1.5,
                vmax_km_s=4.0,
                alpha=20.0,
                convention=bensen_convention,
            )
            lin_measurement = measure(
                lin_trace,
                period_s=period_s,
                vmin_km_s=1.5,
                vmax_km_s=4.0,
                alpha=20.0,
                convention=lin_convention,
            )

        self.assertEqual(prepare.call_count, 2)
        self.assertIsNotNone(bensen_measurement)
        self.assertIsNotNone(lin_measurement)
        for measurement in (bensen_measurement, lin_measurement):
            self.assertAlmostEqual(
                measurement.group_time_s,
                group_time_s,
                delta=0.1,
            )
            self.assertAlmostEqual(
                measurement.phi_tu_rad,
                paper_phi_rad,
                delta=0.03,
            )
        self.assertFalse(
            np.shares_memory(
                bensen_measurement.filtered_waveform,
                lin_measurement.filtered_waveform,
            )
        )

    def test_public_phase_curve_routes_formal_kernels_and_freezes_results(self):
        measure_curve = self.require_module_attribute("measure_phase_curve")
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        dt_s = 0.02
        time_s = np.arange(-30.0, 50.0, dt_s)
        group_time_s = 8.013
        waveform = (
            np.exp(-0.5 * ((time_s - group_time_s) / 4.5) ** 2)
            * np.cos(2.0 * np.pi * (time_s - group_time_s) / 3.0 - 0.4)
        )
        sidelobe_time_s = 5.1
        waveform = waveform + (
            0.2
            * np.exp(-0.5 * ((time_s - sidelobe_time_s) / 0.6) ** 2)
            * np.cos(2.0 * np.pi * (time_s - sidelobe_time_s) / 1.7)
        )
        waveform[int(np.argmin(np.abs(time_s - (group_time_s + 0.28))))] += 0.08
        trace = self.mod.DatTrace(
            "formal_curve", 20.0, dt_s, time_s,
            waveform.copy(), waveform.copy(), waveform.copy(),
            0.0, 0.0, 0.1, 0.1,
        )
        kernels = (
            "gaussian_filter_bank",
            "find_candidate_ridges",
            "select_fundamental_ridge",
            "refine_group_arrival",
            "estimate_instantaneous_frequency",
            "unwrap_phase_along_frequency",
        )
        with contextlib.ExitStack() as stack:
            spies = [
                stack.enter_context(
                    mock.patch.object(
                        self.mod,
                        name,
                        wraps=getattr(self.mod, name),
                    )
                )
                for name in kernels
            ]
            result = measure_curve(
                trace,
                periods_s=np.array([2.88, 3.0, 3.12]),
                velocity_axis_km_s=(
                    self.mod.FtanConfig().group_velocities_km_s
                ),
                alpha=20.0,
                beta1=0.5,
                beta2=1.0,
                convention=convention,
            )
        self.assertEqual(result.convention, convention)
        self.assertEqual(len(result.measurements), 3)
        self.assertTrue(all(spy.called for spy in spies))
        self.assertTrue(
            all(row.convention is convention for row in result.measurements)
        )
        paper_unwrapped = (
            convention.definition.scipy_phase_multiplier
            * result.phase_unwrap.unwrapped_phase_rad
        )
        expected_raw_time = np.array(
            [
                self.mod.raw_phase_travel_time(
                    convention=convention,
                    group_time_s=row.group_time_s,
                    phase_rad=phase,
                    omega_rad_s=row.omega_inst_rad_s,
                )
                for row, phase in zip(result.measurements, paper_unwrapped)
            ]
        )
        np.testing.assert_allclose(
            result.phase_unwrap.raw_phase_time_s,
            expected_raw_time,
            rtol=0.0,
            atol=2e-14,
        )
        self.assertAlmostEqual(result.group_times_s[1], group_time_s, delta=0.2)
        self.assertGreater(
            abs(result.group_times_s[1] / dt_s - round(result.group_times_s[1] / dt_s)),
            1e-3,
        )
        for array in (
            result.periods_s,
            result.velocity_axis_km_s,
            result.group_times_s,
            result.scaled_log_energy,
        ):
            self.assertFalse(array.flags.writeable)

    def test_public_phase_curve_uses_fixed_3p5_anchor_on_formal_51_point_grid(self):
        periods = np.arange(250, 501, 5, dtype=float) / 100.0
        phase_time = 7.0 - 0.4 * periods
        principal = self._principal_phase(
            2.0 * np.pi * phase_time / periods
        )
        group_times = np.zeros(periods.size)
        anchor_3p5 = self.mod.unwrap_phase_along_frequency(
            periods,
            principal,
            group_times,
            phase_time_sign=1,
            anchor_period_s=3.5,
        )
        anchor_3p75 = self.mod.unwrap_phase_along_frequency(
            periods,
            principal,
            group_times,
            phase_time_sign=1,
            anchor_period_s=3.75,
        )
        self.assertEqual(anchor_3p5.status, "accepted")
        self.assertEqual(anchor_3p75.status, "accepted")
        np.testing.assert_allclose(
            anchor_3p75.unwrapped_phase_rad
            - anchor_3p5.unwrapped_phase_rad,
            2.0 * np.pi,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            anchor_3p75.raw_phase_time_s
            - anchor_3p5.raw_phase_time_s,
            periods,
            rtol=0.0,
            atol=2e-14,
        )

        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        dt_s = 0.02
        time_s = np.arange(-10.0, 30.0, dt_s)
        formal_group_time_s = 8.0
        envelope = np.exp(
            -0.5 * ((time_s - formal_group_time_s) / 1.0) ** 2
        )
        analytic = np.vstack(
            [
                envelope
                * np.exp(
                    1j
                    * (
                        2.0
                        * np.pi
                        * (time_s - formal_group_time_s)
                        / period
                        + 0.2
                    )
                )
                for period in periods
            ]
        )
        bank = self.mod.GaussianFilterBankResult(
            filtered_waveforms=analytic.real,
            analytic_signals=analytic,
            envelope=np.abs(analytic),
        )
        trace = self.mod.DatTrace(
            "fixed_anchor",
            20.0,
            dt_s,
            time_s,
            np.zeros_like(time_s),
            np.zeros_like(time_s),
            np.zeros_like(time_s),
            0.0,
            0.0,
            0.1,
            0.1,
        )
        completed_stages = []
        with (
            mock.patch.object(
                self.mod,
                "gaussian_filter_bank",
                return_value=bank,
            ),
            mock.patch.object(
                self.mod,
                "unwrap_phase_along_frequency",
                wraps=self.mod.unwrap_phase_along_frequency,
            ) as unwrap_spy,
        ):
            result = self.mod.measure_phase_curve(
                trace,
                periods_s=periods,
                velocity_axis_km_s=(
                    self.mod.FtanConfig().group_velocities_km_s
                ),
                alpha=20.0,
                beta1=0.5,
                beta2=1.0,
                convention=convention,
                stage_callback=completed_stages.append,
            )

        self.assertIsNotNone(result)
        self.assertEqual(
            completed_stages,
            [
                "filter_bank",
                "dp_ridge",
                "group_arrival_phase_instantaneous_frequency",
                "phase_unwrap",
            ],
        )
        self.assertEqual(unwrap_spy.call_count, 1)
        self.assertEqual(unwrap_spy.call_args.kwargs["anchor_period_s"], 3.5)
        ridge_rows = result.ridge.row_indices
        np.testing.assert_array_equal(
            result.ridge_normalized_log_energy,
            result.scaled_log_energy[
                np.arange(periods.size),
                ridge_rows,
            ],
        )
        sample_times = trace.distance_km / result.velocity_axis_km_s
        amplitude_image = np.vstack(
            [
                np.interp(
                    sample_times,
                    time_s,
                    envelope_row,
                    left=0.0,
                    right=0.0,
                )
                for envelope_row in bank.envelope
            ]
        )
        amplitude_max = np.max(amplitude_image, axis=1, keepdims=True)
        normalized_amplitude = np.zeros_like(amplitude_image)
        np.divide(
            amplitude_image,
            amplitude_max,
            out=normalized_amplitude,
            where=amplitude_max > 0,
        )
        np.testing.assert_array_equal(
            result.ridge_normalized_envelope_amplitude,
            normalized_amplitude[
                np.arange(periods.size),
                ridge_rows,
            ],
        )
        expected_jump = np.zeros(periods.size, dtype=float)
        expected_jump[1:] = np.abs(
            np.diff(result.ridge.group_velocities_km_s)
        )
        np.testing.assert_array_equal(
            result.ridge_adjacent_jump_km_s,
            expected_jump,
        )
        for values in (
            result.ridge_normalized_log_energy,
            result.ridge_normalized_envelope_amplitude,
            result.ridge_adjacent_jump_km_s,
        ):
            self.assertFalse(values.flags.writeable)
        measured_periods = np.array(
            [
                measurement.period_s
                for measurement in result.measurements
            ]
        )
        anchor_index = int(result.phase_unwrap.anchor_index)
        self.assertEqual(
            anchor_index,
            int(np.argmin(np.abs(measured_periods - 3.5))),
        )
        with (
            mock.patch.object(
                self.mod,
                "gaussian_filter_bank",
                return_value=bank,
            ),
            mock.patch.object(
                self.mod,
                "_phase_measurement_snr",
                return_value=float("nan"),
            ),
        ):
            no_noise_window = self.mod.measure_phase_curve(
                trace,
                periods_s=periods,
                velocity_axis_km_s=(
                    self.mod.FtanConfig().group_velocities_km_s
                ),
                alpha=20.0,
                beta1=0.5,
                beta2=1.0,
                convention=convention,
            )
        self.assertIsNotNone(no_noise_window)
        self.assertTrue(
            all(row.snr == 0.0 for row in no_noise_window.measurements)
        )

        original_frequency = self.mod.estimate_instantaneous_frequency

        duplicate_nominal = (2.9, 3.0, 3.1)

        def duplicate_three_periods(*args, **kwargs):
            frequency = original_frequency(*args, **kwargs)
            if any(
                math.isclose(kwargs["nominal_period_s"], value)
                for value in duplicate_nominal
            ):
                return replace(
                    frequency,
                    fitted_phase_slope_rad_s=2.0 * np.pi / 3.0,
                    omega_inst_rad_s=2.0 * np.pi / 3.0,
                    instantaneous_period_s=3.0,
                    period_ratio=3.0 / kwargs["nominal_period_s"],
                    status="accepted",
                )
            return frequency

        with (
            mock.patch.object(
                self.mod,
                "gaussian_filter_bank",
                return_value=bank,
            ),
            mock.patch.object(
                self.mod,
                "estimate_instantaneous_frequency",
                side_effect=duplicate_three_periods,
            ),
        ):
            duplicate_curve = self.mod.measure_phase_curve(
                trace,
                periods_s=periods,
                velocity_axis_km_s=(
                    self.mod.FtanConfig().group_velocities_km_s
                ),
                alpha=20.0,
                beta1=0.5,
                beta2=1.0,
                convention=convention,
            )

        self.assertIsNotNone(duplicate_curve)
        duplicate_indices = np.asarray(
            [
                int(np.flatnonzero(periods == value)[0])
                for value in duplicate_nominal
            ],
            dtype=int,
        )
        self.assertEqual(
            tuple(
                duplicate_curve.measurement_statuses[index]
                for index in duplicate_indices
            ),
            ("duplicate_instantaneous_period",) * 3,
        )
        self.assertTrue(
            all(
                duplicate_curve.measurements[index] is None
                for index in duplicate_indices
            )
        )
        np.testing.assert_array_equal(
            duplicate_curve.instantaneous_periods_s[duplicate_indices],
            np.full(3, 3.0),
        )
        duplicate_targets = self.mod.resample_wang_measurements(
            duplicate_curve.measurements,
            time_s=time_s,
            distance_km=trace.distance_km,
            target_periods_s=(4.0,),
            nominal_periods_s=duplicate_curve.periods_s,
            measurement_statuses=duplicate_curve.measurement_statuses,
            instantaneous_periods_s=(
                duplicate_curve.instantaneous_periods_s
            ),
            ridge_normalized_log_energy=(
                duplicate_curve.ridge_normalized_log_energy
            ),
            ridge_normalized_envelope_amplitude=(
                duplicate_curve.ridge_normalized_envelope_amplitude
            ),
            ridge_adjacent_jump_km_s=(
                duplicate_curve.ridge_adjacent_jump_km_s
            ),
            valid_mask=duplicate_curve.measurement_valid,
        )
        self.assertEqual(duplicate_targets[0].status, "accepted")
        self.assertEqual(
            duplicate_targets[0].continuous_rejection_statuses[
                duplicate_targets[0].continuous_rejection_statuses.index(
                    "duplicate_instantaneous_period"
                )
            ],
            "duplicate_instantaneous_period",
        )

        def reject_one_frequency(*args, **kwargs):
            frequency = original_frequency(*args, **kwargs)
            if math.isclose(kwargs["nominal_period_s"], 4.0):
                return replace(
                    frequency,
                    status="invalid_instantaneous_frequency",
                )
            return frequency

        with (
            mock.patch.object(
                self.mod,
                "gaussian_filter_bank",
                return_value=bank,
            ),
            mock.patch.object(
                self.mod,
                "estimate_instantaneous_frequency",
                side_effect=reject_one_frequency,
            ),
        ):
            partial = self.mod.measure_phase_curve(
                trace,
                periods_s=periods,
                velocity_axis_km_s=(
                    self.mod.FtanConfig().group_velocities_km_s
                ),
                alpha=20.0,
                beta1=0.5,
                beta2=1.0,
                convention=convention,
            )

        self.assertIsNotNone(partial)
        invalid_index = int(np.flatnonzero(periods == 4.0)[0])
        valid_three_index = int(np.flatnonzero(periods == 3.0)[0])
        self.assertIsNone(partial.measurements[invalid_index])
        self.assertTrue(partial.measurement_valid[valid_three_index])
        self.assertFalse(partial.measurement_valid[invalid_index])
        self.assertEqual(
            partial.measurement_statuses[invalid_index],
            "invalid_instantaneous_frequency",
        )
        self.assertTrue(
            np.isnan(partial.instantaneous_periods_s[invalid_index])
        )
        self.assertFalse(partial.measurement_valid.flags.writeable)
        target_rows = self.mod.resample_wang_measurements(
            partial.measurements,
            time_s=time_s,
            distance_km=trace.distance_km,
            target_periods_s=(3.0, 4.0),
            nominal_periods_s=partial.periods_s,
            measurement_statuses=partial.measurement_statuses,
            instantaneous_periods_s=partial.instantaneous_periods_s,
            ridge_normalized_log_energy=(
                partial.ridge_normalized_log_energy
            ),
            ridge_normalized_envelope_amplitude=(
                partial.ridge_normalized_envelope_amplitude
            ),
            ridge_adjacent_jump_km_s=(
                partial.ridge_adjacent_jump_km_s
            ),
            valid_mask=partial.measurement_valid,
        )
        self.assertEqual(target_rows[0].status, "accepted")
        self.assertEqual(
            target_rows[1].status,
            "target_period_not_bracketed",
        )

        def reject_nominal_anchor(*args, **kwargs):
            frequency = original_frequency(*args, **kwargs)
            if math.isclose(kwargs["nominal_period_s"], 3.5):
                return replace(
                    frequency,
                    status="invalid_instantaneous_frequency",
                )
            return frequency

        with (
            mock.patch.object(
                self.mod,
                "gaussian_filter_bank",
                return_value=bank,
            ),
            mock.patch.object(
                self.mod,
                "estimate_instantaneous_frequency",
                side_effect=reject_nominal_anchor,
            ),
        ):
            partial_anchor = self.mod.measure_phase_curve(
                trace,
                periods_s=periods,
                velocity_axis_km_s=(
                    self.mod.FtanConfig().group_velocities_km_s
                ),
                alpha=20.0,
                beta1=0.5,
                beta2=1.0,
                convention=convention,
            )
        anchor_nominal_index = int(np.flatnonzero(periods == 3.5)[0])
        self.assertIsNotNone(partial_anchor)
        self.assertIsNone(
            partial_anchor.measurements[anchor_nominal_index]
        )
        self.assertEqual(
            partial_anchor.measurement_statuses[anchor_nominal_index],
            "invalid_instantaneous_frequency",
        )
        self.assertTrue(
            np.isnan(
                partial_anchor.instantaneous_periods_s[
                    anchor_nominal_index
                ]
            )
        )
        self.assertTrue(np.any(partial_anchor.measurement_valid))
        anchor_targets = self.mod.resample_wang_measurements(
            partial_anchor.measurements,
            time_s=time_s,
            distance_km=trace.distance_km,
            target_periods_s=(3.0, 4.0),
            nominal_periods_s=partial_anchor.periods_s,
            measurement_statuses=partial_anchor.measurement_statuses,
            instantaneous_periods_s=partial_anchor.instantaneous_periods_s,
            ridge_normalized_log_energy=(
                partial_anchor.ridge_normalized_log_energy
            ),
            ridge_normalized_envelope_amplitude=(
                partial_anchor.ridge_normalized_envelope_amplitude
            ),
            ridge_adjacent_jump_km_s=(
                partial_anchor.ridge_adjacent_jump_km_s
            ),
            valid_mask=partial_anchor.measurement_valid,
        )
        self.assertTrue(any(row.status == "accepted" for row in anchor_targets))

        ridge_valid = np.array(result.ridge.valid, copy=True)
        ridge_valid[invalid_index] = False
        ridge_with_gap = replace(result.ridge, valid=ridge_valid)
        with (
            mock.patch.object(
                self.mod,
                "gaussian_filter_bank",
                return_value=bank,
            ),
            mock.patch.object(
                self.mod,
                "select_fundamental_ridge",
                return_value=ridge_with_gap,
            ),
        ):
            partial_ridge = self.mod.measure_phase_curve(
                trace,
                periods_s=periods,
                velocity_axis_km_s=(
                    self.mod.FtanConfig().group_velocities_km_s
                ),
                alpha=20.0,
                beta1=0.5,
                beta2=1.0,
                convention=convention,
            )
        self.assertIsNotNone(partial_ridge)
        self.assertIsNone(partial_ridge.measurements[invalid_index])
        self.assertEqual(
            partial_ridge.measurement_statuses[invalid_index],
            "ridge_invalid",
        )
        self.assertTrue(
            np.isnan(partial_ridge.instantaneous_periods_s[invalid_index])
        )
        self.assertTrue(partial_ridge.measurement_valid[valid_three_index])

        original_unwrap = self.mod.unwrap_phase_along_frequency
        unwrap_call_count = 0

        def reject_one_unwrap_column(*args, **kwargs):
            nonlocal unwrap_call_count
            candidate = original_unwrap(*args, **kwargs)
            unwrap_call_count += 1
            if unwrap_call_count != 1:
                return candidate
            errors = np.array(candidate.prediction_error_s, copy=True)
            bad_local = int(np.argmin(np.abs(np.asarray(args[0]) - 4.0)))
            errors[bad_local] = float(args[0][bad_local])
            return replace(
                candidate,
                prediction_error_s=errors,
                status="phase_unwrap_discontinuous",
            )

        with (
            mock.patch.object(
                self.mod,
                "gaussian_filter_bank",
                return_value=bank,
            ),
            mock.patch.object(
                self.mod,
                "unwrap_phase_along_frequency",
                side_effect=reject_one_unwrap_column,
            ),
        ):
            partial_unwrap = self.mod.measure_phase_curve(
                trace,
                periods_s=periods,
                velocity_axis_km_s=(
                    self.mod.FtanConfig().group_velocities_km_s
                ),
                alpha=20.0,
                beta1=0.5,
                beta2=1.0,
                convention=convention,
            )
        self.assertGreaterEqual(unwrap_call_count, 2)
        self.assertIsNotNone(partial_unwrap)
        self.assertIsNone(partial_unwrap.measurements[invalid_index])
        self.assertEqual(
            partial_unwrap.measurement_statuses[invalid_index],
            "phase_unwrap_discontinuous",
        )
        self.assertTrue(
            np.isfinite(
                partial_unwrap.instantaneous_periods_s[invalid_index]
            )
        )
        self.assertFalse(
            partial_unwrap.instantaneous_periods_s.flags.writeable
        )
        self.assertTrue(partial_unwrap.measurement_valid[valid_three_index])
        partial_unwrap_targets = self.mod.resample_wang_measurements(
            partial_unwrap.measurements,
            time_s=time_s,
            distance_km=trace.distance_km,
            target_periods_s=(3.0,),
            nominal_periods_s=partial_unwrap.periods_s,
            measurement_statuses=partial_unwrap.measurement_statuses,
            instantaneous_periods_s=(
                partial_unwrap.instantaneous_periods_s
            ),
            ridge_normalized_log_energy=(
                partial_unwrap.ridge_normalized_log_energy
            ),
            ridge_normalized_envelope_amplitude=(
                partial_unwrap.ridge_normalized_envelope_amplitude
            ),
            ridge_adjacent_jump_km_s=(
                partial_unwrap.ridge_adjacent_jump_km_s
            ),
            valid_mask=partial_unwrap.measurement_valid,
        )
        audit = partial_unwrap_targets[0]
        audit_index = audit.continuous_rejection_statuses.index(
            "phase_unwrap_discontinuous"
        )
        self.assertEqual(
            audit.rejected_continuous_nominal_periods_s[audit_index],
            periods[invalid_index],
        )
        self.assertEqual(
            audit.rejected_continuous_instantaneous_periods_s[
                audit_index
            ],
            partial_unwrap.instantaneous_periods_s[invalid_index],
        )

    def test_public_phase_measurement_apis_require_explicit_convention(self):
        for name in (
            "measure_phase_curve",
            "measure_single_period",
            "compute_phase_speed_candidates",
        ):
            function = self.require_module_attribute(name)
            self.assertIs(
                inspect.signature(function).parameters["convention"].default,
                inspect.Parameter.empty,
            )
        wrapper_source = inspect.getsource(self.mod.measure_single_period)
        for duplicate_math in (
            "gaussian_narrowband_waveform",
            "hilbert(",
            "np.gradient",
            "np.argmax",
        ):
            self.assertNotIn(duplicate_math, wrapper_source)
        self.assertIn("measure_phase_curve(", wrapper_source)

    def test_measurement_bound_candidates_use_and_validate_convention(self):
        from_measurement = self.require_module_attribute(
            "phase_speed_candidates_from_measurement"
        )
        lin = self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF
        bensen = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        measurement = self.mod.PeriodMeasurement(
            convention=lin,
            period_s=3.0,
            omega_inst_rad_s=2.0 * np.pi / 3.0,
            principal_paper_phase_rad=0.4,
            unwrapped_paper_phase_rad=0.4,
            raw_phase_time_s=self.mod.raw_phase_travel_time(
                convention=lin,
                group_time_s=8.0,
                phase_rad=0.4,
                omega_rad_s=2.0 * np.pi / 3.0,
            ),
            paper_phase_cycle_offset=0,
            group_time_s=8.0,
            group_velocity_km_s=2.5,
            snr=10.0,
            signal_window_start_s=4.0,
            signal_window_end_s=12.0,
            filtered_waveform=np.ones(8),
            envelope=np.ones(8),
        )
        lin_rows = from_measurement(
            measurement,
            distance_km=20.0,
            branch_min=1,
            branch_max=1,
            velocity_bounds=(0.1, 20.0),
        )
        direct = self.mod.compute_phase_speed_candidates(
            phi_tu=measurement.phi_tu_rad,
            omega=measurement.omega_inst_rad_s,
            distance_km=20.0,
            group_velocity_km_s=measurement.group_velocity_km_s,
            convention=lin,
            branch_min=1,
            branch_max=1,
            velocity_bounds=(0.1, 20.0),
        )
        self.assertEqual(lin_rows, direct)
        with self.assertRaisesRegex(ValueError, "convention"):
            from_measurement(
                measurement,
                distance_km=20.0,
                convention=bensen,
            )

    def test_measurement_bound_candidates_preserve_anchored_raw_phase_branch(self):
        period_s = 4.0
        omega = 2.0 * np.pi / period_s
        group_time_s = 10.0
        distance_km = 30.0
        principal_paper_phase = 0.2
        paper_cycle_offset = 1
        unwrapped_paper_phase = (
            principal_paper_phase
            + 2.0 * np.pi * paper_cycle_offset
        )
        for convention in self.mod.PhaseConvention:
            with self.subTest(convention=convention.name):
                principal_raw_time = self.mod.raw_phase_travel_time(
                    convention=convention,
                    group_time_s=group_time_s,
                    phase_rad=principal_paper_phase,
                    omega_rad_s=omega,
                )
                anchored_raw_time = self.mod.raw_phase_travel_time(
                    convention=convention,
                    group_time_s=group_time_s,
                    phase_rad=unwrapped_paper_phase,
                    omega_rad_s=omega,
                )
                self.assertAlmostEqual(
                    anchored_raw_time - principal_raw_time,
                    period_s,
                )
                measurement = self.mod.PeriodMeasurement(
                    convention=convention,
                    period_s=period_s,
                    omega_inst_rad_s=omega,
                    principal_paper_phase_rad=principal_paper_phase,
                    unwrapped_paper_phase_rad=unwrapped_paper_phase,
                    raw_phase_time_s=anchored_raw_time,
                    paper_phase_cycle_offset=paper_cycle_offset,
                    group_time_s=group_time_s,
                    group_velocity_km_s=distance_km / group_time_s,
                    snr=10.0,
                    signal_window_start_s=5.0,
                    signal_window_end_s=15.0,
                    filtered_waveform=np.ones(16),
                    envelope=np.ones(16),
                )
                anchored = (
                    self.mod.phase_speed_candidates_from_measurement(
                        measurement,
                        distance_km=distance_km,
                        branch_min=0,
                        branch_max=1,
                        velocity_bounds=(0.1, 20.0),
                    )
                )
                by_branch = {row.branch: row for row in anchored}
                self.assertAlmostEqual(
                    distance_km / by_branch[0].phase_velocity_km_s,
                    anchored_raw_time,
                )
                self.assertAlmostEqual(
                    distance_km / by_branch[1].phase_velocity_km_s,
                    anchored_raw_time
                    + convention.definition.cycle_phase_sign * period_s,
                )
                legacy = self.mod.compute_phase_speed_candidates(
                    phi_tu=principal_paper_phase,
                    omega=omega,
                    distance_km=distance_km,
                    group_velocity_km_s=distance_km / group_time_s,
                    convention=convention,
                    branch_min=0,
                    branch_max=0,
                    velocity_bounds=(0.1, 20.0),
                )
                self.assertEqual(len(legacy), 1)
                legacy_time = (
                    distance_km / legacy[0].phase_velocity_km_s
                )
                self.assertAlmostEqual(
                    anchored_raw_time - legacy_time,
                    period_s,
                )
                self.assertNotAlmostEqual(
                    by_branch[0].phase_velocity_km_s,
                    legacy[0].phase_velocity_km_s,
                )

    def test_measurement_candidates_recover_positive_time_from_nonpositive_raw_anchor(self):
        period_s = 5.0
        omega = 2.0 * np.pi / period_s
        group_time_s = 1.6
        distance_km = 10.0
        paper_phase = -np.pi
        for convention, recovery_branch in (
            (
                self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
                1,
            ),
            (
                self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF,
                -1,
            ),
        ):
            with self.subTest(convention=convention.name):
                raw_time = self.mod.raw_phase_travel_time(
                    convention=convention,
                    group_time_s=group_time_s,
                    phase_rad=paper_phase,
                    omega_rad_s=omega,
                )
                self.assertAlmostEqual(raw_time, -1.525)
                measurement = self.mod.PeriodMeasurement(
                    convention=convention,
                    period_s=period_s,
                    omega_inst_rad_s=omega,
                    principal_paper_phase_rad=np.pi,
                    unwrapped_paper_phase_rad=paper_phase,
                    raw_phase_time_s=raw_time,
                    paper_phase_cycle_offset=-1,
                    group_time_s=group_time_s,
                    group_velocity_km_s=distance_km / group_time_s,
                    snr=8.0,
                    signal_window_start_s=0.0,
                    signal_window_end_s=5.0,
                    filtered_waveform=np.ones(8),
                    envelope=np.ones(8),
                )
                candidates = (
                    self.mod.phase_speed_candidates_from_measurement(
                        measurement,
                        distance_km=distance_km,
                        branch_min=-1,
                        branch_max=1,
                        velocity_bounds=(0.1, 20.0),
                    )
                )
                self.assertEqual(
                    [row.branch for row in candidates],
                    [recovery_branch],
                )
                recovered_time = (
                    distance_km / candidates[0].phase_velocity_km_s
                )
                self.assertAlmostEqual(recovered_time, raw_time + period_s)
                self.assertGreater(recovered_time, 0.0)

    def test_principal_phase_canonicalizer_maps_exact_negative_axis_to_positive_pi(self):
        canonical = self.require_module_attribute(
            "_canonical_principal_phase_rad"
        )
        for phase in (-3.0 * np.pi, -np.pi, np.pi, 3.0 * np.pi):
            with self.subTest(phase=phase):
                self.assertEqual(canonical(phase), np.pi)

        time_s = np.array([0.0, 1.0])
        exact_negative_real = np.array(
            [complex(-1.0, 0.0), complex(-1.0, 0.0)]
        )
        self.assertEqual(
            self.mod.interpolate_analytic_phase_at_arrival(
                time_s,
                exact_negative_real,
                0.5,
            ),
            np.pi,
        )
        for convention in self.mod.PhaseConvention:
            with self.subTest(convention=convention.name):
                self.assertEqual(
                    self.mod.interpolate_analytic_phase_at_arrival(
                        time_s,
                        exact_negative_real,
                        0.5,
                        convention=convention,
                    ),
                    np.pi,
                )

    def test_period_measurement_requires_positive_pi_boundary_and_preserves_anchored_t0(self):
        period_s = 5.0
        omega = 2.0 * np.pi / period_s
        group_time_s = 10.0
        distance_km = 20.0
        for convention in self.mod.PhaseConvention:
            with self.subTest(convention=convention.name):
                raw_time = self.mod.raw_phase_travel_time(
                    convention=convention,
                    group_time_s=group_time_s,
                    phase_rad=-np.pi,
                    omega_rad_s=omega,
                )
                arguments = {
                    "convention": convention,
                    "period_s": period_s,
                    "omega_inst_rad_s": omega,
                    "unwrapped_paper_phase_rad": -np.pi,
                    "raw_phase_time_s": raw_time,
                    "group_time_s": group_time_s,
                    "group_velocity_km_s": distance_km / group_time_s,
                    "snr": 8.0,
                    "signal_window_start_s": 5.0,
                    "signal_window_end_s": 15.0,
                    "filtered_waveform": np.ones(8),
                    "envelope": np.ones(8),
                }
                with self.assertRaisesRegex(ValueError, "principal"):
                    self.mod.PeriodMeasurement(
                        principal_paper_phase_rad=-np.pi,
                        paper_phase_cycle_offset=0,
                        **arguments,
                    )
                measurement = self.mod.PeriodMeasurement(
                    principal_paper_phase_rad=np.pi,
                    paper_phase_cycle_offset=-1,
                    **arguments,
                )
                self.assertAlmostEqual(raw_time, 6.875)
                candidates = (
                    self.mod.phase_speed_candidates_from_measurement(
                        measurement,
                        distance_km=distance_km,
                        branch_min=0,
                        branch_max=0,
                        velocity_bounds=(0.1, 20.0),
                    )
                )
                self.assertEqual(len(candidates), 1)
                self.assertAlmostEqual(
                    distance_km / candidates[0].phase_velocity_km_s,
                    6.875,
                )
                principal_raw = self.mod.raw_phase_travel_time(
                    convention=convention,
                    group_time_s=group_time_s,
                    phase_rad=np.pi,
                    omega_rad_s=omega,
                )
                self.assertAlmostEqual(principal_raw, 11.875)
                self.assertAlmostEqual(principal_raw - raw_time, period_s)

    def test_formal_curve_canonicalizes_exact_negative_real_phase_without_t0_split(self):
        periods = np.array([4.9, 5.0, 5.1])
        dt_s = 0.02
        time_s = np.arange(-10.0, 30.0, dt_s)
        center = int(np.argmin(np.abs(time_s - 10.0)))
        group_time_s = float(time_s[center])
        distance_km = 30.0
        envelope = np.exp(
            -0.5 * ((time_s - group_time_s) / 1.0) ** 2
        )
        analytic = np.vstack(
            [
                envelope
                * np.exp(
                    1j
                    * (
                        2.0
                        * np.pi
                        * (time_s - group_time_s)
                        / period
                        + np.pi
                    )
                )
                for period in periods
            ]
        )
        analytic[:, center] = complex(-float(envelope[center]), 0.0)
        bank = self.mod.GaussianFilterBankResult(
            filtered_waveforms=analytic.real,
            analytic_signals=analytic,
            envelope=np.abs(analytic),
        )
        trace = self.mod.DatTrace(
            "negative_real_boundary",
            distance_km,
            dt_s,
            time_s,
            np.zeros_like(time_s),
            np.zeros_like(time_s),
            np.zeros_like(time_s),
            0.0,
            0.0,
            0.1,
            0.1,
        )
        for convention in self.mod.PhaseConvention:
            with self.subTest(convention=convention.name):
                with mock.patch.object(
                    self.mod,
                    "gaussian_filter_bank",
                    return_value=bank,
                ):
                    curve = self.mod.measure_phase_curve(
                        trace,
                        periods_s=periods,
                        velocity_axis_km_s=(
                            self.mod.FtanConfig().group_velocities_km_s
                        ),
                        alpha=20.0,
                        beta1=0.5,
                        beta2=1.0,
                        convention=convention,
                    )
                self.assertIsNotNone(curve)
                center_row = curve.measurements[1]
                self.assertEqual(
                    center_row.principal_paper_phase_rad,
                    np.pi,
                )
                self.assertAlmostEqual(
                    center_row.unwrapped_paper_phase_rad,
                    -np.pi,
                )
                self.assertEqual(center_row.paper_phase_cycle_offset, -1)
                self.assertAlmostEqual(center_row.raw_phase_time_s, 6.875)
                candidate = (
                    self.mod.phase_speed_candidates_from_measurement(
                        center_row,
                        distance_km=distance_km,
                        branch_min=0,
                        branch_max=0,
                        velocity_bounds=(0.1, 20.0),
                    )[0]
                )
                self.assertAlmostEqual(
                    distance_km / candidate.phase_velocity_km_s,
                    6.875,
                )

    def test_extraction_measures_one_continuous_curve_including_fixed_anchor(self):
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        trace = self.mod.DatTrace(
            "one_curve",
            30.0,
            0.1,
            np.arange(8.0),
            np.zeros(8),
            np.zeros(8),
            np.zeros(8),
            0.0,
            0.0,
            0.1,
            0.1,
        )

        def fake_curve(_trace, *, periods_s, convention, **_kwargs):
            rows = []
            for period in periods_s:
                omega = 2.0 * np.pi / period
                raw_time = self.mod.raw_phase_travel_time(
                    convention=convention,
                    group_time_s=10.0,
                    phase_rad=0.2,
                    omega_rad_s=omega,
                )
                rows.append(
                    self.mod.PeriodMeasurement(
                        convention=convention,
                        period_s=float(period),
                        omega_inst_rad_s=omega,
                        principal_paper_phase_rad=0.2,
                        unwrapped_paper_phase_rad=0.2,
                        raw_phase_time_s=raw_time,
                        paper_phase_cycle_offset=0,
                        group_time_s=10.0,
                        group_velocity_km_s=3.0,
                        snr=0.0,
                        signal_window_start_s=5.0,
                        signal_window_end_s=15.0,
                        filtered_waveform=np.ones(8),
                        envelope=np.ones(8),
                    )
                )
            return types.SimpleNamespace(
                periods_s=np.asarray(periods_s),
                measurements=tuple(rows),
            )

        for requested in (np.array([3.0, 4.0]), np.array([3.5])):
            with self.subTest(requested=requested):
                with (
                    mock.patch.object(
                        self.mod,
                        "read_dat_trace",
                        return_value=trace,
                    ),
                    mock.patch.object(
                        self.mod,
                        "measure_phase_curve",
                        side_effect=fake_curve,
                    ) as measure_curve,
                    mock.patch.object(
                        self.mod,
                        "measure_single_period",
                    ) as measure_single,
                ):
                    result = self.mod.extract_bensen_phase_curve(
                        "unused.dat",
                        periods_s=requested,
                    )

                self.assertEqual(measure_curve.call_count, 1)
                measure_single.assert_not_called()
                measured_periods = np.asarray(
                    measure_curve.call_args.kwargs["periods_s"]
                )
                self.assertGreaterEqual(measured_periods.size, 3)
                self.assertTrue(np.any(measured_periods == 3.5))
                self.assertTrue(np.all(np.diff(measured_periods) > 0))
                self.assertLessEqual(
                    float(np.max(np.diff(measured_periods))),
                    0.05 + 1e-12,
                )
                for target, measurement in zip(
                    requested,
                    result.measurements,
                ):
                    self.assertIsNotNone(measurement)
                    self.assertEqual(measurement.period_s, target)

    def test_measurement_records_strictly_validate_manual_construction(self):
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        periods = np.array([3.0, 3.5, 4.0])
        group_times = np.full(3, 10.0)

        def make_row(period, **overrides):
            omega = 2.0 * np.pi / period
            values = {
                "convention": convention,
                "period_s": period,
                "omega_inst_rad_s": omega,
                "principal_paper_phase_rad": 0.2,
                "unwrapped_paper_phase_rad": 0.2,
                "raw_phase_time_s": self.mod.raw_phase_travel_time(
                    convention=convention,
                    group_time_s=10.0,
                    phase_rad=0.2,
                    omega_rad_s=omega,
                ),
                "paper_phase_cycle_offset": 0,
                "group_time_s": 10.0,
                "group_velocity_km_s": 3.0,
                "snr": 8.0,
                "signal_window_start_s": 5.0,
                "signal_window_end_s": 15.0,
                "filtered_waveform": np.ones(8),
                "envelope": np.ones(8),
            }
            values.update(overrides)
            return self.mod.PeriodMeasurement(**values)

        rows = tuple(make_row(period) for period in periods)
        ridge = self.mod.RidgeResult(
            row_indices=np.ones(3, dtype=int),
            group_velocities_km_s=np.full(3, 3.0),
            valid=np.ones(3, dtype=bool),
            wang_group_limit_pass_count=3,
            quality=self.mod.RidgeQuality(
                True,
                "accepted",
                1.0,
                0,
                0.0,
                0.0,
                1.0,
            ),
        )
        unwrap = self.mod.PhaseUnwrapResult(
            unwrapped_phase_rad=np.full(3, -0.2),
            cycle_counts=np.zeros(3, dtype=int),
            raw_phase_time_s=np.array(
                [row.raw_phase_time_s for row in rows]
            ),
            prediction_error_s=np.zeros(3),
            sort_order=np.arange(3),
            valid_mask=np.ones(3, dtype=bool),
            anomaly_fraction=0.0,
            max_consecutive_anomalies=0,
            anchor_index=1,
            status="accepted",
        )

        def make_curve(**overrides):
            values = {
                "convention": convention,
                "periods_s": periods,
                "velocity_axis_km_s": np.array([2.0, 3.0, 4.0]),
                "group_times_s": group_times,
                "scaled_log_energy": np.ones((3, 3)),
                "measurements": rows,
                "ridge": ridge,
                "phase_unwrap": unwrap,
                "instantaneous_periods_s": periods,
                "ridge_normalized_log_energy": np.array([0.6, 0.7, 0.8]),
                "ridge_normalized_envelope_amplitude": np.array(
                    [0.7, 0.8, 0.9]
                ),
                "ridge_adjacent_jump_km_s": np.array([0.0, 0.1, 0.2]),
                "status": "accepted",
            }
            values.update(overrides)
            return self.mod.PhaseCurveMeasurement(**values)

        curve = make_curve()
        for array in (
            curve.periods_s,
            curve.velocity_axis_km_s,
            curve.group_times_s,
            curve.scaled_log_energy,
            curve.ridge_normalized_log_energy,
            curve.ridge_normalized_envelope_amplitude,
            curve.ridge_adjacent_jump_km_s,
            curve.instantaneous_periods_s,
            rows[0].filtered_waveform,
            rows[0].envelope,
        ):
            self.assertFalse(array.flags.writeable)

        with self.assertRaises(ValueError):
            make_row(
                3.0,
                filtered_waveform=np.ones(8, dtype=complex),
            )
        with self.assertRaises(ValueError):
            make_row(3.0, unwrapped_paper_phase_rad=0.3)
        with self.assertRaises(ValueError):
            make_curve(
                scaled_log_energy=np.ones((3, 3), dtype=complex)
            )
        with self.assertRaisesRegex(
            ValueError,
            "ridge_normalized_log_energy",
        ):
            make_curve(
                ridge_normalized_log_energy=np.array([0.6, 1.1, 0.8])
            )
        with self.assertRaisesRegex(
            ValueError,
            "ridge_normalized_envelope_amplitude",
        ):
            make_curve(
                ridge_normalized_envelope_amplitude=np.array(
                    [0.7, 0.8, 1.1]
                )
            )
        with self.assertRaises(ValueError):
            make_curve(
                measurements=(make_row(3.1), rows[1], rows[2])
            )
        with self.assertRaises(ValueError):
            make_curve(measurements=(object(), rows[1], rows[2]))
        contradictory_unwrap = self.mod.PhaseUnwrapResult(
            unwrapped_phase_rad=unwrap.unwrapped_phase_rad,
            cycle_counts=np.ones(3, dtype=int),
            raw_phase_time_s=unwrap.raw_phase_time_s,
            prediction_error_s=unwrap.prediction_error_s,
            sort_order=unwrap.sort_order,
            valid_mask=unwrap.valid_mask,
            anomaly_fraction=unwrap.anomaly_fraction,
            max_consecutive_anomalies=unwrap.max_consecutive_anomalies,
            anchor_index=unwrap.anchor_index,
            status=unwrap.status,
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            make_curve(phase_unwrap=contradictory_unwrap)

        with self.assertRaisesRegex(ValueError, "status"):
            make_curve(
                measurement_statuses=(
                    "invalid_instantaneous_frequency",
                    "accepted",
                    "accepted",
                )
            )
        partial_unwrapped = np.array(
            unwrap.unwrapped_phase_rad,
            copy=True,
        )
        partial_raw_time = np.array(
            unwrap.raw_phase_time_s,
            copy=True,
        )
        partial_unwrapped[0] = np.nan
        partial_raw_time[0] = np.nan
        partial_valid = np.array([False, True, True])
        partial_unwrap = self.mod.PhaseUnwrapResult(
            unwrapped_phase_rad=partial_unwrapped,
            cycle_counts=unwrap.cycle_counts,
            raw_phase_time_s=partial_raw_time,
            prediction_error_s=unwrap.prediction_error_s,
            sort_order=unwrap.sort_order,
            valid_mask=partial_valid,
            anomaly_fraction=unwrap.anomaly_fraction,
            max_consecutive_anomalies=unwrap.max_consecutive_anomalies,
            anchor_index=unwrap.anchor_index,
            status="partial_phase_unwrap",
        )
        partial_values = {
            "measurements": (None, rows[1], rows[2]),
            "measurement_valid": partial_valid,
            "phase_unwrap": partial_unwrap,
            "status": "partial_phase_curve",
        }
        with self.assertRaisesRegex(ValueError, "status"):
            make_curve(
                **partial_values,
                measurement_statuses=(
                    "accepted",
                    "accepted",
                    "accepted",
                ),
            )
        with self.assertRaisesRegex(ValueError, "status"):
            make_curve(
                **{
                    **partial_values,
                    "status": "accepted",
                },
                measurement_statuses=(
                    "invalid_instantaneous_frequency",
                    "accepted",
                    "accepted",
                ),
            )

        definition = convention.definition
        with self.assertRaisesRegex(ValueError, "integer"):
            self.mod.PhaseConventionDefinition(
                hilbert_phase_sign=True,
                scipy_phase_multiplier=True,
                phase_time_sign=True,
                formula_phase_sign=1,
                fixed_phase_rad=definition.fixed_phase_rad,
                cycle_phase_sign=1,
                apply_negative_time_derivative=False,
                cycle_count_meaning=definition.cycle_count_meaning,
                formula=definition.formula,
                description=definition.description,
            )

    def test_phase_curve_rejects_nonuniform_or_dt_inconsistent_time(self):
        measure_curve = self.require_module_attribute("measure_phase_curve")
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        base_time = np.arange(-10.0, 20.0, 0.02)
        waveform = np.cos(2.0 * np.pi * base_time / 3.0)
        for time_s, dt_s in (
            (np.r_[base_time[:500], base_time[500:] + 0.001], 0.02),
            (base_time, 0.021),
        ):
            trace = self.mod.DatTrace(
                "bad_time", 20.0, dt_s, time_s,
                waveform, waveform, waveform,
                0.0, 0.0, 0.1, 0.1,
            )
            with self.assertRaisesRegex(ValueError, "uniform|dt_s"):
                measure_curve(
                    trace,
                    periods_s=np.array([2.9, 3.0, 3.1]),
                    velocity_axis_km_s=(
                        self.mod.FtanConfig().group_velocities_km_s
                    ),
                    alpha=20.0,
                    beta1=0.5,
                    beta2=1.0,
                    convention=convention,
                )

    def test_phase_waveform_preparation_copies_bensen_and_validates_axes(self):
        prepare = self.require_module_attribute("prepare_phase_waveform")
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        time_s = np.array([0.0, 0.1, 0.25, 0.5])
        waveform = np.array([1.0, 2.0, -1.0, 0.5])
        original_time = time_s.copy()
        original_waveform = waveform.copy()

        prepared = prepare(time_s, waveform, convention)

        np.testing.assert_array_equal(prepared, waveform)
        self.assertIsNot(prepared, waveform)
        self.assertFalse(prepared.flags.writeable)
        np.testing.assert_array_equal(time_s, original_time)
        np.testing.assert_array_equal(waveform, original_waveform)
        with self.assertRaises(ValueError):
            prepared[0] = 3.0

        bad_calls = (
            (time_s[:, np.newaxis], waveform, convention, "one-dimensional"),
            (time_s, waveform[:-1], convention, "same shape"),
            (
                np.array([0.0, 0.2, 0.1, 0.5]),
                waveform,
                convention,
                "strictly increasing",
            ),
            (
                np.array([0.0, 0.1, np.nan, 0.5]),
                waveform,
                convention,
                "finite",
            ),
            (
                time_s,
                np.array([1.0, np.inf, -1.0, 0.5]),
                convention,
                "finite",
            ),
            (
                np.array([False, True, True, True]),
                waveform,
                convention,
                "boolean",
            ),
            (
                time_s,
                np.array([True, True, False, True]),
                convention,
                "boolean",
            ),
            (time_s, waveform, "lin", "PhaseConvention"),
        )
        for bad_time, bad_waveform, bad_convention, message in bad_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    prepare(bad_time, bad_waveform, bad_convention)

    def test_phase_time_helpers_reject_ambiguous_or_invalid_scalars(self):
        raw_time = self.require_module_attribute("raw_phase_travel_time")
        apply_cycle = self.require_module_attribute("apply_cycle_count")
        bensen = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF

        raw_bad_calls = (
            (bensen, True, 0.2, 1.0, "group_time_s"),
            (bensen, np.array([2.0]), 0.2, 1.0, "group_time_s"),
            (bensen, 0.0, 0.2, 1.0, "group_time_s"),
            (bensen, 2.0, np.nan, 1.0, "phase_rad"),
            (bensen, 2.0, 0.2, False, "omega_rad_s"),
            (bensen, 2.0, 0.2, np.array([1.0]), "omega_rad_s"),
            (bensen, 2.0, 0.2, 0.0, "omega_rad_s"),
            ("bensen", 2.0, 0.2, 1.0, "PhaseConvention"),
        )
        for convention, group_time, phase, omega, message in raw_bad_calls:
            with self.subTest(helper="raw", message=message):
                with self.assertRaisesRegex(ValueError, message):
                    raw_time(
                        convention=convention,
                        group_time_s=group_time,
                        phase_rad=phase,
                        omega_rad_s=omega,
                    )

        cycle_bad_calls = (
            (True, 1, 2.0, bensen, "raw_time_s"),
            (2.0, True, 2.0, bensen, "cycle_count"),
            (2.0, 1.5, 2.0, bensen, "cycle_count"),
            (2.0, 1, np.array([2.0]), bensen, "period_s"),
            (2.0, 1, 0.0, bensen, "period_s"),
            (2.0, 1, 2.0, "bensen", "PhaseConvention"),
        )
        for value, cycle_count, period, convention, message in cycle_bad_calls:
            with self.subTest(helper="cycle", message=message):
                with self.assertRaisesRegex(ValueError, message):
                    apply_cycle(
                        value,
                        cycle_count,
                        period,
                        convention=convention,
                    )

        with self.assertRaisesRegex(ValueError, "positive"):
            apply_cycle(1.0, -1, 2.0)

    def test_candidate_compatibility_uses_only_centralized_fixed_phase(self):
        compute = self.require_module_attribute("compute_phase_speed_candidates")
        lin = self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF
        distance_km = 25.0
        group_velocity_km_s = 2.6
        true_phase_velocity_km_s = 3.3
        period_s = 2.5
        omega_rad_s = 2.0 * np.pi / period_s
        branch = 1
        group_time_s = distance_km / group_velocity_km_s
        true_time_s = distance_km / true_phase_velocity_km_s
        phi_tu_rad = (
            omega_rad_s * (true_time_s - group_time_s + branch * period_s)
            + np.pi / 4.0
        )

        candidates = compute(
            phi_tu=phi_tu_rad,
            omega=omega_rad_s,
            distance_km=distance_km,
            group_velocity_km_s=group_velocity_km_s,
            branch_min=-3,
            branch_max=3,
            convention=lin,
            velocity_bounds=(1.0, 4.0),
        )

        by_branch = {row.branch: row.phase_velocity_km_s for row in candidates}
        self.assertAlmostEqual(
            by_branch[branch],
            true_phase_velocity_km_s,
            places=10,
        )
        central_fixed = (
            self.mod.PhaseConvention.BENSEN_VELOCITY_CCF.definition.fixed_phase_rad
        )
        compatible = compute(
            phi_tu=0.2,
            omega=2.0,
            distance_km=10.0,
            group_velocity_km_s=2.5,
            phase_shift_rad=central_fixed,
            convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
        )
        self.assertTrue(compatible)
        with self.assertRaisesRegex(ValueError, "phase_shift_rad"):
            compute(
                phi_tu=0.2,
                omega=2.0,
                distance_km=10.0,
                group_velocity_km_s=2.5,
                phase_shift_rad=0.0,
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            )

        module_source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertEqual(module_source.count("math.pi / 4.0"), 1)
        compute_source = inspect.getsource(compute)
        self.assertNotIn("pi / 4", compute_source)
        self.assertIsNone(inspect.signature(compute).parameters["phase_shift_rad"].default)

    def test_candidate_helper_rejects_invalid_measurement_inputs(self):
        compute = self.require_module_attribute("compute_phase_speed_candidates")
        valid = {
            "phi_tu": 0.2,
            "omega": 2.0,
            "distance_km": 10.0,
            "group_velocity_km_s": 2.5,
            "convention": self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
        }
        bad_values = (
            ("phi_tu", True),
            ("phi_tu", np.array([0.2])),
            ("phi_tu", np.nan),
            ("omega", False),
            ("omega", 0.0),
            ("distance_km", np.array([10.0])),
            ("distance_km", -1.0),
            ("group_velocity_km_s", np.inf),
            ("group_velocity_km_s", 0.0),
        )
        for name, value in bad_values:
            arguments = dict(valid)
            arguments[name] = value
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    compute(**arguments)

        with self.assertRaisesRegex(ValueError, "branch"):
            compute(**valid, branch_min=True)
        with self.assertRaisesRegex(ValueError, "branch"):
            compute(**valid, branch_min=2, branch_max=1)
        for velocity_bounds in (
            (True, 4.0),
            (1.0, False),
            (np.array(True), 4.0),
            (1.0, np.array(False)),
            "14",
        ):
            with self.subTest(velocity_bounds=velocity_bounds):
                with self.assertRaisesRegex(ValueError, "velocity_bounds"):
                    compute(**valid, velocity_bounds=velocity_bounds)

    def test_single_period_measurement_rejects_invalid_scalar_controls(self):
        measure = self.require_module_attribute("measure_single_period")
        dt_s = 0.02
        time_s = np.arange(-30.0, 50.0, dt_s)
        waveform = np.exp(-0.5 * ((time_s - 8.0) / 4.5) ** 2) * np.cos(
            2.0 * np.pi * (time_s - 8.0) / 3.0
        )
        trace = self.mod.DatTrace(
            pair_name="validation",
            distance_km=20.0,
            dt_s=dt_s,
            time_s=time_s,
            positive_lag=waveform.copy(),
            negative_lag_reversed=waveform.copy(),
            symmetric_waveform=waveform.copy(),
            lon_a=0.0,
            lat_a=0.0,
            lon_b=0.1,
            lat_b=0.1,
        )
        valid = {
            "period_s": 3.0,
            "vmin_km_s": 1.6,
            "vmax_km_s": 5.0,
            "alpha": 20.0,
            "snr_gap_s": 0.5,
            "convention": self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
        }
        invalid_values = (
            True,
            np.array(1.0),
            np.array([1.0]),
            np.nan,
            np.inf,
            0.0,
            -1.0,
        )
        for name in (
            "period_s",
            "vmin_km_s",
            "vmax_km_s",
            "alpha",
            "snr_gap_s",
        ):
            for value in invalid_values:
                arguments = dict(valid)
                arguments[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        measure(trace, **arguments)

        for vmin_km_s, vmax_km_s in ((5.0, 5.0), (5.1, 5.0)):
            with self.subTest(
                vmin_km_s=vmin_km_s,
                vmax_km_s=vmax_km_s,
            ):
                with self.assertRaisesRegex(ValueError, "vmin_km_s"):
                    measure(
                        trace,
                        **{
                            **valid,
                            "vmin_km_s": vmin_km_s,
                            "vmax_km_s": vmax_km_s,
                        },
                    )

        measurement = measure(trace, **{**valid, "alpha": None})
        self.assertIsNotNone(measurement)

    def test_phase_scalar_apis_require_real_numbers(self):
        class FloatLike:
            def __float__(self):
                return 1.0

        bad_scalars = (
            True,
            np.array(1.0),
            "1.0",
            b"1.0",
            1.0 + 0.0j,
            FloatLike(),
            object(),
        )
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        time_s = np.arange(-10.0, 20.0, 0.02)
        waveform = np.cos(2.0 * np.pi * time_s / 3.0)
        trace = self.mod.DatTrace(
            "strict_scalars", 20.0, 0.02, time_s,
            waveform, waveform, waveform,
            0.0, 0.0, 0.1, 0.1,
        )
        for value in bad_scalars:
            with self.subTest(api="_finite_scalar", value=value):
                with self.assertRaisesRegex(ValueError, "value"):
                    self.mod._finite_scalar(value, "value")
            with self.subTest(api="raw_phase_travel_time", value=value):
                with self.assertRaisesRegex(ValueError, "group_time_s"):
                    self.mod.raw_phase_travel_time(
                        convention=convention,
                        group_time_s=value,
                        phase_rad=0.2,
                        omega_rad_s=2.0,
                    )
            with self.subTest(api="gaussian_filter_bank", value=value):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    self.mod.gaussian_filter_bank(
                        waveform,
                        dt_s=0.02,
                        periods_s=np.array([3.0]),
                        alpha=value,
                    )
            with self.subTest(api="measure_phase_curve", value=value):
                with self.assertRaisesRegex(ValueError, "beta1"):
                    self.mod.measure_phase_curve(
                        trace,
                        periods_s=np.array([2.9, 3.0, 3.1]),
                        velocity_axis_km_s=(
                            self.mod.FtanConfig().group_velocities_km_s
                        ),
                        alpha=20.0,
                        beta1=value,
                        beta2=1.0,
                        convention=convention,
                    )

        for value in (1, np.int64(1), np.uint64(1), np.float64(1.0)):
            with self.subTest(valid_real=value):
                self.assertEqual(self.mod._finite_scalar(value, "value"), 1.0)

    def test_phase_array_apis_reject_non_real_numeric_dtypes_without_warning(self):
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        valid_time = np.array([0.0, 0.1, 0.2, 0.3])
        valid_waveform = np.array([1.0, 0.5, -0.5, -1.0])
        bad_arrays = (
            np.array([True, False, True, False]),
            np.array(["0", "1", "2", "3"], dtype=object),
            np.array(["0", "1", "2", "3"], dtype="U1"),
            np.array([b"0", b"1", b"2", b"3"], dtype="S1"),
            np.array([0.0 + 1.0j, 0.1, 0.2, 0.3]),
        )
        for bad in bad_arrays:
            with self.subTest(api="prepare_time", kind=bad.dtype.kind):
                with self.assertRaises(ValueError):
                    self.mod.prepare_phase_waveform(
                        bad,
                        valid_waveform,
                        convention,
                    )
            with self.subTest(api="prepare_waveform", kind=bad.dtype.kind):
                with self.assertRaises(ValueError):
                    self.mod.prepare_phase_waveform(
                        valid_time,
                        bad,
                        convention,
                    )
            with self.subTest(api="filter_waveform", kind=bad.dtype.kind):
                with self.assertRaises(ValueError):
                    self.mod.gaussian_filter_bank(
                        bad,
                        dt_s=0.1,
                        periods_s=np.array([1.0]),
                        alpha=20.0,
                    )
            with self.subTest(api="validate_true", kind=bad.dtype.kind):
                with self.assertRaises(ValueError):
                    self.mod.validate_phase_convention(
                        convention=convention,
                        true_phase_velocities_km_s=bad,
                        recovered_phase_velocities_km_s=np.ones(4),
                        noise_free_mask=np.array([True, True, False, False]),
                    )
            curve_arguments = {
                "periods_s": np.array([0.08, 0.1, 0.12]),
                "velocity_axis_km_s": np.array([1.6, 1.7]),
                "alpha": 20.0,
                "convention": convention,
            }
            valid_trace = self.mod.DatTrace(
                "array_kinds", 0.2, 0.1, valid_time,
                valid_waveform, valid_waveform, valid_waveform,
                0.0, 0.0, 0.1, 0.1,
            )
            for name in ("periods_s", "velocity_axis_km_s"):
                with self.subTest(
                    api="measure_curve_axis",
                    name=name,
                    kind=bad.dtype.kind,
                ):
                    with self.assertRaises(ValueError):
                        self.mod.measure_phase_curve(
                            valid_trace,
                            **{**curve_arguments, name: bad},
                        )
            for trace_field in ("time_s", "symmetric_waveform"):
                values = {
                    "time_s": valid_time,
                    "symmetric_waveform": valid_waveform,
                }
                values[trace_field] = bad
                bad_trace = self.mod.DatTrace(
                    "array_kinds", 0.2, 0.1, values["time_s"],
                    valid_waveform, valid_waveform,
                    values["symmetric_waveform"],
                    0.0, 0.0, 0.1, 0.1,
                )
                with self.subTest(
                    api="measure_curve_trace",
                    field=trace_field,
                    kind=bad.dtype.kind,
                ):
                    with self.assertRaises(ValueError):
                        self.mod.measure_phase_curve(
                            bad_trace,
                            **curve_arguments,
                        )

    def test_public_phase_kernels_reject_non_real_arrays_before_conversion(self):
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        valid_time = np.array([0.0, 0.1, 0.2, 0.3])
        valid_values = np.array([0.1, 0.8, 0.6, 0.2])
        valid_periods = np.array([2.8, 3.0, 3.2])
        valid_velocity = np.array([1.6, 1.7, 1.8, 1.9])
        valid_energy = np.array(
            [
                [0.1, 0.5, 0.3, 0.1],
                [0.1, 0.6, 0.4, 0.1],
                [0.1, 0.7, 0.5, 0.1],
            ]
        )
        bad_arrays = (
            np.array([True, False, True, False]),
            np.array(["0", "1", "2", "3"], dtype=object),
            np.array(["0", "1", "2", "3"], dtype="U1"),
            np.array([b"0", b"1", b"2", b"3"], dtype="S1"),
            np.array([0.0 + 1.0j, 0.1, 0.2, 0.3]),
        )
        for bad in bad_arrays:
            bad_periods = bad[:3]
            bad_grid = np.resize(bad, valid_energy.shape)
            calls = (
                (
                    "normalized_log_energy",
                    lambda: self.mod.normalized_log_energy(
                        bad.reshape(2, 2)
                    ),
                ),
                (
                    "refine_time",
                    lambda: self.mod.refine_group_arrival(
                        bad,
                        valid_values,
                        1,
                    ),
                ),
                (
                    "refine_envelope",
                    lambda: self.mod.refine_group_arrival(
                        valid_time,
                        bad,
                        1,
                    ),
                ),
                (
                    "interpolate_time",
                    lambda: self.mod.interpolate_analytic_phase_at_arrival(
                        bad,
                        np.exp(1j * valid_time),
                        0.15,
                    ),
                ),
                (
                    "frequency_time",
                    lambda: self.mod.estimate_instantaneous_frequency(
                        bad,
                        valid_values,
                        group_time_s=0.15,
                        nominal_period_s=0.2,
                    ),
                ),
                (
                    "frequency_phase",
                    lambda: self.mod.estimate_instantaneous_frequency(
                        valid_time,
                        bad,
                        group_time_s=0.15,
                        nominal_period_s=0.2,
                    ),
                ),
                (
                    "unwrap_periods",
                    lambda: self.mod.unwrap_phase_along_frequency(
                        bad_periods,
                        np.array([0.1, 0.2, 0.3]),
                        np.array([1.0, 1.1, 1.2]),
                        convention=convention,
                    ),
                ),
                (
                    "unwrap_phase",
                    lambda: self.mod.unwrap_phase_along_frequency(
                        valid_periods,
                        bad_periods,
                        np.array([1.0, 1.1, 1.2]),
                        convention=convention,
                    ),
                ),
                (
                    "unwrap_group_time",
                    lambda: self.mod.unwrap_phase_along_frequency(
                        valid_periods,
                        np.array([0.1, 0.2, 0.3]),
                        bad_periods,
                        convention=convention,
                    ),
                ),
                (
                    "ridge_energy",
                    lambda: self.mod.find_candidate_ridges(
                        scaled_log_energy=bad_grid,
                        normalized_envelope_amplitude=valid_energy,
                        periods_s=valid_periods,
                        velocity_axis_km_s=valid_velocity,
                        beta1=0.5,
                        beta2=1.0,
                    ),
                ),
                (
                    "ridge_amplitude",
                    lambda: self.mod.find_candidate_ridges(
                        scaled_log_energy=valid_energy,
                        normalized_envelope_amplitude=bad_grid,
                        periods_s=valid_periods,
                        velocity_axis_km_s=valid_velocity,
                        beta1=0.5,
                        beta2=1.0,
                    ),
                ),
                (
                    "ridge_periods",
                    lambda: self.mod.find_candidate_ridges(
                        scaled_log_energy=valid_energy,
                        normalized_envelope_amplitude=valid_energy,
                        periods_s=bad_periods,
                        velocity_axis_km_s=valid_velocity,
                        beta1=0.5,
                        beta2=1.0,
                    ),
                ),
                (
                    "ridge_velocity",
                    lambda: self.mod.find_candidate_ridges(
                        scaled_log_energy=valid_energy,
                        normalized_envelope_amplitude=valid_energy,
                        periods_s=valid_periods,
                        velocity_axis_km_s=bad,
                        beta1=0.5,
                        beta2=1.0,
                    ),
                ),
            )
            for name, call in calls:
                with self.subTest(api=name, kind=bad.dtype.kind):
                    with self.assertRaises(ValueError):
                        call()

        analytic = np.exp(1j * valid_time)
        phase = self.mod.interpolate_analytic_phase_at_arrival(
            valid_time,
            analytic,
            0.15,
        )
        self.assertTrue(np.isfinite(phase))

    def test_public_phase_kernel_controls_require_real_or_integer_scalars(self):
        class FloatLike:
            def __float__(self):
                return 1.0

        bad_real = (
            True,
            np.array(1.0),
            "1.0",
            b"1.0",
            1.0 + 0.0j,
            FloatLike(),
            object(),
        )
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        time_s = np.linspace(0.0, 4.0, 81)
        phase = 2.0 * np.pi * time_s
        periods = np.array([2.8, 3.0, 3.2])
        energy = np.array(
            [[0.1, 0.8, 0.2], [0.1, 0.9, 0.2], [0.1, 0.8, 0.2]]
        )
        velocity = np.array([1.6, 1.7, 1.8])
        for bad in bad_real:
            calls = (
                (
                    "gaussian_alpha_for_distance",
                    lambda: self.mod.gaussian_alpha_for_distance(bad),
                ),
                (
                    "interpolate_group_time",
                    lambda: self.mod.interpolate_analytic_phase_at_arrival(
                        time_s,
                        np.exp(1j * phase),
                        bad,
                    ),
                ),
                (
                    "frequency_group_time",
                    lambda: self.mod.estimate_instantaneous_frequency(
                        time_s,
                        phase,
                        group_time_s=bad,
                        nominal_period_s=1.0,
                    ),
                ),
                (
                    "frequency_period",
                    lambda: self.mod.estimate_instantaneous_frequency(
                        time_s,
                        phase,
                        group_time_s=2.0,
                        nominal_period_s=bad,
                    ),
                ),
                (
                    "unwrap_anchor",
                    lambda: self.mod.unwrap_phase_along_frequency(
                        periods,
                        np.array([0.1, 0.2, 0.3]),
                        np.array([4.0, 4.1, 4.2]),
                        convention=convention,
                        anchor_period_s=bad,
                    ),
                ),
                (
                    "ridge_beta1",
                    lambda: self.mod.find_candidate_ridges(
                        scaled_log_energy=energy,
                        normalized_envelope_amplitude=energy,
                        periods_s=periods,
                        velocity_axis_km_s=velocity,
                        beta1=bad,
                        beta2=1.0,
                    ),
                ),
                (
                    "velocity_bounds",
                    lambda: self.mod.compute_phase_speed_candidates(
                        phi_tu=0.2,
                        omega=2.0,
                        distance_km=20.0,
                        group_velocity_km_s=2.5,
                        convention=convention,
                        velocity_bounds=(bad, 4.0),
                    ),
                ),
            )
            for name, call in calls:
                with self.subTest(api=name, value=bad):
                    with self.assertRaises(ValueError):
                        call()

        bad_integer = (True, np.array(1), 1.0, "1", FloatLike())
        for bad in bad_integer:
            with self.subTest(api="period_axis", value=bad):
                with self.assertRaises(ValueError):
                    self.mod.normalized_log_energy(
                        np.ones((2, 2)),
                        period_axis=bad,
                    )
            with self.subTest(api="peak_index", value=bad):
                with self.assertRaises(ValueError):
                    self.mod.refine_group_arrival(
                        np.array([0.0, 0.1, 0.2]),
                        np.array([0.0, 1.0, 0.0]),
                        bad,
                    )
            with self.subTest(api="max_candidates", value=bad):
                with self.assertRaises(ValueError):
                    self.mod.find_candidate_ridges(
                        scaled_log_energy=energy,
                        normalized_envelope_amplitude=energy,
                        periods_s=periods,
                        velocity_axis_km_s=velocity,
                        beta1=0.5,
                        beta2=1.0,
                        max_candidates=bad,
                    )

    def test_branch_selection_and_extraction_validate_public_inputs_upfront(self):
        class FloatLike:
            def __float__(self):
                return 1.0

        valid_periods = np.array([2.8, 3.0, 3.2])
        valid_velocity = np.array([2.5, 2.6, 2.7])
        empty_candidates = [[], [], []]
        bad_arrays = (
            np.array([True, False, True]),
            np.array([2.8, 3.0, 3.2], dtype=object),
            np.array(["2.8", "3.0", "3.2"], dtype=object),
            np.array(["2.8", "3.0", "3.2"], dtype="U3"),
            np.array([b"2.8", b"3.0", b"3.2"], dtype="S3"),
            np.array([2.8 + 0.0j, 3.0, 3.2]),
        )
        for bad in bad_arrays:
            with self.subTest(api="select_periods", kind=bad.dtype.kind):
                with self.assertRaises(ValueError):
                    self.mod.select_branch_sequence(
                        periods_s=bad,
                        group_velocities_km_s=valid_velocity,
                        candidate_lists=empty_candidates,
                    )
            with self.subTest(api="select_velocity", kind=bad.dtype.kind):
                with self.assertRaises(ValueError):
                    self.mod.select_branch_sequence(
                        periods_s=valid_periods,
                        group_velocities_km_s=bad,
                        candidate_lists=empty_candidates,
                    )
            with self.subTest(api="extract_periods", kind=bad.dtype.kind):
                with self.assertRaises(ValueError):
                    self.mod.extract_bensen_phase_curve(
                        "/path/that/must/not/be/read.dat",
                        periods_s=bad,
                    )

        bad_real = (
            True,
            np.array(1.0),
            "1.0",
            b"1.0",
            1.0 + 0.0j,
            FloatLike(),
            object(),
        )
        for bad in bad_real:
            for name in ("vmin_km_s", "vmax_km_s", "min_snr"):
                with self.subTest(api="extract_control", name=name, value=bad):
                    with self.assertRaises(ValueError):
                        self.mod.extract_bensen_phase_curve(
                            "/path/that/must/not/be/read.dat",
                            **{name: bad},
                        )

        bad_integer = (True, np.array(1), 1.0, "1", FloatLike())
        for bad in bad_integer:
            with self.subTest(api="iterations", value=bad):
                with self.assertRaises(ValueError):
                    self.mod.select_branch_sequence(
                        periods_s=valid_periods,
                        group_velocities_km_s=valid_velocity,
                        candidate_lists=empty_candidates,
                        iterations=bad,
                    )
            for name in ("branch_min", "branch_max"):
                with self.subTest(api="extract_branch", name=name, value=bad):
                    with self.assertRaises(ValueError):
                        self.mod.extract_bensen_phase_curve(
                            "/path/that/must/not/be/read.dat",
                            **{name: bad},
                        )

        for dtype in (int, np.uint64, float):
            with self.subTest(valid_array_dtype=np.dtype(dtype).kind):
                selected = self.mod.select_branch_sequence(
                    periods_s=np.array([2, 3, 4], dtype=dtype),
                    group_velocities_km_s=np.array(
                        [2, 3, 4],
                        dtype=dtype,
                    ),
                    candidate_lists=empty_candidates,
                    iterations=np.int64(1),
                )
                self.assertEqual(selected.branches, [0, 0, 0])

    def test_single_period_validates_controls_before_trace_eligibility(self):
        measure = self.require_module_attribute("measure_single_period")
        time_s = np.arange(-30.0, 50.0, 0.02)
        waveform = np.cos(2.0 * np.pi * time_s / 3.0)

        def make_trace(*, distance_km, dt_s, name):
            return self.mod.DatTrace(
                pair_name=name,
                distance_km=distance_km,
                dt_s=dt_s,
                time_s=time_s,
                positive_lag=waveform.copy(),
                negative_lag_reversed=waveform.copy(),
                symmetric_waveform=waveform.copy(),
                lon_a=0.0,
                lat_a=0.0,
                lon_b=0.1,
                lat_b=0.1,
            )

        ineligible_traces = (
            make_trace(distance_km=0.0, dt_s=0.02, name="bad_distance"),
            make_trace(distance_km=20.0, dt_s=0.0, name="bad_dt"),
        )
        valid = {
            "period_s": 3.0,
            "vmin_km_s": 1.6,
            "vmax_km_s": 5.0,
            "alpha": 20.0,
            "snr_gap_s": 0.5,
            "convention": self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
        }
        invalid_values = (
            True,
            np.array(1.0),
            np.nan,
            np.inf,
            0.0,
            -1.0,
        )
        for trace in ineligible_traces:
            for name in (
                "period_s",
                "vmin_km_s",
                "vmax_km_s",
                "alpha",
                "snr_gap_s",
            ):
                for value in invalid_values:
                    arguments = {**valid, name: value}
                    with self.subTest(
                        trace=trace.pair_name,
                        name=name,
                        value=value,
                    ):
                        with self.assertRaisesRegex(ValueError, name):
                            measure(trace, **arguments)

            for vmin_km_s, vmax_km_s in ((5.0, 5.0), (5.1, 5.0)):
                with self.subTest(
                    trace=trace.pair_name,
                    vmin_km_s=vmin_km_s,
                    vmax_km_s=vmax_km_s,
                ):
                    with self.assertRaisesRegex(ValueError, "vmin_km_s"):
                        measure(
                            trace,
                            **{
                                **valid,
                                "vmin_km_s": vmin_km_s,
                                "vmax_km_s": vmax_km_s,
                            },
                        )

            with self.subTest(trace=trace.pair_name, convention="invalid"):
                with self.assertRaisesRegex(ValueError, "convention"):
                    measure(trace, **{**valid, "convention": "invalid"})
            with self.subTest(trace=trace.pair_name, alpha=None):
                with mock.patch.object(
                    self.mod,
                    "gaussian_alpha_for_distance",
                    side_effect=AssertionError(
                        "ineligible trace must not compute default alpha"
                    ),
                ) as default_alpha:
                    self.assertIsNone(
                        measure(trace, **{**valid, "alpha": None})
                    )
                default_alpha.assert_not_called()
            with self.subTest(trace=trace.pair_name, controls="valid"):
                self.assertIsNone(measure(trace, **valid))

    def test_synthetic_cases_cover_each_convention_on_formal_ftan_domain(self):
        cases_by_convention = self._phase_convention_synthetic_cases()
        cycle_signs = {
            "BENSEN_VELOCITY_CCF": 1,
            "LIN_NEGATIVE_DERIVATIVE_EGF": -1,
        }
        for convention in self.mod.PhaseConvention:
            cases = cases_by_convention[convention.name]
            self.assertEqual(
                {row[0] for row in cases},
                {2.5, 3.0, 3.5, 4.0, 4.5, 5.0},
            )
            self.assertEqual({row[3] for row in cases}, set(range(-3, 4)))
            self.assertEqual(
                {row[5] for row in cases},
                {"single", "dual", "coda"},
            )
            self.assertEqual(
                (min(row[1] for row in cases), max(row[1] for row in cases)),
                (2.2, 3.4),
            )
            self.assertEqual(
                (min(row[2] for row in cases), max(row[2] for row in cases)),
                (8.0, 25.0),
            )
            for (
                period_s,
                phase_velocity_km_s,
                distance_km,
                cycle_count,
                phi_tu_rad,
                _,
            ) in cases:
                omega_rad_s = 2.0 * np.pi / period_s
                group_time_s = (
                    distance_km / phase_velocity_km_s
                    - (phi_tu_rad - np.pi / 4.0) / omega_rad_s
                    - cycle_signs[convention.name]
                    * cycle_count
                    * period_s
                )
                group_velocity_km_s = distance_km / group_time_s
                self.assertGreaterEqual(group_velocity_km_s, 1.6)
                self.assertLessEqual(group_velocity_km_s, 5.0)
                self.assertNotAlmostEqual(
                    group_velocity_km_s,
                    phase_velocity_km_s,
                    places=4,
                )

    @staticmethod
    def _phase_convention_synthetic_cases():
        return {
            "BENSEN_VELOCITY_CCF": (
                (2.5, 3.4, 25.0, -3, 3.0, "single"),
                (2.5, 2.2, 25.0, 3, -3.0, "single"),
                (3.0, 3.2, 22.0, -2, 3.0, "single"),
                (3.0, 2.2, 25.0, 2, -3.0, "single"),
                (3.5, 2.7, 20.0, -1, 2.1, "single"),
                (4.0, 2.9, 18.0, 1, -2.0, "single"),
                (4.5, 3.1, 25.0, 0, 0.3, "coda"),
                (5.0, 3.25, 25.0, 0, 0.0, "dual"),
                (2.5, 3.4, 8.0, 0, -2.0, "single"),
            ),
            "LIN_NEGATIVE_DERIVATIVE_EGF": (
                (2.5, 2.2, 25.0, -3, -3.0, "single"),
                (2.5, 3.4, 25.0, 3, 3.0, "single"),
                (3.0, 2.2, 25.0, -2, -3.0, "single"),
                (3.0, 3.2, 22.0, 2, 3.0, "single"),
                (3.5, 2.2, 25.0, -1, -3.0, "single"),
                (4.0, 3.4, 20.0, 1, 3.0, "single"),
                (4.5, 3.1, 25.0, 0, 0.3, "coda"),
                (5.0, 3.25, 25.0, 0, 0.0, "dual"),
                (2.5, 3.4, 8.0, 0, -2.0, "single"),
            ),
        }

    def _run_phase_convention_synthetic_case(
        self,
        *,
        synthesis_convention,
        processing_convention,
        specification,
        noisy,
        skip_preparation=False,
        wrong_phase_sign=False,
        wrong_fixed_phase=False,
    ):
        (
            nominal_period_s,
            true_phase_velocity_km_s,
            distance_km,
            cycle_count,
            paper_phi_tu_rad,
            mode,
        ) = specification
        synthesis_oracles = {
            "BENSEN_VELOCITY_CCF": {
                "scipy_phase_multiplier": -1,
                "fixed_phase_rad": -np.pi / 4.0,
                "cycle_phase_sign": 1,
                "apply_negative_time_derivative": False,
            },
            "LIN_NEGATIVE_DERIVATIVE_EGF": {
                "scipy_phase_multiplier": -1,
                "fixed_phase_rad": -np.pi / 4.0,
                "cycle_phase_sign": -1,
                "apply_negative_time_derivative": True,
            },
        }
        synthesis_oracle = synthesis_oracles[synthesis_convention.name]
        omega_rad_s = 2.0 * np.pi / nominal_period_s
        true_phase_time_s = distance_km / true_phase_velocity_km_s
        group_time_s = (
            true_phase_time_s
            - (
                paper_phi_tu_rad + synthesis_oracle["fixed_phase_rad"]
            )
            / omega_rad_s
            - synthesis_oracle["cycle_phase_sign"]
            * cycle_count
            * nominal_period_s
        )
        self.assertGreater(group_time_s, 0.5 * nominal_period_s)
        group_velocity_km_s = distance_km / group_time_s
        self.assertGreaterEqual(group_velocity_km_s, 1.6)
        self.assertLessEqual(group_velocity_km_s, 5.0)
        self.assertNotAlmostEqual(
            group_velocity_km_s,
            true_phase_velocity_km_s,
            places=4,
        )

        dt_s = 0.02
        time_s = np.arange(-30.0, 50.0, dt_s)
        scipy_phi_tu_rad = (
            synthesis_oracle["scipy_phase_multiplier"]
            * paper_phi_tu_rad
        )
        local_time_s = time_s - group_time_s
        phase_waveform = (
            np.exp(
                -0.5
                * (
                    local_time_s / (1.5 * nominal_period_s)
                )
                ** 2
            )
            * np.cos(
                omega_rad_s * local_time_s + scipy_phi_tu_rad
            )
        )
        primary_waveform = np.array(phase_waveform, copy=True)
        competitor_center_s = float("nan")
        competitor_period_s = float("nan")
        competitor_component = np.zeros_like(phase_waveform)
        if mode == "dual":
            competitor_period_s = 0.65 * nominal_period_s
            secondary_omega = 2.0 * np.pi / competitor_period_s
            phase_waveform = phase_waveform + (
                0.40
                * np.exp(
                    -0.5
                    * (
                        local_time_s / nominal_period_s
                    )
                    ** 2
                )
                * np.cos(
                    secondary_omega * local_time_s
                    + scipy_phi_tu_rad
                    + np.pi / 2.0
                )
            )
            primary_waveform = np.array(phase_waveform, copy=True)
            separation_s = 1.2 * nominal_period_s
            competitor_center_s = group_time_s + separation_s
            secondary_time = time_s - competitor_center_s
            competitor_component = (
                0.40
                * np.exp(
                    -0.5
                    * (
                        secondary_time / (0.30 * nominal_period_s)
                    )
                    ** 2
                )
                * np.cos(
                    secondary_omega * secondary_time
                    + scipy_phi_tu_rad
                    + secondary_omega
                    * (competitor_center_s - group_time_s)
                    + 3.0 * np.pi / 4.0
                )
            )
            phase_waveform = phase_waveform + competitor_component
        elif mode == "coda":
            separation_s = 1.2 * nominal_period_s
            competitor_center_s = group_time_s + separation_s
            coda_time = time_s - competitor_center_s
            competitor_period_s = 0.70 * nominal_period_s
            coda_omega = 2.0 * np.pi / competitor_period_s
            phase_waveform = phase_waveform + (
                0.70
                * np.exp(
                    -0.5
                    * (
                        local_time_s / (1.5 * nominal_period_s)
                    )
                    ** 2
                )
                * np.cos(
                    coda_omega * local_time_s
                    + scipy_phi_tu_rad
                    + 3.0 * np.pi / 4.0
                )
            )
            primary_waveform = np.array(phase_waveform, copy=True)
            competitor_component = (
                0.30
                * np.exp(
                    -0.5
                    * (coda_time / (0.30 * nominal_period_s)) ** 2
                )
                * np.cos(
                    coda_omega * coda_time
                    + scipy_phi_tu_rad
                    + coda_omega
                    * (competitor_center_s - group_time_s)
                    + np.pi
                )
            )
            phase_waveform = phase_waveform + competitor_component

        def to_symmetric_ccf(waveform):
            if not synthesis_oracle["apply_negative_time_derivative"]:
                return np.array(waveform, copy=True)
            ccf = np.empty_like(waveform)
            ccf[0] = 0.0
            ccf[1:] = -np.cumsum(
                0.5
                * (waveform[1:] + waveform[:-1])
                * np.diff(time_s)
            )
            return ccf

        primary_symmetric_ccf = to_symmetric_ccf(primary_waveform)
        symmetric_ccf = to_symmetric_ccf(phase_waveform)
        disturbed_symmetric_ccf = symmetric_ccf.copy()
        if noisy:
            noise_amplitude = {
                "single": 0.009,
                "dual": 0.007,
                "coda": 0.008,
            }[mode]
            seed = (
                10_000
                + int(round(10.0 * nominal_period_s))
                + 17 * cycle_count
            )
            symmetric_ccf = symmetric_ccf + (
                noise_amplitude
                * np.random.default_rng(seed).normal(size=time_s.size)
            )
        else:
            noise_amplitude = 0.0
            seed = None
        original_ccf = symmetric_ccf.copy()
        public_trace = self.mod.DatTrace(
            pair_name=(
                f"synthetic_{synthesis_convention.name}_{mode}"
            ),
            distance_km=distance_km,
            dt_s=dt_s,
            time_s=time_s,
            positive_lag=symmetric_ccf.copy(),
            negative_lag_reversed=symmetric_ccf.copy(),
            symmetric_waveform=symmetric_ccf.copy(),
            lon_a=0.0,
            lat_a=0.0,
            lon_b=0.1,
            lat_b=0.1,
        )
        preparation_context = (
            mock.patch.object(
                self.mod,
                "prepare_phase_waveform",
                side_effect=lambda _, waveform, __: np.array(
                    waveform,
                    dtype=float,
                    copy=True,
                ),
            )
            if skip_preparation
            else contextlib.nullcontext()
        )
        with preparation_context:
            public_measurement = self.mod.measure_single_period(
                public_trace,
                period_s=nominal_period_s,
                vmin_km_s=1.6,
                vmax_km_s=5.0,
                alpha=20.0,
                convention=processing_convention,
            )
        if public_measurement is None:
            raise ValueError("public synthetic measurement was invalid")
        np.testing.assert_array_equal(symmetric_ccf, original_ccf)
        snr = public_measurement.snr
        self.assertTrue(np.isfinite(snr))
        self.assertGreater(snr, 1.0)
        measured_phi_rad = public_measurement.phi_tu_rad
        if wrong_phase_sign:
            measured_phi_rad *= -1.0
        measured_period_s = (
            2.0 * np.pi / public_measurement.omega_inst_rad_s
        )
        if wrong_fixed_phase:
            raw_time_s = (
                public_measurement.group_time_s
                + processing_convention.definition.formula_phase_sign
                * measured_phi_rad
                / public_measurement.omega_inst_rad_s
                + np.pi / 4.0 / public_measurement.omega_inst_rad_s
            )
            corrected_time_s = self.mod.apply_cycle_count(
                raw_time_s,
                cycle_count,
                measured_period_s,
                convention=processing_convention,
            )
            recovered_velocity_km_s = distance_km / corrected_time_s
        else:
            if wrong_phase_sign:
                public_candidates = self.mod.compute_phase_speed_candidates(
                    phi_tu=measured_phi_rad,
                    omega=public_measurement.omega_inst_rad_s,
                    distance_km=distance_km,
                    group_velocity_km_s=public_measurement.group_velocity_km_s,
                    branch_min=cycle_count,
                    branch_max=cycle_count,
                    convention=processing_convention,
                    velocity_bounds=(0.1, 20.0),
                )
            else:
                public_candidates = (
                    self.mod.phase_speed_candidates_from_measurement(
                        public_measurement,
                        distance_km=distance_km,
                        branch_min=cycle_count,
                        branch_max=cycle_count,
                        velocity_bounds=(0.1, 20.0),
                    )
                )
            if len(public_candidates) != 1:
                raise ValueError(
                    "public measurement branch did not yield one candidate"
                )
            self.assertEqual(public_candidates[0].branch, cycle_count)
            recovered_velocity_km_s = public_candidates[0].phase_velocity_km_s
        kernel_recovered_velocity_km_s = recovered_velocity_km_s
        kernel_relative_error = abs(
            kernel_recovered_velocity_km_s
            / true_phase_velocity_km_s
            - 1.0
        )
        relative_error = abs(
            recovered_velocity_km_s / true_phase_velocity_km_s - 1.0
        )
        return {
            "true_phase_velocity_km_s": true_phase_velocity_km_s,
            "recovered_phase_velocity_km_s": recovered_velocity_km_s,
            "relative_error": relative_error,
            "kernel_relative_error": kernel_relative_error,
            "noise_free": not noisy,
            "snr": snr,
            "prepared": public_measurement.filtered_waveform,
            "noise_amplitude": noise_amplitude,
            "noise_seed": seed,
            "competitor_center_s": competitor_center_s,
            "competitor_period_s": competitor_period_s,
            "competitor_energy_ratio": float(
                np.sum(competitor_component**2)
                / max(np.sum(primary_waveform**2), 1e-12)
            ),
            "time_s": time_s,
            "primary_symmetric_ccf": primary_symmetric_ccf,
            "disturbed_symmetric_ccf": disturbed_symmetric_ccf,
            "group_time_s": group_time_s,
            "measured_group_time_s": public_measurement.group_time_s,
            "measured_phi_tu_rad": public_measurement.phi_tu_rad,
            "measured_omega_rad_s": public_measurement.omega_inst_rad_s,
            "distance_km": distance_km,
            "nominal_period_s": nominal_period_s,
            "mode": mode,
        }

    def test_full_synthetic_matrix_validates_bensen_and_lin_independently(self):
        cases_by_convention = self._phase_convention_synthetic_cases()
        summaries = []
        correct_results = {}
        for convention in self.mod.PhaseConvention:
            cases = cases_by_convention[convention.name]
            convention_rows = []
            for noisy in (False, True):
                for specification in cases:
                    with self.subTest(
                        convention=convention.name,
                        noisy=noisy,
                        specification=specification,
                    ):
                        row = self._run_phase_convention_synthetic_case(
                            synthesis_convention=convention,
                            processing_convention=convention,
                            specification=specification,
                            noisy=noisy,
                        )
                        convention_rows.append(row)
            correct_results[convention] = convention_rows
            noise_free_errors = [
                row["relative_error"]
                for row in convention_rows
                if row["noise_free"]
            ]
            noisy_errors = [
                row["relative_error"]
                for row in convention_rows
                if not row["noise_free"]
            ]
            noise_free_kernel_errors = [
                row["kernel_relative_error"]
                for row in convention_rows
                if row["noise_free"]
            ]
            noisy_kernel_errors = [
                row["kernel_relative_error"]
                for row in convention_rows
                if not row["noise_free"]
            ]
            self.assertLessEqual(max(noise_free_errors), 0.005)
            self.assertLessEqual(float(np.median(noisy_errors)), 0.02)
            self.assertLessEqual(max(noise_free_kernel_errors), 0.005)
            self.assertLessEqual(
                float(np.median(noisy_kernel_errors)),
                0.02,
            )
            validate = getattr(self.mod, "validate_phase_convention", None)
            self.assertIsNotNone(
                validate,
                "full synthetic matrix requires validate_phase_convention",
            )
            summary = validate(
                convention=convention,
                true_phase_velocities_km_s=np.array(
                    [
                        row["true_phase_velocity_km_s"]
                        for row in convention_rows
                    ]
                ),
                recovered_phase_velocities_km_s=np.array(
                    [
                        row["recovered_phase_velocity_km_s"]
                        for row in convention_rows
                    ]
                ),
                noise_free_mask=np.array(
                    [row["noise_free"] for row in convention_rows]
                ),
            )
            summaries.append(summary)

        bensen_rows = correct_results[
            self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        ]
        lin_rows = correct_results[
            self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF
        ]
        for bensen_row, lin_row in zip(bensen_rows, lin_rows):
            self.assertFalse(
                np.shares_memory(
                    bensen_row["prepared"],
                    lin_row["prepared"],
                )
            )

        self.assertEqual(len(summaries), 2)
        for summary in summaries:
            self.assertEqual(summary.total_count, 2 * len(cases))
            self.assertEqual(summary.valid_count, summary.total_count)
            self.assertEqual(summary.failure_count, 0)
            self.assertLessEqual(summary.noise_free_max_relative_error, 0.005)
            self.assertLessEqual(summary.noisy_median_relative_error, 0.02)
            self.assertEqual(summary.status, "thresholds_passed")
            self.assertEqual(
                json.loads(json.dumps(summary.metadata))["convention"],
                summary.convention.name,
            )
            with self.assertRaises(FrozenInstanceError):
                summary.status = "changed"
            with self.assertRaises(TypeError):
                summary.metadata["status"] = "changed"

    def test_full_synthetic_matrix_routes_every_case_through_public_measurement(self):
        cases_by_convention = self._phase_convention_synthetic_cases()
        rows = []
        kernel_names = (
            "measure_phase_curve",
            "prepare_phase_waveform",
            "gaussian_filter_bank",
            "normalized_log_energy",
            "find_candidate_ridges",
            "select_fundamental_ridge",
            "refine_group_arrival",
            "interpolate_analytic_phase_at_arrival",
            "estimate_instantaneous_frequency",
            "unwrap_phase_along_frequency",
            "_phase_measurement_snr",
            "phase_speed_candidates_from_measurement",
        )
        with contextlib.ExitStack() as stack:
            public_measurement = stack.enter_context(
                mock.patch.object(
                    self.mod,
                    "measure_single_period",
                    wraps=self.mod.measure_single_period,
                )
            )
            kernel_spies = {
                name: stack.enter_context(
                    mock.patch.object(
                        self.mod,
                        name,
                        wraps=getattr(self.mod, name),
                    )
                )
                for name in kernel_names
            }
            for convention in self.mod.PhaseConvention:
                cases = cases_by_convention[convention.name]
                for noisy in (False, True):
                    for specification in cases:
                        rows.append(
                            self._run_phase_convention_synthetic_case(
                                synthesis_convention=convention,
                                processing_convention=convention,
                                specification=specification,
                                noisy=noisy,
                            )
                        )

        self.assertEqual(public_measurement.call_count, 36)
        for name in (
            "measure_phase_curve",
            "prepare_phase_waveform",
            "gaussian_filter_bank",
            "normalized_log_energy",
            "find_candidate_ridges",
            "select_fundamental_ridge",
            "unwrap_phase_along_frequency",
            "phase_speed_candidates_from_measurement",
        ):
            self.assertEqual(kernel_spies[name].call_count, 36)
        self.assertEqual(kernel_spies["refine_group_arrival"].call_count, 108)
        self.assertEqual(
            kernel_spies["estimate_instantaneous_frequency"].call_count,
            108,
        )
        self.assertEqual(
            kernel_spies["interpolate_analytic_phase_at_arrival"].call_count,
            108,
        )
        self.assertEqual(kernel_spies["_phase_measurement_snr"].call_count, 108)
        formal_velocity_axis = self.mod.FtanConfig().group_velocities_km_s
        for call in public_measurement.call_args_list:
            self.assertEqual(call.kwargs["vmin_km_s"], 1.6)
            self.assertEqual(call.kwargs["vmax_km_s"], 5.0)
        for call in kernel_spies["find_candidate_ridges"].call_args_list:
            np.testing.assert_array_equal(
                call.kwargs["velocity_axis_km_s"],
                formal_velocity_axis,
            )
        noise_free_errors = [
            row["relative_error"] for row in rows if row["noise_free"]
        ]
        noisy_errors = [
            row["relative_error"] for row in rows if not row["noise_free"]
        ]
        noisy_rows = [row for row in rows if not row["noise_free"]]
        noisy_snr = np.array([row["snr"] for row in noisy_rows])
        self.assertLessEqual(max(noise_free_errors), 0.005)
        self.assertLessEqual(float(np.median(noisy_errors)), 0.02)
        self.assertLess(float(np.min(noisy_snr)), 4.0)
        self.assertGreater(float(np.max(noisy_snr)), 4.0)
        self.assertGreaterEqual(
            int(np.count_nonzero((noisy_snr > 4.0) & (noisy_snr <= 8.0))),
            4,
        )
        self.assertGreaterEqual(
            len({row["noise_seed"] for row in noisy_rows}),
            6,
        )
        self.assertEqual(
            {row["noise_amplitude"] for row in noisy_rows},
            {0.007, 0.008, 0.009},
        )
        self.assertLessEqual(float(np.percentile(noisy_errors, 90)), 0.002)
        self.assertLessEqual(float(np.percentile(noisy_errors, 95)), 0.004)
        self.assertLessEqual(max(noisy_errors), 0.004)

    def test_synthetic_competitors_form_visible_formal_energy_peaks_and_move_ridges(self):
        velocity = self.mod.FtanConfig().group_velocities_km_s
        cases_by_convention = self._phase_convention_synthetic_cases()
        for convention in self.mod.PhaseConvention:
            challenge_cases = [
                specification
                for specification in cases_by_convention[convention.name]
                if specification[-1] in {"dual", "coda"}
            ]
            self.assertEqual(
                {specification[-1] for specification in challenge_cases},
                {"dual", "coda"},
            )
            for specification in challenge_cases:
                with self.subTest(
                    convention=convention.name,
                    mode=specification[-1],
                ):
                    row = self._run_phase_convention_synthetic_case(
                        synthesis_convention=convention,
                        processing_convention=convention,
                        specification=specification,
                        noisy=False,
                    )
                    signal_start_s = row["distance_km"] / 5.0
                    signal_end_s = row["distance_km"] / 1.6
                    self.assertGreater(row["competitor_center_s"], signal_start_s)
                    self.assertLess(row["competitor_center_s"], signal_end_s)
                    self.assertGreaterEqual(
                        abs(
                            row["competitor_center_s"]
                            - row["group_time_s"]
                        ),
                        row["nominal_period_s"],
                    )
                    self.assertGreaterEqual(row["competitor_energy_ratio"], 0.01)

                    clean_trace = self.mod.DatTrace(
                        pair_name="clean_competitor_control",
                        distance_km=row["distance_km"],
                        dt_s=0.02,
                        time_s=row["time_s"],
                        positive_lag=row["primary_symmetric_ccf"],
                        negative_lag_reversed=row["primary_symmetric_ccf"],
                        symmetric_waveform=row["primary_symmetric_ccf"],
                        lon_a=0.0,
                        lat_a=0.0,
                        lon_b=0.1,
                        lat_b=0.1,
                    )
                    disturbed_trace = self.mod.DatTrace(
                        pair_name="disturbed_competitor",
                        distance_km=row["distance_km"],
                        dt_s=0.02,
                        time_s=row["time_s"],
                        positive_lag=row["disturbed_symmetric_ccf"],
                        negative_lag_reversed=row["disturbed_symmetric_ccf"],
                        symmetric_waveform=row["disturbed_symmetric_ccf"],
                        lon_a=0.0,
                        lat_a=0.0,
                        lon_b=0.1,
                        lat_b=0.1,
                    )
                    challenge_periods = (
                        row["competitor_period_s"]
                        * np.array([0.96, 1.0, 1.04])
                    )
                    curve_arguments = {
                        "periods_s": challenge_periods,
                        "velocity_axis_km_s": velocity,
                        "alpha": 20.0,
                        "beta1": 0.5,
                        "beta2": 1.0,
                        "convention": convention,
                    }
                    clean_curve = self.mod.measure_phase_curve(
                        clean_trace,
                        **curve_arguments,
                    )
                    disturbed_curve = self.mod.measure_phase_curve(
                        disturbed_trace,
                        **curve_arguments,
                    )
                    self.assertIsNotNone(clean_curve)
                    self.assertIsNotNone(disturbed_curve)
                    target_row = 1
                    primary_energy = clean_curve.scaled_log_energy[target_row]
                    disturbed_energy = (
                        disturbed_curve.scaled_log_energy[target_row]
                    )
                    local_peaks = np.flatnonzero(
                        (disturbed_energy[1:-1] > disturbed_energy[:-2])
                        & (
                            disturbed_energy[1:-1]
                            >= disturbed_energy[2:]
                        )
                    ) + 1
                    expected_index = int(
                        np.argmin(
                            np.abs(
                                velocity
                                - row["distance_km"]
                                / row["competitor_center_s"]
                            )
                        )
                    )
                    self.assertGreater(local_peaks.size, 0)
                    peak_index = int(
                        local_peaks[
                            np.argmin(np.abs(local_peaks - expected_index))
                        ]
                    )
                    primary_peak = int(np.argmax(primary_energy))
                    self.assertLessEqual(abs(peak_index - expected_index), 8)
                    self.assertGreaterEqual(abs(peak_index - primary_peak), 10)
                    comparison_start = max(0, expected_index - 8)
                    comparison_stop = min(
                        velocity.size,
                        expected_index + 9,
                    )
                    self.assertGreaterEqual(
                        disturbed_energy[peak_index]
                        - float(
                            np.max(
                                primary_energy[
                                    comparison_start:comparison_stop
                                ]
                            )
                        ),
                        0.15,
                    )
                    between = slice(
                        min(peak_index, primary_peak),
                        max(peak_index, primary_peak) + 1,
                    )
                    self.assertGreaterEqual(
                        disturbed_energy[peak_index]
                        - float(np.min(disturbed_energy[between])),
                        0.15,
                    )
                    ridge_shift_bins = abs(
                        int(disturbed_curve.ridge.row_indices[target_row])
                        - int(clean_curve.ridge.row_indices[target_row])
                    )
                    self.assertGreaterEqual(ridge_shift_bins, 5)
                    for ridge in (clean_curve.ridge, disturbed_curve.ridge):
                        self.assertTrue(np.all(ridge.row_indices > 0))
                        self.assertTrue(
                            np.all(ridge.row_indices < velocity.size - 1)
                        )
                    self.assertGreaterEqual(
                        abs(
                            disturbed_curve.group_times_s[target_row]
                            - clean_curve.group_times_s[target_row]
                        ),
                        2.0 * 0.02,
                    )
                    self.assertLessEqual(row["relative_error"], 0.005)

    def test_synthetic_matrix_rejects_wrong_sign_derivative_fixed_term_and_convention(self):
        cases_by_convention = self._phase_convention_synthetic_cases()
        bensen = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        lin = self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF
        bensen_cases = cases_by_convention[bensen.name]
        lin_cases = cases_by_convention[lin.name]
        wrong_sign_rows = []
        for noisy in (False, True):
            for specification in bensen_cases:
                wrong_sign_rows.append(
                    self._run_phase_convention_synthetic_case(
                        synthesis_convention=bensen,
                        processing_convention=bensen,
                        specification=specification,
                        noisy=noisy,
                        wrong_phase_sign=True,
                    )
                )
        wrong_sign_summary = self.mod.validate_phase_convention(
            convention=bensen,
            true_phase_velocities_km_s=np.array(
                [
                    row["true_phase_velocity_km_s"]
                    for row in wrong_sign_rows
                ]
            ),
            recovered_phase_velocities_km_s=np.array(
                [
                    row["recovered_phase_velocity_km_s"]
                    for row in wrong_sign_rows
                ]
            ),
            noise_free_mask=np.array(
                [row["noise_free"] for row in wrong_sign_rows]
            ),
        )
        self.assertNotEqual(wrong_sign_summary.status, "thresholds_passed")
        self.assertGreater(
            wrong_sign_summary.noise_free_max_relative_error,
            0.02,
        )

        negative_controls = (
            {
                "name": "lin_without_negative_derivative",
                "arguments": {
                    "synthesis_convention": lin,
                    "processing_convention": lin,
                    "specification": lin_cases[4],
                    "noisy": False,
                    "skip_preparation": True,
                },
            },
            {
                "name": "wrong_positive_fixed_phase",
                "arguments": {
                    "synthesis_convention": bensen,
                    "processing_convention": bensen,
                    "specification": bensen_cases[0],
                    "noisy": False,
                    "wrong_fixed_phase": True,
                },
            },
            {
                "name": "lin_processing_of_bensen_input",
                "arguments": {
                    "synthesis_convention": bensen,
                    "processing_convention": lin,
                    "specification": bensen_cases[5],
                    "noisy": False,
                },
            },
        )
        for control in negative_controls:
            with self.subTest(control=control["name"]):
                try:
                    row = self._run_phase_convention_synthetic_case(
                        **control["arguments"]
                    )
                    error = row["relative_error"]
                except ValueError:
                    error = float("inf")
                self.assertGreater(error, 0.02)

    def test_phase_convention_validation_rejects_bad_axes_and_structures_failures(self):
        validate = self.require_module_attribute("validate_phase_convention")
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        valid_arguments = {
            "convention": convention,
            "true_phase_velocities_km_s": np.array([2.5, 3.0]),
            "recovered_phase_velocities_km_s": np.array([2.5, 3.03]),
            "noise_free_mask": np.array([True, False]),
        }
        bad_calls = (
            (
                {
                    **valid_arguments,
                    "true_phase_velocities_km_s": np.array([True, True]),
                },
                "true_phase_velocities_km_s",
            ),
            (
                {
                    **valid_arguments,
                    "true_phase_velocities_km_s": np.array([[2.5, 3.0]]),
                },
                "one-dimensional",
            ),
            (
                {
                    **valid_arguments,
                    "recovered_phase_velocities_km_s": np.array([True, True]),
                },
                "recovered_phase_velocities_km_s",
            ),
            (
                {
                    **valid_arguments,
                    "recovered_phase_velocities_km_s": np.array([2.5]),
                },
                "same shape",
            ),
            (
                {
                    **valid_arguments,
                    "noise_free_mask": np.array([1, 0]),
                },
                "boolean",
            ),
            (
                {
                    **valid_arguments,
                    "valid_mask": np.array([True]),
                },
                "valid_mask",
            ),
            (
                {
                    **valid_arguments,
                    "noise_free_max_tolerance": True,
                },
                "noise_free_max_tolerance",
            ),
            (
                {
                    **valid_arguments,
                    "convention": "bensen",
                },
                "PhaseConvention",
            ),
        )
        for arguments, message in bad_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate(**arguments)

        invalid = validate(
            **{
                **valid_arguments,
                "recovered_phase_velocities_km_s": np.array(
                    [np.nan, -1.0]
                ),
            }
        )
        self.assertEqual(invalid.total_count, 2)
        self.assertEqual(invalid.valid_count, 0)
        self.assertEqual(invalid.failure_count, 2)
        self.assertEqual(invalid.status, "invalid_measurements")
        self.assertTrue(np.isnan(invalid.noise_free_max_relative_error))
        self.assertTrue(np.isnan(invalid.noisy_median_relative_error))
        decoded = json.loads(json.dumps(invalid.metadata))
        self.assertIsNone(decoded["noise_free_max_relative_error"])
        self.assertIsNone(decoded["noisy_median_relative_error"])

    def test_select_branch_sequence_prefers_smooth_reference_like_series(self):
        periods = np.array([0.5, 1.0, 1.5, 2.0], dtype=float)
        group_velocities = np.array([2.05, 2.10, 2.14, 2.18], dtype=float)
        candidate_lists = [
            [
                self.mod.PhaseCandidate(branch=-1, phase_velocity_km_s=1.40, phase_slowness_s_km=1 / 1.40),
                self.mod.PhaseCandidate(branch=0, phase_velocity_km_s=2.20, phase_slowness_s_km=1 / 2.20),
            ],
            [
                self.mod.PhaseCandidate(branch=-1, phase_velocity_km_s=1.32, phase_slowness_s_km=1 / 1.32),
                self.mod.PhaseCandidate(branch=0, phase_velocity_km_s=2.24, phase_slowness_s_km=1 / 2.24),
            ],
            [
                self.mod.PhaseCandidate(branch=-1, phase_velocity_km_s=1.28, phase_slowness_s_km=1 / 1.28),
                self.mod.PhaseCandidate(branch=0, phase_velocity_km_s=2.28, phase_slowness_s_km=1 / 2.28),
            ],
            [
                self.mod.PhaseCandidate(branch=-1, phase_velocity_km_s=1.20, phase_slowness_s_km=1 / 1.20),
                self.mod.PhaseCandidate(branch=0, phase_velocity_km_s=2.31, phase_slowness_s_km=1 / 2.31),
            ],
        ]

        selected = self.mod.select_branch_sequence(
            periods_s=periods,
            group_velocities_km_s=group_velocities,
            candidate_lists=candidate_lists,
        )

        self.assertEqual(selected.branches, [0, 0, 0, 0])
        np.testing.assert_allclose(
            selected.phase_velocities_km_s,
            np.array([2.20, 2.24, 2.28, 2.31]),
            atol=1e-6,
        )

    @staticmethod
    def _ridge_image(
        row_indices,
        *,
        n_velocity=31,
        peak=1.0,
        background=0.0,
    ):
        rows = np.asarray(row_indices, dtype=int)
        image = np.full((rows.size, n_velocity), background, dtype=float)
        image[np.arange(rows.size), rows] = peak
        return image

    @staticmethod
    def _brute_second_order_ridge(energy, velocity, beta1, beta2, blocked=None):
        energy = np.asarray(energy, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        dv = float(velocity[1] - velocity[0])
        if blocked is None:
            blocked = np.zeros_like(energy, dtype=bool)
        best_rows = None
        best_cost = float("inf")
        for rows in itertools.product(range(velocity.size), repeat=energy.shape[0]):
            if any(blocked[col, row] for col, row in enumerate(rows)):
                continue
            row_array = np.asarray(rows, dtype=int)
            cost = -float(np.sum(energy[np.arange(energy.shape[0]), rows]))
            if row_array.size > 1:
                cost += beta1 * dv * float(
                    np.sum(np.abs(np.diff(row_array)))
                )
            if row_array.size > 2:
                cost += beta2 * dv * float(
                    np.sum(np.abs(np.diff(row_array, n=2)))
                )
            if cost < best_cost:
                best_cost = cost
                best_rows = row_array
        return best_rows, best_cost

    def test_exact_l1_distance_transform_matches_brute_force_with_physical_dv(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        rng = np.random.default_rng(20260717)
        for dv in (0.07, 0.20):
            velocity = 1.6 + dv * np.arange(5, dtype=float)
            for case in range(4):
                energy = rng.uniform(0.0, 1.0, size=(5, velocity.size))
                # The masks exercise both low- and high-index endpoint
                # extrapolation in the L1 query x = 2*b-c.
                blocked = np.zeros_like(energy, dtype=bool)
                if case == 1:
                    blocked[1, 1:4] = True
                elif case == 2:
                    blocked[2, :3] = True
                elif case == 3:
                    blocked[3, 2:] = True
                expected_rows, expected_cost = self._brute_second_order_ridge(
                    energy,
                    velocity,
                    beta1=0.9,
                    beta2=1.7,
                    blocked=blocked,
                )
                actual_rows, actual_cost = trace(
                    energy,
                    velocity,
                    beta1=0.9,
                    beta2=1.7,
                    blocked=blocked,
                )
                np.testing.assert_array_equal(actual_rows, expected_rows)
                self.assertAlmostEqual(actual_cost, expected_cost, places=12)

    def test_exact_l1_ties_choose_smallest_predecessor_at_endpoint_and_outside(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        velocity = np.array([1.0, 2.0, 3.0], dtype=float)
        cases = (
            # Terminal state (b=1,c=0) queries x=2, exactly the upper
            # endpoint. Predecessors a=0 and a=2 have equal objective.
            (
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 10.0, 0.0],
                        [10.0, 0.0, 0.0],
                    ]
                ),
                np.array([0, 1, 0]),
            ),
            # Terminal state (b=2,c=0) queries x=4, beyond the upper
            # endpoint. Linear extrapolation must preserve the same lower
            # predecessor tie-break.
            (
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 10.0],
                        [10.0, 0.0, 0.0],
                    ]
                ),
                np.array([0, 2, 0]),
            ),
            # The symmetric x=-2 lower-end extrapolation already has the
            # smaller predecessor at the current backward-scan index.
            (
                np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [10.0, 0.0, 0.0],
                        [0.0, 0.0, 10.0],
                    ]
                ),
                np.array([0, 0, 2]),
            ),
        )
        for energy, expected_rows in cases:
            with self.subTest(expected_rows=expected_rows.tolist()):
                brute_rows, brute_cost = self._brute_second_order_ridge(
                    energy,
                    velocity,
                    beta1=0.0,
                    beta2=0.5,
                )
                actual_rows, actual_cost = trace(
                    energy,
                    velocity,
                    beta1=0.0,
                    beta2=0.5,
                )
                np.testing.assert_array_equal(brute_rows, expected_rows)
                np.testing.assert_array_equal(actual_rows, expected_rows)
                self.assertAlmostEqual(actual_cost, brute_cost, places=12)

    def test_grid_equivalent_first_order_tie_uses_canonical_index_cost(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        velocity = np.array([1.00, 1.13, 1.26], dtype=float)
        energy = np.array(
            [
                [1.0, 2.0 / 3.0, 0.0],
                [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
                [0.0, 2.0 / 3.0, 1.0 / 3.0],
            ],
            dtype=float,
        )
        blocked = np.array(
            [
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ],
            dtype=bool,
        )
        expected_rows, expected_cost = self._brute_second_order_ridge(
            energy,
            velocity,
            beta1=0.3,
            beta2=0.0,
            blocked=blocked,
        )

        actual_rows, actual_cost = trace(
            energy,
            velocity,
            beta1=0.3,
            beta2=0.0,
            blocked=blocked,
        )

        np.testing.assert_array_equal(expected_rows, np.array([0, 0, 1]))
        np.testing.assert_array_equal(actual_rows, expected_rows)
        self.assertAlmostEqual(
            actual_cost,
            -2.294333333333333,
            places=14,
        )
        self.assertAlmostEqual(actual_cost, expected_cost, places=14)

    def test_cost_difference_above_ulp_tie_window_keeps_the_true_optimum(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        velocity = np.array([1.0, 2.0], dtype=float)
        meaningful_delta = 48.0 * np.finfo(float).eps
        energy = np.array(
            [
                [0.0, 0.0],
                [0.0, meaningful_delta],
            ],
            dtype=float,
        )

        rows, cost = trace(
            energy,
            velocity,
            beta1=0.0,
            beta2=0.0,
        )

        np.testing.assert_array_equal(rows, np.array([0, 1]))
        self.assertEqual(cost, -meaningful_delta)

    def test_three_period_path_consumes_the_problem_tie_budget_only_once(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        tolerance = self.require_module_attribute("_ridge_cost_tolerance")
        eps = np.finfo(float).eps
        velocity = np.array([1.0, 2.0], dtype=float)
        energy = eps * np.array(
            [
                [56.0, 78.0],
                [56.0, 82.0],
                [70.0, 87.0],
            ],
            dtype=float,
        )
        expected_rows, strict_minimum = self._brute_second_order_ridge(
            energy,
            velocity,
            beta1=0.0,
            beta2=0.0,
        )
        problem_tolerance = tolerance(
            energy=energy,
            velocity_count=velocity.size,
            dv=1.0,
            beta1=0.0,
            beta2=0.0,
        )

        rows, cost = trace(
            energy,
            velocity,
            beta1=0.0,
            beta2=0.0,
        )

        np.testing.assert_array_equal(expected_rows, np.array([1, 1, 1]))
        self.assertEqual(strict_minimum, -247.0 * eps)
        self.assertEqual(problem_tolerance, 32.0 * eps)
        self.assertLessEqual(cost, strict_minimum + problem_tolerance)
        self.assertFalse(np.array_equal(rows, np.array([0, 0, 1])))

    def test_single_period_terminal_uses_the_same_ulp_tie_rule(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        velocity = np.array([1.0, 2.0], dtype=float)
        roundoff_delta = 16.0 * np.finfo(float).eps

        rows, cost = trace(
            np.array([[0.0, roundoff_delta]], dtype=float),
            velocity,
            beta1=0.0,
            beta2=0.0,
        )

        np.testing.assert_array_equal(rows, np.array([0]))
        self.assertEqual(cost, 0.0)

    def test_fixed_seed_discrete_ties_match_lexicographic_brute_paths(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        rng = np.random.default_rng(240717)
        velocity = 1.0 + 0.13 * np.arange(4, dtype=float)
        for case in range(24):
            energy = rng.integers(0, 4, size=(4, velocity.size)) / 3.0
            blocked = rng.random(energy.shape) < 0.12
            for period_index in range(energy.shape[0]):
                if np.all(blocked[period_index]):
                    blocked[period_index, 0] = False
            beta1, beta2 = (
                ((0.0, 0.0), (0.3, 0.0), (0.3, 0.6))[case % 3]
            )
            expected_rows, expected_cost = self._brute_second_order_ridge(
                energy,
                velocity,
                beta1=beta1,
                beta2=beta2,
                blocked=blocked,
            )
            actual_rows, actual_cost = trace(
                energy,
                velocity,
                beta1=beta1,
                beta2=beta2,
                blocked=blocked,
            )
            with self.subTest(
                case=case,
                beta1=beta1,
                beta2=beta2,
            ):
                np.testing.assert_array_equal(actual_rows, expected_rows)
                self.assertAlmostEqual(actual_cost, expected_cost, places=13)

    def test_exact_ridge_handles_one_and_two_periods_and_breaks_ties_by_low_index(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        velocity = 1.6 + 0.01 * np.arange(8, dtype=float)

        one_rows, one_cost = trace(
            np.zeros((1, velocity.size), dtype=float),
            velocity,
            beta1=0.0,
            beta2=0.0,
        )
        two_rows, two_cost = trace(
            np.zeros((2, velocity.size), dtype=float),
            velocity,
            beta1=0.0,
            beta2=0.0,
        )

        np.testing.assert_array_equal(one_rows, np.array([0]))
        np.testing.assert_array_equal(two_rows, np.array([0, 0]))
        self.assertEqual(one_cost, 0.0)
        self.assertEqual(two_cost, 0.0)

    def test_exact_ridge_reports_unreachable_and_rejects_nonuniform_or_nonfinite_grids(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        velocity = 1.6 + 0.01 * np.arange(8, dtype=float)
        energy = np.zeros((4, velocity.size), dtype=float)
        rows, cost = trace(
            energy,
            velocity,
            beta1=1.0,
            beta2=2.0,
            blocked=np.ones_like(energy, dtype=bool),
        )
        self.assertIsNone(rows)
        self.assertTrue(math.isinf(cost))

        with self.assertRaisesRegex(ValueError, "uniform"):
            trace(
                energy,
                np.array([1.6, 1.61, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68]),
                beta1=1.0,
                beta2=2.0,
            )
        bad = energy.copy()
        bad[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            trace(
                bad,
                velocity,
                beta1=1.0,
                beta2=2.0,
            )

    def test_dynamic_programming_recovers_continuous_peak_and_ignores_isolated_outlier(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 5.05, 0.05)
        velocity = 1.6 + 0.01 * np.arange(341, dtype=float)
        ridge_rows = 80 + np.rint(8.0 * np.sin(np.linspace(0.0, math.pi, periods.size))).astype(int)
        energy = np.full((periods.size, velocity.size), 0.02, dtype=float)
        amplitude = np.full_like(energy, 0.02)
        energy[np.arange(periods.size), ridge_rows] = 1.0
        amplitude[np.arange(periods.size), ridge_rows] = 1.0
        outlier_col = periods.size // 2
        outlier_row = 250
        energy[outlier_col, outlier_row] = 1.4
        amplitude[outlier_col, outlier_row] = 1.0

        candidate = find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=amplitude,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=1.0,
            beta2=2.0,
            max_candidates=1,
        )[0]

        np.testing.assert_array_equal(candidate.row_indices, ridge_rows)
        self.assertNotEqual(candidate.row_indices[outlier_col], outlier_row)
        self.assertTrue(candidate.quality.accepted)

    def test_energy_drives_dp_and_amplitude_only_drives_point_validity(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 3.0, 0.05)
        velocity = 1.6 + 0.01 * np.arange(31, dtype=float)
        energy = np.zeros((periods.size, velocity.size), dtype=float)
        amplitude = np.zeros_like(energy)
        energy[:, 7] = 1.0
        energy[:, 18] = 0.4
        amplitude[:, 7] = 0.149
        amplitude[:, 18] = 1.0

        candidate = find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=amplitude,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=0.0,
            beta2=0.0,
            max_candidates=1,
        )[0]

        np.testing.assert_array_equal(
            candidate.row_indices,
            np.full(periods.size, 7, dtype=int),
        )
        self.assertFalse(np.any(candidate.valid))
        self.assertEqual(candidate.quality.reason, "insufficient_coverage")

    def test_ridge_qc_accepts_one_or_two_gaps_and_rejects_three(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 3.5, 0.05)
        velocity = 1.6 + 0.01 * np.arange(31, dtype=float)
        rows = np.full(periods.size, 12, dtype=int)
        energy = self._ridge_image(rows)
        for gap_length in (1, 2, 3):
            with self.subTest(gap_length=gap_length):
                amplitude = energy.copy()
                amplitude[6 : 6 + gap_length, 12] = 0.149
                candidate = find_candidate_ridges(
                    scaled_log_energy=energy,
                    normalized_envelope_amplitude=amplitude,
                    periods_s=periods,
                    velocity_axis_km_s=velocity,
                    beta1=1.0,
                    beta2=2.0,
                    max_candidates=1,
                )[0]
                if gap_length <= 2:
                    self.assertTrue(candidate.quality.accepted)
                    self.assertEqual(candidate.quality.reason, "accepted")
                else:
                    self.assertFalse(candidate.quality.accepted)
                    self.assertEqual(candidate.quality.reason, "ridge_discontinuous")
                self.assertEqual(candidate.quality.max_gap, gap_length)

    def test_ridge_qc_enforces_coverage_jump_and_valid_boundary_denominators(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 3.5, 0.05)
        velocity = 1.6 + 0.01 * np.arange(41, dtype=float)

        coverage_rows = np.full(periods.size, 20, dtype=int)
        coverage_energy = self._ridge_image(coverage_rows, n_velocity=velocity.size)
        coverage_amp = coverage_energy.copy()
        coverage_amp[[1, 4, 7, 10, 13], 20] = 0.149
        coverage = find_candidate_ridges(
            scaled_log_energy=coverage_energy,
            normalized_envelope_amplitude=coverage_amp,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=0.0,
            beta2=0.0,
            max_candidates=1,
        )[0]
        self.assertLess(coverage.quality.coverage, 0.80)
        self.assertEqual(coverage.quality.reason, "insufficient_coverage")

        jump_rows = np.full(periods.size, 5, dtype=int)
        jump_rows[8:12] = 35
        jump_energy = self._ridge_image(jump_rows, n_velocity=velocity.size)
        jump = find_candidate_ridges(
            scaled_log_energy=jump_energy,
            normalized_envelope_amplitude=jump_energy,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=0.0,
            beta2=0.0,
            max_candidates=1,
        )[0]
        self.assertGreater(jump.quality.jump_fraction, 0.05)
        self.assertEqual(jump.quality.reason, "ridge_jump")

        boundary_rows = np.full(periods.size, 20, dtype=int)
        boundary_rows[:3] = 0
        boundary_energy = self._ridge_image(boundary_rows, n_velocity=velocity.size)
        boundary_amp = boundary_energy.copy()
        # Boundary occupation describes the full traced path. An invalid
        # amplitude point still counts as a DP path point at the outer row.
        boundary_amp[0, 0] = 0.149
        boundary = find_candidate_ridges(
            scaled_log_energy=boundary_energy,
            normalized_envelope_amplitude=boundary_amp,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=0.0,
            beta2=0.0,
            max_candidates=1,
        )[0]
        self.assertAlmostEqual(boundary.quality.boundary_fraction, 3.0 / 20.0)
        self.assertGreater(boundary.quality.boundary_fraction, 0.10)
        self.assertEqual(boundary.quality.reason, "ridge_boundary")

    def test_jump_fraction_uses_only_adjacent_pairs_whose_columns_are_valid(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 3.5, 0.05)
        velocity = 1.6 + 0.01 * np.arange(41, dtype=float)
        rows = np.full(periods.size, 5, dtype=int)
        rows[9:11] = 35
        energy = self._ridge_image(rows, n_velocity=velocity.size)
        amplitude = energy.copy()
        amplitude[8:12, rows[8:12]] = 0.149

        candidate = find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=amplitude,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=0.0,
            beta2=0.0,
            max_candidates=1,
        )[0]

        self.assertEqual(candidate.quality.jump_fraction, 0.0)

    def test_full_grid_corridor_forces_a_crossing_branch_to_detour(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 5.05, 0.05)
        velocity = 1.6 + 0.01 * np.arange(341, dtype=float)
        ridge_a = np.rint(np.linspace(80, 120, periods.size)).astype(int)
        ridge_b = np.rint(np.linspace(120, 80, periods.size)).astype(int)
        ordering = ridge_a - ridge_b
        self.assertLess(np.min(ordering), 0)
        self.assertGreater(np.max(ordering), 0)
        energy = np.zeros((periods.size, velocity.size), dtype=float)
        amplitude = np.zeros_like(energy)
        energy[np.arange(periods.size), ridge_a] = 1.0
        energy[np.arange(periods.size), ridge_b] = 0.9
        amplitude[np.arange(periods.size), ridge_a] = 1.0
        amplitude[np.arange(periods.size), ridge_b] = 0.9

        candidates = find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=amplitude,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=1.0,
            beta2=2.0,
            max_candidates=3,
        )

        self.assertGreaterEqual(len(candidates), 2)
        first, second = candidates[:2]
        np.testing.assert_array_equal(first.row_indices, ridge_a)
        # Far from the intersection the rerun recovers the other physical
        # ridge. At the crossing it must detour around the hard corridor,
        # producing a deterministic invalid gap rather than crossing the mask.
        np.testing.assert_array_equal(second.row_indices[:15], ridge_b[:15])
        np.testing.assert_array_equal(second.row_indices[-15:], ridge_b[-15:])
        self.assertEqual(second.quality.reason, "ridge_discontinuous")
        self.assertGreater(second.quality.max_gap, 2)
        self.assertTrue(
            np.all(
                np.abs(
                    first.group_velocities_km_s
                    - second.group_velocities_km_s
                )
                > 0.05 + 1e-12
            )
        )
        common = first.valid & second.valid
        self.assertGreater(np.count_nonzero(common), 0)
        distinct_fraction = np.mean(
            np.abs(
                first.group_velocities_km_s[common]
                - second.group_velocities_km_s[common]
            )
            >= 0.05 - 1e-12
        )
        self.assertGreaterEqual(distinct_fraction, 0.50)

    def test_full_grid_corridor_keeps_parallel_ridge_at_point_zero_six_km_s(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 5.05, 0.05)
        velocity = 1.6 + 0.01 * np.arange(341, dtype=float)
        primary_rows = np.full(periods.size, 100, dtype=int)
        parallel_rows = np.full(periods.size, 106, dtype=int)
        energy = np.zeros((periods.size, velocity.size), dtype=float)
        amplitude = np.zeros_like(energy)
        energy[:, 100] = 1.0
        amplitude[:, 100] = 1.0
        energy[:, 106] = 0.9
        amplitude[:, 106] = 0.9

        candidates = find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=amplitude,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=1.0,
            beta2=2.0,
            max_candidates=3,
        )

        self.assertGreaterEqual(len(candidates), 2)
        np.testing.assert_array_equal(candidates[0].row_indices, primary_rows)
        np.testing.assert_array_equal(candidates[1].row_indices, parallel_rows)
        np.testing.assert_allclose(
            candidates[1].group_velocities_km_s
            - candidates[0].group_velocities_km_s,
            0.06,
            rtol=0.0,
            atol=1e-12,
        )

    def test_select_fundamental_ridge_prioritizes_wang_limit_then_qc_metrics(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        select = self.require_module_attribute("select_fundamental_ridge")
        periods = np.arange(2.5, 5.05, 0.05)
        velocity = 1.6 + 0.01 * np.arange(341, dtype=float)
        fast = np.full(periods.size, 180, dtype=int)  # 3.40 km/s, high energy
        fundamental = np.full(periods.size, 120, dtype=int)  # 2.80 km/s
        energy = np.zeros((periods.size, velocity.size), dtype=float)
        amplitude = np.zeros_like(energy)
        energy[np.arange(periods.size), fast] = 1.0
        energy[np.arange(periods.size), fundamental] = 0.75
        amplitude[np.arange(periods.size), fast] = 1.0
        amplitude[np.arange(periods.size), fundamental] = 1.0

        candidates = find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=amplitude,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=1.0,
            beta2=2.0,
            max_candidates=2,
        )
        selected = select(candidates, periods_s=periods)

        self.assertTrue(selected.quality.accepted)
        np.testing.assert_allclose(
            selected.group_velocities_km_s,
            velocity[fundamental],
        )
        self.assertGreater(
            selected.wang_group_limit_pass_count,
            candidates[0].wang_group_limit_pass_count,
        )

    def test_select_fundamental_ridge_uses_coverage_energy_then_roughness_ties(self):
        ridge_quality = self.require_module_attribute("RidgeQuality")
        ridge_result = self.require_module_attribute("RidgeResult")
        select = self.require_module_attribute("select_fundamental_ridge")
        periods = np.arange(2.5, 3.0, 0.05)

        def make(values, valid, coverage, energy):
            values = np.asarray(values, dtype=float)
            valid = np.asarray(valid, dtype=bool)
            rows = np.arange(values.size, dtype=int)
            return ridge_result(
                row_indices=rows,
                group_velocities_km_s=values,
                valid=valid,
                wang_group_limit_pass_count=0,
                quality=ridge_quality(
                    accepted=True,
                    reason="accepted",
                    coverage=coverage,
                    max_gap=0,
                    jump_fraction=0.0,
                    boundary_fraction=0.0,
                    normalized_energy_integral=energy,
                ),
            )

        smooth = make(np.full(periods.size, 2.8), np.ones(periods.size, bool), 1.0, 0.8)
        rough = make(
            2.8 + 0.01 * np.tile([0.0, 1.0], periods.size // 2),
            np.ones(periods.size, bool),
            1.0,
            0.8,
        )
        lower_energy = make(np.full(periods.size, 2.7), np.ones(periods.size, bool), 1.0, 0.7)
        lower_coverage = make(
            np.full(periods.size, 2.6),
            np.array([True] * (periods.size - 1) + [False]),
            0.9,
            1.0,
        )

        self.assertIs(select([lower_coverage, smooth], periods_s=periods), smooth)
        self.assertIs(select([lower_energy, smooth], periods_s=periods), smooth)
        self.assertIs(select([rough, smooth], periods_s=periods), smooth)

    def test_empty_candidate_selection_has_explicit_failure_without_column_fallback(self):
        select = self.require_module_attribute("select_fundamental_ridge")
        selected = select([], periods_s=np.arange(2.5, 5.05, 0.05))

        self.assertFalse(selected.quality.accepted)
        self.assertEqual(selected.quality.reason, "no_fundamental_ridge")
        self.assertEqual(selected.row_indices.size, 0)

    def test_all_qc_rejected_candidates_select_explicit_no_fundamental_ridge(self):
        ridge_quality = self.require_module_attribute("RidgeQuality")
        ridge_result = self.require_module_attribute("RidgeResult")
        select = self.require_module_attribute("select_fundamental_ridge")
        periods = np.arange(2.5, 3.0, 0.05)
        rejected = ridge_result(
            row_indices=np.zeros(periods.size, dtype=int),
            group_velocities_km_s=np.full(periods.size, 1.6),
            valid=np.ones(periods.size, dtype=bool),
            wang_group_limit_pass_count=periods.size,
            quality=ridge_quality(
                accepted=False,
                reason="ridge_boundary",
                coverage=1.0,
                max_gap=0,
                jump_fraction=0.0,
                boundary_fraction=1.0,
                normalized_energy_integral=10.0,
            ),
        )

        selected = select([rejected], periods_s=periods)

        self.assertFalse(selected.quality.accepted)
        self.assertEqual(selected.quality.reason, "no_fundamental_ridge")
        self.assertEqual(selected.row_indices.size, 0)

    def test_corridor_mask_is_inclusive_physical_and_zero_common_valid_stops_rerun(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 3.0, 0.05)
        velocity = 1.6 + 0.01 * np.arange(31, dtype=float)
        energy = np.zeros((periods.size, velocity.size), dtype=float)
        amplitude = np.zeros_like(energy)
        energy[:, 10] = 1.0
        amplitude[:, 10] = 1.0
        # Exactly +0.05 km/s is inside the first ridge corridor, while +0.06
        # is the next reachable ridge. It has no valid points, so it must not
        # be emitted as a duplicate/empty candidate.
        energy[:, 15] = 0.9
        amplitude[:, 15] = 1.0
        energy[:, 16] = 0.8

        candidates = find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=amplitude,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=0.0,
            beta2=0.0,
            max_candidates=3,
        )

        self.assertEqual(len(candidates), 1)
        np.testing.assert_array_equal(
            candidates[0].row_indices,
            np.full(periods.size, 10, dtype=int),
        )

    def test_ridge_result_arrays_are_immutable(self):
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 3.0, 0.05)
        velocity = 1.6 + 0.01 * np.arange(31, dtype=float)
        rows = np.full(periods.size, 12, dtype=int)
        energy = self._ridge_image(rows)
        candidate = find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=energy,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=1.0,
            beta2=2.0,
            max_candidates=1,
        )[0]

        for array in (
            candidate.row_indices,
            candidate.group_velocities_km_s,
            candidate.valid,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array[0] = array[0]
        with self.assertRaises(FrozenInstanceError):
            candidate.wang_group_limit_pass_count = 0

    def test_refine_group_arrival_recovers_subsample_quadratic_vertex(self):
        refine = self.require_module_attribute("refine_group_arrival")
        time_s = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.51, 0.63, 0.76, 0.9, 1.05]
        )
        true_arrival_s = 0.73
        envelope = 4.0 - (time_s - true_arrival_s) ** 2
        peak_index = int(np.argmax(envelope))
        original_time = time_s.copy()
        original_envelope = envelope.copy()

        result = refine(time_s, envelope, peak_index)

        self.assertTrue(result.refinement_used)
        self.assertEqual(result.status, "refined")
        local_dt = min(
            time_s[peak_index] - time_s[peak_index - 1],
            time_s[peak_index + 1] - time_s[peak_index],
        )
        self.assertLess(abs(result.group_time_s - true_arrival_s), 0.1 * local_dt)
        self.assertAlmostEqual(result.grid_time_s, time_s[peak_index])
        self.assertLessEqual(abs(result.vertex_offset_samples), 1.0)
        np.testing.assert_array_equal(time_s, original_time)
        np.testing.assert_array_equal(envelope, original_envelope)
        with self.assertRaises(FrozenInstanceError):
            result.group_time_s = true_arrival_s

    def test_refine_group_arrival_is_invariant_to_positive_amplitude_scale(self):
        refine = self.require_module_attribute("refine_group_arrival")
        time_s = np.array([0.0, 0.1, 0.2])
        true_arrival_s = 0.13
        unit_shape = 1.0 - (time_s - true_arrival_s) ** 2
        results = []

        for scale in (1e-18, 1.0, 1e18):
            with self.subTest(scale=scale):
                result = refine(
                    time_s,
                    scale * unit_shape,
                    peak_index=1,
                )
                self.assertTrue(result.refinement_used)
                self.assertEqual(result.status, "refined")
                self.assertLess(
                    abs(result.group_time_s - true_arrival_s),
                    0.1 * 0.1,
                )
                results.append(result)

        self.assertEqual(
            {result.status for result in results},
            {"refined"},
        )
        np.testing.assert_allclose(
            [result.group_time_s for result in results],
            true_arrival_s,
            rtol=0.0,
            atol=2e-15,
        )

    def test_refine_group_arrival_handles_extreme_time_scale_and_singular_fit(self):
        refine = self.require_module_attribute("refine_group_arrival")
        time_s = np.array([1e-200, 2e-200, 4e-200])
        true_arrival_s = 2.5e-200
        envelope = 3.0 - ((time_s - true_arrival_s) / 1e-200) ** 2

        refined = refine(time_s, envelope, peak_index=1)

        self.assertTrue(refined.refinement_used)
        self.assertEqual(refined.status, "refined")
        self.assertLess(
            abs(refined.group_time_s - true_arrival_s),
            0.1 * 1e-200,
        )

        with mock.patch.object(
            self.mod.np.linalg,
            "solve",
            side_effect=np.linalg.LinAlgError("synthetic singular fit"),
        ):
            singular = refine(
                np.array([0.0, 0.1, 0.2]),
                np.array([0.9, 1.0, 0.95]),
                peak_index=1,
            )
        self.assertFalse(singular.refinement_used)
        self.assertEqual(singular.status, "quadratic_fit_singular")
        self.assertEqual(singular.group_time_s, 0.1)

    def test_refine_group_arrival_falls_back_without_searching_a_wider_window(self):
        refine = self.require_module_attribute("refine_group_arrival")
        cases = (
            (np.array([3.0, 2.0, 1.0]), 0, "boundary_peak"),
            (np.array([1.0, 0.0, 1.0]), 1, "not_local_peak"),
            (np.array([1.0, 1.0, 1.0]), 1, "degenerate_quadratic"),
            (np.array([0.0, 1.0, 1.9]), 1, "not_local_peak"),
        )
        time_s = np.array([10.0, 10.5, 11.0])

        for envelope, peak_index, expected_status in cases:
            with self.subTest(expected_status=expected_status):
                result = refine(time_s, envelope, peak_index)
                self.assertFalse(result.refinement_used)
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.group_time_s, time_s[peak_index])
                self.assertEqual(result.vertex_offset_samples, 0.0)

    def test_refine_group_arrival_validates_inputs(self):
        refine = self.require_module_attribute("refine_group_arrival")
        valid_time = np.array([0.0, 0.1, 0.2])
        valid_envelope = np.array([0.0, 1.0, 0.0])
        bad_calls = (
            ((valid_time[:2], valid_envelope, 1), "same shape"),
            ((np.array([0.0, 0.1, 0.05]), valid_envelope, 1), "increasing"),
            ((valid_time, np.array([0.0, np.nan, 0.0]), 1), "finite"),
            ((valid_time, valid_envelope, True), "peak_index"),
            ((valid_time, valid_envelope, 3), "peak_index"),
        )

        for arguments, message in bad_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    refine(*arguments)

    def test_refined_arrival_phase_interpolates_analytic_signal_not_principal_angle(self):
        interpolate = self.require_module_attribute(
            "interpolate_analytic_phase_at_arrival"
        )
        time_s = np.array([0.0, 1.0])
        phases = np.deg2rad(np.array([179.0, -179.0]))
        analytic = np.exp(1j * phases)
        original = analytic.copy()

        phase_at_half = interpolate(time_s, analytic, 0.5)

        self.assertAlmostEqual(abs(phase_at_half), np.pi, places=10)
        self.assertGreater(abs(phase_at_half), 3.0)
        np.testing.assert_array_equal(analytic, original)
        for arrival in (-0.1, 1.1):
            with self.subTest(arrival=arrival):
                with self.assertRaisesRegex(ValueError, "within"):
                    interpolate(time_s, analytic, arrival)

    def test_negative_real_axis_phase_composes_with_principal_unwrap(self):
        interpolate = self.require_module_attribute(
            "interpolate_analytic_phase_at_arrival"
        )
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        negative_axis = complex(-1.0, -0.0)

        interpolated_phase = interpolate(
            np.array([0.0, 1.0]),
            np.array([negative_axis, negative_axis]),
            0.0,
        )

        self.assertEqual(interpolated_phase, np.pi)
        composed = unwrap(
            np.array([3.0, 3.5, 4.0]),
            np.array([0.0, interpolated_phase, 0.0]),
            np.zeros(3),
            phase_time_sign=1,
        )
        self.assertTrue(np.all(np.isfinite(composed.unwrapped_phase_rad)))

        defensive = unwrap(
            np.array([3.0, 3.5, 4.0]),
            np.array([0.0, -np.pi, 0.0]),
            np.zeros(3),
            phase_time_sign=1,
        )
        self.assertEqual(defensive.unwrapped_phase_rad[1], np.pi)

    def test_phase_interpolation_preserves_legal_value_just_inside_negative_pi(self):
        interpolate = self.require_module_attribute(
            "interpolate_analytic_phase_at_arrival"
        )
        inside_angle = float(np.nextafter(-np.pi, np.inf))
        analytic_value = complex(
            math.cos(inside_angle),
            math.sin(inside_angle),
        )
        expected = math.atan2(analytic_value.imag, analytic_value.real)

        interpolated = interpolate(
            np.array([0.0, 1.0]),
            np.array([analytic_value, analytic_value]),
            0.0,
        )

        self.assertEqual(interpolated, expected)
        self.assertEqual(interpolated, inside_angle)
        self.assertNotEqual(interpolated, np.pi)

    def test_phase_interpolation_converts_scipy_hilbert_phase_for_convention(self):
        interpolate = self.require_module_attribute(
            "interpolate_analytic_phase_at_arrival"
        )
        scipy_phase_rad = 0.73
        analytic = np.full(2, np.exp(1j * scipy_phase_rad), dtype=complex)
        time_s = np.array([0.0, 1.0])

        legacy_scipy_phase = interpolate(time_s, analytic, 0.5)
        convention_phase = interpolate(
            time_s,
            analytic,
            0.5,
            convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
        )

        self.assertAlmostEqual(legacy_scipy_phase, scipy_phase_rad)
        self.assertAlmostEqual(convention_phase, -scipy_phase_rad)
        self.assertEqual(
            np.sign(convention_phase),
            self.mod.PhaseConvention.BENSEN_VELOCITY_CCF.definition.hilbert_phase_sign,
        )

    @staticmethod
    def _principal_phase(phase_rad):
        wrapped = np.angle(np.exp(1j * np.asarray(phase_rad, dtype=float)))
        return np.where(
            np.isclose(wrapped, -np.pi, rtol=0.0, atol=1e-14),
            np.pi,
            wrapped,
        )

    def test_instantaneous_frequency_huber_fit_recovers_slope_with_local_outlier(self):
        estimate = self.require_module_attribute("estimate_instantaneous_frequency")
        nominal_period_s = 3.5
        expected_slope = 2.0 * np.pi / nominal_period_s
        time_s = np.arange(0.0, 20.0, 0.02)
        phase = 0.31 + expected_slope * time_s
        phase[np.argmin(abs(time_s - 10.0))] += 1.2
        principal_phase = self._principal_phase(phase)

        result = estimate(
            time_s,
            principal_phase,
            group_time_s=10.0,
            nominal_period_s=nominal_period_s,
        )

        self.assertEqual(result.status, "accepted")
        self.assertAlmostEqual(
            result.fitted_phase_slope_rad_s,
            expected_slope,
            delta=2e-4,
        )
        self.assertAlmostEqual(result.omega_inst_rad_s, expected_slope, delta=2e-4)
        self.assertAlmostEqual(
            result.instantaneous_period_s,
            nominal_period_s,
            delta=4e-4,
        )
        self.assertAlmostEqual(result.period_ratio, 1.0, delta=1e-4)
        self.assertGreaterEqual(result.window_sample_count, 5)

    def test_instantaneous_frequency_accepts_negative_signed_slope(self):
        estimate = self.require_module_attribute("estimate_instantaneous_frequency")
        nominal_period_s = 4.0
        expected_slope = -2.0 * np.pi / nominal_period_s
        time_s = np.arange(0.0, 20.0, 0.02)
        phase = self._principal_phase(0.2 + expected_slope * time_s)

        result = estimate(
            time_s,
            phase,
            group_time_s=10.0,
            nominal_period_s=nominal_period_s,
        )

        self.assertEqual(result.status, "accepted")
        self.assertLess(result.fitted_phase_slope_rad_s, 0.0)
        self.assertAlmostEqual(result.fitted_phase_slope_rad_s, expected_slope, places=9)
        self.assertAlmostEqual(result.omega_inst_rad_s, abs(expected_slope), places=9)
        self.assertAlmostEqual(result.instantaneous_period_s, nominal_period_s, places=9)

    def test_instantaneous_frequency_period_ratio_limits_are_inclusive(self):
        estimate = self.require_module_attribute("estimate_instantaneous_frequency")
        nominal_period_s = 4.0
        time_s = np.arange(0.0, 30.0, 0.01)

        for expected_ratio in (0.90, 1.10):
            phase = self._principal_phase(
                (2.0 * np.pi / (nominal_period_s * expected_ratio)) * time_s
            )
            with self.subTest(expected_ratio=expected_ratio):
                result = estimate(
                    time_s,
                    phase,
                    group_time_s=15.0,
                    nominal_period_s=nominal_period_s,
                )
                self.assertEqual(result.status, "accepted")
                self.assertAlmostEqual(result.period_ratio, expected_ratio, places=10)

        for rejected_ratio in (0.899, 1.101):
            phase = self._principal_phase(
                (2.0 * np.pi / (nominal_period_s * rejected_ratio)) * time_s
            )
            with self.subTest(rejected_ratio=rejected_ratio):
                result = estimate(
                    time_s,
                    phase,
                    group_time_s=15.0,
                    nominal_period_s=nominal_period_s,
                )
                self.assertEqual(
                    result.status,
                    "invalid_instantaneous_frequency",
                )

    def test_instantaneous_frequency_explicitly_rejects_boundary_zero_and_nonfinite(self):
        estimate = self.require_module_attribute("estimate_instantaneous_frequency")
        time_s = np.arange(0.0, 10.0, 0.02)
        nominal_period_s = 3.0
        valid_phase = self._principal_phase(
            (2.0 * np.pi / nominal_period_s) * time_s
        )

        boundary = estimate(
            time_s,
            valid_phase,
            group_time_s=0.5,
            nominal_period_s=nominal_period_s,
        )
        zero = estimate(
            time_s,
            np.zeros_like(time_s),
            group_time_s=5.0,
            nominal_period_s=nominal_period_s,
        )
        nonfinite_phase = valid_phase.copy()
        nonfinite_phase[np.argmin(abs(time_s - 5.0))] = np.nan
        nonfinite = estimate(
            time_s,
            nonfinite_phase,
            group_time_s=5.0,
            nominal_period_s=nominal_period_s,
        )

        for result in (boundary, zero, nonfinite):
            self.assertEqual(
                result.status,
                "invalid_instantaneous_frequency",
            )
            self.assertTrue(np.isnan(result.omega_inst_rad_s))
            self.assertTrue(np.isnan(result.instantaneous_period_s))
            self.assertTrue(np.isnan(result.period_ratio))

        sparse_time = np.array([0.0, 1.0, 2.0, 3.0])
        sparse_phase = self._principal_phase(np.pi * sparse_time)
        too_few = estimate(
            sparse_time,
            sparse_phase,
            group_time_s=1.5,
            nominal_period_s=3.0,
        )
        self.assertEqual(
            too_few.status,
            "invalid_instantaneous_frequency",
        )
        self.assertEqual(too_few.window_sample_count, 4)

    def test_instantaneous_frequency_validates_axes_without_mutating_inputs(self):
        estimate = self.require_module_attribute("estimate_instantaneous_frequency")
        time_s = np.arange(0.0, 10.0, 0.02)
        phase = self._principal_phase((2.0 * np.pi / 3.0) * time_s)
        time_copy = time_s.copy()
        phase_copy = phase.copy()
        result = estimate(
            time_s,
            phase,
            group_time_s=5.0,
            nominal_period_s=3.0,
        )
        np.testing.assert_array_equal(time_s, time_copy)
        np.testing.assert_array_equal(phase, phase_copy)
        with self.assertRaises(FrozenInstanceError):
            result.status = "changed"

        bad_inputs = (
            (time_s[:-1], phase, "same shape"),
            (time_s[::-1], phase, "increasing"),
            (time_s, phase[:, np.newaxis], "one-dimensional"),
        )
        for bad_time, bad_phase, message in bad_inputs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    estimate(
                        bad_time,
                        bad_phase,
                        group_time_s=5.0,
                        nominal_period_s=3.0,
                    )

    def test_phase_unwrap_anchors_at_3p5_and_expands_deterministically_both_ways(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        periods_sorted = np.arange(2.5, 4.51, 0.25)
        phase_time_s = 10.0 + 0.2 * periods_sorted
        true_phase = 2.0 * np.pi * phase_time_s / periods_sorted
        principal = self._principal_phase(true_phase)
        order = np.array([6, 0, 8, 4, 2, 7, 1, 5, 3])
        periods = periods_sorted[order]
        phases = principal[order]
        group_times = (1.0 + 0.3 * periods_sorted)[order]
        period_copy = periods.copy()
        phase_copy = phases.copy()

        result = unwrap(
            periods,
            phases,
            group_times,
            phase_time_sign=1,
            anchor_period_s=3.5,
        )

        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.anchor_index, int(np.flatnonzero(periods == 3.5)[0]))
        phase_cycles = phase_time_s / periods_sorted
        principal_cycle_integer = np.floor(phase_cycles + 0.5).astype(int)
        anchor_cycle_integer = principal_cycle_integer[
            np.flatnonzero(periods_sorted == 3.5)[0]
        ]
        expected_cycles_sorted = (
            principal_cycle_integer - anchor_cycle_integer
        )
        expected_cycles = expected_cycles_sorted[order]
        np.testing.assert_array_equal(result.cycle_counts, expected_cycles)
        np.testing.assert_array_equal(
            result.sort_order,
            np.argsort(periods, kind="stable"),
        )
        np.testing.assert_array_equal(
            result.valid_mask,
            np.ones(periods.size, dtype=bool),
        )
        expected_unwrapped = phases + 2.0 * np.pi * expected_cycles
        np.testing.assert_allclose(result.unwrapped_phase_rad, expected_unwrapped)
        finite_error = result.prediction_error_s[np.isfinite(result.prediction_error_s)]
        np.testing.assert_allclose(finite_error, 0.0, atol=2e-14)
        self.assertEqual(result.anomaly_fraction, 0.0)
        self.assertEqual(result.max_consecutive_anomalies, 0)
        np.testing.assert_array_equal(periods, period_copy)
        np.testing.assert_array_equal(phases, phase_copy)

    def test_convention_unwrap_raw_time_matches_central_formula_pointwise(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        raw_time = self.require_module_attribute("raw_phase_travel_time")
        periods = np.array([2.5, 3.1, 3.7, 4.6, 5.0])
        group_times = np.array([8.2, 8.5, 8.9, 9.4, 9.8])
        scipy_principal = np.array(
            [
                np.nextafter(-np.pi, 0.0),
                -2.4,
                0.2,
                2.5,
                np.pi,
            ]
        )
        for convention in self.mod.PhaseConvention:
            with self.subTest(convention=convention.name):
                result = unwrap(
                    periods,
                    scipy_principal,
                    group_times,
                    convention=convention,
                    anchor_period_s=3.7,
                )
                paper_unwrapped = (
                    convention.definition.scipy_phase_multiplier
                    * result.unwrapped_phase_rad
                )
                expected = np.array(
                    [
                        raw_time(
                            convention=convention,
                            group_time_s=group_time,
                            phase_rad=paper_phase,
                            omega_rad_s=2.0 * np.pi / period,
                        )
                        for period, group_time, paper_phase in zip(
                            periods,
                            group_times,
                            paper_unwrapped,
                        )
                    ]
                )
                np.testing.assert_allclose(
                    result.raw_phase_time_s,
                    expected,
                    rtol=0.0,
                    atol=2e-14,
                )

    def test_phase_unwrap_anchor_and_cycle_ties_are_canonical(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        periods = np.array([4.0, 3.0, 5.0])
        anchor_tie = unwrap(
            periods,
            np.zeros(3),
            np.zeros(3),
            phase_time_sign=1,
            anchor_period_s=3.5,
        )
        self.assertEqual(anchor_tie.anchor_index, 1)

        cycle_tie = unwrap(
            np.array([3.0, 3.5, 4.0]),
            np.array([np.pi, np.pi, 0.0]),
            np.zeros(3),
            phase_time_sign=1,
            anchor_period_s=3.5,
        )
        np.testing.assert_array_equal(cycle_tie.cycle_counts, np.array([0, 0, 0]))

    def test_phase_unwrap_prediction_threshold_is_inclusive_and_flags_excess(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        periods = np.array([3.0, 3.5, 4.0])
        at_limit_phase = np.array([0.0, np.pi / 2.0, 0.0])

        at_limit = unwrap(
            periods,
            at_limit_phase,
            np.zeros(3),
            phase_time_sign=1,
            anchor_period_s=3.5,
        )
        self.assertEqual(at_limit.status, "accepted")
        self.assertAlmostEqual(at_limit.prediction_error_s[1], 0.25 * 3.5)
        self.assertEqual(at_limit.anomaly_fraction, 0.0)

        over_limit_phase = at_limit_phase.copy()
        over_limit_phase[1] *= 1.000001
        over_limit = unwrap(
            periods,
            over_limit_phase,
            np.zeros(3),
            phase_time_sign=1,
            anchor_period_s=3.5,
        )
        self.assertEqual(over_limit.status, "phase_unwrap_discontinuous")
        self.assertGreater(over_limit.prediction_error_s[1], 0.25 * 3.5)
        self.assertEqual(over_limit.anomaly_fraction, 1.0)
        self.assertEqual(over_limit.max_consecutive_anomalies, 1)

    def test_phase_unwrap_reports_consecutive_anomalies_and_cycle_step_failure(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        has_excessive_step = self.require_module_attribute(
            "_has_excessive_cycle_step"
        )
        periods = np.arange(2.5, 5.05, 0.05)
        phase_time_s = 8.0 - 1.1 * periods
        phase_time_s[20] += 0.20 * periods[20]
        phase_time_s[21] -= 0.20 * periods[21]
        principal = self._principal_phase(
            2.0 * np.pi * phase_time_s / periods
        )

        result = unwrap(
            periods,
            principal,
            np.zeros(periods.size),
            phase_time_sign=1,
            anchor_period_s=3.5,
        )

        self.assertEqual(result.status, "phase_unwrap_discontinuous")
        self.assertGreaterEqual(result.max_consecutive_anomalies, 2)
        self.assertFalse(
            has_excessive_step(np.array([0, 1, 0, -1], dtype=int))
        )
        self.assertTrue(
            has_excessive_step(np.array([0, 2], dtype=int))
        )
        self.assertLessEqual(
            int(np.max(np.abs(np.diff(result.cycle_counts)))),
            1,
        )

    def test_phase_unwrap_anomaly_fraction_five_percent_is_inclusive(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        periods = np.arange(2.5, 3.60, 0.05)
        self.assertEqual(periods.size - 2, 20)

        def result_with_spikes(indices):
            phase_time_s = 5.0 - 0.3 * periods
            for index in indices:
                phase_time_s[index] += 0.40 * periods[index]
            principal = self._principal_phase(
                2.0 * np.pi * phase_time_s / periods
            )
            return unwrap(
                periods,
                principal,
                np.zeros(periods.size),
                phase_time_sign=1,
            )

        exactly_five_percent = result_with_spikes([10])
        above_five_percent = result_with_spikes([5, 15])

        self.assertEqual(exactly_five_percent.anomaly_fraction, 0.05)
        self.assertEqual(exactly_five_percent.max_consecutive_anomalies, 1)
        self.assertEqual(exactly_five_percent.status, "accepted")
        self.assertEqual(above_five_percent.anomaly_fraction, 0.10)
        self.assertEqual(above_five_percent.max_consecutive_anomalies, 1)
        self.assertEqual(
            above_five_percent.status,
            "phase_unwrap_discontinuous",
        )

    def test_phase_unwrap_validates_inputs_and_returns_immutable_arrays(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        periods = np.array([3.0, 3.5, 4.0])
        phases = np.array([0.1, 0.2, 0.3])
        group_times = np.array([2.0, 2.2, 2.4])
        result = unwrap(periods, phases, group_times, phase_time_sign=1)

        for array in (
            result.unwrapped_phase_rad,
            result.cycle_counts,
            result.raw_phase_time_s,
            result.prediction_error_s,
            result.sort_order,
            result.valid_mask,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array[0] = array[0]
        with self.assertRaises(FrozenInstanceError):
            result.status = "changed"

        bad_calls = (
            (periods[:-1], phases, group_times, "same shape"),
            (periods, phases, group_times[:-1], "same shape"),
            (np.array([3.0, 3.0, 4.0]), phases, group_times, "unique"),
            (
                np.array([3.0, np.nan, 4.0]),
                phases,
                group_times,
                "positive finite",
            ),
            (periods, np.array([0.1, np.inf, 0.3]), group_times, "finite"),
            (periods, np.array([0.1, 4.0, 0.3]), group_times, "principal"),
            (
                periods,
                np.array([0.1, np.nextafter(-np.pi, -np.inf), 0.3]),
                group_times,
                "principal",
            ),
            (
                periods,
                phases,
                np.array([2.0, np.nan, 2.4]),
                "group_times_s",
            ),
        )
        for bad_periods, bad_phases, bad_group_times, message in bad_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    unwrap(
                        bad_periods,
                        bad_phases,
                        bad_group_times,
                        phase_time_sign=1,
                    )

        with self.assertRaisesRegex(ValueError, "at least three"):
            unwrap(
                periods[:2],
                phases[:2],
                group_times[:2],
                phase_time_sign=1,
            )

        for bad_sign in (0, 2, -2, 1.0, True):
            with self.subTest(bad_sign=bad_sign):
                with self.assertRaisesRegex(ValueError, "phase_time_sign"):
                    unwrap(
                        periods,
                        phases,
                        group_times,
                        phase_time_sign=bad_sign,
                    )

    def test_phase_unwrap_continuity_uses_full_raw_time_including_group_arrival(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        periods = np.arange(2.5, 5.05, 0.05)
        phase_time_s = 6.0 - 0.7 * periods
        principal = self._principal_phase(
            2.0 * np.pi * phase_time_s / periods
        )
        linear_group_time = 4.0 + 0.5 * periods

        stable = unwrap(
            periods,
            principal,
            linear_group_time,
            phase_time_sign=1,
        )
        self.assertEqual(stable.status, "accepted")
        np.testing.assert_allclose(
            stable.prediction_error_s[np.isfinite(stable.prediction_error_s)],
            0.0,
            atol=2e-14,
        )

        anomalous_group_time = linear_group_time.copy()
        anomalous_group_time[20:22] += periods[20:22]
        anomalous = unwrap(
            periods,
            principal,
            anomalous_group_time,
            phase_time_sign=1,
        )
        self.assertEqual(anomalous.status, "phase_unwrap_discontinuous")
        self.assertGreaterEqual(anomalous.max_consecutive_anomalies, 2)

    def test_phase_unwrap_requires_explicit_sign_and_supports_negative_convention(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        periods = np.arange(2.5, 5.05, 0.05)
        phase_time_component = 6.0 - 0.7 * periods
        phase_time_component[20] += 0.20 * periods[20]
        phase_time_component[21] -= 0.20 * periods[21]
        principal = self._principal_phase(
            2.0 * np.pi * phase_time_component / periods
        )
        target_linear_time = 10.0 + 0.2 * periods
        group_times = target_linear_time + phase_time_component

        negative = unwrap(
            periods,
            principal,
            group_times,
            phase_time_sign=-1,
        )
        wrong_positive = unwrap(
            periods,
            principal,
            group_times,
            phase_time_sign=1,
        )

        self.assertEqual(negative.status, "accepted")
        self.assertEqual(wrong_positive.status, "phase_unwrap_discontinuous")
        with self.assertRaises(TypeError):
            unwrap(periods, principal, group_times)

    def test_phase_unwrap_can_take_sign_directly_from_phase_convention(self):
        unwrap = self.require_module_attribute("unwrap_phase_along_frequency")
        periods = np.array([3.0, 3.5, 4.0])
        phases = np.array([0.1, 0.2, 0.3])
        group_times = np.array([8.0, 8.2, 8.4])
        convention = self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF

        direct = unwrap(
            periods,
            phases,
            group_times,
            convention=convention,
        )
        legacy = unwrap(
            periods,
            phases,
            group_times,
            phase_time_sign=convention.definition.phase_time_sign,
        )

        np.testing.assert_array_equal(
            direct.unwrapped_phase_rad,
            legacy.unwrapped_phase_rad,
        )
        np.testing.assert_allclose(
            direct.raw_phase_time_s - legacy.raw_phase_time_s,
            convention.definition.fixed_phase_rad
            * periods
            / (2.0 * np.pi),
        )
        self.assertEqual(direct.status, legacy.status)
        np.testing.assert_allclose(
            direct.prediction_error_s,
            legacy.prediction_error_s,
            equal_nan=True,
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            unwrap(
                periods,
                phases,
                group_times,
                phase_time_sign=-convention.definition.phase_time_sign,
                convention=convention,
            )

    def test_phase_time_prediction_is_invariant_to_global_integer_cycle(self):
        diagnose = self.require_module_attribute(
            "_phase_time_prediction_errors"
        )
        periods = np.array([2.5, 2.8, 3.3, 3.9, 4.6, 5.0])
        raw_time = 8.0 + 0.3 * periods + 0.04 * periods**2

        reference_error, reference_anomaly = diagnose(periods, raw_time)
        shifted_error, shifted_anomaly = diagnose(
            periods,
            raw_time + 7.0 * periods,
        )

        np.testing.assert_allclose(
            shifted_error,
            reference_error,
            rtol=0.0,
            atol=2e-14,
            equal_nan=True,
        )
        np.testing.assert_array_equal(shifted_anomaly, reference_anomaly)

    def test_wang_snr_uses_exact_independent_waveform_rms_windows(self):
        compute = self.require_module_attribute("compute_wang_snr")
        time_s = np.linspace(0.0, 20.0, 201)
        waveform = np.zeros_like(time_s)
        waveform[(time_s >= 0.1) & (time_s <= 2.0)] = 2.0
        waveform[(time_s >= 14.5) & (time_s <= 20.0)] = 4.0
        waveform[(time_s >= 4.0) & (time_s <= 12.5)] = 1.0
        waveform[np.argmin(np.abs(time_s - 8.0))] = -20.0

        result = compute(
            time_s=time_s,
            filtered_waveform=waveform,
            distance_km=20.0,
            period_s=4.0,
        )

        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.leading_status, "accepted")
        self.assertEqual(result.trailing_status, "accepted")
        self.assertAlmostEqual(result.signal_window_start_s, 4.0)
        self.assertAlmostEqual(result.signal_window_end_s, 12.5)
        self.assertAlmostEqual(result.leading_noise_end_s, 2.0)
        self.assertAlmostEqual(result.trailing_noise_start_s, 14.5)
        self.assertEqual(result.signal_sample_count, 86)
        self.assertEqual(result.leading_noise_sample_count, 20)
        self.assertEqual(result.trailing_noise_sample_count, 56)
        self.assertAlmostEqual(result.signal_peak, 20.0)
        self.assertAlmostEqual(result.leading_noise_rms, 2.0)
        self.assertAlmostEqual(result.trailing_noise_rms, 4.0)
        self.assertAlmostEqual(result.leading_snr, 10.0)
        self.assertAlmostEqual(result.trailing_snr, 5.0)
        with self.assertRaises(FrozenInstanceError):
            result.status = "changed"

    def test_wang_snr_reports_each_short_noise_window_independently(self):
        compute = self.require_module_attribute("compute_wang_snr")
        leading_short = compute(
            time_s=np.linspace(0.0, 20.0, 41),
            filtered_waveform=np.ones(41),
            distance_km=10.0,
            period_s=4.0,
        )
        self.assertEqual(
            leading_short.leading_status,
            "insufficient_leading_noise",
        )
        self.assertEqual(leading_short.trailing_status, "accepted")
        self.assertEqual(
            leading_short.status,
            "insufficient_leading_noise",
        )
        self.assertTrue(np.isnan(leading_short.leading_snr))
        self.assertTrue(np.isfinite(leading_short.trailing_snr))

        trailing_short = compute(
            time_s=np.linspace(0.0, 15.0, 151),
            filtered_waveform=np.ones(151),
            distance_km=20.0,
            period_s=4.0,
        )
        self.assertEqual(trailing_short.leading_status, "accepted")
        self.assertEqual(
            trailing_short.trailing_status,
            "insufficient_trailing_noise",
        )
        self.assertEqual(
            trailing_short.status,
            "insufficient_trailing_noise",
        )
        self.assertTrue(np.isfinite(trailing_short.leading_snr))
        self.assertTrue(np.isnan(trailing_short.trailing_snr))

    def test_wang_snr_rejects_nonfinite_inputs_without_warnings(self):
        compute = self.require_module_attribute("compute_wang_snr")
        time_s = np.linspace(0.0, 20.0, 201)
        waveform = np.ones_like(time_s)
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                invalid = waveform.copy()
                invalid[10] = bad
                with self.assertRaisesRegex(ValueError, "finite"):
                    compute(
                        time_s=time_s,
                        filtered_waveform=invalid,
                        distance_km=20.0,
                        period_s=4.0,
                    )

    def test_wang_left_qc_requires_strict_snr_and_exact_period_velocity_limit(self):
        compute = self.require_module_attribute("compute_wang_snr")
        evaluate = self.require_module_attribute("evaluate_wang_left_qc")
        time_s = np.linspace(0.0, 30.0, 301)
        waveform = np.full_like(time_s, 2.0)
        signal = (time_s >= 4.0) & (time_s <= 12.5)
        waveform[signal] = 8.0
        snr_exactly_four = compute(
            time_s=time_s,
            filtered_waveform=waveform,
            distance_km=20.0,
            period_s=4.0,
        )
        self.assertAlmostEqual(snr_exactly_four.leading_snr, 4.0)
        self.assertAlmostEqual(snr_exactly_four.trailing_snr, 4.0)
        self.assertEqual(
            evaluate(
                period_s=4.0,
                group_velocity_km_s=2.8,
                snr=snr_exactly_four,
            ).status,
            "snr_threshold_failed",
        )

        accepted_snr = compute(
            time_s=time_s,
            filtered_waveform=np.where(signal, 10.0, 2.0),
            distance_km=20.0,
            period_s=4.0,
        )
        self.assertEqual(
            evaluate(
                period_s=4.49,
                group_velocity_km_s=3.01,
                snr=accepted_snr,
            ).status,
            "group_velocity_out_of_range",
        )
        self.assertEqual(
            evaluate(
                period_s=4.5,
                group_velocity_km_s=3.3,
                snr=accepted_snr,
            ).status,
            "accepted",
        )
        self.assertEqual(
            evaluate(
                period_s=5.0,
                group_velocity_km_s=1.59,
                snr=accepted_snr,
            ).status,
            "group_velocity_out_of_range",
        )
        for field in (
            "ftan_valid",
            "ridge_valid",
            "group_arrival_valid",
            "phase_valid",
            "instantaneous_frequency_valid",
        ):
            kwargs = {
                "period_s": 4.0,
                "group_velocity_km_s": 2.8,
                "snr": accepted_snr,
                field: False,
            }
            self.assertEqual(
                evaluate(**kwargs).status,
                f"{field}_failed",
            )

    def test_wang_targets_use_exact_two_sided_linear_resampling(self):
        resample = self.require_module_attribute("resample_wang_target_period")
        for target in (3.0, 3.5, 4.0, 5.0):
            periods = np.array([target - 0.05, target + 0.05])
            result = resample(
                continuous_periods_s=periods,
                anchored_raw_phase_time_s=1.0 + 2.0 * periods,
                group_time_s=10.0 + periods,
                signal_peak=12.0 + periods,
                leading_noise_rms=np.full(2, 2.0),
                trailing_noise_rms=np.full(2, 2.5),
                ridge_normalized_log_energy=0.8 + 0.01 * periods,
                ridge_normalized_envelope_amplitude=0.5 + 0.02 * periods,
                ridge_adjacent_jump_km_s=0.02 + 0.01 * periods,
                valid_mask=np.ones(2, dtype=bool),
                distance_km=30.0,
                target_period_s=target,
            )
            with self.subTest(target=target):
                self.assertEqual(result.status, "accepted")
                self.assertTrue(result.accepted)
                self.assertEqual(result.target_period_s, target)
                self.assertEqual(result.support_count, 2)
                self.assertEqual(result.interpolation_method, "linear")
                np.testing.assert_array_equal(result.support_periods_s, periods)
                self.assertAlmostEqual(result.anchored_raw_phase_time_s, 1.0 + 2.0 * target)
                self.assertAlmostEqual(result.group_time_s, 10.0 + target)
                self.assertAlmostEqual(result.signal_peak, 12.0 + target)
                self.assertAlmostEqual(result.leading_noise_rms, 2.0)
                self.assertAlmostEqual(result.trailing_noise_rms, 2.5)
                self.assertAlmostEqual(result.leading_snr, (12.0 + target) / 2.0)
                self.assertAlmostEqual(result.trailing_snr, (12.0 + target) / 2.5)
                self.assertAlmostEqual(result.group_velocity_km_s, 30.0 / (10.0 + target))
                self.assertAlmostEqual(
                    result.ridge_normalized_log_energy,
                    0.8 + 0.01 * target,
                )
                self.assertAlmostEqual(
                    result.ridge_normalized_envelope_amplitude,
                    0.5 + 0.02 * target,
                )
                self.assertAlmostEqual(
                    result.ridge_adjacent_jump_km_s,
                    0.02 + 0.01 * target,
                )
                self.assertFalse(result.support_periods_s.flags.writeable)

    def test_wang_target_uses_pchip_with_three_or_more_support_points(self):
        resample = self.require_module_attribute("resample_wang_target_period")
        periods = np.array([3.91, 3.97, 4.06, 4.09])
        curved = periods**2
        result = resample(
            continuous_periods_s=periods,
            anchored_raw_phase_time_s=curved,
            group_time_s=12.0 + 0.1 * curved,
            signal_peak=20.0 + curved,
            leading_noise_rms=2.0 + 0.01 * curved,
            trailing_noise_rms=2.5 + 0.01 * curved,
            ridge_normalized_log_energy=0.8 + 0.001 * curved,
            ridge_normalized_envelope_amplitude=0.5 + 0.01 * curved,
            ridge_adjacent_jump_km_s=0.05 + 0.001 * curved,
            valid_mask=np.ones(periods.size, dtype=bool),
            distance_km=30.0,
            target_period_s=4.0,
        )

        expected = float(
            PchipInterpolator(periods, curved)(4.0)
        )
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.support_count, 4)
        self.assertEqual(result.interpolation_method, "pchip")
        self.assertAlmostEqual(result.anchored_raw_phase_time_s, expected)
        self.assertAlmostEqual(result.group_time_s, 12.0 + 0.1 * expected)
        self.assertAlmostEqual(result.signal_peak, 20.0 + expected)
        self.assertAlmostEqual(
            result.ridge_normalized_log_energy,
            0.8 + 0.001 * expected,
        )
        self.assertAlmostEqual(
            result.ridge_normalized_envelope_amplitude,
            0.5 + 0.01 * expected,
        )
        self.assertAlmostEqual(
            result.ridge_adjacent_jump_km_s,
            0.05 + 0.001 * expected,
        )

    def test_wang_target_never_extrapolates_or_crosses_large_support_gap(self):
        resample = self.require_module_attribute("resample_wang_target_period")

        def call(periods):
            values = np.ones(len(periods), dtype=float)
            return resample(
                continuous_periods_s=np.asarray(periods, dtype=float),
                anchored_raw_phase_time_s=10.0 * values,
                group_time_s=12.0 * values,
                signal_peak=20.0 * values,
                leading_noise_rms=2.0 * values,
                trailing_noise_rms=2.0 * values,
                ridge_normalized_log_energy=0.9 * values,
                ridge_normalized_envelope_amplitude=0.8 * values,
                ridge_adjacent_jump_km_s=0.1 * values,
                valid_mask=np.ones(len(periods), dtype=bool),
                distance_km=30.0,
                target_period_s=4.0,
            )

        for periods in (
            [4.01, 4.05],
            [3.95, 3.99],
            [3.89, 4.01],
            [3.90, 4.11],
        ):
            with self.subTest(periods=periods):
                result = call(periods)
                self.assertFalse(result.accepted)
                self.assertEqual(
                    result.status,
                    "target_period_not_bracketed",
                )
                self.assertEqual(result.interpolation_method, "none")

    def test_wang_target_rejects_invalid_rowwise_ridge_supports(self):
        resample = self.require_module_attribute("resample_wang_target_period")
        fields = {
            "ridge_normalized_log_energy": np.array([0.7, 0.9]),
            "ridge_normalized_envelope_amplitude": np.array([0.6, 0.8]),
            "ridge_adjacent_jump_km_s": np.array([0.0, 0.2]),
        }
        common = {
            "continuous_periods_s": np.array([2.95, 3.05]),
            "anchored_raw_phase_time_s": np.array([7.0, 7.2]),
            "group_time_s": np.array([10.0, 10.2]),
            "signal_peak": np.array([20.0, 20.0]),
            "leading_noise_rms": np.array([2.0, 2.0]),
            "trailing_noise_rms": np.array([2.0, 2.0]),
            "valid_mask": np.ones(2, dtype=bool),
            "distance_km": 20.0,
            "target_period_s": 3.0,
        }
        invalid = {
            "ridge_normalized_log_energy": np.array([-0.2, 0.4]),
            "ridge_normalized_envelope_amplitude": np.array([1.2, 0.8]),
            "ridge_adjacent_jump_km_s": np.array([-0.1, 0.1]),
        }
        for name, values in invalid.items():
            with self.subTest(field=name):
                kwargs = dict(fields)
                kwargs[name] = values
                with self.assertRaisesRegex(ValueError, name):
                    resample(**common, **kwargs)

    def test_wang_target_result_rejects_contradictory_manual_state(self):
        result_type = self.require_module_attribute(
            "WangTargetPeriodResult"
        )
        values = {
            "target_period_s": 3.0,
            "anchored_raw_phase_time_s": 7.0,
            "group_time_s": 8.0,
            "group_velocity_km_s": 2.5,
            "signal_peak": 10.0,
            "leading_noise_rms": 1.0,
            "trailing_noise_rms": 1.0,
            "leading_snr": 10.0,
            "trailing_snr": 10.0,
            "ridge_normalized_log_energy": 0.8,
            "ridge_normalized_envelope_amplitude": 0.9,
            "ridge_adjacent_jump_km_s": 0.1,
            "support_periods_s": np.array([2.95, 3.05]),
            "support_count": 2,
            "interpolation_method": "linear",
            "accepted": True,
            "status": "accepted",
        }
        result_type(**values)
        result_type(
            **{
                **values,
                "accepted": False,
                "status": "snr_threshold_failed",
            }
        )
        rejected = {
            **values,
            "anchored_raw_phase_time_s": np.nan,
            "group_time_s": np.nan,
            "group_velocity_km_s": np.nan,
            "signal_peak": np.nan,
            "leading_noise_rms": np.nan,
            "trailing_noise_rms": np.nan,
            "leading_snr": np.nan,
            "trailing_snr": np.nan,
            "ridge_normalized_log_energy": np.nan,
            "ridge_normalized_envelope_amplitude": np.nan,
            "ridge_adjacent_jump_km_s": np.nan,
            "interpolation_method": "none",
            "accepted": False,
            "status": "target_period_not_bracketed",
        }
        result_type(**rejected)

        contradictions = (
            {"accepted": True, "status": "snr_threshold_failed"},
            {"accepted": False, "status": "accepted"},
            {"support_count": 1},
            {"interpolation_method": "pchip"},
            {
                "support_periods_s": np.array([2.91, 2.97, 3.05]),
                "support_count": 3,
                "interpolation_method": "linear",
            },
            {
                **rejected,
                "interpolation_method": "linear",
            },
            {"support_periods_s": np.array([3.01, 3.05])},
            {"support_periods_s": np.array([2.89, 3.01])},
            {"support_periods_s": np.array([2.89, 3.11])},
            {"target_period_s": 0.0},
            {"target_period_s": -3.0},
            {"target_period_s": np.nan},
        )
        for overrides in contradictions:
            with self.subTest(overrides=overrides):
                candidate = dict(values)
                candidate.update(overrides)
                with self.assertRaises(ValueError):
                    result_type(**candidate)

    def test_failed_four_second_target_does_not_delete_three_second_target(self):
        resample_many = self.require_module_attribute(
            "resample_wang_target_periods"
        )
        periods = np.array([2.95, 3.05, 3.95, 4.05])
        values = np.ones(periods.size)
        results = resample_many(
            continuous_periods_s=periods,
            anchored_raw_phase_time_s=10.0 + periods,
            group_time_s=12.0 + periods,
            signal_peak=20.0 + periods,
            leading_noise_rms=2.0 * values,
            trailing_noise_rms=2.0 * values,
            ridge_normalized_log_energy=0.9 * values,
            ridge_normalized_envelope_amplitude=0.8 * values,
            ridge_adjacent_jump_km_s=0.1 * values,
            valid_mask=np.array([True, True, False, False]),
            distance_km=30.0,
            target_periods_s=(3.0, 4.0),
        )

        self.assertEqual(tuple(row.target_period_s for row in results), (3.0, 4.0))
        self.assertEqual(results[0].status, "accepted")
        self.assertEqual(
            results[1].status,
            "target_period_not_bracketed",
        )

    def test_phase_snr_wrapper_uniquely_delegates_to_exact_wang_snr(self):
        time_s = np.linspace(0.0, 30.0, 301)
        waveform = np.ones_like(time_s)
        signature = inspect.signature(self.mod._phase_measurement_snr)
        self.assertIn("distance_km", signature.parameters)
        expected = self.mod.compute_wang_snr(
            time_s=time_s,
            filtered_waveform=waveform,
            distance_km=20.0,
            period_s=3.25,
        )
        with mock.patch.object(
            self.mod,
            "compute_wang_snr",
            wraps=self.mod.compute_wang_snr,
        ) as compute:
            value = self.mod._phase_measurement_snr(
                waveform,
                time_s,
                distance_km=20.0,
                period_s=3.25,
            )

        self.assertEqual(compute.call_count, 1)
        self.assertEqual(compute.call_args.kwargs["period_s"], 3.25)
        self.assertEqual(
            value,
            min(expected.leading_snr, expected.trailing_snr),
        )
        source = inspect.getsource(self.mod._phase_measurement_snr)
        self.assertIn("compute_wang_snr(", source)
        self.assertNotIn("np.sqrt", source)
        self.assertNotIn("np.mean", source)

    def test_continuous_measurements_use_t_inst_before_exact_target_resampling(self):
        resample = self.require_module_attribute("resample_wang_measurements")
        self.assertIn(
            "valid_mask",
            inspect.signature(resample).parameters,
        )
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        time_s = np.linspace(0.0, 30.0, 301)
        waveform = np.ones_like(time_s)
        waveform[(time_s >= 4.0) & (time_s <= 12.5)] = 10.0
        rows = []
        for nominal, instantaneous in ((2.9, 2.95), (3.1, 3.05)):
            omega = 2.0 * np.pi / instantaneous
            raw_time = self.mod.raw_phase_travel_time(
                convention=convention,
                group_time_s=8.0 + instantaneous,
                phase_rad=0.2,
                omega_rad_s=omega,
            )
            rows.append(
                self.mod.PeriodMeasurement(
                    convention=convention,
                    period_s=nominal,
                    omega_inst_rad_s=omega,
                    principal_paper_phase_rad=0.2,
                    unwrapped_paper_phase_rad=0.2,
                    raw_phase_time_s=raw_time,
                    paper_phase_cycle_offset=0,
                    group_time_s=8.0 + instantaneous,
                    group_velocity_km_s=20.0 / (8.0 + instantaneous),
                    snr=10.0,
                    signal_window_start_s=4.0,
                    signal_window_end_s=12.5,
                    filtered_waveform=waveform,
                    envelope=np.abs(waveform),
                )
            )
        quality = self.mod.RidgeQuality(
            accepted=True,
            reason="accepted",
            coverage=0.9,
            max_gap=0,
            jump_fraction=0.1,
            boundary_fraction=0.05,
            normalized_energy_integral=2.0,
        )
        with (
            mock.patch.object(
                self.mod,
                "compute_wang_snr",
                wraps=self.mod.compute_wang_snr,
            ) as compute,
            mock.patch.object(
                self.mod,
                "evaluate_wang_left_qc",
                wraps=self.mod.evaluate_wang_left_qc,
            ) as evaluate,
        ):
            result = resample(
                rows,
                time_s=time_s,
                distance_km=20.0,
                target_periods_s=(3.0,),
                nominal_periods_s=np.array([2.9, 3.1]),
                measurement_statuses=("accepted", "accepted"),
                instantaneous_periods_s=np.array([2.95, 3.05]),
                ridge_normalized_log_energy=np.array([0.7, 0.9]),
                ridge_normalized_envelope_amplitude=np.array([0.6, 0.8]),
                ridge_adjacent_jump_km_s=np.array([0.0, 0.2]),
                valid_mask=np.ones(2, dtype=bool),
            )[0]

        self.assertEqual(
            [call.kwargs["period_s"] for call in compute.call_args_list],
            [2.95, 3.05],
        )
        self.assertEqual(evaluate.call_args.kwargs["period_s"], 3.0)
        self.assertEqual(result.target_period_s, 3.0)
        self.assertEqual(result.status, "accepted")
        self.assertAlmostEqual(result.ridge_normalized_log_energy, 0.8)
        self.assertAlmostEqual(
            result.ridge_normalized_envelope_amplitude,
            0.7,
        )
        self.assertAlmostEqual(result.ridge_adjacent_jump_km_s, 0.1)
        with self.assertRaisesRegex(
            ValueError,
            "ridge_normalized_log_energy",
        ):
            resample(
                rows,
                time_s=time_s,
                distance_km=20.0,
                target_periods_s=(3.0,),
                nominal_periods_s=np.array([2.9, 3.1]),
                measurement_statuses=("accepted", "accepted"),
                instantaneous_periods_s=np.array([2.95, 3.05]),
                ridge_normalized_log_energy=np.array([1.2, 0.8]),
                ridge_normalized_envelope_amplitude=np.array([0.6, 0.8]),
                ridge_adjacent_jump_km_s=np.array([0.0, 0.2]),
                valid_mask=np.ones(2, dtype=bool),
            )

        invalid_rows = []
        for row, invalid_velocity in zip(rows, (3.1, 1.5)):
            invalid_group_time = 20.0 / invalid_velocity
            invalid_raw_time = self.mod.raw_phase_travel_time(
                convention=convention,
                group_time_s=invalid_group_time,
                phase_rad=row.unwrapped_paper_phase_rad,
                omega_rad_s=row.omega_inst_rad_s,
            )
            invalid_rows.append(
                replace(
                    row,
                    group_time_s=invalid_group_time,
                    group_velocity_km_s=invalid_velocity,
                    raw_phase_time_s=invalid_raw_time,
                )
            )
        rejected = resample(
            (None, *invalid_rows),
            time_s=time_s,
            distance_km=20.0,
            target_periods_s=(3.0,),
            nominal_periods_s=np.array([2.8, 2.9, 3.1]),
            measurement_statuses=(
                "invalid_instantaneous_frequency",
                "accepted",
                "accepted",
            ),
            instantaneous_periods_s=np.array([np.nan, 2.95, 3.05]),
            ridge_normalized_log_energy=np.array([0.6, 0.7, 0.9]),
            ridge_normalized_envelope_amplitude=np.array([0.5, 0.6, 0.8]),
            ridge_adjacent_jump_km_s=np.array([0.0, 0.1, 0.2]),
            valid_mask=np.array([False, True, True]),
        )[0]
        self.assertFalse(rejected.accepted)
        self.assertEqual(
            rejected.status,
            "target_period_not_bracketed",
        )
        self.assertTrue(
            hasattr(
                rejected,
                "rejected_continuous_nominal_periods_s",
            )
        )
        np.testing.assert_allclose(
            rejected.rejected_continuous_nominal_periods_s,
            np.array([2.8, 2.9, 3.1]),
        )
        np.testing.assert_allclose(
            rejected.rejected_continuous_instantaneous_periods_s,
            np.array([np.nan, 2.95, 3.05]),
            equal_nan=True,
        )
        self.assertEqual(
            rejected.continuous_rejection_statuses,
            (
                "invalid_instantaneous_frequency",
                "group_velocity_out_of_range",
                "group_velocity_out_of_range",
            ),
        )
        self.assertFalse(
            rejected.rejected_continuous_nominal_periods_s.flags.writeable
        )
        self.assertFalse(
            rejected.rejected_continuous_instantaneous_periods_s.flags.writeable
        )

    def test_direct_resampling_audits_all_duplicate_instantaneous_periods(self):
        resample = self.require_module_attribute("resample_wang_measurements")
        convention = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        time_s = np.linspace(0.0, 30.0, 301)
        waveform = np.ones_like(time_s)
        waveform[(time_s >= 4.0) & (time_s <= 12.5)] = 10.0
        nominal = np.array([2.9, 3.0, 3.1, 3.9, 4.1])
        instantaneous = np.array([3.0, 3.0, 3.0, 3.95, 4.05])
        rows = []
        for nominal_period, instantaneous_period in zip(
            nominal,
            instantaneous,
        ):
            omega = 2.0 * np.pi / instantaneous_period
            group_time = 8.0
            rows.append(
                self.mod.PeriodMeasurement(
                    convention=convention,
                    period_s=float(nominal_period),
                    omega_inst_rad_s=omega,
                    principal_paper_phase_rad=0.2,
                    unwrapped_paper_phase_rad=0.2,
                    raw_phase_time_s=self.mod.raw_phase_travel_time(
                        convention=convention,
                        group_time_s=group_time,
                        phase_rad=0.2,
                        omega_rad_s=omega,
                    ),
                    paper_phase_cycle_offset=0,
                    group_time_s=group_time,
                    group_velocity_km_s=2.5,
                    snr=10.0,
                    signal_window_start_s=4.0,
                    signal_window_end_s=12.5,
                    filtered_waveform=waveform,
                    envelope=np.abs(waveform),
                )
            )

        result = resample(
            rows,
            time_s=time_s,
            distance_km=20.0,
            target_periods_s=(4.0,),
            nominal_periods_s=nominal,
            measurement_statuses=("accepted",) * nominal.size,
            instantaneous_periods_s=instantaneous,
            ridge_normalized_log_energy=np.linspace(0.6, 0.9, nominal.size),
            ridge_normalized_envelope_amplitude=np.linspace(
                0.7,
                0.9,
                nominal.size,
            ),
            ridge_adjacent_jump_km_s=np.linspace(0.0, 0.2, nominal.size),
            valid_mask=np.ones(nominal.size, dtype=bool),
        )[0]

        self.assertTrue(result.accepted)
        self.assertEqual(result.target_period_s, 4.0)
        np.testing.assert_array_equal(
            result.rejected_continuous_nominal_periods_s,
            nominal[:3],
        )
        np.testing.assert_array_equal(
            result.rejected_continuous_instantaneous_periods_s,
            instantaneous[:3],
        )
        self.assertEqual(
            result.continuous_rejection_statuses,
            ("duplicate_instantaneous_period",) * 3,
        )

    def test_wrap_periodic_uses_canonical_half_open_boundaries(self):
        wrap = self.require_module_attribute("wrap_periodic")
        period = 2.0
        expected = {
            -3.0: -1.0,
            -1.0: -1.0,
            0.0: 0.0,
            1.0: -1.0,
            3.0: -1.0,
        }
        for value, canonical in expected.items():
            with self.subTest(value=value):
                self.assertEqual(wrap(value, period), canonical)
        np.testing.assert_array_equal(
            wrap(np.array([-1.5, -1.0, 1.0, 1.5]), period),
            np.array([0.5, -1.0, -1.0, -0.5]),
        )
        for bad in (True, 1 + 2j, "1", np.nan, np.inf):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "real numeric"):
                    wrap(bad, period)

    def test_huber_loss_is_deterministic_and_strict(self):
        huber = self.require_module_attribute("huber_loss")
        residual = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        np.testing.assert_allclose(
            huber(residual, delta=0.1),
            np.array([0.015, 0.005, 0.0, 0.005, 0.015]),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertAlmostEqual(huber(0.2, delta=0.1), 0.015, places=15)
        for bad_delta in (True, 0.0, -0.1, np.nan, np.inf):
            with self.subTest(delta=bad_delta):
                with self.assertRaisesRegex(ValueError, "delta"):
                    huber(0.0, delta=bad_delta)

    def test_group_slowness_is_exact_phase_slowness_omega_derivative(self):
        derive = self.require_module_attribute(
            "phase_slowness_to_group_slowness"
        )
        periods = np.arange(2.5, 5.05, 0.05)
        omega = 2.0 * np.pi / periods
        phase_slowness = 0.25 + 0.015 * omega
        expected_group_slowness = 0.25 + 0.030 * omega
        np.testing.assert_allclose(
            derive(periods, phase_slowness),
            expected_group_slowness,
            rtol=0.0,
            atol=2e-14,
        )

    def test_cycle_resolution_honors_convention_and_half_period_ties(self):
        resolve = self.require_module_attribute("resolve_cycle_count")
        bensen = self.mod.PhaseConvention.BENSEN_VELOCITY_CCF
        lin = self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF

        bensen_result = resolve(
            raw_time_s=8.1,
            reference_time_s=14.0,
            period_s=3.0,
            convention=bensen,
        )
        self.assertEqual(bensen_result.cycle_count, 2)
        self.assertAlmostEqual(bensen_result.corrected_time_s, 14.1)
        self.assertFalse(bensen_result.branch_tie)

        lin_result = resolve(
            raw_time_s=19.9,
            reference_time_s=14.0,
            period_s=3.0,
            convention=lin,
        )
        self.assertEqual(lin_result.cycle_count, 2)
        self.assertAlmostEqual(lin_result.corrected_time_s, 13.9)
        self.assertFalse(lin_result.branch_tie)

        tie = resolve(
            raw_time_s=10.5,
            reference_time_s=12.0,
            period_s=3.0,
            convention=bensen,
        )
        self.assertEqual(tie.cycle_count, 0)
        self.assertEqual(tie.corrected_residual_s, -1.5)
        self.assertTrue(tie.branch_tie)
        self.assertLessEqual(
            abs(tie.corrected_residual_s),
            tie.period_s / 2.0,
        )

    def test_reference_folds_are_joint_deterministic_and_complete(self):
        assign = self.require_module_attribute("assign_reference_folds")
        distance = np.repeat(np.arange(1.0, 11.0), 8)
        azimuth = np.tile(np.arange(0.0, 360.0, 45.0), 10)
        first = assign(distance, azimuth)
        second = assign(distance, azimuth)

        np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
        self.assertEqual(first.assignment_hash, second.assignment_hash)
        self.assertEqual(first.fold_count, 5)
        self.assertEqual(first.fold_ids.shape, distance.shape)
        self.assertEqual(set(first.fold_ids.tolist()), set(range(5)))
        self.assertEqual(
            sum(np.count_nonzero(first.fold_ids == fold) for fold in range(5)),
            distance.size,
        )
        for quintile in range(5):
            for azimuth_block in range(8):
                block = (
                    (first.distance_quintile_ids == quintile)
                    & (first.azimuth_block_ids == azimuth_block)
                )
                if np.any(block):
                    self.assertEqual(
                        np.unique(first.fold_ids[block]).size,
                        1,
                    )
        all_indices = set(range(distance.size))
        for fold in range(5):
            self.assertEqual(
                set(first.holdout_indices[fold].tolist()),
                set(np.flatnonzero(first.fold_ids == fold).tolist()),
            )
            self.assertEqual(
                set(first.training_indices[fold].tolist()),
                all_indices - set(first.holdout_indices[fold].tolist()),
            )
        self.assertFalse(first.fold_ids.flags.writeable)

    def test_cv_selection_uses_one_percent_then_smaller_lambdas(self):
        config_class = self.require_module_attribute("ReferenceCvConfig")
        choose = self.require_module_attribute("select_reference_cv_config")

        def config(lambda_s, lambda_g, loss, calls=25):
            return config_class(
                lambda_s=lambda_s,
                lambda_g=lambda_g,
                fold_holdout_losses=np.full(5, loss),
                mean_holdout_loss=loss,
                optimizer_calls=calls,
            )

        configs = (
            config(0.0, 0.1, 1.000),
            config(0.1, 0.0, 1.009),
            config(0.0, 0.0, 1.011),
        )
        chosen = choose(configs)
        self.assertEqual((chosen.lambda_s, chosen.lambda_g), (0.1, 0.0))
        self.assertEqual(sum(item.optimizer_calls for item in configs), 75)

        full_budget = 25 * 5 * 5
        self.assertEqual(full_budget, 625)
        with self.assertRaisesRegex(ValueError, "optimizer_calls"):
            config(0.0, 0.0, 1.0, calls=26)
        boundary = (
            config(0.1, 0.1, 1.0),
            config(0.0, 0.0, 1.01),
        )
        self.assertEqual(choose(boundary), boundary[0])
        just_inside = (
            boundary[0],
            config(0.0, 0.0, 1.0099),
        )
        self.assertEqual(choose(just_inside), just_inside[1])
        zero_best = (
            config(0.1, 0.1, 0.0),
            config(0.0, 0.0, np.finfo(float).eps),
        )
        self.assertEqual(choose(zero_best), zero_best[0])
        zero_tie = (
            zero_best[0],
            config(0.0, 0.0, 0.0),
        )
        self.assertEqual(choose(zero_tie), zero_tie[1])

    def test_reference_start_set_has_required_composition_and_hash_dedup(self):
        generate = self.require_module_attribute("generate_reference_starts")
        periods = np.arange(2.5, 5.05, 0.05)
        first = generate(periods, max_starts=71, seed=20260717)
        second = generate(periods, max_starts=71, seed=20260717)

        self.assertEqual(len(first), 71)
        self.assertEqual(
            [start.kind for start in first[:39]],
            ["endpoint_linear"] * 39,
        )
        self.assertEqual(
            [start.kind for start in first[39:]],
            ["sine_perturbation"] * 32,
        )
        self.assertEqual(
            [start.velocity_hash for start in first],
            [start.velocity_hash for start in second],
        )
        self.assertEqual(len({start.velocity_hash for start in first}), 71)
        for start in first:
            self.assertEqual(start.velocities_km_s.shape, periods.shape)
            self.assertTrue(np.all(start.velocities_km_s >= 1.6))
            self.assertTrue(np.all(start.velocities_km_s <= 4.0))
            self.assertFalse(start.velocities_km_s.flags.writeable)
        self.assertEqual(
            [first[index].base_velocity_km_s for index in range(0, 39, 3)],
            list(np.linspace(1.6, 4.0, 13)),
        )
        self.assertEqual(
            [first[index].endpoint_slope_km_s for index in range(3)],
            [-0.2, 0.0, 0.2],
        )

    def test_basin_clustering_uses_both_strict_thresholds(self):
        cluster = self.require_module_attribute("cluster_reference_solutions")
        velocity = np.full(51, 2.5)
        slowness = 1.0 / velocity
        solutions = (
            types.SimpleNamespace(
                target_velocities_km_s=np.full(4, 2.5),
                phase_slowness_s_km=slowness,
                objective=1.0,
            ),
            types.SimpleNamespace(
                target_velocities_km_s=np.full(4, 2.519),
                phase_slowness_s_km=slowness + 0.0019,
                objective=0.9,
            ),
            types.SimpleNamespace(
                target_velocities_km_s=np.full(4, 2.520),
                phase_slowness_s_km=slowness,
                objective=0.8,
            ),
            types.SimpleNamespace(
                target_velocities_km_s=np.full(4, 2.5),
                phase_slowness_s_km=slowness + 0.002,
                objective=0.7,
            ),
        )
        result = cluster(solutions)

        np.testing.assert_array_equal(result.basin_ids, np.array([0, 0, 1, 2]))
        self.assertEqual(result.representative_indices, (1, 2, 3))
        self.assertFalse(result.basin_ids.flags.writeable)

    def test_basin_clustering_uses_complete_linkage_not_anchor_only(self):
        cluster = self.require_module_attribute("cluster_reference_solutions")
        slowness = np.full(51, 0.4)
        solutions = (
            types.SimpleNamespace(
                target_velocities_km_s=np.full(4, 2.500),
                phase_slowness_s_km=slowness,
                objective=1.0,
            ),
            types.SimpleNamespace(
                target_velocities_km_s=np.full(4, 2.519),
                phase_slowness_s_km=slowness,
                objective=0.8,
            ),
            types.SimpleNamespace(
                target_velocities_km_s=np.full(4, 2.481),
                phase_slowness_s_km=slowness,
                objective=0.7,
            ),
        )
        result = cluster(solutions)
        np.testing.assert_array_equal(result.basin_ids, np.array([0, 0, 1]))
        self.assertEqual(result.representative_indices, (1, 2))

    def test_reference_objective_uses_circular_data_and_derived_group_slowness(self):
        observation_class = self.require_module_attribute(
            "ReferenceObservation"
        )
        objective = self.require_module_attribute("reference_fit_objective")
        periods = np.arange(2.5, 5.05, 0.05)
        velocity = 2.5 + 0.08 * (periods - periods.mean())
        slowness = 1.0 / velocity
        group_slowness = self.mod.phase_slowness_to_group_slowness(
            periods,
            slowness,
        )
        observations = tuple(
            observation_class(
                pair_name=f"P{index}",
                distance_km=30.0 + index,
                azimuth_deg=45.0 * index,
                instantaneous_period_s=float(periods[index * 10]),
                anchored_raw_time_s=float(
                    (30.0 + index) * slowness[index * 10]
                    + (index - 2) * periods[index * 10]
                ),
                group_slowness_s_km=float(group_slowness[index * 10]),
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            )
            for index in range(5)
        )
        exact = objective(
            slowness,
            observations,
            lambda_s=0.0,
            lambda_g=1.0,
            periods_s=periods,
        )
        wrong = objective(
            np.full(periods.size, 1.0 / 3.7),
            observations,
            lambda_s=0.0,
            lambda_g=1.0,
            periods_s=periods,
        )
        self.assertLess(exact, 1e-20)
        self.assertGreater(wrong, exact)

    def test_reference_group_objective_uses_grid_median_and_huber_rho_one(self):
        observation_class = self.require_module_attribute(
            "ReferenceObservation"
        )
        objective = self.require_module_attribute("reference_fit_objective")
        periods = np.round(np.linspace(2.5, 5.0, 51), 12)
        slowness = np.full(51, 0.4)
        group_values = (0.4, 0.4, 1.0)
        observations = tuple(
            observation_class(
                pair_name=f"G{index}",
                distance_km=25.0 + index,
                azimuth_deg=0.0,
                instantaneous_period_s=3.0,
                anchored_raw_time_s=(25.0 + index) * 0.4,
                group_slowness_s_km=group_value,
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            )
            for index, group_value in enumerate(group_values)
        )
        self.assertEqual(
            objective(
                slowness,
                observations,
                lambda_s=0.0,
                lambda_g=1.0,
                periods_s=periods,
            ),
            0.0,
        )

    def test_cross_validation_uses_fixed_starts_and_exact_call_budget(self):
        cross_validate = self.require_module_attribute(
            "cross_validate_reference_fit"
        )
        observation_class = self.require_module_attribute(
            "ReferenceObservation"
        )
        observations = tuple(
            observation_class(
                pair_name=f"P{index}",
                distance_km=20.0 + index,
                azimuth_deg=45.0 * index,
                instantaneous_period_s=3.0 + 0.1 * (index % 5),
                anchored_raw_time_s=(20.0 + index) / 2.8,
                group_slowness_s_km=1.0 / 2.8,
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            )
            for index in range(10)
        )
        calls = []

        def optimizer(objective, x0, bounds, maxiter):
            calls.append((x0.copy(), tuple(bounds), maxiter))
            value = float(objective(x0))
            return types.SimpleNamespace(
                x=x0.copy(),
                fun=value,
                success=True,
                message="spy",
            )

        result = cross_validate(
            observations,
            lambda_values=(0.0,),
            optimizer=optimizer,
        )

        self.assertEqual(result.optimizer_calls, 25)
        self.assertEqual(len(calls), 25)
        self.assertEqual([call[2] for call in calls], [200] * 25)
        expected = np.tile(
            1.0 / np.array([1.7, 2.25, 2.8, 3.35, 3.9]),
            5,
        )
        np.testing.assert_allclose(
            np.array([call[0][0] for call in calls]),
            expected,
        )
        self.assertEqual(len(result.configs), 1)
        self.assertEqual(result.configs[0].optimizer_calls, 25)
        self.assertEqual(
            (result.selected.lambda_s, result.selected.lambda_g),
            (0.0, 0.0),
        )
        config_class = self.require_module_attribute("ReferenceCvConfig")
        result_class = self.require_module_attribute("ReferenceCvResult")
        concentrated = config_class(
            lambda_s=0.0,
            lambda_g=0.0,
            fold_holdout_losses=np.array([0.0, 0.0, 0.0, 0.0, 5.0]),
            mean_holdout_loss=1.0,
            optimizer_calls=25,
        )
        old_digest_source = "|".join(
            (
                result.fold_assignment.assignment_hash,
                "0,0,1,25,0,0,0,0,5",
            )
        )
        concentrated_result = result_class(
            fold_assignment=result.fold_assignment,
            configs=(concentrated,),
            selected=concentrated,
            optimizer_calls=25,
            result_hash=self.mod.hashlib.sha256(
                old_digest_source.encode("ascii")
            ).hexdigest(),
        )
        uniform = replace(
            concentrated,
            fold_holdout_losses=np.ones(5),
        )
        with self.assertRaisesRegex(ValueError, "result_hash"):
            replace(
                concentrated_result,
                configs=(uniform,),
                selected=uniform,
            )

    def test_final_alias_loss_evaluates_each_frozen_fold_without_refitting(self):
        evaluate = self.require_module_attribute(
            "reference_final_fold_holdout_losses"
        )
        observation_class = self.require_module_attribute(
            "ReferenceObservation"
        )
        periods = np.round(np.linspace(2.5, 5.0, 51), 12)
        observations = tuple(
            observation_class(
                pair_name=f"F{index}",
                distance_km=20.0 + index,
                azimuth_deg=45.0 * index,
                instantaneous_period_s=3.0,
                anchored_raw_time_s=(20.0 + index) / 2.5,
                group_slowness_s_km=0.4,
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            )
            for index in range(11)
        )
        assignment = self.mod.assign_reference_folds(
            [row.distance_km for row in observations],
            [row.azimuth_deg for row in observations],
        )
        seen = []
        fold_values = iter((0.0, 0.0, 0.0, 0.0, 5.0))

        def phase_only(_candidate, subset, *, periods_s):
            seen.append(tuple(row.pair_name for row in subset))
            self.assertIs(periods_s, periods)
            return next(fold_values)

        with mock.patch.object(
            self.mod,
            "reference_phase_holdout_loss",
            side_effect=phase_only,
        ):
            losses, mean_loss = evaluate(
                np.full(51, 0.4),
                observations,
                assignment,
                periods_s=periods,
            )
        self.assertEqual(len(seen), 5)
        self.assertEqual(
            seen,
            [
                tuple(observations[index].pair_name for index in holdout)
                for holdout in assignment.holdout_indices
            ],
        )
        np.testing.assert_array_equal(losses, np.array([0, 0, 0, 0, 5]))
        self.assertEqual(mean_loss, 1.0)
        self.assertFalse(losses.flags.writeable)
        base = types.SimpleNamespace(
            objective=1.0,
            holdout_loss=float(np.mean(np.ones(5))),
            target_velocities_km_s=np.full(4, 2.5),
        )
        near_alias = types.SimpleNamespace(
            objective=2.0,
            holdout_loss=float(np.mean(np.full(5, 1.009))),
            target_velocities_km_s=np.full(4, 2.7),
        )
        changed_one_fold = types.SimpleNamespace(
            objective=2.0,
            holdout_loss=float(
                np.mean(np.array([1.009, 1.009, 1.009, 1.009, 1.10]))
            ),
            target_velocities_km_s=np.full(4, 2.7),
        )
        other = tuple(
            types.SimpleNamespace(
                objective=3.0 + index,
                holdout_loss=3.0 + index,
                target_velocities_km_s=np.full(4, 3.0 + index),
            )
            for index in range(3)
        )
        self.assertEqual(
            self.mod.reference_alias_status((base, near_alias, *other)),
            "reference_alias_unresolved",
        )
        self.assertEqual(
            self.mod.reference_alias_status(
                (base, changed_one_fold, *other)
            ),
            "accepted",
        )

    def test_reference_alias_rule_rejects_near_equal_distinct_targets(self):
        decide = self.require_module_attribute("reference_alias_status")
        base = types.SimpleNamespace(
            objective=1.0,
            target_velocities_km_s=np.array([2.5, 2.6, 2.7, 2.8]),
        )
        alias = types.SimpleNamespace(
            objective=1.009,
            target_velocities_km_s=np.array([2.5, 2.6, 2.81, 2.8]),
        )
        other = tuple(
            types.SimpleNamespace(
                objective=1.2 + index,
                target_velocities_km_s=np.full(4, 3.0 + 0.1 * index),
            )
            for index in range(3)
        )
        self.assertEqual(
            decide((base, alias, *other)),
            "reference_alias_unresolved",
        )
        separated = replace(
            self.require_module_attribute("AliasCandidate")(
                objective=1.02,
                target_velocities_km_s=alias.target_velocities_km_s,
            ),
            objective=1.02,
        )
        self.assertEqual(
            decide((base, separated, *other)),
            "accepted",
        )
        exact_boundary = types.SimpleNamespace(
            objective=1.01,
            target_velocities_km_s=alias.target_velocities_km_s,
        )
        self.assertEqual(
            decide((base, exact_boundary, *other)),
            "accepted",
        )
        zero_best = types.SimpleNamespace(
            objective=2.0,
            holdout_loss=0.0,
            target_velocities_km_s=base.target_velocities_km_s,
        )
        zero_alias = types.SimpleNamespace(
            objective=3.0,
            holdout_loss=0.0,
            target_velocities_km_s=alias.target_velocities_km_s,
        )
        self.assertEqual(
            decide((zero_best, zero_alias, *other)),
            "reference_alias_unresolved",
        )
        positive_alias = types.SimpleNamespace(
            objective=3.0,
            holdout_loss=np.finfo(float).eps,
            target_velocities_km_s=alias.target_velocities_km_s,
        )
        self.assertEqual(
            decide((zero_best, positive_alias, *other)),
            "accepted",
        )
        self.assertEqual(
            decide((base, alias, *other[:2])),
            "reference_search_insufficient_minima",
        )

    def test_final_reference_fit_recovers_unambiguous_synthetic_curve_in_budget(self):
        fit = self.require_module_attribute("fit_reference_dispersion")
        observation_class = self.require_module_attribute(
            "ReferenceObservation"
        )
        periods = np.arange(2.5, 5.05, 0.05)
        true_velocity = 2.8
        observations = tuple(
            observation_class(
                pair_name=f"P{index}",
                distance_km=24.0 + 3.0 * index,
                azimuth_deg=45.0 * index,
                instantaneous_period_s=float(periods[(7 * index) % 51]),
                anchored_raw_time_s=float(
                    (24.0 + 3.0 * index) / true_velocity
                    + (index % 3 - 1) * periods[(7 * index) % 51]
                ),
                group_slowness_s_km=1.0 / true_velocity,
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            )
            for index in range(20)
        )
        calls = []
        basin_velocities = (2.8, 2.45, 3.15, 3.5, 3.85)

        def optimizer(objective, x0, bounds, maxiter):
            calls.append(maxiter)
            if maxiter == 200:
                candidate = x0.copy()
            else:
                final_index = sum(value == 500 for value in calls) - 1
                candidate = np.full(
                    x0.size,
                    1.0 / basin_velocities[final_index % 5],
                )
            return types.SimpleNamespace(
                x=candidate,
                fun=float(objective(candidate)),
                success=True,
                message="synthetic-spy",
            )

        result = fit(observations, optimizer=optimizer)

        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.cv_optimizer_calls, 625)
        self.assertEqual(result.final_optimizer_calls, 71)
        self.assertEqual(result.optimizer_calls, 696)
        self.assertEqual(len(result.starts), 71)
        self.assertEqual(len(result.local_solutions), 71)
        self.assertGreaterEqual(len(result.representative_indices), 5)
        self.assertEqual(result.periods_s[0], 2.5)
        self.assertEqual(result.periods_s[-1], 5.0)
        np.testing.assert_allclose(np.diff(result.periods_s), 0.05)
        np.testing.assert_allclose(
            result.target_velocities_km_s,
            np.full(4, true_velocity),
            atol=1e-12,
        )
        self.assertEqual(calls.count(200), 625)
        self.assertEqual(calls.count(500), 71)
        self.assertFalse(result.phase_slowness_s_km.flags.writeable)
        self.assertFalse(result.basin_ids.flags.writeable)
        for changes, message in (
            (
                {
                    "group_slowness_s_km": (
                        result.group_slowness_s_km + 0.01
                    )
                },
                "group_slowness",
            ),
            (
                {
                    "target_velocities_km_s": (
                        result.target_velocities_km_s + 0.01
                    )
                },
                "target_velocities",
            ),
            ({"lambda_g": 1.0}, "lambda"),
            (
                {"status": "reference_alias_unresolved"},
                "status",
            ),
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    replace(result, **changes)
        reversed_representatives = tuple(
            reversed(result.representative_indices)
        )
        with mock.patch.object(
            self.mod,
            "_cluster_converged_reference_solutions",
            return_value=(result.basin_ids, reversed_representatives),
        ):
            with self.assertRaisesRegex(ValueError, "result_hash"):
                replace(
                    result,
                    representative_indices=reversed_representatives,
                )
        solution_index = max(
            range(len(result.local_solutions)),
            key=lambda index: result.local_solutions[index].holdout_loss,
        )
        original_solution = result.local_solutions[solution_index]
        redistributed_folds = original_solution.fold_holdout_losses.copy()
        maximum_index = int(np.argmax(redistributed_folds))
        receiver_index = (maximum_index + 1) % redistributed_folds.size
        transfer = redistributed_folds[maximum_index] / 2.0
        self.assertGreater(transfer, 0.0)
        redistributed_folds[maximum_index] -= transfer
        redistributed_folds[receiver_index] += transfer
        tampered_solutions = list(result.local_solutions)
        tampered_solutions[solution_index] = replace(
            original_solution,
            fold_holdout_losses=redistributed_folds,
        )
        with self.assertRaisesRegex(ValueError, "result_hash"):
            replace(result, local_solutions=tuple(tampered_solutions))

    def test_final_reference_fit_rejects_deliberately_aliased_synthetic_data(self):
        fit = self.require_module_attribute("fit_reference_dispersion")
        observation_class = self.require_module_attribute(
            "ReferenceObservation"
        )
        periods = np.arange(2.5, 5.05, 0.05)
        slowness_difference = 1.0 / 2.5 - 1.0 / 2.8
        observations = tuple(
            observation_class(
                pair_name=f"A{index}",
                distance_km=float(
                    periods[(5 * index) % 51] / slowness_difference
                ),
                azimuth_deg=45.0 * index,
                instantaneous_period_s=float(periods[(5 * index) % 51]),
                anchored_raw_time_s=float(
                    periods[(5 * index) % 51]
                    / slowness_difference
                    / 2.5
                ),
                group_slowness_s_km=1.0 / 2.5,
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            )
            for index in range(20)
        )
        final_velocities = (2.5, 2.8, 3.15, 3.5, 3.85)
        final_call_count = 0

        def optimizer(objective, x0, bounds, maxiter):
            nonlocal final_call_count
            if maxiter == 200:
                candidate = x0.copy()
            else:
                candidate = np.full(
                    x0.size,
                    1.0 / final_velocities[final_call_count % 5],
                )
                final_call_count += 1
            return types.SimpleNamespace(
                x=candidate,
                fun=float(objective(candidate)),
                success=True,
                message="aliased-synthetic-spy",
            )

        result = fit(observations, optimizer=optimizer)

        self.assertEqual(result.status, "reference_alias_unresolved")
        self.assertEqual(result.optimizer_calls, 696)
        representative_velocities = sorted(
            round(
                float(
                    result.local_solutions[index].target_velocities_km_s[0]
                ),
                2,
            )
            for index in result.representative_indices
        )
        self.assertIn(2.5, representative_velocities)
        self.assertIn(2.8, representative_velocities)

    def test_batch_cycle_resolution_uses_each_exact_observation_period(self):
        batch = self.require_module_attribute("resolve_reference_cycles")
        reference_periods = np.arange(2.5, 5.05, 0.05)
        reference_slowness = np.full(51, 0.4)
        distance = np.array([30.0, 30.0])
        periods = np.array([3.13, 5.0])
        reference_time = distance * 0.4
        bensen_raw = reference_time - np.array([2.0, 1.0]) * periods
        lin_raw = reference_time + np.array([2.0, 1.0]) * periods

        bensen = batch(
            raw_times_s=bensen_raw,
            distance_km=distance,
            observation_periods_s=periods,
            reference_periods_s=reference_periods,
            reference_slowness_s_km=reference_slowness,
            convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
        )
        lin = batch(
            raw_times_s=lin_raw,
            distance_km=distance,
            observation_periods_s=periods,
            reference_periods_s=reference_periods,
            reference_slowness_s_km=reference_slowness,
            convention=self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF,
        )
        self.assertEqual([row.cycle_count for row in bensen], [2, 1])
        self.assertEqual([row.cycle_count for row in lin], [2, 1])
        self.assertEqual([row.period_s for row in bensen], [3.13, 5.0])
        for row in (*bensen, *lin):
            self.assertLessEqual(
                abs(row.corrected_residual_s),
                row.period_s / 2.0,
            )

    def test_right_column_huber_fit_resists_outlier_and_bootstrap_is_reproducible(
        self,
    ):
        fit = self.require_module_attribute("fit_right_column_slowness")
        distance = np.linspace(10.0, 100.0, 10)
        true_slowness = 0.4
        travel_time = distance * true_slowness
        travel_time[-1] += 30.0

        first = fit(
            distance,
            travel_time,
            bootstrap_samples=400,
            seed=20260717,
        )
        second = fit(
            distance,
            travel_time,
            bootstrap_samples=400,
            seed=20260717,
        )

        self.assertLess(
            abs(first.huber_slowness_s_km - true_slowness),
            abs(first.ordinary_ls_slowness_s_km - true_slowness),
        )
        self.assertAlmostEqual(
            first.huber_velocity_km_s,
            1.0 / first.huber_slowness_s_km,
        )
        self.assertAlmostEqual(
            first.ordinary_ls_velocity_km_s,
            1.0 / first.ordinary_ls_slowness_s_km,
        )
        self.assertAlmostEqual(
            first.path_velocity_std_km_s,
            float(np.std(distance / travel_time)),
        )
        np.testing.assert_array_equal(
            first.bootstrap_velocity_ci95_km_s,
            second.bootstrap_velocity_ci95_km_s,
        )
        self.assertEqual(
            first.bootstrap_velocity_std_km_s,
            second.bootstrap_velocity_std_km_s,
        )
        self.assertFalse(first.bootstrap_velocity_ci95_km_s.flags.writeable)
        self.assertEqual(first.bootstrap_samples, 400)
        self.assertEqual(first.seed, 20260717)

    def test_full_341_grid_dp_performance_budget(self):
        trace = self.require_module_attribute("_trace_optimal_ridge")
        find_candidate_ridges = self.require_module_attribute("find_candidate_ridges")
        periods = np.arange(2.5, 5.05, 0.05)
        velocity = 1.6 + 0.01 * np.arange(341, dtype=float)
        rows = 110 + np.rint(5.0 * np.sin(np.linspace(0.0, 2.0 * math.pi, periods.size))).astype(int)
        energy = np.full((periods.size, velocity.size), 0.01, dtype=float)
        amplitude = np.full_like(energy, 0.01)
        for offset, peak in ((0, 1.0), (15, 0.9), (-15, 0.8)):
            candidate_rows = rows + offset
            energy[np.arange(periods.size), candidate_rows] = peak
            amplitude[np.arange(periods.size), candidate_rows] = peak

        # Warm the NumPy/SciPy import and allocator path before measuring.
        trace(energy[:3, :11], velocity[:11], beta1=1.0, beta2=2.0)
        started = time.perf_counter()
        trace(energy, velocity, beta1=1.0, beta2=2.0)
        one_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        find_candidate_ridges(
            scaled_log_energy=energy,
            normalized_envelope_amplitude=amplitude,
            periods_s=periods,
            velocity_axis_km_s=velocity,
            beta1=1.0,
            beta2=2.0,
            max_candidates=3,
        )
        three_elapsed = time.perf_counter() - started

        self.assertLess(one_elapsed, 0.35)
        self.assertLess(three_elapsed, 1.05)


class PhaseMatchedFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def require_module_attribute(self, name):
        value = getattr(self.mod, name, None)
        self.assertIsNotNone(value, f"bensen_phase_ftan must define {name}")
        return value

    def test_second_pass_curve_accepts_already_prepared_lin_waveform(self):
        convention = self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF
        config = self.mod.FtanConfig()
        dt_s = 0.04
        time_s = np.arange(0.0, 60.0, dt_s)
        raw = np.exp(-0.5 * ((time_s - 8.0) / 4.0) ** 2) * np.sin(
            2.0 * np.pi * (time_s - 8.0) / 3.5
        )
        prepared = self.mod.prepare_phase_waveform(time_s, raw, convention)
        raw_trace = self.mod.DatTrace(
            "AA__BB", 20.0, dt_s, time_s,
            raw.copy(), raw.copy(), raw.copy(),
            0.0, 0.0, 0.1, 0.1,
        )
        prepared_trace = self.mod.DatTrace(
            "AA__BB", 20.0, dt_s, time_s,
            prepared.copy(), prepared.copy(), prepared.copy(),
            0.0, 0.0, 0.1, 0.1,
        )
        expected = self.mod.measure_phase_curve(
            raw_trace,
            periods_s=config.periods_s,
            velocity_axis_km_s=config.group_velocities_km_s,
            alpha=20.0,
            beta1=1.0,
            beta2=2.0,
            convention=convention,
        )
        actual = self.mod.measure_phase_curve(
            prepared_trace,
            periods_s=config.periods_s,
            velocity_axis_km_s=config.group_velocities_km_s,
            alpha=20.0,
            beta1=1.0,
            beta2=2.0,
            convention=convention,
            waveform_is_prepared=True,
        )
        self.assertIsNotNone(expected)
        self.assertIsNotNone(actual)
        np.testing.assert_allclose(actual.group_times_s, expected.group_times_s)
        np.testing.assert_array_equal(
            actual.measurement_valid,
            expected.measurement_valid,
        )
        valid = np.flatnonzero(actual.measurement_valid)
        np.testing.assert_allclose(
            [actual.measurements[index].raw_phase_time_s for index in valid],
            [expected.measurements[index].raw_phase_time_s for index in valid],
        )

    def test_phase_matched_filter_freezes_bensen_cleaning_parameters(self):
        phase_match = self.require_module_attribute("phase_matched_filter")
        dt_s = 0.05
        time_s = np.arange(0.0, 80.0, dt_s)
        waveform = (
            np.sin(2.0 * np.pi * time_s / 3.0)
            * np.exp(-0.5 * ((time_s - 35.0) / 5.0) ** 2)
            + 0.25
            * np.sin(2.0 * np.pi * time_s / 4.5)
            * np.exp(-0.5 * ((time_s - 60.0) / 2.0) ** 2)
        )
        result = phase_match(
            waveform,
            dt_s=dt_s,
            periods_s=np.array([2.5, 3.0, 3.5, 4.0, 5.0]),
            group_travel_times_s=np.array([31.0, 32.0, 34.0, 36.0, 39.0]),
            first_pass_alpha=30.0,
        )

        expected_center = int(np.argmax(result.compressed_envelope))
        self.assertEqual(result.cut_center_index, expected_center)
        self.assertEqual(result.cut_half_width_s, 10.0)
        self.assertEqual(result.cut_taper_alpha, 0.25)
        self.assertEqual(result.second_pass_alpha, 50.0)
        self.assertEqual(result.cleaned_waveform.shape, waveform.shape)
        self.assertTrue(np.all(np.isfinite(result.cleaned_waveform)))
        self.assertEqual(result.cleaning_window[expected_center], 1.0)
        self.assertTrue(np.all(result.cleaning_window >= 0.0))
        self.assertTrue(np.all(result.cleaning_window <= 1.0))

    def test_phase_matched_filter_uses_twice_alpha_below_cap_and_rejects_bad_curve(self):
        phase_match = self.require_module_attribute("phase_matched_filter")
        waveform = np.zeros(1024)
        waveform[400] = 1.0
        result = phase_match(
            waveform,
            dt_s=0.1,
            periods_s=np.array([2.5, 3.0, 4.0, 5.0]),
            group_travel_times_s=np.array([20.0, 21.0, 22.0, 24.0]),
            first_pass_alpha=12.0,
        )
        self.assertEqual(result.second_pass_alpha, 24.0)
        with self.assertRaisesRegex(ValueError, "same shape"):
            phase_match(
                waveform,
                dt_s=0.1,
                periods_s=np.array([2.5, 3.0]),
                group_travel_times_s=np.array([20.0]),
                first_pass_alpha=12.0,
            )
        with self.assertRaisesRegex(ValueError, "frozen"):
            phase_match(
                waveform,
                dt_s=0.1,
                periods_s=np.array([2.5, 3.0, 4.0, 5.0]),
                group_travel_times_s=np.array([20.0, 21.0, 22.0, 24.0]),
                first_pass_alpha=12.0,
                maximum_period_s=4.0,
            )

    def test_phase_matched_filter_compresses_a_known_dispersive_packet(self):
        phase_match = self.require_module_attribute("phase_matched_filter")
        npts = 4096
        dt_s = 0.05
        periods = np.array([2.5, 3.0, 3.5, 4.0, 5.0])
        group_times = np.array([28.0, 31.0, 35.0, 39.0, 43.0])
        frequency = np.fft.rfftfreq(npts, dt_s)
        curve_frequency = 1.0 / periods
        order = np.argsort(curve_frequency)
        interpolated = np.interp(
            frequency,
            curve_frequency[order],
            group_times[order],
        )
        reference = float(np.median(group_times))
        differential = interpolated - reference
        in_band = (
            (frequency >= np.min(curve_frequency))
            & (frequency <= np.max(curve_frequency))
        )
        differential[~in_band] = 0.0
        omega = 2.0 * np.pi * frequency
        integral = np.zeros_like(omega)
        integral[1:] = np.cumsum(
            0.5
            * (differential[1:] + differential[:-1])
            * np.diff(omega)
        )
        amplitude = np.exp(
            -0.5 * ((frequency - 0.30) / 0.055) ** 2
        )
        spectrum = amplitude * np.exp(
            -1j * (omega * reference + integral)
        )
        waveform = np.fft.irfft(spectrum, npts)
        result = phase_match(
            waveform,
            dt_s=dt_s,
            periods_s=periods,
            group_travel_times_s=group_times,
            first_pass_alpha=12.0,
        )
        time_s = np.arange(npts) * dt_s

        def envelope_width(values):
            envelope = np.abs(self.mod.hilbert(values))
            weights = envelope / np.sum(envelope)
            center = np.sum(time_s * weights)
            return np.sqrt(np.sum(weights * (time_s - center) ** 2))

        self.assertLess(
            envelope_width(result.compressed_waveform),
            0.8 * envelope_width(waveform),
        )

    def test_phase_matching_executes_the_frozen_second_gaussian_ftan(self):
        second_pass = self.require_module_attribute(
            "phase_matched_second_pass_ftan"
        )
        periods = np.array([2.5, 3.0, 3.5, 4.0, 5.0])
        waveform = np.zeros(2048)
        waveform[600:605] = [0.2, 0.6, 1.0, 0.6, 0.2]
        result = second_pass(
            waveform,
            dt_s=0.05,
            periods_s=periods,
            group_travel_times_s=np.array(
                [28.0, 30.0, 32.0, 35.0, 38.0]
            ),
            first_pass_alpha=12.0,
        )
        direct = self.mod.gaussian_filter_bank(
            result.cleaning.cleaned_waveform,
            dt_s=0.05,
            periods_s=periods,
            alpha=24.0,
        )
        np.testing.assert_array_equal(
            result.second_pass_filter_bank.filtered_waveforms,
            direct.filtered_waveforms,
        )
        np.testing.assert_array_equal(
            result.second_pass_filter_bank.envelope,
            direct.envelope,
        )


if __name__ == "__main__":
    unittest.main()
