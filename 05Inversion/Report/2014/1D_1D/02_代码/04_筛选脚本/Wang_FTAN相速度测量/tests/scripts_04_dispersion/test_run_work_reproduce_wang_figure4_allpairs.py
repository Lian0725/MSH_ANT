import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "04_dispersion"
    / "run_work_reproduce_wang_figure4_allpairs.py"
)


def load_module():
    script_dir = str(MODULE_PATH.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "run_work_reproduce_wang_figure4_allpairs",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_fixture_processor(task):
    value = int(task["value"])
    if value == 3:
        return {
            "pair_name": task["pair_name"],
            "ok": False,
            "failure_kind": "expected_scientific_rejection",
            "reason": "fixture_qc",
        }
    return {
        "pair_name": task["pair_name"],
        "ok": True,
        "value": value,
    }


class RunWorkReproduceWangFigure4AllPairsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def _write_stack_fixture(
        self,
        root,
        *,
        relative_path="AA/BB/stack_pws.h5",
        component="ZZ",
        data=None,
        dt_s=0.04,
        maxlag_s=0.08,
        group_name="CrossCorrelation",
        attrs=None,
    ):
        path = Path(root) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        values = (
            np.asarray([5.0, 3.0, 1.0, 7.0, 9.0], dtype=float)
            if data is None
            else np.asarray(data, dtype=float)
        )
        with h5py.File(path, "w") as handle:
            group = handle.require_group(
                f"AuxiliaryData/{group_name}"
            )
            dataset = group.create_dataset(component, data=values)
            dataset.attrs["dt"] = dt_s
            dataset.attrs["maxlag"] = maxlag_s
            for key, value in (attrs or {}).items():
                dataset.attrs[key] = value
        return path

    def test_strict_stack_reader_uses_requested_component_and_symmetric_lags(
        self,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = self._write_stack_fixture(temporary.name)
        with h5py.File(path, "a") as handle:
            dataset = handle["AuxiliaryData/CrossCorrelation"].create_dataset(
                "TT",
                data=np.asarray([4.0, 2.0, 0.0, 6.0, 8.0]),
            )
            dataset.attrs["dt"] = 0.04
            dataset.attrs["maxlag"] = 0.08

        trace = self.mod.read_stack_trace(
            path,
            pair_name="AA__BB",
            component="TT",
        )

        self.assertEqual(trace.pair_name, "AA__BB")
        self.assertEqual(trace.dt_s, 0.04)
        self.assertEqual(trace.maxlag_s, 0.08)
        np.testing.assert_array_equal(
            trace.time_positive_s,
            [0.0, 0.04, 0.08],
        )
        np.testing.assert_array_equal(trace.positive_lag, [0.0, 6.0, 8.0])
        np.testing.assert_array_equal(
            trace.negative_lag_reversed,
            [0.0, 2.0, 4.0],
        )
        np.testing.assert_array_equal(trace.symmetric, [0.0, 4.0, 6.0])
        self.assertGreater(trace.branch_mismatch, 0.0)
        for array in (
            trace.time_positive_s,
            trace.positive_lag,
            trace.negative_lag_reversed,
            trace.symmetric,
        ):
            self.assertFalse(array.flags.writeable)

    def test_strict_stack_reader_reports_structured_input_failures(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        cases = []

        missing = self._write_stack_fixture(
            root / "missing",
            component="TT",
        )
        cases.append((missing, "ZZ", "missing_ZZ"))

        zero = self._write_stack_fixture(
            root / "zero",
            data=np.zeros(5),
        )
        cases.append((zero, "ZZ", "all_zero"))

        invalid_dt = self._write_stack_fixture(
            root / "invalid_dt",
            dt_s=0.0,
        )
        cases.append((invalid_dt, "ZZ", "invalid_dt"))

        bad_length = self._write_stack_fixture(
            root / "bad_length",
            data=np.ones(4),
        )
        cases.append((bad_length, "ZZ", "unexpected_length"))

        multiple = self._write_stack_fixture(root / "multiple")
        with h5py.File(multiple, "a") as handle:
            dataset = handle.require_group(
                "AuxiliaryData/Other"
            ).create_dataset("ZZ", data=np.ones(5))
            dataset.attrs["dt"] = 0.04
            dataset.attrs["maxlag"] = 0.08
        cases.append((multiple, "ZZ", "multiple_matching_components"))

        for path, component, expected_status in cases:
            with self.subTest(status=expected_status):
                with self.assertRaises(self.mod.MeasurementError) as raised:
                    self.mod.read_stack_trace(
                        path,
                        pair_name="AA__BB",
                        component=component,
                    )
                self.assertEqual(raised.exception.status, expected_status)

    def test_input_inventory_and_lineage_are_evidence_based(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        raw_root = root / "raw"
        despiked_root = root / "despiked"
        evidence = {
            "response_removed": True,
            "physical_quantity": "velocity",
            "lag_storage_direction": "negative_to_positive",
        }
        raw = self._write_stack_fixture(
            raw_root,
            attrs=evidence,
        )
        self._write_stack_fixture(
            despiked_root,
            data=np.asarray([5.0, 3.0, 1.0, 7.1, 8.9]),
            attrs=evidence,
        )
        config = root / "preprocessing.yaml"
        config.write_text(
            "response_removed: true\n"
            "physical_quantity: velocity\n"
            "lag_storage_direction: negative_to_positive\n",
            encoding="utf-8",
        )

        audit = self.mod.audit_input_inventory_and_lineage(
            despiked_root,
            component="ZZ",
            preprocessing_config=config,
            raw_stack_root=raw_root,
            phase_sample_limit=100,
        )

        self.assertEqual(audit["stack_file_count"], 1)
        self.assertEqual(audit["valid_component_count"], 1)
        self.assertEqual(audit["dt_distribution"], {"0.04": 1})
        self.assertEqual(audit["sample_count_distribution"], {"5": 1})
        self.assertEqual(audit["maxlag_distribution"], {"0.08": 1})
        self.assertTrue(audit["instrument_response"]["removed"])
        self.assertEqual(
            audit["stack_quantity"]["physical_quantity"],
            "velocity",
        )
        self.assertEqual(
            audit["lag_storage"]["direction"],
            "negative_to_positive",
        )
        self.assertEqual(audit["lineage_status"], "confirmed")
        self.assertEqual(
            audit["instrument_response"]["evidence"][0]["count"],
            2,
        )
        self.assertLessEqual(
            len(
                audit["instrument_response"]["evidence"][0][
                    "example_sources"
                ]
            ),
            5,
        )
        self.assertEqual(
            audit["phase_comparison"]["paired_file_count"],
            1,
        )
        self.assertGreater(
            audit["phase_comparison"]["frequency_bin_count"],
            0,
        )
        self.assertEqual(
            audit["phase_comparison"]["period_band_s"],
            [2.5, 5.0],
        )
        self.assertEqual(raw.name, "stack_pws.h5")

    def test_lineage_unknown_and_contradictory_are_not_guessed(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        unknown_root = root / "unknown"
        self._write_stack_fixture(unknown_root)
        unknown = self.mod.audit_input_inventory_and_lineage(
            unknown_root,
            component="ZZ",
        )
        self.assertEqual(unknown["lineage_status"], "unknown")
        self.assertIsNone(unknown["instrument_response"]["removed"])
        self.assertEqual(
            unknown["stack_quantity"]["physical_quantity"],
            "unknown",
        )

        contradictory_root = root / "contradictory"
        self._write_stack_fixture(
            contradictory_root,
            attrs={"response_removed": True},
        )
        config = root / "contradictory.yaml"
        config.write_text(
            "response_removed: false\n",
            encoding="utf-8",
        )
        contradictory = self.mod.audit_input_inventory_and_lineage(
            contradictory_root,
            component="ZZ",
            preprocessing_config=config,
        )
        self.assertEqual(
            contradictory["instrument_response"]["status"],
            "contradictory",
        )
        self.assertEqual(
            contradictory["lineage_status"],
            "contradictory",
        )

    def test_missing_pyyaml_uses_restricted_scalar_lineage_parser(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        stack_root = root / "stack"
        self._write_stack_fixture(stack_root)
        config = root / "preprocessing.yaml"
        config.write_text("response_removed: true\n", encoding="utf-8")
        with mock.patch.object(self.mod, "yaml", None):
            audit = self.mod.audit_input_inventory_and_lineage(
                stack_root,
                preprocessing_config=config,
            )
        self.assertEqual(audit["lineage_status"], "unknown")
        self.assertEqual(
            audit["preprocessing_config_evidence"]["status"],
            "restricted_scalar_parser",
        )
        self.assertEqual(
            audit["preprocessing_config_evidence"]["path"],
            str(config),
        )

        unsupported_root = root / "unsupported"
        self._write_stack_fixture(
            unsupported_root,
            attrs={
                "response_removed": "yes",
                "physical_quantity": "displacement",
                "lag_storage_direction": "unspecified",
            },
        )
        unsupported = self.mod.audit_input_inventory_and_lineage(
            unsupported_root,
            component="ZZ",
        )
        self.assertEqual(unsupported["lineage_status"], "unknown")
        self.assertIsNone(
            unsupported["instrument_response"]["removed"]
        )
        self.assertEqual(
            unsupported["stack_quantity"]["physical_quantity"],
            "unknown",
        )
        self.assertEqual(
            unsupported["lag_storage"]["direction"],
            "unknown",
        )

    def test_inventory_distributions_include_invalid_length_component(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self._write_stack_fixture(
            root,
            relative_path="AA/BB/stack_pws.h5",
        )
        self._write_stack_fixture(
            root,
            relative_path="CC/DD/stack_pws.h5",
            data=np.ones(4),
        )

        audit = self.mod.audit_input_inventory_and_lineage(
            root,
            component="ZZ",
        )

        self.assertEqual(audit["stack_file_count"], 2)
        self.assertEqual(audit["valid_component_count"], 1)
        self.assertEqual(audit["input_failure_counts"], {"unexpected_length": 1})
        self.assertEqual(sum(audit["dt_distribution"].values()), 2)
        self.assertEqual(sum(audit["maxlag_distribution"].values()), 2)
        self.assertEqual(sum(audit["sample_count_distribution"].values()), 2)
        self.assertEqual(
            audit["sample_count_distribution"],
            {"5": 1, "4": 1},
        )

    def test_unexpected_pair_exception_preserves_type_and_message(self):
        with mock.patch.object(
            self.mod,
            "read_stack_trace",
            side_effect=RuntimeError("synthetic unexpected failure"),
        ):
            result = self.mod.process_one_pair(
                (
                    "unused.h5",
                    "AA",
                    "BB",
                    -122.0,
                    46.0,
                    -121.9,
                    46.0,
                    "ZZ",
                )
            )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["failure_kind"],
            "unexpected_pair_exception",
        )
        self.assertEqual(result["exception_type"], "RuntimeError")
        self.assertEqual(result["reason"], "synthetic unexpected failure")
        self.assertEqual(result["failure_stage"], "strict_hdf5")

    def test_process_one_pair_uses_task_component_and_keeps_period_states_independent(
        self,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = self._write_stack_fixture(
            temporary.name,
            component="TT",
        )
        common = {
            "rejected_continuous_nominal_periods_s": np.array([]),
            "rejected_continuous_instantaneous_periods_s": np.array([]),
            "continuous_rejection_statuses": (),
            "group_time_s": 4.0,
            "group_velocity_km_s": 2.5,
            "leading_snr": 8.0,
            "trailing_snr": 9.0,
            "anchored_raw_phase_time_s": 4.1,
            "signal_peak": 2.0,
            "leading_noise_rms": 0.2,
            "trailing_noise_rms": 0.2,
            "ridge_normalized_log_energy": 0.8,
            "ridge_normalized_envelope_amplitude": 0.9,
            "ridge_adjacent_jump_km_s": 0.02,
            "support_count": 3,
            "interpolation_method": "pchip_t0",
        }
        target_rows = tuple(
            SimpleNamespace(
                target_period_s=period_s,
                accepted=accepted,
                status="accepted" if accepted else rejection,
                **common,
            )
            for period_s, accepted, rejection in (
                (3.0, True, ""),
                (3.5, False, "snr_threshold_failed"),
                (4.0, True, ""),
                (5.0, False, "group_velocity_too_high"),
            )
        )
        curve = SimpleNamespace(
            measurements=(),
            periods_s=np.array([]),
            measurement_statuses=(),
            instantaneous_periods_s=np.array([]),
            ridge_normalized_log_energy=np.array([]),
            ridge_normalized_envelope_amplitude=np.array([]),
            ridge_adjacent_jump_km_s=np.array([]),
            measurement_valid=np.array([], dtype=bool),
        )

        def staged_curve(*_args, **kwargs):
            callback = kwargs["stage_callback"]
            for stage in (
                "filter_bank",
                "dp_ridge",
                "group_arrival_phase_instantaneous_frequency",
                "phase_unwrap",
            ):
                callback(stage)
            return curve

        with (
            mock.patch.object(
                self.mod,
                "measure_phase_curve",
                side_effect=staged_curve,
            ) as measure,
            mock.patch.object(
                self.mod,
                "resample_wang_measurements",
                return_value=target_rows,
            ),
            mock.patch.object(
                self.mod,
                "build_reference_observations_from_task5_curve",
                return_value=[],
            ),
            mock.patch.object(
                self.mod,
                "continuous_curve_audit_rows",
                return_value=[],
            ),
        ):
            result = self.mod.process_one_pair(
                (
                    str(path),
                    "AA",
                    "BB",
                    -122.0,
                    46.0,
                    -121.9,
                    46.0,
                    "TT",
                    {
                        "phase_convention": (
                            "LIN_NEGATIVE_DERIVATIVE_EGF"
                        ),
                        "alpha": 16.0,
                        "beta1": 2.0,
                        "beta2": 4.0,
                    },
                )
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            [row["period_s"] for row in result["measurements"]],
            [3.0, 4.0],
        )
        self.assertEqual(
            result["target_statuses"],
            {
                "3": "accepted",
                "3.5": "snr_threshold_failed",
                "4": "accepted",
                "5": "group_velocity_too_high",
            },
        )
        self.assertEqual(
            [row["target_period_s"] for row in result["rejections"]],
            [3.5, 5.0],
        )
        self.assertEqual(
            result["completed_stages"],
            (
                "strict_hdf5",
                "symmetric_component",
                "filter_bank",
                "dp_ridge",
                "group_arrival_phase_instantaneous_frequency",
                "phase_unwrap",
                "continuous_left_qc",
                "target_period_resampling",
            ),
        )
        trace = measure.call_args.args[0]
        np.testing.assert_array_equal(
            trace.symmetric_waveform,
            [1.0, 5.0, 7.0],
        )
        self.assertEqual(result["input_diagnostics"]["component"], "TT")
        self.assertEqual(
            measure.call_args.kwargs["convention"],
            self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF,
        )
        self.assertEqual(measure.call_args.kwargs["alpha"], 16.0)
        self.assertEqual(measure.call_args.kwargs["beta1"], 2.0)
        self.assertEqual(measure.call_args.kwargs["beta2"], 4.0)
        self.assertGreaterEqual(
            result["input_diagnostics"]["branch_mismatch"],
            0.0,
        )

    def test_preliminary_snr_uses_fixed_candidate_independent_definition(self):
        trace = self.mod.StackTrace(
            pair_name="AA__BB",
            dt_s=0.04,
            maxlag_s=0.12,
            time_positive_s=np.asarray([0.0, 0.04, 0.08, 0.12]),
            positive_lag=np.asarray([1.0, 2.0, 3.0, 4.0]),
            negative_lag_reversed=np.asarray([1.0, 2.0, 3.0, 4.0]),
            symmetric=np.asarray([1.0, 2.0, 3.0, 4.0]),
            branch_mismatch=0.0,
        )
        filtered = np.asarray([[4.0, 3.0, 2.0, 1.0]])
        snr = SimpleNamespace(
            status="accepted",
            leading_snr=7.0,
            trailing_snr=5.0,
        )
        with (
            mock.patch.object(
                self.mod,
                "gaussian_filter_bank",
                return_value=SimpleNamespace(
                    filtered_waveforms=filtered,
                ),
            ) as filter_bank,
            mock.patch.object(
                self.mod,
                "compute_wang_snr",
                return_value=snr,
            ) as compute,
        ):
            value = self.mod.compute_preliminary_snr(
                trace,
                distance_km=20.0,
            )

        self.assertEqual(value, 5.0)
        np.testing.assert_array_equal(
            filter_bank.call_args.kwargs["periods_s"],
            [3.5],
        )
        self.assertEqual(filter_bank.call_args.kwargs["alpha"], 12.0)
        self.assertEqual(compute.call_args.kwargs["period_s"], 3.5)
        np.testing.assert_array_equal(
            compute.call_args.kwargs["filtered_waveform"],
            filtered[0],
        )

    def test_preliminary_snr_row_survives_later_ridge_failure(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = self._write_stack_fixture(temporary.name)
        with (
            mock.patch.object(
                self.mod,
                "compute_preliminary_snr",
                return_value=9.0,
            ),
            mock.patch.object(
                self.mod,
                "measure_phase_curve",
                side_effect=lambda *_args, **kwargs: (
                    kwargs["stage_callback"]("filter_bank"),
                    kwargs["stage_callback"]("dp_ridge"),
                    None,
                )[-1],
            ),
        ):
            result = self.mod.process_one_pair(
                (
                    str(path),
                    "AA",
                    "BB",
                    -122.0,
                    46.0,
                    -121.9,
                    46.0,
                    "ZZ",
                )
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "no_fundamental_ridge")
        self.assertEqual(result["failure_stage"], "dp_ridge")
        self.assertEqual(
            result["preliminary_snr_row"]["preliminary_snr"],
            9.0,
        )
        self.assertEqual(
            result["completed_stages"],
            (
                "strict_hdf5",
                "symmetric_component",
                "filter_bank",
                "dp_ridge",
            ),
        )

    def test_internal_unwrap_exception_is_attributed_to_unwrap_stage(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = self._write_stack_fixture(temporary.name)

        def fail_during_unwrap(*_args, **kwargs):
            callback = kwargs["stage_callback"]
            callback("filter_bank")
            callback("dp_ridge")
            callback("group_arrival_phase_instantaneous_frequency")
            raise RuntimeError("synthetic unwrap failure")

        with (
            mock.patch.object(
                self.mod,
                "compute_preliminary_snr",
                return_value=9.0,
            ),
            mock.patch.object(
                self.mod,
                "measure_phase_curve",
                side_effect=fail_during_unwrap,
            ),
        ):
            result = self.mod.process_one_pair(
                (
                    str(path),
                    "AA",
                    "BB",
                    -122.0,
                    46.0,
                    -121.9,
                    46.0,
                    "ZZ",
                )
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_stage"], "phase_unwrap")
        self.assertEqual(result["exception_type"], "RuntimeError")
        self.assertEqual(
            result["completed_stages"][-1],
            "group_arrival_phase_instantaneous_frequency",
        )

    def test_pick_continuous_ftan_group_curve_prefers_ridge_over_isolated_outlier(self):
        periods_s = np.arange(2.5, 5.6, 0.5, dtype=float)
        velocity_axis = np.linspace(1.8, 3.3, 11, dtype=float)
        ridge_rows = np.array([2, 3, 3, 4, 4, 5, 5], dtype=int)
        image = np.full((velocity_axis.size, periods_s.size), 0.02, dtype=float)

        for col, row in enumerate(ridge_rows):
            image[row, col] = 1.0
            if row > 0:
                image[row - 1, col] = 0.55
            if row + 1 < image.shape[0]:
                image[row + 1, col] = 0.55

        outlier_col = 3
        outlier_row = 9
        image[outlier_row, outlier_col] = 1.8

        curve = self.mod.pick_continuous_ftan_group_curve(
            group_velocity_image=image,
            velocity_axis_km_s=velocity_axis,
            periods_s=periods_s,
        )

        chosen_rows = np.array(
            [int(np.argmin(np.abs(velocity_axis - vel))) for vel in curve],
            dtype=int,
        )
        np.testing.assert_array_equal(chosen_rows, ridge_rows)
        self.assertNotEqual(chosen_rows[outlier_col], outlier_row)

    def test_compatibility_picker_transposes_velocity_period_image_into_unified_kernel(self):
        periods_s = np.arange(2.5, 3.0, 0.05)
        velocity_axis = 1.6 + 0.01 * np.arange(31, dtype=float)
        ridge_rows = np.arange(periods_s.size, dtype=int) + 5
        image = np.zeros((velocity_axis.size, periods_s.size), dtype=float)
        image[ridge_rows, np.arange(periods_s.size)] = 1.0

        curve = self.mod.pick_continuous_ftan_group_curve(
            group_velocity_image=image,
            velocity_axis_km_s=velocity_axis,
            periods_s=periods_s,
        )

        np.testing.assert_allclose(curve, velocity_axis[ridge_rows])

    def test_compatibility_picker_does_not_call_legacy_ftan_autosearch(self):
        periods_s = np.arange(2.5, 3.0, 0.05)
        velocity_axis = 1.6 + 0.01 * np.arange(31, dtype=float)
        image = np.zeros((velocity_axis.size, periods_s.size), dtype=float)
        image[12, :] = 1.0
        original = self.mod.ftan_autosearch

        def forbidden(*_args, **_kwargs):
            raise AssertionError("legacy ftan_autosearch must not run")

        self.mod.ftan_autosearch = forbidden
        try:
            curve = self.mod.pick_continuous_ftan_group_curve(
                group_velocity_image=image,
                velocity_axis_km_s=velocity_axis,
                periods_s=periods_s,
            )
        finally:
            self.mod.ftan_autosearch = original

        np.testing.assert_allclose(curve, velocity_axis[12])
        self.assertNotIn(
            "ftan_autosearch(",
            inspect.getsource(self.mod.pick_continuous_ftan_group_curve),
        )
        self.assertNotIn(
            "ftan_autosearch(",
            inspect.getsource(self.mod.process_one_pair),
        )

    def test_build_measurements_skips_target_period_with_invalid_ridge_amplitude(self):
        periods_s = self.mod.ftan_period_grid()
        velocity_axis = 1.6 + 0.01 * np.arange(341, dtype=float)
        ridge_row = 120
        image = np.full(
            (velocity_axis.size, periods_s.size),
            0.01,
            dtype=float,
        )
        image[ridge_row, :] = 1.0
        invalid_index = int(np.argmin(np.abs(periods_s - 4.0)))
        image[ridge_row, invalid_index] = 0.10
        image[300, invalid_index] = 1.0
        payload = {
            "periods_s": periods_s,
            "velocity_axis_km_s": velocity_axis,
            "group_velocity_image": image,
        }
        original = self.mod.compute_ftan_group_velocity_image
        self.mod.compute_ftan_group_velocity_image = lambda *_args, **_kwargs: payload
        try:
            measurements = self.mod.build_ftan_group_measurements(
                np.ones(128, dtype=float),
                dt_s=0.04,
                distance_km=20.0,
            )
        finally:
            self.mod.compute_ftan_group_velocity_image = original

        self.assertIsNotNone(measurements)
        self.assertIn(3.0, measurements)
        self.assertNotIn(4.0, measurements)
        self.assertIn(5.0, measurements)

    def test_energy_cache_key_separates_phase_convention_but_not_beta(self):
        key_builder = self.mod.ftan_energy_cache_key
        signature = inspect.signature(key_builder)
        self.assertNotIn("beta1", signature.parameters)
        self.assertNotIn("beta2", signature.parameters)

        bensen = key_builder(
            pair_waveform_hash="abc123",
            phase_convention="bensen_velocity_ccf",
            alpha=12.0,
        )
        repeated = key_builder(
            pair_waveform_hash="abc123",
            phase_convention="bensen_velocity_ccf",
            alpha=12.0,
        )
        lin = key_builder(
            pair_waveform_hash="abc123",
            phase_convention="lin_displacement_ccf",
            alpha=12.0,
        )

        self.assertEqual(bensen, repeated)
        self.assertNotEqual(bensen, lin)

    def test_energy_cache_key_rejects_non_string_identifiers_and_bad_alpha(self):
        key_builder = self.mod.ftan_energy_cache_key
        for bad_identifier in (None, b"abc123", 123, ["abc123"]):
            with self.subTest(field="pair_waveform_hash", value=bad_identifier):
                with self.assertRaisesRegex(ValueError, "pair_waveform_hash"):
                    key_builder(
                        pair_waveform_hash=bad_identifier,
                        phase_convention="bensen_velocity_ccf",
                        alpha=12.0,
                    )
            with self.subTest(field="phase_convention", value=bad_identifier):
                with self.assertRaisesRegex(ValueError, "phase_convention"):
                    key_builder(
                        pair_waveform_hash="abc123",
                        phase_convention=bad_identifier,
                        alpha=12.0,
                    )
        for bad_alpha in (
            False,
            True,
            np.bool_(True),
            [12.0],
            np.array([12.0]),
            "not-a-number",
            None,
            0.0,
            -1.0,
            np.nan,
            np.inf,
            -np.inf,
        ):
            with self.subTest(field="alpha", value=bad_alpha):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    key_builder(
                        pair_waveform_hash="abc123",
                        phase_convention="bensen_velocity_ccf",
                        alpha=bad_alpha,
                    )

    def test_ftan_envelope_compatibility_wrapper_is_exact_unified_filter_bank(self):
        waveform = np.sin(np.linspace(0.0, 8.0 * np.pi, 256))
        fs = 25.0
        periods_s = np.array([2.5, 3.0, 4.0, 5.0], dtype=float)
        distance_km = 42.0
        alpha = self.mod.gaussian_alpha_for_distance(distance_km)

        expected = self.mod.gaussian_filter_bank(
            waveform,
            dt_s=1.0 / fs,
            periods_s=periods_s,
            alpha=alpha,
        ).envelope
        actual = self.mod.ftan_envelope_image_calculation(
            waveform,
            fs,
            periods_s,
            distance_km,
        )

        np.testing.assert_array_equal(actual, expected)
        source = inspect.getsource(self.mod.ftan_envelope_image_calculation)
        self.assertIn("gaussian_filter_bank", source)
        self.assertNotIn("np.fft", source)
        self.assertNotIn("np.exp", source)

    def test_compatibility_ftan_image_has_no_envelope_mean_snr_formula(self):
        source = inspect.getsource(self.mod.compute_ftan_group_velocity_image)
        self.assertNotIn("envelope_noise", source)
        self.assertNotIn("noise_mean", source)
        self.assertNotIn('"snr_t"', source)

    def test_snap_group_time_to_local_envelope_peak_stays_near_ftan_prediction(self):
        time_s = np.linspace(0.0, 12.0, 1201)
        envelope = (
            0.35 * np.exp(-((time_s - 3.1) / 0.20) ** 2)
            + 1.20 * np.exp(-((time_s - 5.2) / 0.18) ** 2)
            + 0.90 * np.exp(-((time_s - 8.7) / 0.20) ** 2)
        )

        index, snapped_time_s = self.mod.snap_group_time_to_local_envelope_peak(
            time_s,
            envelope,
            predicted_time_s=5.0,
            period_s=3.0,
        )

        self.assertAlmostEqual(snapped_time_s, 5.2, places=1)
        self.assertAlmostEqual(time_s[index], snapped_time_s, places=6)

    def test_wang_snr_compatibility_entry_delegates_to_math_core(self):
        time_s = np.linspace(0.0, 10.0, 1001)
        filtered_waveform = np.ones_like(time_s)
        filtered_waveform[(time_s >= 2.0) & (time_s <= 6.25)] = 10.0
        signature = inspect.signature(self.mod.wang_leading_trailing_snr)
        self.assertIn("filtered_waveform", signature.parameters)

        snr = self.mod.wang_leading_trailing_snr(
            time_s=time_s,
            filtered_waveform=filtered_waveform,
            distance_km=10.0,
            period_s=3.0,
        )

        self.assertAlmostEqual(snr.signal_peak, 10.0, places=7)
        self.assertAlmostEqual(snr.leading_snr, 10.0, places=7)
        self.assertAlmostEqual(snr.trailing_snr, 10.0, places=7)
        source = inspect.getsource(self.mod.wang_leading_trailing_snr)
        self.assertIn("compute_wang_snr(", source)
        self.assertNotIn("np.sqrt", source)
        self.assertNotIn("np.mean", source)

    def test_group_pick_stability_rejects_peak_far_from_ftan_prediction(self):
        self.assertTrue(
            self.mod.group_pick_is_stable(
                predicted_time_s=5.0,
                snapped_time_s=5.35,
                period_s=3.0,
                max_fraction_period=0.25,
            )
        )
        self.assertFalse(
            self.mod.group_pick_is_stable(
                predicted_time_s=5.0,
                snapped_time_s=6.1,
                period_s=3.0,
                max_fraction_period=0.25,
            )
        )

    def test_no_preleft_nearfield_screen_or_075_threshold_exists(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertFalse(
            hasattr(self.mod, "passes_preleft_nearfield_screen")
        )
        self.assertNotIn("PRELEFT_MIN_WAVELENGTHS", source)
        self.assertNotIn("0.75", source)

    def test_ftan_period_grid_is_exact_formal_config_grid(self):
        expected = self.mod.FtanConfig().periods_s
        actual = self.mod.ftan_period_grid()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[0], 2.5)
        self.assertEqual(actual[-1], 5.0)
        np.testing.assert_allclose(np.diff(actual), 0.05)

    def test_formal_runner_uses_core_target_resampling_without_preleft_screen(self):
        self.assertEqual(self.mod.TARGET_PERIODS_S, (3.0, 3.5, 4.0, 5.0))
        source = inspect.getsource(self.mod.process_one_pair)
        self.assertIn("measure_phase_curve(", source)
        self.assertIn("resample_wang_measurements(", source)
        self.assertIn("valid_mask=curve.measurement_valid", source)
        self.assertIn("periods_s=FtanConfig().periods_s", source)
        self.assertNotIn("periods_s=ftan_period_grid()", source)
        self.assertNotIn("passes_preleft_nearfield_screen(", source)
        self.assertNotIn("PRELEFT_MIN_WAVELENGTHS", source)
        self.assertNotIn("wang_leading_trailing_snr(", source)

    def test_runner_fits_reference_from_task5_continuous_left_rows_only(self):
        process_source = inspect.getsource(self.mod.process_one_pair)
        self.assertIn('"continuous_observations"', process_source)
        self.assertIn("curve.measurements", process_source)
        self.assertIn("curve.instantaneous_periods_s", process_source)
        self.assertIn("curve.measurement_valid", process_source)
        self.assertNotIn("DisperPicker", process_source)
        helper_source = inspect.getsource(
            self.mod.build_reference_observations_from_task5_curve
        )
        self.assertIn("evaluate_wang_left_qc(", helper_source)
        self.assertIn("compute_wang_snr(", helper_source)
        self.assertIn("rejected_continuous_nominal_periods_s", helper_source)

        main_source = inspect.getsource(self.mod.main)
        self.assertIn("fit_reference_dispersion(reference_observations)", main_source)
        self.assertIn('"reference_fit_status"', main_source)
        self.assertIn(
            "resolve_reference_cycles(",
            main_source,
        )
        self.assertIn(
            '"continuous_reference_cycles.csv"',
            main_source,
        )
        self.assertIn(
            '"reference_alias_solutions.csv"',
            main_source,
        )
        payload_source = inspect.getsource(self.mod.build_period_payload)
        self.assertIn("resolve_reference_cycles(", payload_source)
        self.assertNotIn("estimate_reference_velocity(", payload_source)
        self.assertNotIn("correct_phase_ambiguity(", payload_source)
        module_source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("DEFAULT_REFERENCE_VELOCITIES", module_source)
        self.assertNotIn("def estimate_reference_velocity(", module_source)
        self.assertNotIn("def correct_phase_ambiguity(", module_source)

    def test_target_cycle_audit_precedes_one_wavelength_filter(self):
        rows = [
            self.mod.PhaseMeasurement(
                pair_name="SHORT",
                distance_km=5.0,
                period_s=3.0,
                group_time_s=2.0,
                group_velocity_km_s=2.5,
                leading_snr=8.0,
                trailing_snr=8.0,
                phi_tu_rad=0.2,
                raw_travel_time_s=2.0,
            ),
            self.mod.PhaseMeasurement(
                pair_name="LONG",
                distance_km=20.0,
                period_s=3.0,
                group_time_s=8.0,
                group_velocity_km_s=2.5,
                leading_snr=8.0,
                trailing_snr=8.0,
                phi_tu_rad=0.2,
                raw_travel_time_s=8.0,
            ),
        ]
        periods = np.round(np.linspace(2.5, 5.0, 51), 12)
        payload = self.mod.build_period_payload(
            rows,
            SimpleNamespace(
                status="accepted",
                periods_s=periods,
                phase_slowness_s_km=np.full(51, 0.4),
                phase_velocities_km_s=np.full(51, 2.5),
            ),
        )
        self.assertEqual(len(payload["left_rows"]), 2)
        self.assertEqual(len(payload.get("corrected_rows", [])), 2)
        self.assertEqual(len(payload["right_rows"]), 1)
        self.assertEqual(payload["corrected_rows"][0]["pair_name"], "SHORT")
        self.assertFalse(
            payload["corrected_rows"][0]["passes_one_wavelength"]
        )
        self.assertFalse(payload["corrected_rows"][0]["right_column"])
        self.assertEqual(
            payload["corrected_rows"][0]["right_qc_status"],
            "fails_one_wavelength",
        )
        self.assertEqual(payload["right_rows"][0]["pair_name"], "LONG")
        self.assertTrue(payload["right_rows"][0]["right_column"])
        self.assertEqual(
            payload["right_rows"][0]["right_qc_status"],
            "accepted",
        )
        for field in (
            "N",
            "cycle_count",
            "branch_tie",
            "reference_time_s",
            "reference_slowness_s_km",
            "corrected_time_s",
            "corrected_residual_s",
            "cycle_period_s",
        ):
            self.assertIn(field, payload["corrected_rows"][0])

    def test_right_column_accepts_exact_wavelength_and_half_period_residual(self):
        period_s = 3.0
        reference_velocity = 2.5
        boundary_distance = reference_velocity * period_s
        reference_time = boundary_distance / reference_velocity
        row = self.mod.PhaseMeasurement(
            pair_name="BOUNDARY",
            distance_km=boundary_distance,
            period_s=period_s,
            group_time_s=reference_time,
            group_velocity_km_s=reference_velocity,
            leading_snr=8.0,
            trailing_snr=8.0,
            phi_tu_rad=0.2,
            raw_travel_time_s=reference_time + period_s / 2.0,
        )
        periods = np.round(np.linspace(2.5, 5.0, 51), 12)

        payload = self.mod.build_period_payload(
            [row],
            SimpleNamespace(
                status="accepted",
                periods_s=periods,
                phase_slowness_s_km=np.full(51, 0.4),
                phase_velocities_km_s=np.full(51, reference_velocity),
            ),
        )

        self.assertEqual(len(payload["right_rows"]), 1)
        accepted = payload["right_rows"][0]
        self.assertTrue(accepted["passes_one_wavelength"])
        self.assertTrue(accepted["right_column"])
        self.assertTrue(accepted["branch_tie"])
        self.assertAlmostEqual(
            abs(accepted["corrected_residual_s"]),
            period_s / 2.0,
        )

    def test_target_cycle_resolution_honors_frozen_lin_convention(self):
        period_s = 3.0
        reference_velocity = 2.5
        distance_km = 20.0
        reference_time_s = distance_km / reference_velocity
        row = self.mod.PhaseMeasurement(
            pair_name="LIN_PAIR",
            distance_km=distance_km,
            period_s=period_s,
            group_time_s=reference_time_s,
            group_velocity_km_s=reference_velocity,
            leading_snr=8.0,
            trailing_snr=8.0,
            phi_tu_rad=0.2,
            raw_travel_time_s=reference_time_s + period_s,
        )
        periods = np.round(np.linspace(2.5, 5.0, 51), 12)
        payload = self.mod.build_period_payload(
            [row],
            SimpleNamespace(
                status="accepted",
                periods_s=periods,
                phase_slowness_s_km=np.full(51, 0.4),
                phase_velocities_km_s=np.full(51, reference_velocity),
            ),
            convention=self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF,
        )
        corrected = payload["corrected_rows"][0]
        self.assertEqual(corrected["cycle_count"], 1)
        self.assertAlmostEqual(corrected["corrected_time_s"], reference_time_s)

    def test_target_periods_can_retain_different_right_column_counts(self):
        reference_velocity = 2.5
        reference_periods = np.round(np.linspace(2.5, 5.0, 51), 12)
        reference_fit = SimpleNamespace(
            status="accepted",
            periods_s=reference_periods,
            phase_slowness_s_km=np.full(51, 1.0 / reference_velocity),
            phase_velocities_km_s=np.full(51, reference_velocity),
        )
        distances = (8.0, 9.0, 11.0, 13.0)
        counts = []
        for period_s in self.mod.TARGET_PERIODS_S:
            rows = [
                self.mod.PhaseMeasurement(
                    pair_name=f"P{period_s:.1f}_{index}",
                    distance_km=distance,
                    period_s=period_s,
                    group_time_s=distance / reference_velocity,
                    group_velocity_km_s=reference_velocity,
                    leading_snr=8.0,
                    trailing_snr=8.0,
                    phi_tu_rad=0.2,
                    raw_travel_time_s=distance / reference_velocity,
                )
                for index, distance in enumerate(distances)
            ]
            payload = self.mod.build_period_payload(rows, reference_fit)
            counts.append(len(payload["right_rows"]))
            self.assertEqual(
                len(payload["corrected_rows"]),
                len(distances),
            )

        self.assertEqual(counts, [4, 3, 2, 1])

    def test_reference_observations_require_task5_left_qc_and_support_acceptance(self):
        rows = (
            SimpleNamespace(
                filtered_waveform=np.ones(20),
                group_time_s=10.0,
                raw_phase_time_s=9.0,
                group_velocity_km_s=2.0,
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            ),
            SimpleNamespace(
                filtered_waveform=np.ones(20),
                group_time_s=10.0,
                raw_phase_time_s=9.5,
                group_velocity_km_s=2.0,
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            ),
        )
        curve = SimpleNamespace(
            measurements=rows,
            measurement_valid=np.array([True, True]),
            periods_s=np.array([2.9, 3.0]),
            instantaneous_periods_s=np.array([2.95, 3.05]),
        )
        target_rows = (
            SimpleNamespace(
                rejected_continuous_nominal_periods_s=np.array([3.0]),
            ),
        )
        accepted_qc = SimpleNamespace(accepted=True, status="accepted")
        with (
            mock.patch.object(
                self.mod,
                "compute_wang_snr",
                return_value=SimpleNamespace(
                    signal_peak=8.0,
                    leading_noise_rms=1.0,
                    trailing_noise_rms=1.0,
                    leading_snr=8.0,
                    trailing_snr=8.0,
                ),
            ) as compute,
            mock.patch.object(
                self.mod,
                "evaluate_wang_left_qc",
                return_value=accepted_qc,
            ) as evaluate,
        ):
            result = self.mod.build_reference_observations_from_task5_curve(
                pair_name="AA__BB",
                curve=curve,
                target_rows=target_rows,
                time_s=np.arange(20, dtype=float) + 1.0,
                distance_km=20.0,
                azimuth_deg=90.0,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["instantaneous_period_s"], 2.95)
        self.assertEqual(result[0]["anchored_raw_time_s"], 9.0)
        self.assertEqual(compute.call_count, 2)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(
            [call.kwargs["period_s"] for call in compute.call_args_list],
            [2.95, 3.05],
        )

        with (
            mock.patch.object(
                self.mod,
                "compute_wang_snr",
                return_value=SimpleNamespace(
                    signal_peak=8.0,
                    leading_noise_rms=1.0,
                    trailing_noise_rms=1.0,
                    leading_snr=8.0,
                    trailing_snr=8.0,
                ),
            ),
            mock.patch.object(
                self.mod,
                "evaluate_wang_left_qc",
                side_effect=(
                    SimpleNamespace(accepted=False, status="snr_threshold_failed"),
                    accepted_qc,
                ),
            ),
        ):
            rejected = self.mod.build_reference_observations_from_task5_curve(
                pair_name="AA__BB",
                curve=curve,
                target_rows=target_rows,
                time_s=np.arange(20, dtype=float) + 1.0,
                distance_km=20.0,
                azimuth_deg=90.0,
            )
        self.assertEqual(rejected, [])

    def test_stage_b_continuous_left_rows_have_the_frozen_hash_schema(self):
        measurement = SimpleNamespace(
            filtered_waveform=np.ones(40),
            group_time_s=10.0,
            raw_phase_time_s=9.0,
            group_velocity_km_s=2.0,
            convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
        )
        curve = SimpleNamespace(
            measurements=(measurement,),
            measurement_valid=np.array([True]),
            periods_s=np.array([3.0]),
            instantaneous_periods_s=np.array([3.05]),
            ridge_normalized_log_energy=np.array([0.8]),
            ridge_normalized_envelope_amplitude=np.array([0.7]),
            ridge_adjacent_jump_km_s=np.array([0.02]),
            selected_ridge=SimpleNamespace(
                quality=SimpleNamespace(
                    coverage=0.95,
                    max_gap=1,
                    jump_fraction=0.02,
                    boundary_fraction=0.0,
                    normalized_energy_integral=20.0,
                )
            ),
        )
        snr = SimpleNamespace(
            signal_peak=10.0,
            leading_noise_rms=1.0,
            trailing_noise_rms=1.25,
            leading_snr=10.0,
            trailing_snr=8.0,
            status="accepted",
        )
        with mock.patch.object(
            self.mod,
            "compute_wang_snr",
            return_value=snr,
        ), mock.patch.object(
            self.mod,
            "evaluate_wang_left_qc",
            return_value=SimpleNamespace(accepted=True, status="accepted"),
        ):
            rows = self.mod.build_reference_observations_from_task5_curve(
                pair_name="AA__BB",
                curve=curve,
                target_rows=(),
                time_s=np.arange(40, dtype=float),
                distance_km=20.0,
                azimuth_deg=90.0,
            )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for field in (
            "pair_name",
            "T_inst",
            "t0",
            "U",
            "signal_peak",
            "leading_rms",
            "trailing_rms",
            "ridge_fields",
            "ridge_valid",
            "instantaneous_period_valid",
        ):
            self.assertIn(field, row)
        self.assertEqual(row["T_inst"], 3.05)
        self.assertEqual(row["t0"], 9.0)
        self.assertEqual(row["U"], 2.0)
        self.assertEqual(row["leading_snr"], 10.0)
        self.assertEqual(row["trailing_snr"], 8.0)
        self.assertEqual(row["ridge_fields"]["nominal_period_s"], 3.0)
        self.assertEqual(
            len(self.mod.wang_ftan_validation.hash_left_observation_table(rows)),
            64,
        )

    def test_reference_observations_reject_duplicate_instantaneous_periods_before_target_resampling(
        self,
    ):
        measurements = tuple(
            SimpleNamespace(
                filtered_waveform=np.ones(20),
                group_time_s=10.0,
                raw_phase_time_s=9.0 + index,
                group_velocity_km_s=2.0,
                convention=self.mod.PhaseConvention.BENSEN_VELOCITY_CCF,
            )
            for index in range(2)
        )
        curve = SimpleNamespace(
            measurements=measurements,
            measurement_valid=np.array([True, True]),
            periods_s=np.array([2.9, 3.0]),
            instantaneous_periods_s=np.array([3.05, 3.05]),
        )

        result = self.mod.build_reference_observations_from_task5_curve(
            pair_name="AA__BB",
            curve=curve,
            target_rows=(),
            time_s=np.arange(20, dtype=float) + 1.0,
            distance_km=20.0,
            azimuth_deg=90.0,
        )

        self.assertEqual(result, [])

    def test_rejection_rows_include_every_continuous_and_failed_target_status(self):
        common = {
            "rejected_continuous_nominal_periods_s": np.array([2.8, 3.1]),
            "rejected_continuous_instantaneous_periods_s": np.array(
                [np.nan, 3.05]
            ),
            "continuous_rejection_statuses": (
                "invalid_instantaneous_frequency",
                "group_velocity_out_of_range",
            ),
        }
        targets = (
            SimpleNamespace(
                target_period_s=3.0,
                accepted=True,
                status="accepted",
                **common,
            ),
            SimpleNamespace(
                target_period_s=4.0,
                accepted=False,
                status="target_period_not_bracketed",
                **common,
            ),
        )

        rows = self.mod.wang_rejection_rows("AA__BB", targets)

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["stage"] for row in rows],
            [
                "continuous_observation",
                "continuous_observation",
                "target_period",
            ],
        )
        self.assertEqual(
            [row["reason"] for row in rows],
            [
                "invalid_instantaneous_frequency",
                "group_velocity_out_of_range",
                "target_period_not_bracketed",
            ],
        )
        self.assertEqual(rows[0]["nominal_period_s"], 2.8)
        self.assertIsNone(rows[0]["instantaneous_period_s"])
        self.assertEqual(rows[1]["instantaneous_period_s"], 3.05)
        self.assertEqual(rows[2]["target_period_s"], 4.0)
        for row in rows:
            self.assertEqual(
                set(row),
                {
                    "pair_name",
                    "stage",
                    "reason",
                    "failure_kind",
                    "nominal_period_s",
                    "instantaneous_period_s",
                    "target_period_s",
                },
            )

        process_source = inspect.getsource(self.mod.process_one_pair)
        self.assertIn('"rejections": wang_rejection_rows(', process_source)
        main_source = inspect.getsource(self.mod.main)
        self.assertIn(
            'failures.extend(result.get("rejections", []))',
            main_source,
        )

    def _run_main_fixture(
        self,
        results,
        reference_status="accepted",
        reference_error=None,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output_dir = Path(temporary.name)
        stale_figure = (
            output_dir
            / "figures"
            / "wang_figure4_reproduction.png"
        )
        stale_figure.parent.mkdir(parents=True, exist_ok=True)
        stale_figure.write_bytes(b"stale success figure")
        (output_dir / "report.html").write_text(
            "stale success report",
            encoding="utf-8",
        )
        (output_dir / "continuous_reference_cycles.csv").write_text(
            "stale continuous cycles",
            encoding="utf-8",
        )
        for name in (
            "measurements_initial_qc.csv",
            "measurements_corrected.csv",
            "measurements_right_qc.csv",
            "fit_summary.csv",
            "reference_alias_solutions.csv",
            "reference_cv_audit.csv",
        ):
            (output_dir / name).write_text(
                f"stale {name}",
                encoding="utf-8",
            )
        (output_dir / "unrelated-user-note.txt").write_text(
            "must survive runner cleanup",
            encoding="utf-8",
        )
        args = SimpleNamespace(
            stations_csv=output_dir / "stations.csv",
            stack_root=output_dir / "stacks",
            output_dir=output_dir,
            bbox=None,
            bbox_mode="both",
            limit_pairs=None,
            max_workers=1,
            chunksize=1,
        )

        class FakePool:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def imap_unordered(self, _function, _tasks, chunksize):
                self.chunksize = chunksize
                return iter(results)

        tasks = [("unused",) * 7 for _ in results]
        reference_periods = np.arange(2.5, 5.05, 0.05)
        fold_ids = np.arange(5, dtype=int)
        fold_assignment = SimpleNamespace(
            assignment_hash="b" * 64,
            fold_ids=fold_ids,
            distance_quintile_ids=fold_ids,
            azimuth_block_ids=fold_ids,
            training_indices=tuple(
                np.flatnonzero(fold_ids != fold) for fold in range(5)
            ),
            holdout_indices=tuple(
                np.flatnonzero(fold_ids == fold) for fold in range(5)
            ),
        )
        cv_config = SimpleNamespace(
            lambda_s=0.0,
            lambda_g=0.0,
            fold_holdout_losses=np.arange(5, dtype=float),
            mean_holdout_loss=2.0,
            optimizer_calls=25,
        )
        cv_result = SimpleNamespace(
            fold_assignment=fold_assignment,
            configs=(cv_config,),
            selected=cv_config,
            optimizer_calls=625,
            result_hash="c" * 64,
        )
        reference_fit = SimpleNamespace(
            status=reference_status,
            periods_s=reference_periods,
            phase_slowness_s_km=np.full(51, 0.4),
            phase_velocities_km_s=np.full(51, 2.5),
            result_hash="a" * 64,
            cv_optimizer_calls=625,
            final_optimizer_calls=71,
            optimizer_calls=696,
            cv_result=cv_result,
            lambda_s=0.0,
            lambda_g=0.0,
        )
        with (
            mock.patch.object(self.mod, "parse_args", return_value=args),
            mock.patch.object(self.mod, "load_station_coords", return_value={}),
            mock.patch.object(
                self.mod,
                "iter_stack_tasks",
                return_value=iter(tasks),
            ),
            mock.patch.object(self.mod, "Pool", FakePool),
            mock.patch.object(self.mod, "plot_figure") as plot,
            mock.patch.object(
                self.mod,
                "fit_reference_dispersion",
                return_value=reference_fit,
                side_effect=reference_error,
            ),
        ):
            return_code = self.mod.main([])
        metadata_path = output_dir / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else None
        )
        return return_code, output_dir, metadata, plot

    def test_reference_fit_failure_cleans_stale_success_and_writes_diagnostics(self):
        measurement = {
            "pair_name": "AA__BB",
            "distance_km": 20.0,
            "period_s": 3.0,
            "group_time_s": 8.0,
            "group_velocity_km_s": 2.5,
            "leading_snr": 8.0,
            "trailing_snr": 8.0,
            "phi_tu_rad": 0.2,
            "raw_travel_time_s": 7.5,
        }
        continuous = [
            {
                "pair_name": f"AA__B{index}",
                "distance_km": 20.0 + index,
                "azimuth_deg": 45.0 * index,
                "instantaneous_period_s": 3.0,
                "anchored_raw_time_s": (20.0 + index) / 2.5,
                "group_slowness_s_km": 0.4,
                "convention": "BENSEN_VELOCITY_CCF",
            }
            for index in range(5)
        ]
        return_code, output_dir, metadata, plot = self._run_main_fixture(
            [
                {
                    "pair_name": "AA__BB",
                    "ok": True,
                    "measurements": [measurement],
                    "continuous_observations": continuous,
                    "rejections": [],
                }
            ],
            reference_status="reference_alias_unresolved",
        )
        self.assertEqual(return_code, 2)
        self.assertEqual(
            metadata["terminal_failure_reason"],
            "reference_alias_unresolved",
        )
        self.assertEqual(metadata.get("reference_cv_optimizer_calls"), 625)
        self.assertEqual(metadata.get("reference_final_optimizer_calls"), 71)
        self.assertEqual(metadata["reference_optimizer_calls"], 696)
        common_expected = {
            "stack_root": str(output_dir / "stacks"),
            "stations_csv": str(output_dir / "stations.csv"),
            "target_periods_s": list(self.mod.TARGET_PERIODS_S),
            "expected_scientific_rejection_count": 1,
            "unexpected_pair_exception_count": 0,
        }
        self.assertLessEqual(set(common_expected), set(metadata))
        for key, expected in common_expected.items():
            self.assertEqual(metadata[key], expected)
        failure_row_count = max(
            0,
            len(
                (output_dir / "failures.csv")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            - 1,
        )
        self.assertEqual(metadata["failure_count"], failure_row_count)
        self.assertEqual(
            metadata["failure_count"],
            metadata["expected_scientific_rejection_count"]
            + metadata["unexpected_pair_exception_count"],
        )
        self.assertFalse((output_dir / "report.html").exists())
        self.assertFalse(
            (
                output_dir
                / "figures"
                / "wang_figure4_reproduction.png"
            ).exists()
        )
        self.assertTrue(
            (output_dir / "reference_alias_solutions.csv").exists()
        )
        self.assertTrue((output_dir / "reference_cv_audit.csv").exists())
        self.assertNotIn(
            "stale reference_alias_solutions.csv",
            (output_dir / "reference_alias_solutions.csv").read_text(
                encoding="utf-8"
            ),
        )
        self.assertIn(
            "reference_fit",
            (output_dir / "failures.csv").read_text(encoding="utf-8"),
        )
        plot.assert_not_called()

    def test_main_terminal_gates_publish_failure_metadata_not_success_artifacts(self):
        cases = (
            ("zero_tasks", []),
            (
                "all_pairs_failed",
                [
                    {
                        "pair_name": "AA__BB",
                        "ok": False,
                        "reason": "boom",
                        "failure_kind": "unexpected_pair_exception",
                    }
                ],
            ),
            (
                "no_accepted_measurements",
                [
                    {
                        "pair_name": "AA__BB",
                        "ok": True,
                        "measurements": [],
                        "rejections": [
                            {
                                "pair_name": "AA__BB",
                                "stage": "target_period",
                                "reason": "target_period_not_bracketed",
                                "failure_kind": (
                                    "expected_scientific_rejection"
                                ),
                                "nominal_period_s": None,
                                "instantaneous_period_s": None,
                                "target_period_s": 3.0,
                            }
                        ],
                    }
                ],
            ),
        )
        for expected_reason, results in cases:
            with self.subTest(reason=expected_reason):
                return_code, output_dir, metadata, plot = (
                    self._run_main_fixture(results)
                )
                self.assertNotEqual(return_code, 0)
                self.assertIsNotNone(metadata)
                self.assertEqual(metadata["run_status"], "failed")
                self.assertEqual(
                    metadata["terminal_failure_reason"],
                    expected_reason,
                )
                plot.assert_not_called()
                self.assertFalse(
                    (
                        output_dir
                        / "figures"
                        / "wang_figure4_reproduction.png"
                    ).exists()
                )
                self.assertFalse((output_dir / "report.html").exists())
                self.assertFalse(
                    (output_dir / "reference_alias_solutions.csv").exists()
                )
                self.assertTrue(
                    (output_dir / "unrelated-user-note.txt").exists()
                )
                self.assertEqual(
                    metadata["removed_stale_artifacts"],
                    [
                        "figures/wang_figure4_reproduction.png",
                        "report.html",
                        "measurements_initial_qc.csv",
                        "measurements_corrected.csv",
                        "measurements_right_qc.csv",
                        "continuous_reference_cycles.csv",
                        "fit_summary.csv",
                        "reference_alias_solutions.csv",
                        "reference_cv_audit.csv",
                    ],
                )
                failure_row_count = max(
                    0,
                    len(
                        (output_dir / "failures.csv")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    )
                    - 1,
                )
                self.assertEqual(
                    metadata["failure_count"],
                    failure_row_count,
                )
                self.assertEqual(
                    metadata["failure_count"],
                    metadata["expected_scientific_rejection_count"]
                    + metadata["unexpected_pair_exception_count"],
                )

        _, output_dir, metadata, _ = self._run_main_fixture(cases[1][1])
        failures = (
            output_dir / "failures.csv"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            metadata["unexpected_pair_exception_count"],
            1,
        )
        self.assertIn("unexpected_pair_exception", failures)

    def test_reference_alias_rows_include_complete_start_definition(self):
        velocities = np.linspace(1.6, 4.0, 51)
        start = SimpleNamespace(
            start_index=0,
            velocity_hash="d" * 64,
            kind="sine_perturbation",
            base_velocity_km_s=2.7,
            endpoint_slope_km_s=0.0,
            velocities_km_s=velocities,
        )
        solution = SimpleNamespace(
            converged=True,
            objective=0.2,
            fold_holdout_losses=np.full(5, 0.1),
            holdout_loss=0.1,
            basin_id=0,
            optimizer_message="accepted",
            target_velocities_km_s=np.array([2.6, 2.7, 2.8, 2.9]),
            phase_slowness_s_km=1.0 / velocities,
        )
        rows = self.mod.reference_alias_solution_rows(
            SimpleNamespace(
                representative_indices=(0,),
                starts=(start,),
                local_solutions=(solution,),
            )
        )
        self.assertEqual(rows[0]["start_base_velocity_km_s"], 2.7)
        self.assertEqual(rows[0]["start_endpoint_slope_km_s"], 0.0)
        self.assertEqual(
            json.loads(rows[0]["fold_holdout_losses"]),
            [0.1] * 5,
        )
        self.assertEqual(
            json.loads(rows[0]["start_velocities_km_s"]),
            velocities.tolist(),
        )

    def test_reference_fold_precondition_failure_is_structured_and_cleans(self):
        measurement = {
            "pair_name": "AA__BB",
            "distance_km": 20.0,
            "period_s": 3.0,
            "group_time_s": 8.0,
            "group_velocity_km_s": 2.5,
            "leading_snr": 8.0,
            "trailing_snr": 8.0,
            "phi_tu_rad": 0.2,
            "raw_travel_time_s": 7.5,
        }
        continuous = [
            {
                "pair_name": f"AA__B{index}",
                "distance_km": 20.0,
                "azimuth_deg": 0.0,
                "instantaneous_period_s": 3.0,
                "anchored_raw_time_s": 8.0,
                "group_slowness_s_km": 0.4,
                "convention": "BENSEN_VELOCITY_CCF",
            }
            for index in range(5)
        ]
        try:
            return_code, output_dir, metadata, plot = self._run_main_fixture(
                [
                    {
                        "pair_name": "AA__BB",
                        "ok": True,
                        "measurements": [measurement],
                        "continuous_observations": continuous,
                        "rejections": [],
                    }
                ],
                reference_error=ValueError(
                    "at least five joint distance-azimuth blocks are required"
                ),
            )
        except ValueError as error:
            self.fail(f"reference precondition escaped main: {error}")
        self.assertEqual(return_code, 2)
        self.assertEqual(
            metadata["terminal_failure_reason"],
            "reference_insufficient_fold_blocks",
        )
        self.assertIn(
            "at least five joint distance-azimuth blocks",
            metadata["reference_fit_error"],
        )
        common_expected = {
            "stack_root": str(output_dir / "stacks"),
            "stations_csv": str(output_dir / "stations.csv"),
            "target_periods_s": list(self.mod.TARGET_PERIODS_S),
            "expected_scientific_rejection_count": 1,
            "unexpected_pair_exception_count": 0,
        }
        self.assertLessEqual(set(common_expected), set(metadata))
        for key, expected in common_expected.items():
            self.assertEqual(metadata[key], expected)
        failure_row_count = max(
            0,
            len(
                (output_dir / "failures.csv")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            - 1,
        )
        self.assertEqual(metadata["failure_count"], failure_row_count)
        self.assertEqual(
            metadata["failure_count"],
            metadata["expected_scientific_rejection_count"]
            + metadata["unexpected_pair_exception_count"],
        )
        self.assertFalse((output_dir / "report.html").exists())
        self.assertFalse(
            (output_dir / "reference_alias_solutions.csv").exists()
        )
        self.assertFalse((output_dir / "reference_cv_audit.csv").exists())
        self.assertIn(
            "reference_insufficient_fold_blocks",
            (output_dir / "failures.csv").read_text(encoding="utf-8"),
        )
        plot.assert_not_called()

    def test_main_success_path_still_returns_zero(self):
        measurement = {
            "pair_name": "AA__BB",
            "distance_km": 20.0,
            "period_s": 3.0,
            "group_time_s": 8.0,
            "group_velocity_km_s": 2.5,
            "leading_snr": 8.0,
            "trailing_snr": 8.0,
            "phi_tu_rad": 0.2,
            "raw_travel_time_s": 7.5,
        }
        return_code, output_dir, metadata, plot = self._run_main_fixture(
            [
                {
                    "pair_name": "AA__BB",
                    "ok": True,
                    "measurements": [measurement],
                    "continuous_observations": [
                        {
                            "pair_name": f"AA__B{index}",
                            "distance_km": 20.0 + index,
                            "azimuth_deg": 45.0 * index,
                            "instantaneous_period_s": 3.0,
                            "anchored_raw_time_s": (20.0 + index) / 2.5,
                            "group_slowness_s_km": 0.4,
                            "convention": "BENSEN_VELOCITY_CCF",
                        }
                        for index in range(5)
                    ],
                    "rejections": [],
                }
            ]
        )
        self.assertEqual(return_code, 0)
        self.assertEqual(metadata["run_status"], "success")
        audit = (output_dir / "reference_cv_audit.csv").read_text(
            encoding="utf-8"
        )
        self.assertIn("observation_assignment", audit)
        self.assertIn("fold_membership", audit)
        self.assertIn("config_fold_loss", audit)
        self.assertTrue(
            (output_dir / "measurements_corrected.csv").exists()
        )
        self.assertTrue(
            (output_dir / "measurements_right_qc.csv").exists()
        )
        fit_summary = (output_dir / "fit_summary.csv").read_text(
            encoding="utf-8"
        )
        for field in (
            "huber_velocity_km_s",
            "ordinary_ls_velocity_km_s",
            "path_velocity_std_km_s",
            "bootstrap_velocity_ci95_low_km_s",
            "bootstrap_velocity_ci95_high_km_s",
            "bootstrap_seed",
        ):
            self.assertIn(field, fit_summary)
        self.assertEqual(
            metadata["right_column_fit"]["bootstrap_seed"],
            20260717,
        )
        self.assertEqual(
            metadata["right_column_fit"]["bootstrap_unit"],
            "station pair",
        )
        self.assertTrue((output_dir / "input_inventory.json").exists())
        input_inventory = json.loads(
            (output_dir / "input_inventory.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            metadata["preliminary_snr_inventory"]["definition"]["period_s"],
            3.5,
        )
        self.assertEqual(
            len(metadata["preliminary_snr_inventory"]["rows_sha256"]),
            64,
        )
        self.assertNotIn("preliminary_snr", input_inventory)
        plot.assert_called_once()

    def test_figure_and_metadata_follow_all_four_exact_targets(self):
        plot_source = inspect.getsource(self.mod.plot_figure)
        self.assertIn("len(FIGURE4_PERIODS_S)", plot_source)
        self.assertEqual(self.mod.FIGURE4_PERIODS_S, (3.0, 4.0, 5.0))
        self.assertEqual(self.mod.TARGET_PERIODS_S, (3.0, 3.5, 4.0, 5.0))
        self.assertIn("axes[-1, 0]", plot_source)
        self.assertIn("axes[-1, 1]", plot_source)
        main_source = inspect.getsource(self.mod.main)
        self.assertNotIn('"preleft_min_wavelengths"', main_source)
        self.assertNotIn('"guard_fraction_period"', main_source)
        self.assertIn('"period_for_guard": "T_inst"', main_source)
        self.assertIn('"target_qc_period": "T_target"', main_source)
        report_source = inspect.getsource(self.mod.write_report_html)
        self.assertIn("<code>3.5 s</code>", report_source)
        self.assertIn("八联图", report_source)
        self.assertIn("Huber 过原点稳健走时拟合", report_source)
        defaults = self.mod.parse_args([])
        self.assertEqual(
            defaults.stack_root.name,
            "STACK_SPIKE_REMOVED_DIAGFIT_20260628",
        )
        self.assertEqual(defaults.raw_stack_root.name, "STACK")

    def test_stage_b_freeze_writer_never_disguises_failure_as_success(self):
        validation = self.mod.wang_ftan_validation
        failed_budget = validation.evaluate_stage_b_budget(
            candidate_benchmark_elapsed_s=86_400.0,
            candidate_benchmark_work_units=1.0,
            stage_b_candidate_work_units=2.0,
            reference_benchmark_elapsed_s=1.0,
            reference_benchmark_optimizer_calls=10,
            distinct_measurement_class_count=300,
            worker_count=24,
            measured_peak_memory_bytes=1,
            available_memory_bytes=10,
        )
        benchmark_evidence = validation.StageBBenchmarkEvidence(
            candidate_grid_elapsed_s=1.0,
            ten_single_reference_fits_elapsed_s=1.0,
            lambda_cv_elapsed_s=1.0,
            twenty_half_samples_elapsed_s=1.0,
            measured_peak_memory_bytes=1,
            available_memory_bytes=10,
            cache_hit_fraction=0.0,
            benchmark_input_sha256="f" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            failed = validation.StageBRunResult(
                status="insufficient_triplet_support",
                return_code=2,
                budget=failed_budget,
                benchmark_evidence=benchmark_evidence,
                selection=None,
                candidate_results=(),
                measurement_classes={},
                class_evidence={},
                decision=None,
                phase_matching_diagnostics={},
                frozen_parameters=None,
                audit={"reason": "insufficient_triplet_support"},
            )
            stale = output_dir / "frozen_parameters.json"
            stale.write_text('{"stage_b_status":"passed"}', encoding="utf-8")
            return_code = self.mod.write_stage_b_freeze_decision(
                output_dir,
                failed,
            )
            self.assertNotEqual(return_code, 0)
            self.assertFalse(stale.exists())
            failure = json.loads(
                (output_dir / "stage_b_decision.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                failure["stage_b_status"],
                "insufficient_triplet_support",
            )
            self.assertFalse(failure["formal_full_run_allowed"])

    def test_execute_stage_b_injects_the_real_second_pass_ftan_core(self):
        result = object()
        output_dir = Path("/tmp/stage-b-output")
        with mock.patch.object(
            self.mod.wang_ftan_validation,
            "run_stage_b_validation",
            return_value=result,
        ) as run_validation, mock.patch.object(
            self.mod,
            "write_stage_b_freeze_decision",
            return_value=0,
        ) as write_decision:
            return_code = self.mod.execute_stage_b(
                output_dir,
                inventory_rows=(),
                phase_matched_second_pass_ftan=lambda **kwargs: object(),
            )
        self.assertEqual(return_code, 0)
        self.assertIs(
            run_validation.call_args.kwargs[
                "phase_matched_second_pass_ftan"
            ],
            self.mod.phase_matched_second_pass_ftan,
        )
        write_decision.assert_called_once_with(output_dir, result)

    def test_execute_stage_b_persists_nonzero_when_validation_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stale = output / "frozen_parameters.json"
            stale.write_text("stale success", encoding="utf-8")
            with mock.patch.object(
                self.mod.wang_ftan_validation,
                "run_stage_b_validation",
                side_effect=ValueError("phase matching incomplete"),
            ):
                code = self.mod.execute_stage_b(output)
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertFalse(stale.exists())
        self.assertEqual(code, 2)
        self.assertEqual(metadata["run_status"], "failed")
        self.assertEqual(
            metadata["terminal_failure_reason"],
            "stage_b_validation_error",
        )
        self.assertIn("phase matching incomplete", metadata["detail"])

    def test_stage_cli_rejects_stage_c_without_one_unmodified_freeze(self):
        with self.assertRaises(SystemExit):
            self.mod.parse_args(
                [
                    "--stage",
                    "B",
                    "--phase-convention",
                    "BENSEN_VELOCITY_CCF",
                    "--alpha",
                    "12",
                    "--beta1",
                    "1",
                    "--beta2",
                    "2",
                    "--resume",
                ]
            )

        with self.assertRaises(SystemExit):
            self.mod.parse_args(["--stage", "C"])
        with self.assertRaises(SystemExit):
            self.mod.parse_args(
                [
                    "--stage",
                    "C",
                    "--frozen-parameters",
                    "/tmp/frozen_parameters.json",
                    "--alpha",
                    "12",
                ]
            )

    def test_stage_cli_missing_stack_or_station_input_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_stack = root / "stacks"
            existing_stack.mkdir()
            stations = root / "stations.csv"
            stations.write_text(
                "station_code,longitude,latitude\nAA,0,0\n",
                encoding="utf-8",
            )
            cases = (
                (root / "missing-stacks", stations),
                (existing_stack, root / "missing-stations.csv"),
            )
            for index, (stack_root, stations_csv) in enumerate(cases):
                with self.subTest(index=index):
                    output = root / f"out-{index}"
                    output.mkdir()
                    stale = (
                        output / "frozen_parameters.json",
                        output / "stage_b_decision.json",
                        output / "report.html",
                    )
                    for path in stale:
                        path.write_text("stale success", encoding="utf-8")
                    return_code = self.mod.main(
                        [
                            "--stage",
                            "B",
                            "--stack-root",
                            str(stack_root),
                            "--stations-csv",
                            str(stations_csv),
                            "--output-dir",
                            str(output),
                            "--max-workers",
                            "1",
                        ]
                    )
                    self.assertNotEqual(return_code, 0)
                    metadata = json.loads(
                        (output / "metadata.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(metadata["run_status"], "failed")
                    self.assertIn(
                        metadata["terminal_failure_reason"],
                        ("missing_stack_root", "missing_stations_csv"),
                    )
                    self.assertTrue(all(not path.exists() for path in stale))

    def test_stage_b_cli_uses_formal_validation_not_legacy_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stack_root = root / "stacks"
            stack_root.mkdir()
            stations = root / "stations.csv"
            stations.write_text(
                "station_code,longitude,latitude\nAA,0,0\nBB,1,0\n",
                encoding="utf-8",
            )
            output = root / "output"
            inventory = {
                "stack_file_count": 1,
                "valid_component_count": 1,
                "dt_distribution": {"0.04": 1},
                "sample_count_distribution": {"7501": 1},
                "maxlag_distribution": {"150": 1},
                "lineage_status": "confirmed",
            }
            task = (
                str(stack_root / "AA" / "BB" / "stack_pws.h5"),
                "AA",
                "BB",
                0.0,
                0.0,
                1.0,
                0.0,
            )
            with mock.patch.object(
                self.mod,
                "audit_input_inventory_and_lineage",
                return_value=inventory,
            ), mock.patch.object(
                self.mod,
                "load_station_coords",
                return_value={"AA": (0.0, 0.0), "BB": (1.0, 0.0)},
            ), mock.patch.object(
                self.mod,
                "iter_stack_tasks",
                return_value=iter((task,)),
            ), mock.patch.object(
                self.mod,
                "formal_runtime_code_sha256",
                return_value="b" * 64,
            ), mock.patch.object(
                self.mod,
                "formal_scientific_config_sha256",
                return_value="c" * 64,
            ), mock.patch.object(
                self.mod,
                "run_stage_a_test_suite",
                return_value=0,
            ), mock.patch.object(
                self.mod,
                "execute_stage_b",
                return_value=7,
            ) as execute_stage_b, mock.patch.object(
                self.mod,
                "run_checkpointed_pair_tasks",
            ) as legacy_checkpoints:
                return_code = self.mod.main(
                    [
                        "--stage",
                        "B",
                        "--stack-root",
                        str(stack_root),
                        "--stations-csv",
                        str(stations),
                        "--output-dir",
                        str(output),
                        "--max-workers",
                        "1",
                    ]
                )
            self.assertEqual(return_code, 7)
            execute_stage_b.assert_called_once()
            legacy_checkpoints.assert_not_called()
            kwargs = execute_stage_b.call_args.kwargs
            self.assertTrue(kwargs["inventory_rows"])
            self.assertTrue(callable(kwargs["measure_candidate"]))
            self.assertTrue(callable(kwargs["fit_full_reference"]))
            self.assertTrue(callable(kwargs["fit_split_half_reference"]))
            self.assertTrue(callable(kwargs["run_phase_matching"]))

    def test_stage_b_candidate_adapter_measures_only_selected_real_tasks(self):
        task_by_pair = {
            "AA__BB": ("aa-bb.h5", "AA", "BB", 0.0, 0.0, 1.0, 0.0),
            "AA__CC": ("aa-cc.h5", "AA", "CC", 0.0, 0.0, 2.0, 0.0),
        }
        left_row = {
            "pair_name": "AA__BB",
            "T_inst": 3.05,
            "t0": 8.0,
            "U": 2.5,
            "signal_peak": 10.0,
            "leading_rms": 1.0,
            "trailing_rms": 1.0,
            "leading_snr": 10.0,
            "trailing_snr": 10.0,
            "ridge_fields": {"outermost_velocity_cell": True},
            "ridge_valid": True,
            "instantaneous_period_valid": True,
        }
        selection = SimpleNamespace(selected_pair_names=("AA__BB",))
        candidate = {
            "candidate_id": "BENSEN-a12-b1-b2",
            "phase_convention": "BENSEN_VELOCITY_CCF",
            "alpha": 12.0,
            "beta1": 1.0,
            "beta2": 2.0,
        }
        with mock.patch.object(
            self.mod,
            "process_one_pair",
            return_value={
                "pair_name": "AA__BB",
                "ok": True,
                "continuous_observations": [left_row],
            },
        ) as process:
            measured = self.mod.measure_stage_b_candidate_from_tasks(
                candidate=candidate,
                selection=selection,
                task_by_pair=task_by_pair,
                component="ZZ",
                max_workers=1,
                synthetic_validation_status="accepted",
            )
        process.assert_called_once()
        submitted = process.call_args.args[0]
        self.assertEqual(submitted[:7], task_by_pair["AA__BB"])
        self.assertEqual(submitted[7], "ZZ")
        self.assertEqual(submitted[8]["alpha"], 12.0)
        self.assertEqual(measured["continuous_left_rows"], (left_row,))
        self.assertEqual(
            measured["accepted_outermost_velocity_cell_count"],
            1,
        )
        self.assertEqual(measured["synthetic_validation_status"], "accepted")
        self.assertEqual(measured["processed_pair_count"], 1)
        self.assertEqual(measured["successful_pair_count"], 1)
        self.assertEqual(measured["unexpected_pair_exception_count"], 0)

    def test_stage_b_candidate_synthetic_gate_executes_alpha_and_beta_checks(self):
        caches = ({}, {})
        common = {
            "phase_convention": "BENSEN_VELOCITY_CCF",
            "alpha": 12.0,
            "beta1": 0.0,
        }
        rejected = self.mod.stage_b_candidate_synthetic_status(
            {**common, "candidate_id": "bad", "beta2": 0.0},
            phase_alpha_cache=caches[0],
            beta_cache=caches[1],
        )
        accepted = self.mod.stage_b_candidate_synthetic_status(
            {**common, "candidate_id": "good", "beta2": 1.0},
            phase_alpha_cache=caches[0],
            beta_cache=caches[1],
        )
        self.assertEqual(rejected, "rejected")
        self.assertEqual(accepted, "accepted")
        self.assertEqual(len(caches[0]), 1)
        self.assertEqual(len(caches[1]), 2)

    def test_stage_b_routes_candidate_specific_synthetic_status_to_measurement(self):
        task = ("aa-bb.h5", "AA", "BB", 0.0, 0.0, 1.0, 0.0)
        candidate = {
            "candidate_id": "candidate",
            "phase_convention": "BENSEN_VELOCITY_CCF",
            "alpha": 12.0,
            "beta1": 0.0,
            "beta2": 0.0,
        }
        selection = SimpleNamespace(selected_pair_names=("AA__BB",))

        def execute(_output_dir, **kwargs):
            measured = kwargs["measure_candidate"](
                candidate=candidate,
                selection=selection,
            )
            self.assertEqual(
                measured["synthetic_validation_status"],
                "rejected",
            )
            return 9

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            self.mod,
            "run_stage_a_test_suite",
            return_value=0,
        ), mock.patch.object(
            self.mod,
            "build_stage_b_inventory_rows",
            return_value=(
                {
                    "pair_name": "AA__BB",
                    "distance_km": 10.0,
                    "azimuth_deg": 0.0,
                    "preliminary_snr": 10.0,
                },
            ),
        ), mock.patch.object(
            self.mod,
            "build_stage_b_closure_triplets",
            return_value=((), ()),
        ), mock.patch.object(
            self.mod,
            "stage_b_candidate_synthetic_status",
            return_value="rejected",
        ) as synthetic, mock.patch.object(
            self.mod,
            "measure_stage_b_candidate_from_tasks",
            side_effect=lambda **kwargs: {
                "synthetic_validation_status": kwargs[
                    "synthetic_validation_status"
                ]
            },
        ), mock.patch.object(
            self.mod,
            "execute_stage_b",
            side_effect=execute,
        ):
            frozen_inventory = {"lineage_status": "confirmed"}
            inventory_hash = self.mod._canonical_json_sha256(frozen_inventory)
            code = self.mod.run_stage_b_from_tasks(
                tasks=(task,),
                station_coordinates={"AA": (0.0, 0.0), "BB": (1.0, 0.0)},
                input_inventory=frozen_inventory,
                input_inventory_sha256=inventory_hash,
                code_sha256="b" * 64,
                config_sha256="c" * 64,
                output_dir=Path(tmp),
                component="ZZ",
                max_workers=1,
            )
            persisted_inventory = json.loads(
                (Path(tmp) / "input_inventory.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(code, 9)
        self.assertEqual(persisted_inventory, frozen_inventory)
        self.assertEqual(
            self.mod._canonical_json_sha256(persisted_inventory),
            inventory_hash,
        )
        synthetic.assert_called_once_with(
            candidate,
            phase_alpha_cache={},
            beta_cache={},
        )

    def test_stage_b_target_adapter_resamples_then_resolves_cycles(self):
        def row(period, raw_time):
            return {
                "pair_name": "AA__BB",
                "distance_km": 20.0,
                "azimuth_deg": 90.0,
                "T_inst": period,
                "t0": raw_time,
                "U": 2.5,
                "group_time_s": 8.0,
                "signal_peak": 10.0,
                "leading_rms": 1.0,
                "trailing_rms": 1.0,
                "leading_snr": 10.0,
                "trailing_snr": 10.0,
                "ridge_fields": {
                    "normalized_log_energy": 0.8,
                    "normalized_envelope_amplitude": 0.7,
                    "adjacent_jump_km_s": 0.02,
                    "outermost_velocity_cell": False,
                },
            }

        periods = self.mod.FtanConfig().periods_s
        reference = SimpleNamespace(
            periods_s=periods,
            phase_slowness_s_km=np.full(periods.size, 0.4),
        )
        corrected = self.mod.stage_b_corrected_target_rows(
            (row(2.95, 7.9), row(3.05, 8.1)),
            reference_fit=reference,
        )
        self.assertEqual(len(corrected), 1)
        result = corrected[0]
        self.assertIsInstance(
            result,
            self.mod.wang_ftan_validation.CorrectedTargetObservation,
        )
        self.assertEqual(result.pair_name, "AA__BB")
        self.assertEqual(result.target_period_s, 3.0)
        self.assertAlmostEqual(result.reference_time_s, 8.0)
        self.assertLessEqual(abs(result.reference_residual_s), 1.5)
        self.assertTrue(result.left_qc_accepted)

    def test_stage_b_full_reference_adapter_returns_typed_evidence(self):
        left_rows = (
            {
                "pair_name": "AA__BB",
                "distance_km": 20.0,
                "azimuth_deg": 90.0,
                "T_inst": 3.0,
                "t0": 8.0,
                "U": 2.5,
                "group_time_s": 8.0,
                "signal_peak": 10.0,
                "leading_rms": 1.0,
                "trailing_rms": 1.0,
                "leading_snr": 10.0,
                "trailing_snr": 10.0,
                "ridge_fields": {},
            },
        )
        starts = tuple(
            SimpleNamespace(
                basin_id=index,
                phase_slowness_s_km=np.full(51, 0.4 + index * 1e-4),
            )
            for index in range(5)
        )
        fit = SimpleNamespace(
            status="accepted",
            lambda_s=0.01,
            lambda_g=0.1,
            optimizer_calls=753,
            local_solutions=starts,
            representative_indices=tuple(range(5)),
            result_hash="d" * 64,
        )
        corrected = self.mod.wang_ftan_validation.CorrectedTargetObservation(
            pair_name="AA__BB",
            target_period_s=3.0,
            raw_time_s=8.0,
            cycle_count=0,
            corrected_time_s=8.0,
            reference_time_s=8.0,
            reference_residual_s=0.0,
            leading_snr=10.0,
            trailing_snr=10.0,
            left_qc_accepted=True,
        )
        with mock.patch.object(
            self.mod,
            "fit_reference_dispersion",
            return_value=fit,
        ) as fitter, mock.patch.object(
            self.mod,
            "stage_b_corrected_target_rows",
            return_value=(corrected,),
        ):
            evidence = self.mod.fit_stage_b_full_reference(
                left_rows=left_rows,
                candidate_ids=("c001",),
                maximum_optimizer_calls=753,
            )
        fitter.assert_called_once()
        observation = fitter.call_args.args[0][0]
        self.assertEqual(observation.pair_name, "AA__BB")
        self.assertEqual(observation.instantaneous_period_s, 3.0)
        self.assertIsInstance(
            evidence,
            self.mod.wang_ftan_validation.FullReferenceEvidence,
        )
        self.assertEqual(evidence.status, "accepted")
        self.assertEqual(evidence.alias_status, "accepted")
        self.assertEqual(evidence.corrected_rows, (corrected,))
        self.assertEqual(len(evidence.basin_starts), 5)

    def test_stage_b_split_half_adapter_reuses_lambdas_without_cv(self):
        rows = (
            {
                "pair_name": "AA__BB",
                "distance_km": 20.0,
                "azimuth_deg": 90.0,
                "T_inst": 3.0,
                "t0": 8.0,
                "U": 2.5,
                "convention": "BENSEN_VELOCITY_CCF",
            },
        )
        starts = tuple(
            SimpleNamespace(phase_slowness_s_km=np.full(51, 0.38 + 0.01 * i))
            for i in range(5)
        )
        full = SimpleNamespace(basin_starts=starts)
        target = np.full(51, 0.4)
        with mock.patch.object(
            self.mod,
            "reference_fit_objective",
            side_effect=lambda candidate, *_args, **_kwargs: float(
                np.sum((np.asarray(candidate) - target) ** 2)
            ),
        ):
            evidence = self.mod.fit_stage_b_split_half_reference(
                left_rows=rows,
                full_reference=full,
                split_index=0,
                seed=20260717,
                side="A",
                lambda_s=0.01,
                lambda_g=0.1,
                basin_starts=starts,
                maxiter=300,
            )
        self.assertIsInstance(
            evidence,
            self.mod.wang_ftan_validation.SplitHalfFitEvidence,
        )
        self.assertEqual(evidence.status, "accepted")
        self.assertEqual(evidence.optimizer_calls, 5)
        self.assertEqual(evidence.cv_optimizer_calls, 0)
        self.assertEqual(evidence.maxiter, 300)
        np.testing.assert_allclose(evidence.target_velocities_km_s, 2.5)

    def test_stage_b_split_half_rejects_finite_nonconverged_solutions(self):
        rows = (
            {
                "pair_name": "AA__BB",
                "distance_km": 20.0,
                "azimuth_deg": 90.0,
                "T_inst": 3.0,
                "t0": 8.0,
                "U": 2.5,
                "convention": "BENSEN_VELOCITY_CCF",
            },
        )
        starts = (
            SimpleNamespace(phase_slowness_s_km=np.full(51, 0.4)),
        )
        full = SimpleNamespace(basin_starts=starts)
        failed_result = SimpleNamespace(
            x=np.full(51, 0.4),
            success=False,
        )
        with mock.patch.object(
            self.mod,
            "minimize",
            return_value=failed_result,
        ), mock.patch.object(
            self.mod,
            "reference_fit_objective",
            return_value=0.0,
        ):
            evidence = self.mod.fit_stage_b_split_half_reference(
                left_rows=rows,
                full_reference=full,
                split_index=0,
                seed=20260717,
                side="A",
                lambda_s=0.01,
                lambda_g=0.1,
                basin_starts=starts,
                maxiter=300,
            )
        self.assertEqual(evidence.status, "rejected")

    def test_stage_b_benchmark_adapter_requires_and_records_fixed_workload(self):
        workload = {
            "candidate_grid_elapsed_s": 2.0,
            "ten_single_reference_fits_elapsed_s": 3.0,
            "lambda_cv_elapsed_s": 4.0,
            "twenty_half_samples_elapsed_s": 5.0,
            "measured_peak_memory_bytes": 1024,
            "available_memory_bytes": 4096,
            "cache_hit_fraction": 0.8,
            "benchmark_input_sha256": "a" * 64,
        }
        with mock.patch.object(
            self.mod,
            "_run_stage_b_benchmark_workload",
            return_value=workload,
        ) as run:
            evidence = self.mod.benchmark_stage_b_runtime(
                candidate_count=300,
                synthetic_left_observation_count=2000,
                synthetic_waveform_count=20,
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
                max_workers=8,
            )
        run.assert_called_once_with(max_workers=8)
        self.assertIsInstance(
            evidence,
            self.mod.wang_ftan_validation.StageBBenchmarkEvidence,
        )
        self.assertEqual(evidence.candidate_grid_elapsed_s, 2.0)
        with self.assertRaisesRegex(ValueError, "fixed workload"):
            self.mod.benchmark_stage_b_runtime(
                candidate_count=299,
                synthetic_left_observation_count=2000,
                synthetic_waveform_count=20,
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
                max_workers=8,
            )

    def test_uncached_benchmark_projection_matches_current_production_path(self):
        projected = self.mod.project_uncached_candidate_grid_seconds(
            filter_bank_elapsed_s=2.0,
            ridge_elapsed_s=3.0,
            beta_grid_count=25,
        )
        self.assertEqual(projected, 53.0)
        with self.assertRaisesRegex(ValueError, "benchmark timing"):
            self.mod.project_uncached_candidate_grid_seconds(
                filter_bank_elapsed_s=-1.0,
                ridge_elapsed_s=3.0,
                beta_grid_count=25,
            )

    def test_stage_b_phase_matching_rejects_incomplete_single_pair_diagnostic(self):
        periods = self.mod.FtanConfig().periods_s
        task_by_pair = {
            "AA__BB": ("pair.h5", "AA", "BB", 0.0, 0.0, 1.0, 0.0),
        }
        trace = SimpleNamespace(
            time_positive_s=np.arange(100, dtype=float) * 0.04,
            symmetric=np.ones(100),
            positive_lag=np.ones(100),
            negative_lag_reversed=np.ones(100),
            dt_s=0.04,
        )
        curve = SimpleNamespace(group_times_s=np.full(periods.size, 8.0))
        closure = SimpleNamespace(
            period_summaries={
                period: SimpleNamespace(median_absolute_cycles=0.1)
                for period in self.mod.TARGET_PERIODS_S
            }
        )
        class_evidence = {
            "continuous_left_rows": (
                {
                    "pair_name": "AA__BB",
                    "ridge_fields": {"coverage": 0.9},
                },
            ),
            "closure": closure,
        }
        candidate = {
            "candidate_id": "c001",
            "phase_convention": "BENSEN_VELOCITY_CCF",
            "alpha": 12.0,
            "beta1": 1.0,
            "beta2": 2.0,
            "accepted_boundary_fraction": 0.01,
        }
        second_pass_result = object()
        executor = mock.Mock(return_value=second_pass_result)
        with mock.patch.object(
            self.mod,
            "read_stack_trace",
            return_value=trace,
        ), mock.patch.object(
            self.mod,
            "measure_phase_curve",
            return_value=curve,
        ), mock.patch.object(
            self.mod,
            "prepare_phase_waveform",
            return_value=np.ones(100),
        ), mock.patch.object(
            self.mod.wang_ftan_validation,
            "hash_phase_matching_second_pass_output",
            return_value="e" * 64,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "complete real-pair phase-matching diagnostic",
            ):
                self.mod.run_stage_b_phase_matching(
                    candidate=candidate,
                    class_evidence=class_evidence,
                    execute_second_pass_ftan=executor,
                    task_by_pair=task_by_pair,
                    component="ZZ",
                )
        executor.assert_called_once()

    def test_stage_b_phase_matching_remeasures_complete_real_pair_class(self):
        periods = self.mod.FtanConfig().periods_s
        task_by_pair = {
            "AA__BB": ("ab.h5", "AA", "BB", 0.0, 0.0, 0.1, 0.0),
            "BB__CC": ("bc.h5", "BB", "CC", 0.1, 0.0, 0.2, 0.0),
        }
        trace = SimpleNamespace(
            time_positive_s=np.arange(1500, dtype=float) * 0.04,
            symmetric=np.ones(1500),
            positive_lag=np.ones(1500),
            negative_lag_reversed=np.ones(1500),
            dt_s=0.04,
        )
        curve = SimpleNamespace(group_times_s=np.full(periods.size, 8.0))
        matched_result = SimpleNamespace(
            cleaning=SimpleNamespace(cleaned_waveform=np.ones(1500))
        )
        closure = SimpleNamespace(
            period_summaries={
                period: SimpleNamespace(median_absolute_cycles=0.10)
                for period in self.mod.TARGET_PERIODS_S
            }
        )
        matched_closure = SimpleNamespace(
            period_summaries={
                period: SimpleNamespace(median_absolute_cycles=0.08)
                for period in self.mod.TARGET_PERIODS_S
            }
        )
        class_evidence = {
            "continuous_left_rows": tuple(
                {
                    "pair_name": pair_name,
                    "ridge_fields": {
                        "coverage": 0.80,
                        "outermost_velocity_cell": False,
                    },
                }
                for pair_name in task_by_pair
            ),
            "closure": closure,
            "triplet_rows_geometry_valid": (),
        }
        candidate = {
            "candidate_id": "c001",
            "phase_convention": "BENSEN_VELOCITY_CCF",
            "alpha": 12.0,
            "beta1": 1.0,
            "beta2": 2.0,
            "accepted_boundary_fraction": 0.01,
        }
        matched_rows = {
            pair_name: [
                {
                    "pair_name": pair_name,
                    "T_inst": 3.0,
                    "t0": 8.0,
                    "U": 2.5,
                    "signal_peak": 10.0,
                    "leading_rms": 1.0,
                    "trailing_rms": 1.0,
                    "ridge_fields": {
                        "coverage": 0.95,
                        "outermost_velocity_cell": False,
                    },
                    "ridge_valid": True,
                    "instantaneous_period_valid": True,
                    "distance_km": 20.0,
                    "azimuth_deg": 90.0,
                    "convention": "BENSEN_VELOCITY_CCF",
                }
            ]
            for pair_name in task_by_pair
        }
        executor = mock.Mock(return_value=matched_result)
        with mock.patch.object(
            self.mod,
            "read_stack_trace",
            return_value=trace,
        ), mock.patch.object(
            self.mod,
            "measure_phase_curve",
            return_value=curve,
        ) as measure, mock.patch.object(
            self.mod,
            "build_reference_observations_from_task5_curve",
            side_effect=lambda pair_name, **_kwargs: matched_rows[pair_name],
        ), mock.patch.object(
            self.mod,
            "fit_stage_b_full_reference",
            return_value=SimpleNamespace(corrected_rows=()),
        ), mock.patch.object(
            self.mod.wang_ftan_validation,
            "evaluate_triplet_closure",
            return_value=matched_closure,
        ), mock.patch.object(
            self.mod.wang_ftan_validation,
            "hash_phase_matching_second_pass_output",
            side_effect=("d" * 64, "e" * 64),
        ):
            evidence = self.mod.run_stage_b_phase_matching(
                candidate=candidate,
                class_evidence=class_evidence,
                execute_second_pass_ftan=executor,
                task_by_pair=task_by_pair,
                component="ZZ",
            )
        self.assertEqual(executor.call_count, 2)
        self.assertEqual(measure.call_count, 4)
        self.assertEqual(
            evidence.matched_output_sha256,
            self.mod.wang_ftan_validation.hash_phase_matching_execution_hashes(
                ("d" * 64, "e" * 64)
            ),
        )
        self.assertTrue(evidence.diagnostic.design_revision_required)

    def test_stage_a_cli_runs_validation_without_real_stack_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "stage-a"
            with mock.patch.object(
                self.mod,
                "run_stage_a_test_suite",
                return_value=5,
            ) as run, mock.patch.object(
                self.mod,
                "audit_input_inventory_and_lineage",
            ) as inventory:
                code = self.mod.main(
                    [
                        "--stage",
                        "A",
                        "--stack-root",
                        str(Path(tmp) / "does-not-exist"),
                        "--stations-csv",
                        str(Path(tmp) / "does-not-exist.csv"),
                        "--output-dir",
                        str(output),
                    ]
                )
        self.assertEqual(code, 5)
        run.assert_called_once_with(output)
        inventory.assert_not_called()

    def test_formal_inventory_rejects_zero_component_and_mixed_sampling(self):
        baseline = {
            "stack_file_count": 2,
            "valid_component_count": 2,
            "dt_distribution": {"0.04": 2},
            "sample_count_distribution": {"7501": 2},
            "maxlag_distribution": {"150": 2},
        }
        self.assertIsNone(
            self.mod.formal_input_inventory_failure_reason(baseline)
        )
        cases = (
            (
                {**baseline, "valid_component_count": 0},
                "zero_valid_components",
            ),
            (
                {
                    **baseline,
                    "dt_distribution": {"0.04": 1, "0.05": 1},
                },
                "mixed_dt",
            ),
            (
                {
                    **baseline,
                    "sample_count_distribution": {"7501": 1, "7503": 1},
                },
                "mixed_sample_length",
            ),
            (
                {
                    **baseline,
                    "maxlag_distribution": {"150": 1, "160": 1},
                },
                "mixed_maxlag",
            ),
        )
        for inventory, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    self.mod.formal_input_inventory_failure_reason(
                        inventory
                    ),
                    expected,
                )

    def test_formal_science_gate_separates_qc_empty_from_exceptions(self):
        counts = {3.0: 10, 4.0: 10, 5.0: 10}
        self.assertIsNone(
            self.mod.formal_science_failure_reason(
                input_count=100,
                unexpected_exception_count=1,
                left_count_by_period=counts,
                right_count_by_period=counts,
            )
        )
        self.assertEqual(
            self.mod.formal_science_failure_reason(
                input_count=100,
                unexpected_exception_count=2,
                left_count_by_period=counts,
                right_count_by_period=counts,
            ),
            "unexpected_exception_fraction_exceeded",
        )
        self.assertEqual(
            self.mod.formal_science_failure_reason(
                input_count=100,
                unexpected_exception_count=0,
                left_count_by_period={**counts, 4.0: 0},
                right_count_by_period=counts,
            ),
            "empty_left_target_period_4s",
        )
        self.assertEqual(
            self.mod.formal_science_failure_reason(
                input_count=100,
                unexpected_exception_count=0,
                left_count_by_period=counts,
                right_count_by_period={**counts, 5.0: 0},
            ),
            "empty_right_target_period_5s",
        )

    def test_stage_c_freeze_loader_requires_hash_lineage_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = {"candidate_results": [{"candidate_id": "c001"}]}
            evidence_payload = json.dumps(
                evidence,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            evidence_hash = __import__("hashlib").sha256(
                evidence_payload
            ).hexdigest()
            evidence_path = root / "stage_b_validation_evidence.json"
            evidence_path.write_bytes(evidence_payload)
            frozen_path = root / "frozen_parameters.json"
            manifest = {
                "stage_b_status": "passed",
                "candidate_id": "c001",
                "phase_convention": "BENSEN_VELOCITY_CCF",
                "alpha": 12.0,
                "beta1": 1.0,
                "beta2": 2.0,
                "input_inventory_sha256": "a" * 64,
                "code_sha256": "b" * 64,
                "config_sha256": "c" * 64,
                "validation_table_sha256": evidence_hash,
            }
            frozen_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete|structure"):
                self.mod.load_stage_c_frozen_parameters(
                    frozen_path,
                    expected_input_inventory_sha256="a" * 64,
                    expected_code_sha256="b" * 64,
                    expected_config_sha256="c" * 64,
                )

            evidence_path.unlink()
            with self.assertRaisesRegex(ValueError, "validation evidence"):
                self.mod.load_stage_c_frozen_parameters(
                    frozen_path,
                    expected_input_inventory_sha256="a" * 64,
                    expected_code_sha256="b" * 64,
                    expected_config_sha256="c" * 64,
                )
            evidence_path.write_bytes(evidence_payload)
            bad = {**manifest, "stage_b_status": "failed"}
            frozen_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "passed"):
                self.mod.load_stage_c_frozen_parameters(
                    frozen_path,
                    expected_input_inventory_sha256="a" * 64,
                    expected_code_sha256="b" * 64,
                    expected_config_sha256="c" * 64,
                )

    def test_stage_c_recomputes_freeze_decision_from_complete_candidate_grid(self):
        grid = self.mod.wang_ftan_validation.build_candidate_grid(
            phase_conventions=(
                "BENSEN_VELOCITY_CCF",
                "LIN_NEGATIVE_DERIVATIVE_EGF",
            ),
            alpha_candidates=self.mod.FtanConfig().alpha_candidates,
            beta1_candidates=self.mod.FtanConfig().beta1_candidates,
            beta2_candidates=self.mod.FtanConfig().beta2_candidates,
        )
        left_hash = "f" * 64
        gates = {
            name: True
            for name in (
                "synthetic_passes",
                "ridge_passes",
                "instantaneous_period_passes",
                "alias_passes",
                "triplet_passes",
                "half_sample_passes",
                "boundary_passes",
            )
        }
        candidates = []
        for index, source in enumerate(grid):
            row = dict(source)
            row.update(
                {
                    **gates,
                    "closure_median_cycles": 0.1 if index == 0 else 1.0,
                    "left_observation_sha256": left_hash,
                }
            )
            candidates.append(row)
        arbitrary = candidates[1]
        evidence = {
            "budget": {"accepted": True, "status": "accepted"},
            "benchmark_evidence": {"benchmark_input_sha256": "a" * 64},
            "selection": {"seed": 20260717, "max_random_pairs": 2000},
            "candidate_results": candidates,
            "measurement_classes": {
                left_hash: [row["candidate_id"] for row in candidates]
            },
            "class_evidence": {left_hash: {"placeholder": True}},
            "phase_matching_diagnostics": {
                phase: {
                    "phase_convention": phase,
                    "second_pass_ftan_executed": True,
                    "diagnostic": {
                        "freeze_raw_ftan": True,
                        "design_revision_required": False,
                        "status": "raw_ftan_frozen",
                    },
                }
                for phase in (
                    "BENSEN_VELOCITY_CCF",
                    "LIN_NEGATIVE_DERIVATIVE_EGF",
                )
            },
        }
        manifest = {
            name: arbitrary[name]
            for name in (
                "candidate_id",
                "phase_convention",
                "alpha",
                "beta1",
                "beta2",
            )
        }
        manifest.update(
            {
                "lineage_status": "confirmed",
                "lineage_preferred_phase_convention": (
                    "BENSEN_VELOCITY_CCF"
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "decision"):
            self.mod._validate_stage_c_evidence_structure(evidence, manifest)
        selected_manifest = {
            name: candidates[0][name]
            for name in (
                "candidate_id",
                "phase_convention",
                "alpha",
                "beta1",
                "beta2",
            )
        }
        selected_manifest.update(
            {
                "lineage_status": "confirmed",
                "lineage_preferred_phase_convention": (
                    "BENSEN_VELOCITY_CCF"
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "measurement-class"):
            self.mod._validate_stage_c_evidence_structure(
                evidence,
                selected_manifest,
            )

    def test_stage_c_rejects_freeze_before_any_pair_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stack_root = root / "stacks"
            stack_root.mkdir()
            stations = root / "stations.csv"
            stations.write_text(
                "station_code,longitude,latitude\nAA,0,0\n",
                encoding="utf-8",
            )
            frozen = root / "frozen_parameters.json"
            frozen.write_text("{}", encoding="utf-8")
            output = root / "output"
            inventory = {
                "stack_file_count": 1,
                "valid_component_count": 1,
                "dt_distribution": {"0.04": 1},
                "sample_count_distribution": {"7501": 1},
                "maxlag_distribution": {"150": 1},
            }
            with mock.patch.object(
                self.mod,
                "audit_input_inventory_and_lineage",
                return_value=inventory,
            ), mock.patch.object(
                self.mod,
                "load_stage_c_frozen_parameters",
                side_effect=ValueError("freeze mismatch"),
            ) as load_freeze, mock.patch.object(
                self.mod,
                "run_checkpointed_pair_tasks",
            ) as checkpoint:
                return_code = self.mod.main(
                    [
                        "--stage",
                        "C",
                        "--stack-root",
                        str(stack_root),
                        "--stations-csv",
                        str(stations),
                        "--output-dir",
                        str(output),
                        "--frozen-parameters",
                        str(frozen),
                        "--max-workers",
                        "1",
                    ]
                )
            self.assertNotEqual(return_code, 0)
            load_freeze.assert_called_once()
            checkpoint.assert_not_called()
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["terminal_failure_reason"],
                "stage_c_frozen_parameters_invalid",
            )

    def test_checkpoint_resume_is_atomic_deterministic_and_conservative(self):
        tasks = tuple(
            {"pair_name": f"P{index:02d}", "value": index}
            for index in range(5)
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir()
            stale_extra = checkpoint_dir / "chunk_999999.json"
            stale_extra.write_text("stale", encoding="utf-8")
            first = self.mod.run_checkpointed_pair_tasks(
                tasks,
                output_dir=output_dir,
                chunk_size=2,
                config_sha256="c" * 64,
                process_task=checkpoint_fixture_processor,
                max_workers=1,
                resume=False,
                maximum_new_chunks=1,
            )
            self.assertFalse(first["complete"])
            self.assertFalse(stale_extra.exists())
            self.assertEqual(first["new_chunk_indices"], (0,))
            self.assertEqual(
                sorted(
                    path.name
                    for path in (output_dir / "checkpoints").glob("*.json")
                ),
                ["chunk_000000.json"],
            )
            self.assertFalse(
                list((output_dir / "checkpoints").glob("*.tmp"))
            )

            resumed = self.mod.run_checkpointed_pair_tasks(
                tasks,
                output_dir=output_dir,
                chunk_size=2,
                config_sha256="c" * 64,
                process_task=checkpoint_fixture_processor,
                max_workers=1,
                resume=True,
            )
            self.assertTrue(resumed["complete"])
            self.assertEqual(resumed["new_chunk_indices"], (1, 2))
            self.assertEqual(
                [row["pair_name"] for row in resumed["results"]],
                [f"P{index:02d}" for index in range(5)],
            )
            self.assertEqual(resumed["input_count"], 5)
            self.assertEqual(resumed["successful_pair_count"], 4)
            self.assertEqual(
                resumed["expected_scientific_rejection_count"],
                1,
            )
            self.assertEqual(
                resumed["unexpected_pair_exception_count"],
                0,
            )
            self.assertEqual(
                resumed["input_count"],
                resumed["successful_pair_count"]
                + resumed["expected_scientific_rejection_count"]
                + resumed["unexpected_pair_exception_count"],
            )

            repeated = self.mod.run_checkpointed_pair_tasks(
                tasks,
                output_dir=output_dir,
                chunk_size=2,
                config_sha256="c" * 64,
                process_task=checkpoint_fixture_processor,
                max_workers=1,
                resume=True,
            )
            self.assertEqual(repeated["new_chunk_indices"], ())
            self.assertEqual(repeated["results"], resumed["results"])

    def test_checkpoint_single_and_multi_process_content_is_identical(self):
        tasks = tuple(
            {"pair_name": f"P{index:02d}", "value": index}
            for index in range(5)
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = self.mod.run_checkpointed_pair_tasks(
                tasks,
                output_dir=root / "single",
                chunk_size=2,
                config_sha256="c" * 64,
                process_task=checkpoint_fixture_processor,
                max_workers=1,
                resume=False,
            )
            multi = self.mod.run_checkpointed_pair_tasks(
                tasks,
                output_dir=root / "multi",
                chunk_size=2,
                config_sha256="c" * 64,
                process_task=checkpoint_fixture_processor,
                max_workers=2,
                resume=False,
            )
            self.assertEqual(single["results"], multi["results"])
            self.assertEqual(
                single["scientific_content_sha256"],
                multi["scientific_content_sha256"],
            )

    def test_formal_success_lineage_records_stage_freeze_and_checkpoint_hashes(self):
        manifest = {
            "stage_b_status": "passed",
            "candidate_id": "candidate",
            "phase_convention": "LIN_NEGATIVE_DERIVATIVE_EGF",
            "alpha": 12.0,
            "beta1": 1.0,
            "beta2": 2.0,
            "input_inventory_sha256": "a" * 64,
            "code_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "validation_table_sha256": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp) / "frozen_parameters.json"
            frozen.write_text(json.dumps(manifest), encoding="utf-8")
            payload = self.mod.formal_success_lineage_metadata(
                stage="C",
                component="ZZ",
                input_inventory_sha256="a" * 64,
                code_sha256="b" * 64,
                config_sha256="c" * 64,
                frozen_manifest=manifest,
                frozen_parameters_path=frozen,
                checkpoint_run={
                    "scientific_content_sha256": "e" * 64,
                    "lineage": {
                        "frozen_lineage": {"stage_b_status": "passed"}
                    },
                },
            )
        self.assertEqual(payload["stage"], "C")
        self.assertEqual(
            payload["frozen_candidate"]["candidate_id"],
            "candidate",
        )
        self.assertEqual(payload["scientific_content_sha256"], "e" * 64)
        self.assertEqual(len(payload["frozen_parameters_file_sha256"]), 64)

    def test_formal_output_contract_names_every_required_audit_artifact(self):
        required = set(self.mod.FORMAL_REQUIRED_OUTPUTS)
        self.assertEqual(
            required,
            {
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
            },
        )

    def test_figure4_plot_uses_three_paper_periods_and_csv_scatter_counts(self):
        per_period = {}
        for period_s, left_count, right_count in (
            (3.0, 2, 1),
            (3.5, 9, 8),
            (4.0, 3, 2),
            (5.0, 4, 3),
        ):
            per_period[period_s] = {
                "left_rows": [
                    {
                        "distance_km": 8.0 + index,
                        "raw_travel_time_s": 3.0 + index * 0.1,
                    }
                    for index in range(left_count)
                ],
                "right_rows": [
                    {
                        "distance_km": 8.0 + index,
                        "corrected_travel_time_s": 3.0 + index * 0.1,
                    }
                    for index in range(right_count)
                ],
                "reference_velocity_km_s": 2.8,
                "fit_velocity_km_s": 2.7,
                "std_velocity_km_s": 0.1,
            }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "paper.png"
            audit = self.mod.plot_figure(
                output,
                per_period,
                paper_scale=True,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
        self.assertEqual(audit["periods_s"], [3.0, 4.0, 5.0])
        self.assertEqual(audit["subplot_shape"], [3, 2])
        self.assertEqual(
            audit["left_scatter_count_by_period"],
            {"3": 2, "4": 3, "5": 4},
        )
        self.assertEqual(
            audit["right_scatter_count_by_period"],
            {"3": 1, "4": 2, "5": 3},
        )
        self.assertTrue(audit["shared_axis_limits"])
        self.assertTrue(audit["reference_half_period_lines_both_columns"])
        self.assertTrue(audit["right_fit_line_and_statistics"])

    def test_formal_output_validator_rejects_empty_files_and_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.mod.validate_formal_outputs(
                root,
                expected_left_count_by_period={3.0: 2, 4.0: 3, 5.0: 4},
                expected_right_count_by_period={3.0: 1, 4.0: 2, 5.0: 3},
            )
            self.assertFalse(result["accepted"])
            self.assertEqual(result["status"], "formal_output_missing")

            for relative in self.mod.FORMAL_REQUIRED_OUTPUTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".csv":
                    path.write_text("placeholder\nvalue\n", encoding="utf-8")
                elif path.suffix == ".json":
                    path.write_text("{}\n", encoding="utf-8")
                else:
                    path.write_bytes(b"not-empty")
            examples = root / "figures" / "ftan_examples"
            examples.mkdir(parents=True)
            for index in range(3):
                (examples / f"example_{index}.png").write_bytes(b"png")
            candidate_rows = [
                "candidate_id,phase_convention,alpha,beta1,beta2,status"
            ] + [
                f"c{index},BENSEN_VELOCITY_CCF,5,0,0,rejected"
                for index in range(300)
            ]
            (root / "candidate_grid_results.csv").write_text(
                "\n".join(candidate_rows) + "\n",
                encoding="utf-8",
            )
            split_rows = ["split_index,seed,period_s,absolute_difference_km_s"] + [
                f"{index},{20260717 + index},{period},0.01"
                for index in range(20)
                for period in (3.0, 3.5, 4.0, 5.0)
            ]
            (root / "split_half_stability.csv").write_text(
                "\n".join(split_rows) + "\n",
                encoding="utf-8",
            )
            (root / "split_half_membership.csv").write_text(
                "split_index,half,pair_name\n"
                + "\n".join(
                    f"{index},A,P{index}" for index in range(20)
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "cycle_count_distribution.csv").write_text(
                "period_s,cycle_count,measurement_count,branch_tie_count\n"
                "3,-1,2,1\n",
                encoding="utf-8",
            )
            spatial_rows = [
                "diagnostic_dimension,bin_index,measurement_count"
            ] + [
                f"distance_quintile,{index},1" for index in range(5)
            ] + [
                f"azimuth_45deg,{index},1" for index in range(8)
            ]
            (root / "reference_spatial_diagnostics.csv").write_text(
                "\n".join(spatial_rows) + "\n",
                encoding="utf-8",
            )
            (root / "phase_matching_comparison.csv").write_text(
                "phase_convention,raw_valid_ridge_coverage,matched_valid_ridge_coverage\n"
                "BENSEN_VELOCITY_CCF,0.8,0.81\n"
                "LIN_NEGATIVE_DERIVATIVE_EGF,0.82,0.83\n",
                encoding="utf-8",
            )
            left_rows = ["period_s,pair_name"]
            right_rows = ["period_s,pair_name"]
            for period, count in ((3, 2), (4, 3), (5, 4)):
                left_rows.extend(
                    f"{period},L{period}_{index}" for index in range(count)
                )
            for period, count in ((3, 1), (4, 2), (5, 3)):
                right_rows.extend(
                    f"{period},R{period}_{index}" for index in range(count)
                )
            (root / "measurements_left_qc.csv").write_text(
                "\n".join(left_rows) + "\n",
                encoding="utf-8",
            )
            (root / "measurements_right_qc.csv").write_text(
                "\n".join(right_rows) + "\n",
                encoding="utf-8",
            )
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "run_status": "success",
                        "stage": "C",
                        "exit_status": 0,
                        "git_commit_sha": "a" * 40,
                        "python_version": "3.12",
                        "dependency_versions": {"numpy": "2.0"},
                        "stack_root": "/stack",
                        "stations_csv": "/stations.csv",
                        "input_file_count": 9,
                        "input_inventory_sha256": "b" * 64,
                        "code_sha256": "c" * 64,
                        "config_sha256": "d" * 64,
                        "phase_convention": "LIN_NEGATIVE_DERIVATIVE_EGF",
                        "frozen_candidate": {"candidate_id": "selected"},
                        "started_at": "start",
                        "finished_at": "finish",
                        "host": "work",
                    }
                ),
                encoding="utf-8",
            )
            (root / "frozen_parameters.json").write_text(
                json.dumps(
                    {
                        "stage_b_status": "passed",
                        "candidate_id": "selected",
                        "phase_convention": "LIN_NEGATIVE_DERIVATIVE_EGF",
                        "alpha": 12.0,
                        "beta1": 1.0,
                        "beta2": 2.0,
                    }
                ),
                encoding="utf-8",
            )
            accepted = self.mod.validate_formal_outputs(
                root,
                expected_left_count_by_period={3.0: 2, 4.0: 3, 5.0: 4},
                expected_right_count_by_period={3.0: 1, 4.0: 2, 5.0: 3},
            )
            self.assertTrue(accepted["accepted"], accepted)

            (root / "measurements_right_qc.csv").write_text(
                "period_s,pair_name\n3,R3_0\n",
                encoding="utf-8",
            )
            mismatch = self.mod.validate_formal_outputs(
                root,
                expected_left_count_by_period={3.0: 2, 4.0: 3, 5.0: 4},
                expected_right_count_by_period={3.0: 1, 4.0: 2, 5.0: 3},
            )
            self.assertFalse(mismatch["accepted"])
            self.assertEqual(mismatch["status"], "formal_output_count_mismatch")

    def test_stage_b_audit_tables_flatten_selected_class_without_losing_membership(self):
        selected_id = "selected"
        selected_hash = "a" * 64
        candidates = [
            {
                "candidate_id": selected_id,
                "phase_convention": "LIN_NEGATIVE_DERIVATIVE_EGF",
                "alpha": 12.0,
                "beta1": 1.0,
                "beta2": 2.0,
                "left_observation_sha256": selected_hash,
                "synthetic_passes": True,
            },
            {
                "candidate_id": "rejected",
                "phase_convention": "BENSEN_VELOCITY_CCF",
                "alpha": 5.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "left_observation_sha256": "b" * 64,
                "synthetic_passes": False,
            },
        ]
        splits = []
        for index in range(20):
            splits.append(
                {
                    "split_index": index,
                    "seed": 20260717 + index,
                    "a_pair_names": ["AA__BB"],
                    "b_pair_names": ["BB__CC"],
                    "stratum_half_counts": {"(0, 1, 2)": [1, 1]},
                    "stratum_by_pair": {
                        "AA__BB": [0, 1, 2],
                        "BB__CC": [0, 1, 2],
                    },
                    "snr_field": "candidate_left_snr",
                    "odd_stratum_extra_side": "A" if index % 2 == 0 else "B",
                    "membership_sha256": f"{index:064x}",
                }
            )
        evidence = {
            "candidate_results": candidates,
            "class_evidence": {
                selected_hash: {
                    "reference": {
                        "corrected_rows": [
                            {
                                "pair_name": "AA__BB",
                                "target_period_s": 3.0,
                                "cycle_count": -1,
                                "branch_tie": False,
                                "reference_time_s": 4.0,
                                "corrected_time_s": 4.1,
                            },
                            {
                                "pair_name": "BB__CC",
                                "target_period_s": 3.0,
                                "cycle_count": 0,
                                "branch_tie": True,
                                "reference_time_s": 5.0,
                                "corrected_time_s": 5.1,
                            },
                        ]
                    },
                    "split_plan": {
                        "base_seed": 20260717,
                        "plan_sha256": "c" * 64,
                        "splits": splits,
                    },
                    "split_half_absolute_differences_km_s": [
                        [0.01, 0.02, 0.03, 0.04] for _ in range(20)
                    ],
                    "half_stability": {
                        "accepted": True,
                        "status": "accepted",
                        "period_summaries": {
                            str(period): {
                                "period_s": period,
                                "median_absolute_difference_km_s": 0.02,
                                "p90_absolute_difference_km_s": 0.04,
                                "accepted": True,
                                "status": "accepted",
                            }
                            for period in (3.0, 3.5, 4.0, 5.0)
                        },
                    },
                    "closure": {
                        "accepted": True,
                        "status": "accepted",
                        "period_summaries": {},
                        "triplet_rows": [
                            {
                                "triplet_id": "A_B_C",
                                "period_s": 3.0,
                                "raw_closure_residual_s": 0.3,
                                "corrected_closure_residual_s": 0.1,
                                "corrected_closure_residual_cycles": 0.033,
                            }
                        ],
                    },
                }
            },
            "phase_matching_diagnostics": {
                "BENSEN_VELOCITY_CCF": {
                    "phase_convention": "BENSEN_VELOCITY_CCF",
                    "diagnostic": {
                        "status": "raw_ftan_frozen",
                        "freeze_raw_ftan": True,
                        "design_revision_required": False,
                        "raw_valid_ridge_coverage": 0.8,
                        "matched_valid_ridge_coverage": 0.81,
                    },
                },
                "LIN_NEGATIVE_DERIVATIVE_EGF": {
                    "phase_convention": "LIN_NEGATIVE_DERIVATIVE_EGF",
                    "diagnostic": {
                        "status": "raw_ftan_frozen",
                        "freeze_raw_ftan": True,
                        "design_revision_required": False,
                        "raw_valid_ridge_coverage": 0.82,
                        "matched_valid_ridge_coverage": 0.83,
                    },
                },
            },
        }
        tables = self.mod.stage_b_audit_rows(
            evidence,
            {"candidate_id": selected_id},
        )
        self.assertEqual(len(tables["candidate_grid_results"]), 2)
        self.assertTrue(tables["candidate_grid_results"][0]["selected"])
        self.assertEqual(
            tables["cycle_count_distribution"],
            [
                {
                    "period_s": 3.0,
                    "cycle_count": -1,
                    "measurement_count": 1,
                    "branch_tie_count": 0,
                },
                {
                    "period_s": 3.0,
                    "cycle_count": 0,
                    "measurement_count": 1,
                    "branch_tie_count": 1,
                },
            ],
        )
        self.assertEqual(len(tables["split_half_stability"]), 80)
        self.assertEqual(len(tables["split_half_membership"]), 40)
        self.assertEqual(
            {row["split_index"] for row in tables["split_half_membership"]},
            set(range(20)),
        )
        self.assertEqual(len(tables["phase_matching_comparison"]), 2)
        self.assertEqual(len(tables["triplet_closure"]), 1)

    def test_continuous_curve_audit_rows_keep_accepted_and_rejected_periods(self):
        measurement = SimpleNamespace(
            group_time_s=4.0,
            raw_phase_time_s=4.2,
            filtered_waveform=np.ones(21),
        )
        curve = SimpleNamespace(
            convention=self.mod.PhaseConvention.LIN_NEGATIVE_DERIVATIVE_EGF,
            periods_s=np.asarray([3.0, 3.1, 3.2]),
            instantaneous_periods_s=np.asarray([3.01, np.nan, 3.19]),
            measurement_valid=np.asarray([True, False, True]),
            measurement_statuses=("accepted", "phase_invalid", "accepted"),
            measurements=(measurement, None, measurement),
            velocity_axis_km_s=np.asarray([1.6, 1.7, 1.8]),
            ridge=SimpleNamespace(
                row_indices=np.asarray([1, 2, 0]),
                group_velocities_km_s=np.asarray([1.7, 1.8, 1.6]),
                valid=np.asarray([True, False, True]),
                quality=SimpleNamespace(
                    coverage=2.0 / 3.0,
                    max_gap=1,
                    jump_fraction=0.0,
                    boundary_fraction=2.0 / 3.0,
                    normalized_energy_integral=1.2,
                ),
            ),
            ridge_normalized_log_energy=np.asarray([0.8, 0.0, 0.7]),
            ridge_normalized_envelope_amplitude=np.asarray([0.9, 0.0, 0.8]),
            ridge_adjacent_jump_km_s=np.asarray([0.0, 0.1, 0.2]),
        )
        snr = SimpleNamespace(
            signal_peak=10.0,
            leading_noise_rms=1.0,
            trailing_noise_rms=1.25,
            leading_snr=10.0,
            trailing_snr=8.0,
        )
        left = SimpleNamespace(accepted=True, status="accepted")
        with mock.patch.object(self.mod, "compute_wang_snr", return_value=snr), mock.patch.object(
            self.mod,
            "evaluate_wang_left_qc",
            return_value=left,
        ):
            rows = self.mod.continuous_curve_audit_rows(
                pair_name="AA__BB",
                source_code="AA",
                receiver_code="BB",
                source_lon=-122.2,
                source_lat=46.1,
                receiver_lon=-122.1,
                receiver_lat=46.2,
                distance_km=12.0,
                azimuth_deg=35.0,
                curve=curve,
                time_s=np.arange(21) * 0.1,
            )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["left_qc_status"], "accepted")
        self.assertTrue(rows[0]["left_qc_accepted"])
        self.assertEqual(rows[1]["measurement_status"], "phase_invalid")
        self.assertFalse(rows[1]["left_qc_accepted"])
        self.assertIsNone(rows[1]["raw_travel_time_s"])
        self.assertEqual(rows[2]["ridge_row_index"], 0)
        self.assertTrue(rows[2]["outermost_velocity_cell"])
        required = {
            "pair_name",
            "source_code",
            "receiver_code",
            "source_lon",
            "source_lat",
            "receiver_lon",
            "receiver_lat",
            "distance_km",
            "azimuth_deg",
            "nominal_period_s",
            "instantaneous_period_s",
            "group_time_s",
            "group_velocity_km_s",
            "leading_snr",
            "trailing_snr",
            "ridge_normalized_log_energy",
            "raw_travel_time_s",
            "measurement_status",
            "left_qc_status",
        }
        self.assertTrue(required.issubset(rows[0]))

    def test_diagnostic_plotters_write_nonempty_auditable_images(self):
        reference_rows = [
            {"period_s": period, "reference_velocity_km_s": velocity}
            for period, velocity in ((3.0, 2.7), (4.0, 2.8), (5.0, 2.9))
        ]
        split_rows = [
            {
                "split_index": split,
                "period_s": period,
                "absolute_difference_km_s": 0.01 + split * 0.0001,
                "median_absolute_difference_km_s": 0.012,
                "p90_absolute_difference_km_s": 0.018,
            }
            for split in range(20)
            for period in (3.0, 4.0, 5.0)
        ]
        candidate_rows = [
            {
                "candidate_id": "lin",
                "phase_convention": "LIN_NEGATIVE_DERIVATIVE_EGF",
                "closure_median_cycles": 0.05,
                "selected": True,
            },
            {
                "candidate_id": "bensen",
                "phase_convention": "BENSEN_VELOCITY_CCF",
                "closure_median_cycles": 0.08,
                "selected": False,
            },
        ]
        phase_rows = [
            {
                "phase_convention": "LIN_NEGATIVE_DERIVATIVE_EGF",
                "raw_valid_ridge_coverage": 0.8,
                "matched_valid_ridge_coverage": 0.82,
            },
            {
                "phase_convention": "BENSEN_VELOCITY_CCF",
                "raw_valid_ridge_coverage": 0.75,
                "matched_valid_ridge_coverage": 0.74,
            },
        ]
        triplets = [
            {
                "period_s": 3.0,
                "raw_closure_residual_s": 0.3,
                "corrected_closure_residual_s": 0.1,
            },
            {
                "period_s": 4.0,
                "raw_closure_residual_s": -0.2,
                "corrected_closure_residual_s": -0.05,
            },
        ]
        periods = np.asarray([3.0, 4.0, 5.0])
        velocity = np.asarray([1.6, 2.0, 2.4, 2.8, 3.2])
        energy = np.asarray(
            [
                [0.1, 0.2, 0.9, 0.3, 0.1],
                [0.1, 0.2, 0.8, 0.4, 0.1],
                [0.1, 0.3, 0.7, 0.5, 0.1],
            ]
        )
        ridge = SimpleNamespace(
            group_velocities_km_s=np.asarray([2.4, 2.4, 2.4]),
            valid=np.asarray([True, True, True]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.mod.plot_reference_dispersion_stability(
                root / "reference.png",
                reference_rows,
                split_rows,
            )
            self.mod.plot_phase_convention_validation(
                root / "phase.png",
                candidate_rows,
                phase_rows,
            )
            self.mod.plot_triplet_closure(
                root / "closure.png",
                triplets,
            )
            audit = self.mod.plot_ftan_example(
                root / "example.png",
                pair_name="AA__BB",
                distance_km=12.0,
                periods_s=periods,
                velocity_axis_km_s=velocity,
                normalized_envelope_amplitude=energy,
                scaled_log_energy=energy,
                selected_ridge=ridge,
                beta1=1.0,
                beta2=2.0,
            )
            for name in ("reference.png", "phase.png", "closure.png", "example.png"):
                path = root / name
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)
        self.assertEqual(audit["velocity_grid_count"], 5)
        self.assertLessEqual(audit["candidate_ridge_count"], 3)
        self.assertEqual(audit["exclusion_corridor_km_s"], 0.05)
        self.assertEqual(audit["energy_scale"], [0.0, 1.0])

    def test_formal_chinese_report_separates_validation_wang_difference_and_evidence(self):
        metadata = {
            "run_status": "success",
            "stage": "C",
            "git_commit_sha": "a" * 40,
            "python_version": "3.12",
            "dependency_versions": {"numpy": "2.0"},
            "stack_root": "/server/private/stack",
            "stations_csv": "/server/private/stations.csv",
            "input_file_count": 123,
            "input_inventory_sha256": "b" * 64,
            "code_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "phase_convention": "LIN_NEGATIVE_DERIVATIVE_EGF",
            "frozen_candidate": {
                "candidate_id": "lin-a12-b1-2",
                "alpha": 12.0,
                "beta1": 1.0,
                "beta2": 2.0,
            },
            "started_at": "2026-07-19T00:00:00",
            "finished_at": "2026-07-19T01:00:00",
            "host": "work",
            "exit_status": 0,
        }
        summary = [
            {
                "period_s": 3.0,
                "initial_count": 100,
                "right_qc_count": 80,
                "fit_velocity_km_s_display": "2.70",
                "std_velocity_km_s_display": "0.12",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.html"
            self.mod.write_formal_report_html(
                output,
                metadata=metadata,
                summary_rows=summary,
                formal_validation={
                    "candidate_count": 300,
                    "split_count": 20,
                    "ftan_example_count": 3,
                },
            )
            document = output.read_text(encoding="utf-8")
        self.assertIn("方法验证结果", document)
        self.assertIn("与 Wang 的数据差异", document)
        self.assertIn("差异证据", document)
        self.assertIn("300", document)
        self.assertIn("20", document)
        self.assertIn("DisperPicker", document)
        self.assertNotIn('src="/server/', document)
        self.assertIn(
            'src="figures/wang_figure4_ftan_paper_scale.png"',
            document,
        )


if __name__ == "__main__":
    unittest.main()
