import importlib.util
import inspect
import json
import multiprocessing as mp
import os
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "04_dispersion"
    / "wang_ftan_validation.py"
)
BENSEN_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "04_dispersion"
    / "bensen_phase_ftan.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "wang_ftan_validation",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_bensen_module():
    spec = importlib.util.spec_from_file_location(
        "bensen_phase_ftan_for_validation_test",
        BENSEN_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load module spec from {BENSEN_MODULE_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WangFtanValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()
        cls.bensen = load_bensen_module()

    def test_stage_b_stratified_selection_is_deterministic_and_adds_closure_edges(
        self,
    ):
        rows = [
            {
                "pair_name": f"S{index:04d}__R{index:04d}",
                "distance_km": 8.0 + (index % 100) * 0.2,
                "azimuth_deg": float((index * 17) % 360),
                "preliminary_snr": (
                    np.nan if index % 97 == 0 else 1.0 + (index % 30)
                ),
                "disperpicker_velocity_km_s": 99.0,
                "paths_csv_row": index,
            }
            for index in range(2200)
        ]
        baseline = self.mod.select_stage_b_pairs(
            rows,
            max_random_pairs=2000,
            seed=20260717,
        )
        unselected = sorted(
            set(row["pair_name"] for row in rows)
            - set(baseline.random_pair_names)
        )
        closure_edges = tuple(unselected[:7])

        first = self.mod.select_stage_b_pairs(
            rows,
            closure_edge_pair_names=closure_edges,
            max_random_pairs=2000,
            seed=20260717,
        )
        second = self.mod.select_stage_b_pairs(
            list(reversed(rows)),
            closure_edge_pair_names=reversed(closure_edges),
            max_random_pairs=2000,
            seed=20260717,
        )

        self.assertEqual(len(first.random_pair_names), 2000)
        self.assertEqual(first.random_pair_names, second.random_pair_names)
        self.assertEqual(first.selected_pair_names, second.selected_pair_names)
        self.assertEqual(first.membership_sha256, second.membership_sha256)
        self.assertEqual(len(first.membership_sha256), 64)
        self.assertTrue(
            set(closure_edges).issubset(first.selected_pair_names)
        )
        self.assertEqual(len(first.selected_pair_names), 2007)
        self.assertEqual(first.seed, 20260717)
        self.assertEqual(first.max_random_pairs, 2000)
        self.assertEqual(first.distance_quantile_probabilities, (0.2, 0.4, 0.6, 0.8))
        self.assertEqual(first.azimuth_sector_width_deg, 45.0)
        self.assertEqual(first.snr_quantile_probabilities, (1.0 / 3.0, 2.0 / 3.0))
        self.assertFalse(first.distance_quintile_edges_km.flags.writeable)
        self.assertFalse(first.snr_tertile_edges.flags.writeable)
        self.assertEqual(
            sum(first.stratum_random_counts.values()),
            2000,
        )

        changed_legacy = [
            {
                **row,
                "disperpicker_velocity_km_s": -999.0,
                "paths_csv_row": "changed",
            }
            for row in rows
        ]
        unchanged = self.mod.select_stage_b_pairs(
            changed_legacy,
            closure_edge_pair_names=closure_edges,
            max_random_pairs=2000,
            seed=20260717,
        )
        self.assertEqual(
            first.membership_sha256,
            unchanged.membership_sha256,
        )
        source = inspect.getsource(self.mod.select_stage_b_pairs)
        self.assertIn('row["preliminary_snr"]', source)
        self.assertNotIn("DisperPicker", MODULE_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("paths.csv", MODULE_PATH.read_text(encoding="utf-8"))

    def test_triplet_geometry_requires_between_cross_track_and_distance_closure(
        self,
    ):
        accepted = self.mod.evaluate_triplet_geometry(
            station_a=(0.0, 0.0),
            station_b=(0.1, 0.003),
            station_c=(0.2, 0.0),
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.status, "accepted")
        self.assertGreaterEqual(accepted.projection_fraction, 0.0)
        self.assertLessEqual(accepted.projection_fraction, 1.0)
        self.assertLessEqual(accepted.cross_track_km, 0.5)
        self.assertLessEqual(
            abs(
                accepted.distance_ab_km
                + accepted.distance_bc_km
                - accepted.distance_ac_km
            ),
            0.5,
        )

        outside = self.mod.evaluate_triplet_geometry(
            station_a=(0.0, 0.0),
            station_b=(0.3, 0.0),
            station_c=(0.2, 0.0),
        )
        self.assertFalse(outside.accepted)
        self.assertEqual(outside.status, "station_b_outside_ac")

        cross_track = self.mod.evaluate_triplet_geometry(
            station_a=(0.0, 0.0),
            station_b=(0.1, 0.01),
            station_c=(0.2, 0.0),
        )
        self.assertFalse(cross_track.accepted)
        self.assertEqual(cross_track.status, "cross_track_too_large")

    def test_triplet_closure_uses_corrected_times_and_enforces_support_metrics(
        self,
    ):
        rows = []
        for period_s in (3.0, 3.5, 4.0, 5.0):
            for index in range(100):
                perturbation = 0.01 * np.sin(index)
                rows.append(
                    {
                        "triplet_id": f"T{index:03d}",
                        "period_s": period_s,
                        "distance_ab_km": 10.0,
                        "distance_bc_km": 10.0,
                        "distance_ac_km": 20.0,
                        "raw_time_ab_s": 1.0,
                        "raw_time_bc_s": 4.0,
                        "raw_time_ac_s": 8.0,
                        "corrected_time_ab_s": 4.0 + perturbation,
                        "corrected_time_bc_s": 4.0 - perturbation,
                        "corrected_time_ac_s": 8.0,
                        "left_ab": True,
                        "left_bc": True,
                        "left_ac": True,
                        "snr_ab": 9.0,
                        "snr_bc": 9.5,
                        "snr_ac": 10.0,
                    }
                )

        result = self.mod.evaluate_triplet_closure(
            rows,
            target_periods_s=(3.0, 3.5, 4.0, 5.0),
            minimum_support=100,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(
            tuple(result.period_summaries),
            (3.0, 3.5, 4.0, 5.0),
        )
        for period_s, summary in result.period_summaries.items():
            self.assertEqual(summary.support_count, 100)
            self.assertLessEqual(summary.median_absolute_cycles, 0.15)
            self.assertLessEqual(summary.absolute_bias_cycles, 0.05)
            self.assertTrue(summary.accepted)
            period_rows = [
                row
                for row in result.triplet_rows
                if row["period_s"] == period_s
            ]
            self.assertEqual(len(period_rows), 100)
            self.assertTrue(
                all(
                    abs(row["corrected_closure_residual_s"]) < 1e-12
                    for row in period_rows
                )
            )
            self.assertTrue(
                all(
                    row["raw_closure_residual_s"] == -3.0
                    for row in period_rows
                )
            )

        insufficient = self.mod.evaluate_triplet_closure(
            rows[:-1],
            target_periods_s=(3.0, 3.5, 4.0, 5.0),
            minimum_support=100,
        )
        self.assertFalse(insufficient.accepted)
        self.assertEqual(
            insufficient.period_summaries[5.0].status,
            "insufficient_triplet_support",
        )

        nonfinite = [
            {
                **row,
                "corrected_time_ac_s": np.nan,
            }
            for row in rows
        ]
        nonfinite_result = self.mod.evaluate_triplet_closure(
            nonfinite,
            target_periods_s=(3.0, 3.5, 4.0, 5.0),
            minimum_support=100,
        )
        self.assertFalse(nonfinite_result.accepted)
        self.assertTrue(
            all(
                summary.support_count == 0
                for summary in nonfinite_result.period_summaries.values()
            )
        )

        duplicated = [
            {
                **rows[0],
                "triplet_id": "DUPLICATE",
            }
            for _ in range(100)
        ]
        with self.assertRaisesRegex(ValueError, "duplicate triplet"):
            self.mod.evaluate_triplet_closure(
                duplicated,
                target_periods_s=(3.0,),
                minimum_support=100,
            )

    def test_twenty_half_sample_splits_are_stratified_hashed_and_reproducible(
        self,
    ):
        rows = [
            {
                "pair_name": f"AA{index:03d}__BB{index:03d}",
                "distance_km": 8.0 + (index % 25),
                "azimuth_deg": float((index * 31) % 360),
                "candidate_left_snr": 2.0 + (index % 11),
            }
            for index in range(257)
        ]
        first = self.mod.build_half_sample_splits(
            rows,
            split_count=20,
            base_seed=20260717,
        )
        second = self.mod.build_half_sample_splits(
            reversed(rows),
            split_count=20,
            base_seed=20260717,
        )

        self.assertEqual(first.plan_sha256, second.plan_sha256)
        self.assertEqual(len(first.splits), 20)
        all_names = {row["pair_name"] for row in rows}
        for index, split in enumerate(first.splits):
            self.assertEqual(split.seed, 20260717 + index)
            self.assertTrue(set(split.a_pair_names).isdisjoint(split.b_pair_names))
            self.assertEqual(
                set(split.a_pair_names).union(split.b_pair_names),
                all_names,
            )
            self.assertEqual(len(split.membership_sha256), 64)
            self.assertEqual(set(split.stratum_by_pair), all_names)
            self.assertEqual(split.snr_field, "candidate_left_snr")
            for key, counts in split.stratum_half_counts.items():
                self.assertLessEqual(abs(counts[0] - counts[1]), 1)
                if sum(counts) % 2:
                    expected_extra_side = "A" if index % 2 == 0 else "B"
                    self.assertEqual(
                        split.odd_stratum_extra_side,
                        expected_extra_side,
                    )

    def test_half_sample_stability_and_left_table_equivalence_hash(self):
        target_periods = (3.0, 3.5, 4.0, 5.0)
        differences = np.tile(
            np.asarray([0.02, 0.025, 0.01, 0.03]),
            (20, 1),
        )
        passed = self.mod.evaluate_half_sample_stability(
            differences,
            target_periods_s=target_periods,
        )
        self.assertTrue(passed.accepted)
        self.assertTrue(
            all(summary.accepted for summary in passed.period_summaries.values())
        )

        failed_differences = differences.copy()
        failed_differences[:, 2] = np.linspace(0.01, 0.08, 20)
        failed = self.mod.evaluate_half_sample_stability(
            failed_differences,
            target_periods_s=target_periods,
        )
        self.assertFalse(failed.accepted)
        self.assertFalse(failed.period_summaries[4.0].accepted)

        base_row = {
            "pair_name": "AA__BB",
            "T_inst": 3.1,
            "t0": 8.2,
            "U": 2.4,
            "signal_peak": 5.0,
            "leading_rms": 0.2,
            "trailing_rms": 0.3,
            "ridge_fields": {
                "energy": 0.8,
                "amplitude": 0.9,
                "jump": 0.02,
            },
            "legacy_unused": 999,
        }
        first_hash = self.mod.hash_left_observation_table([base_row])
        repeated_hash = self.mod.hash_left_observation_table(
            [{**base_row, "legacy_unused": -999}]
        )
        changed_hash = self.mod.hash_left_observation_table(
            [{**base_row, "t0": 8.3}]
        )
        self.assertEqual(first_hash, repeated_hash)
        self.assertNotEqual(first_hash, changed_hash)
        self.assertEqual(len(first_hash), 64)
        numpy_row = {
            **base_row,
            "T_inst": np.float64(3.1),
            "ridge_fields": {
                "energy": np.asarray([0.8, 0.9], dtype=np.float64),
                "valid": np.asarray([True, False]),
            },
        }
        self.assertEqual(
            self.mod.hash_left_observation_table([numpy_row]),
            self.mod.hash_left_observation_table(
                [
                    {
                        **numpy_row,
                        "ridge_fields": {
                            "energy": np.asarray(
                                [0.8, 0.9],
                                dtype=np.float64,
                            ),
                            "valid": np.asarray([True, False]),
                        },
                    }
                ]
            ),
        )
        positive_zero = self.mod.hash_left_observation_table(
            [{**base_row, "t0": 0.0}]
        )
        negative_zero = self.mod.hash_left_observation_table(
            [{**base_row, "t0": -0.0}]
        )
        self.assertNotEqual(positive_zero, negative_zero)

    def test_candidate_boundary_fraction_is_a_separate_five_percent_gate(self):
        exact = self.mod.evaluate_candidate_boundary_fraction(
            accepted_measurement_count=100,
            accepted_outermost_velocity_cell_count=5,
        )
        self.assertTrue(exact.accepted)
        self.assertEqual(exact.status, "accepted")
        self.assertEqual(exact.accepted_boundary_fraction, 0.05)

        failed = self.mod.evaluate_candidate_boundary_fraction(
            accepted_measurement_count=100,
            accepted_outermost_velocity_cell_count=6,
        )
        self.assertFalse(failed.accepted)
        self.assertEqual(failed.status, "candidate_boundary_fraction_exceeded")
        self.assertEqual(failed.accepted_boundary_fraction, 0.06)

        with self.assertRaisesRegex(ValueError, "measurement count"):
            self.mod.evaluate_candidate_boundary_fraction(
                accepted_measurement_count=0,
                accepted_outermost_velocity_cell_count=0,
            )
        with self.assertRaisesRegex(ValueError, "outermost"):
            self.mod.evaluate_candidate_boundary_fraction(
                accepted_measurement_count=10,
                accepted_outermost_velocity_cell_count=11,
            )

    def test_stage_b_candidate_input_integrity_enforces_one_percent_exceptions(self):
        check = self.mod.stage_b_candidate_input_integrity_passes
        self.assertTrue(
            check(
                {
                    "processed_pair_count": 100,
                    "successful_pair_count": 90,
                    "expected_scientific_rejection_count": 9,
                    "unexpected_pair_exception_count": 1,
                },
                selected_pair_count=100,
            )
        )
        self.assertFalse(
            check(
                {
                    "processed_pair_count": 100,
                    "successful_pair_count": 90,
                    "expected_scientific_rejection_count": 8,
                    "unexpected_pair_exception_count": 2,
                },
                selected_pair_count=100,
            )
        )
        with self.assertRaisesRegex(ValueError, "conservation"):
            check(
                {
                    "processed_pair_count": 100,
                    "successful_pair_count": 90,
                    "expected_scientific_rejection_count": 9,
                    "unexpected_pair_exception_count": 2,
                },
                selected_pair_count=100,
            )

    def test_phase_matching_is_diagnostic_and_cannot_silently_replace_raw_ftan(
        self,
    ):
        raw = {3.0: 0.20, 3.5: 0.18, 4.0: 0.16, 5.0: 0.14}
        improved = {period: value * 0.89 for period, value in raw.items()}
        revision = self.mod.evaluate_phase_matching_diagnostic(
            raw_closure_median_cycles=raw,
            matched_closure_median_cycles=improved,
            raw_valid_ridge_coverage=0.72,
            matched_valid_ridge_coverage=0.80,
            raw_phase_convention="LIN_NEGATIVE_DERIVATIVE_EGF",
            matched_phase_convention="LIN_NEGATIVE_DERIVATIVE_EGF",
            raw_boundary_fraction=0.03,
            matched_boundary_fraction=0.03,
            narrowband_sidelobe_validation_passed=True,
        )
        self.assertTrue(revision.design_revision_required)
        self.assertFalse(revision.freeze_raw_ftan)
        self.assertEqual(
            revision.status,
            "phase_matching_design_revision_required",
        )
        self.assertTrue(all(revision.period_reduction_passes.values()))

        narrowband_sidelobe = self.mod.evaluate_phase_matching_diagnostic(
            raw_closure_median_cycles=raw,
            matched_closure_median_cycles=improved,
            raw_valid_ridge_coverage=0.72,
            matched_valid_ridge_coverage=0.80,
            raw_phase_convention="LIN_NEGATIVE_DERIVATIVE_EGF",
            matched_phase_convention="LIN_NEGATIVE_DERIVATIVE_EGF",
            raw_boundary_fraction=0.03,
            matched_boundary_fraction=0.03,
            narrowband_sidelobe_validation_passed=False,
        )
        self.assertFalse(narrowband_sidelobe.design_revision_required)
        self.assertTrue(narrowband_sidelobe.freeze_raw_ftan)
        self.assertEqual(narrowband_sidelobe.status, "raw_ftan_frozen")
        self.assertTrue(
            all(narrowband_sidelobe.period_reduction_passes.values())
        )
        self.assertFalse(
            narrowband_sidelobe.narrowband_sidelobe_validation_passed
        )

        convention_changed = self.mod.evaluate_phase_matching_diagnostic(
            raw_closure_median_cycles=raw,
            matched_closure_median_cycles=improved,
            raw_valid_ridge_coverage=0.72,
            matched_valid_ridge_coverage=0.80,
            raw_phase_convention="BENSEN_VELOCITY_CCF",
            matched_phase_convention="LIN_NEGATIVE_DERIVATIVE_EGF",
            raw_boundary_fraction=0.03,
            matched_boundary_fraction=0.02,
            narrowband_sidelobe_validation_passed=True,
        )
        self.assertTrue(convention_changed.freeze_raw_ftan)
        self.assertFalse(convention_changed.phase_convention_unchanged)
        with self.assertRaisesRegex(ValueError, "four target periods"):
            self.mod.evaluate_phase_matching_diagnostic(
                raw_closure_median_cycles={3.0: 0.2},
                matched_closure_median_cycles={3.0: 0.1},
                raw_valid_ridge_coverage=0.72,
                matched_valid_ridge_coverage=0.80,
                raw_phase_convention="LIN",
                matched_phase_convention="LIN",
                raw_boundary_fraction=0.03,
                matched_boundary_fraction=0.02,
                narrowband_sidelobe_validation_passed=True,
            )

    def test_candidate_grid_and_freeze_decision_apply_gates_and_tie_breaks(self):
        grid = self.mod.build_candidate_grid(
            phase_conventions=(
                "BENSEN_VELOCITY_CCF",
                "LIN_NEGATIVE_DERIVATIVE_EGF",
            ),
            alpha_candidates=(5, 8, 12, 16, 20, 25),
            beta1_candidates=(0, 0.5, 1, 2, 4),
            beta2_candidates=(0, 1, 2, 4, 8),
        )
        self.assertEqual(len(grid), 300)
        self.assertEqual(len({row["candidate_id"] for row in grid}), 300)

        gate_fields = {
            "synthetic_passes": True,
            "ridge_passes": True,
            "instantaneous_period_passes": True,
            "alias_passes": True,
            "triplet_passes": True,
            "half_sample_passes": True,
            "boundary_passes": True,
        }
        rows = [
            {
                "candidate_id": "BENSEN_worse",
                "phase_convention": "BENSEN",
                "alpha": 12.0,
                "beta1": 1.0,
                "beta2": 2.0,
                "closure_median_cycles": 0.100,
                **gate_fields,
            },
            {
                "candidate_id": "BENSEN_simple",
                "phase_convention": "BENSEN",
                "alpha": 8.0,
                "beta1": 0.5,
                "beta2": 0.0,
                "closure_median_cycles": 0.104,
                **gate_fields,
            },
            {
                "candidate_id": "LIN_best",
                "phase_convention": "LIN",
                "alpha": 5.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "closure_median_cycles": 0.130,
                **gate_fields,
            },
            {
                "candidate_id": "rejected",
                "phase_convention": "LIN",
                "alpha": 5.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "closure_median_cycles": 0.001,
                **{**gate_fields, "triplet_passes": False},
            },
        ]
        decision = self.mod.freeze_ftan_candidate(
            rows,
            lineage_status="unknown",
            lineage_preferred_phase_convention=None,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.status, "passed")
        self.assertEqual(
            decision.selected_candidate["candidate_id"],
            "BENSEN_simple",
        )
        self.assertNotIn("rejected", decision.eligible_candidate_ids)

    def test_phase_convention_within_five_percent_requires_confirmed_lineage(self):
        gates = {
            "synthetic_passes": True,
            "ridge_passes": True,
            "instantaneous_period_passes": True,
            "alias_passes": True,
            "triplet_passes": True,
            "half_sample_passes": True,
            "boundary_passes": True,
        }
        rows = [
            {
                "candidate_id": "BENSEN",
                "phase_convention": "BENSEN",
                "alpha": 5.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "closure_median_cycles": 0.100,
                **gates,
            },
            {
                "candidate_id": "LIN",
                "phase_convention": "LIN",
                "alpha": 5.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "closure_median_cycles": 0.104,
                **gates,
            },
        ]
        unknown = self.mod.freeze_ftan_candidate(
            rows,
            lineage_status="unknown",
            lineage_preferred_phase_convention=None,
        )
        self.assertFalse(unknown.accepted)
        self.assertEqual(unknown.status, "phase_convention_unidentifiable")

        confirmed = self.mod.freeze_ftan_candidate(
            rows,
            lineage_status="confirmed",
            lineage_preferred_phase_convention="LIN",
        )
        self.assertTrue(confirmed.accepted)
        self.assertEqual(confirmed.selected_candidate["candidate_id"], "LIN")

        exact_boundary = [
            rows[0],
            {
                **rows[1],
                "closure_median_cycles": 0.105,
            },
        ]
        exact = self.mod.freeze_ftan_candidate(
            exact_boundary,
            lineage_status="unknown",
            lineage_preferred_phase_convention=None,
        )
        self.assertTrue(exact.accepted)
        self.assertEqual(exact.selected_candidate["candidate_id"], "BENSEN")

    def test_frozen_manifest_requires_success_and_complete_hash_lineage(self):
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
        decision = self.mod.freeze_ftan_candidate(
            [
                {
                    "candidate_id": "only",
                    "phase_convention": "LIN",
                    "alpha": 12.0,
                    "beta1": 1.0,
                    "beta2": 2.0,
                    "closure_median_cycles": 0.08,
                    **gates,
                }
            ],
            lineage_status="confirmed",
            lineage_preferred_phase_convention="LIN",
        )
        hashes = {
            "input_inventory_sha256": "a" * 64,
            "code_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "validation_table_sha256": "d" * 64,
        }
        manifest = self.mod.build_frozen_parameters_manifest(
            decision,
            **hashes,
        )
        self.assertEqual(manifest["stage_b_status"], "passed")
        self.assertEqual(manifest["candidate_id"], "only")
        self.assertEqual(manifest["lineage_status"], "confirmed")
        self.assertEqual(
            manifest["lineage_preferred_phase_convention"],
            "LIN",
        )
        for key, value in hashes.items():
            self.assertEqual(manifest[key], value)

        failed = self.mod.freeze_ftan_candidate(
            [{**decision.selected_candidate, "triplet_passes": False}],
            lineage_status="confirmed",
            lineage_preferred_phase_convention="LIN",
        )
        self.assertFalse(failed.accepted)
        with self.assertRaisesRegex(ValueError, "passed decision"):
            self.mod.build_frozen_parameters_manifest(failed, **hashes)

    def test_stage_b_budget_uses_full_workload_and_memory_hard_gates(self):
        passed = self.mod.evaluate_stage_b_budget(
            candidate_benchmark_elapsed_s=20.0,
            candidate_benchmark_work_units=100.0,
            stage_b_candidate_work_units=1000.0,
            reference_benchmark_elapsed_s=10.0,
            reference_benchmark_optimizer_calls=10,
            distinct_measurement_class_count=2,
            worker_count=2,
            measured_peak_memory_bytes=2_000_000_000,
            available_memory_bytes=10_000_000_000,
        )
        self.assertTrue(passed.accepted)
        self.assertEqual(passed.status, "accepted")
        self.assertEqual(passed.optimizer_calls_per_class, 953)
        self.assertIn("953", passed.reference_projection_formula)
        atomic_waves = self.mod.evaluate_stage_b_budget(
            candidate_benchmark_elapsed_s=1.0,
            candidate_benchmark_work_units=1.0,
            stage_b_candidate_work_units=1.0,
            reference_benchmark_elapsed_s=1.0,
            reference_benchmark_optimizer_calls=953,
            distinct_measurement_class_count=25,
            worker_count=24,
            measured_peak_memory_bytes=1,
            available_memory_bytes=10,
        )
        self.assertEqual(atomic_waves.projected_reference_seconds, 2.0)
        self.assertIn("ceil", atomic_waves.reference_projection_formula)

        time_failed = self.mod.evaluate_stage_b_budget(
            candidate_benchmark_elapsed_s=86_400.0,
            candidate_benchmark_work_units=1.0,
            stage_b_candidate_work_units=2.0,
            reference_benchmark_elapsed_s=10.0,
            reference_benchmark_optimizer_calls=10,
            distinct_measurement_class_count=1,
            worker_count=1,
            measured_peak_memory_bytes=1,
            available_memory_bytes=10,
        )
        self.assertFalse(time_failed.accepted)
        self.assertEqual(time_failed.status, "stage_b_budget_exceeded")

        memory_failed = self.mod.evaluate_stage_b_budget(
            candidate_benchmark_elapsed_s=1.0,
            candidate_benchmark_work_units=1.0,
            stage_b_candidate_work_units=1.0,
            reference_benchmark_elapsed_s=1.0,
            reference_benchmark_optimizer_calls=10,
            distinct_measurement_class_count=1,
            worker_count=1,
            measured_peak_memory_bytes=8,
            available_memory_bytes=10,
        )
        self.assertFalse(memory_failed.accepted)
        self.assertGreater(memory_failed.memory_fraction, 0.70)

        with self.assertRaisesRegex(ValueError, "classes"):
            self.mod.evaluate_stage_b_budget(
                candidate_benchmark_elapsed_s=1.0,
                candidate_benchmark_work_units=1.0,
                stage_b_candidate_work_units=1.0,
                reference_benchmark_elapsed_s=1.0,
                reference_benchmark_optimizer_calls=10,
                distinct_measurement_class_count=301,
                worker_count=24,
                measured_peak_memory_bytes=1,
                available_memory_bytes=10,
            )

    def test_measurement_class_jobs_use_bounded_independent_processes(self):
        parent_pid = os.getpid()

        def evaluator(value):
            checksum = 0
            for index in range(200_000):
                checksum = (checksum + value + index) % 1_000_003
            return {
                "value": value,
                "pid": os.getpid(),
                "checksum": checksum,
            }

        rows = self.mod.execute_measurement_class_processes(
            tuple(range(8)),
            evaluator=evaluator,
            max_workers=2,
        )
        worker_pids = {row["pid"] for row in rows}
        self.assertEqual([row["value"] for row in rows], list(range(8)))
        self.assertNotIn(parent_pid, worker_pids)
        self.assertEqual(len(worker_pids), 2)
        with self.assertRaisesRegex(ValueError, r"\[1, 24\]"):
            self.mod.execute_measurement_class_processes(
                (1,),
                evaluator=evaluator,
                max_workers=25,
            )

    def test_stage_b_orchestrator_stops_before_science_when_budget_fails(self):
        grid = self.mod.build_candidate_grid(
            phase_conventions=(
                "BENSEN_VELOCITY_CCF",
                "LIN_NEGATIVE_DERIVATIVE_EGF",
            ),
            alpha_candidates=(5, 8, 12, 16, 20, 25),
            beta1_candidates=(0, 0.5, 1, 2, 4),
            beta2_candidates=(0, 1, 2, 4, 8),
        )
        events = []

        def benchmark_stage_b(**kwargs):
            events.append(("benchmark", kwargs))
            return self.mod.StageBBenchmarkEvidence(
                candidate_grid_elapsed_s=86_400.0,
                ten_single_reference_fits_elapsed_s=86_400.0,
                lambda_cv_elapsed_s=86_400.0,
                twenty_half_samples_elapsed_s=86_400.0,
                measured_peak_memory_bytes=1,
                available_memory_bytes=10,
                cache_hit_fraction=0.0,
                benchmark_input_sha256="f" * 64,
            )

        def forbidden(*args, **kwargs):
            self.fail("scientific callback ran after budget rejection")

        result = self.mod.run_stage_b_validation(
            inventory_rows=[
                {
                    "pair_name": "AA__BB",
                    "distance_km": 10.0,
                    "azimuth_deg": 0.0,
                    "preliminary_snr": 9.0,
                }
            ],
            station_coordinates={
                "AA": (0.0, 0.0),
                "BB": (0.1, 0.0),
            },
            closure_triplets=(),
            closure_edge_pair_names=(),
            candidate_grid=grid,
            lineage_status="unknown",
            lineage_preferred_phase_convention=None,
            benchmark_stage_b=benchmark_stage_b,
            measure_candidate=forbidden,
            fit_full_reference=forbidden,
            fit_split_half_reference=forbidden,
            run_phase_matching=forbidden,
            input_inventory_sha256="a" * 64,
            code_sha256="b" * 64,
            config_sha256="c" * 64,
            max_workers=24,
        )
        self.assertEqual(result.return_code, 2)
        self.assertEqual(result.status, "stage_b_budget_exceeded")
        self.assertIsNone(result.frozen_parameters)
        self.assertEqual(events[0][0], "benchmark")
        self.assertEqual(events[0][1]["candidate_count"], 300)
        self.assertEqual(events[0][1]["synthetic_waveform_count"], 20)
        self.assertEqual(events[0][1]["half_start_count"], 5)

    def test_stage_b_orchestrator_runs_one_reference_per_bitwise_class(self):
        grid = self.mod.build_candidate_grid(
            phase_conventions=(
                "BENSEN_VELOCITY_CCF",
                "LIN_NEGATIVE_DERIVATIVE_EGF",
            ),
            alpha_candidates=(5, 8, 12, 16, 20, 25),
            beta1_candidates=(0, 0.5, 1, 2, 4),
            beta2_candidates=(0, 1, 2, 4, 8),
        )
        inventory = []
        station_coordinates = {}
        triplets = []
        left_rows = []
        for index in range(100):
            pair_ab = f"A{index:03d}__B{index:03d}"
            pair_bc = f"B{index:03d}__C{index:03d}"
            pair_ac = f"A{index:03d}__C{index:03d}"
            station_a = f"A{index:03d}"
            station_b = f"B{index:03d}"
            station_c = f"C{index:03d}"
            latitude = index * 0.01
            station_coordinates[station_a] = (0.0, latitude)
            station_coordinates[station_b] = (0.09, latitude)
            station_coordinates[station_c] = (0.18, latitude)
            triplets.append(
                {
                    "triplet_id": f"T{index:03d}",
                    "station_a_code": station_a,
                    "station_b_code": station_b,
                    "station_c_code": station_c,
                    "pair_ab_name": pair_ab,
                    "pair_bc_name": pair_bc,
                    "pair_ac_name": pair_ac,
                }
            )
            for pair_name, distance_km in (
                (pair_ab, 10.0),
                (pair_bc, 10.0),
                (pair_ac, 20.0),
            ):
                inventory.append(
                    {
                        "pair_name": pair_name,
                        "distance_km": distance_km,
                        "azimuth_deg": float(index % 8) * 45.0,
                        "preliminary_snr": 10.0,
                    }
                )
                for period_s in (3.0, 3.5, 4.0, 5.0):
                    travel_time = distance_km / 2.5
                    left_rows.append(
                        {
                            "pair_name": pair_name,
                            "target_period_s": period_s,
                            "T_inst": period_s,
                            "t0": travel_time,
                            "U": 2.5,
                            "signal_peak": 2.0,
                            "leading_rms": 0.2,
                            "trailing_rms": 0.2,
                            "ridge_fields": {
                                "energy": 0.9,
                                "amplitude": 0.8,
                                "jump": 0.01,
                            },
                            "ridge_valid": True,
                            "instantaneous_period_valid": True,
                            "distance_km": distance_km,
                            "azimuth_deg": float(index % 8) * 45.0,
                            "leading_snr": 10.0,
                            "trailing_snr": 10.0,
                        }
                    )
        inventory = list(
            {
                row["pair_name"]: row for row in inventory
            }.values()
        )
        left_rows = tuple(left_rows)
        manager = mp.Manager()
        self.addCleanup(manager.shutdown)
        events = manager.list()

        def benchmark_stage_b(**kwargs):
            events.append("benchmark")
            return self.mod.StageBBenchmarkEvidence(
                candidate_grid_elapsed_s=1.0,
                ten_single_reference_fits_elapsed_s=1.0,
                lambda_cv_elapsed_s=1.0,
                twenty_half_samples_elapsed_s=1.0,
                measured_peak_memory_bytes=1,
                available_memory_bytes=10,
                cache_hit_fraction=0.5,
                benchmark_input_sha256="f" * 64,
            )

        def measure_candidate(**kwargs):
            events.append("measure")
            rows = left_rows
            if kwargs["candidate"]["candidate_id"] == grid[-1][
                "candidate_id"
            ]:
                changed = [dict(row) for row in left_rows]
                changed[0] = {
                    **changed[0],
                    "t0": np.nextafter(changed[0]["t0"], np.inf),
                }
                rows = tuple(changed)
            return {
                "synthetic_validation_status": "accepted",
                "continuous_left_rows": rows,
                "accepted_outermost_velocity_cell_count": 0,
            }

        def fit_full_reference(**kwargs):
            events.append("full_reference")
            self.assertNotIn("closure", kwargs)
            self.assertEqual(kwargs["maximum_optimizer_calls"], 753)
            class_left_rows = kwargs["left_rows"]
            return self.mod.FullReferenceEvidence(
                status="accepted",
                alias_status="accepted",
                lambda_s=1.0,
                lambda_g=1.0,
                basin_starts=tuple(range(5)),
                optimizer_calls=753,
                corrected_rows=tuple(
                    self.mod.CorrectedTargetObservation(
                        pair_name=row["pair_name"],
                        target_period_s=row["target_period_s"],
                        raw_time_s=row["t0"],
                        cycle_count=0,
                        corrected_time_s=row["t0"],
                        reference_time_s=row["t0"],
                        reference_residual_s=0.0,
                        leading_snr=10.0,
                        trailing_snr=10.0,
                        left_qc_accepted=True,
                    )
                    for row in class_left_rows
                ),
                result_sha256="e" * 64,
            )

        def fit_split_half_reference(**kwargs):
            events.append(f"half_{kwargs['split_index']}_{kwargs['side']}")
            self.assertEqual(kwargs["maxiter"], 300)
            self.assertEqual(kwargs["lambda_s"], 1.0)
            self.assertEqual(kwargs["lambda_g"], 1.0)
            self.assertEqual(kwargs["basin_starts"], tuple(range(5)))
            return self.mod.SplitHalfFitEvidence(
                status="accepted",
                target_velocities_km_s=[2.5, 2.5, 2.5, 2.5],
                lambda_s=1.0,
                lambda_g=1.0,
                optimizer_calls=5,
                cv_optimizer_calls=0,
                maxiter=300,
            )

        def run_phase_matching(**kwargs):
            events.append("phase_matching")
            raw = {period: 0.1 for period in (3.0, 3.5, 4.0, 5.0)}
            candidate = kwargs["candidate"]
            periods_s = np.arange(2.5, 5.0 + 0.025, 0.05)
            dt_s = 0.05
            time_s = np.arange(4096) * dt_s
            waveform = np.exp(
                -0.5 * ((time_s - 45.0) / 8.0) ** 2
            ) * np.cos(2.0 * np.pi * time_s / 3.5)
            second_pass = tuple(
                kwargs["execute_second_pass_ftan"](
                    waveform=waveform,
                    dt_s=dt_s,
                    periods_s=periods_s,
                    group_travel_times_s=np.full(periods_s.shape, 45.0),
                    first_pass_alpha=candidate["alpha"],
                )
                for _ in range(2)
            )
            diagnostic = self.mod.evaluate_phase_matching_diagnostic(
                raw_closure_median_cycles=raw,
                matched_closure_median_cycles=raw,
                raw_valid_ridge_coverage=0.8,
                matched_valid_ridge_coverage=0.8,
                raw_phase_convention=candidate["phase_convention"],
                matched_phase_convention=candidate["phase_convention"],
                raw_boundary_fraction=0.0,
                matched_boundary_fraction=0.0,
                narrowband_sidelobe_validation_passed=True,
            )
            return self.mod.PhaseMatchingRunEvidence(
                candidate_id=candidate["candidate_id"],
                phase_convention=candidate["phase_convention"],
                first_pass_alpha=candidate["alpha"],
                second_pass_alpha=min(2.0 * candidate["alpha"], 50.0),
                cut_half_width_s=10.0,
                cut_taper_alpha=0.25,
                second_pass_ftan_executed=True,
                raw_output_sha256="1" * 64,
                matched_output_sha256=(
                    self.mod.hash_phase_matching_execution_hashes(
                        tuple(
                            self.mod.hash_phase_matching_second_pass_output(
                                row
                            )
                            for row in second_pass
                        )
                    )
                ),
                raw_closure_median_cycles=raw,
                matched_closure_median_cycles=raw,
                raw_valid_ridge_coverage=0.8,
                matched_valid_ridge_coverage=0.8,
                raw_boundary_fraction=0.0,
                matched_boundary_fraction=0.0,
                diagnostic=diagnostic,
            )

        result = self.mod.run_stage_b_validation(
            inventory_rows=inventory,
            station_coordinates=station_coordinates,
            closure_triplets=triplets,
            closure_edge_pair_names=[
                row["pair_name"] for row in inventory
            ],
            candidate_grid=grid,
            lineage_status="confirmed",
            lineage_preferred_phase_convention=(
                "LIN_NEGATIVE_DERIVATIVE_EGF"
            ),
            benchmark_stage_b=benchmark_stage_b,
            measure_candidate=measure_candidate,
            fit_full_reference=fit_full_reference,
            fit_split_half_reference=fit_split_half_reference,
            run_phase_matching=run_phase_matching,
            input_inventory_sha256="a" * 64,
            code_sha256="b" * 64,
            config_sha256="c" * 64,
            max_workers=24,
            phase_matched_second_pass_ftan=(
                self.bensen.phase_matched_second_pass_ftan
            ),
        )
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.status, "passed")
        self.assertEqual(len(result.candidate_results), 300)
        self.assertEqual(len(result.measurement_classes), 2)
        self.assertEqual(events.count("full_reference"), 2)
        self.assertEqual(
            len([event for event in events if event.startswith("half_")]),
            80,
        )
        self.assertEqual(events.count("phase_matching"), 2)
        self.assertEqual(result.audit["reference_worker_process_count"], 2)
        self.assertNotIn(
            os.getpid(),
            result.audit["reference_worker_pids"],
        )
        self.assertEqual(
            result.audit["phase_matching_second_pass_execution_count"],
            4,
        )
        self.assertEqual(
            result.frozen_parameters["stage_b_status"],
            "passed",
        )
        persisted_payload = json.loads(
            json.dumps(
                self.mod.stage_b_validation_evidence_payload(result),
                allow_nan=False,
            )
        )
        for persisted_class in persisted_payload["class_evidence"].values():
            restored = self.mod._unpack_stage_b_class_result(
                persisted_class
            )
            self.assertTrue(
                self.mod._successful_class_evidence_is_complete(restored)
            )
        stripped_class_evidence = {
            key: {
                "candidate_ids": evidence["candidate_ids"],
                "continuous_left_rows": evidence["continuous_left_rows"],
                "continuous_left_rows_sha256": key,
            }
            for key, evidence in result.class_evidence.items()
        }
        stripped_payload = self.mod._stage_b_evidence_components(
            budget=result.budget,
            benchmark_evidence=result.benchmark_evidence,
            selection=result.selection,
            candidate_results=result.candidate_results,
            measurement_classes=result.measurement_classes,
            class_evidence=stripped_class_evidence,
            phase_matching_diagnostics=result.phase_matching_diagnostics,
        )
        stripped_hash = self.mod.stage_b_validation_evidence_sha256(
            stripped_payload
        )
        stripped_frozen = {
            **dict(result.frozen_parameters),
            "validation_table_sha256": stripped_hash,
        }
        with self.assertRaisesRegex(
            ValueError,
            "evidence",
        ):
            self.mod.StageBRunResult(
                status="passed",
                return_code=0,
                budget=result.budget,
                benchmark_evidence=result.benchmark_evidence,
                selection=result.selection,
                candidate_results=result.candidate_results,
                measurement_classes=result.measurement_classes,
                class_evidence=stripped_class_evidence,
                decision=result.decision,
                phase_matching_diagnostics=(
                    result.phase_matching_diagnostics
                ),
                frozen_parameters=stripped_frozen,
                audit=result.audit,
            )

        def rebuild_with_class_evidence(class_evidence):
            payload = self.mod._stage_b_evidence_components(
                budget=result.budget,
                benchmark_evidence=result.benchmark_evidence,
                selection=result.selection,
                candidate_results=result.candidate_results,
                measurement_classes=result.measurement_classes,
                class_evidence=class_evidence,
                phase_matching_diagnostics=(
                    result.phase_matching_diagnostics
                ),
            )
            frozen = {
                **dict(result.frozen_parameters),
                "validation_table_sha256": (
                    self.mod.stage_b_validation_evidence_sha256(payload)
                ),
            }
            return self.mod.StageBRunResult(
                status="passed",
                return_code=0,
                budget=result.budget,
                benchmark_evidence=result.benchmark_evidence,
                selection=result.selection,
                candidate_results=result.candidate_results,
                measurement_classes=result.measurement_classes,
                class_evidence=class_evidence,
                decision=result.decision,
                phase_matching_diagnostics=(
                    result.phase_matching_diagnostics
                ),
                frozen_parameters=frozen,
                audit=result.audit,
            )

        first_hash = next(iter(result.class_evidence))
        inconsistent_half = {
            key: dict(value)
            for key, value in result.class_evidence.items()
        }
        inconsistent_half[first_hash][
            "split_half_absolute_differences_km_s"
        ] = np.full((20, 4), 0.20)
        with self.assertRaisesRegex(ValueError, "evidence"):
            rebuild_with_class_evidence(inconsistent_half)

        inconsistent_closure = {
            key: dict(value)
            for key, value in result.class_evidence.items()
        }
        inconsistent_closure[first_hash][
            "triplet_rows_geometry_valid"
        ] = tuple(
            {
                **dict(row),
                "corrected_time_ac_s": (
                    float(row["corrected_time_ac_s"])
                    + (
                        float(row["period_s"])
                        if float(row["period_s"]) == 3.0
                        else 0.0
                    )
                ),
            }
            for row in inconsistent_closure[first_hash][
                "triplet_rows_geometry_valid"
            ]
        )
        with self.assertRaisesRegex(ValueError, "evidence"):
            rebuild_with_class_evidence(inconsistent_closure)

    def test_stage_b_orchestrator_returns_nonzero_for_zero_triplet_support(self):
        grid = self.mod.build_candidate_grid(
            phase_conventions=(
                "BENSEN_VELOCITY_CCF",
                "LIN_NEGATIVE_DERIVATIVE_EGF",
            ),
            alpha_candidates=(5, 8, 12, 16, 20, 25),
            beta1_candidates=(0, 0.5, 1, 2, 4),
            beta2_candidates=(0, 1, 2, 4, 8),
        )
        left_rows = tuple(
            {
                "pair_name": "AA__BB",
                "target_period_s": period_s,
                "T_inst": period_s,
                "t0": 4.0,
                "U": 2.5,
                "signal_peak": 2.0,
                "leading_rms": 0.2,
                "trailing_rms": 0.2,
                "ridge_fields": {
                    "energy": 0.9,
                    "amplitude": 0.8,
                    "jump": 0.01,
                },
                "ridge_valid": True,
                "instantaneous_period_valid": True,
                "distance_km": 10.0,
                "azimuth_deg": 0.0,
                "leading_snr": 10.0,
                "trailing_snr": 10.0,
            }
            for period_s in (3.0, 3.5, 4.0, 5.0)
        )

        def benchmark_stage_b(**kwargs):
            return self.mod.StageBBenchmarkEvidence(
                candidate_grid_elapsed_s=1.0,
                ten_single_reference_fits_elapsed_s=1.0,
                lambda_cv_elapsed_s=1.0,
                twenty_half_samples_elapsed_s=1.0,
                measured_peak_memory_bytes=1,
                available_memory_bytes=10,
                cache_hit_fraction=0.0,
                benchmark_input_sha256="f" * 64,
            )

        result = self.mod.run_stage_b_validation(
            inventory_rows=[
                {
                    "pair_name": "AA__BB",
                    "distance_km": 10.0,
                    "azimuth_deg": 0.0,
                    "preliminary_snr": 10.0,
                }
            ],
            station_coordinates={
                "AA": (0.0, 0.0),
                "BB": (0.1, 0.0),
            },
            closure_triplets=(),
            closure_edge_pair_names=(),
            candidate_grid=grid,
            lineage_status="confirmed",
            lineage_preferred_phase_convention=(
                "LIN_NEGATIVE_DERIVATIVE_EGF"
            ),
            benchmark_stage_b=benchmark_stage_b,
            measure_candidate=lambda **kwargs: {
                "synthetic_validation_status": "accepted",
                "continuous_left_rows": left_rows,
                "accepted_outermost_velocity_cell_count": 0,
            },
            fit_full_reference=lambda **kwargs: (
                self.mod.FullReferenceEvidence(
                    status="accepted",
                    alias_status="accepted",
                    lambda_s=1.0,
                    lambda_g=1.0,
                    basin_starts=tuple(range(5)),
                    optimizer_calls=10,
                    corrected_rows=(),
                    result_sha256="e" * 64,
                )
            ),
            fit_split_half_reference=lambda **kwargs: self.fail(
                "half fit ran without accepted triplet closure"
            ),
            run_phase_matching=lambda **kwargs: self.fail(
                "phase matching ran after triplet failure"
            ),
            input_inventory_sha256="a" * 64,
            code_sha256="b" * 64,
            config_sha256="c" * 64,
            max_workers=24,
        )
        self.assertEqual(result.return_code, 2)
        self.assertEqual(result.status, "insufficient_triplet_support")
        self.assertIsNone(result.frozen_parameters)


if __name__ == "__main__":
    unittest.main()
