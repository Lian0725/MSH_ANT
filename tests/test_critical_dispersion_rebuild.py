import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUTOPICK = ROOT / "04DispersionData" / "Reports" / "Code" / "04_dispersion_autopick"
REBUILD = AUTOPICK / "rebuild_04dispersion_from_03cc_stackdata.sh"
CONVERTER = AUTOPICK / "convert_1d_stack_to_dat.py"
RUNNER = AUTOPICK / "run_dispersion_gpu_mi09.py"
VERIFIER = AUTOPICK / "verify_disperpicker_full_run.py"


def load_module(path: Path, name: str, *, postpone_annotations=False):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if postpone_annotations:
        source = "from __future__ import annotations\n" + path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


def write_executable(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def write_stack(path: Path, *, valid_dataset=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        if not valid_dataset:
            handle.create_dataset("wrong", data=np.ones(7))
            return
        ds = handle.create_dataset(
            "AuxiliaryData/Allstack_pws/ZZ",
            data=np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32),
        )
        ds.attrs["dt"] = 0.5
        ds.attrs["maxlag"] = 1.5
        ds.attrs["latS"] = 0.0
        ds.attrs["lonS"] = 0.0
        ds.attrs["latR"] = 0.0
        ds.attrs["lonR"] = 0.0
        ds.attrs["ngood_hours"] = 12


def write_stationxml(path: Path, station: str, lat: float, lon: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?xml version='1.0' encoding='UTF-8'?>
<FDSNStationXML xmlns="http://www.fdsn.org/xml/station/1">
  <Network code="1D">
    <Station code="{station}">
      <Latitude>{lat}</Latitude>
      <Longitude>{lon}</Longitude>
    </Station>
  </Network>
</FDSNStationXML>
""",
        encoding="utf-8",
    )


def write_dat(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "-122.200000 46.100000\n"
        "-122.100000 46.200000\n"
        "0.0400 1.0 1.0\n",
        encoding="utf-8",
    )


def write_curve(path: Path, *, nonzero=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = 2.8 if nonzero else 0.0
    lines = ["-122.200000 46.100000", "-122.100000 46.200000"]
    lines.extend(f"{0.2 + i * 0.1:.2f} {value:.3f} 5.000 0.800" for i in range(49))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_npz(path: Path, *, failure_reason=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "group_image": np.ones((701, 49)),
        "phase_image": np.ones((701, 49)),
        "periods": np.linspace(0.2, 5.0, 49),
        "velocities": np.linspace(0.5, 4.0, 701),
        "velocity_axis_km_s": np.linspace(0.5, 4.0, 701),
        "actual_velocity_axis_km_s": np.linspace(2.0, 4.0, 401),
        "snr": np.ones(49),
        "distance_km": 12.0,
    }
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    np.savez_compressed(path, **payload)


class RebuildShellTests(unittest.TestCase):
    def test_dat_rebuild_passes_current_stack_and_stationxml_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Final"
            (root / "03CC_StackData/2014/1D_1D/STACK").mkdir(parents=True)
            (root / "01RawData/2014/MetaData/1D").mkdir(parents=True)
            capture = Path(tmp) / "args.txt"
            fake_python = Path(tmp) / "fake-python"
            write_executable(
                fake_python,
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n",
            )
            env = os.environ.copy()
            env.update(
                FINAL_ROOT=str(root),
                PY_FTAN=str(fake_python),
                CAPTURE_ARGS=str(capture),
            )

            result = subprocess.run(
                ["bash", str(REBUILD), "dat-unspiked"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            args = capture.read_text(encoding="utf-8").splitlines()
            self.assertIn(str(root / "03CC_StackData/2014/1D_1D/STACK"), args)
            self.assertIn("--stationxml-dir", args)
            self.assertIn(str(root / "01RawData/2014/MetaData/1D"), args)

    def test_extract_passes_1d_glob_and_propagates_failed_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Final"
            dat_dir = root / "04DispersionData/2014/1D_1D/NonRemoveSpikes/DatData/dat_all"
            dat_dir.mkdir(parents=True)
            capture_dir = Path(tmp) / "captures"
            capture_dir.mkdir()
            fake_python = Path(tmp) / "fake-gpu"
            write_executable(
                fake_python,
                """#!/usr/bin/env bash
idx=unknown
prev=
for arg in "$@"; do
  if [[ "$prev" == "--shard_index" ]]; then idx="$arg"; fi
  prev="$arg"
done
printf '%s\n' "$@" > "$CAPTURE_DIR/$idx.txt"
if [[ "$idx" == "1" ]]; then exit 7; fi
exit 0
""",
            )
            env = os.environ.copy()
            env.update(
                FINAL_ROOT=str(root),
                PY_GPU=str(fake_python),
                SHARDS="2",
                CAPTURE_DIR=str(capture_dir),
                VERIFY_OUTPUTS="0",
            )

            result = subprocess.run(
                ["bash", str(REBUILD), "extract-unspiked"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0, result.stderr + result.stdout)
            capture = capture_dir / "0.txt"
            self.assertTrue(capture.exists(), result.stderr + result.stdout)
            args = capture.read_text(encoding="utf-8").splitlines()
            self.assertIn("--dat_glob", args)
            self.assertIn("1D.*.dat", args)

    def test_successful_extract_runs_verifier_with_current_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Final"
            dat_dir = root / "04DispersionData/2014/1D_1D/NonRemoveSpikes/DatData/dat_all"
            dat_dir.mkdir(parents=True)
            fake_gpu = Path(tmp) / "fake-gpu"
            write_executable(fake_gpu, "#!/usr/bin/env bash\nexit 0\n")
            verify_capture = Path(tmp) / "verify-args.txt"
            fake_verify = Path(tmp) / "fake-verify"
            write_executable(
                fake_verify,
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$VERIFY_ARGS\"\n",
            )
            env = os.environ.copy()
            env.update(
                FINAL_ROOT=str(root),
                PY_GPU=str(fake_gpu),
                PY_VERIFY=str(fake_verify),
                SHARDS="1",
                VERIFY_ARGS=str(verify_capture),
            )

            result = subprocess.run(
                ["bash", str(REBUILD), "extract-unspiked"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue(verify_capture.exists(), result.stderr + result.stdout)
            args = verify_capture.read_text(encoding="utf-8").splitlines()
            self.assertIn("--dat-dir", args)
            self.assertIn(str(dat_dir), args)
            self.assertIn("--dat-glob", args)
            self.assertIn("1D.*.dat", args)
            self.assertIn("--expected-shards", args)
            self.assertIn("1", args)

    def test_extract_verifies_only_fresh_logs_for_current_shard_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Final"
            dat_dir = root / "04DispersionData/2014/1D_1D/NonRemoveSpikes/DatData/dat_all"
            dat_dir.mkdir(parents=True)
            stale_logs = root / "04DispersionData/Reports/RebuildLogs/nonremove_logs"
            stale_logs.mkdir(parents=True)
            (stale_logs / "shard_9.log").write_text("处理失败: stale\n", encoding="utf-8")
            fake_gpu = Path(tmp) / "fake-gpu"
            write_executable(fake_gpu, "#!/usr/bin/env bash\nexit 0\n")
            fake_verify = Path(tmp) / "fake-verify"
            write_executable(
                fake_verify,
                """#!/usr/bin/env bash
logs=
expected=
prev=
for arg in "$@"; do
  if [[ "$prev" == "--logs-dir" ]]; then logs="$arg"; fi
  if [[ "$prev" == "--expected-shards" ]]; then expected="$arg"; fi
  prev="$arg"
done
count=$(find "$logs" -maxdepth 1 -type f -name 'shard_*.log' | wc -l | tr -d ' ')
[[ "$count" == "$expected" ]]
""",
            )
            env = os.environ.copy()
            env.update(
                FINAL_ROOT=str(root),
                PY_GPU=str(fake_gpu),
                PY_VERIFY=str(fake_verify),
                SHARDS="1",
            )

            result = subprocess.run(
                ["bash", str(REBUILD), "extract-unspiked"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


class ConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = load_module(CONVERTER, "critical_converter_test")

    def test_missing_stationxml_coordinate_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stack_root = root / "STACK"
            stack = stack_root / "1D.4001/1D.4002/stack.h5"
            write_stack(stack)
            xml_dir = root / "xml"
            write_stationxml(xml_dir / "1D.4001.xml", "4001", 46.1, -122.2)

            rc = self.converter.main(
                [
                    "--stack-root",
                    str(stack_root),
                    "--out-dir",
                    str(root / "dat"),
                    "--stationxml-dir",
                    str(xml_dir),
                ]
            )

            self.assertNotEqual(rc, 0)
            self.assertFalse((root / "dat/1D.4001__1D.4002.dat").exists())

    def test_failed_conversion_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stack_root = root / "STACK"
            write_stack(stack_root / "1D.4001/1D.4002/stack.h5", valid_dataset=False)

            rc = self.converter.main(
                ["--stack-root", str(stack_root), "--out-dir", str(root / "dat")]
            )

            self.assertNotEqual(rc, 0)

    def test_empty_stack_selection_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stack_root = root / "STACK"
            stack_root.mkdir()

            rc = self.converter.main(
                ["--stack-root", str(stack_root), "--out-dir", str(root / "dat")]
            )

            self.assertNotEqual(rc, 0)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module(
            RUNNER,
            "critical_gpu_runner_test",
            postpone_annotations=True,
        )

    def test_zero_dat_glob_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = self.runner.main(
                [
                    "--dat_dir",
                    str(root),
                    "--out_dir",
                    str(root / "curves"),
                    "--dat_glob",
                    "1D.*.dat",
                    "--skip_qc_plot",
                ]
            )
            self.assertNotEqual(rc, 0)

    def test_fully_resumed_shard_logs_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dat = root / "dat/1D.4001__1D.4002.dat"
            write_dat(dat)
            curves = root / "curves"
            write_curve(curves / "GDisp.1D.4001__1D.4002.txt")
            write_curve(curves / "CDisp.1D.4001__1D.4002.txt")
            pixels = root / "pixels"
            write_npz(pixels / "1D.4001__1D.4002.npz")

            with self.assertLogs(self.runner.logger, level="INFO") as captured:
                rc = self.runner.main(
                    [
                        "--dat_dir",
                        str(dat.parent),
                        "--out_dir",
                        str(curves),
                        "--energy_dir",
                        str(pixels),
                        "--dat_glob",
                        "1D.*.dat",
                        "--resume_existing",
                        "--skip_qc_plot",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertIn("处理完成", "\n".join(captured.output))

    def test_resume_rejects_legacy_failure_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dat = root / "dat/1D.4001__1D.4002.dat"
            write_dat(dat)
            curves = root / "curves"
            write_curve(curves / "GDisp.1D.4001__1D.4002.txt", nonzero=False)
            write_curve(curves / "CDisp.1D.4001__1D.4002.txt", nonzero=False)
            pixels = root / "pixels"
            write_npz(pixels / "1D.4001__1D.4002.npz", failure_reason="interpolation failed")

            complete = self.runner.pair_outputs_exist(
                str(dat), str(curves), energy_dir=str(pixels)
            )

            self.assertFalse(complete)

    def test_resume_rejects_truncated_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dat = root / "dat/1D.4001__1D.4002.dat"
            write_dat(dat)
            curves = root / "curves"
            curves.mkdir()
            (curves / "GDisp.1D.4001__1D.4002.txt").write_text(
                "truncated but nonempty\n", encoding="utf-8"
            )
            write_curve(curves / "CDisp.1D.4001__1D.4002.txt")
            pixels = root / "pixels"
            write_npz(pixels / "1D.4001__1D.4002.npz")

            self.assertFalse(
                self.runner.pair_outputs_exist(
                    str(dat), str(curves), energy_dir=str(pixels)
                )
            )

    def test_resume_rejects_npz_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dat = root / "dat/1D.4001__1D.4002.dat"
            write_dat(dat)
            curves = root / "curves"
            write_curve(curves / "GDisp.1D.4001__1D.4002.txt")
            write_curve(curves / "CDisp.1D.4001__1D.4002.txt")
            pixels = root / "pixels"
            pixels.mkdir()
            np.savez_compressed(pixels / "1D.4001__1D.4002.npz", unexpected=np.ones(1))

            self.assertFalse(
                self.runner.pair_outputs_exist(
                    str(dat), str(curves), energy_dir=str(pixels)
                )
            )

    def test_processing_exception_returns_error_and_removes_partial_outputs(self):
        self.assertTrue(hasattr(self.runner, "PairStatus"), "runner must expose PairStatus")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dat = root / "dat/1D.4001__1D.4002.dat"
            write_dat(dat)
            curves = root / "curves"
            pixels = root / "pixels"
            write_curve(curves / "GDisp.1D.4001__1D.4002.txt")
            write_curve(curves / "CDisp.1D.4001__1D.4002.txt")
            write_npz(pixels / "1D.4001__1D.4002.npz")

            with mock.patch.object(
                self.runner.gpu_backend,
                "build_gfcn",
                side_effect=ValueError("synthetic image failure"),
            ):
                status = self.runner.process_one_pair(
                    str(dat),
                    str(curves),
                    qc_plot_dir=None,
                    backend="numpy",
                    energy_dir=str(pixels),
                )

            self.assertIs(status, self.runner.PairStatus.ERROR)
            self.assertFalse((curves / "GDisp.1D.4001__1D.4002.txt").exists())
            self.assertFalse((curves / "CDisp.1D.4001__1D.4002.txt").exists())
            self.assertFalse((pixels / "1D.4001__1D.4002.npz").exists())

    def test_no_pick_is_successful_shard_but_error_is_not(self):
        self.assertTrue(hasattr(self.runner, "PairStatus"), "runner must expose PairStatus")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dat = root / "dat/1D.4001__1D.4002.dat"
            write_dat(dat)
            args = [
                "--dat_dir",
                str(dat.parent),
                "--out_dir",
                str(root / "curves"),
                "--dat_glob",
                "1D.*.dat",
                "--skip_qc_plot",
            ]
            with mock.patch.object(self.runner, "load_picker_and_qc", return_value=(object(), object())):
                with mock.patch.object(
                    self.runner,
                    "process_one_pair",
                    return_value=self.runner.PairStatus.NO_PICK,
                ):
                    self.assertEqual(self.runner.main(args), 0)
                with mock.patch.object(
                    self.runner,
                    "process_one_pair",
                    return_value=self.runner.PairStatus.ERROR,
                ):
                    self.assertNotEqual(self.runner.main(args), 0)


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = load_module(VERIFIER, "critical_verifier_test")

    def make_layout(self, root: Path, *, failure_reason=None):
        dat = root / "dat/1D.4001__1D.4002.dat"
        write_dat(dat)
        write_curve(root / "curves/GDisp.1D.4001__1D.4002.txt")
        write_curve(root / "curves/CDisp.1D.4001__1D.4002.txt")
        write_npz(root / "pixels/1D.4001__1D.4002.npz", failure_reason=failure_reason)
        logs = root / "logs"
        logs.mkdir(parents=True)
        (logs / "shard_0.log").write_text(
            "处理完成: 成功=1, 无有效拾取=0, 异常=0, 共=1\n", encoding="utf-8"
        )

    @staticmethod
    def verifier_args(root: Path):
        return [
            "--dat-dir",
            str(root / "dat"),
            "--curves-dir",
            str(root / "curves"),
            "--pixels-dir",
            str(root / "pixels"),
            "--logs-dir",
            str(root / "logs"),
            "--expected-shards",
            "1",
            "--sample-size",
            "1",
        ]

    def run_verifier(self, root: Path):
        try:
            return self.verifier.main(self.verifier_args(root))
        except SystemExit as exc:
            self.fail(f"verifier rejected the current-layout CLI: {exc}")

    def test_current_layout_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_layout(root)
            self.assertEqual(self.run_verifier(root), 0)

    def test_failure_marker_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_layout(root, failure_reason="legacy interpolation failure")
            self.assertNotEqual(self.run_verifier(root), 0)

    def test_dat_glob_limits_verification_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_layout(root)
            write_dat(root / "dat/XD.MS01__1D.4001.dat")
            args = self.verifier_args(root) + ["--dat-glob", "1D.*.dat"]
            try:
                rc = self.verifier.main(args)
            except SystemExit as exc:
                self.fail(f"verifier rejected --dat-glob: {exc}")
            self.assertEqual(rc, 0)

    def test_empty_dat_directory_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("dat", "curves", "pixels", "logs"):
                (root / name).mkdir()
            self.assertNotEqual(self.run_verifier(root), 0)

    def test_legacy_output_dir_uses_legacy_log_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dat(root / "dat_ge10/1D.4001__1D.4002.dat")
            write_curve(root / "curves_ge10/GDisp.1D.4001__1D.4002.txt")
            write_curve(root / "curves_ge10/CDisp.1D.4001__1D.4002.txt")
            write_npz(root / "full_pixel_data_ge10/1D.4001__1D.4002.npz")
            logs = root / "logs"
            logs.mkdir()
            for index in range(24):
                (logs / f"dispersion24_shard_{index}_of_24.log").write_text(
                    "处理完成: 成功=0, 失败=0, 共=0\n", encoding="utf-8"
                )

            rc = self.verifier.main(
                ["--output-dir", str(root), "--sample-size", "1"]
            )

            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
