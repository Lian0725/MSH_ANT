# Critical Dispersion Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the archived 04-dispersion rebuild use the correct dataset/metadata paths and fail reliably on empty inputs, conversion errors, shard exceptions, or invalid placeholder outputs.

**Architecture:** Keep the existing four-file pipeline. The shell wrapper supplies explicit paths and propagates child exit codes; the converter and GPU runner expose truthful return codes; the verifier checks the actual directory contract and legacy failure markers. One consolidated unittest module exercises all four components with temporary synthetic data.

**Tech Stack:** Bash, Python 3, unittest, NumPy, h5py

---

### Task 1: Add failing regression tests

**Files:**
- Create: `tests/test_critical_dispersion_rebuild.py`

- [ ] Test that `dat-unspiked` passes the `2014/1D_1D` stack and `01RawData/2014/MetaData/1D` StationXML directory to the converter.
- [ ] Test that extraction passes `--dat_glob 1D.*.dat` and propagates a failed shard status.
- [ ] Test that conversion returns nonzero for a missing StationXML coordinate and for a failed conversion.
- [ ] Test that zero DAT matches return nonzero, legacy `failure_reason` NPZ files are not resumable, and processing exceptions are distinct from normal no-pick results.
- [ ] Test that the verifier accepts explicit current-layout directories, rejects zero DAT inputs, and rejects a nonempty NPZ `failure_reason`.
- [ ] Run: `python3 -m unittest -v tests.test_critical_dispersion_rebuild`
- [ ] Expected: failures corresponding to the current stale paths, hard-coded verifier layout, and false-success behavior.
- [ ] Commit the red tests.

### Task 2: Fix rebuild wiring and converter exit status

**Files:**
- Modify: `04DispersionData/Reports/Code/04_dispersion_autopick/rebuild_04dispersion_from_03cc_stackdata.sh`
- Modify: `04DispersionData/Reports/Code/04_dispersion_autopick/convert_1d_stack_to_dat.py`

- [ ] Change archived roots from `2014/1D` to `2014/1D_1D`.
- [ ] Add overridable `STATIONXML_DIR` and `DAT_GLOB` variables; pass StationXML and `1D.*.dat` explicitly.
- [ ] Collect shard PIDs, wait each PID, and return nonzero if any shard fails.
- [ ] Require StationXML coordinates for every pair when `--stationxml-dir` is supplied.
- [ ] Return nonzero for any conversion failure or an empty candidate set; preserve distance-filter skips as success.
- [ ] Run the focused shell/converter tests until green.
- [ ] Commit the rebuild/converter fix.

### Task 3: Make extraction status truthful

**Files:**
- Modify: `04DispersionData/Reports/Code/04_dispersion_autopick/run_dispersion_gpu_mi09.py`

- [ ] Introduce explicit `success`, `no_pick`, and `error` pair statuses.
- [ ] Keep valid no-pick outputs and count them separately without failing the shard.
- [ ] On exceptions, remove partial pair outputs and return `error`; do not create all-zero placeholders.
- [ ] Reject zero DAT glob matches.
- [ ] Make resume reject zero-byte, unreadable, or legacy failure-marked outputs.
- [ ] Return nonzero if any pair has `error`.
- [ ] Run the focused runner tests until green.
- [ ] Commit the runner fix.

### Task 4: Update verifier and integrate it into rebuild

**Files:**
- Modify: `04DispersionData/Reports/Code/04_dispersion_autopick/verify_disperpicker_full_run.py`
- Modify: `04DispersionData/Reports/Code/04_dispersion_autopick/rebuild_04dispersion_from_03cc_stackdata.sh`

- [ ] Accept explicit DAT, curve, NPZ, and log directories plus expected shard count.
- [ ] Validate nonempty DAT input, exact pair sets, zero-byte files, sampled shapes, log completion/error counts, and NPZ failure markers.
- [ ] Invoke the verifier after all extraction shards succeed and save its JSON report below the rebuild log directory.
- [ ] Run all consolidated tests until green.
- [ ] Commit the verifier integration.

### Task 5: Final verification and GitHub synchronization

**Files:**
- Modify only if needed: `README.md` or `04DispersionData/Reports/README_04DispersionData.md`

- [ ] Run `python3 -m unittest -v tests.test_critical_dispersion_rebuild`.
- [ ] Run Python compile checks for all three modified Python scripts.
- [ ] Run `bash -n` on the rebuild wrapper.
- [ ] Review `git diff origin/main...HEAD` and confirm inversion/map/result files are unchanged.
- [ ] Obtain independent code review and resolve any blocking findings.
- [ ] Push `codex/fix-critical-pipeline` to `origin` and verify the remote branch commit.
