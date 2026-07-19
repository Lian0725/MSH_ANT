# Wang FTAN Stage B 24-Core Parallel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen Stage B benchmark use all 24 physical work-server cores while preserving the 300-candidate science grid, deterministic evidence, correct budget projection, and nonzero failures.

**Architecture:** Keep the existing runner and validation module boundaries. Add explicit benchmark evidence for worker-summed cost versus pool wall time, execute the fixed FTAN and optimizer jobs in sequentially owned fork pools, and sample aggregate parent-plus-child RSS while each pool is alive. Keep the existing real pair pool and measurement-class pool unchanged and non-overlapping.

**Tech Stack:** Python 3.12, `multiprocessing` fork pools, NumPy/SciPy/h5py, pinned pure-Python `threadpoolctl==3.5.0`, standard-library `/proc` RSS sampling, `unittest`, JSON evidence.

---

## File map

- Modify `scripts/04_dispersion/wang_ftan_validation.py`: benchmark evidence schema, conservation validation, exact budget formulas, and four-phase pool lifecycle audit.
- Modify `scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py`: deterministic benchmark jobs, 24-process execution, aggregate RSS sampler, threadpool evidence, server launch validation, and explicit real-pair pool lifecycle evidence.
- Modify `tests/scripts_04_dispersion/test_wang_ftan_validation.py`: evidence and worker-wave budget regression tests.
- Modify `tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py`: parallel execution, task conservation, threadpool, memory, error, and pool-lifetime tests.

### Task 1: Freeze parallel benchmark evidence and budget formulas

**Files:**
- Modify: `scripts/04_dispersion/wang_ftan_validation.py:650-725,1740-1875`
- Test: `tests/scripts_04_dispersion/test_wang_ftan_validation.py`

- [ ] **Step 1: Write failing evidence-schema tests**

Add tests constructing `StageBBenchmarkEvidence` with named fields for candidate/filter/ridge worker sums, FTAN/fitting pool wall times, the three fit worker sums, task counts, requested/actual workers, worker PIDs, threadpool evidence, and aggregate RSS evidence. Assert missing counts, duplicate PIDs, parent-only RSS, or inconsistent worker sums raise `ValueError`.

- [ ] **Step 2: Run the targeted schema tests and verify RED**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests.scripts_04_dispersion.test_wang_ftan_validation.WangFtanValidationTests.test_parallel_benchmark_evidence_requires_complete_worker_contract
```

Expected: FAIL because the new evidence fields/validation do not exist.

- [ ] **Step 3: Implement the minimal immutable evidence contract**

Replace ambiguous benchmark elapsed fields with explicit fields such as:

```python
candidate_filter_worker_sum_s: float
candidate_ridge_worker_sum_s: float
candidate_worker_cost_sum_s: float
candidate_pool_wall_s: float
ten_fit_worker_sum_s: float
cv_fit_worker_sum_s: float
half_fit_worker_sum_s: float
fit_pool_wall_s: float
ftan_task_count: int
ridge_search_count: int
ten_fit_task_count: int
cv_fit_task_count: int
half_fit_task_count: int
ftan_requested_worker_count: int
ftan_actual_worker_pids: Tuple[int, ...]
ftan_creator_pid: int
ftan_pool_started_monotonic_s: float
ftan_pool_ended_monotonic_s: float
fit_requested_worker_count: int
fit_actual_worker_pids: Tuple[int, ...]
fit_creator_pid: int
fit_pool_started_monotonic_s: float
fit_pool_ended_monotonic_s: float
ftan_aggregate_peak_rss_bytes: int
fit_aggregate_peak_rss_bytes: int
```

Validate `candidate_worker_cost_sum_s == 25 * filter_sum + ridge_sum` within floating tolerance; counts are exactly 240/6000 and 10/125/200; both benchmark pools separately provide the requested PID count, creator PID, valid interval and aggregate RSS evidence; creator PIDs agree; and the FTAN interval ends before the fit interval starts. Add RED cases for a 23-PID FTAN pool, a 23-PID fit pool, overlapping intervals and 10/125/200 count imbalance.

Add a separate final `StageBPoolLifecycleAudit` (or an equivalently validated object in `StageBRunResult.audit`) because real-pair and measurement-class pools run only after benchmark construction and the budget gate. It contains benchmark FTAN/fit plus real-pair/class status, creator, worker PIDs and intervals. When budget rejects before real measurement, the latter two statuses must be exactly `not_run_budget_rejected` with no fabricated PID or interval; after a full Stage B run they must carry real production evidence.

- [ ] **Step 4: Write failing exact-budget tests**

Add two tests proving:

```python
candidate_wall = candidate_worker_sum / 20 * ceil(selected_pairs / workers)
reference_wall = reference_worker_sum / 335 * 953 * ceil(classes / workers)
```

Changing only either pool-wall field must not change the projected budget. With fixed worker sums, worker=1 and worker=24 must apply exactly one worker-wave reduction.

- [ ] **Step 5: Run budget tests and verify RED**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests.scripts_04_dispersion.test_wang_ftan_validation.WangFtanValidationTests.test_parallel_budget_ignores_pool_wall \
tests.scripts_04_dispersion.test_wang_ftan_validation.WangFtanValidationTests.test_parallel_budget_applies_worker_waves_once
```

Expected: FAIL against the old candidate formula and old evidence names.

- [ ] **Step 6: Implement the exact formulas and verify GREEN**

Run all validation tests:

```bash
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests/scripts_04_dispersion/test_wang_ftan_validation.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add scripts/04_dispersion/wang_ftan_validation.py \
  tests/scripts_04_dispersion/test_wang_ftan_validation.py
git commit -m "refactor: define parallel Stage B budget evidence"
```

### Task 2: Execute the fixed benchmark in non-overlapping fork pools

**Files:**
- Modify: `scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py:5030-5250`
- Test: `tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py`

- [ ] **Step 1: Write failing deterministic-job tests**

Require job builders to produce exactly 240 unique FTAN task IDs and 335 unique fit task IDs. Verify FTAN jobs collectively represent 6000 beta searches and fit groups are exactly 10/125/200. Shuffle worker result order and assert the deterministic summary hash is unchanged.

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_benchmark_job_builders_preserve_fixed_workload \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_parallel_benchmark_result_order_is_deterministic
```

Expected: FAIL because job builders and deterministic aggregators do not exist.

- [ ] **Step 3: Implement top-level picklable job builders/evaluators**

Create top-level frozen job/result dataclasses or JSON-safe tuples. Each FTAN worker handles one waveform/convention/alpha task and all 25 beta combinations. Each fit worker handles one frozen optimizer task. Return stable task ID, elapsed components, PID, start/end monotonic timestamps, threadpool evidence, and output digest.

- [ ] **Step 4: Write failing fork-pool ownership tests**

Use lightweight injected evaluators to prove:

- explicit fork context is selected;
- the requested worker count is created;
- FTAN pool is joined before fit pool starts;
- no pool is created by a daemon worker;
- worker exceptions, missing IDs, or duplicate IDs fail without partial evidence.
- a barrier fixture causes all requested workers to report intersecting task start/end intervals, proving real overlapping work rather than merely 24 distinct historical PIDs;
- an orchestration integration fixture records benchmark FTAN, benchmark fit, real-pair and measurement-class pool creator/interval evidence and rejects any overlap or non-main creator.

Run and verify RED:

```bash
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_parallel_benchmark_pools_are_non_overlapping \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_all_stage_b_pool_phases_have_one_main_creator \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_requested_workers_have_overlapping_work_intervals
```

Expected: FAIL because pool lifecycle evidence and barrier overlap checks do not exist.

- [ ] **Step 5: Implement sequentially owned fork pools**

The Stage B main PID creates the FTAN pool, consumes all results, and closes/joins it. Only then create the fit pool. Sort results by stable task ID before aggregation. Record creator PID and both pool intervals.

Wrap the existing `measure_stage_b_candidate_from_tasks` pool and `execute_measurement_class_processes` pool without changing their scientific evaluators: assert the creator is the Stage B main PID, use explicit fork context, record creator/PIDs/start/end, and guarantee `close/join` on success or `terminate/join` on exception. Propagate the real-pair phase envelope across sequential candidates and the class-pool envelope into the separate final `StageBPoolLifecycleAudit`. Validate the four ordered non-overlapping phases:

```text
benchmark FTAN -> benchmark fit -> all real-pair pools -> measurement-class pool
```

Make the orchestration integration RED test GREEN using these production audit fields, not mocks that fabricate lifecycle evidence. Add a budget-rejection GREEN test proving pair/class statuses are `not_run_budget_rejected` and contain no fabricated lifecycle values.

- [ ] **Step 6: Write failing aggregate-memory tests**

Spawn lightweight workers that allocate a known byte array and overlap behind a barrier. Assert sampled `parent RSS + live child RSS` exceeds parent-only RSS and increases with worker count. Reject samples without every active worker PID.

Run and verify RED:

```bash
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_aggregate_rss_sampler_includes_live_workers \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_aggregate_rss_sampler_stops_after_worker_error
```

Expected: FAIL because the sampler and cleanup contract do not exist.

- [ ] **Step 7: Implement the pool RSS sampler**

Use a small platform adapter: Linux reads resident pages from `/proc/<pid>/statm` and multiplies by `SC_PAGE_SIZE`; local non-Linux tests inject a deterministic reader rather than weakening formal Linux evidence. After pool creation and before blocking result collection, start a `threading.Thread` sampler with a `threading.Event`; sample parent and every pool-worker PID at a fixed interval. In `finally`, close or terminate the pool, join it, set the sampler stop event, and join the sampler thread. Preserve separate FTAN/fit peak breakdown, timestamp, active PIDs, baseline, and system available memory. Test both normal completion and worker-exception cleanup; do not multiply `RUSAGE_SELF` by workers.

- [ ] **Step 8: Run runner tests and verify GREEN**

```bash
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py
```

Expected: PASS.

- [ ] **Step 9: Commit Task 2**

```bash
git add scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py \
  tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py
git commit -m "feat: parallelize frozen Stage B benchmark"
```

### Task 3: Enforce one numerical-library thread per worker

**Files:**
- Modify: `scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py`
- Test: `tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py`

- [ ] **Step 1: Write failing launch-contract tests**

Require formal Stage B to reject absent/non-1 values for `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`. Require every worker's real `threadpoolctl.threadpool_info()` result to report `num_threads == 1` for every loaded numerical backend.

- [ ] **Step 2: Verify RED**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_formal_stage_b_subprocess_requires_preimport_thread_limits \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_benchmark_workers_report_single_thread_backends
```

Expected: FAIL because formal Stage B currently accepts the environment without the frozen thread contract.

- [ ] **Step 3: Implement the launch and worker checks**

Install and verify the local dependency before running the RED/GREEN subprocess tests:

```bash
/opt/homebrew/anaconda3/envs/seis/bin/python -m pip install threadpoolctl==3.5.0
/opt/homebrew/anaconda3/envs/seis/bin/python -c 'import threadpoolctl; assert threadpoolctl.__version__ == "3.5.0"'
```

Install the same version into the frozen server venv, recording it in Stage A evidence. Validate the environment at formal Stage B entry and validate real threadpool evidence from every worker. Add positive and negative `subprocess.run` tests that set or omit the five variables before a fresh Python interpreter imports the runner; do not rely on the already-imported unittest process. Keep the server command responsible for setting variables before Python starts.

- [ ] **Step 4: Verify targeted and complete test matrix**

```bash
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests/scripts_04_dispersion/test_bensen_phase_ftan.py \
tests/scripts_04_dispersion/test_wang_ftan_validation.py \
tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py
```

Expected: the 223 pre-existing tests plus every new plan test PASS with no warnings.

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py \
  tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py
git commit -m "fix: enforce Stage B worker thread isolation"
```

### Task 4: Server verification, restart, and supervision

**Files:**
- Local source: `/Users/lgx/Projects/MSH_ANT/.worktrees/wang-ftan-phase-velocity`
- Server code: `/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/code`
- New server output: `/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/stage_b_<COMMIT>_24core`
- GitHub clone: `/Users/lgx/Projects/MSH_ANT_upload`, branch `main`, remote `origin`

- [ ] **Step 1: Sync exact production/test files to work and verify SHA-256**

Set the frozen implementation ID once:

```bash
COMMIT=$(git rev-parse --short HEAD)
test -n "$COMMIT"
```

Copy exactly these files:

```text
scripts/04_dispersion/bensen_phase_ftan.py
scripts/04_dispersion/wang_ftan_validation.py
scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py
tests/scripts_04_dispersion/test_bensen_phase_ftan.py
tests/scripts_04_dispersion/test_wang_ftan_validation.py
tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py
```

From the local worktree run:

```bash
rsync -a --relative \
  scripts/04_dispersion/bensen_phase_ftan.py \
  scripts/04_dispersion/wang_ftan_validation.py \
  scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py \
  tests/scripts_04_dispersion/test_bensen_phase_ftan.py \
  tests/scripts_04_dispersion/test_wang_ftan_validation.py \
  tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py \
  work:/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/code/
```

Create and compare complete ordered manifests:

```bash
shasum -a 256 \
  scripts/04_dispersion/bensen_phase_ftan.py \
  scripts/04_dispersion/wang_ftan_validation.py \
  scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py \
  tests/scripts_04_dispersion/test_bensen_phase_ftan.py \
  tests/scripts_04_dispersion/test_wang_ftan_validation.py \
  tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py \
  > /tmp/wang-ftan-local.sha256
ssh work 'cd /mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/code && sha256sum \
  scripts/04_dispersion/bensen_phase_ftan.py \
  scripts/04_dispersion/wang_ftan_validation.py \
  scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py \
  tests/scripts_04_dispersion/test_bensen_phase_ftan.py \
  tests/scripts_04_dispersion/test_wang_ftan_validation.py \
  tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py' \
  > /tmp/wang-ftan-server.sha256
diff -u /tmp/wang-ftan-local.sha256 /tmp/wang-ftan-server.sha256
```

Any mismatch stops the run.

- [ ] **Step 2: Run Stage A in the frozen server environment**

Install and verify the dependency:

```bash
ssh work '/mnt/data_hdd/lgx/MSH_ANT/.venvs/2014_wang_pws/bin/python -m pip install threadpoolctl==3.5.0'
```

Run Stage A from the local shell with an explicit remote working directory and prefix:

```bash
ssh work "cd /mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/code && \
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/mnt/data_hdd/lgx/MSH_ANT/.venvs/2014_wang_pws/bin/python \
scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py \
--stage A --output-dir /mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/stage_a_${COMMIT}_24core"
```

Expected: exit 0 and `stage_a_evidence.json` records every test passing and `threadpoolctl==3.5.0`.

- [ ] **Step 3: Run a reduced-work diagnostic parallel benchmark**

Run the new targeted server unittests for lightweight/injected 24-worker overlap, threadpool and RSS fixtures:

```bash
ssh work "cd /mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/code && \
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/mnt/data_hdd/lgx/MSH_ANT/.venvs/2014_wang_pws/bin/python -m unittest \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_requested_workers_have_overlapping_work_intervals \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_benchmark_workers_report_single_thread_backends \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_aggregate_rss_sampler_includes_live_workers \
tests.scripts_04_dispersion.test_run_work_reproduce_wang_figure4_allpairs.RunWorkReproduceWangFigure4AllPairsTests.test_benchmark_job_builders_preserve_fixed_workload \
tests.scripts_04_dispersion.test_wang_ftan_validation.WangFtanValidationTests.test_parallel_benchmark_evidence_requires_complete_worker_contract"
```

Expected: exit 0, exactly 24 overlapping worker PIDs and complete conservation. Do not use this diagnostic as scientific Stage B evidence.

- [ ] **Step 4: Start formal Stage B in a new output directory**

From the local shell launch a detached remote wrapper. Keep control files outside the scientific output directory so formal stale-artifact cleanup cannot remove them:

```bash
ssh work "CONTROL=/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/stage_b_${COMMIT}_24core_control; \
export CONTROL; mkdir -p \"\$CONTROL\"; rm -f \"\$CONTROL/wrapper.pid\" \"\$CONTROL/wrapper.pid.tmp\" \"\$CONTROL/python.pid\" \"\$CONTROL/python.pid.tmp\" \"\$CONTROL/exit_status\" \"\$CONTROL/exit_status.tmp\"; \
cd /mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/code; \
nohup bash -lc 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/mnt/data_hdd/lgx/MSH_ANT/.venvs/2014_wang_pws/bin/python scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py \
--stage B --stack-root /mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK_SPIKE_REMOVED_DIAGFIT_20260628 \
--stations-csv /mnt/data_hdd/lgx/MSH_ANT/inversion/phase_velocity_maps_aant_2014/wang_1d1d_server_wang_gvmax_min1lambda_cdisp/stations.csv \
--output-dir /mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/stage_b_${COMMIT}_24core \
--component ZZ --max-workers 24 & python_pid=\$!; echo \"\$python_pid\" > \"\$CONTROL/python.pid.tmp\"; mv \"\$CONTROL/python.pid.tmp\" \"\$CONTROL/python.pid\"; wait \"\$python_pid\"; rc=\$?; echo \"\$rc\" > \"\$CONTROL/exit_status.tmp\"; mv \"\$CONTROL/exit_status.tmp\" \"\$CONTROL/exit_status\"; exit \"\$rc\"' \
> \"\$CONTROL/run.log\" 2>&1 < /dev/null & wrapper_pid=\$!; echo \"\$wrapper_pid\" > \"\$CONTROL/wrapper.pid.tmp\"; mv \"\$CONTROL/wrapper.pid.tmp\" \"\$CONTROL/wrapper.pid\"; cat \"\$CONTROL/wrapper.pid\""
```

Expected: no manual scientific overrides; input inventory and Stage A preflight are written before benchmark evidence. Any nonzero hash/test/preflight status stops further work.

- [ ] **Step 5: Verify live resource use**

Check main/child PIDs, 24 active workers during benchmark pools, total CPU utilization, RSS, and initial output artifacts:

```bash
ssh work "CONTROL=/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/stage_b_${COMMIT}_24core_control; \
python_pid=\$(cat \"\$CONTROL/python.pid\"); test -n \"\$python_pid\"; \
ps -p \"\$python_pid\" -o pid,etime,time,pcpu,rss,nlwp,stat,args; \
pgrep -P \"\$python_pid\" | sort -n; \
ps --ppid \"\$python_pid\" -o pid,pcpu,rss,stat,args; \
find /mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_phase_20260717/stage_b_${COMMIT}_24core -maxdepth 1 -type f -printf '%f %s bytes\n' | sort; \
test ! -f \"\$CONTROL/exit_status\" || { echo EXIT_STATUS=\$(cat \"\$CONTROL/exit_status\"); tail -100 \"\$CONTROL/run.log\"; }"
```

Expected during a benchmark pool: 24 child PIDs with overlapping active intervals and aggregate CPU approaching 24 cores. If CPU remains low because the job is between pools or I/O-bound, distinguish that from a failed worker pool before changing code.

- [ ] **Step 6: Keep the 30-minute heartbeat active**

Call the Codex automation tool with `{mode: "view", id: "wang-ftan-stage-b"}` and confirm status `ACTIVE`; do not create a duplicate and do not expose or rewrite its recurrence rule. If the ID is missing or paused, stop and request explicit automation creation/reactivation rather than silently replacing it. The heartbeat reads the control PID/log/exit-status plus scientific output every 30 minutes. It must not start Stage C unless Stage B exits 0 and deep evidence validation passes.

- [ ] **Step 7: Sync verified code to GitHub**

Copy the verified files into the existing layout:

```bash
UPLOAD='/Users/lgx/Projects/MSH_ANT_upload/05Inversion/Report/2014/1D_1D/02_代码/04_筛选脚本/Wang_FTAN相速度测量'
rsync -a scripts/04_dispersion/bensen_phase_ftan.py scripts/04_dispersion/wang_ftan_validation.py scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py "$UPLOAD/scripts/04_dispersion/"
rsync -a tests/scripts_04_dispersion/test_bensen_phase_ftan.py tests/scripts_04_dispersion/test_wang_ftan_validation.py tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py "$UPLOAD/tests/scripts_04_dispersion/"
rsync -a docs/superpowers/specs/2026-07-19-wang-ftan-stage-b-parallel-design.md "$UPLOAD/"
rsync -a docs/superpowers/plans/2026-07-19-wang-ftan-stage-b-parallel.md "$UPLOAD/"
cd "$UPLOAD"
MPLCONFIGDIR=/private/tmp/mpl-wang-ftan-upload PYTHONWARNINGS=error \
/opt/homebrew/anaconda3/envs/seis/bin/python -m unittest \
tests/scripts_04_dispersion/test_bensen_phase_ftan.py \
tests/scripts_04_dispersion/test_wang_ftan_validation.py \
tests/scripts_04_dispersion/test_run_work_reproduce_wang_figure4_allpairs.py
```

Expected: all existing and new tests pass. Then:

```bash
git -C /Users/lgx/Projects/MSH_ANT_upload fetch origin main
git -C /Users/lgx/Projects/MSH_ANT_upload status --short --branch
git -C /Users/lgx/Projects/MSH_ANT_upload add '05Inversion/Report/2014/1D_1D/02_代码/04_筛选脚本/Wang_FTAN相速度测量'
git -C /Users/lgx/Projects/MSH_ANT_upload diff --cached --check
git -C /Users/lgx/Projects/MSH_ANT_upload commit -m 'perf: parallelize Wang FTAN Stage B'
git -C /Users/lgx/Projects/MSH_ANT_upload pull --rebase origin main
git -C /Users/lgx/Projects/MSH_ANT_upload push origin main
git -C /Users/lgx/Projects/MSH_ANT_upload fetch origin main
test "$(git -C /Users/lgx/Projects/MSH_ANT_upload rev-parse HEAD)" = "$(git -C /Users/lgx/Projects/MSH_ANT_upload rev-parse origin/main)"
```

Any test/hash/diff/pull/rebase conflict exits nonzero and stops before push or before claiming synchronization.
