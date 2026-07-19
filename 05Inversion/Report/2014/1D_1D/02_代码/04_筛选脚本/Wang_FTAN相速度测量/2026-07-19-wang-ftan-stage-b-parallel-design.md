# Wang FTAN Stage B 24 核并行设计

**日期：** 2026-07-19

## 1. 目标

在不改变 Stage B 的 300 组候选参数、分层样本、三角闭合差、分半稳定性、相位匹配和科学门槛的前提下，使固定性能基准及其后的正式 Stage B 工作持续利用 work 服务器的 24 个物理核。新运行必须保留确定性、可审计性和非零失败状态。

## 2. 已确认的根因

work 服务器有 24 个在线物理核和约 58 GiB 可用内存。旧 Stage B 虽使用 `--max-workers 24`，但 `_run_stage_b_benchmark_workload` 内的 240 个波形/相位/alpha 组合、6,000 次脊线搜索和 335 次参考曲线拟合均由单进程 `for` 循环执行。`max_workers` 在该函数中只参与参数校验和内存投影，因此旧进程平均仅使用约 2 核。

正式真实候选测量已经按台站对使用进程池，测量类参考拟合也已经按类使用进程池；本次不重写这两套已验证的科学流程。

## 3. 选定方案

采用 24 个独立进程，每个进程限制为 1 个 BLAS/OpenMP 线程。

### 3.1 FTAN 候选基准

- 固定生成 240 个任务：`20 waveform × 2 phase convention × 6 alpha`。
- 每个任务仍完整执行 25 个 `beta1 × beta2` 组合和最多 3 条脊线，不减少工作量。
- 使用 `fork` 进程池并行执行，最多 24 个 worker。
- worker 返回任务编号、滤波耗时、脊线耗时、输入摘要和 PID。
- 主进程按任务编号排序后求和，禁止依赖并行完成顺序。

### 3.2 参考拟合基准

- 固定生成 335 个独立任务：10 个单参考拟合、125 个 lambda 交叉验证拟合、200 个分半拟合。
- 使用同一并发上限的进程池执行。
- 每个任务保留原来的 `maxiter`、lambda 和起始曲线。
- 三组耗时分别记录，不合并或遗漏。

### 3.3 进程池所有权与生命周期

- 所有进程池只允许由 Stage B 主进程顺序创建，使用显式 `multiprocessing.get_context("fork")`。
- benchmark FTAN pool 必须在拟合 pool 创建前完成 `close/join`；benchmark callback 返回前两个 pool 均必须销毁。
- benchmark 完成后才允许逐候选创建现有 pair pool；全部候选测量完成后才允许创建 measurement-class pool。
- 禁止 daemon/worker 创建子进程池，禁止任意两个 pool 生命周期重叠。
- evidence 记录 creator PID、worker PID 和 pool 开始/结束时刻；验证器拒绝 creator 不是 Stage B 主进程或 pool 时间区间重叠的证据。

### 3.4 线程与资源控制

服务器必须在启动 Python 前使用命令环境前缀设置：

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `VECLIB_MAXIMUM_THREADS=1`

这样并发上限是 24 个计算进程，而不是 24 个进程各自再创建多线程。正式命令继续使用 `--max-workers 24`。

不能在 benchmark 函数或 pool initializer 中才设置这些变量，因为 runner 顶层已经导入 h5py、NumPy 和 SciPy，fork worker 会继承已初始化的线程池。每个 worker 还必须使用 `threadpoolctl.threadpool_info()` 记录实际数值库线程数；任一 BLAS/OpenMP 后端的 `num_threads != 1` 都使正式 benchmark 非零失败。仅检查 `os.environ` 不足以通过验证。

## 4. 预算计时口径

并行基准必须同时保存：

1. 每类任务的 worker 耗时总和，用作单任务成本和科学预算外推；字段名必须包含 `_worker_sum_`；
2. 进程池实际墙钟时间，用来证明 24 核并行带来的现实加速；
3. 实际 worker PID 数和请求 worker 数，用来证明并行确实发生；
4. 峰值内存与可用内存比例。

现有 `StageBBenchmarkEvidence` 将使用明确的 `_worker_sum_` 和 `_pool_wall_` 字段，禁止使用含糊的 `elapsed` 字段。预算只使用 worker-sum，pool-wall 只作为实际加速证据，禁止进入科学投影。

固定预算公式为：

```text
candidate_projected_wall
  = candidate_worker_cost_sum / 20
    * ceil(selected_pair_count / worker_count)

reference_projected_wall
  = (ten_fit_worker_sum + cv_fit_worker_sum + half_fit_worker_sum) / 335
    * 953
    * ceil(distinct_measurement_class_count / worker_count)
```

其中当前生产路径对每个 beta 组合重新计算滤波，因此：

```text
candidate_worker_cost_sum
  = 25 * sum(filter_elapsed_over_240_tasks)
    + sum(ridge_elapsed_over_240_tasks)
```

不得直接累加每个任务的一次 filter 与 ridge，否则会漏计 24 份滤波成本。验证器同时检查请求 worker 数、实际 worker PID 数、240 个 FTAN 任务、6,000 个 beta 脊线任务、335 个拟合任务及各组计数守恒。worker-sum 固定时，worker 从 1 改为 24 只能通过上述 `ceil` 各折算一次。

跨进程峰值内存定义为两个 benchmark pool 运行期间按固定间隔采样的：

```text
Stage B 主进程 RSS + 所有存活 benchmark worker RSS
```

采样器记录采样间隔、峰值时刻、主进程 PID、当时活跃 worker PID、各 PID RSS、运行前基线和系统可用内存。不得使用 `RUSAGE_SELF × worker_count` 或只使用父进程 RSS；缺少全部活跃 worker 覆盖时验证器拒绝证据。

## 5. 确定性和失败处理

- 任务列表、参数组合和输入摘要必须与旧串行基准完全一致。
- 并行返回结果按稳定任务 ID 排序后聚合。
- 任一 worker 异常都使 Stage B 返回非零；不允许用缺失任务的部分结果生成预算结论。
- 验证任务数必须严格为 240 和 335，FTAN beta 组合总数必须为 6,000。
- `max_workers=24` 的正式服务器 benchmark 必须创建恰好 24 个 worker，并证明它们存在时间重叠；测试环境可使用较小显式并发值验证同一契约。
- Stage B 通过预算和科学验证前，Stage C 仍禁止启动。

## 6. 测试方案

测试先行，至少覆盖：

- 正式服务器 `max_workers=24` 时创建恰好 24 个具有重叠工作区间的 PID；
- `max_workers=1` 与并行结果具有相同的唯一任务 ID、参数/输入摘要、240/6,000/335 完成计数和确定性结果摘要；不要求易受调度影响的耗时相等；
- 并行完成顺序变化不改变聚合结果；
- 使用真实 `threadpoolctl` 证明每个 worker 的 BLAS/OpenMP 线程数为 1；
- worker 异常、缺任务或重复任务返回非零；
- pool-wall 单独改变时预算不变；worker-sum 固定、worker=1/24 时 candidate/reference waves 均只折算一次；
- filter 耗时严格乘 25；
- 子进程分配已知内存时，采样到的跨进程峰值随并发子进程上升，parent-only evidence 被拒绝；
- benchmark pool、pair pool 和 class pool 不重叠，worker/daemon 不能创建 pool；
- 现有 223 项 FTAN 测试全部继续通过。

## 7. 服务器重启和监督

1. 先在本地完成 RED/GREEN 测试和全套回归。
2. 同步生产脚本与测试到 work，并执行服务器 Stage A。
3. 使用新的独立输出目录启动 Stage B，避免复用旧的部分输出。
4. 启动后检查 24 核利用率、worker PID 数、内存和输出证据。
5. 建立当前任务的 30 分钟 heartbeat：检查远端 PID、CPU、RSS、worker 数、最新输出文件和退出状态；发现异常时报告，但不自动修改科学参数或启动 Stage C。
