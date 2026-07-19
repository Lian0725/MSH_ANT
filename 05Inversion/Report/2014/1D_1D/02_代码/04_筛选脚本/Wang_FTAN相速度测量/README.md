# Wang 风格 FTAN 相速度测量

本目录是 2014 年 1D–1D 台站对从双边互相关函数直接提取 2.5–5.0 s Rayleigh 波相速度的当前实现。它不读取 DisperPicker 输出，也不使用 Wang et al. (2017) 的报告速度作为先验或调参目标。

## 文件

- `scripts/04_dispersion/bensen_phase_ftan.py`：Gaussian FTAN、群速度脊线、解析相位、整数周分支、参考频散和 Wang 左/右列筛选的数学内核。
- `scripts/04_dispersion/wang_ftan_validation.py`：Stage A/B 的确定性抽样、闭合差、分半稳定性、候选网格、预算与正式输出验证器。
- `scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py`：服务器入口；执行 Stage A、B、C，并生成测量表、Figure 4 风格散点图和中文 HTML 报告。
- `方法设计与验证门槛.md`：公式、数据流、Wang/Bensen/Lin 对应关系、预注册参数和禁止事后调图的门槛。
- `tests/scripts_04_dispersion/`：上述生产代码的回归测试；保留仓库相对路径后可直接运行。

## 三阶段执行约束

1. Stage A 检查运行环境、输入谱系和全部自动测试。
2. Stage B 在按距离、45° 方位扇区和初步 SNR 分层的冻结样本上比较完整的 300 组候选参数，并执行三角闭合差、分半稳定性、空间偏差、相位约定和计算预算硬门槛。
3. 只有 Stage B 的正式证据通过验证器后，Stage C 才能在全量台站对上运行；缺失、空表、数量不守恒或异常退出均返回非零状态，不能伪装成功。

服务器示例：

```bash
python scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py \
  --stage B \
  --stack-root /path/to/STACK_ROOT \
  --stations-csv /path/to/stations.csv \
  --output-dir /path/to/stage_b_output \
  --component ZZ \
  --max-workers 24
```

Stage C 必须显式读取 Stage B 生成并通过验证的冻结参数：

```bash
python scripts/04_dispersion/run_work_reproduce_wang_figure4_allpairs.py \
  --stage C \
  --stack-root /path/to/STACK_ROOT \
  --stations-csv /path/to/stations.csv \
  --output-dir /path/to/stage_c_output \
  --frozen-parameters /path/to/stage_b_output/frozen_parameters.json \
  --component ZZ \
  --max-workers 24
```

## 当前验证状态（2026-07-19）

- 本地与 work 服务器 Stage A：223 项测试通过。
- 20 台站对诊断试算：20 个输入均有终态，其中 6 条连续 FTAN 曲线成功、14 条按科学条件拒绝、0 个未预期异常；该小样本不作为正式科学结果。
- 正式 Stage B 正在 work 服务器执行。完成前不得声称参数已冻结，也不得启动 Stage C 或二维反演。

当前代码基线为开发提交 `420cd858`。正式结果应以输出目录中的 `input_inventory.json`、`stage_b_decision.json`、`stage_b_validation_evidence.json`、`frozen_parameters.json` 和 `metadata.json` 为准，而不是以图片是否接近 Wang 论文为准。
