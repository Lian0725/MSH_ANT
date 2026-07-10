# README_04DispersionData

> 更新日期：2026-07-09  
> 目标目录：`/mnt/data_hdd/MSH_ANT_Final/04DispersionData`  
> 本文档对应的代码快照来源：`work:/mnt/data_hdd/lgx/MSH_ANT/code/MSH_ANT/scripts`

## 1. 本目录是做什么的

`04DispersionData` 保存的是 2014 年 1D 异地台网互相关叠加之后，进一步做自动频散曲线提取时产生的全部中间结果和最终结果。它不是原始互相关波形目录，而是面向后续频散拾取、重画图、质量检查和参数追溯的结果整理目录。

当前顶层结构如下：

```text
04DispersionData/
├── 2014/
│   └── 1D/
│       ├── NonRemoveSpikes/
│       └── RemoveSpikes/
├── 2017/
└── Reports/
```

含义如下：

| 路径 | 含义 |
|---|---|
| `2014/1D/NonRemoveSpikes` | 2014 年 1D 台网、未去除 1 s 固定相位尖峰的叠加波形所对应的 DAT、自动拾取曲线和频散能量图数据。 |
| `2014/1D/RemoveSpikes` | 2014 年 1D 台网、已去除 1 s 固定相位尖峰之后的同一批叠加波形所对应的 DAT、自动拾取曲线和频散能量图数据。 |
| `2017` | 预留目录；截至 2026-07-09 该目录下没有同步进来的文件。 |
| `Reports` | 本次整理后的说明文档和服务器代码快照目录。后续读者应先看这里。 |

## 2. 2014/1D 结果目录说明

### 2.1 未去尖峰结果

```text
2014/1D/NonRemoveSpikes/
├── Curves/
│   └── curves_all_finalct001/
├── DatData/
│   └── dat_all/
└── DispersionNPZ/
```

截至 2026-07-09 已核对的文件数量：

| 路径 | 文件数 | 说明 |
|---|---:|---|
| `DatData/dat_all` | 402423 | 每个台对一个 DAT 文件。 |
| `Curves/curves_all_finalct001` | 804846 | 每个台对两条曲线文件：`GDisp.*.txt` 和 `CDisp.*.txt`。因此这里是 `402423 × 2`。 |
| `DispersionNPZ` | 402423 | 每个台对一个 `.npz`，保存频散能量图数值。 |

这套目录对应的服务器结果根目录是：

```text
/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_unspiked_20260702
```

其中：

| 最终目录 | 服务器来源 |
|---|---|
| `DatData/dat_all` | `.../dat_all` |
| `Curves/curves_all_finalct001` | `.../curves_all_finalct001` |
| `DispersionNPZ` | 2026-07-09 额外回填的 NPZ；不是 2026-07-02 当天原始目录自带内容。 |

也就是说，未去尖峰版本的曲线是 2026-07-02 已经生成好的，但频散能量 `.npz` 是后面用 GPU 的 NPZ-only 回填脚本重新补齐的。

### 2.2 去尖峰结果

```text
2014/1D/RemoveSpikes/
├── Curves/
│   └── curves_all_finalct001/
├── DatData/
│   └── dat_all/
└── DispersionNPZ/
    └── full_pixel_data_all/
```

截至 2026-07-09 已核对的文件数量：

| 路径 | 文件数 | 说明 |
|---|---:|---|
| `DatData/dat_all` | 402423 | 每个台对一个 DAT 文件。 |
| `Curves/curves_all_finalct001` | 804846 | 每个台对两条曲线文件：`GDisp.*.txt` 和 `CDisp.*.txt`。 |
| `DispersionNPZ/full_pixel_data_all` | 402423 | 每个台对一个 `.npz`。 |

这套目录对应的服务器结果根目录是：

```text
/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_20260701
```

其中服务器目录本身就同时带有：

```text
dat_all/
curves_all_finalct001/
full_pixel_data_all/
```

### 2.3 为什么 `Curves` 下面会有 80 万条文件

这里最容易让人误会。`Curves/curves_all_finalct001` 不是“一对台站一个文件”，而是“一对台站两类曲线各一个文件”：

```text
GDisp.<pair>.txt   群速度曲线
CDisp.<pair>.txt   相速度曲线
```

所以：

```text
402423 对台站 × 2 = 804846 个曲线文件
```

这不是重复计算，也不是多出了一倍的台对。

### 2.4 `full_pixel_data_all` 或 `DispersionNPZ` 里面是什么

这里保存的不是图片本身，而是“画频散能量图所需的数值矩阵”。每个台对一个 `.npz`，核心目的是：

1. 下次改图时不必重新读取所有 H5 或重新跑 EGFAnalysis。
2. 可以直接在 `.npz` 上改 colormap、叠加参考线、调文字位置、改筛选逻辑。
3. 支持重新生成 Wang Figure 4/5/6 一类图，而不必重新跑自动拾取。

典型 `.npz` 关键字段如下：

| Key | 含义 |
|---|---|
| `group_image` | 群速度频散能量图，固定为 `701 × 49`。 |
| `phase_image` | 经过第一轮群速度引导后、用于最终相速度拾取的频散图，固定为 `701 × 49`。 |
| `phase_image_raw` | 未做时变滤波的原始相速度图。 |
| `periods` | 周期轴，49 个点，对应 `0.2-5.0 s`，步长 `0.1 s`。 |
| `velocities` / `velocity_axis_km_s` | CNN 固定输入速度轴，`0.5-4.0 km/s`，步长 `0.005 km/s`，共 701 点。 |
| `actual_velocity_axis_km_s` | 当前台对真实参与计算的速度轴，从 `actual_start_v` 到 `4.0 km/s`。 |
| `snr` | 每个周期点对应的 SNR。 |
| `distance_km` | 台间距。 |
| `actual_start_v` | 当前台对实际生效的最低速度；远台对可能被内部逻辑自动抬高。 |
| `configured_start_v` | 脚本配置中设定的 `StartV`。 |
| `backend` | `cupy` 或 `numpy`。 |
| `image_elapsed_s` | 该台对生成能量图所花时间。 |

## 3. 频散曲线自动提取代码和代码归档说明

`Reports/Code` 下保存的是这次整理时，从 `work` 服务器直接拷贝下来的代码快照。

目录结构：

```text
Reports/
├── README_04DispersionData.md
├── Environment/
└── Code/
    ├── 02_cc_stack_and_spike_removal/
    └── 04_dispersion_autopick/
```

其中：

### 3.1 `Code/02_cc_stack_and_spike_removal`

| 文件 | 作用 |
|---|---|
| `run_2014_1d_wang_pws.py` | 2014 年 1D Wang/Lin 互相关与 PWS 叠加的主入口。 |
| `wang_1d_pws.py` | 上述主入口的核心函数、参数 dataclass 和 PWS 实现。 |
| `run_pipeline_2014_1d_wang_pws.sh` | 服务器上实际串联 `manifest -> correlate -> export -> audit -> rsync` 的 pipeline 脚本快照。 |
| `watchdog_2014_1d_wang_pws.sh` | 上游互相关/PWS 生产过程的监控脚本快照。 |
| `detect_1s_spikes_wang_style.py` | 识别 1 s 周期固定相位尖峰、构建模板、输出诊断图。 |
| `apply_spike_removal_moveout_compare.py` | 在 moveout 子集上做 before/after 对比，并缓存数据。 |
| `apply_spike_removal_to_all_stacks.py` | 把去尖峰模板批量作用到全部 `STACK` H5。 |

### 3.2 `Code/04_dispersion_autopick`

| 文件 | 作用 |
|---|---|
| `convert_1d_stack_to_dat.py` | 把叠加后的 H5 波形转成 EGFAnalysis/DisperPicker 可读的 DAT。 |
| `run_dispersion_mi09.py` | CPU 版自动提取脚本。 |
| `run_dispersion_gpu_mi09.py` | GPU 版自动提取脚本。当前 2014 全量结果主要看这个版本。 |
| `export_dispersion_npz_gpu_only.py` | 仅回填 `.npz` 的 GPU 脚本；不再重跑最终 GDisp/CDisp。 |
| `gpu_dispersion_backend.py` | GPU 版群速度/相速度图构建后端。 |
| `verify_disperpicker_full_run.py` | 用于全量结果的完整性检查。 |
| `watch_finalize_wang_disperpicker.sh` | 后处理/验证/恢复/同步的监控脚本快照。 |
| `EGFAnalysisPy/EGFAnalysisTimeFreq.py` | EGFAnalysis 核心实现。 |
| `EGFAnalysisPy/config.py` | EGFAnalysis 配置默认值。 |
| `EGFAnalysisPy/DisperPicker/pick.py` | DisperPicker CNN 拾取逻辑。 |
| `EGFAnalysisPy/DisperPicker/qc.py` | 拾取后的质量控制逻辑。 |
| `EGFAnalysisPy/DisperPicker/train_cnn.py` | DisperPicker CNN 网络定义。 |
| `EGFAnalysisPy/DisperPicker/plot/` | `pick.py` 依赖的绘图辅助模块。 |
| `EGFAnalysisPy/DisperPicker/reader/` | DisperPicker 数据读取辅助模块。 |
| `EGFAnalysisPy/DisperPicker/tflib/` | DisperPicker TensorFlow 层实现。 |
| `EGFAnalysisPy/DisperPicker/saver/` | DisperPicker 预训练模型权重与 `checkpoint` 文件。 |
| `rebuild_04dispersion_from_03cc_stackdata.sh` | 直接从 `03CC_StackData` 重建 `04DispersionData` 的便捷脚本。 |

### 3.3 `Environment`

这个目录保存的是从 `work` 服务器直接导出的环境快照，便于之后尽可能还原当时的运行环境。

关键文件包括：

| 文件 | 作用 |
|---|---|
| `README_SERVER_ENVIRONMENTS.md` | 环境总说明。 |
| `wang_pws_summary.json` | 互相关/PWS 环境核心版本摘要。 |
| `ftan_summary.json` | CPU 版频散提取环境摘要。 |
| `disp_gpu_summary.json` | GPU 版频散提取环境摘要。 |
| `*_pip_freeze.txt` | 三套环境的完整包版本清单。 |
| `disp_gpu_tensorflow_build_info.json` | TensorFlow CUDA / cuDNN 构建信息。 |
| `disp_gpu_cupy_show_config.txt` | CuPy / CUDA 运行配置。 |
| `nvidia_smi.txt` | 服务器 GPU、驱动和显存信息。 |
| `nvcc_version.txt` | 系统 `nvcc` 版本。 |

## 4. 自动频散提取的结果链路

当前 `04DispersionData` 中的 2014/1D 结果，实际处理链如下：

```text
预处理后的 1D 日波形
-> 2014 1D Wang/Lin hourly XCORR
-> PWS 叠加，得到 stack_pws.h5
-> 可选：去除 1 s 固定相位尖峰
-> H5 -> DAT
-> EGFAnalysis + DisperPicker 自动提取 GDisp / CDisp
-> 可选：单独回填 DispersionNPZ
-> 同步到 Lenovo 的 04DispersionData
```

## 5. 互相关与叠加参数

这一部分是频散提取的上游参数，必须记录，因为 DAT 与频散图都直接依赖它。

### 5.1 上游生产代码

上游 2014 1D 互相关和叠加生产脚本是：

```text
Code/02_cc_stack_and_spike_removal/run_2014_1d_wang_pws.py
Code/02_cc_stack_and_spike_removal/wang_1d_pws.py
```

服务器 pipeline 脚本快照中记录的运行入口是：

```text
/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/run_pipeline.sh
```

该脚本内部顺序是：

```text
manifest -> correlate --resume --fft-threads 4 -> export -> audit -> rsync 到 lenovo
```

### 5.2 输入数据

| 项目 | 路径 |
|---|---|
| 预处理后 mseed | `/mnt/data_hdd/lgx/MSH_ANT/staging/2014_1D_new_preprocessed/mseed_25Hz_resp_vel_prefilt_0p005_0p01_10_12` |
| StationXML | `/mnt/data_hdd/lgx/MSH_ANT/staging/2014_1D_new_preprocessed/xml` |
| 原始 PWS stack 输出 | `/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK` |
| 去尖峰后 stack 输出 | `/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK_SPIKE_REMOVED_DIAGFIT_20260628` |

### 5.3 互相关参数

`wang_1d_pws.py` 中的 `WangPwsConfig` 给出的参数如下：

| 参数 | 值 |
|---|---:|
| `network` | `1D` |
| `channel` | `DPZ` |
| `start_date` | `2014-07-18` |
| `end_date` | `2014-08-05` |
| `sampling_rate` | `25.0 Hz` |
| `cc_len` | `3600 s` |
| `step` | `1800 s` |
| overlap | `1800 s`，即 50% overlap |
| `inc_hours` | `6` |
| 每完整日窗口数 | `47` |
| 每日 block 窗口数 | `[12, 12, 12, 11]` |
| `maxlag` | `150 s` |
| 输出点数 | `7501` |
| `freqmin` | `0.2 Hz` |
| `freqmax` | `10.0 Hz` |
| `freq_norm` | `RMA` |
| `smoothspect_N` | `40` |
| `time_norm` | `NO` |
| `smooth_N` | `40` |
| `cc_method` | `XCORR` |
| `rm_resp` | `NO`，因为输入已经是去响应后的速度波形 |
| `max_over_std` | `90` |
| `substack` | `True` |
| `substack_windows` | `1` |
| `hourly_normalization` | `post_cc_maxabs` |

### 5.4 每个小时窗的处理步骤

对每个台站、每天、每个 1 小时窗，脚本内部实际做的是：

1. 读取采样率为 `25 Hz` 的 `DPZ` 速度波形。
2. 截取 `3600 s` 窗，按 `1800 s` 滑动。
3. 若窗口含 NaN 或长度不完整，则跳过该窗。
4. 对窗数据进行：
   - 去均值；
   - 线性去趋势；
   - `5%` cosine taper。
5. 做 RFFT。
6. 仅在 `0.2-10 Hz` 频带内做 RMA 白化：

```python
spectrum_white = spectrum / moving_average(abs(spectrum), smoothspect_N)
```

7. 两个台站做互相关：

```python
cc = irfft(conj(source_spectrum) * receiver_spectrum)
cropped = concatenate(cc[-nlag:], cc[:nlag + 1])
```

8. 每条小时互相关再做一次 `post_cc_maxabs` 归一化。

### 5.5 叠加参数

| 参数 | 值 |
|---|---:|
| `stack_method` | `PWS` |
| `pws_power` | `2.0` |
| 叠加对象 | 每个小时窗的互相关结果 |
| 输出 H5 数据集 | `AuxiliaryData/Allstack_pws/ZZ` |

等效公式：

```python
linear = mean(rows, axis=0)
phase = analytic_signal(row) / abs(analytic_signal(row))
weight = abs(mean(phase, axis=0)) ** pws_power
stack = linear * weight
```

## 6. H5 -> DAT 转换参数

使用脚本：

```text
Code/04_dispersion_autopick/convert_1d_stack_to_dat.py
```

DAT 格式：

```text
line 1: source lon lat
line 2: receiver lon lat
line 3+: time_s positive_lag negative_lag_reversed
```

具体转换逻辑：

```python
nlag = round(maxlag / dt)
green_ab = data[nlag + 1 :]
green_ba = data[nlag - 1 :: -1]
time_axis = np.arange(1, nlag + 1) * dt
```

关键点：

1. 使用的是正 lag 一侧和反转后的负 lag 一侧。
2. 每个 DAT 文件内部会用 `max(abs(green_ab), abs(green_ba))` 做一次幅值归一化。
3. 支持分片并行：
   - `--num-shards`
   - `--shard-index`
4. 当前全量生产可以使用 `--allow-zero-ngood`，即便属性里 `ngood=0`，只要波形非零也允许转出 DAT。
5. 可选支持：
   - `--stationxml-dir`
   - `--min-distance-km`
   - `--max-distance-km`

## 7. 自动频散提取参数

### 7.1 当前归档代码版本

请以 `Code/04_dispersion_autopick/run_dispersion_gpu_mi09.py` 为当前 2014 全量结果的主口径。

同时需要区分两类运行：

1. `2026-07-01 / 2026-07-02` 的原始全量曲线生产。
2. `2026-07-09` 的未去尖峰 NPZ-only 回填。

二者共享同一套 EGFAnalysis/GPU 图像构建参数，但第二类不再重跑最终曲线。

### 7.2 EGFAnalysis / 图像构建参数

`run_dispersion_gpu_mi09.py` 中 `CONFIG` 的值如下：

| 参数 | 值 | 说明 |
|---|---:|---|
| `isEGF` | `False` | 输入是互相关函数，不是直接 EGF。 |
| `StartT` | `0.2 s` | 最短周期。 |
| `EndT` | `5.0 s` | 最长周期。 |
| `DeltaT` | `0.1 s` | 周期步长。 |
| `StartV` | `2.0 km/s` | 配置中的最低速度。 |
| `EndV` | `4.0 km/s` | 最高速度。 |
| `DeltaV` | `0.005 km/s` | 速度采样间隔。 |
| `MinDist` | `10.0 km` | 只做日志提示，不再跳过短距台对。 |
| `WinAlpha` | `0.1` | 主信号窗 Tukey 余弦比例。 |
| `NoiseTime` | `5.0 s` | 噪声窗长度。 |
| `MinSNR` | `5.0` | EGFAnalysis 内部最小信噪比阈值。 |
| `WinPeriodNum` | `5` | 时变滤波窗口周期数。 |
| `WinMinTime` | `5 s` | 时变滤波最小时窗长度。 |
| `FilterKaiserPara` | `6` | Kaiser 窗参数。 |
| `MaxFilterLengthLog` | `14` | 最大滤波长度的 `log2`。 |

周期轴点数固定为：

```text
(5.0 - 0.2) / 0.1 + 1 = 49
```

DisperPicker CNN 固定输入速度轴是：

```text
0.5-4.0 km/s, step=0.005 km/s, 共 701 点
```

因此即便当前配置 `StartV=2.0`，脚本仍会把真实计算图像 pad 到 `701 × 49` 后再送入 CNN。

### 7.3 相速度参考周期参数

如果不手动指定 `--phase_ref_t`，脚本会按台间距自动生成相速度追踪起始列：

```python
auto_index = int(
    0.6 * min(
        n_period - 1,
        round((distance_km / 1.5 / 3.2 - start_t) / delta_t),
    )
)
```

可选参数：

| 参数 | 作用 |
|---|---|
| `--phase_ref_t` | 手动固定参考周期。 |
| `--phase_ref_t_max` | 限制自动参考周期的上限。 |

### 7.4 两轮 CNN 拾取参数

第一轮：

| 项目 | 值 |
|---|---:|
| 目的 | 先获得群速度种子，用来驱动时变滤波后的相速度图。 |
| `pick.mean_confidence_G` | `0` |
| `ct` | `0.01` |

第二轮：

| 项目 | 值 |
|---|---:|
| 目的 | 生成最终 GDisp / CDisp 结果。 |
| `pick.mean_confidence_G` | `0.3` |
| `ct` | 脚本默认 `2.0`，但当前 `curves_all_finalct001` 目录实际运行使用的是 `0.01`。 |

这里一定要注意：

```text
curves_all_finalct001
```

这个目录名明确说明当前同步进来的最终曲线结果是按：

```text
--final_ct 0.01
```

跑出来的，而不是脚本默认值 `2.0`。

### 7.5 QC 参数

群速度 QC：

| 参数 | 值 |
|---|---:|
| `v_range` | `[2.0, 4.0]` |
| `diff_range` | `[-0.07, 0.08]` |
| `upward` | `0.4` |
| `each_stage_upward` | `0.3` |
| `min_len` | `round(49 / 5) = 10` |
| `skip` | `False` |

相速度 QC：

| 参数 | 值 |
|---|---:|
| `v_range` | `[2.0, 4.0]` |
| `diff_range` | `[-0.1, 0.1]` |
| `upward` | `0` |
| `each_stage_upward` | `0.2` |
| `min_len` | `10` |
| `skip` | `True` |

### 7.6 输出文件格式

每个曲线文件格式如下：

```text
lon1 lat1
lon2 lat2
period_s velocity_km_s snr confidence
```

也就是：

| 列号 | 含义 |
|---|---|
| 1 | 周期（秒） |
| 2 | 速度（km/s） |
| 3 | SNR |
| 4 | 该周期点的拾取置信度 |

## 8. 去除尖峰说明

这一部分对应 `RemoveSpikes` 结果，是整个 2014 频散结果链中非常重要的一步。

### 8.1 使用的代码

核心脚本：

| 文件 | 作用 |
|---|---|
| `Code/02_cc_stack_and_spike_removal/detect_1s_spikes_wang_style.py` | 从距离分箱诊断图中识别 1 s 周期尖峰并构建模板。 |
| `Code/02_cc_stack_and_spike_removal/apply_spike_removal_moveout_compare.py` | 在 moveout 子集上演示 before/after，并保存缓存。 |
| `Code/02_cc_stack_and_spike_removal/apply_spike_removal_to_all_stacks.py` | 把模板应用到全部 `STACK` H5。 |

### 8.2 模板是怎么构建的

模板构建流程如下：

1. 从 `STACK` 目录遍历所有唯一台对。
2. 用蓄水池抽样抽取最多 `500000` 条候选路径：
   - `size=500000`
   - `seed=20260619`
3. 读取样本后对每条台对先做归一化。
4. 按距离做分箱，默认箱宽：

```text
0.5 km
```

5. 从距离分箱后的 coherent trace 中估计“每秒内固定相位最强的位置”。
6. 以该相位为中心，在 `1-15 s` 的每个整数秒附近截取小窗。
7. 对这些窗取中位数，构成一个重复尖峰模板。

模板构建关键参数：

| 参数 | 值 |
|---|---:|
| `template_source` | `diagnostic` |
| `start_second` | `1` |
| `end_second` | `15` |
| `half_width_s` | `0.16 s` |
| `seed` | `20260619` |
| 样本上限 | `500000` |

模板构建代码核心：

```python
offsets, template = build_repeating_spike_template(
    time,
    template_reference,
    best_phase_s,
    start_second=1,
    end_second=15,
    half_width_s=0.16,
)
```

### 8.3 每条 stack 如何扣除尖峰

对每条叠加波形，不是重新做互相关，而是在已经叠加好的 `stack_pws.h5` 上直接减模板。

对单条波形的振幅缩放因子用最小二乘一维拟合：

```python
scale = dot(observed, template) / dot(template, template)
```

然后在正 lag 和负 lag 对称位置同时减去：

```python
corrected[pos_index] -= scale * amplitude
corrected[neg_index] -= scale * amplitude
```

也就是说：

1. 尖峰模板的形状固定。
2. 每个台对只拟合一个标量 `scale`。
3. 只改叠加后的波形，不回溯重跑上游互相关和 PWS。

### 8.4 去尖峰后写出的内容

全量输出目录：

```text
/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK_SPIKE_REMOVED_DIAGFIT_20260628
```

每个 H5 在原有属性基础上新增：

| 属性 | 含义 |
|---|---|
| `spike_removed=YES` | 已执行去尖峰。 |
| `spike_phase_s` | 固定相位位置。 |
| `spike_template_source` | 模板来源；本批次为 `diagnostic`。 |
| `spike_scale` | 该台对拟合得到的模板缩放因子。 |

此外还会写出：

| 文件 | 含义 |
|---|---|
| `spike_scales.csv` | 每条台对的缩放因子清单。 |
| `spike_template.csv` | 模板形状数值。 |
| `report.html` | 全量处理摘要。 |

### 8.5 去尖峰不会改变什么

以下参数在去尖峰前后都不变：

- `cc_len`
- `step`
- `maxlag`
- `freqmin`
- `freqmax`
- `freq_norm`
- `time_norm`
- `smoothspect_N`
- `pws_power`
- `stack_method`

它只是叠加后波形的一个后处理步骤。

## 9. 代码版本与运行指南

## 9.0 现在这份 Reports 是否已经足够复现

和上一版整理相比，这次已经额外补齐了下面这些原先缺失、但实际运行必需的资产：

- `DisperPicker` 的 `train_cnn.py`
- `DisperPicker/plot/`
- `DisperPicker/reader/`
- `DisperPicker/tflib/`
- `DisperPicker/saver/` 里的模型权重和 `checkpoint`
- 服务器三套 Python 环境的版本导出
- 服务器 GPU / CUDA / 驱动信息
- 便捷重建脚本 `rebuild_04dispersion_from_03cc_stackdata.sh`

因此在“代码和模型文件是否齐全”这个层面，现在的 `Reports` 已经比上一版完整得多，可以支持后续读者重新搭环境并重跑。

## 9.1 服务器环境口径

需要区分三类运行环境：

| 阶段 | 主要脚本 | 环境/解释器 |
|---|---|---|
| 互相关 + PWS 叠加 | `run_2014_1d_wang_pws.py` | `run_pipeline.sh` 里记录的是 `/mnt/data_hdd/lgx/MSH_ANT/.venvs/2014_wang_pws/bin/python`，同时 `conda activate noise`。 |
| CPU 版频散提取/校验 | `run_dispersion_mi09.py`、`verify_disperpicker_full_run.py` | 相关脚本快照中常见 `PY=/mnt/data_hdd/lgx/MSH_ANT/envs/ftan/bin/python`。 |
| GPU 版频散提取/NPZ 回填 | `run_dispersion_gpu_mi09.py`、`export_dispersion_npz_gpu_only.py` | `disp_gpu` 环境；日志中可见 TensorFlow/CuPy 来自 `/mnt/data_hdd/lgx/MSH_ANT/envs/disp_gpu/`。 |

系统级摘要见：

```text
Environment/README_SERVER_ENVIRONMENTS.md
```

快速摘要如下：

| 项目 | 值 |
|---|---|
| OS | `Ubuntu 24.04.4 LTS` |
| Kernel | `Linux 6.8.0-124-generic` |
| GPU | `NVIDIA GeForce RTX 5090 D` |
| Driver | `595.71.05` |
| `nvidia-smi` 报告 CUDA | `13.2` |
| `nvcc --version` | `12.0.140` |
| TensorFlow 2.21 build CUDA | `12.5.1` |
| TensorFlow 2.21 build cuDNN | `9` |

## 9.2 上游 2014 互相关 + PWS 的运行方式

服务器已有 pipeline 脚本快照：

```bash
/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/run_pipeline.sh
```

其核心命令顺序是：

```bash
run_python manifest
run_python correlate --resume --fft-threads 4
run_python export
run_python audit
rsync 到 lenovo
```

资源分配口径：

| 参数 | 口径 |
|---|---|
| `--fft-threads` | `4` |
| `OMP_NUM_THREADS` | `1` |
| `OPENBLAS_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `NUMEXPR_NUM_THREADS` | `1` |
| `SCIPY_NUM_THREADS` | 由脚本内部设置为 `fft_threads` |
| `--workers` | 如不手动指定，由 `choose_worker_count(os.cpu_count())` 自动决定 |

这套设置的思想是：

1. 进程级并发负责“按 source 台站分任务”。
2. 单进程内部 FFT 保持有限线程，避免 CPU 线程过度竞争。

## 9.3 H5 -> DAT 运行示例

未去尖峰 stack 转 DAT：

```bash
/mnt/data_hdd/lgx/MSH_ANT/envs/ftan/bin/python \
  /mnt/data_hdd/lgx/MSH_ANT/code/MSH_ANT/scripts/04_dispersion/convert_1d_stack_to_dat.py \
  --stack-root /mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK \
  --stationxml-dir /mnt/data_hdd/lgx/MSH_ANT/data/metadata/2014/1D \
  --out-dir /mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_unspiked_20260702/dat_all \
  --source-glob '1D.*' \
  --receiver-glob '1D.*' \
  --allow-zero-ngood
```

去尖峰 stack 转 DAT，只需把 `--stack-root` 改为：

```text
/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK_SPIKE_REMOVED_DIAGFIT_20260628
```

如需并行转换，可同时启动多个 shard：

```bash
for i in $(seq 0 15); do
  /mnt/data_hdd/lgx/MSH_ANT/envs/ftan/bin/python \
    /mnt/data_hdd/lgx/MSH_ANT/code/MSH_ANT/scripts/04_dispersion/convert_1d_stack_to_dat.py \
    --stack-root ... \
    --stationxml-dir ... \
    --out-dir ... \
    --allow-zero-ngood \
    --num-shards 16 \
    --shard-index "$i" \
    > logs/dat_shard_${i}.log 2>&1 &
done
wait
```

## 9.4 GPU 全量自动提取运行示例

### A. 原始全量曲线生产口径

从 2026-07-01 / 2026-07-02 的日志可确认：

- 使用 `run_dispersion_gpu_mi09.py`
- `backend=cupy`
- `StartV=2.0`
- `final_ct=0.01`
- `resume_existing=True`
- 实际是 `16` 个 shard 并发

日志目录名称已经直接记录了这一点：

```text
logs_gpu_finalct001_16w_20260701
logs_gpu_finalct001_16w_20260702
```

等效命令示例如下：

```bash
for i in $(seq 0 15); do
  CUDA_VISIBLE_DEVICES=0 \
  /mnt/data_hdd/lgx/MSH_ANT/envs/disp_gpu/bin/python \
    /mnt/data_hdd/lgx/MSH_ANT/code/MSH_ANT/scripts/04_dispersion/run_dispersion_gpu_mi09.py \
    --dat_dir /mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_20260701/dat_all \
    --out_dir /mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_20260701/curves_all_finalct001 \
    --skip_qc_plot \
    --backend cupy \
    --final_ct 0.01 \
    --num_shards 16 \
    --shard_index "$i" \
    --resume_existing \
    > /mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_20260701/logs_gpu_finalct001_16w_20260701/shard_${i}.log 2>&1 &
done
wait
```

未去尖峰版本只需把 `dat_dir` 和 `out_dir` 改成：

```text
/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_unspiked_20260702/dat_all
/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_unspiked_20260702/curves_all_finalct001
```

### B. 2026-07-09 的 NPZ-only 回填口径

未去尖峰版本最初没有成套 `full_pixel_data_all`，因此后面使用：

```text
Code/04_dispersion_autopick/export_dispersion_npz_gpu_only.py
```

单独回填 `.npz`，而不再重复生成曲线。等效命令示例如下：

```bash
for i in $(seq 0 23); do
  CUDA_VISIBLE_DEVICES=0 \
  /mnt/data_hdd/lgx/MSH_ANT/envs/disp_gpu/bin/python \
    /mnt/data_hdd/lgx/MSH_ANT/code/MSH_ANT/scripts/04_dispersion/export_dispersion_npz_gpu_only.py \
    --dat_dir /mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_unspiked_20260702/dat_all \
    --energy_dir /mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_allpairs_unspiked_20260702_backfill_gpu_20260709/full_pixel_data_all \
    --backend cupy \
    --num_shards 24 \
    --shard_index "$i" \
    --resume_existing \
    > logs/npz_only_shard_${i}.log 2>&1 &
done
wait
```

这个回填口径与原始 16 shard 曲线生产不同：

| 项目 | 原始曲线生产 | NPZ-only 回填 |
|---|---|---|
| 脚本 | `run_dispersion_gpu_mi09.py` | `export_dispersion_npz_gpu_only.py` |
| 是否输出 GDisp/CDisp | 是 | 否 |
| 是否输出 NPZ | 可选 | 只输出 NPZ |
| 2026-07-09 实际并发口径 | 无 | `24` shards |

## 9.5 从 03CC_StackData 直接重建 04DispersionData

这次额外放入了一个便捷脚本：

```text
Code/04_dispersion_autopick/rebuild_04dispersion_from_03cc_stackdata.sh
```

这个脚本默认直接使用：

```text
/mnt/data_hdd/MSH_ANT_Final/03CC_StackData
/mnt/data_hdd/MSH_ANT_Final/04DispersionData
```

因此它的目标就是从你现在整理好的 `03` 和 `04/Reports` 出发，重建：

- `DatData/dat_all`
- `Curves/curves_all_finalct001`
- `DispersionNPZ` 或 `DispersionNPZ/full_pixel_data_all`

常用示例：

```bash
# 未去尖峰版本：重建 DAT + 曲线 + NPZ
bash Code/04_dispersion_autopick/rebuild_04dispersion_from_03cc_stackdata.sh all-unspiked

# 去尖峰版本：重建 DAT + 曲线 + NPZ
bash Code/04_dispersion_autopick/rebuild_04dispersion_from_03cc_stackdata.sh all-spiked
```

可配置环境变量：

| 变量 | 作用 | 默认值 |
|---|---|---|
| `FINAL_ROOT` | 顶层根目录 | `/mnt/data_hdd/MSH_ANT_Final` |
| `PY_FTAN` | DAT 转换解释器 | `/mnt/data_hdd/lgx/MSH_ANT/envs/ftan/bin/python` |
| `PY_GPU` | GPU 频散提取解释器 | `/mnt/data_hdd/lgx/MSH_ANT/envs/disp_gpu/bin/python` |
| `BACKEND` | `cupy` 或 `numpy` | `cupy` |
| `SHARDS` | 并行 shard 数 | `16` |
| `FINAL_CT` | 最终 DisperPicker `ct` | `0.01` |
| `RESUME_EXISTING` | 是否启用 `--resume_existing` | `1` |

## 9.6 验证和同步

全量结果校验脚本：

```text
Code/04_dispersion_autopick/verify_disperpicker_full_run.py
```

它检查的内容包括：

1. 每个 DAT 是否都有 `GDisp`、`CDisp`、`NPZ`。
2. 随机抽样的曲线文件列数、行数、数值格式是否正确。
3. `.npz` 是否包含必要字段，且图像尺寸是否是 `701 × 49`。
4. 日志里是否出现 `处理失败`、`Traceback` 等错误。

后处理/恢复/同步脚本快照：

```text
Code/04_dispersion_autopick/watch_finalize_wang_disperpicker.sh
```

注意：这个脚本是后期“补跑/验证/rsync”的辅助脚本快照，其中默认 `SHARDS=24`，主要用于恢复和收尾，不应和 2026-07-01/02 的原始 `16` shard 曲线生产口径混为一谈。

## 10. 历史版本差异说明

仓库里还保留过更早的 DisperPicker 说明文档和旧日志，其中有过一套常见口径：

| 参数 | 旧口径 | 当前 `04DispersionData` 口径 |
|---|---:|---:|
| `StartV` | `1.0 km/s` | `2.0 km/s` |
| `final_ct` | 旧结果各异 | 当前同步结果明确是 `0.01` |
| 目录名 | 例如 `wang_1d1d_2014_ge18` | 当前是 `RemoveSpikes` / `NonRemoveSpikes` |

因此后续如果有人看到旧文档里的 `StartV=1.0`，不要直接套用到本目录。  
对本目录的参数追溯，应当以 `Reports/Code` 中归档的服务器脚本快照和本 README 为准。

## 11. 可复现性的边界

现在这份 `Reports` 已经尽量补成“接近自包含”的离线重建包，但仍要区分两层含义：

### 11.1 现在可以做到的

如果后续读者：

1. 按 `Environment/` 里的版本说明重建出足够接近的三套环境；
2. 使用 `03CC_StackData` 里的原始 / 去尖峰 H5；
3. 使用 `Reports/Code` 里的完整脚本和 DisperPicker 模型；

那么就可以重新生成：

- `DAT`
- `GDisp/CDisp`
- `NPZ`

也就是功能上可以复现 `04DispersionData` 的全部核心结果。

### 11.2 仍然不应承诺的

即使环境尽量对齐，也不应承诺以下两点：

1. 所有 `.txt` / `.npz` 文件逐字节完全一致。
2. 不同 GPU、驱动或 TensorFlow/CuPy 组合下毫无任何浮点差异。

更现实也更专业的表述应当是：

```text
结果等价复现
```

而不是：

```text
二进制级完全一致复现
```
