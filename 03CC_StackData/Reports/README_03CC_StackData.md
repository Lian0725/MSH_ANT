# 03CC StackData 报告、代码与 Moveout 缓存说明

## 1. 目录用途

本目录：

- `/mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports`

用于集中保存 2014 年 `1D` 台网互相关与叠加结果的说明文档、生产代码副本、moveout 图片，以及后续快速重绘所需的 `.npz` 缓存文件。

整理后的目录结构约定为：

- 外层保留：
  - `README_03CC_StackData.md`
  - `report.html`
- `code/`
  - 保存互相关、叠加、去尖峰和 moveout 重绘脚本副本。
- `images/`
  - 保存 PNG/JPG 图片、`.npz` 缓存以及 CSV 结果表。

这次整理的目标是：

- 把 2014 年这批 `1D` 数据实际生产用的互相关/叠加代码放进本目录。
- 把 moveout 作图代码也放进本目录，并让它优先使用同目录依赖，尽量自包含。
- 把原来的 `README-03-cc-stackdata.md` 改名为 `README_03CC_StackData.md`。
- 详细记录互相关、叠加和 moveout 绘图参数。
- 后续若仅修改图片风格，不再回读整批 H5，而是直接基于 `.npz` 重绘。

## 2. 当前目录内文件角色

### 2.1 moveout 与缓存相关

- `code/apply_spike_removal_moveout_compare.py`
  - moveout 主绘图脚本。
  - 支持两种模式：
    - 全流程模式：从 H5 读数据，写 `.npz`，再出图。
    - 缓存重绘模式：直接从 `.npz` 读取并重绘 PNG/HTML。

- `code/export_moveout_cache_from_existing_result.py`
  - 把已有的历史结果目录转换成 `.npz` 缓存。
  - 不重新跑 spike removal。
  - 不重新扫描全量 2014 STACK。
  - 只读取已筛好的 moveout 子集。

- `code/reproduce_wang_figure2_1d4529.py`
  - moveout 绘图的直接依赖。
  - 负责 record section 的三个滤波面板绘制、红色速度参考线和红色速度标签。

- `code/detect_1s_spikes_wang_style.py`
  - spike removal 的直接依赖。
  - 负责 1 s 重复尖峰的诊断、模板构建和拟合。

- `images/moveout_before_after_data.npz`
  - moveout 的数值缓存。
  - 后续改颜色、加参考线、调整版式时优先读取这个文件，而不是重新回读 H5。

### 2.2 2014 互相关 / 叠加生产代码

- `code/run_2014_1d_wang_pws.py`
  - 2014 `1D` 数据实际生产用的主入口脚本。
  - 负责 `manifest -> correlate -> export -> audit`。

- `code/wang_1d_pws.py`
  - 上述入口脚本的核心实现。
  - 定义配置参数、建窗规则、PWS 累加器、manifest 构建、checkpoint 写读等。

- `code/run_pipeline.sh`
  - 服务器实际调度脚本。
  - 负责激活环境、运行 `manifest -> correlate --resume --fft-threads 4 -> export -> audit`，并把结果 `rsync` 到 lenovo。

### 2.3 结果文件

- `images/moveout_before.png`
- `images/moveout_after.png`
- `images/moveout_before_after_compare.png`
- `images/receivers_used.csv`
- `images/spike_scales.csv`
- `report.html`

### 2.4 Wang Figure 3 四联图资产

- `code/wang_figure3_uniform/render_wang_figure3_from_npz.py`
  - 从 4 个 NPZ 直接重绘 panel A-D，并输出统一尺寸的四联图。

- `code/wang_figure3_uniform/compose_wang_figure3_panels.py`
  - 把已有的 4 张单图重新拼成统一 2×2 组合图。

- `code/wang_figure3_uniform/panel_data_index.md`
  - 记录 panel A-D 各自对应的原始图片路径、NPZ 路径和数组键名。

- `images/wang_figure3_uniform/panel_a.png`
- `images/wang_figure3_uniform/panel_b.png`
- `images/wang_figure3_uniform/panel_c.png`
- `images/wang_figure3_uniform/panel_d.png`
- `images/wang_figure3_uniform/wang_figure3_four_panel_uniform.png`

- `images/wang_figure3_uniform/wang_figure3a_panel_data.npz`
- `images/wang_figure3_uniform/subset_strict_data.npz`
- `images/wang_figure3_uniform/distance_bin_wiggle_panel_data.npz`
- `images/wang_figure3_uniform/wang_figure3d_bandpassed_fill_scaled_2p00_no_fill_data.npz`

## 3. 2014 这批互相关与叠加数据的实际生产链条

### 3.1 生产入口

实际生产入口脚本：

- `run_2014_1d_wang_pws.py`

对应生产调度脚本：

- `run_pipeline.sh`

### 3.2 输入数据

预处理后的输入波形：

- `/mnt/data_hdd/lgx/MSH_ANT/staging/2014_1D_new_preprocessed/mseed_25Hz_resp_vel_prefilt_0p005_0p01_10_12`

StationXML：

- `/mnt/data_hdd/lgx/MSH_ANT/staging/2014_1D_new_preprocessed/xml`

### 3.3 输出目录

生产输出根目录：

- `/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620`

其中关键子目录：

- `manifest/`
- `checkpoints/`
- `logs/`
- `qc/`
- `STACK/`

同步到 lenovo 后的结果目录：

- `/mnt/data_hdd/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620`

### 3.4 生产步骤

`run_pipeline.sh` 中实际执行顺序：

1. `manifest`
2. `correlate --resume --fft-threads 4`
3. `export`
4. `audit`
5. `rsync` 到 lenovo

## 4. 互相关与叠加参数

下面的参数来自 `wang_1d_pws.py` 里的 `WangPwsConfig`，以及 `run_pipeline.sh`。

### 4.1 基本数据参数

- 台网：`1D`
- 分量：`DPZ`
- 时间范围：`2014-07-18` 到 `2014-08-05`
- 采样率：`25.0 Hz`
- 采样间隔：`0.04 s`
- 输入数据类型：预处理后的速度波形

### 4.2 互相关建窗参数

- `cc_len = 3600 s`
  - 单个互相关窗长度为 1 小时。

- `step = 1800 s`
  - 窗滑动步长为 30 分钟。

- overlap = `1800 / 3600 = 50%`
  - 即相邻 1 小时窗重叠 50%。

- 每天完整窗口数：`47`
  - 对应起始时间从 `00:00` 到 `23:00`，每 30 分钟一个窗。

- `inc_hours = 6`
  - 计算时按 6 小时为一个 block 组织。
  - 每天分成 4 个 6 小时 block。
  - `build_day_blocks()` 会把 `read_end` 延长一个 `step`，用于保证 block 边界处窗覆盖连续。

### 4.3 最大延时与输出长度

- `maxlag = 150 s`
  - 最终只保留 `[-150 s, +150 s]`。

- 输出点数：
  - `2 * round(maxlag / dt) + 1`
  - 这里 `dt = 0.04 s`
  - 即 `2 * 3750 + 1 = 7501` 点。

### 4.4 互相关方法

- `cc_method = "XCORR"`
  - 采用频域互相关。

核心过程：

1. 对每个时间窗先做去均值、去线性趋势、余弦 taper。
2. 计算频谱。
3. 在指定频带内做白化。
4. 源台和接收台频谱做：
   - `conj(source) * receiver`
5. 反变换回时域。
6. 截取 `[-maxlag, +maxlag]`。

### 4.5 时域与频域归一化参数

- `time_norm = "NO"`
  - 互相关前不做时间域 one-bit / RMA 归一化。

- `freq_norm = "RMA"`
  - 频域白化使用运行平均振幅归一化。

- `smoothspect_N = 40`
  - 在实现中进入 `moving_average(values, half_width=40)`。
  - 实际平滑窗口宽度为：
    - `2 * 40 + 1 = 81` 点。

- `smooth_N = 40`
  - 配置中保留，但当前生产脚本主流程里没有单独作为时域归一化步骤使用，因为 `time_norm = "NO"`。

### 4.6 频带参数

- `freqmin = 0.2 Hz`
- `freqmax = 10.0 Hz`

注意：

- 这是互相关前频域白化/保留的工作频带。
- 它不是 moveout 图三面板显示时的带通周期范围。

### 4.7 每窗预处理细节

在 `prepare_window()` 中，每个 1 小时窗会做：

1. 去均值
2. 去线性趋势
3. `cosine_taper(fraction=0.05)`

即：

- taper 占窗长 `5%`
- 窗两端做余弦收边

### 4.8 异常窗控制

- `max_over_std = 90`

这是配置中记录的异常控制阈值，用于与 Wang/Lin 风格处理保持一致。

### 4.9 小时级互相关归一化

- `hourly_normalization = "post_cc_maxabs"`

也就是：

- 每个小时级 CCF 在进入最终叠加前，按该小时 CCF 的 `max(abs(CCF))` 归一化。

这一步通过：

- `maxabs_normalize_rows()`

实现。

### 4.10 叠加方式

- `stack_method = "PWS"`

即：

- 最终叠加采用 `Phase-Weighted Stack`

并且：

- `pws_power = 2.0`

`StreamingPWSAccumulator.finalize()` 的实际形式是：

1. 先对所有小时级 CCF 线性平均
2. 再根据解析信号相位一致性计算权重
3. 权重为：
   - `abs(mean(phase_unit(rows))) ** 2.0`
4. 最终：
   - `linear_stack * phase_weight`

### 4.11 substack 参数

- `substack = True`
- `substack_windows = 1`

当前这套生产脚本主要输出最终 `STACK`，并保留与历史流水线一致的 substack 配置字段。

### 4.12 checkpoint / resume

- `correlate` 阶段支持 `--resume`
- 每个源台站一个 checkpoint：
  - `checkpoints/<source>.h5`

checkpoint 中保存：

- 已累加的线性和
- 已累加的相位和
- `ngood`
- 已处理日期
- manifest hash

因此中断后可以续算。

### 4.13 并行与线程参数

`run_pipeline.sh` 中明确设置：

- `FFT_THREADS = 4`
- 运行参数：
  - `correlate --resume --fft-threads 4`

环境变量设置：

- `HDF5_USE_FILE_LOCKING=FALSE`
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `VECLIB_MAXIMUM_THREADS=1`

目的：

- 限制 BLAS/OpenMP 过度抢线程
- 让 FFT 线程数更可控

worker 数：

- `WORKERS=''`
- 即默认由代码自动选择

`choose_worker_count()` 的逻辑是：

- `max(1, min(24, cpu_count - 6))`

## 5. 导出结果参数

`export` 阶段会把每个台站对写为：

- `STACK/<source>/<receiver>/stack_pws.h5`

H5 中 `AuxiliaryData/Allstack_pws/ZZ` 关键属性包括：

- `dt`
- `maxlag`
- `ngood_hours`
- `stack_method = PWS`
- `pws_power = 2.0`
- `network = 1D`
- `component = ZZ`
- `station_source`
- `station_receiver`
- `input_hash`

## 6. moveout 图使用的是哪批结果

这次 moveout 图对应的历史结果目录是：

- `/mnt/data_hdd/MSH_ANT/parameter_tests/1d_moveout_before_after_spike_removal_20260625_fixed`

其原始未去尖峰 STACK 根目录是：

- `/mnt/data_hdd/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK`

其 cleaned 子集目录是：

- `/mnt/data_hdd/MSH_ANT/parameter_tests/1d_moveout_before_after_spike_removal_20260625_fixed/stack_cleaned_subset`

实际参与 moveout 的接收台数量：

- `172`

## 7. moveout 绘图参数

### 7.1 时间窗

- `lag_window = 15.0 s`

即绘图使用：

- `-15 s ~ +15 s`

### 7.2 三个显示面板

来自 `default_panel_specs()`：

1. `(a)` `1-10 s`
   - 周期带：`1.0–10.0 s`
   - 速度参考线：`7.0, 3.0, 1.6 km/s`

2. `(b)` `1.5-2.5 s`
   - 周期带：`1.5–2.5 s`
   - 速度参考线：`7.0, 3.0 km/s`

3. `(c)` `2.5-5 s`
   - 周期带：`2.5–5.0 s`
   - 速度参考线：`3.0, 1.6 km/s`

### 7.3 单道显示样式

- `amplitude_km = 0.9`
- `line_width = 0.32`
- `line_alpha = 0.72`
- `top_pad_km = 1.0`
- `bottom_pad_km = 0.75`

### 7.4 红色速度标签位置

这次已专门修改：

- 红色速度数字整体下移

原因：

- 避免与横轴刻度数字、线条末端和其他文字重合。

## 8. 为什么现在优先使用 NPZ

当前目录里保存：

- `images/moveout_before_after_data.npz`

这个文件保存的是：

- before / after 每条道的 `time_s`
- `window_trace`
- `distance_km`
- 接收台坐标
- 对应原始 H5 路径
- 对应 cleaned H5 路径
- 缩放因子等说明

因此后续如果只是：

- 改颜色
- 改线宽
- 加参考线
- 改红色速度标签位置
- 改标题和版式

只需要读取：

- `images/moveout_before_after_data.npz`

不再需要重新回读那 172 个 H5，更不需要重新扫描全量 2014 STACK。

## 9. 常用命令

### 9.1 从已有历史结果生成缓存

```bash
/home/lenovo/anaconda3/envs/noise/bin/python \
  /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/code/export_moveout_cache_from_existing_result.py \
  --existing-output /mnt/data_hdd/MSH_ANT/parameter_tests/1d_moveout_before_after_spike_removal_20260625_fixed \
  --cache-file /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/images/moveout_before_after_data.npz
```

### 9.2 只从 NPZ 重绘 moveout

```bash
/home/lenovo/anaconda3/envs/noise/bin/python \
  /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/code/apply_spike_removal_moveout_compare.py \
  --from-cache /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/images/moveout_before_after_data.npz \
  --output /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports
```

### 9.3 只从 NPZ 重绘 Wang Figure 3 四联图

```bash
python /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/code/wang_figure3_uniform/render_wang_figure3_from_npz.py \
  --panel-a /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/images/wang_figure3_uniform/wang_figure3a_panel_data.npz \
  --panel-b /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/images/wang_figure3_uniform/subset_strict_data.npz \
  --panel-c /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/images/wang_figure3_uniform/distance_bin_wiggle_panel_data.npz \
  --panel-d /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/images/wang_figure3_uniform/wang_figure3d_bandpassed_fill_scaled_2p00_no_fill_data.npz \
  --output /mnt/data_hdd/MSH_ANT_Final/03CC_StackData/Reports/images/wang_figure3_uniform
```

## 10. 当前目录最终应包含

- `README_03CC_StackData.md`
- `report.html`
- `code/run_2014_1d_wang_pws.py`
- `code/wang_1d_pws.py`
- `code/run_pipeline.sh`
- `code/apply_spike_removal_moveout_compare.py`
- `code/reproduce_wang_figure2_1d4529.py`
- `code/detect_1s_spikes_wang_style.py`
- `code/export_moveout_cache_from_existing_result.py`
- `images/moveout_before_after_data.npz`
- `images/moveout_before.png`
- `images/moveout_after.png`
- `images/moveout_before_after_compare.png`
- `images/integer_spike_contrast_before_after.png`
- `images/coherent_before_after.png`
- `images/spike_template.png`
- `images/receivers_used.csv`
- `images/spike_scales.csv`
- `code/wang_figure3_uniform/render_wang_figure3_from_npz.py`
- `code/wang_figure3_uniform/compose_wang_figure3_panels.py`
- `code/wang_figure3_uniform/panel_data_index.md`
- `images/wang_figure3_uniform/panel_a.png`
- `images/wang_figure3_uniform/panel_b.png`
- `images/wang_figure3_uniform/panel_c.png`
- `images/wang_figure3_uniform/panel_d.png`
- `images/wang_figure3_uniform/wang_figure3_four_panel_uniform.png`
- `images/wang_figure3_uniform/wang_figure3a_panel_data.npz`
- `images/wang_figure3_uniform/subset_strict_data.npz`
- `images/wang_figure3_uniform/distance_bin_wiggle_panel_data.npz`
- `images/wang_figure3_uniform/wang_figure3d_bandpassed_fill_scaled_2p00_no_fill_data.npz`
