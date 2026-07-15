# MSH_ANT — Mount St. Helens Ambient Noise Tomography

**作者**：Lian Guoxuan (`@Lian0725`)
**处理管线**：从原始地震波形到二维相速度反演的端到端流程
**参考文献**：Wang et al. (2017) *Ambient noise tomography across Mount St. Helens using a dense seismic array*

---

## 仓库范围

本仓库**只存放代码、关键最终图表与结果 CSV/NPZ**，不包含以下原始/中间数据：

| 未包含内容 | 位置 | 体积 |
|---|---|---|
| 原始地震波形 (.mseed) | `01RawData/{2014,2017}/` | ~541 GB |
| 预处理波形 | `02PreprocessData/{2014,2017}/` | ~90 GB |
| 互相关 + 叠加数据 (.h5) | `03CC_StackData/{2014,2017}/` | ~36 GB |
| 频散中间产物 (Curves/DatData/DispersionNPZ) | `04DispersionData/{2014,2017}/` | ~403 GB |

若需重建完整数据流，请按各阶段 `Reports/README_*.md` 里的说明本地重跑管线。

---

## 目录结构

```
MSH_ANT/
├── 01RawData/Reports/                  # 原始波形下载与元数据说明
├── 02PreprocessData/Reports/           # 预处理（去均值/去响应/降采样）说明
├── 03CC_StackData/                     # 互相关 + PWS 叠加
│   └── Reports/
│       ├── code/                       # 处理脚本
│       └── images/                     # moveout 汇总图
├── 04DispersionData/                   # DisperPicker CNN 频散提取
│   └── Reports/
│       ├── Code/
│       │   ├── 02_cc_stack_and_spike_removal/
│       │   └── 04_dispersion_autopick/  # 含 DisperPicker + EGFAnalysisPy 主流程
│       ├── Environment/                # 环境与依赖说明
│       ├── figures/                    # 汇总图表
│       └── scripts/                    # 辅助脚本
└── 05Inversion/                        # Wang Figure4 筛选 + 二维相速度反演
    ├── Report/2014/1D_1D/              # 汇总报告
    └── 2014/1D_1D/                     # 筛选后测量 CSV + 反演结果 NPZ
        ├── 01_去除尖峰数据/
        └── 02_不去除尖峰数据/
            ├── 01_筛选后数据/figure4_screening/
            └── 02_反演结果数据/phase_velocity_maps_3_3p5_4/
```

---

## 关键脚本

| 位置 | 功能 |
|---|---|
| `04DispersionData/Reports/Code/04_dispersion_autopick/run_dispersion_mi09.py` | DisperPicker CNN 频散拾取主入口 |
| `04DispersionData/Reports/Code/04_dispersion_autopick/convert_1d_stack_to_dat.py` | h5 stack → .dat 输入格式转换 |
| `05Inversion/Report/2014/1D_1D/02_代码/04_筛选脚本/plot_disperpicker_wang_figure4.py` | Wang 论文风格 Figure 4 筛选（`--paper-standard` 分支） |
| `05Inversion/Report/2014/1D_1D/02_代码/05_反演脚本/local_phase_velocity_maps.py` | 二维相速度反演 |

---

## 已修复的关键 Bug

### ① `ngood_hours` 字段名兼容
`convert_1d_stack_to_dat.py` 读取 HDF5 stack 时旧版为 `ngood`、新版为 `ngood_hours`，
现兼容读取：
```python
ngood = int(attrs.get("ngood_hours", attrs.get("ngood", 0)))
```

### ② DisperPicker CNN 权重预检
`run_dispersion_mi09.py` 启动前检查 `saver/{checkpoint, -10000.*}` 是否齐全，
避免因 rsync 白名单遗漏无扩展名 `checkpoint` 文件导致的 `FileNotFoundError`：
```python
_verify_disperpicker_saver()   # 缺失即 SystemExit + 打印修复命令
```

### ③ 20160715 — 04 频散重建可靠性改进

本次更新修复了 2014 1D 台网频散重建中“空输出仍被视为成功”的关键风险；仅修改
`04DispersionData` 的重建与验证代码，未改变 05 反演流程或既有结果数据。

- **重建输入路径统一为 `1D_1D`**：重建脚本使用正确的 stack、去尖峰 stack 与输出目录，
  并显式接收 2014 1D 的 `StationXML` 路径。
- **StationXML 与 DAT 选择受检**：转换前要求存在可用台站坐标；缺少任一台站坐标、没有匹配
  stack pair 或转换失败时，脚本返回非零状态。`DAT_GLOB` 统一传递给提取与验证步骤。
- **异常不再伪装为成功**：DisperPicker 对每个台对区分“成功”“无有效拾取”“处理异常”；
  异常时清理局部曲线/NPZ，进程以非零状态退出，不再写入零值占位产物供 `--resume_existing`
  误判为完成。
- **断点续跑只跳过有效产物**：恢复任务会验证 `GDisp/CDisp` 的坐标和周期行数，以及 NPZ 的
  必需数组、维度和失败标记；空文件、截断文件、缺键或带失败原因的 NPZ 都会重算。
- **分片失败可追溯**：重建脚本逐一等待全部分片并汇总退出状态；任一分片失败即终止后续流程。
- **提取完成后强制全量验证**：验证器检查 DAT、曲线、能量 NPZ、分片日志和预期分片数，
  输出 JSON/日志报告；验证不通过则整个重建命令失败。

对应实现位于：
`04DispersionData/Reports/Code/04_dispersion_autopick/`。

---

## 运行环境

- **conda env `noise`**：ObsPy 1.4.2（预处理 / 互相关 / 叠加）
- **conda env `ftan`**：Python 3.6.13, TensorFlow 1.13.1（DisperPicker CNN）

具体依赖见 `04DispersionData/Reports/Environment/`。

---

## 参考文献

- Wang, Y., Allen, R. M., & Liu, K. H. (2017). Ambient noise tomography across Mount St. Helens
  using a dense seismic array. *Journal of Geophysical Research: Solid Earth*, 122(6), 4492–4508.
- Bensen, G. D., et al. (2007). Processing seismic ambient noise data to obtain reliable broad-band
  surface wave dispersion measurements. *Geophysical Journal International*, 169(3), 1239–1260.
- Lin, F.-C., Ritzwoller, M. H., & Snieder, R. (2008). Surface wave tomography of the western United
  States from ambient seismic noise: Rayleigh and Love wave phase velocity maps.
  *Geophysical Journal International*, 173(1), 281–298.

---

## 数据可用性

原始波形数据来自 Wang et al. (2017) 论文所述的 Mount St. Helens 密集台阵（1D 台网 + 永久台站），
可通过 IRIS FDSN Web Services 获取。本仓库不再镜像分发。
