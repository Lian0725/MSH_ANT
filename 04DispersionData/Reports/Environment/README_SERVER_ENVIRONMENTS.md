# README_SERVER_ENVIRONMENTS

> 更新日期：2026-07-09  
> 适用范围：`/mnt/data_hdd/MSH_ANT_Final/04DispersionData/Reports/Environment`

## 1. 这组文件是做什么的

这个目录保存的是从 `work` 服务器直接导出的环境快照，目的是尽量完整保留“当时跑 2014 频散提取时的真实软件环境”。

这里分两类文件：

1. 摘要文件  
   便于快速查看 Python 版本、核心库版本、GPU/CUDA 信息。

2. 原始导出文件  
   便于之后严格对照、重建或排查环境差异。

## 2. 服务器机器环境摘要

从导出文件确认：

| 项目 | 值 |
|---|---|
| OS | `Ubuntu 24.04.4 LTS` |
| Kernel | `Linux 6.8.0-124-generic` |
| GPU | `NVIDIA GeForce RTX 5090 D` |
| GPU 显存 | `32607 MiB` |
| NVIDIA Driver | `595.71.05` |
| `nvidia-smi` 报告 CUDA | `13.2` |
| `nvcc --version` | `CUDA 12.0.140` |

注意这里存在三层 CUDA 口径：

1. 驱动层：`nvidia-smi` 显示 `CUDA Version: 13.2`
2. 本机 `nvcc`：`12.0.140`
3. TensorFlow 2.21 构建信息：`cuda_version = 12.5.1`

这说明服务器上运行 GPU 版频散提取时，并不是单纯依赖系统 `nvcc` 版本，而是由 Python 环境内打包的 CUDA 运行时共同决定。

## 3. 三套 Python 环境

### 3.1 `wang_pws`

用途：

- 2014 1D Wang/Lin 互相关
- 小时窗互相关与 PWS 叠加
- 生成 `03CC_StackData` 对应的 `STACK`

解释器路径：

```text
/mnt/data_hdd/lgx/MSH_ANT/.venvs/2014_wang_pws/bin/python
```

核心版本：

| 包 | 版本 |
|---|---:|
| Python | `3.12.3` |
| NumPy | `2.4.6` |
| SciPy | `1.18.0` |
| Matplotlib | `3.11.0` |
| h5py | `3.16.0` |
| ObsPy | `1.5.0` |

对应文件：

| 文件 | 含义 |
|---|---|
| `wang_pws_python_version.txt` | `python -V` |
| `wang_pws_pip_version.txt` | `pip --version` |
| `wang_pws_summary.json` | 核心包版本摘要 |
| `wang_pws_pip_freeze.txt` | 全量 `pip freeze` 输出 |

### 3.2 `ftan`

用途：

- CPU 版频散提取
- 历史 DisperPicker / TensorFlow 1 兼容运行
- DAT 转换与部分老脚本

解释器路径：

```text
/mnt/data_hdd/lgx/MSH_ANT/envs/ftan/bin/python
```

核心版本：

| 包 | 版本 |
|---|---:|
| Python | `3.6.13` |
| NumPy | `1.19.0` |
| SciPy | `1.5.2` |
| Matplotlib | `3.3.4` |
| h5py | `3.1.0` |
| TensorFlow | `1.13.1` |
| geopy | `2.2.0` |

对应文件：

| 文件 | 含义 |
|---|---|
| `ftan_python_version.txt` | `python -V` |
| `ftan_pip_version.txt` | `pip --version` |
| `ftan_summary.json` | 核心包版本摘要 |
| `ftan_pip_freeze.txt` | 全量 `pip freeze` 输出 |

### 3.3 `disp_gpu`

用途：

- GPU 版频散能量图构建
- 全量 `GDisp/CDisp` 自动提取
- `NPZ-only` 回填

解释器路径：

```text
/mnt/data_hdd/lgx/MSH_ANT/envs/disp_gpu/bin/python
```

核心版本：

| 包 | 版本 |
|---|---:|
| Python | `3.12.3` |
| NumPy | `2.5.0` |
| SciPy | `1.18.0` |
| Matplotlib | `3.11.0` |
| h5py | `3.14.0` |
| TensorFlow | `2.21.0` |
| CuPy | `14.1.1` |
| geopy | `2.4.1` |

从 TensorFlow 构建信息确认：

| 项目 | 值 |
|---|---|
| TensorFlow CUDA build | `12.5.1` |
| cuDNN | `9` |
| CUDA build target | `sm_60, sm_70, sm_80, sm_89, compute_90` |

从 CuPy 配置确认：

| 项目 | 值 |
|---|---|
| CUDA Build Version | `12090` |
| CUDA Driver Version | `13020` |
| CUDA Runtime Version | `12090` |
| NCCL Build Version | `22501` |
| NCCL Runtime Version | `23007` |
| Device | `NVIDIA GeForce RTX 5090 D` |
| Compute Capability | `120` |

对应文件：

| 文件 | 含义 |
|---|---|
| `disp_gpu_python_version.txt` | `python -V` |
| `disp_gpu_pip_version.txt` | `pip --version` |
| `disp_gpu_summary.json` | 核心包版本摘要 |
| `disp_gpu_pip_freeze.txt` | 全量 `pip freeze` 输出 |
| `disp_gpu_tensorflow_build_info.json` | TensorFlow 构建信息 |
| `disp_gpu_cupy_show_config.txt` | CuPy / CUDA 详细配置 |

## 4. 系统级导出文件

| 文件 | 含义 |
|---|---|
| `system_uname.txt` | `uname -a` 输出 |
| `system_lsb_release.txt` | `lsb_release -a` 输出 |
| `system_os_release.txt` | `/etc/os-release` |
| `nvidia_smi.txt` | 完整 `nvidia-smi` 输出 |
| `nvidia_smi_query.csv` | GPU 名称、驱动版本、显存摘要 |
| `nvcc_version.txt` | `nvcc --version` 输出 |

## 5. 如何使用这些文件

### 5.1 如果目标是“理解原环境”

按下面顺序看最快：

1. `README_SERVER_ENVIRONMENTS.md`
2. `*_summary.json`
3. `disp_gpu_tensorflow_build_info.json`
4. `disp_gpu_cupy_show_config.txt`
5. `*_pip_freeze.txt`

### 5.2 如果目标是“重建接近服务器的环境”

推荐流程：

1. 先按 `*_summary.json` 建出接近的 Python 主版本。
2. 再参考 `*_pip_freeze.txt` 装同版本包。
3. GPU 环境要额外对齐：
   - NVIDIA 驱动
   - TensorFlow 2.21 的 CUDA 口径
   - CuPy 14.1.1 的 CUDA 口径
4. 最后用 `Reports/Code` 里的脚本跑一个小样本台对验证。

## 6. 关于“完全一致复现”的说明

这些环境导出文件已经足够让后续读者非常接近服务器环境，但仍需注意：

1. GPU 数值运算存在浮点级差异。
2. TensorFlow / CuPy / 驱动三者的兼容性会影响运行速度与极小数值差别。
3. 因此更现实的目标是：

```text
结果等价复现
```

而不是保证所有 `.txt` / `.npz` 文件逐字节完全相同。
