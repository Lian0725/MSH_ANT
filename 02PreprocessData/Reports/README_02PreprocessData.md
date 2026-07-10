# README_02PreprocessData

## 目录用途

本说明文档记录 `MSH_ANT_Final/02PreprocessData` 中 2014 年 1D 台网预处理波形的来源、目录结构、预处理脚本与详细参数，便于后续复现、核查和数据交接。

## 相关路径

- 原始 1D 波形输入目录：`/mnt/data_hdd/MSH_ANT_Final/01RawData/2014/WaveData/1D`
- 2014 年 1D StationXML 目录：`/mnt/data_hdd/MSH_ANT_Final/01RawData/2014/MetaData/1D`
- 预处理输出目录：`/mnt/data_hdd/MSH_ANT_Final/02PreprocessData/2014/WaveData/1D`
- 预处理脚本放置目录：`/mnt/data_hdd/MSH_ANT_Final/02PreprocessData/Reports`
- 预处理脚本文件：`/mnt/data_hdd/MSH_ANT_Final/02PreprocessData/Reports/preprocess_2014_1d_simple_vel.py`

## 当前输出目录概况

基于目录扫描，这批 2014 年 1D 预处理数据具有以下特征：

- 台站目录数：`898`
- MiniSEED 文件数：`11799`
- 文件组织方式：每个台站一个子目录，每天一个 `.mseed` 文件
- 文件命名方式：`YYYYMMDD.mseed`
- 典型日期范围：`2014-07-18` 到 `2014-08-05`

## 预处理目标

本批数据的目标是把 2014 年 1D 台网原始连续波形统一整理为：

- 已合并分段的连续日波形
- 已统一降采样到 `25 Hz`
- 已去仪器响应
- 输出物理量为速度 `VEL`
- 输出为便于后续环境噪声互相关使用的 `MiniSEED`

## 详细预处理参数

### 输入数据

- 输入波形：`1D.*/*.mseed`
- 输入元数据：与台站同名的 `StationXML` 文件，例如 `1D.4037.xml`

### 处理流程

按每个台站每日文件独立处理，顺序如下：

1. 读取原始 `MiniSEED`
2. 读取对应台站 `StationXML`
3. 将 trace 数据转换为 `float64`
4. `merge(method=1, fill_value="interpolate")`
5. 重采样到 `25 Hz`
6. `detrend("demean")`
7. `detrend("linear")`
8. `remove_response(...)` 去仪器响应，输出速度 `VEL`
9. 将输出数据转为 `float32`
10. 以 `MiniSEED` 格式写入目标目录

### 去响应参数

- `output="VEL"`
- `pre_filt=(0.005, 0.01, 10.0, 12.0)`
- `water_level=60`
- `plot=False`

### 采样率与重采样规则

- 目标采样率：`25.0 Hz`
- 若原采样率是 `25 Hz` 的整数倍，则优先使用 `decimate(...)`
- 否则使用 `resample(25.0, no_filter=False)`

### merge 参数

- `method=1`
- `fill_value="interpolate"`

### detrend 参数

- 第一遍：`demean`
- 第二遍：`linear`

### 输出参数

- 输出目录：`/mnt/data_hdd/MSH_ANT_Final/02PreprocessData/2014/WaveData/1D`
- 输出格式：`MSEED`
- 输出编码：`FLOAT32`
- 输出单位：速度 `VEL`

## 脚本说明

脚本 `preprocess_2014_1d_simple_vel.py` 使用的默认目录已经指向 `MSH_ANT_Final` 目录结构：

- `RAW_DIR=/mnt/data_hdd/MSH_ANT_Final/01RawData/2014/WaveData/1D`
- `XML_DIR=/mnt/data_hdd/MSH_ANT_Final/01RawData/2014/MetaData/1D`
- `OUT_DIR=/mnt/data_hdd/MSH_ANT_Final/02PreprocessData/2014/WaveData/1D`

同时也支持通过环境变量覆盖：

- `MSH_1D_RAW_DIR`
- `MSH_1D_XML_DIR`
- `MSH_1D_OUT_DIR`

## 运行方式

### 全量处理

```bash
python /mnt/data_hdd/MSH_ANT_Final/02PreprocessData/Reports/preprocess_2014_1d_simple_vel.py
```

### 仅处理指定台站

先准备一个台站列表文本，例如：

```text
1D.4001
1D.4003
1D.4004
```

然后运行：

```bash
python /mnt/data_hdd/MSH_ANT_Final/02PreprocessData/Reports/preprocess_2014_1d_simple_vel.py station_list.txt
```

## 运行日志与行为说明

- 若目标输出文件已存在且非空，脚本打印 `SKIP`
- 若缺少对应 `StationXML`，脚本打印 `MISS`
- 若某个文件处理失败，脚本打印 `FAIL`
- 每处理 `100` 个文件，或全部结束时，打印一次进度

## 与目录名参数的一致性

这批数据对应的核心参数可以概括为：

- `25 Hz`
- `remove_response -> VEL`
- `pre_filt=(0.005, 0.01, 10, 12)`

若后续需要用目录名进一步显式标注参数，推荐采用类似：

`mseed_25Hz_resp_vel_prefilt_0p005_0p01_10_12`

但当前 `MSH_ANT_Final` 中已经统一归档到：

`/mnt/data_hdd/MSH_ANT_Final/02PreprocessData/2014/WaveData/1D`

## 备注

- 目前未在该目录旁找到独立历史运行日志文件，因此本说明基于现有脚本、目录结构和现存输出数据整理。
- 该 README 的目的，是把这批数据的关键技术参数固定下来，避免后续使用时只知道“已预处理”，却不清楚具体做法。
