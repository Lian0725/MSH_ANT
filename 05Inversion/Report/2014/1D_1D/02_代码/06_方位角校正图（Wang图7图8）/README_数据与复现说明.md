# 2014 1D 台阵：Wang (2017) 图7、图8风格方位角诊断

## 本目录内容

- `reproduce_wang_fig7_fig8.py`：图7与图8的可复跑绘图、方位校正与结果导出脚本。
- 图片目录 `../../03_图片/03_方位角校正图（Wang图7图8）/`：最终的图7和图8 PNG。

本结果使用 2014 年 Mount St. Helens 1D 密集台阵的去尖峰互相关与严格相位走时拾取，参照 Wang et al. (2017, doi:10.1002/2016JB013769) 的图7、图8进行数值诊断。它是本项目数据上的**Wang 风格复现**，并非论文原图的逐像素复制。

## 输入数据与筛选

运行环境为服务器 `work`，根目录为 `/mnt/data_hdd/lgx/MSH_ANT`。

| 输入 | 服务器路径 | 用途 |
|---|---|---|
| 严格相位走时 | `outputs/reports/wang_figure4_disperpicker_phase_periods_3_3p5_4_left_quality_right_strict_20260701/measurements_period_corrected.csv` | 图7上排、图8与方位校正。|
| 台站坐标 | `outputs/reports/wang_fig5_fig6_disperpicker_paper_standard_3_3p5_4s_20260701/data/stations.csv` | 台间方位角与波束空间投影。|
| 去尖峰 PWS CCF | `stack/2014/1D_WANG_PWS_150s_20260620/STACK_SPIKE_REMOVED_DIAGFIT_20260628` | 图7下排的非对称 CCF 波束。|

相位走时来自 CDisp。其严格筛选包括 SNR、群速度和参考波长距离条件；本次图7/图8实际保留的走时数为 3 s: 25,474，3.5 s: 22,947，4 s: 18,759。图7下排从 402,423 条可用去尖峰 CCF 中，以固定随机种子 `20160708` 均匀抽取 20,000 条，保证可重复。

## 计算定义

### 图7上排：走时残差

对每个周期，以过原点的最小二乘慢度拟合常速参考模型：

\[
s_\mathrm{ref}=\frac{\sum_i D_i t_i}{\sum_i D_i^2},\qquad
V_\mathrm{ref}=1/s_\mathrm{ref},\qquad
r_i=t_i-D_i/V_\mathrm{ref}.
\]

上排显示 3、3.5、4 s 的残差方位—距离极坐标分布。半径为 0–25 km；残差色标固定为 -0.3 至 +0.3 s；灰色单元表示该距离—方位格内没有通过严格筛选的走时，并非人工掩膜。

### 图7下排：CCF 波束

下排显示 2、3、4 s。对每条去尖峰、非对称 CCF，在目标频率的傅里叶系数上施加平面波试探移时，并对所有 CCF 的复相位相干性求平均。方位为来波源方向，慢度半径为 0–0.6 s/km，包含零慢度，因此图心为实际计算结果。

图中振幅为显示用归一化振幅：每个周期的原始相干度除以该周期峰值后乘以 2。色标为 0–2，并用短蓝色低值段和扩展红—洋红高值段近似论文图7的显示色阶。原始相干度峰值、峰值方位与慢度写入输出 `metadata.json`。

### 图8：偶次方位角校正

每 20° 方位箱计算残差均值与均值标准误，随后按 Wang et al. 的形式拟合：

\[
f(\theta)=a+b\cos(2\theta)+c\sin(2\theta)+d\cos(4\theta)+e\sin(4\theta).
\]

仅保留偶次项，以满足对称 CCF 走时残差的 180° 对称性。脚本同时输出每条走时的 `azimuth_correction_s`、`corrected_travel_time_s` 与校正后残差，可作为后续反演输入。

## 复跑命令

服务器缺省没有 Times New Roman；本次使用的字体副本位于隔离实验目录：

```bash
/mnt/data_hdd/lgx/MSH_ANT/.venvs/2014_wang_pws/bin/python \
  reproduce_wang_fig7_fig8.py \
  --measurements-csv /mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_figure4_disperpicker_phase_periods_3_3p5_4_left_quality_right_strict_20260701/measurements_period_corrected.csv \
  --stations-csv /mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_fig5_fig6_disperpicker_paper_standard_3_3p5_4s_20260701/data/stations.csv \
  --stack-root /mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620/STACK_SPIKE_REMOVED_DIAGFIT_20260628 \
  --output-dir /mnt/data_hdd/lgx/MSH_ANT/experiments/wang_fig78_20260712/output/final \
  --beam-sample-size 20000 \
  --font-file '/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_fig78_20260712/fonts/Times New Roman.ttf'
```

## 方法边界

论文使用 phase-FTAN 拾取及全 CCF 的时域移时叠加最大振幅；本项目使用 CDisp 相位走时和窄带频域相干波束。两者检验的是相同的方位依赖与试探移时思想，但波束幅值定义不同。因此应以该图判断本项目数据中方位性偏差及其校正必要性，而不应将其解释为对原论文图形的逐像素重现。
