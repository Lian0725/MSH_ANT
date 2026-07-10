"""
run_dispersion_mi09.py
=======================
使用 EGFAnalysisPy 对 MI09 台站的所有 1D 台对进行频散曲线提取。

运行环境: work 服务器, conda noise 环境
运行方式:
    /home/lgxwork/miniconda3/envs/noise/bin/python run_dispersion_mi09.py \
        --dat_dir /mnt/Data/lgxwork/MSH_ANT/outputs/dispersion/by_station/dispersion_MI09/dat/MI09 \
        --out_dir /mnt/Data/lgxwork/MSH_ANT/outputs/dispersion/by_station/dispersion_MI09/curves \
        [--test_pair XD.MI09__1D.4001]   # 仅测试单台对

输出目录结构（不修改原始数据）:
    outputs/dispersion/by_station/dispersion_MI09/
        dat/MI09/            ← 由 convert_stack_to_dat.py 生成
        curves/
            GDisp.<pair>.dat ← 群速度频散曲线
            CDisp.<pair>.dat ← 相速度频散曲线
        qc_plot/             ← 频散能量图+拾取结果可视化
"""

import sys
import os
import argparse
import glob
import logging
import traceback
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────
#  将 EGFAnalysisPy 目录加入 Python 路径
#  脚本位于: scripts/04_dispersion/run_dispersion_mi09.py
#  EGFAnalysisPy 位于: scripts/04_dispersion/EGFAnalysisPy/
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EGF_DIR = os.path.join(SCRIPT_DIR, 'EGFAnalysisPy')
if EGF_DIR not in sys.path:
    sys.path.insert(0, EGF_DIR)

# DisperPicker 使用 os.getcwd() 定位模型文件，必须切换到 EGFAnalysisPy/ 目录
os.chdir(EGF_DIR)

# DisperPicker 子目录也需要加入路径
DISPER_DIR = os.path.join(EGF_DIR, 'DisperPicker')
if DISPER_DIR not in sys.path:
    sys.path.insert(0, DISPER_DIR)


def _verify_disperpicker_saver():
    """启动前预检 DisperPicker CNN 权重文件是否齐全。

    背景: TensorFlow 的 saver/checkpoint 是无扩展名文本文件, 若归档/克隆流程
    按扩展名白名单迁移代码, 会漏掉此文件, 导致 Pick.__init__ 抛
    FileNotFoundError 且信息隐晦。此函数提供清晰错误 + 修复指令。
    """
    saver_dir = os.path.join(DISPER_DIR, "saver")
    ckpt = os.path.join(saver_dir, "checkpoint")
    weights = [
        "-10000.data-00000-of-00001",
        "-10000.index",
        "-10000.meta",
    ]
    missing = []
    if not os.path.isfile(ckpt):
        missing.append(ckpt)
    for w in weights:
        wp = os.path.join(saver_dir, w)
        if not os.path.isfile(wp):
            missing.append(wp)
    if not missing:
        return
    msg = (
        "\n[FATAL] DisperPicker CNN 权重文件缺失, 无法加载模型:\n"
        + "\n".join("  - " + m for m in missing)
        + "\n\n最常见成因: 归档/克隆流程用了扩展名白名单, 漏掉了无扩展名的\n"
          "  checkpoint 文件。修复方式(整目录复制 saver/):\n"
          "  cp -r <MSH_ANT_Final>/04DispersionData/Reports/Code/04_dispersion_autopick/\n"
          "        EGFAnalysisPy/DisperPicker/saver/. \\\n"
          "     " + saver_dir + "/\n"
    )
    raise SystemExit(msg)

import numpy as np
import EGFAnalysisTimeFreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
#  全局参数（已与用户确认）
# ─────────────────────────────────────────────────────────────────
GreenFcnObjectsType = EGFAnalysisTimeFreq.gfcn_analysis.GreenFcnObjectsType

CONFIG = dict(
    isEGF=False,          # 互相关函数（非EGF），内部会做 Hilbert 变换
    StartT=0.2,           # 最短周期 0.2s（对应 5Hz）
    EndT=5.0,             # 最长周期 5.0s（对应 0.2Hz）
    DeltaT=0.1,           # 周期步长 0.1s
    StartV=2.0,           # 最小相速度 2.0 km/s（Wang Figure 4 自动拾取实验口径）
    EndV=4.0,             # 最大相速度 4.0 km/s
    DeltaV=0.005,         # 速度分辨率（固定默认値）
    MinDist=10.0,         # 旧的短台距诊断阈值（km）
                          # 2017 permanent-YI 现按 2014 XD/1D 口径执行：
                          # 不再用这个阈值跳过台对，只保留日志提示
    GreenFcnObjects=GreenFcnObjectsType.A_add_B,  # 正负滞后平均
    WinAlpha=0.1,         # 窗函数余弦比例
    NoiseTime=5.0,        # 噪声段时长（s），兼顾短记录和近台对可用噪声窗
    MinSNR=5.0,           # 最小信噪比阈値
    WinPeriodNum=5,       # 相速度时变滤波窗口周期数
    WinMinTime=5,         # 最小窗口时长（s）
    FilterKaiserPara=6,   # Kaiser 窗形状因子
    MaxFilterLengthLog=14, # FFT 点数上限（2^14）
)

GROUP_QC_CMAP = "cividis"
PHASE_QC_CMAP = "coolwarm"


@dataclass(frozen=True)
class PhaseRefIndex:
    index: int
    period_s: float
    source: str


def _period_count(start_t=None, end_t=None, delta_t=None):
    start_t = CONFIG['StartT'] if start_t is None else start_t
    end_t = CONFIG['EndT'] if end_t is None else end_t
    delta_t = CONFIG['DeltaT'] if delta_t is None else delta_t
    return round((end_t - start_t) / delta_t) + 1


def _clamp_period_index(index, n_period):
    return max(0, min(n_period - 1, int(index)))


def _period_to_index(period_s, start_t=None, end_t=None, delta_t=None):
    start_t = CONFIG['StartT'] if start_t is None else start_t
    end_t = CONFIG['EndT'] if end_t is None else end_t
    delta_t = CONFIG['DeltaT'] if delta_t is None else delta_t
    n_period = _period_count(start_t, end_t, delta_t)
    index = round((float(period_s) - start_t) / delta_t)
    return _clamp_period_index(index, n_period)


def phase_ref_index_for_distance(distance_km, phase_ref_t=None, phase_ref_t_max=None,
                                 start_t=None, end_t=None, delta_t=None):
    """Resolve the DisperPicker phase reference column for one station pair."""
    start_t = CONFIG['StartT'] if start_t is None else start_t
    end_t = CONFIG['EndT'] if end_t is None else end_t
    delta_t = CONFIG['DeltaT'] if delta_t is None else delta_t
    n_period = _period_count(start_t, end_t, delta_t)

    if phase_ref_t is not None:
        index = _period_to_index(phase_ref_t, start_t, end_t, delta_t)
        source = "manual"
    else:
        auto_index = int(
            0.6 * min(
                n_period - 1,
                round((float(distance_km) / 1.5 / 3.2 - start_t) / delta_t),
            )
        )
        index = _clamp_period_index(auto_index, n_period)
        source = "auto"
        if phase_ref_t_max is not None:
            cap_index = _period_to_index(phase_ref_t_max, start_t, end_t, delta_t)
            if index > cap_index:
                index = cap_index
                source = "auto_capped"

    period_s = start_t + index * delta_t
    return PhaseRefIndex(index=index, period_s=period_s, source=source)


def configure_picker_phase_ref(pick, distance_km, phase_ref_t=None, phase_ref_t_max=None):
    ref = phase_ref_index_for_distance(
        distance_km=distance_km,
        phase_ref_t=phase_ref_t,
        phase_ref_t_max=phase_ref_t_max,
    )
    pick.ref_T = ref.index
    return ref


def pair_name_from_dat_path(dat_path):
    return os.path.splitext(os.path.basename(dat_path))[0]


def pair_outputs_exist(dat_path, out_dir, energy_dir=None):
    pair_name = pair_name_from_dat_path(dat_path)
    required = [
        os.path.join(out_dir, f"GDisp.{pair_name}.txt"),
        os.path.join(out_dir, f"CDisp.{pair_name}.txt"),
    ]
    if energy_dir:
        required.append(os.path.join(energy_dir, pair_name + '.npz'))
    return all(os.path.exists(path) for path in required)


def filter_dat_files(dat_files, out_dir, energy_dir=None,
                     num_shards=1, shard_index=0,
                     resume_existing=False, max_pairs=None):
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    selected = [
        path
        for index, path in enumerate(dat_files)
        if index % num_shards == shard_index
    ]

    skipped_existing = 0
    if resume_existing:
        remaining = []
        for path in selected:
            if pair_outputs_exist(path, out_dir, energy_dir=energy_dir):
                skipped_existing += 1
            else:
                remaining.append(path)
        selected = remaining

    if max_pairs is not None and max_pairs > 0:
        selected = selected[:max_pairs]

    return selected, skipped_existing


def _plot_dispersion_qc(group_image, phase_image, group_v, phase_v,
                        T, actual_start_v, end_v, pair_name, dist, out_path):
    """
    自定义 QC 绘图，修正坐标轴问题：
    - 速度轴使用实际有效范围（actual_start_v ~ end_v），去掉 pad 的零値区域
    - 图幅足够大避免标签重叠
    - 不依赖 Config 类
    """
    # 速度点数（EGFAnalysisTimeFreq 经 [::-1] 后：行0 = 最低速 actual_start_v，升序）
    model_nv = group_image.shape[0]  # 701
    actual_nv = round((end_v - actual_start_v) / CONFIG['DeltaV']) + 1

    # 裁切掉 pad 的零値行（顶部），只保留有效速度范围
    g_img = group_image[model_nv - actual_nv:, :]  # shape (actual_nv, num_T)
    p_img = phase_image[model_nv - actual_nv:, :]  # shape (actual_nv, num_T)

    # 坐标轴
    x = T  # 周期轴
    # EGFAnalysisTimeFreq 输出经过 [::-1] 翻转：行0 = 最低速度(actual_start_v)
    # 裁切后 g_img[0] = actual_start_v, g_img[-1] = end_v
    # 所以 y 轴必须升序排列，使 pcolormesh 正确映射
    y = np.linspace(actual_start_v, end_v, actual_nv)  # 速度轴（升序）

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{pair_name}  dist={dist:.1f} km', fontsize=14)

    # 左图：群速度
    ax = axes[0]
    z_max = np.abs(g_img).max() or 1.0
    pcm = ax.pcolormesh(
        x,
        y,
        g_img,
        shading='auto',
        cmap=GROUP_QC_CMAP,
        vmin=0,
        vmax=z_max,
    )
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    # 绘制群速度曲线
    nz = np.nonzero(group_v)[0]
    if len(nz):
        ax.plot(T[nz], group_v[nz], '--k', linewidth=2, label='Group (CNN)')
        ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(T[0], T[-1])
    ax.set_ylim(actual_start_v, end_v)
    ax.set_xlabel('Period (s)', fontsize=13)
    ax.set_ylabel('Group Velocity (km/s)', fontsize=13)
    ax.set_title('Group velocity', fontsize=13)
    ax.tick_params(labelsize=11)

    # 右图：相速度
    ax = axes[1]
    z_abs = np.abs(p_img).max() or 1.0
    pcm = ax.pcolormesh(
        x,
        y,
        p_img,
        shading='auto',
        cmap=PHASE_QC_CMAP,
        vmin=-z_abs,
        vmax=z_abs,
    )
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    # 绘制相速度曲线
    nz = np.nonzero(phase_v)[0]
    if len(nz):
        ax.plot(T[nz], phase_v[nz], '--k', linewidth=2, label='Phase (CNN)')
    # 绘制群速度曲线参考线
    nz_g = np.nonzero(group_v)[0]
    if len(nz_g):
        ax.plot(T[nz_g], group_v[nz_g], '--w', linewidth=1.5, label='Group ref')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(T[0], T[-1])
    ax.set_ylim(actual_start_v, end_v)
    ax.set_xlabel('Period (s)', fontsize=13)
    ax.set_ylabel('Phase Velocity (km/s)', fontsize=13)
    ax.set_title('Phase velocity', fontsize=13)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def process_one_pair(dat_path: str, out_dir: str, qc_plot_dir: str,
                     pick=None, qc_mod=None,
                     phase_ref_t=None, phase_ref_t_max=None,
                     energy_dir=None) -> bool:
    """
    对单个台对 DAT 文件进行频散提取，使用 DisperPicker CNN 拾取。
    pick: 预初始化的 Pick 对象（避免每次重新加载模型）
    返回 True 表示成功提取到频散曲线，False 表示失败或信噪比不足。
    """
    pair_name = pair_name_from_dat_path(dat_path)

    try:
        # 0. 记录短台距信息，但不再跳过台对。
        #    2017 permanent-YI 这里改为和 2014 XD/1D 一致：
        #    即使后面拾取不到有效频散点，也仍然输出 qc_plot。
        with open(dat_path, 'r') as _f:
            lon_a, lat_a = map(float, _f.readline().split())
            lon_b, lat_b = map(float, _f.readline().split())
        from geopy.distance import geodesic
        _dist_km = geodesic((lat_a, lon_a - 360), (lat_b, lon_b - 360)).km
        if _dist_km < CONFIG['MinDist']:
            logger.info(
                f"{pair_name}: 短台距提示（dist={_dist_km:.1f}km < "
                f"legacy MinDist={CONFIG['MinDist']}km），继续生成频散能量图"
            )

        # 1. 初始化 EGFAnalysisTimeFreq（读取 DAT 文件，建立分析对象）
        gfcn = EGFAnalysisTimeFreq.gfcn_analysis(
            DataFileName=dat_path,
            isEGF=CONFIG['isEGF'],
            StartT=CONFIG['StartT'],
            EndT=CONFIG['EndT'],
            DeltaT=CONFIG['DeltaT'],
            StartV=CONFIG['StartV'],
            EndV=CONFIG['EndV'],
            DeltaV=CONFIG['DeltaV'],
            GreenFcnObjects=CONFIG['GreenFcnObjects'],
            WinAlpha=CONFIG['WinAlpha'],
            NoiseTime=CONFIG['NoiseTime'],
            MinSNR=CONFIG['MinSNR'],
        )

        dist = gfcn.StaDist
        sta_info = [dist,
                    gfcn.Longitude_A, gfcn.Latitude_A,
                    gfcn.Longitude_B, gfcn.Latitude_B]

        # 2. 计算群速度能量图（步骤1：无时变滤波）
        group_image = gfcn.GroupVelocityImgCalculate()

        # 3. 计算相速度能量图（步骤1：无时变滤波）
        phase_image = gfcn.PhaseVelocityImgCalculate(
            TimeVariableFilter=gfcn.TimeVariableFilterType.no,
            WinPeriodNum=CONFIG['WinPeriodNum'],
            WinMinTime=CONFIG['WinMinTime'],
            FilterKaiserPara=CONFIG['FilterKaiserPara'],
            MaxFilterLengthLog=CONFIG['MaxFilterLengthLog'],
        )

        snr = gfcn.SNR_T
        num_T = group_image.shape[1]  # 周期点数 = 49
        num_V = group_image.shape[0]  # 实际速度点数（从 gfcn.StartV 到 EndV）

        # DisperPicker CNN 模型期望 701 速度点（对应原始 StartV=0.5, EndV=4.0, DeltaV=0.005）。
        # 当前实验设置 StartV=2.0，实际有效速度行数通常为 401；低速端 padding 后仍保持
        # 701 x 49 的模型输入矩阵，便于后续把全像素数据按任意 colormap 重画。
        MODEL_NV = round((CONFIG['EndV'] - 0.5) / CONFIG['DeltaV']) + 1  # = 701 （模型固定）
        MODEL_NT = num_T  # 周期轴不受 StartV 重置影响

        # 4. DisperPicker CNN 拾取（步骤1）
        if pick is None:
            from pick import Pick
            pick = Pick()
        if qc_mod is None:
            import qc as qc_mod

        # 图像 pad 到模型期望尺寸 (MODEL_NV × MODEL_NT)
        batch_group_image = np.zeros((1, MODEL_NV, MODEL_NT))
        batch_phase_image = np.zeros((1, MODEL_NV, MODEL_NT))
        # VPoint 是从 EndV 到 StartV 降序排列，低速在数组末尾（图像底部）
        # 如果 num_V < MODEL_NV，则图像在顶部（高速端）多出的行补零
        batch_group_image[0, MODEL_NV - num_V:, :num_T] = group_image
        batch_phase_image[0, MODEL_NV - num_V:, :num_T] = phase_image

        pick.mean_confidence_G = 0  # 第一轮放宽以获得群速度
        phase_ref = configure_picker_phase_ref(
            pick,
            distance_km=dist,
            phase_ref_t=phase_ref_t,
            phase_ref_t_max=phase_ref_t_max,
        )
        logger.debug(
            "%s: phase ref_T=%d (T=%.2fs, %s) before first pick",
            pair_name, phase_ref.index, phase_ref.period_s, phase_ref.source,
        )

        group_velocity, phase_velocity, _, _ = pick.pick(
            group_image=batch_group_image,
            phase_image=batch_phase_image,
            sta_info=[sta_info],
            snr=[snr],
            file_list=[pair_name],
            ct=0.01,
            save_result=False,
        )

        # 5. 时变滤波：用群速度加窗，重新计算相速度能量图（步骤2）
        if np.count_nonzero(group_velocity[0]) > 0:
            gfcn.GroupDisperCurve = group_velocity[0]
            phase_image2 = gfcn.PhaseVelocityImgCalculate(
                TimeVariableFilter=gfcn.TimeVariableFilterType.obs,
                WinPeriodNum=CONFIG['WinPeriodNum'],
                WinMinTime=CONFIG['WinMinTime'],
                FilterKaiserPara=CONFIG['FilterKaiserPara'],
                MaxFilterLengthLog=CONFIG['MaxFilterLengthLog'],
            )
            batch_phase_image2 = np.zeros((1, MODEL_NV, MODEL_NT))
            batch_phase_image2[0, MODEL_NV - phase_image2.shape[0]:, :phase_image2.shape[1]] = phase_image2
        else:
            batch_phase_image2 = batch_phase_image
            logger.warning(f"{pair_name}: 群速度拾取失败，跳过时变滤波")

        # 6. DisperPicker 第二轮拾取（最终结果）
        pick.mean_confidence_G = 0.3
        phase_ref = configure_picker_phase_ref(
            pick,
            distance_km=dist,
            phase_ref_t=phase_ref_t,
            phase_ref_t_max=phase_ref_t_max,
        )
        logger.debug(
            "%s: phase ref_T=%d (T=%.2fs, %s) before final pick",
            pair_name, phase_ref.index, phase_ref.period_s, phase_ref.source,
        )

        group_velocity, phase_velocity, prob_G, prob_C = pick.pick(
            group_image=batch_group_image,
            phase_image=batch_phase_image2,
            sta_info=[sta_info],
            snr=[snr],
            file_list=[pair_name],
            ct=2,
            save_result=False,  # 手动保存到指定路径
        )

        # 7. 质量控制
        T = np.linspace(CONFIG['StartT'], CONFIG['EndT'],
                        round((CONFIG['EndT'] - CONFIG['StartT']) / CONFIG['DeltaT']) + 1)

        if energy_dir:
            os.makedirs(energy_dir, exist_ok=True)
            velocity_axis = np.linspace(0.5, CONFIG['EndV'], MODEL_NV)
            np.savez_compressed(
                os.path.join(energy_dir, pair_name + '.npz'),
                group_image=batch_group_image[0],
                phase_image=batch_phase_image2[0],
                periods=T,
                velocities=velocity_axis,
                velocity_axis_km_s=velocity_axis,
                actual_velocity_axis_km_s=np.linspace(
                    float(gfcn.StartV),
                    CONFIG['EndV'],
                    round((CONFIG['EndV'] - float(gfcn.StartV)) / CONFIG['DeltaV']) + 1,
                ),
                snr=np.asarray(snr),
                actual_start_v=float(gfcn.StartV),
                configured_start_v=float(CONFIG['StartV']),
                end_v=float(CONFIG['EndV']),
                delta_v=float(CONFIG['DeltaV']),
                distance_km=float(dist),
                noise_time=float(CONFIG['NoiseTime']),
            )

        snr_for_qc = snr if np.count_nonzero(snr) > 0 else np.ones(len(T))

        group_v = qc_mod.qc(group_velocity[0], snr_for_qc,
                             v_range=[CONFIG['StartV'], CONFIG['EndV']],
                             diff_range=[-0.07, 0.08],
                             upward=0.4, each_stage_upward=0.3,
                             min_len=round(len(T) / 5), skip=False)

        phase_v = qc_mod.qc(phase_velocity[0], snr_for_qc,
                             v_range=[CONFIG['StartV'], CONFIG['EndV']],
                             diff_range=[-0.1, 0.1],
                             upward=0, each_stage_upward=0.2,
                             min_len=round(len(T) / 5), skip=True)

        # 8. 保存结果到指定输出目录
        has_result = False

        os.makedirs(out_dir, exist_ok=True)

        # 群速度文件
        g_path = os.path.join(out_dir, f"GDisp.{pair_name}.txt")
        with open(g_path, 'w') as f:
            f.write(f"{sta_info[1]:.8f}    {sta_info[2]:.8f}\n")
            f.write(f"{sta_info[3]:.8f}    {sta_info[4]:.8f}\n")
            for t_val, g_val, s_val, pg in zip(T, group_v, snr_for_qc, prob_G):
                f.write(f"{t_val:.2f}  {g_val:.3f}  {s_val:.3f}  {pg:.3f}\n")

        # 相速度文件
        c_path = os.path.join(out_dir, f"CDisp.{pair_name}.txt")
        with open(c_path, 'w') as f:
            f.write(f"{sta_info[1]:.8f}    {sta_info[2]:.8f}\n")
            f.write(f"{sta_info[3]:.8f}    {sta_info[4]:.8f}\n")
            for t_val, c_val, s_val, pc in zip(T, phase_v, snr_for_qc, prob_C):
                f.write(f"{t_val:.2f}  {c_val:.3f}  {s_val:.3f}  {pc:.3f}\n")

        # QC 可视化图（自定义绘图，修正坐标轴）
        if qc_plot_dir:
            os.makedirs(qc_plot_dir, exist_ok=True)
            try:
                _plot_dispersion_qc(
                    group_image=batch_group_image[0],
                    phase_image=batch_phase_image2[0],
                    group_v=group_v,
                    phase_v=phase_v,
                    T=T,
                    actual_start_v=float(gfcn.StartV),
                    end_v=CONFIG['EndV'],
                    pair_name=pair_name,
                    dist=dist,
                    out_path=os.path.join(qc_plot_dir, pair_name + '.jpg'),
                )
            except Exception as e:
                logger.warning(f"{pair_name}: 绘图失败（不影响结果）: {e}")

        # 判断是否有有效频散点
        if np.count_nonzero(group_v) > 0 or np.count_nonzero(phase_v) > 0:
            has_result = True
            logger.info(f"{pair_name}: dist={dist:.1f}km, "
                        f"G_pts={np.count_nonzero(group_v)}, "
                        f"C_pts={np.count_nonzero(phase_v)}")
        else:
            logger.warning(f"{pair_name}: 未提取到有效频散点")

        return has_result

    except Exception as e:
        logger.error(f"{pair_name} 处理失败: {e}")
        logger.debug(traceback.format_exc())
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="EGFAnalysisPy 频散提取 - MI09 台站")
    parser.add_argument('--dat_dir', required=True,
                        help="DAT 文件目录，如 .../dispersion_MI09/dat/MI09")
    parser.add_argument('--out_dir', required=True,
                        help="频散曲线输出目录，如 .../dispersion_MI09/curves")
    parser.add_argument('--qc_plot_dir', default=None,
                        help="QC 图输出目录，默认为 out_dir/../qc_plot")
    parser.add_argument('--skip_qc_plot', action='store_true',
                        help="跳过逐台对 QC 图片，仅保存频散曲线。")
    parser.add_argument('--dat_glob', default="XD.*.dat",
                        help="DAT 文件匹配模式，默认 XD.*.dat")
    parser.add_argument('--test_pair', default=None,
                        help="仅处理指定台对（如 XD.MI09__1D.4001），用于测试")
    parser.add_argument('--mean_confidence_c', type=float, default=None,
                        help="覆盖 DisperPicker 相速度候选平均概率阈值，例如 0.35")
    parser.add_argument('--phase_ref_t', type=float, default=None,
                        help="固定相速度拾取参考周期（秒），用于从指定周期列开始追踪")
    parser.add_argument('--phase_ref_t_max', type=float, default=None,
                        help="自动参考周期的上限（秒），例如 2.0 可锚定短周期稳定段")
    parser.add_argument('--energy_dir', '--full_pixel_data_dir', dest='energy_dir', default=None,
                        help="可选：保存群/相速度全像素频散矩阵 NPZ 的目录")
    parser.add_argument('--num_shards', '--num-shards', dest='num_shards', type=int, default=1,
                        help="并行分片总数；每个进程处理 index %% num_shards == shard_index 的 DAT")
    parser.add_argument('--shard_index', '--shard-index', dest='shard_index', type=int, default=0,
                        help="当前进程分片编号，范围 0 <= shard_index < num_shards")
    parser.add_argument('--max_pairs', '--max-pairs', dest='max_pairs', type=int, default=None,
                        help="可选：过滤和分片后最多处理多少个 DAT；0 或负数表示不限制")
    parser.add_argument('--resume_existing', '--resume-existing', dest='resume_existing', action='store_true',
                        help="跳过已经存在 GDisp/CDisp 输出的台对；若保存全像素数据则同时要求 NPZ 已存在")
    args = parser.parse_args(argv)

    if args.num_shards < 1:
        logger.error("--num_shards must be >= 1")
        return 2
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        logger.error("--shard_index must satisfy 0 <= shard_index < --num_shards")
        return 2

    if args.skip_qc_plot:
        args.qc_plot_dir = None
    elif args.qc_plot_dir is None:
        args.qc_plot_dir = os.path.join(os.path.dirname(args.out_dir), 'qc_plot')

    logger.info("=" * 60)
    logger.info("EGFAnalysisPy 频散提取")
    logger.info(f"DAT 目录:  {args.dat_dir}")
    logger.info(f"输出目录:  {args.out_dir}")
    logger.info(f"QC 图目录: {args.qc_plot_dir if args.qc_plot_dir else 'disabled'}")
    logger.info(f"全像素数据: {args.energy_dir if args.energy_dir else 'disabled'}")
    logger.info(f"参数配置: StartT={CONFIG['StartT']}s, EndT={CONFIG['EndT']}s, "
                f"DeltaT={CONFIG['DeltaT']}s")
    logger.info(f"         StartV={CONFIG['StartV']}, EndV={CONFIG['EndV']}, "
                f"DeltaV={CONFIG['DeltaV']} km/s")
    logger.info(
        f"         legacy MinDist={CONFIG['MinDist']}km "
        f"(只提示不跳过；按 2014 XD/1D 口径保留全部台对 qc_plot)"
    )
    logger.info(f"         MinSNR={CONFIG['MinSNR']}, NoiseTime={CONFIG['NoiseTime']}s")
    logger.info(f"         GreenFcnObjects=A_add_B, isEGF=False")
    logger.info(
        f"         shard={args.shard_index}/{args.num_shards}, "
        f"resume_existing={args.resume_existing}, max_pairs={args.max_pairs}"
    )
    if args.mean_confidence_c is not None:
        logger.info(f"         mean_confidence_C override={args.mean_confidence_c:.3f}")
    if args.phase_ref_t is not None:
        logger.info(f"         phase_ref_t override={args.phase_ref_t:.2f}s")
    elif args.phase_ref_t_max is not None:
        logger.info(f"         phase_ref_t_max={args.phase_ref_t_max:.2f}s")
    logger.info(f"         拾取方式: DisperPicker CNN")
    logger.info("=" * 60)

    if not os.path.isdir(args.dat_dir):
        logger.error(f"DAT 目录不存在: {args.dat_dir}")
        return 1

    # 获取 DAT 文件列表
    if args.test_pair:
        dat_files = glob.glob(os.path.join(args.dat_dir, f"{args.test_pair}.dat"))
        if not dat_files:
            logger.error(f"未找到测试台对: {args.test_pair}.dat")
            return 1
    else:
        dat_files = sorted(glob.glob(os.path.join(args.dat_dir, args.dat_glob)))

    total_dat_files = len(dat_files)
    if not args.test_pair:
        dat_files, skipped_existing = filter_dat_files(
            dat_files,
            out_dir=args.out_dir,
            energy_dir=args.energy_dir,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            resume_existing=args.resume_existing,
            max_pairs=args.max_pairs,
        )
    else:
        skipped_existing = 0
    logger.info(
        f"DAT 总数: {total_dat_files}; 本分片待处理: {len(dat_files)}; "
        f"已完成跳过: {skipped_existing}"
    )

    if not dat_files:
        logger.info("没有需要处理的 DAT，直接结束")
        return 0

    # 启动前预检 CNN 权重完整性, 提前给出清晰错误 (归档易漏 saver/checkpoint)
    _verify_disperpicker_saver()
    # 提前初始化 Pick 和 qc 模块，避免每个台对重复加载 TensorFlow 模型
    logger.info("正在加载 DisperPicker CNN 模型...")
    from pick import Pick
    import qc as qc_mod
    picker = Pick()
    if args.mean_confidence_c is not None:
        picker.mean_confidence_C = args.mean_confidence_c
    logger.info("CNN 模型加载完成，开始批量处理")

    success = 0
    failed = 0

    for i, dat_path in enumerate(dat_files):
        pair_name = os.path.splitext(os.path.basename(dat_path))[0]
        logger.info(f"\n[{i+1}/{len(dat_files)}] {pair_name}")
        if process_one_pair(dat_path, args.out_dir, args.qc_plot_dir,
                            pick=picker, qc_mod=qc_mod,
                            phase_ref_t=args.phase_ref_t,
                            phase_ref_t_max=args.phase_ref_t_max,
                            energy_dir=args.energy_dir):
            success += 1
        else:
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"处理完成: 成功={success}, 失败={failed}, 共={len(dat_files)}")
    logger.info(f"频散曲线保存至: {args.out_dir}")
    logger.info("=" * 60)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
