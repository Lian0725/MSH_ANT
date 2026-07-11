"""
GPU-backed EGFAnalysisPy + DisperPicker dispersion extraction.

This runner keeps the legacy Wang Figure 4 output contract, but replaces the
expensive group/phase image construction with ``gpu_dispersion_backend``. The
DisperPicker CNN still runs the original two-pass picking workflow.
"""

import argparse
import glob
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
EGF_DIR = SCRIPT_DIR / "EGFAnalysisPy"
DISPER_DIR = EGF_DIR / "DisperPicker"
for _path in (SCRIPT_DIR, EGF_DIR, DISPER_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gpu_dispersion_backend as gpu_backend  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


CONFIG = dict(
    isEGF=False,
    StartT=0.2,
    EndT=5.0,
    DeltaT=0.1,
    StartV=2.0,
    EndV=4.0,
    DeltaV=0.005,
    MinDist=10.0,
    WinAlpha=0.1,
    NoiseTime=5.0,
    MinSNR=5.0,
    WinPeriodNum=5,
    WinMinTime=5,
    FilterKaiserPara=6,
    MaxFilterLengthLog=14,
)

MODEL_START_V = 0.5
GROUP_QC_CMAP = "cividis"
PHASE_QC_CMAP = "coolwarm"


@dataclass(frozen=True)
class PhaseRefIndex:
    index: int
    period_s: float
    source: str


class PairStatus(Enum):
    SUCCESS = "success"
    NO_PICK = "no_pick"
    ERROR = "error"


def backend_config():
    return gpu_backend.DispersionConfig(
        isEGF=CONFIG["isEGF"],
        StartT=CONFIG["StartT"],
        EndT=CONFIG["EndT"],
        DeltaT=CONFIG["DeltaT"],
        StartV=CONFIG["StartV"],
        EndV=CONFIG["EndV"],
        DeltaV=CONFIG["DeltaV"],
        WinAlpha=CONFIG["WinAlpha"],
        NoiseTime=CONFIG["NoiseTime"],
        MinSNR=CONFIG["MinSNR"],
        WinPeriodNum=CONFIG["WinPeriodNum"],
        WinMinTime=CONFIG["WinMinTime"],
        FilterKaiserPara=CONFIG["FilterKaiserPara"],
        MaxFilterLengthLog=CONFIG["MaxFilterLengthLog"],
    )


def model_velocity_count():
    return round((CONFIG["EndV"] - MODEL_START_V) / CONFIG["DeltaV"]) + 1


def period_count(start_t=None, end_t=None, delta_t=None):
    start_t = CONFIG["StartT"] if start_t is None else start_t
    end_t = CONFIG["EndT"] if end_t is None else end_t
    delta_t = CONFIG["DeltaT"] if delta_t is None else delta_t
    return round((end_t - start_t) / delta_t) + 1


def _clamp_period_index(index, n_period):
    return max(0, min(n_period - 1, int(index)))


def _period_to_index(period_s, start_t=None, end_t=None, delta_t=None):
    start_t = CONFIG["StartT"] if start_t is None else start_t
    end_t = CONFIG["EndT"] if end_t is None else end_t
    delta_t = CONFIG["DeltaT"] if delta_t is None else delta_t
    n_period = period_count(start_t, end_t, delta_t)
    index = round((float(period_s) - start_t) / delta_t)
    return _clamp_period_index(index, n_period)


def phase_ref_index_for_distance(
    distance_km,
    phase_ref_t=None,
    phase_ref_t_max=None,
    start_t=None,
    end_t=None,
    delta_t=None,
):
    start_t = CONFIG["StartT"] if start_t is None else start_t
    end_t = CONFIG["EndT"] if end_t is None else end_t
    delta_t = CONFIG["DeltaT"] if delta_t is None else delta_t
    n_period = period_count(start_t, end_t, delta_t)

    if phase_ref_t is not None:
        index = _period_to_index(phase_ref_t, start_t, end_t, delta_t)
        source = "manual"
    else:
        auto_index = int(
            0.6
            * min(
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

    return PhaseRefIndex(index=index, period_s=start_t + index * delta_t, source=source)


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


def pair_output_paths(dat_path, out_dir, energy_dir=None):
    pair_name = pair_name_from_dat_path(dat_path)
    required = [
        os.path.join(out_dir, f"GDisp.{pair_name}.txt"),
        os.path.join(out_dir, f"CDisp.{pair_name}.txt"),
    ]
    if energy_dir:
        required.append(os.path.join(energy_dir, pair_name + ".npz"))
    return required


def pair_outputs_exist(dat_path, out_dir, energy_dir=None):
    required = pair_output_paths(dat_path, out_dir, energy_dir=energy_dir)
    for path in required:
        try:
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                return False
        except OSError:
            return False

    if energy_dir:
        npz_path = required[-1]
        try:
            with np.load(npz_path, allow_pickle=False) as payload:
                if "failure_reason" in payload.files:
                    reason = str(np.asarray(payload["failure_reason"]).item()).strip()
                    if reason:
                        return False
        except Exception:
            return False
    return True


def remove_pair_outputs(dat_path, out_dir, energy_dir=None):
    for path in pair_output_paths(dat_path, out_dir, energy_dir=energy_dir):
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("unable to remove partial output %s: %s", path, exc)


def filter_dat_files(
    dat_files,
    out_dir,
    energy_dir=None,
    num_shards=1,
    shard_index=0,
    resume_existing=False,
    max_pairs=None,
):
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    selected = [
        path for index, path in enumerate(dat_files) if index % num_shards == shard_index
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


def pad_velocity_image(image, actual_start_v):
    model_nv = model_velocity_count()
    image = np.asarray(image)
    padded = np.zeros((model_nv, image.shape[1]), dtype=image.dtype)
    offset = round((float(actual_start_v) - MODEL_START_V) / CONFIG["DeltaV"])
    if offset < 0 or offset + image.shape[0] > model_nv:
        raise ValueError(
            f"Cannot pad velocity image: actual_start_v={actual_start_v}, "
            f"rows={image.shape[0]}, model_rows={model_nv}"
        )
    padded[offset : offset + image.shape[0], : image.shape[1]] = image
    return padded


def _haversine_km(lat_a, lon_a, lat_b, lon_b):
    radius_km = 6371.0
    lat1 = np.radians(float(lat_a))
    lat2 = np.radians(float(lat_b))
    dlat = lat2 - lat1
    dlon = np.radians(float(lon_b) - float(lon_a))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * radius_km * np.arcsin(np.sqrt(a)))


def _display_lon(lon):
    lon = float(lon)
    return lon + 360.0 if lon < 0 else lon


def station_info_from_dat(dat_path):
    with open(dat_path, "r", encoding="utf-8") as handle:
        lon_a, lat_a = map(float, handle.readline().split()[:2])
        lon_b, lat_b = map(float, handle.readline().split()[:2])
    dist = _haversine_km(lat_a, lon_a, lat_b, lon_b)
    return [dist, _display_lon(lon_a), lat_a, _display_lon(lon_b), lat_b]


def _plot_dispersion_qc(
    group_image,
    phase_image,
    group_v,
    phase_v,
    periods,
    actual_start_v,
    end_v,
    pair_name,
    dist,
    out_path,
):
    model_nv = group_image.shape[0]
    actual_nv = round((end_v - actual_start_v) / CONFIG["DeltaV"]) + 1
    offset = model_nv - actual_nv
    g_img = group_image[offset:, :]
    p_img = phase_image[offset:, :]
    y = np.linspace(actual_start_v, end_v, actual_nv)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{pair_name}  dist={dist:.1f} km", fontsize=14)

    ax = axes[0]
    z_max = np.abs(g_img).max() or 1.0
    pcm = ax.pcolormesh(
        periods,
        y,
        g_img,
        shading="auto",
        cmap=GROUP_QC_CMAP,
        vmin=0,
        vmax=z_max,
    )
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    nz = np.nonzero(group_v)[0]
    if len(nz):
        ax.plot(periods[nz], group_v[nz], "--k", linewidth=2, label="Group (CNN)")
        ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(periods[0], periods[-1])
    ax.set_ylim(actual_start_v, end_v)
    ax.set_xlabel("Period (s)", fontsize=13)
    ax.set_ylabel("Group Velocity (km/s)", fontsize=13)
    ax.set_title("Group velocity", fontsize=13)

    ax = axes[1]
    z_abs = np.abs(p_img).max() or 1.0
    pcm = ax.pcolormesh(
        periods,
        y,
        p_img,
        shading="auto",
        cmap=PHASE_QC_CMAP,
        vmin=-z_abs,
        vmax=z_abs,
    )
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    nz = np.nonzero(phase_v)[0]
    if len(nz):
        ax.plot(periods[nz], phase_v[nz], "--k", linewidth=2, label="Phase (CNN)")
    nz_g = np.nonzero(group_v)[0]
    if len(nz_g):
        ax.plot(periods[nz_g], group_v[nz_g], "--w", linewidth=1.5, label="Group ref")
    if len(nz) or len(nz_g):
        ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(periods[0], periods[-1])
    ax.set_ylim(actual_start_v, end_v)
    ax.set_xlabel("Period (s)", fontsize=13)
    ax.set_ylabel("Phase Velocity (km/s)", fontsize=13)
    ax.set_title("Phase velocity", fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_picker_and_qc(mean_confidence_c=None):
    os.environ["DISPERPICKER_ALLOW_GPU"] = "1"
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.chdir(EGF_DIR)

    from pick import Pick
    import qc as qc_mod

    picker = Pick()
    if mean_confidence_c is not None:
        picker.mean_confidence_C = mean_confidence_c
    return picker, qc_mod


def process_one_pair(
    dat_path: str,
    out_dir: str,
    qc_plot_dir: str | None,
    pick=None,
    qc_mod=None,
    phase_ref_t=None,
    phase_ref_t_max=None,
    energy_dir=None,
    backend="cupy",
    final_ct=2.0,
) -> PairStatus:
    pair_name = pair_name_from_dat_path(dat_path)
    start_time = time.time()
    config = backend_config()

    try:
        gfcn = gpu_backend.build_gfcn(Path(dat_path), config)
        dist = gfcn.StaDist
        if dist < CONFIG["MinDist"]:
            logger.info(
                "%s: 短台距提示（dist=%.1fkm < legacy MinDist=%.1fkm），继续处理",
                pair_name,
                dist,
                CONFIG["MinDist"],
            )

        sta_info = [
            dist,
            gfcn.Longitude_A,
            gfcn.Latitude_A,
            gfcn.Longitude_B,
            gfcn.Latitude_B,
        ]

        image_start = time.time()
        group_image = gpu_backend.group_velocity_image(gfcn, backend=backend)
        phase_image_raw = gpu_backend.phase_velocity_image(
            gfcn,
            config,
            backend=backend,
            time_variable=False,
        )
        image_elapsed = time.time() - image_start

        snr = np.asarray(gfcn.SNR_T)
        periods = np.linspace(
            CONFIG["StartT"],
            CONFIG["EndT"],
            period_count(),
        )
        model_nt = len(periods)
        batch_group_image = np.zeros((1, model_velocity_count(), model_nt))
        batch_phase_image = np.zeros((1, model_velocity_count(), model_nt))
        batch_group_image[0] = pad_velocity_image(group_image, gfcn.StartV)
        batch_phase_image[0] = pad_velocity_image(phase_image_raw, gfcn.StartV)

        if pick is None or qc_mod is None:
            pick, qc_mod = load_picker_and_qc()

        pick.mean_confidence_G = 0
        phase_ref = configure_picker_phase_ref(
            pick,
            distance_km=dist,
            phase_ref_t=phase_ref_t,
            phase_ref_t_max=phase_ref_t_max,
        )
        logger.debug(
            "%s: phase ref_T=%d (T=%.2fs, %s) before first pick",
            pair_name,
            phase_ref.index,
            phase_ref.period_s,
            phase_ref.source,
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

        if np.count_nonzero(group_velocity[0]) > 0:
            gfcn.GroupDisperCurve = np.asarray(group_velocity[0], dtype=float)
            phase_image = gpu_backend.phase_velocity_image(
                gfcn,
                config,
                backend=backend,
                time_variable=True,
            )
            batch_phase_image2 = np.zeros((1, model_velocity_count(), model_nt))
            batch_phase_image2[0] = pad_velocity_image(phase_image, gfcn.StartV)
        else:
            phase_image = phase_image_raw
            batch_phase_image2 = batch_phase_image
            logger.warning("%s: 群速度拾取失败，跳过时变滤波", pair_name)

        pick.mean_confidence_G = 0.3
        phase_ref = configure_picker_phase_ref(
            pick,
            distance_km=dist,
            phase_ref_t=phase_ref_t,
            phase_ref_t_max=phase_ref_t_max,
        )
        group_velocity, phase_velocity, prob_G, prob_C = pick.pick(
            group_image=batch_group_image,
            phase_image=batch_phase_image2,
            sta_info=[sta_info],
            snr=[snr],
            file_list=[pair_name],
            ct=float(final_ct),
            save_result=False,
        )

        if energy_dir:
            os.makedirs(energy_dir, exist_ok=True)
            velocity_axis = np.linspace(MODEL_START_V, CONFIG["EndV"], model_velocity_count())
            np.savez_compressed(
                os.path.join(energy_dir, pair_name + ".npz"),
                group_image=batch_group_image[0],
                phase_image=batch_phase_image2[0],
                phase_image_raw=batch_phase_image[0],
                periods=periods,
                velocities=velocity_axis,
                velocity_axis_km_s=velocity_axis,
                actual_velocity_axis_km_s=np.linspace(
                    float(gfcn.StartV),
                    CONFIG["EndV"],
                    round((CONFIG["EndV"] - float(gfcn.StartV)) / CONFIG["DeltaV"]) + 1,
                ),
                snr=snr,
                actual_start_v=float(gfcn.StartV),
                configured_start_v=float(CONFIG["StartV"]),
                end_v=float(CONFIG["EndV"]),
                delta_v=float(CONFIG["DeltaV"]),
                distance_km=float(dist),
                noise_time=float(CONFIG["NoiseTime"]),
                backend=str(backend),
                image_elapsed_s=float(image_elapsed),
            )

        snr_for_qc = snr if np.count_nonzero(snr) > 0 else np.ones(len(periods))
        group_v = qc_mod.qc(
            group_velocity[0],
            snr_for_qc,
            v_range=[CONFIG["StartV"], CONFIG["EndV"]],
            diff_range=[-0.07, 0.08],
            upward=0.4,
            each_stage_upward=0.3,
            min_len=round(len(periods) / 5),
            skip=False,
        )
        phase_v = qc_mod.qc(
            phase_velocity[0],
            snr_for_qc,
            v_range=[CONFIG["StartV"], CONFIG["EndV"]],
            diff_range=[-0.1, 0.1],
            upward=0,
            each_stage_upward=0.2,
            min_len=round(len(periods) / 5),
            skip=True,
        )

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"GDisp.{pair_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{sta_info[1]:.8f}    {sta_info[2]:.8f}\n")
            f.write(f"{sta_info[3]:.8f}    {sta_info[4]:.8f}\n")
            for t_val, g_val, s_val, pg in zip(periods, group_v, snr_for_qc, prob_G):
                f.write(f"{t_val:.2f}  {g_val:.3f}  {s_val:.3f}  {pg:.3f}\n")

        with open(os.path.join(out_dir, f"CDisp.{pair_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{sta_info[1]:.8f}    {sta_info[2]:.8f}\n")
            f.write(f"{sta_info[3]:.8f}    {sta_info[4]:.8f}\n")
            for t_val, c_val, s_val, pc in zip(periods, phase_v, snr_for_qc, prob_C):
                f.write(f"{t_val:.2f}  {c_val:.3f}  {s_val:.3f}  {pc:.3f}\n")

        if qc_plot_dir:
            os.makedirs(qc_plot_dir, exist_ok=True)
            try:
                _plot_dispersion_qc(
                    group_image=batch_group_image[0],
                    phase_image=batch_phase_image2[0],
                    group_v=group_v,
                    phase_v=phase_v,
                    periods=periods,
                    actual_start_v=float(gfcn.StartV),
                    end_v=CONFIG["EndV"],
                    pair_name=pair_name,
                    dist=dist,
                    out_path=os.path.join(qc_plot_dir, pair_name + ".jpg"),
                )
            except Exception as exc:
                logger.warning("%s: 绘图失败（不影响结果）: %s", pair_name, exc)

        has_result = np.count_nonzero(group_v) > 0 or np.count_nonzero(phase_v) > 0
        elapsed = time.time() - start_time
        if has_result:
            logger.info(
                "%s: dist=%.1fkm, G_pts=%d, C_pts=%d, image=%.2fs, total=%.2fs",
                pair_name,
                dist,
                np.count_nonzero(group_v),
                np.count_nonzero(phase_v),
                image_elapsed,
                elapsed,
            )
        else:
            logger.warning("%s: 未提取到有效频散点，total=%.2fs", pair_name, elapsed)
        return PairStatus.SUCCESS if has_result else PairStatus.NO_PICK

    except Exception as exc:
        logger.error("%s 处理失败: %s", pair_name, exc)
        logger.debug(traceback.format_exc())
        remove_pair_outputs(dat_path, out_dir, energy_dir=energy_dir)
        logger.warning("%s: 已清理异常产生的部分输出", pair_name)
        return PairStatus.ERROR


def format_seconds(seconds):
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def main(argv=None):
    parser = argparse.ArgumentParser(description="GPU/CuPy EGFAnalysisPy 频散提取")
    parser.add_argument("--dat_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--qc_plot_dir", default=None)
    parser.add_argument("--skip_qc_plot", action="store_true")
    parser.add_argument("--dat_glob", default="XD.*.dat")
    parser.add_argument("--test_pair", default=None)
    parser.add_argument("--mean_confidence_c", type=float, default=None)
    parser.add_argument("--phase_ref_t", type=float, default=None)
    parser.add_argument("--phase_ref_t_max", type=float, default=None)
    parser.add_argument("--energy_dir", "--full_pixel_data_dir", dest="energy_dir", default=None)
    parser.add_argument("--num_shards", "--num-shards", dest="num_shards", type=int, default=1)
    parser.add_argument("--shard_index", "--shard-index", dest="shard_index", type=int, default=0)
    parser.add_argument("--max_pairs", "--max-pairs", dest="max_pairs", type=int, default=None)
    parser.add_argument("--resume_existing", "--resume-existing", dest="resume_existing", action="store_true")
    parser.add_argument("--backend", choices=["cupy", "numpy"], default="cupy")
    parser.add_argument(
        "--final_ct",
        "--final-ct",
        dest="final_ct",
        type=float,
        default=2.0,
        help="Final DisperPicker distance-constraint multiplier. Default 2.0; use 0.01 to nearly disable the 2T cutoff.",
    )
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
        args.qc_plot_dir = os.path.join(os.path.dirname(args.out_dir), "qc_plot")

    logger.info("=" * 60)
    logger.info("GPU/CuPy EGFAnalysisPy 频散提取")
    logger.info("DAT 目录:  %s", args.dat_dir)
    logger.info("输出目录:  %s", args.out_dir)
    logger.info("QC 图目录: %s", args.qc_plot_dir if args.qc_plot_dir else "disabled")
    logger.info("全像素数据: %s", args.energy_dir if args.energy_dir else "disabled")
    logger.info("backend=%s; StartV=%.1f, EndV=%.1f, DeltaV=%.3f km/s", args.backend, CONFIG["StartV"], CONFIG["EndV"], CONFIG["DeltaV"])
    logger.info("final DisperPicker ct=%s", args.final_ct)
    logger.info(
        "shard=%d/%d, resume_existing=%s, max_pairs=%s",
        args.shard_index,
        args.num_shards,
        args.resume_existing,
        args.max_pairs,
    )
    logger.info("=" * 60)

    if not os.path.isdir(args.dat_dir):
        logger.error("DAT 目录不存在: %s", args.dat_dir)
        return 1

    if args.test_pair:
        dat_files = glob.glob(os.path.join(args.dat_dir, f"{args.test_pair}.dat"))
        if not dat_files:
            logger.error("未找到测试台对: %s.dat", args.test_pair)
            return 1
    else:
        dat_files = sorted(glob.glob(os.path.join(args.dat_dir, args.dat_glob)))

    total_dat_files = len(dat_files)
    if total_dat_files == 0:
        logger.error("DAT glob matched no files: %s", os.path.join(args.dat_dir, args.dat_glob))
        return 1
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
        "DAT 总数: %d; 本分片待处理: %d; 已完成跳过: %d",
        total_dat_files,
        len(dat_files),
        skipped_existing,
    )
    if not dat_files:
        logger.info("没有需要处理的 DAT，直接结束")
        logger.info("处理完成: 成功=0, 无有效拾取=0, 异常=0, 共=0")
        return 0

    logger.info("正在加载 DisperPicker CNN 模型...")
    picker, qc_mod = load_picker_and_qc(mean_confidence_c=args.mean_confidence_c)
    logger.info("CNN 模型加载完成，开始批量处理")

    success = 0
    no_pick = 0
    errors = 0
    run_start = time.time()

    for i, dat_path in enumerate(dat_files):
        pair_name = pair_name_from_dat_path(dat_path)
        if args.resume_existing and pair_outputs_exist(
            dat_path,
            args.out_dir,
            energy_dir=args.energy_dir,
        ):
            logger.info("[%d/%d] %s already completed; skip", i + 1, len(dat_files), pair_name)
            continue
        logger.info("[%d/%d] %s", i + 1, len(dat_files), pair_name)
        status = process_one_pair(
            dat_path,
            args.out_dir,
            args.qc_plot_dir,
            pick=picker,
            qc_mod=qc_mod,
            phase_ref_t=args.phase_ref_t,
            phase_ref_t_max=args.phase_ref_t_max,
            energy_dir=args.energy_dir,
            backend=args.backend,
            final_ct=args.final_ct,
        )
        if status is PairStatus.SUCCESS:
            success += 1
        elif status is PairStatus.NO_PICK:
            no_pick += 1
        else:
            errors += 1

        done = i + 1
        elapsed = time.time() - run_start
        avg = elapsed / done
        eta = avg * (len(dat_files) - done)
        logger.info(
            "progress=%d/%d success=%d no_pick=%d errors=%d avg=%.2fs ETA=%s",
            done,
            len(dat_files),
            success,
            no_pick,
            errors,
            avg,
            format_seconds(eta),
        )

    logger.info("=" * 60)
    logger.info(
        "处理完成: 成功=%d, 无有效拾取=%d, 异常=%d, 共=%d",
        success,
        no_pick,
        errors,
        len(dat_files),
    )
    logger.info("频散曲线保存至: %s", args.out_dir)
    logger.info("=" * 60)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
