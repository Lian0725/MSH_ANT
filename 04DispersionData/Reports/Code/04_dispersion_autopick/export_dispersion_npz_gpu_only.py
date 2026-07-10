"""
Export DisperPicker-style dispersion energy NPZ files only.

This keeps the same image-generation and first-pass group-pick logic as
``run_dispersion_gpu_mi09.py`` so the saved NPZ structure stays compatible with
the existing reporting workflow, but it skips the final CNN pick, QC, and curve
text outputs. The goal is to backfill energy images as fast as possible.
"""

import argparse
import glob
import os
import time

import numpy as np

import run_dispersion_gpu_mi09 as gpu_run


logger = gpu_run.logger


def write_empty_npz(dat_path, energy_dir, backend, failure_reason):
    pair_name = gpu_run.pair_name_from_dat_path(dat_path)
    sta_info = gpu_run.station_info_from_dat(dat_path)
    periods = np.linspace(gpu_run.CONFIG["StartT"], gpu_run.CONFIG["EndT"], gpu_run.period_count())
    velocity_axis = np.linspace(gpu_run.MODEL_START_V, gpu_run.CONFIG["EndV"], gpu_run.model_velocity_count())
    actual_axis = np.linspace(
        gpu_run.CONFIG["StartV"],
        gpu_run.CONFIG["EndV"],
        round((gpu_run.CONFIG["EndV"] - gpu_run.CONFIG["StartV"]) / gpu_run.CONFIG["DeltaV"]) + 1,
    )
    image = np.zeros((gpu_run.model_velocity_count(), len(periods)), dtype=float)
    os.makedirs(energy_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(energy_dir, pair_name + ".npz"),
        group_image=image,
        phase_image=image,
        phase_image_raw=image,
        periods=periods,
        velocities=velocity_axis,
        velocity_axis_km_s=velocity_axis,
        actual_velocity_axis_km_s=actual_axis,
        snr=np.zeros_like(periods),
        actual_start_v=float(gpu_run.CONFIG["StartV"]),
        configured_start_v=float(gpu_run.CONFIG["StartV"]),
        end_v=float(gpu_run.CONFIG["EndV"]),
        delta_v=float(gpu_run.CONFIG["DeltaV"]),
        distance_km=float(sta_info[0]),
        noise_time=float(gpu_run.CONFIG["NoiseTime"]),
        backend=str(backend),
        failure_reason=str(failure_reason),
        image_elapsed_s=0.0,
    )


def npz_exists(dat_path, energy_dir):
    pair_name = gpu_run.pair_name_from_dat_path(dat_path)
    return os.path.exists(os.path.join(energy_dir, pair_name + ".npz"))


def filter_dat_files(dat_files, energy_dir, num_shards=1, shard_index=0, resume_existing=False):
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
            if npz_exists(path, energy_dir):
                skipped_existing += 1
            else:
                remaining.append(path)
        selected = remaining
    return selected, skipped_existing


def process_one_pair(
    dat_path,
    energy_dir,
    pick=None,
    phase_ref_t=None,
    phase_ref_t_max=None,
    backend="cupy",
):
    pair_name = gpu_run.pair_name_from_dat_path(dat_path)
    start_time = time.time()
    config = gpu_run.backend_config()

    try:
        gfcn = gpu_run.gpu_backend.build_gfcn(gpu_run.Path(dat_path), config)
        dist = gfcn.StaDist
        if dist < gpu_run.CONFIG["MinDist"]:
            logger.info(
                "%s: 短台距提示（dist=%.1fkm < legacy MinDist=%.1fkm），继续处理",
                pair_name,
                dist,
                gpu_run.CONFIG["MinDist"],
            )

        sta_info = [
            dist,
            gfcn.Longitude_A,
            gfcn.Latitude_A,
            gfcn.Longitude_B,
            gfcn.Latitude_B,
        ]

        image_start = time.time()
        group_image = gpu_run.gpu_backend.group_velocity_image(gfcn, backend=backend)
        phase_image_raw = gpu_run.gpu_backend.phase_velocity_image(
            gfcn,
            config,
            backend=backend,
            time_variable=False,
        )

        snr = np.asarray(gfcn.SNR_T)
        periods = np.linspace(
            gpu_run.CONFIG["StartT"],
            gpu_run.CONFIG["EndT"],
            gpu_run.period_count(),
        )
        model_nt = len(periods)
        batch_group_image = np.zeros((1, gpu_run.model_velocity_count(), model_nt))
        batch_phase_image = np.zeros((1, gpu_run.model_velocity_count(), model_nt))
        batch_group_image[0] = gpu_run.pad_velocity_image(group_image, gfcn.StartV)
        batch_phase_image[0] = gpu_run.pad_velocity_image(phase_image_raw, gfcn.StartV)

        if pick is None:
            pick, _ = gpu_run.load_picker_and_qc()

        pick.mean_confidence_G = 0
        phase_ref = gpu_run.configure_picker_phase_ref(
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
        group_velocity, _, _, _ = pick.pick(
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
            phase_image = gpu_run.gpu_backend.phase_velocity_image(
                gfcn,
                config,
                backend=backend,
                time_variable=True,
            )
            batch_phase_image2 = np.zeros((1, gpu_run.model_velocity_count(), model_nt))
            batch_phase_image2[0] = gpu_run.pad_velocity_image(phase_image, gfcn.StartV)
        else:
            phase_image = phase_image_raw
            batch_phase_image2 = batch_phase_image
            logger.warning("%s: 群速度拾取失败，跳过时变滤波", pair_name)

        image_elapsed = time.time() - image_start

        os.makedirs(energy_dir, exist_ok=True)
        velocity_axis = np.linspace(
            gpu_run.MODEL_START_V,
            gpu_run.CONFIG["EndV"],
            gpu_run.model_velocity_count(),
        )
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
                gpu_run.CONFIG["EndV"],
                round((gpu_run.CONFIG["EndV"] - float(gfcn.StartV)) / gpu_run.CONFIG["DeltaV"]) + 1,
            ),
            snr=snr,
            actual_start_v=float(gfcn.StartV),
            configured_start_v=float(gpu_run.CONFIG["StartV"]),
            end_v=float(gpu_run.CONFIG["EndV"]),
            delta_v=float(gpu_run.CONFIG["DeltaV"]),
            distance_km=float(dist),
            noise_time=float(gpu_run.CONFIG["NoiseTime"]),
            backend=str(backend),
            image_elapsed_s=float(image_elapsed),
        )

        elapsed = time.time() - start_time
        logger.info(
            "%s: npz_only dist=%.1fkm, G_seed_pts=%d, image=%.2fs, total=%.2fs",
            pair_name,
            dist,
            np.count_nonzero(group_velocity[0]),
            image_elapsed,
            elapsed,
        )
        return True

    except Exception as exc:
        logger.error("%s 处理失败: %s", pair_name, exc)
        try:
            write_empty_npz(
                dat_path=dat_path,
                energy_dir=energy_dir,
                backend=backend,
                failure_reason="npz_only failure: %s: %s" % (type(exc).__name__, exc),
            )
            logger.warning("%s: 已写出失败占位 NPZ，避免 resume 重复处理", pair_name)
        except Exception as empty_exc:
            logger.error("%s: 写出失败占位 NPZ 失败: %s", pair_name, empty_exc)
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GPU/CuPy export dispersion NPZ only")
    parser.add_argument("--dat_dir", required=True)
    parser.add_argument("--energy_dir", required=True)
    parser.add_argument("--dat_glob", default="1D.*.dat")
    parser.add_argument("--test_pair", default=None)
    parser.add_argument("--phase_ref_t", type=float, default=None)
    parser.add_argument("--phase_ref_t_max", type=float, default=None)
    parser.add_argument("--num_shards", "--num-shards", dest="num_shards", type=int, default=1)
    parser.add_argument("--shard_index", "--shard-index", dest="shard_index", type=int, default=0)
    parser.add_argument("--resume_existing", "--resume-existing", dest="resume_existing", action="store_true")
    parser.add_argument("--backend", choices=["cupy", "numpy"], default="cupy")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isdir(args.dat_dir):
        logger.error("DAT 目录不存在: %s", args.dat_dir)
        return 1

    if args.test_pair:
        dat_files = glob.glob(os.path.join(args.dat_dir, args.test_pair + ".dat"))
        if not dat_files:
            logger.error("未找到测试台对: %s.dat", args.test_pair)
            return 1
        skipped_existing = 0
    else:
        dat_files = sorted(glob.glob(os.path.join(args.dat_dir, args.dat_glob)))
        dat_files, skipped_existing = filter_dat_files(
            dat_files,
            energy_dir=args.energy_dir,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            resume_existing=args.resume_existing,
        )

    logger.info("=" * 60)
    logger.info("GPU/CuPy dispersion NPZ-only export")
    logger.info("DAT 目录:  %s", args.dat_dir)
    logger.info("NPZ 目录:  %s", args.energy_dir)
    logger.info(
        "shard=%d/%d, resume_existing=%s, pending=%d, skipped_existing=%d, backend=%s",
        args.shard_index,
        args.num_shards,
        args.resume_existing,
        len(dat_files),
        skipped_existing,
        args.backend,
    )
    logger.info("=" * 60)

    if not dat_files:
        logger.info("没有需要处理的 DAT，直接结束")
        return 0

    logger.info("正在加载 DisperPicker CNN 模型（仅首轮群速度种子）...")
    picker, _ = gpu_run.load_picker_and_qc()
    logger.info("模型加载完成，开始导出 NPZ")

    success = 0
    failed = 0
    elapsed_sum = 0.0
    for index, dat_path in enumerate(dat_files, 1):
        pair_name = gpu_run.pair_name_from_dat_path(dat_path)
        if args.resume_existing and npz_exists(dat_path, args.energy_dir):
            logger.info("[%d/%d] %s already completed; skip", index, len(dat_files), pair_name)
            continue

        logger.info("[%d/%d] %s", index, len(dat_files), pair_name)
        t0 = time.time()
        ok = process_one_pair(
            dat_path=dat_path,
            energy_dir=args.energy_dir,
            pick=picker,
            phase_ref_t=args.phase_ref_t,
            phase_ref_t_max=args.phase_ref_t_max,
            backend=args.backend,
        )
        elapsed_sum += time.time() - t0
        if ok:
            success += 1
        else:
            failed += 1

        avg = elapsed_sum / index
        remaining = max(0, len(dat_files) - index)
        logger.info(
            "progress=%d/%d success=%d failed=%d avg=%.2fs ETA=%s",
            index,
            len(dat_files),
            success,
            failed,
            avg,
            gpu_run.format_seconds(avg * remaining),
        )

    logger.info(
        "NPZ-only export finished: success=%d failed=%d total=%d",
        success,
        failed,
        len(dat_files),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
