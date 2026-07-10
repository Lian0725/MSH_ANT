"""CuPy/Numpy backend for EGFAnalysisPy dispersion image calculation.

This module keeps the original EGFAnalysisTimeFreq parsing and metadata rules,
but moves the expensive frequency/filter image construction into vectorized FFT
operations. Use ``backend="cupy"`` for GPU execution and ``backend="numpy"`` for
local regression tests.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import interpolate, signal
from scipy.signal import windows


SCRIPT_DIR = Path(__file__).resolve().parent
EGF_DIR = SCRIPT_DIR / "EGFAnalysisPy"
if str(EGF_DIR) not in sys.path:
    sys.path.insert(0, str(EGF_DIR))

import EGFAnalysisTimeFreq  # noqa: E402


@dataclass(frozen=True)
class DispersionConfig:
    isEGF: bool = False
    StartT: float = 0.2
    EndT: float = 5.0
    DeltaT: float = 0.1
    StartV: float = 2.0
    EndV: float = 4.0
    DeltaV: float = 0.005
    WinAlpha: float = 0.1
    NoiseTime: float = 5.0
    MinSNR: float = 5.0
    WinPeriodNum: int = 5
    WinMinTime: float = 5.0
    FilterKaiserPara: float = 6.0
    MaxFilterLengthLog: int = 14


@dataclass
class DispersionImages:
    group_image: np.ndarray
    phase_image: np.ndarray
    snr: np.ndarray
    distance_km: float
    periods: np.ndarray
    velocities: np.ndarray
    backend: str
    phase_image_obs: Optional[np.ndarray] = None


def _array_module(backend: str):
    if backend == "numpy":
        return np
    if backend == "cupy":
        import cupy as cp

        return cp
    raise ValueError("backend must be 'numpy' or 'cupy'")


def _asnumpy(xp, value):
    if xp is np:
        return np.asarray(value)
    return xp.asnumpy(value)


def _next_pow2(value: int) -> int:
    return int(2 ** math.ceil(math.log2(max(1, int(value)))))


def _hilbert_axis_last(xp, data):
    n = int(data.shape[-1])
    spectrum = xp.fft.fft(data, axis=-1)
    h = xp.zeros(n, dtype=spectrum.dtype)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    analytic = xp.fft.ifft(spectrum * h.reshape((1,) * (data.ndim - 1) + (n,)), axis=-1)
    return analytic


def _envelope_image_fft(gfcn, xp, wave):
    alfa_x = np.array([0, 100, 250, 500, 1000, 2000, 4000, 20000], dtype=float)
    alfa_y = np.array([5, 8, 12, 20, 25, 35, 50, 75], dtype=float)
    gaussian_alpha = float(np.interp(gfcn.StaDist, alfa_x, alfa_y))

    t_point = np.asarray(gfcn.TPoint, dtype=float)
    pt_num = int(np.asarray(wave).shape[0])
    nfft = _next_pow2(max(pt_num, int(1024 * gfcn.SampleF)))

    wave_x = xp.asarray(wave, dtype=xp.float64)
    wave_fft = xp.fft.fft(wave_x, nfft)
    half = wave_fft[: nfft // 2 + 1]
    fxx = xp.arange(nfft // 2 + 1, dtype=xp.float64) / float(nfft) * float(gfcn.SampleF)
    fc = xp.asarray(1.0 / t_point, dtype=xp.float64).reshape((-1, 1))
    hf = xp.exp(-gaussian_alpha * (fxx.reshape((1, -1)) - fc) ** 2 / (fc**2))
    yy_half = half.reshape((1, -1)) * hf
    yy_fft = xp.concatenate([yy_half, xp.conj(yy_half[:, -2:0:-1])], axis=1)
    yy = xp.real(xp.fft.ifft(yy_fft, nfft, axis=1))
    envelope = xp.abs(_hilbert_axis_last(xp, yy))
    return _asnumpy(xp, envelope[:, :pt_num])


def _interp_velocity_image(gfcn, image_by_period, start_index, end_index, phase_shift_by_period=None):
    out = []
    time_index = np.arange(start_index, end_index)
    time = time_index * gfcn.SampleT
    for period_index in range(gfcn.NumCtrT):
        if phase_shift_by_period is None:
            trav_pt_v = gfcn.StaDist / (time * gfcn.SampleT if False else time)
        else:
            center_t = phase_shift_by_period[period_index]
            trav_pt_v = gfcn.StaDist / (time - center_t / 8.0)
            trav_pt_v[~np.isfinite(trav_pt_v)] = 100.0
        values = image_by_period[period_index, start_index:end_index]
        out.append(
            interpolate.interp1d(
                trav_pt_v[::-1],
                values[::-1],
                kind="cubic",
                bounds_error=False,
                fill_value=0,
            )(gfcn.VPoint)
        )
    return np.transpose(np.asarray(out))[::-1]


def group_velocity_image(gfcn, backend: str = "cupy"):
    xp = _array_module(backend)
    envelope_signal = _envelope_image_fft(gfcn, xp, gfcn.WinWaveClip)
    amp_s_t = np.max(envelope_signal, axis=1)
    noise = gfcn.NoiseWinWave * windows.tukey(gfcn.NoisePt, 0.2)
    envelope_noise = _envelope_image_fft(gfcn, xp, noise)
    gfcn.SNR_T = amp_s_t / np.mean(envelope_noise, axis=1)

    gfcn.HighSNRIndex = np.where(gfcn.SNR_T > gfcn.MinSNR)
    gfcn.SNRIndex = np.zeros(gfcn.NumCtrT)
    gfcn.SNRIndex[gfcn.HighSNRIndex] = 1
    for ii in range(1, gfcn.NumCtrT - 1):
        if gfcn.SNRIndex[ii] == 0:
            if (
                gfcn.SNR_T[ii] > gfcn.MinSNR / 2
                and gfcn.SNRIndex[ii - 1] == 1
                and gfcn.SNRIndex[ii + 1] == 1
            ):
                gfcn.SNRIndex[ii] = 1

    trav_pt_v = gfcn.StaDist / (np.asarray(range(gfcn.StartWin, gfcn.EndWin + 1)) * gfcn.SampleT)
    group_rows = []
    for i in range(gfcn.NumCtrT):
        values = envelope_signal[i, gfcn.StartWin : gfcn.EndWin + 1] / amp_s_t[i]
        group_rows.append(
            interpolate.interp1d(
                trav_pt_v[::-1],
                values[::-1],
                kind="cubic",
                bounds_error=False,
                fill_value=0,
            )(gfcn.VPoint)
        )
    group_image = np.transpose(np.asarray(group_rows))[::-1]
    gfcn.GroupVelocityImg = group_image
    return group_image.copy()


def _fir_filter_bank(gfcn, config: DispersionConfig):
    bandwidth = gfcn.DeltaT
    exponential = min(math.ceil(np.log2(1024 * gfcn.SampleF)), config.MaxFilterLengthLog)
    filter_length = int(2**exponential)
    filters = []
    for num_t in range(gfcn.NumCtrT):
        ctr_t = gfcn.StartT + num_t * gfcn.DeltaT
        low_f = (2 / gfcn.SampleF) / (ctr_t + 0.5 * bandwidth)
        high_f = (2 / gfcn.SampleF) / (ctr_t - 0.5 * bandwidth)
        filters.append(
            signal.firwin(
                filter_length + 1,
                [low_f, high_f],
                pass_zero=False,
                window=("kaiser", config.FilterKaiserPara),
            )
        )
    return np.asarray(filters), filter_length


def _time_variable_waves(gfcn, config: DispersionConfig, filter_length: int, time_variable: bool):
    half_filter_num = int(filter_length / 2)
    base_win_wave = np.concatenate((np.copy(gfcn.WinWaveClip), np.zeros(half_filter_num)))
    total_len = gfcn.WaveClipPt + half_filter_num
    if not time_variable:
        return base_win_wave.reshape((1, -1))

    if not hasattr(gfcn, "GroupDisperCurve"):
        return base_win_wave.reshape((1, -1))

    group_time = gfcn.StaDist / gfcn.GroupDisperCurve
    bad = np.where(group_time == np.inf)
    group_time[bad] = gfcn.StaDist / gfcn.StartV

    waves = np.zeros((gfcn.NumCtrT, total_len))
    for num_t in range(gfcn.NumCtrT):
        ctr_t = gfcn.StartT + num_t * gfcn.DeltaT
        winpt = np.round(np.maximum(config.WinPeriodNum * ctr_t, config.WinMinTime) * gfcn.SampleF)
        if winpt % 2 == 1:
            winpt = winpt + 1
        winpt = int(winpt)
        win_tukey = signal.windows.tukey(winpt, 0.2)
        grouppt = winpt + round(group_time[num_t] * gfcn.SampleF + 1)
        tmp_wave = np.concatenate((np.zeros(winpt), base_win_wave[: gfcn.WaveClipPt], np.zeros(winpt)))
        start = int(grouppt - winpt // 2)
        stop = int(grouppt + winpt // 2)
        if start < 0 or stop > tmp_wave.shape[0]:
            start_clip = max(0, start)
            stop_clip = min(tmp_wave.shape[0], stop)
            tukey_start = start_clip - start
            tukey_stop = tukey_start + (stop_clip - start_clip)
            tmp_wave[start_clip:stop_clip] *= win_tukey[tukey_start:tukey_stop]
            tmp_wave[:start_clip] = 0
            tmp_wave[stop_clip:] = 0
        else:
            tmp_wave[start:stop] *= win_tukey
            tmp_wave[:start] = 0
            tmp_wave[stop:] = 0
        waves[num_t, : gfcn.WaveClipPt] = tmp_wave[winpt : winpt + gfcn.WaveClipPt]
    return waves


def _batch_lfilter_filtfilt_fir(xp, waves, filters):
    waves_x = xp.asarray(waves, dtype=xp.float64)
    filters_x = xp.asarray(filters, dtype=xp.float64)
    num_t = filters.shape[0]
    n_signal = waves.shape[1]
    n_filter = filters.shape[1]
    nfft = _next_pow2(n_signal + n_filter - 1)
    filter_fft = xp.fft.rfft(filters_x, nfft, axis=1)
    if waves.shape[0] == 1:
        wave_fft = xp.fft.rfft(waves_x[0], nfft).reshape((1, -1))
        first = xp.fft.irfft(filter_fft * wave_fft, nfft, axis=1)[:, :n_signal]
    else:
        wave_fft = xp.fft.rfft(waves_x, nfft, axis=1)
        first = xp.fft.irfft(filter_fft * wave_fft, nfft, axis=1)[:, :n_signal]
    rev_fft = xp.fft.rfft(first[:, ::-1], nfft, axis=1)
    second = xp.fft.irfft(filter_fft * rev_fft, nfft, axis=1)[:, :n_signal]
    return second[:, ::-1][:num_t]


def phase_velocity_image(
    gfcn,
    config: DispersionConfig,
    backend: str = "cupy",
    time_variable: bool = False,
):
    xp = _array_module(backend)
    filters, filter_length = _fir_filter_bank(gfcn, config)
    waves = _time_variable_waves(gfcn, config, filter_length, time_variable=time_variable)
    filtered = _batch_lfilter_filtfilt_fir(xp, waves, filters)
    phase_img = _asnumpy(xp, filtered[:, : gfcn.WaveClipPt])
    max_abs = np.max(np.abs(phase_img), axis=1)
    max_abs[max_abs == 0] = 1.0
    phase_img = phase_img / max_abs.reshape((-1, 1))

    timeptnum = np.array(range(gfcn.StartWin, gfcn.EndWin))
    time_axis = timeptnum * gfcn.SampleT
    phase_rows = []
    for i in range(gfcn.NumCtrT):
        center_t = gfcn.StartT + i * gfcn.DeltaT
        trav_pt_v = gfcn.StaDist / (time_axis - center_t / 8)
        trav_pt_v[trav_pt_v == math.inf] = 100
        phase_rows.append(
            interpolate.interp1d(
                trav_pt_v[::-1],
                phase_img[i][gfcn.StartWin : gfcn.EndWin][::-1],
                kind="cubic",
                bounds_error=False,
                fill_value=0,
            )(gfcn.VPoint)
        )
    phase_image = np.transpose(np.asarray(phase_rows))[::-1]
    gfcn.PhaseVelocityImg = phase_image
    return phase_image.copy()


def build_gfcn(dat_path: Path, config: DispersionConfig):
    return EGFAnalysisTimeFreq.gfcn_analysis(
        DataFileName=str(dat_path),
        isEGF=config.isEGF,
        StartT=config.StartT,
        EndT=config.EndT,
        DeltaT=config.DeltaT,
        StartV=config.StartV,
        EndV=config.EndV,
        DeltaV=config.DeltaV,
        GreenFcnObjects=EGFAnalysisTimeFreq.gfcn_analysis.GreenFcnObjectsType.A_add_B,
        WinAlpha=config.WinAlpha,
        NoiseTime=config.NoiseTime,
        MinSNR=config.MinSNR,
    )


def compute_dispersion_images(
    dat_path: Path,
    config: Optional[DispersionConfig] = None,
    backend: str = "cupy",
    include_time_variable_phase: bool = False,
    group_curve: Optional[np.ndarray] = None,
):
    config = config or DispersionConfig()
    gfcn = build_gfcn(dat_path, config)
    group_image = group_velocity_image(gfcn, backend=backend)
    phase_image = phase_velocity_image(gfcn, config, backend=backend, time_variable=False)
    phase_image_obs = None
    if include_time_variable_phase:
        if group_curve is not None:
            gfcn.GroupDisperCurve = np.asarray(group_curve, dtype=float)
        phase_image_obs = phase_velocity_image(gfcn, config, backend=backend, time_variable=True)
    return DispersionImages(
        group_image=group_image,
        phase_image=phase_image,
        phase_image_obs=phase_image_obs,
        snr=np.asarray(gfcn.SNR_T),
        distance_km=float(gfcn.StaDist),
        periods=np.asarray(gfcn.TPoint),
        velocities=np.asarray(gfcn.VPoint),
        backend=backend,
    )


def _write_npz(path: Path, images: DispersionImages):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "group_image": images.group_image,
        "phase_image": images.phase_image,
        "periods": images.periods,
        "velocities": images.velocities,
        "snr": images.snr,
        "distance_km": images.distance_km,
        "backend": images.backend,
    }
    if images.phase_image_obs is not None:
        payload["phase_image_obs"] = images.phase_image_obs
    np.savez_compressed(path, **payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="GPU/CuPy dispersion image smoke backend.")
    parser.add_argument("--dat", required=True, type=Path)
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="cupy")
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--include-time-variable-phase", action="store_true")
    args = parser.parse_args(argv)

    start = time.time()
    images = compute_dispersion_images(
        args.dat,
        backend=args.backend,
        include_time_variable_phase=args.include_time_variable_phase,
    )
    elapsed = time.time() - start
    report = {
        "dat": str(args.dat),
        "backend": args.backend,
        "elapsed_s": elapsed,
        "group_shape": list(images.group_image.shape),
        "phase_shape": list(images.phase_image.shape),
        "distance_km": images.distance_km,
        "snr_finite": bool(np.all(np.isfinite(images.snr))),
    }
    if args.compare_cpu:
        cpu_gfcn = build_gfcn(args.dat, DispersionConfig())
        cpu_group = cpu_gfcn.GroupVelocityImgCalculate()
        cpu_phase = cpu_gfcn.PhaseVelocityImgCalculate(
            TimeVariableFilter=cpu_gfcn.TimeVariableFilterType.no,
            WinPeriodNum=DispersionConfig().WinPeriodNum,
            WinMinTime=DispersionConfig().WinMinTime,
            FilterKaiserPara=DispersionConfig().FilterKaiserPara,
            MaxFilterLengthLog=DispersionConfig().MaxFilterLengthLog,
        )
        report["compare_cpu"] = {
            "group_max_abs": float(np.max(np.abs(images.group_image - cpu_group))),
            "phase_max_abs": float(np.max(np.abs(images.phase_image - cpu_phase))),
        }
    if args.output_npz:
        _write_npz(args.output_npz, images)
        report["output_npz"] = str(args.output_npz)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
