"""Waveform-level check: EGF (-d/dt) convention vs stored CCF raw travel times.

For a random sample of pairs with existing target measurements, re-extract the
FTAN phase at the recorded group arrival from the despiked .dat waveforms, with
and without the negative time derivative, and verify that the raw phase travel
time difference equals +T/4 (i.e. a +pi/2 phase-convention offset).
"""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert

MEAS = Path(
    "/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_dat_20260724/"
    "fixed_bensen_alpha12_b1_b2_1/target_measurements.jsonl"
)
DAT_DIR = Path(
    "/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/"
    "wang_disperpicker_fig4_xmlcoords_allpairs_unspiked_20260702/dat_all"
)
OUT = Path(__file__).resolve().parent
ALPHA = 12.0
SAMPLE_PER_PERIOD = 60
SEED = 20260724


def gaussian_filter_analytic(waveform, dt_s, period_s, alpha):
    npts = waveform.size
    nfft = int(2 ** np.ceil(np.log2(max(npts, 1024))))
    spectrum = fft(waveform, nfft)
    freq = np.fft.fftfreq(nfft, d=dt_s)
    center_hz = 1.0 / period_s
    taper = np.exp(-alpha * ((np.abs(freq) - center_hz) / center_hz) ** 2)
    filtered = np.real(ifft(spectrum * taper))[:npts]
    return np.asarray(hilbert(filtered), dtype=complex)


def phase_at(time_s, analytic, arrival_s):
    re = float(np.interp(arrival_s, time_s, analytic.real))
    im = float(np.interp(arrival_s, time_s, analytic.imag))
    # scipy_phase_multiplier = -1 (forward-transform phase)
    return -math.atan2(im, re)


def raw_time(group_time_s, phi_rad, omega):
    return group_time_s + (phi_rad - math.pi / 4.0) / omega


def main():
    rows = []
    with MEAS.open() as fh:
        for line in fh:
            x = json.loads(line)
            if math.isfinite(float(x["raw_travel_time_s"])):
                rows.append(x)
    rng = random.Random(SEED)
    by_period = {}
    for x in rows:
        by_period.setdefault(float(x["period_s"]), []).append(x)

    results = []
    for period, group in sorted(by_period.items()):
        sample = rng.sample(group, min(SAMPLE_PER_PERIOD, len(group)))
        for x in sample:
            dat = DAT_DIR / f"{x['pair_name']}.dat"
            if not dat.exists():
                continue
            table = np.loadtxt(dat, skiprows=2, ndmin=2)
            time_s = table[:, 0]
            symmetric = 0.5 * (table[:, 1] + table[:, 2])
            keep = time_s >= 0.0
            time_s = time_s[keep]
            symmetric = symmetric[keep]

            t_inst = float(x["instantaneous_period_s"])
            omega = 2.0 * math.pi / t_inst
            tu = float(x["group_time_s"])
            if not (time_s[0] < tu < time_s[-1]):
                continue

            nominal = float(x["nominal_period_s"])
            analytic_ccf = gaussian_filter_analytic(
                symmetric, float(time_s[1] - time_s[0]), nominal, ALPHA
            )
            egf_waveform = -np.gradient(symmetric, time_s)
            analytic_egf = gaussian_filter_analytic(
                egf_waveform, float(time_s[1] - time_s[0]), nominal, ALPHA
            )

            t_ccf = raw_time(tu, phase_at(time_s, analytic_ccf, tu), omega)
            t_egf = raw_time(tu, phase_at(time_s, analytic_egf, tu), omega)

            delta = t_egf - t_ccf
            delta_wrapped = delta - round(delta / t_inst) * t_inst
            stored_diff = t_ccf - float(x["raw_travel_time_s"])
            stored_diff_wrapped = (
                stored_diff - round(stored_diff / t_inst) * t_inst
            )
            results.append(
                {
                    "pair_name": x["pair_name"],
                    "period_s": period,
                    "instantaneous_period_s": t_inst,
                    "delta_egf_minus_ccf_s": delta,
                    "delta_wrapped_s": delta_wrapped,
                    "delta_wrapped_over_T": delta_wrapped / t_inst,
                    "stored_diff_wrapped_s": stored_diff_wrapped,
                }
            )

    with (OUT / "sample_results.jsonl").open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    print(f"validated points: {len(results)}")
    for period in sorted(by_period):
        sub = [r for r in results if r["period_s"] == period]
        if not sub:
            continue
        frac = np.array([r["delta_wrapped_over_T"] for r in sub])
        repro = np.array([abs(r["stored_diff_wrapped_s"]) for r in sub])
        print(
            f"T={period}: n={len(sub)} "
            f"delta/T median={np.median(frac):+.4f} "
            f"(expect +0.25) p5={np.percentile(frac, 5):+.4f} "
            f"p95={np.percentile(frac, 95):+.4f} | "
            f"stored-raw reproduction |diff| median={np.median(repro):.4f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
