# FTAN_Wang

Wang et al. (2017) Figure 4 FTAN processing code and the latest reproduced
figure.  The source data are intentionally not included: the full pair-result
archive is about 18 GB and remains on the `work` server.

## Contents

- `code/bensen_phase_ftan.py` — FTAN, phase picking, Wang SNR/group-velocity
  QC, target-period resampling, and cycle-resolution utilities.
- `code/run_work_reproduce_wang_figure4_allpairs.py` — all-pair Stage-B FTAN
  runner used to generate the target measurements.
- `code/wang_ftan_validation.py` — validation helpers used by the runner.
- `code/check_egf_vs_ccf_phase.py` — waveform-level verification of the
  `+T/4` conversion from the stored Bensen CCF phase convention to the EGF
  convention used in the plotted Figure 4.
- `code/rebuild_wang_figure4_egf_no_preleft.py` — latest Figure 4 renderer.
  It retains all SNR/group-velocity-qualified points in the left column,
  applies the phase-cycle correction and one-wavelength screen only to the
  right column, uses Wang's tall panel geometry, and plots only the two green
  `±T/2` boundaries enclosing the selected branch.
- `figure/wang_figure4_egf_no_preleft_wang_aspect_times.png` — latest output
  with Times New Roman typography.

## Re-rendering

The renderer reads the measurement archives from the experiment root on
`work`.  It requires a Times New Roman `.ttf` file.  The font file is not
included here because it is a system font; pass a locally licensed copy with
`--font-path`:

```bash
python code/rebuild_wang_figure4_egf_no_preleft.py \
  --font-path '/path/to/Times New Roman.ttf'
```

The rendered server output is:

```text
/mnt/data_hdd/lgx/MSH_ANT/experiments/wang_ftan_dat_20260724/
egf_convention_check_no_preleft/wang_figure4_egf_no_preleft.png
```
