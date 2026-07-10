#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT='/mnt/data_hdd/lgx/MSH_ANT'
CODE_ROOT='/mnt/data_hdd/lgx/MSH_ANT/code/MSH_ANT'
RUN_ROOT='/mnt/data_hdd/lgx/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620'
LENOVO_HOST='lenovo'
LENOVO_RESULT='/mnt/data_hdd/MSH_ANT/stack/2014/1D_WANG_PWS_150s_20260620'
PYTHON_BIN='/mnt/data_hdd/lgx/MSH_ANT/.venvs/2014_wang_pws/bin/python'
CONDA_ENV='noise'
WORKERS=''
FFT_THREADS='4'

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/checkpoints" "$RUN_ROOT/qc" "$RUN_ROOT/STACK"
cd "$CODE_ROOT"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

run_python() {
  "$PYTHON_BIN" "$CODE_ROOT/scripts/02_cc/run_2014_1d_wang_pws.py" "$@"
}

{
  echo "[pipeline] start $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  run_python manifest
  args=(correlate --resume --fft-threads "$FFT_THREADS")
  if [[ -n "$WORKERS" ]]; then
    args+=(--workers "$WORKERS")
  fi
  run_python "${args[@]}"
  run_python export
  run_python audit
  echo "[pipeline] transferring results to lenovo"
  ssh "$LENOVO_HOST" "mkdir -p '$LENOVO_RESULT'"
  rsync -aH --partial --append-verify --info=progress2     "$RUN_ROOT/" "$LENOVO_HOST:$LENOVO_RESULT/"
  echo "[pipeline] finished $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} 2>&1 | tee -a "$RUN_ROOT/logs/pipeline.log"
