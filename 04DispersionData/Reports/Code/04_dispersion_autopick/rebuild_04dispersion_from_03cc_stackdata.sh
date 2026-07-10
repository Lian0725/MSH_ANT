#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_ROOT="${FINAL_ROOT:-/mnt/data_hdd/MSH_ANT_Final}"
REPORT_ROOT="${FINAL_ROOT}/04DispersionData/Reports"
STACK_ROOT="${FINAL_ROOT}/03CC_StackData/2014/1D/STACK"
STACK_SPIKE_ROOT="${FINAL_ROOT}/03CC_StackData/2014/1D/STACK_SPIKE_REMOVED_DIAGFIT_20260628"
NONREMOVE_ROOT="${FINAL_ROOT}/04DispersionData/2014/1D/NonRemoveSpikes"
REMOVE_ROOT="${FINAL_ROOT}/04DispersionData/2014/1D/RemoveSpikes"
LOG_ROOT="${REPORT_ROOT}/RebuildLogs"

PY_FTAN="${PY_FTAN:-/mnt/data_hdd/lgx/MSH_ANT/envs/ftan/bin/python}"
PY_GPU="${PY_GPU:-/mnt/data_hdd/lgx/MSH_ANT/envs/disp_gpu/bin/python}"
BACKEND="${BACKEND:-cupy}"
SHARDS="${SHARDS:-16}"
FINAL_CT="${FINAL_CT:-0.01}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"

usage() {
  cat <<'EOF'
Usage:
  rebuild_04dispersion_from_03cc_stackdata.sh <command>

Commands:
  dat-unspiked        Rebuild DAT files from 03CC_StackData/STACK
  dat-spiked          Rebuild DAT files from 03CC_StackData/STACK_SPIKE_REMOVED_DIAGFIT_20260628
  extract-unspiked    Rebuild GDisp/CDisp/NPZ from NonRemoveSpikes DAT
  extract-spiked      Rebuild GDisp/CDisp/NPZ from RemoveSpikes DAT
  all-unspiked        Run DAT + dispersion extraction for NonRemoveSpikes
  all-spiked          Run DAT + dispersion extraction for RemoveSpikes

Environment variables:
  FINAL_ROOT          Default: /mnt/data_hdd/MSH_ANT_Final
  PY_FTAN             DAT conversion interpreter
  PY_GPU              GPU dispersion interpreter
  BACKEND             cupy or numpy; default cupy
  SHARDS              Number of parallel dispersion shards; default 16
  FINAL_CT            Final DisperPicker ct; default 0.01
  RESUME_EXISTING     1 to pass --resume_existing, 0 to omit

Notes:
  1. This script does not delete existing outputs.
  2. For a strict from-scratch rebuild, clear the target DAT/Curves/DispersionNPZ
     directories manually before running.
  3. This script assumes the runtime assets under Reports/Code are complete,
     including EGFAnalysisPy/DisperPicker/saver model weights.
EOF
}

need_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

need_exec() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "Interpreter is not executable: $path" >&2
    exit 1
  fi
}

run_convert() {
  local stack_root="$1"
  local dat_dir="$2"
  local label="$3"
  local log_file="${LOG_ROOT}/${label}_convert.log"

  mkdir -p "$dat_dir" "$LOG_ROOT"
  need_exec "$PY_FTAN"
  need_file "$stack_root"
  need_file "${SCRIPT_DIR}/convert_1d_stack_to_dat.py"

  {
    echo "[convert] start $(date '+%F %T %Z')"
    echo "[convert] stack_root=$stack_root"
    echo "[convert] dat_dir=$dat_dir"
    "$PY_FTAN" "${SCRIPT_DIR}/convert_1d_stack_to_dat.py" \
      --stack-root "$stack_root" \
      --out-dir "$dat_dir" \
      --component ZZ \
      --source-glob '1D.*' \
      --receiver-glob '1D.*' \
      --allow-zero-ngood
    echo "[convert] done $(date '+%F %T %Z')"
  } 2>&1 | tee "$log_file"
}

run_extract() {
  local dat_dir="$1"
  local curves_dir="$2"
  local npz_dir="$3"
  local label="$4"
  local log_dir="${LOG_ROOT}/${label}_logs"

  mkdir -p "$curves_dir" "$npz_dir" "$log_dir"
  need_exec "$PY_GPU"
  need_file "$dat_dir"
  need_file "${SCRIPT_DIR}/run_dispersion_gpu_mi09.py"
  need_file "${SCRIPT_DIR}/EGFAnalysisPy/DisperPicker/saver/checkpoint"

  local resume_flag=()
  if [[ "$RESUME_EXISTING" == "1" ]]; then
    resume_flag=(--resume_existing)
  fi

  echo "[extract] start $(date '+%F %T %Z')"
  echo "[extract] dat_dir=$dat_dir"
  echo "[extract] curves_dir=$curves_dir"
  echo "[extract] npz_dir=$npz_dir"
  echo "[extract] shards=$SHARDS backend=$BACKEND final_ct=$FINAL_CT resume=$RESUME_EXISTING"

  for ((i=0; i<SHARDS; i++)); do
    CUDA_VISIBLE_DEVICES=0 \
    "$PY_GPU" "${SCRIPT_DIR}/run_dispersion_gpu_mi09.py" \
      --dat_dir "$dat_dir" \
      --out_dir "$curves_dir" \
      --skip_qc_plot \
      --energy_dir "$npz_dir" \
      --backend "$BACKEND" \
      --final_ct "$FINAL_CT" \
      --num_shards "$SHARDS" \
      --shard_index "$i" \
      "${resume_flag[@]}" \
      > "${log_dir}/shard_${i}.log" 2>&1 &
  done
  wait

  echo "[extract] done $(date '+%F %T %Z')"
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 1
fi

case "$cmd" in
  dat-unspiked)
    run_convert "$STACK_ROOT" "${NONREMOVE_ROOT}/DatData/dat_all" "nonremove"
    ;;
  dat-spiked)
    run_convert "$STACK_SPIKE_ROOT" "${REMOVE_ROOT}/DatData/dat_all" "remove"
    ;;
  extract-unspiked)
    run_extract \
      "${NONREMOVE_ROOT}/DatData/dat_all" \
      "${NONREMOVE_ROOT}/Curves/curves_all_finalct001" \
      "${NONREMOVE_ROOT}/DispersionNPZ" \
      "nonremove"
    ;;
  extract-spiked)
    run_extract \
      "${REMOVE_ROOT}/DatData/dat_all" \
      "${REMOVE_ROOT}/Curves/curves_all_finalct001" \
      "${REMOVE_ROOT}/DispersionNPZ/full_pixel_data_all" \
      "remove"
    ;;
  all-unspiked)
    run_convert "$STACK_ROOT" "${NONREMOVE_ROOT}/DatData/dat_all" "nonremove"
    run_extract \
      "${NONREMOVE_ROOT}/DatData/dat_all" \
      "${NONREMOVE_ROOT}/Curves/curves_all_finalct001" \
      "${NONREMOVE_ROOT}/DispersionNPZ" \
      "nonremove"
    ;;
  all-spiked)
    run_convert "$STACK_SPIKE_ROOT" "${REMOVE_ROOT}/DatData/dat_all" "remove"
    run_extract \
      "${REMOVE_ROOT}/DatData/dat_all" \
      "${REMOVE_ROOT}/Curves/curves_all_finalct001" \
      "${REMOVE_ROOT}/DispersionNPZ/full_pixel_data_all" \
      "remove"
    ;;
  *)
    usage
    exit 1
    ;;
esac
