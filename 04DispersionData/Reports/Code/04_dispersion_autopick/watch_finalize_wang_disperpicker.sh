#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_20260630}"
CODE_ROOT="${CODE_ROOT:-/mnt/data_hdd/lgx/MSH_ANT/code/MSH_ANT}"
PY="${PY:-/mnt/data_hdd/lgx/MSH_ANT/envs/ftan/bin/python}"
SUPERVISOR_PID="${SUPERVISOR_PID:-1501903}"
DEST_HOST="${DEST_HOST:-lenovo}"
DEST_DIR="${DEST_DIR:-/mnt/data_hdd/lgx/MSH_ANT/outputs/reports/wang_disperpicker_fig4_xmlcoords_20260630}"
LOG_DIR="$OUT/logs"
LOG_FILE="$LOG_DIR/finalize_watch.log"
VERIFY_JSON="$OUT/verification_summary.json"
SHARDS="${SHARDS:-24}"
MAX_VERIFY_ATTEMPTS="${MAX_VERIFY_ATTEMPTS:-2}"

mkdir -p "$LOG_DIR"

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG_FILE"
}

run_verify() {
  "$PY" "$CODE_ROOT/scripts/04_dispersion/verify_disperpicker_full_run.py" \
    --output-dir "$OUT" \
    --sample-size 1000 \
    --report-json "$VERIFY_JSON" >> "$LOG_FILE" 2>&1
}

run_resume_pass() {
  log "resume_pass_start shards=$SHARDS"
  for i in $(seq 0 $((SHARDS - 1))); do
    "$PY" "$CODE_ROOT/scripts/04_dispersion/run_dispersion_mi09.py" \
      --dat_dir "$OUT/dat_ge10" \
      --out_dir "$OUT/curves_ge10" \
      --dat_glob "1D.*.dat" \
      --skip_qc_plot \
      --full_pixel_data_dir "$OUT/full_pixel_data_ge10" \
      --num_shards "$SHARDS" \
      --shard_index "$i" \
      --resume_existing > "$LOG_DIR/dispersion24_shard_${i}_of_${SHARDS}.log" 2>&1 &
  done
  wait
  log "resume_pass_done"
}

log "watcher_start supervisor=$SUPERVISOR_PID output=$OUT dest=$DEST_HOST:$DEST_DIR"

while ps -p "$SUPERVISOR_PID" >/dev/null 2>&1; do
  npz_count=$(find "$OUT/full_pixel_data_ge10" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  g_count=$(find "$OUT/curves_ge10" -maxdepth 1 -name 'GDisp.*.txt' 2>/dev/null | wc -l)
  c_count=$(find "$OUT/curves_ge10" -maxdepth 1 -name 'CDisp.*.txt' 2>/dev/null | wc -l)
  worker_count=$(pgrep -P "$SUPERVISOR_PID" -f 'run_dispersion_mi09.py' 2>/dev/null | wc -l || true)
  log "still_running workers=$worker_count npz=$npz_count g=$g_count c=$c_count"
  sleep 600
done

log "supervisor_exited"
attempt=1
while true; do
  if run_verify; then
    log "verification_ok attempt=$attempt"
    break
  fi
  log "verification_failed attempt=$attempt"
  if [ "$attempt" -ge "$MAX_VERIFY_ATTEMPTS" ]; then
    exit 1
  fi
  attempt=$((attempt + 1))
  run_resume_pass
done

log "rsync_start"
ssh "$DEST_HOST" "mkdir -p '$DEST_DIR'"
rsync -a --partial --info=stats2,progress2 \
  "$OUT/dat_ge10" \
  "$OUT/curves_ge10" \
  "$OUT/full_pixel_data_ge10" \
  "$OUT/logs" \
  "$VERIFY_JSON" \
  "$DEST_HOST:$DEST_DIR/" >> "$LOG_FILE" 2>&1
log "rsync_done"

ssh "$DEST_HOST" "find '$DEST_DIR/full_pixel_data_ge10' -maxdepth 1 -name '*.npz' | wc -l; find '$DEST_DIR/curves_ge10' -maxdepth 1 -name 'GDisp.*.txt' | wc -l; find '$DEST_DIR/curves_ge10' -maxdepth 1 -name 'CDisp.*.txt' | wc -l" >> "$LOG_FILE" 2>&1
log "watcher_done"
