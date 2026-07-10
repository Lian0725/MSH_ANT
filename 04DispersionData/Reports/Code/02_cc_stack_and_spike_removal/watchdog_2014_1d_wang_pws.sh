#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/mnt/data_hdd/lgx/MSH_ANT}"
RUN_ROOT="${RUN_ROOT:-${WORK_ROOT}/stack/2014/1D_WANG_PWS_150s_20260620}"
CODE_ROOT="${CODE_ROOT:-${WORK_ROOT}/code/MSH_ANT}"
SESSION="${SESSION:-wang2014pws}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${RUN_ROOT}/logs/watchdog.log}"
RUN_SCRIPT="${RUN_SCRIPT:-${RUN_ROOT}/run_pipeline.sh}"
PIPELINE_LOG="${PIPELINE_LOG:-${RUN_ROOT}/logs/pipeline.log}"
COMPLETION_MARKER="${COMPLETION_MARKER:-${RUN_ROOT}/qc/pipeline_complete.json}"
MIN_FREE_GB="${MIN_FREE_GB:-200}"
MAX_LOG_STALE_SECONDS="${MAX_LOG_STALE_SECONDS:-7200}"
MAX_CHECKPOINT_STALE_SECONDS="${MAX_CHECKPOINT_STALE_SECONDS:-10800}"

mkdir -p "$(dirname "$WATCHDOG_LOG")"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$WATCHDOG_LOG"
}

mtime_age() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo 999999999
    return
  fi
  echo $(( $(date +%s) - $(stat -c %Y "$path") ))
}

latest_file() {
  local root="$1"
  local pattern="$2"
  find "$root" -type f -name "$pattern" -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}'
}

pipeline_completed() {
  if [[ -f "$COMPLETION_MARKER" ]]; then
    return 0
  fi
  if [[ -f "$PIPELINE_LOG" ]] && grep -Fq "[pipeline] finished" "$PIPELINE_LOG"; then
    return 0
  fi
  return 1
}

if [[ ! -d /mnt/data_hdd ]]; then
  log "ERROR /mnt/data_hdd is unavailable; refusing to run on system disk"
  exit 1
fi

free_gb=$(df -BG --output=avail /mnt/data_hdd | awk 'NR==2 {gsub(/G/, "", $1); print $1}')
if [[ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]]; then
  log "ERROR low disk space on /mnt/data_hdd: ${free_gb}G available, need ${MIN_FREE_GB}G"
  exit 1
fi

if command -v free >/dev/null 2>&1; then
  log "INFO memory $(free -h | awk 'NR==2 {print "used="$3",available="$7}') disk_free=${free_gb}G"
else
  log "INFO disk_free=${free_gb}G"
fi

if pipeline_completed; then
  log "INFO pipeline already completed; not restarting tmux session ${SESSION}"
  log "OK watchdog check complete"
  exit 0
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ -x "$RUN_SCRIPT" ]]; then
    log "WARN tmux session ${SESSION} missing; restarting with ${RUN_SCRIPT}"
    cd "$CODE_ROOT"
    tmux new-session -d -s "$SESSION" "$RUN_SCRIPT"
  else
    log "ERROR tmux session ${SESSION} missing and run script is not executable: ${RUN_SCRIPT}"
    exit 1
  fi
fi

latest_log=$(latest_file "${RUN_ROOT}/logs" '*.log')
if [[ -n "${latest_log:-}" ]]; then
  log_age=$(mtime_age "$latest_log")
  if [[ "$log_age" -gt "$MAX_LOG_STALE_SECONDS" ]]; then
    log "WARN latest log stale age=${log_age}s file=${latest_log}"
  fi
else
  log "WARN no log files found under ${RUN_ROOT}/logs"
fi

latest_checkpoint=$(latest_file "${RUN_ROOT}/checkpoints" '*.h5')
if [[ -n "${latest_checkpoint:-}" ]]; then
  checkpoint_age=$(mtime_age "$latest_checkpoint")
  if [[ "$checkpoint_age" -gt "$MAX_CHECKPOINT_STALE_SECONDS" ]]; then
    log "WARN latest checkpoint stale age=${checkpoint_age}s file=${latest_checkpoint}"
  fi
else
  log "WARN no checkpoints found yet under ${RUN_ROOT}/checkpoints"
fi

log "OK watchdog check complete"
