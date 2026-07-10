#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPORT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
INVERSION_ROOT=$(cd "$REPORT_DIR/../../.." && pwd)
YEAR_DIR="$INVERSION_ROOT/2014/1D_1D"
FINAL_PARENT=$(cd "$INVERSION_ROOT/.." && pwd)
WORKERS="${WORKERS:-12}"
SPIKE_MODE="remove"
OUT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spike-mode)
      SPIKE_MODE="$2"
      shift 2
      ;;
    --output-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1" >&2
      exit 1
      ;;
  esac
done

case "$SPIKE_MODE" in
  remove)
    CURVE_KIND_DIR="RemoveSpikes"
    DEFAULT_OUT_ROOT="$YEAR_DIR/01_去除尖峰数据"
    ;;
  nonremove)
    CURVE_KIND_DIR="NonRemoveSpikes"
    DEFAULT_OUT_ROOT="$YEAR_DIR/02_不去除尖峰数据"
    ;;
  *)
    echo "--spike-mode 只能是 remove 或 nonremove" >&2
    exit 1
    ;;
esac

OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"
CURVES_DIR="$FINAL_PARENT/04DispersionData/2014/1D_1D/$CURVE_KIND_DIR/Curves/curves_all_finalct001"
SCREEN_OUT="$OUT_ROOT/01_筛选后数据/figure4_screening"
INV_OUT="$OUT_ROOT/02_反演结果数据/phase_velocity_maps_3_3p5_4"

mkdir -p "$SCREEN_OUT" "$INV_OUT"

python3 "$REPORT_DIR/02_代码/04_筛选脚本/plot_disperpicker_wang_figure4.py" \
  --curves-dir "$CURVES_DIR" \
  --group-curves-dir "$CURVES_DIR" \
  --output-dir "$SCREEN_OUT" \
  --periods 3.0,3.5,4.0 \
  --paper-standard \
  --min-snr 4.0 \
  --group-vmin 0.0 \
  --min-wavelengths 1.0 \
  --disperpicker-final-ct 0.01 \
  --workers "$WORKERS"

python3 "$REPORT_DIR/02_代码/05_二维成图脚本/plot_wang_fig5_fig6_from_disperpicker.py" \
  --measurements-csv "$SCREEN_OUT/measurements_period_corrected.csv" \
  --curves-dir "$CURVES_DIR" \
  --output-dir "$INV_OUT" \
  --periods 3.0,3.5,4.0

printf '重建完成，模式=%s，输出目录=%s\n' "$SPIKE_MODE" "$OUT_ROOT"
