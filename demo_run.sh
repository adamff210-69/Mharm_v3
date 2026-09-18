#!/usr/bin/env bash
# Multi-HARM demo driver: runs the full pipeline in the v3 execution order.
# Usage:  source demo_env.sh && bash demo_run.sh            # all stages
#         source demo_run.sh 3                              # only stage 03
#         source demo_env.sh && bash demo_run.sh 4 5 6 7 8  # stages 04..08
# Every stage is idempotent/resumable — safe to kill and re-run.
set -e
cd "$(dirname "$0")"

# NOTE (v3.1 fix): the old run() took only "$1"/"$2", so the "--with-model"
# flag on stage 09 was parsed away and never reached python — the demo silently
# skipped the forward-included latency measurement the runbook promises.
# Stages are now "number|full command line", executed as argv arrays.
stages=(
  "01|python 01_setup_and_validate.py"
  "02|python 02_build_dataset.py"
  "03|python 03_extract_signals.py"
  "04|python 04_calibrate_hstar.py"
  "05|python 05_baseline_attn_tracker.py"
  "06|python 06_calibrate_general.py"
  "07|python 07_calibrate_specialists.py"
  "08|python 08_meta_decision.py"
  "09|python 09_experiments_analysis.py --with-model --calib-sweep"
  "10|python 10_figures_report.py"
  "11|python 11_reproducibility.py"
)

if [ $# -eq 0 ]; then
  wanted="01 02 03 04 05 06 07 08 09 10 11"
else
  wanted="$*"
fi

for entry in "${stages[@]}"; do
  n="${entry%%|*}"
  cmd="${entry#*|}"
  if echo " $wanted " | grep -q " $n "; then
    echo ""
    echo "=============================================================="
    echo "  STAGE ${n}  ->  ${cmd}   $(date +%H:%M:%S)"
    echo "=============================================================="
    # shellcheck disable=SC2086
    read -r -a argv <<< "$cmd"
    "${argv[@]}"
  fi
done
echo ""
echo "DEMO PIPELINE COMPLETE. Key artifacts:"
echo "  out/experiments/SUMMARY.md    <- success-criteria check + span audit + §4.4"
echo "  out/experiments/table_48.json <- the spine table"
echo "  out/report/RESULTS.md         <- paper-facing report (§4.8/§4.9 tables)"
echo "  out/figures/                  <- all figures (incl. fig_calib_sweep.png)"
