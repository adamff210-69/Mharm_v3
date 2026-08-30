#!/usr/bin/env bash
# Multi-HARM demo driver: runs the full pipeline in the v3 execution order.
# Usage:  source demo_env.sh && bash demo_run.sh            # all stages
#         source demo_env.sh && bash demo_run.sh 3          # only stage 03
#         source demo_env.sh && bash demo_run.sh 4 5 6 7 8  # stages 04..08
# Every stage is idempotent/resumable — safe to kill and re-run.
set -e
cd "$(dirname "$0")"

run() {
  n="$1"; script="$2"
  echo ""
  echo "=============================================================="
  echo "  STAGE ${n}  ->  ${script}   $(date +%H:%M:%S)"
  echo "=============================================================="
  python "${script}"
}

stages=(
  "01 01_setup_and_validate.py"
  "02 02_build_dataset.py"
  "03 03_extract_signals.py"
  "04 04_calibrate_hstar.py"
  "05 05_baseline_attn_tracker.py"
  "06 06_calibrate_general.py"
  "07 07_calibrate_specialists.py"
  "08 08_meta_decision.py"
  "09 09_experiments_analysis.py --with-model"
  "10 10_figures_report.py"
)

if [ $# -eq 0 ]; then
  wanted="01 02 03 04 05 06 07 08 09 10"
else
  wanted="$*"
fi

for s in "${stages[@]}"; do
  n="${s%% *}"
  if echo " $wanted " | grep -q " $n "; then
    # shellcheck disable=SC2086
    run "$n" $s
  fi
done
echo ""
echo "DEMO PIPELINE COMPLETE. Key artifacts:"
echo "  out/experiments/SUMMARY.md   <- success-criteria check + span audit"
echo "  out/experiments/table_48.json<- the spine table"
echo "  out/report/RESULTS.md        <- paper-facing report (4.8/4.9 tables)"
echo "  out/figures/                 <- all figures"
