# Multi-HARM — reduced-size DEMO config for a Colab T4.
# Source this before each run cell (or before demo_run.sh):
#   !source ./demo_env.sh && python 03_extract_signals.py
#
# Sizes: 400 clean + 400 injected (4 types x 5 goals x 20) = 800 samples.
# Expected on a free T4 (4-bit, eager attention):
#   03 extraction ~30-75 min | everything else minutes.
# Total door-to-door: ~1-1.5 h (incl. ~5 GB model download).
#
# FASTER variant (~half the time): N_CLEAN=200, N_INJ_PER_CELL=15,
# N_BASE_PAIRS=450  (still >=5/cell so all splits stay non-degenerate).

export MULTI_HARM_N_CLEAN=400
export MULTI_HARM_N_INJ_PER_CELL=20
export MULTI_HARM_N_BASE_PAIRS=900

# 4-bit deployment condition (default on CUDA anyway; explicit for clarity)
export MULTI_HARM_QUANT=nf4

# Keep the v3 defaults otherwise (calib sizes 160, target FPR 5%, seed 42)
