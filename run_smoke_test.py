#!/usr/bin/env python3
"""End-to-end smoke test with a TINY model on CPU (gpt2, synthetic data).

Validates the entire pipeline code path (tokenization, token-range
validation gate, signal extraction, all calibrations, meta decision,
every table, figures, reproducibility) without a GPU or an 8B download.
Results from this run are meaningless scientifically — it is a
correctness harness only.

Run:  python run_smoke_test.py [--keep]
"""
import os
import shutil
import subprocess
import sys

STAGES = [
    ("02_build_dataset.py", []),
    ("03_extract_signals.py", []),
    ("04_calibrate_hstar.py", []),
    ("05_baseline_attn_tracker.py", []),
    ("06_calibrate_general.py", []),
    ("07_calibrate_specialists.py", []),
    ("08_meta_decision.py", []),
    ("09_experiments_analysis.py", []),
    ("10_figures_report.py", []),
    ("11_reproducibility.py", []),
]


def main():
    keep = "--keep" in sys.argv
    if not keep:
        for d in ("data", "out"):
            if os.path.exists(d):
                shutil.rmtree(d)

    env = dict(os.environ,
               MULTI_HARM_MODEL_NAME="gpt2",
               MULTI_HARM_TEST_MODE="true",
               MULTI_HARM_SYNTHETIC_CLEAN="true",
               MULTI_HARM_MAX_SEQ_LEN="384")
    results = []
    for script, extra in STAGES:
        print(f"\n{'=' * 70}\nSMOKE: {script}\n{'=' * 70}")
        r = subprocess.run([sys.executable, script] + extra,
                           env=env, capture_output=True, text=True, timeout=1800)
        ok = r.returncode == 0
        tail = (r.stdout or "")[-1500:]
        err = (r.stderr or "")[-1500:]
        results.append((script, ok))
        print(tail)
        if not ok:
            print("STDERR:\n" + err)
            break

    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    for s, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {s}")
    failed = [s for s, ok in results if not ok]
    print(f"\n{'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
