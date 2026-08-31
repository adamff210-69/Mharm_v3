# Multi-HARM Framework Audit — what I tested, what was wrong, what I fixed

Date: 2026-08-31 (sandbox, branch `arena/01a05384-mharm-v3`)

## 1. What I tested

| Area | How | Result |
|---|---|---|
| `02_build_dataset.py` | Ran with synthetic clean, test mode | PASS. 120 rows, clean/injected crosstab correct. Now emits a loud synthetic-source warning. |
| `03_extract_signals.py` | Attempted (gpt2 smoke) | **BLOCKED by sandbox** — `huggingface.co` is unreachable, so the 8B/2B model can't download here. The code path itself was audited by inspection. |
| `04_calibrate_hstar.py` | Ran on a synthetic signal cache | PASS after fixes |
| `05_baseline_attn_tracker.py` | Ran on synthetic cache | PASS |
| `06_calibrate_general.py` | Ran on synthetic cache | PASS |
| `07_calibrate_specialists.py` | Ran on synthetic cache | PASS |
| `08_meta_decision.py` | Ran on synthetic cache | PASS |
| `09_experiments_analysis.py` | Ran on synthetic cache | PASS (all tables, §4.3/4.4/4.8, latency, SUMMARY.md) |
| `10_figures_report.py` | Ran on synthetic cache | PASS (7 figures + RESULTS.md) |
| `11_reproducibility.py` | Ran | PASS (zip now actually contains all scripts + config) |

Note: the synthetic-cache numbers are meaningless scientifically — they only
prove the code paths run and the metrics/tables no longer silently collapse.

## 2. Real bugs found and fixed

### 2.1 `auroc()` silently returned 0.5 for every AUROC (highest-impact)
`multi_harm_common/metrics.py` called
`roc_auc_score(y, s, pos_label=1)` with a runtime `pos_label` variable.
Newer scikit-learn (>=1.6) rejects that, the `except Exception` swallowed the
error, and every call returned `0.5`. This is the same root cause your Colab run
had to patch by hand.

**Fix:** call `roc_auc_score(y, s)`; keep `pos_label=1` as API compatibility;
raise visibly on any other label value instead of returning 0.5 silently.

### 2.2 Model config names were hard-coded
- `get_n_layers()` used `model.config.n_layer` → crashed on Mistral (`num_hidden_layers`).
- `01_setup_and_validate.py` used `model.config.n_head` → crashes on Mistral (`num_attention_heads`).

**Fix:** added `get_n_heads()` and made `get_n_layers()` read any of
`num_hidden_layers / n_layer / num_layers / n_layers`, and heads read any of
`num_attention_heads / num_heads / n_head / n_heads`.

### 2.3 In-sample head selection made half-split AUROCs meaningless
`calibrate_pooled_hstar()` and `calibrate_specialist()` selected the best
attention head on the **same** calibration rows they then evaluated. With ~30
rows and 128 candidate heads, that guarantees near-1.0 AUROC even with no
signal — which is exactly why your pilot showed `attention=1.0` while pooled
H* showed 0.5.

**Fix:** stratified fit/eval split. Head selection is done on a fit half; the
reported `att_head` / `att_shared` AUROC is measured on the held-out half.

### 2.4 Probe/L* hidden AUROC also had in-sample + stratification bugs
- `select_l_star` / `fit_probe` used a **plain shuffle**, which on small
  calibration sets can put the entire held-out split in one class → AUROC 0.5
  even when the signal is perfect.
- `n_fit = max(10, ...)` could exceed a small calibration set, so "held-out"
  fell back onto training rows.

**Fix:** `_stratified_split()` keeps both classes in both halves (when
possible) and the fit size is bounded so at least one held-out row remains.

### 2.5 `meta_decision` threshold used `or` instead of `is not None`
`general_score > (general_theta or 1e9)` treats a valid `theta=0.0` as "no
threshold". Fixed to `general_theta if general_theta is not None else 1e9`.

### 2.6 `10_figures_report.py` had a boolean-precedence bug
`tableA['attr_accuracy'] or 0 > 0.75` parses as
`tableA['attr_accuracy'] or (0 > 0.75)`, printing the real accuracy value as
the "Status" instead of `MET/NOT MET`. Fixed.

### 2.7 `11_reproducibility.py` produced an incomplete zip
The staging loop skipped every top-level `*.py` (and `config.py`) because they
were already in `codefiles`, so the repro zip had only `multi_harm_common` +
README/requirements — not the pipeline scripts. Fixed; verified zip now lists
`01…11`, `config.py`, `multi_harm_common/*`, `run_smoke_test.py`.

### 2.8 Silent synthetic clean-data fallback
When MS-MARCO isn't reachable, `02` used synthetic clean text without an
explicit warning, making it easy to present synthetic numbers as real data.

**Fix:** dataset now carries `source_id`; `02` writes `data/clean_source.json`
and prints a loud warning if the source is synthetic.

## 3. Remaining risks (not fixed, should be stated honestly)

1. **Could not run 03 here** because the sandbox has no HF access, so the
   model/extraction path (token ranges, attention tensors, `hidden_states`) is
   only code-reviewed, not executed. It must be re-run on Colab/T4.
2. **Pilot is tiny.** Your Colab run used `n_clean=30`, `n_inj_per_cell=6`
   (~150 samples). Any AUROC / correlation / attribution number from it is
   diagnostic only. The review line should be:
   *"The pipeline runs end-to-end at pilot scale; these are not final
   results."*
3. **`probe_fit_frac` / alpha.** Now calibrated on held-out halves, but the
   detection threshold `theta` is still tuned on the full calibration set
   (FPR-budget). This is a minor but real leaking path; document it.
4. **Attention signal appears weak/absent at pilot scale** (H* ≈ 0.5, row-1
   AUROC ≤0.15 after the per-type fix). That is a legitimate finding, not a
   bug; don't show it as "attention works".

## 4. Re-run order (after copying then patched files to Colab)

```bash
# env (demo size recommended)
export MULTI_HARM_N_CLEAN=400
export MULTI_HARM_N_INJ_PER_CELL=20
export MULTI_HARM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3

# clear stale outputs
rm -rf "$MULTI_HARM_OUT_DIR/calib" "$MULTI_HARM_OUT_DIR/meta" \
       "$MULTI_HARM_OUT_DIR/experiments" "$MULTI_HARM_OUT_DIR/figures" \
       "$MULTI_HARM_OUT_DIR/report" "$MULTI_HARM_OUT_DIR/repro"

python 02_build_dataset.py --force   # confirm NOT synthetic
python 03_extract_signals.py          # §2.0 gate + extraction
python 04_calibrate_hstar.py
python 05_baseline_attn_tracker.py
python 06_calibrate_general.py
python 07_calibrate_specialists.py
python 08_meta_decision.py
python 09_experiments_analysis.py --with-model
python 10_figures_report.py
python 11_reproducibility.py
```

## 5. Files changed in this audit (uncommitted)

- `multi_harm_common/metrics.py` — real AUROC (no silent 0.5)
- `multi_harm_common/model.py` — portable layer/head count
- `multi_harm_common/signals.py` — stratified holdout for L*/probe
- `multi_harm_common/calibrate.py` — honest H*/half-split + alpha
- `multi_harm_common/detect.py` — meta threshold fix
- `multi_harm_common/dataset.py` + `02_build_dataset.py` — explicit clean source
- `01_setup_and_validate.py` — uses `get_n_heads()`
- `05/06/09/10` — per-type AUROC slices include clean rows
- `10_figures_report.py` — attribution status-line bug
- `11_reproducibility.py` — complete repro zip
