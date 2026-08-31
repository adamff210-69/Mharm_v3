# Multi-HARM Pilot Run — Diagnosis (what is weak / wrong / the fix)

Reviewed against the Colab run you pasted (Mistral-7B-Instruct-v0.3, T4,
`/content/multi-harm`).

**Bottom line:** the pipeline runs end-to-end, but this specific run cannot
support the paper's claims. Two things dominate: (1) the run is far too small,
and (2) the **attention signal is effectively dead**, while a hidden-selection
artifact makes the "half-split attention AUROC" look perfect.

---

## 1. The run is not the intended scale (biggest problem)

Your env overrides in the notebook:

- `MULTI_HARM_N_CLEAN = 30`
- `MULTI_HARM_N_INJ_PER_CELL = 6`
- → total **150 samples** (30 clean + 120 injected).

After the 60/20/20 stratified split:

- train ≈ 18 clean + ~60 injected
- val ≈ 6 clean + ~24 injected (≈5 injects per type)
- test ≈ 6 clean + ~24 injected (≈5 injects per type)

What that means:

- Every "AUROC = 1.0" is computed on **≈10 rows** (6 clean + ~5 injects of
  that type). A probe with 4,096 features will separate 6 vs 5 samples with a
  hull curve; this is not evidence of robustness.
- Correlations (§4.3: 0.79–0.96) and attribution (0.35) are also on ≤30 val
  samples.
- The spec calls for 150–200 calibration samples and a 2,000-sample corpus;
  the demo runbook itself says **800** (400 clean + 4×5×20). You ran **150**.

**Fix:** re-run at the demo scale at minimum:
`MULTI_HARM_N_CLEAN=400` `MULTI_HARM_N_INJ_PER_CELL=20`, ideally the full spec.
Keep `MULTI_HARM_SYNTHETIC_CLEAN=false` and confirm `02` actually loaded
MS-MARCO. If it printed the "falling back to synthetic" warning, the clean
text is the short templated synth set and every below result is inflated.

---

## 2. The attention signal is not working (root cause of most weirdness)

`04_calibrate_hstar.py` output:

```
best head: layer 28, head 0 | AUROC 0.5000
top-10: 28x0..28x9  all 0.5000
```

Then after you patched `05` per-type masks:

```
4.8 ROW 1 — attention-only, shared calibration (test):
  topic 0.0833 | naive 0.1500 | fake 0.0333 | combined 0.0000
  overall AUROC 0.0667
```

So the shared attention ratio is **not just uninformative — it is at/below
chance**. That is why every specialist picks `alpha = 0` and why the "fused"
detector is really a hidden-state probe. The calibration correctly found that
blending attention hurts.

Notes on why this is not a patch problem:

- `04` reported 0.5 *before* any per-type patch; your `05` patch simply made
  the per-type row honest. It did not make the signal worse.
- The "attention half AUROC = 1.0000" in the half-split table is an artifact
  (see #3) and should not be read as evidence that attention works.
- In this repo `forward_signals` only caches the **last 4 layers** for
  attention masses. If the distraction effect lives in earlier layers, it is
  invisible by construction.

**Fix / investigate:**
1. Confirm the signal is genuinely absent at this model/scale: plot R for
   clean vs injected at the pooled head on a bigger set. If it is flat, either
   the "distraction effect" does not hold for Mistral-7B, or the
   query→injection-sum operationalization is wrong.
2. Try caching attention from more layers (`attn_last_k` 4 → 12–16) and/or
   middle layers before concluding attention is dead.
3. Check `m_qp/m_qi` on synthetic samples with very short passages: clean
   `W_i = tail_len = 48` (your clean `W_i` mean was 48.0), and if a passage
   tokenizes under ~48 tokens the "body" is tiny, so R loses meaning. Make
   sure clean passages are long enough or clamp the pseudo-injection region to
   `min(tail_len, len(passage)//2)`.
4. Do not present the current R as a working attention-based detector.

---

## 3. The "half-split" AUROC is a hidden in-sample artifact (code bug)

`multi_harm_common/calibrate.py::calibrate_specialist` does:

1. Build per-specialist calibration ids (~18 clean + ~15 injects).
2. `head_res = select_h_star(masses, widths, labels, ...)` — **selects the
   best of all heads using every one of those calibration samples.**
3. Then reports `auroc(labels, r_head)` **on the very same samples**.

That is selection AUC on the selection set. With ~33 samples and 128 candidate
heads, some head will be 1.0 by chance. This is exactly why the half-split
table says `attention=1.0` while pooled H* says `0.5` for every head. The v3
"half-split" number is therefore meaningless as reported.

**Fix:** evaluate the attention half on data that was not used to pick the
head (and not used for L*/probe/alpha either). Minimal real split:

- split the calibration ids into a "selection/fit" half and an "evaluation"
  half;
- run `select_h_star` on the selection/fit half only;
- compute `att_head` AUROC on the evaluation half.

The same principle already applies to `select_l_star` (it uses a 70/30
hold-out), so the code is inconsistent between the two halves.

---

## 4. `get_n_layers` is still wrong for Mistral / fresh clones

`multi_harm_common/model.py` at cell [21] of your notebook still shows:

```python
def get_n_layers(model) -> int:
    return int(model.config.n_layer)
```

Mistral (`MistralConfig`) exposes `num_hidden_layers`, **not** `n_layer`. On a
fresh clone this crashes `03`. (Your run likely survived because the extracted
copy you executed had a patch injected earlier, or because the patch did not
survive to the file you inspected later.)

I fixed this in the repo:

```python
def get_n_layers(model) -> int:
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    ...
```

Copy this updated `multi_harm_common/model.py` to Colab and re-run `03` on the
proper dataset.

---

## 5. Per-type AUROC was still one-class-broken in your run order

This is the bug we patched in `05`, `06`, `09` (and I also caught `10`). Your
transcript:

- `06` was patched (cells 43–45) → row 2 per-type went 0.5 → 1.0.
- `09` was patched (cells 49–50) → Table C went to 1.0.
- `05` was patched **last** (cell 54) — after `09` (cell 51) already read the
  old `row1_attn_shared.json`, so **Table B still shows the stale 0.5 for
  attn-shared**. After re-running `09`, that row should read ~0.0333–0.1500
  (which is the honest, though bad, signal).
- `10_figures_report.py` was never patched in the notebook; its per-type ROC
  curves share the same `ev["types"] == t` bug.

**Fix:** re-copy the four patched files from this repo and run in strict
order `04 → 05 → 06 → 07 → 08 → 09 → 10`, after clearing stale
`out/calib` and `out/experiments`.

---

## 6. "Fusion" is not happening — don't call it fusion

Every specialist reports `alpha = 0.00`. The fused score is:

```
S = alpha*z(R) + (1-alpha)*z(P) = z(P)
```

So all reported "fused" numbers are the **hidden-layer probe alone**. The
half-split table, Table B, and Table C all inherit this. In the review, say:
*"At pilot scale, calibration found zero weight for attention, so the detector
is currently residual/hidden-only; whether attention contributes is deferred
to the full run."*

---

## 7. Attribution result is genuinely poor — and expected

- Attribution accuracy **0.35** vs target **>0.75** (Table D).
- Specialist correlation §4.3 **0.79–0.96** vs target `<0.5` → NOT MET.
- §4.4: topic specialist is the only reliable one; naive/combined catch ~0.
- Because `alpha=0`, every specialist is a lightly-differently-calibrated
  hidden probe, so they are highly redundant and the argmax effectively
  labels nearly everything "topic".

These are real, explainable, and should be reported as **NOT MET** with the
mechanism, not as a wall of 1.0s.

---

## 8. Latency result is fine, but label it correctly

```
1 spec  2.2221 ms
5 specs 11.3466 ms
extra   9.1245 ms
```

This is **scoring-only** overhead (no forward pass). It supports the
"specialists are cheap on a shared forward pass" claim. It does **not** yet
measure the spec's total `< 0.5%` forward+scoring overhead — that requires
`09 --with-model` (which was not run).

---

## 9. Quick checklist to make the next run trustworthy

1. Set dataset to `N_CLEAN=400, N_INJ_PER_CELL=20` (or full 2000).
2. Confirm clean data is **real MS-MARCO**, not synthetic fallback.
3. Re-apply `get_n_layers` fix (done in this repo).
4. Re-apply the per-type AUROC slice fixes in `05/06/09/10` (done in this repo).
5. Fix the half-split in `calibrate_specialist` to use a real fit/eval split
   for the attention head.
6. Clear `out/calib` + `out/experiments`; run `04→05→06→07→08→09→10`.
7. Report the pilot as diagnostic only; lead with the negative findings and
   the latency result.

---

## Files changed in this repo (uncommitted)

- `multi_harm_common/model.py` — robust `get_n_layers`.
- `05_baseline_attn_tracker.py` — per-type AUROC includes clean.
- `06_calibrate_general.py` — per-type AUROC + hidden baseline include clean.
- `09_experiments_analysis.py` — Table B/C/§4.8 + top-1-vs-top-K include clean.
- `10_figures_report.py` — per-type ROC curves include clean.
