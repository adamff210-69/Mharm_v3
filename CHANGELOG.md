# Multi-HARM CHANGELOG

**v3.1.0** (this branch) vs **v3.0** = tag `base-v3`, commit `6cc0dd517001bead0d46b60dd818f7244deeff03`.

v3.0 is preserved unmodified on `main` / `base-v3`. Nothing here changes the
study design; every entry is a correctness, portability or honesty fix, and the
three that break the results outright are listed first. `git diff base-v3..HEAD`
shows the full delta.

---

## P0 — results were wrong regardless of the model

### 1. `metrics.auroc` returned a constant 0.5 for *every* call
`sklearn.metrics.roc_auc_score` has **no `pos_label` argument** (that belongs to
`roc_curve`/`precision_recall_curve`). v3.0 passed `pos_label=pos_label`, every
call raised `TypeError`, and the bare `except Exception: return 0.5` swallowed it.
Consequences, project-wide and silent:

* pooled `H*` and per-specialist `H*_s` head selection was an **arbitrary** pick
  (all 128 candidate AUROCs tied at 0.5);
* `L*` layer selection likewise (argmax over a tie);
* the fusion weight grid always returned its first candidate, **α = 0.0**, i.e.
  "fusion" silently degenerated to the residual half alone for every specialist;
* every AUROC in Tables A–E, §4.3, §4.8, the figures and both reports read
  `0.5000`, which looks like a flat result rather than a crash.

Fix (`multi_harm_common/metrics.py`): binary labels go straight to
`roc_auc_score`, non-binary ones are reduced one-vs-rest on `pos_label`,
mismatched/NaN inputs are handled, and errors are **no longer swallowed**.
`auroc_selftest()` was added and `04_calibrate_hstar.py` aborts if it fails, so a
broken metric can never again produce a whole experiment.

*Found by `run_offline_check.py`, not by reading the code — the base version had
this in a `try/except` that made it invisible to both the smoke test and a human.*

### 2. Every §4.8 per-type AUROC was computed on a single-class subset
`05`, `06` and `09` all filtered to `attack_type == t` and then took an AUROC —
excluding the `clean` rows (tagged `attack_type == "clean"`), so the subset was
all-positive and, with bug #1 fixed, would still have been exactly 0.5.

Fix: one definition, `detect.type_vs_clean_ids()` / `detect.per_type_auroc()` —
{injected of this type} ∪ {all clean rows of that split} — used by row 1, row 2,
the hidden-only baseline, Table B, Table C (both the halves and the top-1/top-K
ablation) and §4.8 row 3, so the three rows cannot drift apart. `spread` now uses
`detect.spread_of`, which skips degenerate (`None`) cells instead of crashing.

### 3. `get_n_layers` read `config.n_layer` — a GPT-2 attribute
`model.config.n_layer` / `model.config.n_head` exist on GPT-2/NeoX, **not** on
Llama/Mistral/Qwen (`num_hidden_layers` / `num_attention_heads`), which is the
default model (`meta-llama/Meta-Llama-3.1-8B-Instruct`). v3.0 crashed with
`AttributeError` in `01` and `03` on the real target — while `run_smoke_test.py`
(gpt2) passed. Fix: attribute-name lookup chains with an explicit error naming
the config class and the attributes tried, plus `get_n_heads()`.

## P1 — data integrity and cache correctness

* **Skipped samples desynchronized the cache.** `03` marked an unvalidatable
  sample as done in the checkpoint *without* writing signals for it, so every
  later stage died on a bare `KeyError` from deep inside the cache — and a re-run
  could not heal it. Now: `03` writes `out/validation/extraction_report.json`
  (ids + reasons + clipped list), aborts if more than
  `MULTI_HARM_MAX_SKIP_FRAC` (5%) of the dataset is unextractable, and every stage
  from `04` on routes the frame through `sigcache.usable_df()`, which excludes
  unextracted rows and says so.
* **Span widths were measured on untruncated tokens while masses were sliced from
  truncated ones.** `encode_sample` clipped `ids` to `max_seq_len` but left
  `passage_range`/`inj_range` beyond it, so `W_p`/`W_i` counted tokens the model
  never saw — re-introducing, for long samples, exactly the length confound the
  width-invariant ratio exists to remove (`combined` payloads are the longest, so
  the bias was type-dependent). `chat.encode_sample` now clips the spans, marks a
  sample invalid if a needed span disappears entirely, and computes the clean
  pseudo-tail from the *clipped* passage; `model.forward_signals` re-clips
  defensively, derives widths from the clipped ranges and reports `clipped`.
* **The documented chunked parquet writer was dead code — and broken.**
  `README`/`config.chunk_rows` claim chunked writes; `ParquetSinker` was never
  called, and `sigcache.save_row` appended row-by-row, re-reading and re-writing
  the whole file each time (once per sample, per file) — quadratic I/O over ~18
  files. `ParquetSinker` is now wired in per output file (and fixed: it passed
  `write_index=False`, a fastparquet kwarg, to the pyarrow engine, so it raised
  `TypeError` — one reason it stayed unused). It reads the existing row count on
  init, so a resumed run appends instead of truncating; `flush_sinks()` runs
  *before* each checkpoint save and `close_sinks()` before `finish()`, so a kill
  can only cost a re-do, never a claimed-but-missing row.
* **Cache schema versioning.** `data/signals/schema.json` records the extraction
  schema; `configure()`/`load_cache()` refuse a mismatch with the exact
  delete-and-re-run instruction, so a v3.0 cache cannot be read as v3.1.
* **Cache provenance, and a `03` that refuses a foreign cache.** `03` writes
  `data/signals/sigcache_meta.json` (model, quantization, row/skip/clipped counts,
  `max_seq_len`, dataset fingerprint) and aborts before touching the GPU if that
  record disagrees with the dataset in front of it, because the extraction
  checkpoint keys on sample id: regenerated ids would be marked done and would be
  silently absent from every stage. `--fresh` rebuilds. `11` archives the file.
* **Dataset fingerprint covers config, not just bytes.** `02` now records
  `version`, sizes, `max_seq_len`, `synthetic_clean`, `test_mode` alongside the
  sha256 and reports *why* a cache was invalidated. `n_base_pairs` 1,650 → 2,200,
  because the full run needs 2,000 distinct hosts (at 1,650 the same passage was
  reused as a clean row and an injected host) and the warning claimed "cycling
  with jitter", which was never implemented — it now says what actually happens.
  `base_idx` is kept in `dataset.parquet` (v3.0 dropped it), so a paired
  same-host-passage analysis remains possible.

## P2 — calibration statistics

* **α, the residual z-stats and θ were tuned on *in-sample* probe scores.**
  The deployed probe is a logistic regression on a 4,096-dim last-token vector
  fit on ~112 samples; its in-sample probabilities are far more separable and far
  more extreme than what it produces on new data. Choosing the fusion weight on
  them biases α toward whichever half overfits the calibration set and leaves the
  residual scale mismatched at inference. `signals.crossfit_probs()` now produces
  **out-of-fold** `P(injection)` (folds = `MULTI_HARM_PROBE_CROSSFIT_FOLDS`,
  default 2; set 1 for v3.0 behaviour) and `alpha`, `p_mu/p_sd` and `theta` are
  chosen from it. `auroc.hid_insample` / `hid_oof_gap` are logged so the
  overfitting gap is visible in `specialists.json`.
* **Standardizer leakage in `L*` selection and probe fitting.** `StandardScaler`
  was fit on all calibration rows, including the 30% "held-out" fraction used to
  score the layer, so the "no optimistic bias" claim did not hold. Both
  `select_l_star` and `fit_probe` now fit the scaler on the fit fraction only, and
  `fit_probe` returns `{fit_auroc, eval_auroc, n_fit, n_eval}`.
* **The 07 half-split table is labelled as calibration-set output**, with a
  `fused (held-out)` column next to it, because α is grid-searched to maximize
  that exact number (fused ≥ max(att, hid) holds on it by construction).
  `RESULTS.md` and `SUMMARY.md` carry the same caveat.
* **FPR-resolution and feasibility reporting.** `choose_theta` now returns
  `feasible`, `n_neg`, `min_nonzero_fpr` and `fpr_resolution_limited`; `07` prints
  how many clean samples the budget needs to be resolvable (a pilot run's θ is
  otherwise a single extreme order statistic, which is why its test FPR can be far
  from target while behaving exactly as specified), and `08` lists types whose
  threshold fell back to "never fire" — those can only be caught by
  `HARM_general`, capping both their detection rate and attribution accuracy.

## P2 — reporting, portability, latency

* `demo_run.sh` passed stages as `"09 script --flag"` into a `run()` that read
  only `$1`/`$2`, so **`--with-model` never reached python** and the demo silently
  skipped the forward-included latency number its own runbook promises. Stages are
  now full command lines, executed as argv arrays; `--calib-sweep` is included and
  `11` is in the default list.
* `10_figures_report.py` loaded `sec43_pearson_val.json`, which nothing wrote
  (`09` wrote only `.csv`), and then never used the variable. `09` now writes the
  JSON, and `10` renders §4.3 and §4.4 in `RESULTS.md`.
* `crit(tableA['attr_accuracy'] or 0 > 0.75)` parsed as `acc or (0 > 0.75)`, so
  the attribution criterion printed **MET for any nonzero accuracy**. Parenthesized.
* `figures.fig_quant_compare` called `axvline(str(L))` against a categorical bar
  axis (version-dependent `TypeError`, and unreachable from the CPU smoke test
  because `03` skips §2.3 off-GPU). Bars now sit on numeric positions with labels
  via `set_xticks`, and the marker is only drawn if that layer is present.
* `env.py` makes torch optional (`require_torch()` names the two stages that need
  it), so `02` and `04`–`11` — including the reproducibility zip — run on a bare
  numpy/pandas install. `model.load_model` places bitsandbytes models via
  `device_map` instead of a post-hoc `.to("cuda")` on a quantized module.
* `config.py` gained `version` / `base_spec` (stamped into the dataset
  fingerprint, `REPRODUCIBILITY.md`, `RESULTS.md`), `max_skip_frac`,
  `probe_crossfit_folds`. Attack-type lists in `calibrate.py`, `detect.py`, `05`,
  `06`, `07` now come from `config.ATTACK_TYPES` (six duplicated literals could
  drift from the env-overridable taxonomy).

## New: what was added to catch this class of bug

* **`run_offline_check.py`** — end-to-end integration check for stages `02` and
  `04`–`11` with **no model, no GPU, no download**: it builds the real dataset,
  fabricates a signal cache with *planted* structure (each attack type visible only
  on its own head, width independent of intensity, weak-but-real residual signal)
  through the production writer, then asserts the pipeline behaves correctly on
  it: no per-type AUROC is the degenerate 0.5; each specialist selects its own
  planted head; row 3 beats row 2 by more than the 3-pt criterion and shrinks the
  per-type spread; `corr(R, W_i)` is far smaller for the invariant ratio than for
  the naive sum ratio on identical data; the trailing partial parquet chunk
  survives; a stale schema and a `pos_label`-style metric failure are refused.
  This is what found P0 #1. `python run_offline_check.py [--keep]`.
* **`09 --calib-sweep`** — re-calibrates every specialist at
  `calib_n ∈ {40, 80, 160}` (cached signals only) and reports
  `fig_calib_sweep.png` + `calib_size_sweep.{csv,json}`, so "does the
  specialization gain survive a small calibration budget" is a measurement
  instead of limitation #4.
* **`09` §4.4 output** — `sec44_cross_val.json` with `own_type_fraction`, the
  attribution number that is *not* biased by each specialist being calibrated on
  its own type, surfaced in both reports next to Table D.
* **Robustness claims moved into the pipeline.** `09` now reports
  `out/validation/width_invariance.json` (`corr(R, W_i)` for the deployed
  per-token ratio *and* for v3.0's raw sum-ratio on identical data — on the
  offline fixture: −0.086 vs +0.914) and
  `out/experiments/sec43_head_by_type.json`/`.csv` (mean mass of every layer×head
  by attack type, which type each head prefers, and whether calibration's head is
  that argmax). `10` tables both in `RESULTS.md`, and prints an explicit
  **FPR-resolution-limited** caveat for any specialist whose `theta` had to fall
  back to a single order statistic, instead of letting an off-target FPR read as
  a tuning miss.
* `run_smoke_test.py` now runs `09` with `--calib-sweep`, verifies ~20 expected
  artifacts exist, and fails if the §4.8 table is uniformly 0.5.
* **`run_offline_check.py --stage-only NN`** re-runs the fixture through one
  stage; the check now also requires the artifacts above to exist *and carry the
  right fields*, asserts `repro_manifest.json` hashes artifacts and records the
  git commit and cache provenance, asserts the zip contains both, and unit-tests
  `chat.py`'s span mapping / clipping and `choose_theta`'s resolution flag
  (`1/n_neg`) directly, since those depend on data sizes rather than wiring.

## Removed (dead in v3.0)

`io_utils.read_jsonl` / `append_jsonl` / `Checkpoint.mark_done_batch`,
`metrics.spread` (unused; §4.8 used its own inline `max-min`),
`dataset._seed_hash`, `Config.effective_n`, `Config.forward_dtype_note`,
a `drop(columns=["base_idx"])`, and the wrong return annotation on
`metrics.confusion`.

## Not changed (deliberate)

The attack taxonomy, the 4×5×50 design, the pooled-then-specialized calibration
order, `per_spec` meta rule, span-width-invariant `R`, `tail_len` pseudo-injection
for clean samples, and all v3 success criteria are untouched — they are the study
design, not bugs. Known limitations that remain, and should be stated in the
paper rather than coded around: 1,000 injected samples come from **20 payload
strings** (5 goals × 4 wrappers, verbatim), so a probe can key on template lexicon
instead of attack semantics, and `fake`/`combined` share a `[SYSTEM NOTICE]`
delimiter; injections are always appended at the end of the passage, so the
residual half also sees a position cue; attribution-by-argmax remains structurally
favourable to the true type (§4.4 is the honest companion number).
