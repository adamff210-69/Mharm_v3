# Multi-HARM

**Attack-type-calibrated fusion of attention and residual-stream signals for
prompt injection detection** — implementation of the v3 master spec
(`HARM_Master_Implementation_Prompt_v3.md`).

> **This tree is v3.1** (`config.version = 3.1.0`): the v3 study design with the
> correctness fixes collected in [`CHANGELOG.md`](CHANGELOG.md). Three of them
> change the numbers — `metrics.auroc` returned a constant 0.5 on every call, the
> §4.8 per-type AUROCs were computed on single-class subsets, and
> `model.config.n_layer` does not exist on Llama. The **unmodified v3.0 baseline is
> tag `base-v3`** (commit `6cc0dd5`, on `main`); compare with
> `git diff base-v3..HEAD`. Nothing here alters the hypothesis, the taxonomy, the
> calibration order or the success criteria.

One shared forward pass per sample → attention-ratio signal (last 4 layers,
per head) + last-token hidden states (candidate layers `[N/4, 3N/4]`) →
**5 independently calibrated specialists** (one per attack type + general
fallback) scored by cheap dot products → **meta-decision** →
INJECTED/SAFE + **attack-type attribution** (argmax over specialist scores,
no trained classification head).

The paper's spine is the **§4.8 table**: attention-only shared calibration
(replicating Attention Tracker) vs fused-shared vs fused-specialized — a
direct test of whether attack-type specialization helps *beyond* the
attention signal's reported cross-attack generalization.

---

## Layout

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters; every field overridable via `MULTI_HARM_<FIELD>` env vars |
| `multi_harm_common/` | Importable core: `env`, `io_utils`, `chat` (prompt + token ranges + §2.0 validation), `model`, `signals`, `sigcache`, `calibrate`, `detect`, `dataset`, `metrics`, `figures` |
| `01_setup_and_validate.py` | Env report, model load, forward-pass shape smoke test |
| `02_build_dataset.py` | 1,000 clean MS-MARCO pairs + 1,000 injected (4 types × 5 goals × 50), 60/20/20 split stratified by (type, goal) |
| `03_extract_signals.py` | **§2.0 validation gate**, resumable chunked extraction; `--quant-compare` runs **§2.3** (fp-ref vs 4-bit) |
| `04_calibrate_hstar.py` | Pooled H* (per-head AUROC, 150–200 samples, §2.1) |
| `05_baseline_attn_tracker.py` | **§4.8 row 1** — attention-only, shared calibration |
| `06_calibrate_general.py` | **§4.8 row 2** — HARM_general (fused, shared) + PIShield-style hidden-only baseline |
| `07_calibrate_specialists.py` | 4 type specialists: L* → probe → h_base → α → θ, **half-split AUROC logged per specialist** (Phase 3 addition) |
| `08_meta_decision.py` | Meta layer evaluated on val; FPR criterion check; `global_max` alternative threshold |
| `09_experiments_analysis.py` | Tables A–E, §4.3 (pairwise ρ), §4.4 (cross-specialist), §4.8 spine table, latency, `SUMMARY.md` with the success-criteria check. `--calib-sweep` re-calibrates every specialist at calib_n 40/80/160 |
| `run_offline_check.py` | **No-GPU integration check** for `02` + `04`–`11`: plants a synthetic signal cache and asserts the pipeline recovers it (see below) |
| `run_tiny_model_check.py` | **No-network, no-GPU check of the model stages** (`01`, `03`) and then `04`–`11`, against a locally built random-init Llama-architecture model |
| `demo_kaggle.ipynb` / `KAGGLE.md` | Kaggle kernel staging (settings, torch/bitsandbytes caveat, resume recipe) |
| `CHANGELOG.md` | Every v3.0 → v3.1 change, with the reason |
| `10_figures_report.py` | All figures + `out/report/RESULTS.md` (§4.8 lead table, **§4.9 BAGEL/Luna-2 differentiation table** with citations, novelty claim, limitations) |
| `11_reproducibility.py` | `REPRODUCIBILITY.md` + zip (code, config, hashes, results) |
| `run_smoke_test.py` | End-to-end correctness harness: **gpt2 on CPU, synthetic data** (no GPU needed) |

## Execution order (per v3)

```bash
pip install -r requirements.txt

python run_offline_check.py                  # 0. no GPU needed: proves 02/04-11
python 01_setup_and_validate.py              # 1. env + model
python 02_build_dataset.py                   # 2. dataset
python 03_extract_signals.py --quant-compare # 4. §2.3 fp-ref vs 4-bit (BEFORE trusting 4-bit)
python 03_extract_signals.py                 # 3. §2.0 gate + full extraction
python 04_calibrate_hstar.py                 # 5. H* (§2.1)
python 05_baseline_attn_tracker.py           # 6. §4.8 row 1 — reference point first
python 06_calibrate_general.py               # 7. §4.8 row 2
python 07_calibrate_specialists.py           # 8. specialists + half-split logging
python 08_meta_decision.py                   # 9. meta layer (val)
python 09_experiments_analysis.py --with-model --calib-sweep   # 10. tables
python 10_figures_report.py                  # 11-12. §4.9 table, figures
python 11_reproducibility.py                 # 13-14. package
```

Everything from `04` onward runs on **cached signals only** (no model
forward passes — and no torch either; `02`/`04`–`11` import cleanly without it) — re-running a calibration variant after a design tweak
takes seconds. The expensive step is `03` (~2–5 h on a free T4 for 2,000
samples, resumable: a restarted Colab session picks up where it stopped via
`out/progress/extract.json`).

**Stale-cache protection** (two layers, because a mixed cache is not a crash —
it is a silently wrong number):
`02` writes a fingerprint of the dataset and, when a later run changes sizes or
mode (smoke test → full run), clears the signal cache and extraction checkpoint
so old rows cannot mix into a new run. `03` then writes
`data/signals/sigcache_meta.json` next to the cache — model, quantization, row
count, skip count, clipped count, `max_seq_len` and the dataset fingerprint it
was built from — and refuses to start if that record disagrees with the dataset
in front of it (re-run with `--fresh` to rebuild). `11` copies that file into the
reproducibility zip and hashes every artifact into `repro_manifest.json`.

### Colab quick start

```python
!pip install -q -r requirements.txt        # in the first cell
!git clone <your-repo> multi-harm && %cd multi-harm
# then run each numbered script as !python 0X_....py (each in its own cell)
```

### Kaggle / local GPU

Plain `python` — no Colab-specific code. Move the project folder, `pip
install -r requirements.txt`, and run the same order. Useful env-var
overrides:

| Var | Example | Effect |
|---|---|---|
| `MULTI_HARM_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.3` | switch base model |
| `MULTI_HARM_QUANT` | `nf4` / `fp16` / `int8` / `fp32` | force dtype (default `auto`: nf4 on CUDA) |
| `MULTI_HARM_MAX_SEQ_LEN` | `1536` | longer contexts |
| `MULTI_HARM_DATA_DIR` / `MULTI_HARM_OUT_DIR` | `/kaggle/input/...` | relocate data (read-only input dirs work for `data`) |
| `MULTI_HARM_UNSEEN_TYPE` | `fake` | change the §4.5 held-out type |
| `MULTI_HARM_TARGET_FPR` | `0.05` | overall FPR target (per-specialist budget is `target_fpr/4`) |
| `MULTI_HARM_CALIB_PER_SPECIALIST` | `160` | clean samples per specialist. `theta` needs at least `1/(target_fpr/4)` ≈ 80 negatives before the budget is resolvable at all; below that `07`/`10` print an FPR-resolution note instead of pretending the target was met |
| `MULTI_HARM_PROBE_CROSSFIT_FOLDS` | `1` | out-of-fold folds used to pick α / z-stats / θ (`1` = v3.0 in-sample behaviour) |
| `MULTI_HARM_MAX_SKIP_FRAC` | `0.05` | share of the dataset `03` may fail to extract before aborting |
| `MULTI_HARM_CHUNK_ROWS` | `500` | rows buffered per parquet rewrite in `03` |

## Correctness harnesses (no GPU)

Two checks, in this order. Both must pass before you spend T4 hours.

```bash
python run_offline_check.py                  # 1. stages 02 + 04-11, no model at all
python run_offline_check.py --stage-only 09  #    one stage, for iterating
python run_tiny_model_check.py --full        # 2. stages 01-11, real forward pass,
                                             #    no download (~1 min, CPU)
python run_smoke_test.py                     # 3. the same with gpt2 (~500 MB dl)
```

`run_tiny_model_check.py` is the one to run on a machine with torch but no
network, or before a Colab/Kaggle session: it builds a random-init 6-layer ×
4-head `LlamaForCausalLM` plus a byte-level BPE tokenizer trained on the dataset,
saves them to a temp dir and points `MULTI_HARM_MODEL_NAME` at it, so `01`'s
shape gate, `03`'s §2.0 gate, the chunked cache writer, checkpointing, the
provenance record **and the staleness guard** all actually execute — including
`04`–`11` on genuine attention matrices. It asserts the cache holds
`attn_last_k × n_heads` mass rows per sample, that `m_qi <= m_qp` for every head
(which is what a span/clipping mismatch breaks first), and that a cache from a
different dataset is refused rather than silently reused. It exits 0 with a note
when torch is unavailable, so it is safe to leave in any environment.

`run_offline_check.py` is the one that earns its keep: it builds the real dataset,
writes a **fabricated signal cache with planted structure** through the production
parquet writer, and asserts the analysis half of the pipeline recovers it — no
per-type AUROC may be the degenerate 0.5, each specialist must select its own
planted head, row 3 must beat row 2 by more than the 3-point criterion *and*
shrink the per-type spread, and `corr(R, W_i)` must be far smaller for the
width-invariant ratio than for the naive sum ratio on identical data; the
cache-provenance and `usable_df` ghost-row guards have to fire. It also runs
torch-free unit checks on the parts that need no model at all — the char-span →
token-span mapping and the `max_seq_len` clipping in `chat.py`, which is where a
mismatch would be invisible in the numbers. Needs only
numpy/pandas/scikit-learn/pyarrow/matplotlib, because `env.py` makes torch
optional for every stage except `01`/`03`. It is also what caught the
`roc_auc_score(pos_label=...)` bug that made every AUROC in v3.0 a constant 0.5.

Runs the **entire pipeline** with `gpt2` on CPU and deterministic synthetic
clean pairs (20 clean + 4 types × 5 goals × 5 injected, short sequences).
Add `--keep` to keep the resulting `data/`/`out/` for inspection (by
default the smoke run wipes them so a real run starts clean).
This verifies token-range mapping, the §2.0 gate, extraction/checkpointing,
all calibrations, every table, the meta layer, figures and the zip.
Scientific content of this run is meaningless — it is a wiring test. Run it
before burning T4 hours.

## Design decisions (documented assumptions)

The v2 scaffold was not available; v3 fixes the *structure* (specialists,
calibration order, experiments) but not every formula. Where v3 left a
choice, this is the one implemented — change in one place, noted here:

1. **Attention signal R.** `R(l,h) = (m_qi/W_i) / ((m_qp − m_qi)/(W_p − W_i) + ε)`
   — the **per-token** attention intensity on the injection region relative to the
   per-token intensity on the passage **body** (the injection *excluded*: it is a
   subset of the passage, so it would dilute its own denominator). `m_qi`, `m_qp`
   are sums over
   (query rows × span columns) and are divided by the span widths, which
   makes R **invariant to injection length and query length**. This matters:
   `combined` payloads are several times longer than `naive` ones, so a raw
   sum ratio would have made `combined` mechanically easier to detect —
   a pure span-length artifact, not the hypothesis under test. Widths are measured
   on spans already clipped to `max_seq_len`, so they describe the tokens the model
   actually saw. For **clean**
   samples the "injection span" is the last `tail_len` (48) tokens of the
   passage (a pseudo-injection region). This operationalizes the
   "distraction effect" (Attention Tracker) for the passage/query layout.
   The per-head, per-layer *masses* and span widths are cached; the
   per-column-mean normalization happens at cache load, and any other ratio
   variant can be re-derived without another forward pass. `09` writes a
   span-width audit (`out/experiments/span_width.json`) reporting per-type
   payload lengths and the residual R↔width correlation for the paper.
2. **§2.1 deviation (deliberate).** v3 suggests caching raw
   `outputs.attentions[-4:]` (~268 MB/sample) to re-run head selection at a
   different top-K. The per-head masses (~1 KB/sample) determine every ratio
   we can form, so head selection at any top-K re-runs from cache; raw
   attention is not cached. If you change the *definition* of R (new spans),
   re-extract.
3. **Residual signal P(injection).** Linear probe (logistic regression on
   standardized last-token hidden state at L*). L* chosen by per-layer probe
   AUROC with a 70/30 fit/eval split inside the calibration set (no
   optimistic bias). `h_base` = mean clean embedding at L*; cosine distance
   to `h_base` is stored as a diagnostic (not the primary residual signal).
4. **Fusion.** `S = α·z(R) + (1−α)·z(P)`, per-specialist α from a 0–1 grid
   (step 0.05) maximizing calibration AUROC; z-stats from the same
   calibration set (standard, documented mild-leakage convention; the probe
   itself is evaluated on its held-out fraction).
5. **Thresholds.** Per-specialist θ chosen at the FPR budget
   `target_fpr/4` (union bound keeps the meta OR-rule near the 5% overall
   FPR target); general gets the full `target_fpr`. The `global_max` meta
   alternative (single θ on the max score, tuned on val) is reported
   alongside as an ablation.
6. **Meta rule (default `per_spec`).** INJECTED if any type specialist's
   fused score exceeds its own θ; attribution = argmax over fired specialists
   (ties → highest score); HARM_general fires only as fallback
   (attribution "general"). SAFE otherwise.
7. **Attack types / goals.** `naive` (raw instruction), `fake` (fake system
   notice), `topic` (on-topic prose framing), `combined` (fake + raw +
   topic). Five instruction-level goals (answer override, info leak, format
   hijack, role override, persuasion) — see `GOALS`/`ATTACK_WRAPPERS` in
   `config.py`. If you have the v2 set, replace these and rebuild the
   dataset.
8. **§2.3 on a T4.** 8B in fp16 (~16 GB) does not fit a 16 GB T4 with
   activations, so the reference chain is `fp16 → bf16 → int8` (first
   loadable wins — on a T4 that is usually **int8**, a legitimate
   near-full-precision reference). The correlation analysis is identical; the
   report states which reference was actually used.
4a. **Cross-fitted fusion statistics (v3.1).** `alpha`, the residual z-stats and
   `theta` are chosen on **out-of-fold** `P(injection)` (`signals.crossfit_probs`,
   `probe_crossfit_folds=2`), not on the deployed probe's in-sample probabilities.
   A 4,096-dim logistic probe fit on ~112 samples is far more separable and far
   more extreme in-sample than at inference, which biases α toward whichever half
   overfits the calibration set and mismatches the residual scale afterwards. Set
   `MULTI_HARM_PROBE_CROSSFIT_FOLDS=1` for the v3.0 behaviour; the deployed probe is
   still fit on the full calibration set (more data), only the *statistics* are
   cross-fitted. `auroc.hid_insample` and `hid_oof_gap` record the overfit gap.
9. **Clean pairs.** MS-MARCO golden query-passage pairs (HF `ms_marco`,
   `passage_ranked` dev); fallback to `data/clean_pairs.csv`
   (`passage,query` columns) or `--synthetic` (offline test only).
10. **Per-type AUROC = type vs clean (v3.1).** Every per-attack-type AUROC in the
    §4.8 table, Table B/C and the figures is computed on
    {injected of that type} ∪ {all clean rows of that split} via
    `detect.type_vs_clean_ids`. AUROC is a two-class statistic; the v3.0 subsets
    were all-positive and therefore pinned at 0.5. Rows 1/2/3 share the helper so
    they stay comparable by construction.
11. **Skipped samples are excluded, not fatal (v3.1).** If `03` cannot validate an
    encoding it records the id in `out/validation/extraction_report.json` and moves
    on; every later stage restricts itself to extracted rows (`sigcache.usable_df`)
    and prints what it dropped. Above `max_skip_frac` (5%) of the dataset
    unextractable, `03` aborts instead of calibrating on a silently shrunken set.
12. **Model metadata (v3.1).** Layer/head counts come from a lookup chain
    (`num_hidden_layers|n_layer|num_layers`, `num_attention_heads|n_head|num_heads`)
    so Llama/Mistral/Qwen work, not only GPT-2.
13. **Cache schema marker (v3.1).** `data/signals/schema.json` pins the extraction
    schema (now 2: clipped widths); `03` and every reader refuse a mismatch with the
    exact delete-and-rerun command, so a v3.0 cache cannot be read as v3.1.
14. **Robustness claims are measured, in the pipeline, not only in the tests
    (v3.1).** `09` writes `out/validation/width_invariance.json` —
    `corr(R, W_i)` for the deployed per-token ratio beside the same correlation for
    v3.0's raw sum-ratio, on the identical data — and
    `out/experiments/sec43_head_by_type.json`, the mean mass of *every* layer×head
    by attack type with which type each head prefers, which is what §4.3's
    "no single head fires for all four types" actually asserts. `10` reproduces both
    in `RESULTS.md` and adds an FPR-resolution caveat whenever `07` had to fall back
    to a single order statistic.

## Success criteria (where each is checked)

| Metric | Target | Where |
|---|---|---|
| Multi-HARM mean ASR | < 8% | `09` Table A → `SUMMARY.md` |
| **§4.8 specialized-vs-shared fused gain** | **> 3 AUROC pts** | `09` → `table_48.json` / `SUMMARY.md` (primary claim) |
| Mean FPR | < 5% | `08` (val) + `09` (test) |
| Attribution accuracy | > 75% | `09` Table D |
| Pairwise specialist-score ρ | < 0.5 per pair | `09` §4.3 |
| Latency overhead vs single specialist | < 0.5% | `09 --with-model` (forward) / numpy-only (scoring) |
| Unseen-attack ASR | within 10 pts of seen | `09` Table E |
| fp-ref vs 4-bit correlation at L* | > 0.9 | `03 --quant-compare`, checked in `10` |
| (guard) AUROC metric is computing at all | perfect=1.0 / inverted=0.0 | `04` self-test, aborts on failure; also asserted by `run_offline_check.py` |
| (context) specialization vs calibration size | monotone, not cliff-edged | `09 --calib-sweep` → `calib_size_sweep.csv`, `fig_calib_sweep.png` |
| (context) honest attribution | §4.4 own-type fraction, not Table D argmax | `09` → `sec44_cross_val.json` |
| (robustness) `R` uncorrelated with payload width | `|corr|` below v3.0's sum-ratio on the same data | `09` → `out/validation/width_invariance.json` |
| (robustness) no head wins all four types | `one_head_wins_every_type = False` | `09` → `sec43_head_by_type.json`, tabled by `10` |
| (guard) signal cache provenance recorded | file exists, hashed into the zip | `03` → `data/signals/sigcache_meta.json`, `11` → `repro_manifest.json` |

## Timeline (free T4, per v3)

~4–6 days across sessions: `03` dominates (~3 h extraction + 30 min
quant-compare; `02`/`04`–`11` are minutes on CPU); `01` model download ~5 GB; everything else is minutes.
Each script is idempotent — kill and restart anywhere.
