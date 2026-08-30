# Multi-HARM

**Attack-type-calibrated fusion of attention and residual-stream signals for
prompt injection detection** — implementation of the v3 master spec
(`HARM_Master_Implementation_Prompt_v3.md`).

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
| `09_experiments_analysis.py` | Tables A–E, §4.3 (pairwise ρ), §4.4 (cross-specialist), §4.8 spine table, latency, `SUMMARY.md` with the success-criteria check |
| `10_figures_report.py` | All figures + `out/report/RESULTS.md` (§4.8 lead table, **§4.9 BAGEL/Luna-2 differentiation table** with citations, novelty claim, limitations) |
| `11_reproducibility.py` | `REPRODUCIBILITY.md` + zip (code, config, hashes, results) |
| `run_smoke_test.py` | End-to-end correctness harness: **gpt2 on CPU, synthetic data** (no GPU needed) |

## Execution order (per v3)

```bash
pip install -r requirements.txt

python 01_setup_and_validate.py              # 1. env + model
python 02_build_dataset.py                   # 2. dataset
python 03_extract_signals.py --quant-compare # 4. §2.3 fp-ref vs 4-bit (BEFORE trusting 4-bit)
python 03_extract_signals.py                 # 3. §2.0 gate + full extraction
python 04_calibrate_hstar.py                 # 5. H* (§2.1)
python 05_baseline_attn_tracker.py           # 6. §4.8 row 1 — reference point first
python 06_calibrate_general.py               # 7. §4.8 row 2
python 07_calibrate_specialists.py           # 8. specialists + half-split logging
python 08_meta_decision.py                   # 9. meta layer (val)
python 09_experiments_analysis.py --with-model  # 10. all tables (test)
python 10_figures_report.py                  # 11-12. §4.9 table, figures
python 11_reproducibility.py                 # 13-14. package
```

Everything from `04` onward runs on **cached signals only** (no model
forward passes) — re-running a calibration variant after a design tweak
takes seconds. The expensive step is `03` (~2–5 h on a free T4 for 2,000
samples, resumable: a restarted Colab session picks up where it stopped via
`out/progress/extract.json`).

**Stale-cache protection:** `02` writes a fingerprint of the dataset. If a
later run uses different sizes/mode (e.g. smoke test → full run), `02`
detects the mismatch and automatically clears the signal cache and
extraction checkpoint so old rows can never mix into a new run.

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
| `MULTI_HARM_TARGET_FPR` | `0.03` | tighter FPR budget |

## Correctness harness (no GPU)

```bash
python run_smoke_test.py
```

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

1. **Attention signal R.** `R(l,h) = (m_qi/W_i) / (m_qp/W_p + ε)` — the
   **per-token** attention intensity on the injection region relative to the
   per-token intensity on the whole passage. `m_qi`, `m_qp` are sums over
   (query rows × span columns) and are divided by the span widths, which
   makes R **invariant to injection length and query length**. This matters:
   `combined` payloads are several times longer than `naive` ones, so a raw
   sum ratio would have made `combined` mechanically easier to detect —
   a pure span-length artifact, not the hypothesis under test. For **clean**
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
9. **Clean pairs.** MS-MARCO golden query-passage pairs (HF `ms_marco`,
   `passage_ranked` dev); fallback to `data/clean_pairs.csv`
   (`passage,query` columns) or `--synthetic` (offline test only).

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

## Timeline (free T4, per v3)

~4–6 days across sessions: `03` dominates (~3 h extraction + 30 min
quant-compare); `01` model download ~5 GB; everything else is minutes.
Each script is idempotent — kill and restart anywhere.
