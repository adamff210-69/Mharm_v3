# Multi-HARM — Project Review Demo (Colab T4, ~1–1.5 h)

Reduced-size real-data run: **400 clean MS-MARCO + 400 injected
(4 attack types × 5 goals × 20) = 800 samples**, Llama-3.1-8B-Instruct
4-bit (nf4). Same pipeline, same code as the full 2,000-sample run —
only the dataset size changes (env vars in `demo_env.sh`).

**Integrity line to use at the review:** *"These numbers are from the
800-sample pilot run; the full 2,000-sample run is queued for this
week. All pipeline logic, validation gates, and reporting are identical."*

## Colab quick start

```
1. Repo is already on GitHub:  !git clone <your-repo-url> && %cd <repo>
   (or upload multi_harm_code.zip:  !unzip multi_harm_code.zip && %cd multi-harm)
2. Run demo_colab.ipynb top to bottom (or:  !source ./demo_env.sh && bash demo_run.sh)
```

## Timing on a free T4 (with buffer)

| Stage | Command | Time | What you'll see |
|---|---|---|---|
| install | `pip install -r requirements.txt` | ~2 min | — |
| 01 | setup + model | 10–20 min (5 GB download) | env report, 3-sentence shape smoke test |
| 02 | dataset | ~3 min | 800 rows, split crosstab |
| 03 | extraction | **30–75 min** | **§2.0 validation gate 12/12**, then progress+ETA |
| 04 | pooled H\* | seconds | best (layer, head) + top-10 table |
| 05 | 4.8 row 1 | seconds | attention-only shared per-type AUROC |
| 06 | 4.8 row 2 | seconds | HARM_general + PIShield-style baseline |
| 07 | specialists | seconds | **half-split AUROC table (Phase 3 addition)** |
| 08 | meta (val) | seconds | FPR/ASR/attribution on val |
| 09 | all tables | ~5 min (loads model for latency) | Tables A–E, §4.3/4.4, **§4.8 spine table**, span-width audit |
| 10 | figures+report | seconds | `RESULTS.md` + 7 PNGs |

**Critical path is stage 03.** Stages 04–10 run on cached signals
(seconds to minutes) — if you're close to review time with 03 still
running, you can already present: env/gate (01–03 top), and the plan
for the rest. But realistically 03 finishes well inside an hour.

## Decision tree

- **No T4 attached (CPU runtime):** restart the runtime a few times
  (free-tier assignment is flaky); if persistent, use Colab Pro or a
  Kaggle notebook (same commands work; set `MULTI_HARM_DATA_DIR` if the
  input dir is read-only).
- **OOM in 03 (shouldn't at 4-bit):** check nothing else is using the
  GPU; 1024-token 8B nf4 fits a T4 with margin.
- **Session dies mid-03:** restart the SAME runtime, re-run the 03 cell
  — it resumes from `out/progress/extract.json` (skip: already-extracted ids).
- **02 fails to find MS-MARCO:** the loader tries several HF configs,
  then suggests a local CSV; worst case drop a `data/clean_pairs.csv`
  (`passage,query` columns, ≥900 rows) and re-run 02.
- **Review moved up:** 04+05+06 (≈40 min in) already give you the
  §4.8 rows 1–2 + the H\* table — present those and the 07/08/09
  numbers as "minutes away".

## What to show (in this order)

1. **§2.0 gate output** (12/12 token-range checks) — "we validate the
   alignment that silently invalidates attention-based detectors."
2. **The §4.8 spine table** (`out/experiments/table_48.json` /
   `SUMMARY.md`) — attention-only shared vs fused-shared vs
   fused-specialized, per-type AUROC + spread. This is the paper's
   lead result.
3. **Half-split table** (07 output) — "specialization is carried by
   the residual half / by the attention half" — state whichever the
   numbers say, including if the gain is small (the v3 spec plans for
   the negative outcome as a publishable result).
4. **Success-criteria table** in `SUMMARY.md` — honest MET/NOT MET
   status against the v3 targets.
5. **Span-width audit** (09) — combined payloads are ~1.7× longer than
   naive; R is span-width-invariant by construction (per-token
   intensity ratio vs passage body). Good defensive talking point.
6. **Latency** (09 `--with-model`) — 5 specialists cost <0.5% over one
   shared forward pass.

## Expected anti-patterns (don't be surprised)

- Pilot AUROCs will be noisier than the full run (n≈160 calib, n≈40
  test per type). If a per-type AUROC looks odd, that's sampling noise,
  not a pipeline bug (the pipeline is gated at 03).
- Attribution accuracy on the pilot can swing a lot; the confusion
  table in Table D is more informative than the single number.
- `corr(R, W_i)` injected-only can look non-zero on 40 samples; on the
  full run (200/test) it's the meaningful number.
