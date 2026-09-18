# Running Multi-HARM v3.1 on Kaggle

Same code, same stages as the Colab demo — Kaggle just needs four settings
handled up front. `demo_kaggle.ipynb` has all of this as cells; this page is the
why.

## 1. Kernel settings

| Setting | Value | Why |
|---|---|---|
| Accelerator | **Nvidia T4 x2** (or P100) | `03` extraction and `09 --with-model`; everything else runs on CPU |
| Internet | **ON** | `pip`, and the 5 GB model download. With Internet OFF, attach the model as a Kaggle **Model/Dataset** and point `MULTI_HARM_MODEL_NAME` at `/kaggle/input/...` |
| Environment variables | add `HF_TOKEN` (secret) | `meta-llama/Llama-3.1-8B-Instruct` is gated. No token → use an ungated model (below) |
| Persistence | none needed | write to `/kaggle/working`; see §4 for surviving a session end |

## 2. Do **not** `pip install -r requirements.txt`

Kaggle's image already ships a CUDA-correct `torch` + `bitsandbytes` pair.
`requirements.txt` lists `torch`, so a wholesale install re-resolves it and can
break that pairing (the failure mode is a `libcublasLt.so.* not found` or an
nf4 `RuntimeError` at load time, not a clean error message). Install only what
is missing:

```bash
!pip install -q "transformers>=4.45" accelerate bitsandbytes datasets
```

`02` and `04`–`11` need nothing beyond numpy/pandas/scikit-learn/pyarrow/
matplotlib, all preinstalled — `env.py` makes torch optional precisely so those
stages never depend on the GPU environment.

## 3. Environment: per-cell `source` does not persist

`!source ./demo_env.sh && python 03...` works only inside that one cell — the
same caveat as Colab, but on Kaggle people more often split stages across cells.
Set the variables in the kernel once instead:

```python
import os
os.environ.update({
    "MULTI_HARM_DATA_DIR": "/kaggle/working/data",
    "MULTI_HARM_OUT_DIR": "/kaggle/working/out",
    "MULTI_HARM_QUANT": "nf4",
    "MULTI_HARM_N_CLEAN": "400",          # demo sizes (800 samples total)
    "MULTI_HARM_N_INJ_PER_CELL": "20",
    "MULTI_HARM_N_BASE_PAIRS": "900",
})
```

Full-run sizes instead: `N_CLEAN=1000 N_INJ_PER_CELL=25 N_BASE_PAIRS=2200`.
Note `MULTI_HARM_CALIB_PER_SPECIALIST` (160 by default): with 160 clean
calibration rows the smallest achievable FPR is 1/160 = 0.00625, which is just
under the per-specialist budget of `target_fpr/4` = 0.0125 — so θ is resolvable,
but only barely. Raise it to 400 for the paper-facing run; if it is not
resolvable, `07` and `RESULTS.md` say so explicitly rather than letting an
off-target FPR read as a tuning miss.

## 4. Order, timing, and resuming

```
0  python run_offline_check.py            # ~21 s, CPU: proves 02 + 04-11
0b python run_tiny_model_check.py --full   # ~3 min, CPU: proves 01 + 03 wiring
1  python 01_setup_and_validate.py         # model load + §2.0 shape gate (5 GB dl)
2  python 02_build_dataset.py              # ~3 min
3  python 03_extract_signals.py --quant-compare   # §2.3, ~30 min (optional)
4  python 03_extract_signals.py            # 30-75 min demo / 2-5 h full  ← the long one
5  python 04_..._08                        # seconds each, cached signals only
6  python 09_experiments_analysis.py --calib-sweep --with-model
7  python 10_figures_report.py && python 11_reproducibility.py
```

Run the two CPU checks *before* the model download: they are what caught v3.0's
`roc_auc_score(pos_label=...)` bug, and `run_tiny_model_check.py --full` exercises
`01`/`03` against a real transformer forward pass (a tiny random-init
Llama-architecture model built locally — no download, meaningless numbers,
correct wiring).

`03` checkpoints to `out/progress/extract.json` every 25 rows and flushes the
parquet sinks first, so a mid-run kill costs at most a re-do: re-run cell 03 in
the **same session** and it resumes. Across sessions, `/kaggle/working` is
per-run, so:

1. **Save Version** (outputs keep `data/signals` + `out/`).
2. New kernel → attach the old version's output as input.
3. `shutil.copytree("/kaggle/input/<prev>/data", "/kaggle/working/data", dirs_exist_ok=True)`
   plus the same for `out/`, then re-run cell 03 without `--fresh`.

If that abort with *"data/signals holds N rows extracted from a DIFFERENT
dataset (changed: ...)"*, you changed sizes or the model between sessions —
that is v3.1's provenance guard, and it is right: re-run `02`, then `03 --fresh`.
The same record is what `11` puts into the repro zip as `signals_meta.json`.

## 5. Model choice

The repo default is `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated). Ungated drop-in
alternatives, one env var each:

```python
os.environ["MULTI_HARM_MODEL_NAME"] = "microsoft/Phi-3.5-mini-instruct"      # 3.8B, Apache
os.environ["MULTI_HARM_MODEL_NAME"] = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # fast pilot
```

Swapping is safe in v3.1 and *was not* safe in v3.0: layer/head counts now come
from a lookup over `num_hidden_layers|n_layer|num_layers` and
`num_attention_heads|n_head|num_heads`, and `01` prints the architecture it
resolved. If a config exposes none of them, `01` raises naming what it tried,
instead of `03` dying 40 minutes into extraction.

## 6. Two Kaggle-specific failure modes worth knowing

- **Attention memory.** Signals come from `output_attentions`, i.e.
  layers × heads × T² per sample. Eager attention is therefore mandatory (the
  loader forces it) and peak memory grows with `T²`: 32 layers × 32 heads ×
  1024² ≈ 2.1 GB at fp16, ≈ 4.8 GB at 1536. On a T4, keep `MAX_SEQ_LEN=1024`;
  `03` now prints how many samples had spans clipped by the limit, so a payload
  that does not fit is visible instead of quietly diluting `R`.
- **CPU-only kernels** run everything except `01`/`03`: `run_offline_check.py`,
  `02`, `04`–`11` need no GPU and no torch. Useful for iterating on tables and
  figures against an already-extracted cache (attach it as a dataset).

## 7. Outputs to download

From the kernel's Output panel (or `Save Version`):

- `out/report/RESULTS.md` — lead §4.8 table, Table A, §4.3 head-level
  specialization, §4.4 honest attribution, FPR-resolution caveats
- `out/experiments/SUMMARY.md` — every success criterion with MET/NOT MET
- `out/figures/*.png` — 8 figures incl. `fig_calib_sweep.png`
- `out/validation/width_invariance.json` — the width-confound measurement
- `out/repro/multi_harm_repro.zip` — code + artifacts + `repro_manifest.json`
  (SHA-256 of every artifact, git commit, cache provenance)
