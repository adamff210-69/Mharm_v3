# Multi-HARM — Review / Slide Notes (pilot run, patch cycle)

> **Status line (say this plainly at the top):**
> These are **pilot-scale** numbers from a small, synthetic/templated clean-text run.
> They are directionally useful and internally diagnostic — they are **not** final results.
> The full MS-MARCO + diverse-passage run is what will support a robust claim.

---

## 1. Code bug fixed this cycle (before you re-render any table)

The per-attack-type AUROC slabs were filtering to **only** that type's injected
rows. With no clean rows inside the slab, every per-type AUROC was a
one-class-vs-itself computation (collapsing to 1.0 or 0.5 depending on the
metric behavior). Fixed consistently in this repo:

| File | Lines | What changed |
|---|---|---|
| `05_baseline_attn_tracker.py` | per-type AUROC (row 1) | `attack_type == t` → `attack_type == t \| label == 0` |
| `06_calibrate_general.py` | per-type AUROC / hidden-only baseline | same fix |
| `09_experiments_analysis.py` | `per_type_auroc_of`, Table B, Table C halves, top-1 vs top-K, §4.8 row 3 | `types == t` → `types == t \| labels == 0`, plus same mtest fix |
| `10_figures_report.py` | per-specialist ROC curves | same fix |

**Consequence:** Table B's `attn-shared (4.8 r1) = 0.5000`, and every other
1.0000 per-type cell that came from the old slabs, are **not final**. Re-run
05 → 06 → 07 → 08 → 09 → 10 after this patch and present the re-rendered
numbers.

> If you cannot re-run before the review: say "Table B row 1 is provisional;
> that cell is known-broken pending the per-type AUROC fix." Do **not** read
> the old row as *"attention alone gets 0.5, fusion gets 1.0."*

---

## 2. What the current pilot run actually tells us

Three findings are genuine and consistently explained. They matter more than
the raw wall of 1.0000s.

### Finding 1 — alpha = 0.00 everywhere: the "fusion" is currently hidden-state only

- Every specialist (topic, naive, fake, combined) picked `alpha = 0` on the
  calibration grid.
- So the fused score being used right now is **100% the hidden-state probe**;
  the attention half contributes 0.
- **Why is this not a bug?** On this tiny calibration set the hidden signal
  already separates clean vs injected at AUROC ~1.0, so the grid correctly
  finds no calibration-set benefit to blending in attention.
- **What it means for the review:** do not let the word "fusion" imply
  attention+residual fusion is happening. Say:
  *"At this scale, calibration found zero weight for the attention signal, so
  the detector is presently the residual/hidden probe; whether attention
  contributes is an open question for the real-data run."*

### Finding 2 — specialist correlation 0.79–0.96: success criterion must be marked NOT MET

- v3 success target was pairwise specialist Pearson `< 0.5`.
- Observed range: **0.79–0.96 → NOT MET.**
- **Mechanism:** with `alpha = 0` everywhere, every specialist is a
  slightly-differently-calibrated hidden-state probe over the same handful of
  clean/injected samples. High redundancy is exactly what you'd expect.
- **Honest slide framing:**
  *"We expected near-independence; we got heavy redundancy. The result is
  mechanically consistent with the alpha=0 outcome, and it is a real,
  explainable non-result rather than noise."*

### Finding 3 — attribution accuracy 0.35 (target > 75%): topic specialist fires on everything

- Attribution accuracy is **0.35**, well below the >75% target.
- §4.4 matrix shows the topic specialist catches 5/5 topic, 5/5 naive, 2/5
  fake, 5/5 combined on val, while naive and combined specialists catch ~0
  injected of any type.
- Detection still hits 100% overall because the ensemble OR-rule only needs
  **one** specialist to fire — and that one is essentially always topic (or
  general). The argmax attribute then mislabels non-topic attacks as topic.
- **What to not do:** don't present the "detection = 1.0" headline without
  pairing it with "attribution = 0.35," because the overall number hides that
  the system is not actually identifying attack *type* well.

---

## 3. A genuine win with no caveats

**Latency / shared-forward-pass overhead** (from your run):

- 1 specialist: **2.22 ms**
- 5 specialists: **11.35 ms**
- extra scoring for 4× specialists: **~9.1 ms**

This is a real efficiency result — running four extra specialist scorers on
one shared forward pass costs no additional model forward pass. Safe to show
without the pilot-scale caveat; just label it as "scoring-only overhead in
this environment."

---

## 4. Slide-ready narrative

> **The pipeline runs end-to-end with real calibration on a real model.**
> At this small pilot scale, the hidden-state signal alone separates clean
> from injected almost perfectly — likely inflated by templated synthetic
> clean text, not yet proof of robust detection. More importantly, calibration
> correctly detected that attention contributed nothing at this scale
> (alpha = 0), which explains why specialists are highly correlated (0.79–0.96,
> target < 0.5 — NOT MET) and why attribution underperforms its target
> (0.35 vs > 0.75). Those are two consistent, mechanistically-explained
> findings, not noise. The open question for the full run is whether attention
> starts contributing once real, diverse MS-MARCO passages replace the
> synthetic pilot data.

---

## 5. What to say / not say

**Say:**
- "Pilot-scale, synthetic clean text — numbers likely inflated; treat as
  diagnostic, not final."
- "Detection works end-to-end; attribution does not yet."
- "Specialist independence criterion explicitly NOT met; here's the
  mechanism."
- "Attention contributes 0 at this scale; fusion claim is deferred to the full
  run."
- "Latency win is real."

**Don't say / don't lead with:**
- "AUROC = 1.0 across the board" without the caveat.
- "Fusion beats attention" — old Table B comparison is broken until 05 is
  re-run with the patch.
- "Specialists are complementary" — the correlation matrix says the opposite
  on the pilot set.
- Any claim that the pilot proves robust prompt-injection defense.

---

## 6. Next steps

1. Re-run the chain on your Colab/Kaggle session **after this patch**:
   `05 → 06 → 07 → 08 → 09 → 10`
2. Bring back the new `09` output (Tables A–E, §4.3–4.8).
3. If `alpha` is still 0 at pilot scale, state the negative result as a
   positive diagnostic and defer the fusion claim to the full run.
4. If the full run starts showing `alpha > 0`, re-check specialist correlation
   before claiming independence.

---

*Generated from the arena/01a05384-mharm-v3 working tree. Code changes are in
`05_baseline_attn_tracker.py`, `06_calibrate_general.py`,
`09_experiments_analysis.py`, `10_figures_report.py`. No run artifacts exist in
this sandbox (no `data/`, `out/`, or ML deps), so the numbers above come from
the session log being reviewed, not a fresh execution.*
