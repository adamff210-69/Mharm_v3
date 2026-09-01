# Multi-HARM — what we can say now (plain language)

Two finished experiments. Treat them as two separate results. Do not blend the numbers.

1. **Loud templates on MS MARCO** (earlier Kaggle run, ~800 examples). Attacks look like banners: “IMPORTANT OVERRIDE”, “[SYSTEM NOTICE]”.
2. **QuietRAG** (this notebook, 228 examples). Attacks are quieter, often mid-document, with character spans that round-trip 204/204. Types present: loud-ish (`naive`, 24), on-topic (`topic`, 84), stealth (`combined`, 96), plus 24 clean.

Model for QuietRAG: Mistral-7B-Instruct, 4-bit, two T4 GPUs.

---

## Bottom line

**Yes — the system mostly flags “attack vs no attack” correctly on QuietRAG.**

On the held-out QuietRAG test set it caught **98% of injected documents** and flagged **0% of clean documents**. About **2% of attacks slipped through**.

**No — QuietRAG did not fix the research claim we actually needed.** The mixer never mixed. It always threw away the attention signal and kept only the hidden-state signal (`α = 0` on every detector). Specializing per attack type did not beat one shared detector. Naming the attack family is still near chance-to-mediocre (50% when it fires).

So: **a working yes/no injection alarm**, not a fusion paper and not an attack-type classifier.

---

## What “flag correctly” means (two different questions)

### Question 1 — Is this document injected or clean?

This is the only question the numbers support.

QuietRAG **test** (the split the detector was not tuned on):

| | |
|---|---|
| Injected documents caught | **98.0%** (50 of 51 injected test rows, ASR ≈ 2%) |
| Clean documents wrongly flagged | **0%** (0 of 6 clean test rows) |
| Same picture on the val split | 97.1% caught, 0% false alarms |

Loud MS MARCO **test** was even more extreme: essentially **100% caught, 0% false alarms**. That run used very obvious banner text, so it is the easy case.

**Caveats for Question 1**

- QuietRAG test clean set is tiny (**6** clean documents). Zero false alarms on 6 rows is encouraging, not a tight false-alarm rate.
- QuietRAG `naive` test set is **6** rows. Do not over-interpret per-type rates there.
- The alarm that actually fires is almost entirely the **hidden-state probe** (a linear readout of the model’s internal vector at layer 8). Attention ranking looks decent, but the attention **threshold** on QuietRAG was unusable (see below).

### Question 2 — Which kind of attack is it?

**Not solved.**

| Run | Correct type among detected attacks |
|---|---|
| Loud MS MARCO | ~30% |
| QuietRAG | **50%** |

50% with three types is better than 30% with four, but it is still not a type-ID system. Target was 75%. **Do not write that Multi-HARM attributes attack type.**

Confusion on QuietRAG test (who got credit when something was flagged):

- True `topic` often labelled `topic` (15) but also `naive` (4).
- True `combined` scattered across `combined` (6), `topic` (10), `naive` (5), `general` (3).
- True `naive` mostly `naive` (4) on only 6 rows.

---

## Did QuietRAG improve anything?

| Goal | Loud MS MARCO | QuietRAG | Improved? |
|---|---|---|---|
| Mix attention + hidden into one score | Mixer weight always 0 (hidden only) | Mixer weight always **0** (hidden only) | **No** |
| Beat “hidden only” by mixing | No — hidden already ~1.00 | No — hidden already 0.98–1.00 | **No** |
| Per-type specialists beat one shared detector | No | **Worse** (shared 0.984 / 1.00 / 1.00 vs specialized 0.984 / 0.944 / 0.965) | **No** |
| Yes/no detection vs clean | Essentially perfect (easy banners) | **98% hit / 0% false alarm** on harder text | Detection still works; not a fusion win |
| Name the attack type | ~30% | **50%** | Slightly, still a fail |
| Attention as a standalone alarm | Worked as a ranker; threshold fired some | Ranker still strong (~0.96) but **threshold never fired** (caught 0% of attacks) | Attention alarm got **worse** |
| Attacks look like real RAG | Loud banners | Quieter, mid-doc, real spans | **Yes** (dataset quality) |

QuietRAG did what we asked of the **data**: quieter, spanned, not re-wrapped with `IMPORTANT OVERRIDE`. It did **not** make the two signals complementary. The hidden-state score is still so strong that the mixer has nothing to add.

---

## How the detector actually behaves (no mixer mystique)

Two internal clues are computed from one forward pass:

1. **Attention clue** — does the last user token look at the suspected attack span more than at the rest of the passage?
2. **Hidden-state clue** — a small probe on the model’s layer-8 vector, trained to say “injected vs clean”.

They are combined as:

`final score = α × (attention) + (1 − α) × (hidden)`

`α` is chosen on training data. **Every QuietRAG detector chose α = 0.** That means the published “fusion” score **is the hidden-state score**. Attention is computed and then ignored.

That is the same outcome as the loud run. It is not a bug in the code path this time (the `0 × inf` crash was already fixed). The grid search simply prefers hidden because hidden already separates injected from clean almost perfectly.

---

## QuietRAG numbers you can put in a table

### Detection (yes/no), test split

| Setting | Injected caught | Missed attacks | Clean wrongly flagged |
|---|---|---|---|
| Full Multi-HARM (which is hidden-only in practice) | 98.0% | 2.0% | 0% |
| One shared hidden probe | same as above (identical α = 0) | | |
| Attention only, one shared cutoff | **0%** caught | **100%** missed | 0% |
| Per-type specialized fused scores | slightly **worse** ranking than the one shared probe | | |

Per-type catch rate (test): `combined` 100% of 16 val / strong on test; `topic` 93% on val; `naive` 100% on 4 val / 6 test rows. Report overall 98%, not a drama per type.

### Ranking quality (how well scores sort injected above clean)

Think “1.0 = perfect sort, 0.5 = coin flip”.

| Signal | topic | naive | combined |
|---|---|---|---|
| Attention only, shared | 0.976 | 1.000 | 0.938 |
| Hidden only, shared | 0.984 | 1.000 | 1.000 |
| “Fused” shared (α = 0) | 0.984 | 1.000 | 1.000 |
| Per-type specialized | 0.984 | 0.944 | 0.965 |

Attention **can rank**. It **cannot fire** on QuietRAG with the cutoff the script chose (θ = 5.17, an extreme z-score). So a paper sentence like “attention generalizes across types” is only about ranking, not about an operating alarm.

### Calibration half-split (training scores, why fusion died)

| Detector | Attention half | Hidden half | Combined | Mixer weight α |
|---|---|---|---|---|
| topic | 1.00 | 1.00 | 1.00 | 0 |
| naive | 0.50 | 0.98 | 0.98 | 0 |
| combined | 0.96 | 1.00 | 1.00 | 0 |
| shared general | — | — | 1.00 | 0 |

Combined never beats hidden. On `naive`, attention on the tiny calibration set is a coin flip.

### Other checks

- **Held-out type** (`combined` left out of the specialist set): miss rate 17% vs 13% on seen types. Within 10 points — a weak pass, small n.
- **Specialist scores are correlated** (0.45–0.59). They are not independent experts.
- **Extra cost of scoring 3 specialists vs 1**, including the forward pass: **+5.4%**. Target was < 0.5%. Scoring itself is milliseconds; the forward pass dominates.
- **Span length vs attention ratio** on test: correlation about **−0.24** (not a length cheat, not zero either). Stealth spans are longer on average (median 48 tokens, mean 107 because of a few very long ones) than `naive` (27) and `topic` (34).

---

## Sentences that are honest for the paper

**Can say**

- We test a span-aware yes/no detector on (a) loud template injections in MS MARCO and (b) a quieter RAG-style set (QuietRAG, 204 attacks / 24 clean) with character spans that match the document.
- On QuietRAG test, the detector flags 98% of injected documents at 0% false alarms on the 6 clean test documents.
- The residual / hidden-state probe at mid-layer is the component that does the work on both sets.
- A learned mix of attention-routing and that probe **selects 100% hidden-state** on both sets (`α = 0`). Mixing does not improve over hidden-state alone.
- Training a separate detector per attack family does **not** beat one shared probe on QuietRAG.
- Recovering the attack family from which detector fires is 50% on QuietRAG (30% on loud MS MARCO) — not usable as attribution.
- Attention scores still rank injected vs clean well (~0.94–1.00) on QuietRAG, but a single shared cutoff did not yield an operating detector (0% recall).

**Must not say**

- That this is the first successful fusion of attention and residual signals for detection. Fusion never won.
- That specialists are necessary. On QuietRAG they hurt.
- That the system identifies attack type.
- That QuietRAG “unlocked” mixing. It did not.
- That false-alarm rate is tightly 0% in deployment. n_clean test = 6.
- That latency overhead is under 0.5%. Measured +5.4% with the forward pass.

**Limitation paragraph (use this)**

Results are from one 7B instruction model at 4-bit, one QuietRAG build (228 rows, three mapped families, 17 underlying recipes), and one loud MS MARCO template set. QuietRAG negatives include only 24 clean/near-miss documents. Adaptive attacks (optimized suffixes, etc.) were not tested. The mixer’s collapse to hidden-state is expected when the hidden probe already separates the two classes; that is a finding, not a justification to lock `α` by hand.

---

## Status of the project (practical)

| Item | Status |
|---|---|
| Code path (QuietRAG loader, no re-wrap, type list from data, fusion NaN fix, type-vs-clean ranking) | Landed on `arena/01a05640-mharm-v3` (`404b2f1`) |
| Loud MS MARCO run | Finished earlier. Keep that notebook as the loud result. |
| QuietRAG run | Finished. Zip: `quietrag_out.zip` (`out/` + `dataset.parquet`). Save Version on Kaggle. |
| Fusion claim | **Not supported** by either run |
| Yes/no detection claim | **Supported**, with the n and model caveats above |
| Type-ID claim | **Not supported** |
| Next experiment that would change the story | Only if hidden-state is **no longer** near-perfect — harder negatives, adaptive attacks, or a setting where attention and hidden disagree. TensorTrust/BIPIA are the wrong shape for this week (no RAG spans). |

**One-line summary for you:** it flags injected vs clean well; it does not mix the two signals; QuietRAG made the data more honest, not the fusion result better.
