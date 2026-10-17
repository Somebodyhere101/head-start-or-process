# SCAFFOLD Stage 3: replication at 0.6B, a B task without repeated-string cues, and a preregistered sensitivity control

Written 2026-10-17 ~10:00 +04:00, before any Stage-3 run. It builds on PREREGISTRATION.md, PREREGISTRATION_STAGE2.md, DEVIATIONS.md (D1–D5, E1–E2) and RESULTS.md (the Stage-2 results at 1.7B). It is frozen by sha256 in PREREGISTRATIONS_SHA256SUMS together with the code it runs. The author delegated the design decisions; they are recorded in §8.

## 1. Why Stage 3

Stage 2 (Qwen3-1.7B) found three things:
1. Δ_K ≈ 0 (equivalent to 0 on B_ind).
2. No scaffold.
3. A start × network interaction that E2 classed as generic output saturation.

It left three gaps:
- the preregistered sensitivity control (PC-H2) failed;
- B_ind's cues are repeated strings, like A's infrastructure cue, so A's learned feature partly predicts B_ind (A starts at about 0.6);
- one model size.

Stage 3 closes these three gaps.

## 2. New task: B_para (bworld.py, rule 'bpara')

- Software-domain situations. Time and infrastructure are generated as in ced, independently of the label; infrastructure is still a repeated string.
- Two rule cues whose sameness is **semantic, never a repeated string**:
  - **Batch value**, written in three notations: numeral, English word, arithmetic ("12" / "twelve" / "3 times 4").
  - **Supplier city**, written as name, nickname and description ("Chicago" / "the Windy City" / "Chi-Town").
- Every item uses exactly one of each form for each cue. "Same" items repeat the value or city; "different" items use three different values or cities. The form pattern is therefore identical across labels, and no batch or city string repeats within an item.
- Rule: one shared cause iff same batch value AND same city.
- Verified (generator, 4,000 items): P(label | A's rule true) = 0.501 vs 0.499 when false; P(label | same infrastructure) = 0.508 vs 0.492.
- Items, question format, balanced accuracy and item counts are as in Stage 2 (interpret question; 192 training situations; 96 evaluation situations).

## 3. Models, adapters, B seeds

**Qwen3-0.6B (replication of the full design).**
- A adapters: ST2s_20260919, ST2s_20260921, ST2s_20260922. These are the O5 v1-recipe adapters (far c/i/p 0.997–1.000).
- Controls are trained now with the unchanged v1 recipe on the same seeds: q06shuf_<seed> (outcomes permuted) and q06single_<seed> (infrastructure-only rule).
- B tasks: B_ind, B_ext, B_para.

**Qwen3-1.7B (new task only).**
- The six Stage-2 A adapters, with the existing shuf_<seed> and single_<seed> controls.
- B task: B_para.

**B seeds.** Pilot: 1–3. Stage 3: 201, 202, 203; none of these were used before.

## 4. Arms (same items, order, φ initialisation and update v1′ within each model × task × B seed)

| Cell | Arms |
|---|---|
| Base cell | base; **base_lr2** (Adam learning rate ×2, forward identical: the preregistered sensitivity control) |
| Each A adapter | A_raw; prior; A_matched; A_matched_fade |
| Each control adapter (shuf, single) | A_raw; prior; A_matched. The raw and prior arms make the saturation test confirmatory |

The permuted-weight control is dropped: it was inert at 1.7B (letter KL ≈ 0, G = base). Stage-0 localisation is not repeated as a gate. A light Stage 0 on the three 0.6B adapters is run as a descriptive replication of the letter-readout finding.

## 5. Budget calibration (pilot)

- Base and A_raw on pilot B seeds 1–3, 6 passes, evaluated every 96 situations, for (0.6B: B_ind, B_ext, B_para) and (1.7B: B_para).
- p* is the smallest number of passes at which at least one seed reaches balanced accuracy ≥ 0.85.
  - If p* = 1 and all 3 seeds reach 0.85 within 1 pass, the budget is half a pass (96 training situations).
  - If no seed reaches 0.85 within 6 passes, the budget is 6 passes and the task is flagged "hard". It still runs; the sensitivity rule decides whether it is interpretable.
- G is the mean balanced accuracy at 8 evenly spaced points over the budget.
- pilot_select.py applies these rules mechanically and writes results/stage3_budgets.json.

## 6. Decision rules (per model × task; analysis by analyze3.py as frozen)

Contrasts are paired within adapter × B seed, with a hierarchical bootstrap (2,000 resamples, adapters then B seeds). Effects use 95 % CIs, equivalence uses 90 % CIs, the smallest effect of interest is 0.10, and the equivalence bound is ±0.05.

**Sensitivity condition.** |mean over B seeds of G(base_lr2) − G(base)| ≥ 0.07. If it is not met, that model × task is "uninterpretable" for null claims; estimates are still reported.

| Verdict | Rule |
|---|---|
| **H1 (no process effect)** | Sensitivity met, and Δ_K equivalent to 0 (90 % CI within ±0.05) |
| **H2 (process)** | Δ_K ≥ 0.10 with 95 % CI > 0, and Δ_K − Δ_K(shuf) ≥ 0.10 and Δ_K − Δ_K(single) ≥ 0.10 (point estimates) |
| **Scaffold** | H2, and S ≥ 0.10 with 95 % CI > 0 |
| **Inconclusive** | Anything else |

**Saturation test.** Interaction = (G(raw) − G(prior)) − (G(matched) − G(base)).
- If A's interaction is < 0.05, it is "absent".
- Otherwise:
  - shuf's interaction ≥ 0.5 × A's → **generic saturation**;
  - shuf's interaction ≤ 0.25 × A's, with its 95 % CI excluding A's point estimate → **abstraction-specific**;
  - otherwise → unresolved.

**Confirmatory claims.**
- **C1.** At 0.6B, H1 holds on B_ind and B_para.
- **C2.** At 1.7B, H1 holds on B_para.
- **C3.** No scaffold anywhere: S is not ≥ 0.10 with CI > 0 in any model × task.
- **C4.** Where an interaction is present, it is generic saturation.

## 7. Forecasts (written before any run)

| Claim | Forecast |
|---|---|
| C1 | 0.65 (0.6B may sit near the floor on B_para) |
| C2 | 0.7 |
| C3 | 0.85 |
| C4 | 0.6 |
| Sensitivity condition met on every model × task | 0.7 |
| B_para flagged "hard" at 0.6B | 0.35 |

## 8. Decisions (author delegated)

1. **B_para's semantic sameness.** Number notations and city aliases. A small LM knows these equivalences, and they remove literal string repetition entirely.
2. **The learning-rate ×2 benchmark** replaces PC-H2 as the sensitivity control. It was the only exploratory benchmark that moved G by a known process change in both directions.
3. **The permuted control is dropped,** since it was inert.
4. **The saturation test becomes confirmatory.**
5. **Run on the laptop.** DQ is occupied by paper 6 (26 jobs left). The thermal rule (start each job below 62 °C) is kept.
