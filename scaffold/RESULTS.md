# SCAFFOLD results (Qwen3-1.7B), 2026-10-17

Preregistration: Stage 1 frozen 2026-10-16 12:35 (sha256 37506f0c…); Stage 2 frozen 23:32 (0e369def…), with code hashes in PREREGISTRATIONS_SHA256SUMS. Deviations D1–D5 and exploratory analyses E1–E2 are in DEVIATIONS.md. The full narrative is in the session's ASSESSMENT.md §6.

- **Pilot** (PILOT_OUTCOME.md): budgets B_ind 1 pass, B_ext 3 passes. Output-matching exactness ≤ 8e-6.
- **Stage 0** (STAGE0_OUTCOME.md): the gate fails for 6/6 adapters. No single-site, letter-free address for A exists. The final-position change at layers 24–26 is a 4-d letter readout (64 % excess letter-copy, 0 meaning flips).
- **Stage 2** (results/stage2/ANALYSIS.json). The frozen verdict is "uninterpretable" on both tasks, because PC-H2 failed calibration (results/pc_calib/selection.json).

| Contrast | B_ind | B_ext |
|---|---|---|
| Δ_raw | +0.156 [0.098, 0.211] | +0.283 [0.204, 0.348] |
| Δ_start | +0.067 [0.036, 0.099] | +0.169 [0.135, 0.203] |
| Δ_K | −0.014 [−0.054, 0.024]; equivalent within ±0.05 | −0.029 [−0.098, 0.047] |
| S | −0.035 [−0.070, 0.004] | −0.273 [−0.347, −0.202] |
| S_fade | +0.056 [−0.020, 0.145] | −0.048 [−0.123, 0.019] |
| Δ_K − shuf / single / perm | +0.004 / −0.101 / −0.043 | +0.063 / +0.032 / +0.030 |

- **E1** (results/EXPLORE_E1.json):
  - Learning rate ×2: +0.074 (B_ind), −0.109 (B_ext). Learning rate ×4: −0.126, −0.178.
  - Replicate floor: 0.000.
  - Start × network interaction: +0.103 (B_ind), +0.143 (B_ext).
  - Forecast (results/forecast): FAILED, with no statistic reaching Spearman 0.5. A's kernel is 10–3,000× larger in scale and about 10× less aligned with B.
- **E2** (results/EXPLORE_E2.json): **generic saturation**. Shuffled-outcome controls reproduce the interaction: +0.069 (B_ind), +0.129 (B_ext).

**Conclusion.** At this scale, A affects B-learning through information (head start, reuse) and generic output saturation. It does not change how B forms, and it is not a scaffold.
