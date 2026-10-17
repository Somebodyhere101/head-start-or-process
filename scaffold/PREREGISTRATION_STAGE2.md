# SCAFFOLD Stage 2 addendum (decisive grid)

Builds on PREREGISTRATION.md (sha256 37506f0c…, frozen 2026-10-16 12:35 +04:00), DEVIATIONS.md D1–D4, PILOT_OUTCOME.md and STAGE0_OUTCOME.md. It is frozen by sha256 in PREREGISTRATIONS_SHA256SUMS before the first Stage-2 run. Code is frozen by the sha256 of the files listed in the same file. No Stage-2 arm, and no output-matched, fade or control arm on any B seed, had been run when this was frozen.

## 1. Stage-0 outcome and its consequence

**The gate fails for 6 of 6 adapters** (STAGE0_OUTCOME.md): maximum meaning-flip 0.00 at every site and layer, including whole-residual swaps at both decision-option tokens and at the final position. A's final-position change is a 4-d letter readout.

Per preregistration §5:
- the localisation clause is dropped, and the A_matched_rm / A_matched_rand arms are not run;
- this is reported as a finding: A's acquired skill has no single-site, letter-free address in Qwen3-1.7B.

## 1b. PC-H2 strength, fixed before Stage 2 on the pilot seeds

The smoke test suggested the ×4 amplification may destabilise learning, and a failed sensitivity check makes every Stage-2 verdict uninterpretable. The factor is therefore chosen by pc_select.py on pilot B seeds 1–3 (not reused in Stage 2), with the Stage-2 budgets:
- the smallest factor in {2, 4, 8} whose mean G gain over base is ≥ 0.10 on both B rules;
- if none qualifies, PC-H2 counts as failed and Stage-2 verdicts are reported as uninterpretable for lack of demonstrated sensitivity, with all effect estimates still shown;
- the chosen factor is written to results/pc_calib/factor.txt and used unchanged for every Stage-2 PC_H2 arm.

## 2. Fixed parameters
- **B seeds:** 101, 102, 103 (the pilot used 1–3; none are reused).
- **Budgets** (from the pilot, by the frozen rule): B_ind 1 pass over 192 situations, evaluated every 24; B_ext 3 passes over 192, evaluated every 72. In both cases G is the mean of 8 evaluation points.
- **A adapters:** ST2_20260919, ST2_20260921, ST2_20260922, ST2b_20260920, P1_20260923, P1_20260924. P1_20260925–28 are held out.
- **Control adapters** (results/controls; v1 recipe on the ST2 seeds 19, 21, 22):
  - shuf_<seed>: outcomes permuted;
  - single_<seed>: infrastructure-only rule;
  - perm_<seed>: A's LoRA factors row/column-permuted, norm-matched with the function destroyed.

  Each control's software interpret accuracy and normalised letter KL are reported beside A's.

  Measured already for perm: accuracy 0.516, equal to base; normalised letter KL 0.001–0.012, against 10.35 for A. It is therefore norm-matched but not KL-matched, and is read as a weight-scale control only. shuf and single are trained tonight, before the grid starts.
- **Update:** v1′ (DEVIATIONS D1) for every arm.

## 3. Arms

Every arm, per rule × B seed, uses the same items, order, φ initialisation and update.

| Arm | Network during B-learning | Output offset | Role |
|---|---|---|---|
| base | θ0 | none | reference |
| PC_H2 | θ0; each update's component in an 8-d calibration-gradient subspace amplified ×4 | none | sensitivity: a process-only effect by construction |
| A_raw | θA | none | the confounded effect |
| prior | θ0 | +(lpA − lp0) | the information-matched prior shift (H1 null) |
| A_matched | θA | −(lpA − lp0) | **Δ_K** |
| A_matched_fade | θ0 + c·ΔA, with c = 1 for the first two thirds of the budget, then linear to 0 | −(lp_{A,c} − lp0), recomputed per c | scaffold test without abrupt removal |
| control A_matched | θ0 + ΔA′ | −(lpA′ − lp0) | generic-adaptation control |

Plain-vs-plain replicate floor: the base arm is run again for B seed 101 on both rules (results/stage2_rep).

## 4. Primary contrasts and decision rules

Primary contrasts, analysed by analyze.py exactly as frozen:
- Δ_K = G(A_matched) − G(base);
- Δ_K − Δ_K(control), for each of shuf and single (and perm, secondary);
- S = scaffold(A_matched) − scaffold(base), where scaffold means θ0 + φ;
- S_fade = final(A_matched_fade) − final(base).

The hierarchical bootstrap, Holm correction, TOST (±0.05) and smallest effect of interest (0.10) are as in PREREGISTRATION.md §7.

| Verdict | Rule |
|---|---|
| **Uninterpretable** | PC_H2 does not raise G by ≥ 0.10 over base (mean over B seeds) on that rule |
| **H1 (head start only)** | Δ_K is equivalent to 0 (90 % CI within ±0.05) |
| **H2 (process)** | Δ_K ≥ 0.10 with 95 % CI > 0, and Δ_K exceeds Δ_K(shuf) and Δ_K(single) by ≥ 0.10 each. Localisation is dropped if the Stage-0 gate failed. On B_ind this is the decisive test; on B_ext a positive is read as H1-reuse unless S > 0 |
| **H2-strong (scaffold)** | H2, and S ≥ 0.10 with 95 % CI > 0. S_fade is reported beside S as the less biased removal |
| **Inconclusive** | anything else |

## 5. Secondary (reported, not decisive)
- Δ_start and Δ_raw.
- Phrase-shift accuracy for each arm.
- Near-miss profiles.
- Gradient norms.
- The forecast (forecast.py, run after the grid on the same cells): Spearman of F_res, F_sign and CKA_type with G across arm × cell.
- The 0.6B replication, in a later addendum.

## 6. Forecasts
Unchanged from PREREGISTRATION.md §8, with one addition after the Stage-0 result: P(PC_H2 detected on both rules) = 0.7.
