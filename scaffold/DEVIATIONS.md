# SCAFFOLD deviations log

Each entry is recorded before any run it affects.

## D1 (2026-10-16 ~13:05 +04:00, before any pilot or Stage-0 run): B-learning update v1′ (normalised KL anchor; mass term on raw output)

**What happened.** In a smoke test (16 items, B seed 9, results/smoke), the output-matched arm stopped with a non-finite gradient.

**Cause.**
- The v1 A adapters are extremely saturated: A's non-chosen letters sit at about −116 to −134 nats, while base's sit at −12 to −28 (`dbg_tables.py`). Output-matching offsets therefore reach +125.
- The v1 KL anchor, Σ exp(lp_raw)(lp_raw − lp0), and the mass term, −logsumexp(lp_raw), both assume raw letter log-probs ≤ 0.
- With offsets, once B-learning raises a suppressed letter of A, the matched value becomes positive and exp overflows.
- The audit also found that the v1 "letter KL" is not a KL: it is unnormalised and can be negative.

**Change (applied to every arm, so arms stay comparable).**
- Policy: unchanged, renorm((lp_raw + off) / τ).
- KL anchor: a proper KL between the four-letter renormalised distributions, KL(q_t ‖ q_0), with q = softmax(lp_raw + off), weight β = 0.03 unchanged.
- Mass term: −logsumexp over the letters of the model's raw output lp_raw (no offset), weight γ = 0.1 unchanged.
- All other settings are v1: ε 0.1, τ 2, moving baseline, Adam 5e-4, batch 4, clip 1.0.

**Why output matching is still the intended construction.** The matched policy is softmax(z_base + Δz_A(φ)) up to per-item constants: base's starting logits plus the change that φ produces through A's network. A logit change of a given size therefore moves the matched learner and the base learner identically. The only difference between them is the map from φ to Δz, i.e. the kernel.

**Effect on the preregistration.** §2 ("frozen v1 exact update") now reads "v1′ as above". No hypothesis, arm, metric or threshold changes. No pilot or Stage-0 result had been seen; the only numbers seen were the 16-item smoke runs, which are discarded.

## D2 (2026-10-16 ~13:30 +04:00, after the Stage-0 smoke test on one adapter and before the main Stage-0 run): option-token sites added; decision-direction estimator changed

**Smoke result** (ST2_20260919, 8 situations per domain × 4 rotations, `results/stage0_smoke`):
- **U_pca4 (the M2 analogue) is a letter readout.** At the final position it copies the donor's letter on 100 % of interchange pairs, with 0 % meaning flips (layers 24 and 26; necessity 0.69 and 1.0). This confirms the audit's letter-readout finding.
- **No letter-free signal at the final position.** Every letter-residualised candidate is inert at layers 12–26: 0 flips, copy at chance.
- **Reading:** the final position carries which letter to emit, and any letter-agnostic decision variable must be computed upstream.

**Changes for the main Stage-0 run:**
1. **New sites.** Candidates are also fitted and patched at the last token of the two decision-option texts ("…shared cause affecting all three", "…separate causes"), at layers 8, 12, 16, 20 and 24. Sites: optS, optP, and both jointly.
2. **New estimator.** w_dec becomes the class-mean difference of letter-residualised deltas. The joint least-squares fit is underdetermined (d = 2048 > n) and gave a minimum-norm, effectively random direction.
3. **New candidate.** U_res16 is added at the final position.
4. **Cost saving.** Necessity and sufficiency are computed only for the 3 candidates with the highest fit-domain meaning-flip, plus the controls.

The selection rule (on fit domains: maximum meaning-flip with letter-copy ≤ 0.2) and the gate (on software: flip ≥ 0.5, copy ≤ 0.2, necessity ≥ 0.5) are unchanged.

## D3 (2026-10-16 ~13:50 +04:00, after a second 4-situation smoke run and before the main Stage-0 run): flip and copy measured as excess over the unpatched pair rates; full-residual reference swap added

**Problem.** In the smoke runs, a patch that changes nothing leaves each recipient on its own answer. That answer coincides with the donor's chosen letter on about 22 % of pairs, so raw "letter-copy" sits at about 0.22 for an inert patch. The preregistered "letter-copy ≤ 0.2" therefore rejected every candidate, inert ones included.

**Clarification.** The preregistration meant copying *caused* by the patch. Flip and copy are now reported as excess over the same pairs' unpatched rates; raw rates are kept in the JSON. Thresholds are unchanged.

**Addition.** A whole-residual swap at each site ('full') gives an upper bound on what any subspace at that site can carry. It is reference only and excluded from selection.

**Smoke-run observation (4 situations per domain; not a result).** (See D4 for the main-run adaptation.) No candidate at any option-token site or final-position layer 16–26 flipped the decision. The final-position U_pca4 at layers 24–26 copies the donor's letter on 100 % of pairs.

## D4 (2026-10-16 ~14:15 +04:00, after the main Stage-0 run of adapter 1 and before adapters 2–6): adaptive light mode for adapters 2–6

**Adapter 1 (ST2_20260919, full candidate set, 32 situations per domain × 4 rotations, 41 min)** found no candidate that passes the gate.
- Every candidate at every option-token site and layer 8–24 gave 0 meaning-flip and 0 excess copy, **including the whole-residual swap**.
- At the final position, layers 16–20 are inert even under the whole-residual swap.
- At layers 24–26, U_pca4 equals the whole-residual swap (excess letter-copy 0.64, flip 0). The final-position change is entirely "which letter".

**Change.** A whole-residual swap transfers all of the donor's information at that site, so it bounds what any subspace swap there can do. For adapters 2–6:
- Stage 0 runs the whole-residual swap at every site and layer, plus U_pca4 at the final position.
- The full candidate set is re-run only where a whole-residual swap reaches meaning-flip ≥ 0.2 on the fit domains ("ESCALATED" in the logs).

The gate and selection rule are unchanged. The reason is cost: about 41 minutes per adapter at the full set, with GPU time shared with another application.

## D5 (2026-10-17 06:40 +04:00, after the grid finished, before any analysis output was read): analysis crash fix
analyze.py declared hboot(cells, fn, ...) with an unused required argument `fn`, so it crashed on the first call. `fn` is now optional (`fn=None`); no computation changes. The replicate-floor runs (results/stage2_rep) did not execute because their output directory did not exist (the PowerShell redirect failed and the log shows a 0-second "exit 0"). They are rerun below.

## E1 (2026-10-17 06:45 +04:00, after reading the frozen analysis): exploratory analyses, labelled as such
The frozen verdict is "uninterpretable" on both rules: PC-H2 failed its calibration (pc_calib/selection.json; mean G gain bind −0.06/−0.03/−0.05, bext +0.07/+0.08/+0.01 for factors 2/4/8). The following are **post hoc and exploratory**. They do not change the frozen verdicts and are reported separately.
1. **Sensitivity benchmark with known pure-process changes.** The base network with Adam learning rate ×2 and ×4 (arms base_lr2, base_lr4; forward pass identical to base) on Stage-2 B seeds 101–103, both rules. Its G gain over base sets the scale of process effects the G metric can detect at this n.
2. **Start × network factorial.** Raw − prior isolates the network effect given A's starting predictions; matched − base isolates it given base's. Both get the same hierarchical bootstrap.
3. **Replicate floor.** The base arm rerun on B seed 101, both rules.
4. **Forecast (preregistered §6, secondary).** forecast.py on B seed 101, both rules, all adapters.

## E2 (2026-10-17 06:55 +04:00, exploratory, written before running): is the start × network interaction specific to A's abstraction?

**Observation (E1, post hoc).** A's network minus the base network, both starting from A's predictions (A_raw − prior):
- bind +0.089 [0.025, 0.159];
- bext +0.114 [0.044, 0.174].

From base's predictions (Δ_K) the same difference is about 0, so the interaction is +0.103 (bind) and +0.143 (bext); 16/18 and 17/18 cells are positive.

**Candidate explanation.** A's letter readout is saturated and high-gain: normalised letter KL 10.35, and U directions move letter logits about 22× more than random. A's network can therefore overturn its own extreme priors, which the base network cannot do from the same start. That would make the effect generic to output saturation, not specific to the abstraction.

**Test.** Run the A_raw and prior arms with the shuf and single control adapters (3 seeds each) on B seeds 101–103, both rules. shuf is about as saturated as A (normalised letter KL 7.6–11.4) but learned no rule (accuracy 0.47–0.53).

**Reading, fixed now.** For each rule, compute the control's interaction ((raw − prior) − (matched − base)), using its own matched arm from Stage 2.
- If shuf's interaction is ≥ 0.5 × A's, A's effect is classed as **generic saturation**.
- If shuf's interaction is ≤ 0.25 × A's and its 95 % CI excludes A's point estimate, it is classed as **abstraction-specific**.
- Anything else is unresolved.

single is reported beside it.

## D6 (2026-10-17 ~09:35 +04:00): Stage 3 stopped by the author during its pilot

The author stopped the Stage-3 chain (PREREGISTRATION_STAGE3.md, frozen 09:28, sha256 89b30e9f…) to proceed directly to the paper draft.
- **Completed:** one pilot cell (Qwen3-0.6B, B_ind, pilot B seed 1, base and A_raw arms; results/stage3_pilot_q06).
- **Not run:** no budget selection, no control training, no Stage-3 grid.
- **Reporting:** no Stage-3 result is reported or analysed. The preregistration stays on file as planned future work, and the paper reports Stage 3 as preregistered and not run.

## E3 (2026-10-17, post hoc, written after all Stage-2, E1 and E2 results were read): robustness of the Stage-2 contrasts

Prompted by a review of the paper draft. Purely exploratory: no frozen verdict changes. Script: robustness.py → results/EXPLORE_E3.json.

1. **Trajectory noise.**
   - Every contrast so far was taken against a single base run per B seed. The three perm controls (letter KL to base ≤ 0.01, base-like kernel) are numerically perturbed replicates of base, so base plus the three output-matched perm runs form a four-run reference per seed.
   - Reference SD of G within a seed: 0.027 (bind), 0.077 (bext). The bext base run on seed 101 lies +0.127 above its reference mean.
2. **Crossed design and few clusters.** Adapters and B seeds are crossed, not nested, and the base run is shared within a seed. Intervals are recomputed three ways:
   - a three-way bootstrap (adapters, B seeds, reference runs within seed);
   - t intervals over the six adapter means (df 5);
   - a stratified permutation test (A runs vs reference runs, permuted within seed).
3. **Components of the E2 interaction.**
   - The E2 statistic ((raw − prior) − (matched − base)) mixes two components. The motivating observation concerned only the first, raw − prior.
   - That component is compared between A and each control group with a two-way bootstrap.

## E4 (2026-10-17, post hoc, prompted by an independent review of the draft): forecast evaluation and adapter-level kernel check

Script: forecast_eval.py → results/FORECAST_EVAL.json.
1. **Preregistered forecast (Stage-2 §6).**
   - Measure: Spearman of F_res, F_sign and CKA_type at t = 0 with G, across arm × cell. forecast.py ran on B seed 101 only.
   - The draft reported "no statistic reaches 0.5", but no saved script computed it; it is now computed.
   - Result: fails on both rules (largest 0.23).
2. **Exploratory.** Across adapters, does a t = 0 kernel statistic of the output-matched frame rank the mean Δ_K?
   - All four statistics are reported, for the six A adapters (exact permutation p) and for all 15 networks.
   - F_sign ranks the six A adapters at Spearman 0.943 on both rules (exact two-sided p = 0.017 each), and all 15 networks at about 0.60.
   - It is a hypothesis for the four held-out adapters, not a finding.

Also found in review, and disclosed in the paper:
- The mass term is computed on the raw output, so it is A's in A_matched.
- A_matched_fade applies offsets tabulated on a 0.1 grid of the scale while the weights use the exact scale.
- The control table's "letter KL" measures disagreement with base, not saturation.

## E5 (2026-10-17, written before running): t = 0 kernel statistics for the four held-out adapters, to be registered as a forecast

- **What runs.** forecast.py on P1_20260925–28, B seed 101, both rules, output-matched frame only. It uses the same settings as the six Stage-2 adapters. No B training is run, so no Δ_K outcome for these adapters is known or computed.
- **What follows.** The F_sign values and the predicted ranking of Δ_K go into FSIGN_FORECAST.md. That file is hashed, and the author registers it publicly before the preprint is posted and before any held-out B-learning run.
