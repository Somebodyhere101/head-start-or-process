# SCAFFOLD: does an acquired abstraction change how the next one is learned? Stage 1 preregistration

Written 2026-10-16 before any run in this directory. The author delegated every design decision below to the assistant; decisions and reasons are recorded in §9. This file is frozen by its sha256, written in `PREREGISTRATIONS_SHA256SUMS` before the first Stage-0 or pilot run. Stage 2 (the decisive grid) gets its own frozen addendum after Stage 0 and the pilot, and before any Stage-2 arm runs. The full motivation and evidence audit are in the assessment document of the 2026-10-16 session (`ASSESSMENT.md`).

## 1. Question

Does an abstraction A, acquired from outcome feedback, change the *process* by which a later abstraction B is learned, beyond giving B a head start or information? The strongest form ("scaffold"): A is needed for B to *form* but not for B to be *used*.

## 2. Definitions

- θ0 is Qwen3-1.7B. A is an existing recipe-v1 adapter (rank-8 q/v LoRA trained on outcome-labelled plant and clinic situations under the rule "one shared cause iff same time window AND same infrastructure"). It is merged into the weights: θA = θ0 + ΔA.
- B-learning: a fresh rank-8 q/v LoRA φ is trained on top, with the frozen v1 exact update (exp_state.consequence_pass_exact semantics: ε 0.1, τ 2, mass 0.1, KL anchor 0.03, Adam lr 5e-4, batch 4, gradient clip 1.0). Training uses the `interpret` question only, on software-domain situations.
- **Learning** means updates to φ. Activation hooks used during B-learning are removed before testing unless a condition says otherwise. Nothing is given in context.
- **Output matching.** On every prompt x the learner's four letter log-probabilities are lp_θ(x) − sg[lp_θA(x) − lp_θ0(x)], with the offset table precomputed from the frozen θA and θ0. At t = 0 its policy, KL anchor and mass term are exactly base's; its Jacobian with respect to φ is A's.
- **Prior shift (the H1 null).** Base model with offset +sg[lp_θA − lp_θ0]. This is A's starting predictions on base's kernel.
- **r_A.** A's identified, letter-agnostic representation (Stage 0). "Removing r_A during learning" means subtracting the per-prompt vector P_r(h_θA(x) − h_θ0(x)) at r_A's layer and final position on every forward pass during B-learning, and not at test.
- **Scaffold test.** The learned φ is evaluated on θ0 + φ (A unmerged).

## 3. Hypotheses

Let G be the mean balanced held-out B accuracy over the evaluation points of the budget (§6).

- **Δ_start** = G(prior shift) − G(base).
- **Δ_K** = G(A output-matched) − G(base).
- **H1 (head start / information):** Δ_K equivalent to 0 (TOST ±0.05).
- **H1-reuse:** Δ_K ≥ 0.10 on B_ext only, and S ≤ 0.
- **H2 (process):** Δ_K ≥ 0.10 on B_ind with 95% CI excluding 0. It must exceed each A′ control by ≥ 0.10; removing r_A during learning must remove ≥ 50% of Δ_K; removing a random subspace of the same size must remove ≤ 20%.
- **H2-strong (scaffold):** H2, and S = acc(θ0 + φ_{A,matched}) − acc(θ0 + φ_{base}) ≥ 0.10 with CI excluding 0.

## 4. Tasks (generator in `bworld.py`; default CED code is untouched)

All B situations use the software frame of `ced/world.py` with four cues:
- time window and infrastructure, as in CED;
- reporting desk (", reported via <desk>");
- a new work-order cue (", under <work order>"), same or different across the three lines.

All B items carry all four phrases, so the text distribution is identical across B tasks. Only the rule differs.

- **B_ind = desk AND work order.** Time and infrastructure are drawn independently (p = 0.5 each), so A's rule is statistically independent of B's label. Test items: half positives, half near-misses (one of the two cues false, the two types balanced).
- **B_ext = time AND infrastructure AND desk** (the D3 and3 rule). The work-order cue is random. Test items: half positives, half near-misses (each of the three cues false, balanced).
- **Training sets:** 192 situations (half positive, half near-miss). **Evaluation:** 96 fresh situations per B seed.
- **Phrase-shift evaluation** for the form of B: 96 situations whose desk and work-order phrases come from disjoint pools.
- **Balanced accuracy:** mean of (accuracy on positives, mean accuracy over near-miss types). A constant policy scores 0.50, a single-cue policy at most 0.75, the conjunction 1.00.

## 5. Stages

**Stage 0 (identification of r_A; exploratory measurement with a gate).**
- **Adapters:** ST2_20260919, ST2_20260921, ST2_20260922, ST2b_20260920, P1_20260923, P1_20260924. P1_20260925–28 are held out for a later confirmation.
- **Data:** for each adapter, interpret prompts from fresh items (plant and clinic for fitting, software for testing), each presented in all four option rotations so letter identity is decorrelated from the decision.
- **Candidates**, at layers {12, 16, 20, 24, 26}, final position:
  - (a) U_M2, the saved M2 structure (reference);
  - (b) U_res: top-k PCs of A-minus-base deltas after regressing out the one-hot of the correct letter;
  - (c) w_dec: the letter-adjusted decision-label regression direction (k = 1);
  - k ∈ {1, 2, 4}.
- **Tests on software:**
  - letter-span overlap: base unembedding rows of A–D times the final-norm gain, centred;
  - necessity: subtract P δ in A;
  - sufficiency: add P δ in base;
  - interchange: for a common/separate item pair with different letter rotations, replace P h in A from the other item. "Meaning-flip" = the choice moves to the other decision's option; "letter-copy" = the choice moves to the donor's letter.
- **Controls:** a random subspace of equal k, and the base letter-identity subspace at the same layer.
- **Gate (per adapter; selection on the fit domains, confirmation on software):** meaning-flip ≥ 0.5, letter-copy ≤ 0.2, necessity ≥ 0.5 of A's software interpret gain.
- **If no candidate passes for at least 4 of 6 adapters,** Stage 2 runs at the adapter level: "remove A during learning" becomes scaling ΔA to 0 during B-learning only, and the localisation clause is dropped. This is reported as a finding.

**Stage P (budget calibration; base and A_raw only, plus an exactness check of output matching).**
- B seeds 1–3, both B tasks, up to 6 passes over the 192 training situations, evaluated every 96 situations.
- **Budget rule:** the smallest number of passes at which base reaches balanced accuracy ≥ 0.85 on 1 or 2 of the 3 pilot seeds per B task (neither floor nor ceiling). If base never reaches 0.85 within 6 passes, or reaches it on 3 of 3 seeds within 1 pass, the task difficulty is adjusted (training size or cue salience) and the change is recorded before Stage 2.
- **Exactness check:** at t = 0 the output-matched arm's letter log-probabilities equal base's on every prompt (maximum absolute difference ≤ 1e-3, as bf16 allows).

**Stage 1 (instrument sensitivity; run inside the Stage-2 grid on a subset).**
- PC-H2 is a learner whose forward function is exactly base's, but whose φ-gradients are rescaled along the top-8 gradient subspace of a separate calibration set of B items (×4). That is a process effect by construction.
- The battery must classify PC-H2 as Δ_K > 0 and S > 0, and the prior-shift arm as Δ_K = 0.
- If PC-H2 is not detected, Stage-2 nulls are reported as uninterpretable.

**Stage 2 (decisive grid; parameters frozen in a Stage-2 addendum).**
- 6 A adapters × 3 B seeds × 2 B tasks.
- **Arms:** base; A_raw; prior shift; A output-matched; A output-matched with r_A removed during learning; A output-matched with a random equal-k subspace removed during learning; A′_shuf, A′_single and A′_rand output-matched; PC-H2 on a subset.
- **Control adapters:** trained in this project with the v1 recipe on the ST2 seeds:
  - A′_shuf: outcomes shuffled across situations;
  - A′_single: the infrastructure-only rule;
  - A′_rand: a random rank-8 adapter scaled to A's mean KL.
- Replication: Qwen3-0.6B (O5 adapters ST2s_*), same design.

## 6. Measurements

- **G:** balanced held-out accuracy at 8 evenly spaced points over the budget (after updates; t = 0 is recorded separately).
- **Labels-to-criterion (0.85):** restricted mean survival time.
- **S:** final accuracy of θ0 + φ. Also θA + φ with r_A removed at test.
- **Persistence:** for the arms with removal during learning, the final test uses an intact θA.
- **Form of B:** near-miss profile, phrase-shift accuracy, and the share of φ's layer-L effect inside r_A.
- **Forecast:** before any B update, the fresh φ's empirical NTK on B's training prompts (gradient of the vindicated-letter log-prob with respect to φ's B matrices) and its centred alignment with B's ±1 labels. Prediction: across arm × seed, Spearman(alignment, G) ≥ 0.5.
- **Diagnostics:**
  - gradient norm and Adam step size per arm;
  - a normalised letter KL (renormalised over the four letters), replacing CED's "letter KL";
  - a plain-vs-plain replicate floor (base arm run twice on two B seeds).

## 7. Analysis

- All contrasts are paired within A seed × B seed. Confidence intervals come from a hierarchical bootstrap (2,000 resamples, over A seeds, then B seeds).
- Holm correction across the four primary contrasts (Δ_K, A′ comparison, localisation, S) within each B task.
- Smallest effect of interest 0.10; equivalence bounds ±0.05.
- Every arm, failure and deviation is reported. Thresholds are not changed after results.

## 8. Forecasts (written before any run)

- Δ_start ≥ 0.10 on B_ext: 0.9.
- Δ_K ≥ 0.10 on B_ind: 0.2.
- H2-strong: 0.07.
- Stage-0 gate passes for ≥ 4 of 6 adapters: 0.35. The M2 structure is 58–60% letter readout, and M1 found nothing at the situation tokens.

## 9. Decisions made by the assistant (author delegated)

1. **Interpret question only.** Connect and pursue options name the infrastructure, which would bias B_ind.
2. **B in the software domain.** A never trained there; this matches L1, X1 and D3.
3. **The work-order cue,** rather than reusing time or infrastructure, so that B_ind is independent of A by construction.
4. **Six adapters for the test, four held out.** This makes confirmation possible without new training.
5. **The v1 exact update.** It reveals every letter's reward, so the evidence stream is identical across arms. The confidence-weighting confound is handled by output matching rather than by changing the update.
6. **B seeds vary items, option orders and φ initialisation.** This fixes the hard-coded seed in `ced/later_learning.py`.
7. **Laptop RTX 5070 for Stage 0 and the pilot; DQ RTX 5090 for the grid** when the paper-6 queue allows. GPU temperature is checked before launches, with one job on the card at a time.
