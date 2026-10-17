# Frozen forecast: F_sign ranks the held-out adapters' output-matched learning (E5)

Frozen 2026-10-17 10:47 +0400. No B-learning run has been made with any of these four adapters.

## Background
In Stage 2 (Qwen3-1.7B), the t = 0 sign-gradient kernel alignment F_sign of the output-matched frame ranked the six used A adapters' mean Δ_K at
Spearman 0.943 on both B tasks (post hoc, E4; one of four statistics tried; forecast_eval.py). This file turns that observation into a test on
the four adapters designated as held out in PREREGISTRATION.md (P1_20260925–28).

## Measured predictor (forecast.py, B seed 101, output-matched frame, t = 0; same settings as the six)
| adapter | F_sign B_ind | predicted rank B_ind | F_sign B_ext | predicted rank B_ext |
|---|---|---|---|---|
| P1_20260925 | 34.0 | 1 | 46.6 | 1 |
| P1_20260926 | 25.5 | 4 | 13.6 | 2 |
| P1_20260927 | 26.6 | 3 | 9.3 | 3 |
| P1_20260928 | 28.2 | 2 | 4.1 | 4 |

Rank 1 = largest predicted output-matched learning (G of A_matched); the prediction is that rank order of G(A_matched) follows F_sign.

## Outcome (to be run later)
- For each held-out adapter and each rule (B_ind, B_ext): the A_matched arm on ten new B seeds 301–310, with the Stage-2 recipe and budgets
  unchanged (frozen recipe v1 with D1; B_ind one pass, B_ext three passes; eight evaluation points), as in the conditions where E4 was found.
- Outcome per adapter and rule: mean G(A_matched) over the ten seeds. (The shared base reference cancels in a ranking across adapters, so the
  ranking of Δ_K equals the ranking of G(A_matched).) A replicated base reference (base plus perturbed replicates) is run on the same seeds so
  that levels can also be reported, but the test below uses only the ranking.

## Test
- Primary statistic: the mean over the two rules of Spearman(F_sign, mean G(A_matched)) across the four held-out adapters.
- Null: exact, permuting adapter labels independently within each rule (24 x 24 = 576 equally likely outcomes).
- Confirmed if the statistic is >= 0.800 (one-sided exact p = 0.0312 <= 0.05). Disconfirmed if it is <= 0. Otherwise inconclusive.
- Secondary (reported, not decisive): each rule separately; whether the adapter with the lowest F_sign on a rule has the lowest G(A_matched);
  and the same test at the amended Stage-3 budgets if those are run.
- With four adapters the test has low power; a non-confirmation is weak evidence against the hypothesis, a confirmation moderate evidence for it.

## Commitments
- This file is hashed into PREREGISTRATIONS_SHA256SUMS and is to be registered on a public registry by the author before the preprint is posted
  and before any held-out B-learning run.
- Any deviation is logged in DEVIATIONS.md before results are read.
