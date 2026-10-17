# Stage 0 outcome (recorded 2026-10-16 23:40 +04:00)

**Sources.**
- **Adapter 1 (ST2_20260919):** full candidate set; results/stage0/ST2_20260919.json and .log.
- **Adapters 2–6 (ST2_20260921, ST2_20260922, ST2b_20260920, P1_20260923, P1_20260924):** light mode (DEVIATIONS D4); results/stage0/<adapter>.stdout.txt.
- **Bug note.** The light runs crashed *after* all interchange passes had finished and been logged, but before the JSON was written: a KeyError when looking up a control candidate light mode does not compute (fixed in stage0.py). The logged fit-domain flip and copy values below are complete for every site and layer. Test-domain values and necessity/sufficiency for adapters 2–6 were not written. They are not needed for the gate decision, because the gate requires a fit-domain flip ≥ 0.5 before any test-domain check.

**Accuracy.** Meaning accuracy on fresh items (32 situations per domain × 4 rotations): base 0.504 (fit) / 0.516 (software); every A adapter 1.000 / 1.000.

**Interchange results.** Values are fit-domain excess meaning-flip / excess letter-copy.

| Site | Layers | All 6 adapters |
|---|---|---|
| Option token "…shared cause affecting all three" (optS), whole-residual swap | 8, 12, 16, 20, 24 | 0.00 / 0.00 |
| Option token "…separate causes" (optP), whole-residual swap | 8, 12, 16, 20, 24 | 0.00 / 0.00 |
| Both option tokens (opt2), whole-residual swap | 8, 12, 16, 20, 24 | 0.00 / 0.00 |
| Final position, whole-residual swap | 16 | 0.00 / 0.00 |
| Final position, whole-residual swap | 20 | flip 0.00; copy 0.00 / 0.20 / 0.07 / 0.43 / 0.19 / 0.00 |
| Final position, U_pca4 (4-d) | 24, 26 | 0.00 / 0.64, identical to the whole-residual swap |
| Adapter 1 only: w_dec, U_res2/4/16, L_base, rand at every site and layer | — | 0.00 flip |

**Gate.** Flip ≥ 0.5 with copy ≤ 0.2 and necessity ≥ 0.5: **FAILS for 6 of 6 adapters** (maximum flip anywhere is 0.00).

**Reading.**
- What A adds at the final position is "which letter to emit". A 4-d subspace at layers 24–26 carries all of it: the swap copies the donor's letter, never its meaning.
- No single position carries a letter-free decision variable in the whole residual, at any tested layer: neither the final position (through layer 20) nor either decision-option token (layers 8–24).
- So A's decision is either computed redundantly across many positions, or assembled at the final position in layers 21–23 from many positions at once, in a letter-bound form.
- This independently confirms the audit's finding that the M2 "four directions" are the letter readout, and extends it: there is no letter-agnostic single-site address for A to replace them.

**Consequence (preregistration §5).** Stage 2 runs at the adapter level. The localisation clause (removing r_A during learning) and the A_matched_rm / A_matched_rand arms are dropped. Δ_K, the control comparisons, S and S_fade remain.
