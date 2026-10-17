# Stage P outcome (budget calibration), recorded 2026-10-16 13:20 +04:00 from results/pilot/*.json

Arms run: base and A_raw (adapter ST2_20260919); B seeds 1–3; 6 passes over 192 training situations; evaluation every 96 situations (96 fresh items, balanced interpret accuracy). This is calibration only: no output-matched, removal or control arm was trained.

**Exactness.** Fresh zero adapter: max |Δlp| = 0 in all 6 cells. Matched-versus-base max |Δlp|: 5.2e-6 to 7.6e-6. PASS (≤ 1e-3).

**Base learning curves (balanced accuracy at 96, 192, 288, … situations).**

| B task | Seed | Curve | First point ≥ 0.85 |
|---|---|---|---|
| B_ind | 1 | 0.50, 0.55, 1.00, 0.69, 1.00 … | 288 |
| B_ind | 2 | 0.52, 0.65, 0.96, 1.00 … | 288 |
| B_ind | 3 | 0.49, 0.92, 0.93, 1.00 … | 192 |
| B_ext | 1 | 0.50, 0.55, 0.55, 0.73, 0.55, 0.88, 0.93 … | 576 (0.95 final) |
| B_ext | 2 | ≈ 0.43–0.52 throughout | never (0.48 final) |
| B_ext | 3 | ≈ 0.49–0.56 until the final point | never (0.84 final) |

**Budget rule** (preregistration §5): the smallest number of passes at which base reaches ≥ 0.85 on 1 or 2 of 3 seeds.

| B task | Within 1 pass | Within 3 passes | Budget |
|---|---|---|---|
| B_ind | 1 of 3 | 3 of 3 | **1 pass (192 situations)**; 3-of-3 within 1 pass did not occur, so no difficulty adjustment |
| B_ext | 0 of 3 | 1 of 3 | **3 passes (576 situations)**; the never-reach clause does not apply, since seed 1 reached 0.88 |

G uses 8 evenly spaced evaluation points: every 24 situations for B_ind and every 72 for B_ext.

**A_raw, descriptive (not a hypothesis test).**

| B task | Start | G vs base G (seeds 1 / 2 / 3) |
|---|---|---|
| B_ind | 0.53–0.58 (no head start, as designed) | 0.917 / 0.997 / 0.786 vs 0.895 / 0.927 / 0.944 |
| B_ext | 0.78–0.79 (head start) | 0.988 / 0.940 / 0.997 vs 0.788 / 0.477 / 0.542 |

**Design consequence found here (affects the Stage-2 addendum).**
- The A_raw adapters evaluated with A unmerged (θ0 + φ) score 0.51 / 0.60 / 0.79 (B_ind) and 0.53 / 0.55 / 0.52 (B_ext). The base adapters score 0.84–1.00.
- An adapter learned on top of A is therefore co-adapted with A, and an abrupt unmerge is biased against the scaffold hypothesis. This matches Yang et al. 2026 (arXiv:2609.10287), where abrupt removal fails and gradual removal succeeds.
- The Stage-2 addendum adds a gradual-fade arm, with ΔA scaled linearly from 1 to 0 over the last third of the B budget. The abrupt unmerge stays the preregistered S; the fade arm is reported beside it.
