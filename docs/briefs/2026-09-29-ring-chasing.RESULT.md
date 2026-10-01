# Ring-chasing discount result

## Change

No model change. The pre-registered ring-chasing arm failed, so every
ring-chasing shape is rejected on the current frame (v6.0.2). Two diagnostic
arms, which could never be adopted, located the failure.
`scripts/eval_ringchase_gated.py` keeps the arms and the diagnostic
`--oracle` mode for the reopen condition below.

## Evidence

### Pre-registered arm

The arm is `--gate continuous`, run at 10 seeds on the v6.0.0 frame (885
rows). It computes `latent - delta * w * (latent - floor)`, with
`w = clip((p_last_2y - p0) / (1 - p0), 0, 1)`. The hazard is the frozen
`scripts/build_retirement_hazard.py` fit.

| Check | Bar | Result |
|---|---|---|
| Corrected paired dSel | t > 2 | +0.01167, t = +0.88, FAIL |
| delta interior | most pools | at the grid edge in 25 of 50 pools, FAIL |
| A1 without LeBron James 2026 | improves | 0.8466 -> 0.8460, FAIL |
| C2 worst segment | <= $0.30M | +$0.172M, pass |

With LeBron included, A1 goes from 0.8357 to 0.8454. His row alone moves
from $46.45M to $14.09M against an actual $3.88M.

The arm moved 91 rows: 51 better and 40 worse.

| Moved rows | n | MAE gain | Champion bias |
|---|---:|---:|---:|
| Minimum signings | 48 | +$74.5M (+$42.2M without LeBron) | +$2.68M |
| All other signings | 43 | -$26.0M | -$2.59M |

MAE improves, but R-squared falls without LeBron. The gains are many small
floor-level corrections. The losses deepen large under-predictions on paid
veterans: Chris Paul 2024 -$5.0M, Ian Mahinmi 2019, Maxi Kleber 2023, Khris
Middleton 2026, and Marcus Smart 2026.

Earlier three-seed screens used a one-sample t that overstates the evidence:

| Gate | t |
|---|---:|
| age, no ring | +0.19 |
| age, earnings | +1.30, LeBron alone |
| hard retire gate | +2.74 with delta capped at 1.5; +1.74 with the cap at 3.0 |

### Estimator check

Realized label: the player's last Basketball Reference season falls within
two seasons of the signing. The label is known only when the season after
that window has been observed, so only for frame seasons up to 2023.

On 158 veteran rows aged 30 or older with a known label (76 were last
contracts):

- The hazard's `p_last_2y` has AUC 0.725. Age alone has AUC 0.562.
- By hazard quartile, the realized last-contract rate is 0.22, 0.38, 0.59,
  and 0.72.

The estimator ranks the outcome reasonably well.

Champion residual (prediction minus actual) on the same rows:

| | Not Minimum | Minimum |
|---|---:|---:|
| Last contract | -$1.37M (n=24) | +$1.45M (n=52) |
| Not last contract | -$1.35M (n=63) | +$0.92M (n=19) |

Among paid signings, a last contract shows no discount at all.

### Oracle arms (diagnostic)

These arms use the same pipeline and seeds as the pre-registered arm, with the
realized label in place of the hazard. Rows with an unknown label are not
pulled, so LeBron James 2026 and Chris Paul 2024 and 2025 stay at the
champion. A1 on the known-label rows covers the 558 frame rows from seasons
2019 to 2023.

| Arm | Label = 1 | paired dSel t | C2 worst | Moved Minimum rows | Moved other rows | A1 known-label rows |
|---|---:|---:|---:|---|---|---|
| `--oracle last` | 76 | -0.43 | +$0.382M, FAIL | 31 of 31 better, +$56.9M | 3 of 19 better, -$22.1M | 0.8288 -> 0.8351 |
| `--oracle last_rich` (earnings >= P75) | 45 | -0.50 | +$0.379M, FAIL | 18 of 18 better, +$37.3M | 3 of 12 better, -$4.4M | 0.8288 -> 0.8363 |

In both arms delta stays inside the grid, with a mean near 0.75. The
last-contract signings that received ordinary pay are damaged, including
Malcolm Brogdon 2023, Paul Millsap 2020, Patrick Beverley 2022, and Robert
Covington 2023. The largest gains are buyout cases, Kemba Walker 2021
(+$16.1M) and Hassan Whiteside 2020. These belong to the waiver work, not to
ring chasing.

## Decision

The failure comes from the concept, not from the estimator:

- Perfect knowledge of the last contract fails the gate.
- Adding the earnings condition also fails.
- In seasons 2019 to 2023, a last contract predicts a discount only when the
  signing ends at the minimum. That is route information, not horizon
  information.

Reject every ring-chasing shape on this frame. Do not try another pull shape,
gate, or hazard on the same rows.

The star case cannot be tested yet. Every high-earning veteran with a
reported discount signed in 2024 or later, and none of their labels is
known: LeBron James 2026, Chris Paul 2025, James Harden 2025, Al Horford 2025,
and Kevin Durant 2026.

Reopen only when those horizons are observed. Then run
`python scripts/eval_ringchase_gated.py --seeds 10 --gate continuous --oracle last_rich`
and require the oracle to pass before building any ex-ante gate.

## Artifacts

- `scripts/eval_ringchase_gated.py`
- `scripts/build_retirement_hazard.py`, `data/processed/retirement_hazard.csv`
- `outputs/models/ringchase_gated_continuous_{oof.csv,eval.json}`
- `outputs/models/ringchase_gated_continuous_oracle_last_{oof.csv,eval.json}`
- `outputs/models/ringchase_gated_continuous_oracle_last_rich_{oof.csv,eval.json}`
