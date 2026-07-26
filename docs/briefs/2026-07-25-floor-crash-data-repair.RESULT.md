# RESULT — Floor-crash impact join repair

Date: 2026-07-25

## Final scope

This result adopts only the impact identity/workload repair. A proposed
buyout/waiver/prepriced exclusion was rejected after target semantics were
re-examined and has been fully removed. The evaluation frame remains **944
rows**.

A signing team's incremental salary after a buyout is the contract price this
project predicts. Old-team guaranteed compensation is context, not a reason to
delete the new contract.

Kendrick Nunn 2020 is a separate continuation-data defect, recorded as
ISSUES #31. It must be fixed through the common contract-span/year-in-contract
machinery, not an invalid-observation list.

## Join diagnosis

Two source failures overlap:

1. DARKO, LEBRON and LAKER sometimes spell one player differently. Rows sharing
   stable `(nba_id, season)` were left split by the old name-season merge.
2. LAKER history genuinely omits some regular-season rows. Those require narrow,
   sourced workload corrections; an alias cannot recover absent source data.

`src/features/impact_identity.py` now coalesces source rows by
`(nba_id, season)`, preferring the spelling that joins the salary row. It also
fills missing age from the player's invariant `age - season` history, never
from the league median. `impact_metric_corrections.csv` carries sourced
age/games/mpg corrections for the omissions that remain.

Patrick Beverley's prior pay is separately corrected to the $13M contractual
base in `salary_corrections.csv`; the old table retained only a $506,508
rest-of-season payment.

## Five reviewed rows

| Player-season | age | mpg | availability | prior pay |
|---|---:|---:|---:|---:|
| Montrezl Harrell 2023 | 29 | 11.9 | 0.6841 | $2.46M |
| DeMarcus Cousins 2020 | 29 | 0.0 | 0.1372 | $3.50M |
| Bol Bol 2023 | 23 | 21.5 | 0.5561 | $2.20M |
| Malik Beasley 2023 | 26 | 25.8 | 0.8732 | $15.56M |
| Patrick Beverley 2023 | 34 | 27.1 | 0.7110 | $13.00M |

Age follows the project's priced-season convention, hence Cousins/Beverley
29/34 rather than signing-date ages 30/35.

Undefined LAKER rate statistics such as usage, AST% and RAPM remain missing and
use the ordinary model imputation. No unsupported values were invented.

## Evaluation

The row set and player grouping are unchanged, so the old and new 944-row suite
folds pair directly over 10 seeds.

| Metric | old | repaired |
|---|---:|---:|
| A1 R2 | 0.7989 | **0.8073** |
| A1 MAE | $2.916M | **$2.874M** |
| A2 R2 | 0.8515 | **0.8579** |
| A2 MAE | $3.114M | **$3.079M** |
| B1 R2 | 0.8356 | **0.8406** |
| B1 MAE | $3.221M | **$3.195M** |

Selection-pool paired A1 delta is **+0.00989 +/- 0.00447, t=2.21**.
Per-fold deltas are
`[+0.00241, +0.00053, +0.02312, +0.01779, +0.00560]`: all five are positive.
Pooled paired delta is **+0.00921 +/- 0.00405, t=2.27**.

The five rows' OOF absolute-error changes are:

| Player-season | old | repaired | change |
|---|---:|---:|---:|
| Montrezl Harrell 2023 | $7.60M | $2.62M | -$4.98M |
| DeMarcus Cousins 2020 | $5.82M | $1.04M | -$4.78M |
| Bol Bol 2023 | $5.64M | $4.43M | -$1.21M |
| Malik Beasley 2023 | $5.52M | $7.08M | +$1.56M |
| Patrick Beverley 2023 | $5.15M | $5.29M | +$0.14M |

Correctness does not depend on every repaired row improving its residual.
Beasley's observed 25.8-minute role must not be replaced by a favorable median.

## Verification

- evaluation frame restored to 944 rows;
- no `invalid_observation`, `observation_exclusions` or `prepriced`
  implementation remains;
- three focused identity/workload tests pass;
- compilation passes;
- full 10-seed evaluation suite completed and rewrote the 944-row OOF reference.

The repair moves published numbers and should take the next version when
committed and tagged. No commit or tag is made here.
