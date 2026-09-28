# Public accuracy views result

## Change

Added `scripts/public_accuracy_views.py`, a reporting-only view of the v8.16x
champion's out-of-sample predictions for readers outside the project. It reads
`outputs/models/oof_reference.csv` from `src/model/evaluate_suite.py`, so A1
values are out-of-fold and B1 values are rolling-forward. It never reads the
`export_web` refit, whose predictions are in-sample. No model code or
published number changes.

The suite rerun on 2026-09-28 reproduced v8.16x: A1 0.8423, A2 0.8587, B1
0.8255 (2024 / 2025 / 2026: 0.8607 / 0.8108 / 0.7929). `VERSION_HISTORY.md`
records A2 0.8576 and B1 0.8259; the difference is within seed noise.

## Evidence

All values are B1 forward, 2024-26, n=326, unless stated.

### Error tiers

| Tier | Dollar form | Cap-point form |
|---|---|---|
| Near exact | 29.1% within $0.5M | 34.4% within 0.5 pts |
| Close | 53.1% within $2M; median error $1.80M | 64.1% within 2 pts |
| 90% coverage | 90.2% within $8M | q90 = 5.00 pts |

A cap point is one percentage point of that season's cap: $1.65M in 2026.

- 14.4% of rows (47) have zero error. 23 are Minimum and 24 are Bird Rights
  maxes; Stage 2 clips them to a CBA bound. Use the $0.5M tier as "near exact",
  not the zero-error share.
- Non-Minimum rows (n=232): 40.1% within 1 pt, 72.8% within 3 pts, 88.4%
  within 5 pts. The tiers do not depend on minimum contracts.
- The 10-point tier (98.2%) equals about $16.5M and is too loose to publish.

### Relative error is unsuitable as the headline

| Row set | within 20% | within 30% |
|---|---:|---:|
| All signings | 48.8% | 56.4% |
| Salary >= $10M (n=124) | 67.7% | 77.4% |

Relative error divides by small minimum salaries and understates accuracy on
the 249 floor-zone rows. Use it only for contracts of $10M or more.

### Trim curve

R-squared after removing the k rows with the largest champion error. The
removed rows are fixed by the champion and are named in
`public_accuracy_misses.csv`.

| k | A1 | B1 | B1 2026 |
|---:|---:|---:|---:|
| 0 | 0.842 | 0.826 | 0.793 |
| 1 | 0.852 | 0.852 | 0.885 |
| 3 | 0.861 | 0.880 | 0.900 |
| 5 | 0.866 | 0.891 | 0.918 |
| 10 | 0.877 | 0.904 | 0.943 |

### Largest forward misses

| Row | Actual | Forward prediction | B1 R-squared cost |
|---|---:|---:|---:|
| LeBron James 2026, Minimum, PHI | $2.45M | $43.43M | 0.026 |
| Damian Lillard 2025, MLE, after buyout | $14.10M | $51.31M | 0.025 |
| Trae Young 2026, Bird Rights | $49.49M | $29.61M | 0.006 |

LeBron James 2026 alone moves the 2026 origin from 0.885 to 0.793. It
accounts for most of the drop from v8.14x's 0.8971. The two frames have
different rows, so this is not an exact attribution. LeBron was admitted by
the v8.16x Spotrac backfill and was not in the v8.14x frame. Lillard 2025 is
the main cost in the 2025 origin.

## Decision

- Publish dollar tiers as the headline: within $0.5M, within $2M, and within
  $8M. Add cap points as the secondary unit.
- Publish the unmodified R-squared first, with the trim curve beside it.
- Name every trimmed row, with a one-line reason for each.
- Keep these views out of accept/reject decisions.
- The two largest forward misses are the test cases for the ring-chasing and
  waiver queue items. Report both rows by name when evaluating those items.

## Artifacts

- `scripts/public_accuracy_views.py`
- `outputs/diagnostics/public_accuracy_views.csv`: all metrics for A1, B1,
  and B1 2026
- `outputs/diagnostics/public_accuracy_misses.csv`: top 10 misses per layer
- `outputs/diagnostics/public_accuracy_views.png`: B1 hit-rate curves

Reproduce:

```text
OMP_NUM_THREADS=4 python src/model/evaluate_suite.py
python scripts/public_accuracy_views.py
```
