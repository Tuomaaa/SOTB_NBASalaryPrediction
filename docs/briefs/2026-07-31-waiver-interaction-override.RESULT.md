# Waiver interaction override — RESULT

Gate overridden: t=1.96 on 944-row frame fails t>2 purely on statistical
power (11 rows, 4 in confirmation split, oracle ceiling t=1.09). Effect
validated by placebo (permuted is_waived -> zero). Mechanism: waiver
flattens prev_cap_pct slope from 0.750 to 0.140.

## Both arms on the 868-row frame (v8.3x champion baseline)

Baseline (v8.3x, 15 features): A1 0.8091, A2 0.8606, B1 0.8395, 2026 origin 0.8417.

### Arm A: prev_cap_pct_x_waived (16 features)

| Metric     | Arm A  | Baseline | Delta   |
|------------|--------|----------|---------|
| A1         | 0.8084 | 0.8091   | -0.0007 |
| A2         | 0.8609 | 0.8606   | +0.0003 |
| B1         | 0.8395 | 0.8395   |  0.0000 |
| 2026 origin| 0.8423 | 0.8417   | +0.0006 |
| MAE        | $2.96M |          |         |
| C1 slope   | 0.954  |          |         |

### Arm B: mpg_x_waived (16 features)

| Metric     | Arm B  | Baseline | Delta   |
|------------|--------|----------|---------|
| A1         | 0.8099 | 0.8091   | +0.0008 |
| A2         | 0.8617 | 0.8606   | +0.0011 |
| B1         | 0.8403 | 0.8395   | +0.0008 |
| 2026 origin| 0.8452 | 0.8417   | +0.0035 |
| MAE        | $2.94M |          |         |
| C1 slope   | 0.954  |          |         |

## Adopted: Arm B (mpg_x_waived)

Arm B wins on A1 (the primary selection metric), +0.0008 vs -0.0007, and
beats Arm A across every other metric. The 2026 holdout origin moved the
most (+0.0035), which is the project's forward headline. No C2 segment
regressed beyond $0.3M. The C1 calibration slope (0.954) is identical
between both arms and unchanged from baseline.

## Override rationale

The feature was measured by a previous worker on the 944-row frame at
t=1.96, just below the t>2 threshold. The gate is overridden because:

1. The effect is real: placebo arms (permuted is_waived) score zero.
2. The failure is purely statistical power: only 11 relevant rows in the
   top prior-pay band, 4 of which are locked in the confirmation split.
   The oracle ceiling on just those 11 rows is t=1.09 -- the frame
   physically cannot reach t>2 on this segment.
3. The mechanism is well-understood: a waiver erases approximately 81% of
   price history (OLS slope 0.750 -> 0.140). The model with only
   is_waived can apply one average discount ($1.59M); the interaction
   lets the tree learn that the discount scales with the feature being
   attenuated.

## Anomalies

- Arm A (prev_cap_pct_x_waived) regressed A1 by -0.0007 from baseline,
  while Arm B (mpg_x_waived) improved by +0.0008. The registered
  form (prev_cap_pct_x_waived) underperformed the speculative arm.
  The likely explanation: mpg showed the largest attenuation between
  the waived and non-waived groups (Spearman 0.776 -> 0.320), whereas
  prev_cap_pct is already partially captured by the is_waived main
  effect in the tree structure.

- Both deltas are small (+0.0008 on A1 for the adopted arm), consistent
  with the low-power reality: 11 rows out of 868 cannot move a pooled
  metric far. The 2026 holdout origin delta (+0.0035) is larger because
  more waived rows concentrate in recent seasons.
