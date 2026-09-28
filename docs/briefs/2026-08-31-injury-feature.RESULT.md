# Spotrac injury feature result

## Change and rationale

The experiment parsed cached Spotrac injury/DNP tables into 11,697 dated events
for 761 players. It tested fixed ex-ante scores based on log games missed,
one-year or two-year time decay, injury-family severity, and same-family
recurrence. An event enters only after its first missed-game date. An event
still open at the signing cutoff receives one observed game, because its final
games-missed total would use future information.

The feature is rejected. The Spotrac table largely recompresses information
already present in `availability_3yr` and `mpg`, and 745 of 870 evaluation rows
have positive burden. More importantly, Spotrac records missed-game intervals
rather than diagnosis dates. Damian Lillard's Achilles row starts on October
22, 2025, after his July 19 signing, although the tear occurred during the
prior playoffs. The source therefore misses the event that motivated the task.

## Directional paired screen

This is a three-seed, five-fold directional screen on the 870-row evaluation
frame. The user stopped the 10-seed formal gate after the screen was uniformly
negative. These values must not be reported as a completed formal gate.

| Arm | Paired Delta selection R-squared | t | Per-fold delta |
|---|---:|---:|---|
| One-year time decay | -0.00201 | -3.07 | -0.00062, -0.00386, -0.00217, -0.00047, -0.00292 |
| Two-year time decay | -0.00190 | -2.88 | -0.00078, -0.00269, -0.00121, -0.00070, -0.00411 |
| Type weighted | -0.00173 | -2.38 | -0.00029, -0.00300, -0.00066, -0.00078, -0.00391 |
| Type plus recurrence | -0.00239 | -3.20 | -0.00105, -0.00359, -0.00079, -0.00187, -0.00465 |
| Games observed | -0.00180 | -2.18 | -0.00002, -0.00255, -0.00080, -0.00096, -0.00466 |
| Event count | -0.00172 | -2.97 | -0.00132, -0.00322, -0.00077, -0.00034, -0.00295 |
| Coverage control | -0.00018 | -0.22 | +0.00163, -0.00262, -0.00098, +0.00155, -0.00048 |
| Season control | -0.00096 | -2.22 | -0.00086, -0.00252, -0.00108, -0.00020, -0.00014 |

The recurrence arm is negative in every fold. Coverage is approximately flat,
so missing-page coverage does not explain the loss. Type weighting reduces the
damage relative to recurrence weighting but does not add independent signal.

## Data audit

- URL-table players: 923.
- Players with a cached mapped page: 829.
- Players with at least one parsed event: 761.
- Parsed events: 11,697.
- Evaluation rows with a matched signing date: 786 of 870.
- Evaluation rows with positive ex-ante injury burden: 745 of 870.
- Rows with an unparseable `TBD` end date: 34. These do not enter a score that
  requires both dates.

## Decision

Reject the feature and remove it from the active queue. Do not add any injury
score to `FEATURE_COLS`. Reopen only if a source supplies diagnosis date,
surgery or season-ending status, expected return, and signing-time recovery
status. Reweighting the current missed-game table is not new information.

## Reproducibility

- `src/features/injury_history.py`
- `scripts/build_injury_features.py`
- `scripts/eval_injury_feature.py`
- `tests/test_injury_history.py`
- `data/processed/spotrac_injuries.csv`

Commands:

```bash
python scripts/build_injury_features.py
python -m unittest tests.test_injury_history -v
OMP_NUM_THREADS=6 python scripts/eval_injury_feature.py --screen --seeds 3
```

Proposed commit message: `Reject Spotrac injury-history feature`
