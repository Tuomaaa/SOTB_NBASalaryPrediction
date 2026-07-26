# RESULT — Previous-waiver feature

Date: 2026-07-26

## Decision

Ship `is_waived` as the fifteenth model feature. Do not ship
`is_waived_known`; it remains an audit field and the coverage-control arm.

No buyout, waiver, or supplemental-pay observation is removed. The new team's
contract is the market price this project predicts. A recent waiver is
pre-signing context about the player and the contract being exited, not a reason
to invalidate the next observed salary. Kendrick Nunn 2020 remains the separate
continuation defect in ISSUES #31.

## Definition and source

`is_waived = 1` when Spotrac records an explicit waiver or buyout in the fixed
365 days before the signing date that prices the row. Events on or after the
signing are excluded, so the feature cannot read future information. A
same-season signing is used only when no dated contract span covers the row;
priced candidates are matched by AAV and an unpriced fallback uses the latest
dated signing.

The source is the dated transaction history on cached Spotrac player pages.
`scripts/build_waiver_features.py --fetch` fetches only uncached pages, sleeps
three seconds after each live request, stops on failure, and can resume from the
cache. The parsed artifacts contain:

| Artifact | Count |
|---|---:|
| Evaluation-player pages fetched | 511 |
| Dated transactions | 6,849 |
| Explicit waiver/buyout events | 559 |

The complete 3,113-row scoring table has 2,294 source-known rows (73.7%) and 227
positives. The 944-row evaluation frame has 792 known rows (83.9%) and 122
positives. Unknown source/signing coverage stays NaN in the feature table;
`is_waived_known` makes that status explicit.

## Case audit

All seven cases selected from the floor-crash review hit the expected prior
event:

| Player-season | Prior event |
|---|---|
| Andre Drummond 2021 | Cleveland buyout, 2021-03-26 |
| Reggie Jackson 2020 | Detroit buyout, 2020-02-18 |
| Reggie Jackson 2024 | Charlotte buyout, 2024-07-23 |
| Blake Griffin 2021 | Detroit buyout, 2021-03-05 |
| Reggie Bullock 2023 | San Antonio waiver, 2023-10-01 |
| Austin Rivers 2019 | Phoenix waiver, 2018-12-18 |
| Spencer Dinwiddie 2024 | Toronto waiver, 2024-02-08 |

The row-level audit columns retain the exact event date and Spotrac text.

## Evaluation

### Final-data decision pair

After the concurrent player-row missingness repair landed in the same worktree,
the decision pair was rerun on the final 944-row table with identical
player-grouped folds, locked selection pool, and ten seeds.

| Arm | A1 | paired delta, selection | t | delta A2 | delta B1 |
|---|---:|---:|---:|---:|---:|
| incumbent, 14 features | 0.8124 | — | — | — | — |
| `is_waived` | **0.8214** | **+0.00666 +/- 0.00291** | **2.29** | +0.0042 | +0.0157 |

The feature still clears the selection threshold, and A2 and B1 move in the
same direction. Relative calibration excess changes by +0.00031, inside the
+0.005 guard. The worst C2 absolute-bias growth is $0.127M in the Unknown
signing-mechanism slice, below the $0.30M guard. The paired result is saved to
`outputs/models/waiver_feature_current_pair.json`.

### Coverage control

The complete four-arm run predates only that concurrent missingness repair and
uses the same 944 player-season keys. It isolates the Spotrac coverage channel:

| Arm | A1 | paired delta, selection | t |
|---|---:|---:|---:|
| incumbent | 0.8073 | — | — |
| `is_waived` | 0.8180 | +0.00816 +/- 0.00345 | 2.36 |
| coverage only | 0.8077 | +0.00029 +/- 0.00029 | 1.02 |
| waiver + coverage | 0.8191 | +0.00950 +/- 0.00373 | 2.55 |

The coverage-only arm is small and does not clear the gate. The gain is
therefore not explained by which players have a usable Spotrac page. Adding the
coverage flag buys a little more fit, but it encodes source availability rather
than the requested basketball/economic fact, so only `is_waived` ships. The
complete control result is
`outputs/models/waiver_feature_evaluation.json`.

## Integration

The historical feature is attached during the dataset rebuild and reproduced
by `scripts/rebuild_training_data.py`. `predict.py` and `export_web.py`
compute the same 365-day status as of the latest dated transaction in the cached
source; they do not use transactions after that cutoff.

The deterministic rebuild finishes at 3,113 rows and reproduces all stable
columns at 100%, including both waiver columns. The production filter chain
remains 944 rows and `FEATURE_COLS` contains 15 features.

The full current 15-feature evaluation suite also completes: champion A1 is
0.8215, A2 is 0.8632, and B1 is 0.8528. It rewrites the standard 944-row OOF
reference.

Focused tests cover transaction wording, the strict pre-signing window,
unknown coverage, same-season fallback, and the audited join repairs. No commit
or version tag is made from this dirty shared worktree.
