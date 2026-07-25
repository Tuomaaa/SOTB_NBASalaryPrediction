# Task brief: ISSUES #27 and #26 — one wrong fact in two places, one stale feature

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-28-ceiling-consistency.RESULT.md`.

**Two other workers are out.** File ownership is exclusive:

| worker | owns |
|---|---|
| **you** | `train.py`'s `_compute_max_eligible` and the loaders it uses, `data/raw/raw_external/designated_ineligible.csv`, `data/processed/training_data_v2.csv` |
| stage-3 wiring | `extension_cap.py`, `evaluate_suite.py`, `train.py`'s **Stage-2/3 composition**, `export_web.py`, `predict.py` |
| ISSUES cleanup | `ISSUES.md` — **do not touch it**; put your entries in the RESULT |

You and the wiring worker both touch `train.py`. **Your half is
`_compute_max_eligible` and the ceiling loaders; theirs is the prediction
composition.** They do not overlap, but say in your RESULT exactly which
functions you changed so the architect can merge cleanly.

**This change moves the champion**, so it is a version-bump change and its
numbers must be measured, not assumed.

## ISSUES #27 — the same wrong fact, in the ceiling path

ISSUES #23 fixed a Designated-Veteran exemption that Marcus Smart was granted
from a DPOY he won **nine months after** signing his August-2021 extension.
That fix went into `extension_cap._designated_veteran`, which now anchors the
award test on the **signing** season. `train._compute_max_eligible` has the
same supermax branch, has no signing instrument at all, and therefore still
reads Smart 2022 at a **35% tier ceiling, $43.28M**, in both
`max_eligible_pct` and `tier_ceiling_pct`. His real ceiling is 30%.

Direction matters: a too-high ceiling makes the max push aim $25.8M too high
for that row — exactly the damage the extension clip removes downstream. Fix
it at the source.

**Two routes, and the brief wants the second attempted first:**

1. **The one-line patch** — add `marcus smart, 2022` to
   `designated_ineligible.csv`. Fixes both paths immediately, but it is
   another hand-curated row on a list that already needs a new entry every
   summer (ISSUES #6).
2. **The real fix** — give `_compute_max_eligible` the same signing-date
   instrument `extension_cap` already uses, so the award test is anchored on
   the signing season everywhere. **If you take this route you must audit
   every supermax row**, not just Smart: report which rows change, and
   whether any of the 14 designated-veteran rows the extension work
   identified (Lillard 2021, Booker 2024, Mitchell 2025, Giannis 2021,
   Harden 2019, Brown 2024, Tatum 2025, Embiid 2023, Wall 2019, Randle 2022,
   Towns 2024, Smart 2022, Jokic 2023, Gobert 2021) gain or lose eligibility.
   Note the known asymmetry the extension worker documented: the signing
   anchor catches deals signed two summers early (Wall, Harden), while the
   paying anchor catches players whose debut-based service count understates
   them (Embiid). Keep the **service-year** anchor unioned; only the **award**
   anchor moves to the signing season.

Take route 2 if the audit comes out clean. If it moves rows you cannot
justify one by one, fall back to route 1 and say why — a curated row you
understand beats a rule you do not.

**Verify**: Smart 2022 reads `tier_ceiling_pct == 0.30`; no row is paid above
its own ceiling; the frame stays 944; `is_max_contract` count is reported
before and after (Smart's $17.46M is far below either tier, so his flag should
not move — if the count changes at all, explain every row).

## ISSUES #26 — the stored feature is stale

`salary_corrections.csv` is applied inside `_load_prev_season_cap_pct`, which
the ceiling rules read at load time. The `prev_cap_pct` **feature** is baked
into `training_data_v2.csv` by `phase3.build_contract_features`, which was
last run before the corrections existed. So Marcus Smart 2022, Ivica Zubac
2025 and Aaron Gordon 2026 **train on the stale prior** ($13.84M / $11.74M /
$22.84M) while their ceilings use the corrected one. One number, two answers.

Rebuild so the feature agrees. Precedent for how: the prev-cap and
feature-batch workers applied **stage 3 in isolation** rather than the whole
chain, because a full rebuild churns the cosmetic `team`/`agent`/`position`
columns (ISSUES #5's known non-determinism). **Verify that only
`prev_cap_pct` moves**, cell by cell, and report how many rows.

## Measurement

Both fixes move the champion, so measure them together and separately:

- paired ΔSel vs the same-run champion, 10 seeds × 5 GroupKFold folds,
  **per-fold** pairing (the fold is the unit; a seed-paired t inflates ~3×);
- A1 / A2 / B1 including **B1 per origin**;
- C2 fixed segments;
- the max zone's size and MAE before and after.

These are correctness fixes: adopt unless the paired delta is significantly
negative (t < −2), and report the number either way. Do not argue for a gate
change.

## Traps

- Never clamp a ceiling to observed pay (`docs/QUEUE.md`, standing decision) —
  the over-ceiling count is our most sensitive detector of data defects and
  has found three separate bugs this month.
- The ceiling is the Stage-1 censor bound. Moving it changes which rows are
  treated as censored, so `is_max_contract` deserves its own before/after
  line even if you expect no change.
- Do not touch `ISSUES.md`.

## Deliverable

Branch + RESULT: which route you took for #27 and why, the supermax audit if
you took route 2, the rebuild's changed-column verification, the paired
numbers with per-fold values and per-origin forward figures, and the ISSUES
text you would have written. Expected effort: half a day to a day.
