# RESULT: ISSUES.md cleanup — duplicate numbers, fixed entries, stale titles

**Brief**: `docs/briefs/2026-07-28-issues-cleanup.md`
**Pin**: `master` (9813a0f)
**Branch**: `worker/issues-cleanup` (worktree `../BBall-worker-issues-cleanup`)
**Files touched**: `ISSUES.md` and this RESULT. Nothing else — no full renumber,
so the sanctioned cross-reference exception was not used. `master` advanced two
commits (`5fb89b1`, `7c317e9`) while I worked; neither touched `ISSUES.md`, so
the branch merges clean.

## What changed

Four entries deleted after running each one's own Verify block (#3, #18, the
`age − 19` #20, #21), one rewritten down to its residual (#17), one renumbered
to break the `#25` collision (Spotrac totals → **#28**), and the entries sorted
into monotonic numeric order. **A full renumber was declined**: 198
`ISSUES #`-style references exist across 47 files, and roughly half of them
name numbers that are already retired (#7–#11, #13, #15, #16, #19, #22, #23),
so the number space is already historical. Renumbering survivors would collide
with meanings the archive has fixed in place.

The header now carries the convention that would have prevented both
collisions — numbers are permanent, never reused, a new entry takes `max + 1`
over every number ever used (next is **29**) — and the "ordered roughly by how
much damage each one does" claim, which the file contradicted, is replaced by
numeric ordering with the reason stated.

## Per-entry verdicts

| # | Entry | Verdict | Action |
|---|---|---|---|
| 2 | Continuation rows, the part left in | open (deliberate) | keep |
| 3 | Single-seed vs 10-seed CV R² | **fixed** | deleted |
| 4 | Signing-mechanism residual 10% | open | keep |
| 5 | Team/position missing 5.4% | open | keep |
| 6 | Boundary rules on curated tables | open | keep |
| 12 | AAV coverage, non-random | open (informational) | keep |
| 17 | 2018 prior salary coverage | **partly fixed** | rewritten |
| 20 | Three zone-gate protocol defects | open | keep + status line |
| 20 | `age − 19` experience fallback | **fixed** | deleted |
| 21 | Veteran-extension raise caps | **fixed** | deleted |
| 24 | Two CBA boundaries, both right | open | keep, moved into order |
| 25 | Spotrac totals vs our schedules | open | **renumbered → 28** |
| 25 | τ-objective collateral term | open | keeps 25 |
| 26 | Stored `prev_cap_pct` stale | open, in flight | keep |
| 27 | Smart 2022 supermax ceiling | open, in flight | keep |

### Verification output for each deletion

**#3 — train.py writes the 10-seed figure.** `train.py:1046-1061`:

```python
    # The canonical CV R² is the 10-seed average (same protocol as
    # evaluate_suite.py). Write it into grabit_results.json so the
    # file's headline matches evaluation_suite.json's A1.
    canon = _multiseed_grabit_cv(df, sigma=0.02)
    ...
    with open(model_dir / "grabit_results.json", "w") as f:
        json.dump(canon, f, indent=2)
```

`_multiseed_grabit_cv` writes `cv_r2_mean` from the 10-seed average, keeps the
single-seed value under the distinct name `cv_r2_single_seed`, and records
`n_seeds`. The entry's own Fix is done and it explicitly scoped itself to
`train.py`'s output ("`export_web.py` no longer reads that file … this is now
only about keeping `train.py`'s own output honest"). One residual noted below.

**#18 — the awards name join.**

```
  norm('Kevin Durant^'       ) = 'kevin durant'
  norm('Bam Adebayost1'      ) = 'bam adebayo'
  norm('Ben Simmonscovid2'   ) = 'ben simmons'
  norm('Nikola Jokić^'       ) = 'nikola jokic'

player_name_norm  season  award_score_cum  all_nba_cum
       ja morant    2021             3.00          0.0
       ja morant    2023             6.95          1.0
     luka doncic    2021             6.55          1.0
     luka doncic    2025            12.00          5.0
    kevin durant    2019            21.35          2.0
```

Both halves of the Verify block pass: `norm("Kevin Durant^") == "kevin durant"`,
and `award_score_cum > 0` for Luka and Ja in their post-rookie seasons.

**#20 (`age − 19`) — real service years.**

```
  service years: 1296 from debut, 1 age-19 fallback
  frame rows = 944
player_name_norm  season  cap_pct  tier_ceiling_pct  is_max_contract
   austin reaves    2026     0.25              0.25             True
    jimmy butler    2019     0.30              0.30             True
  max zone (is_max_contract) = 70
  over-tier count (cap_pct > max_eligible_pct) = 0     [CAP_TOL = 1e-4]
  debut-season table entries = 5367
```

Every clause of the Verify block passes: Reaves 2026 at 0.25 and True, Butler
2019 at 0.30 and True, max zone 68 → 70, over-tier 0, frame 944. The fallback
that fired on 223 of 944 rows now fires on **1 of 1297**.

**#21 — the raise caps are implemented.** `src/model/extension_cap.py` exists
and runs; the hard gate is clean:

```
Extension cap: 156 first-year extension rows (73 rookie-scale, 83 veteran,
                                              13 designated-veteran)
  pay-above-base allowances applied: dejounte murray 2024 +$4.02M,
                                     marcus smart 2022 +$0.25M
  rows with a computed extension cap: 156
  rows at their cap within 0.5%:      63
  over_cap_rows(CAP_TOL=1e-4):        0
  p.j. washington 2026: cap_pct 0.1201  ext_cap_pct 0.1201
```

P.J. Washington 2026 — the entry's $29.7M headline row — now reads a ceiling
equal to his pay to four decimals.

### Delete or rewrite #21 — deleted, and why

The brief left this open. **Deleted.** Every thread the entry opened is closed,
not merely retitled:

- *Treatment 1, ceiling correction*: implemented as `ext_cap_pct`, and its
  non-entry into `max_eligible_pct` is a **decision**, not an omission — the
  cap binds only conditional on choosing to extend, so it is a told-parameter
  correction. It is the Stage-3 told-route clip landed at 4ae4f63, and the
  wire-stage3 worker is adopting it now.
- *Treatment 2, Stage-1 censoring*: adjudicated in the extension-route brief —
  these rows are **not** right-censored, the same "choice, not constraint"
  that killed censoring good players on minimums.
- *The counterweight-band hypothesis* ("the brake that closed the route-mixture
  line is measuring a data bug"): **refuted**, per
  `2026-07-26-extension-route.RESULT.md` §"The counterweight-band hypothesis is
  refuted".

Rewriting it would leave an entry whose whole content is "this landed" — a
changelog line in a worklist. The knowledge survives in `extension_cap.py`'s
docstrings, the extension-route RESULT, and VERSION_HISTORY.

### Why #17 was rewritten rather than deleted

The 2026-07-26 cleanup-debt RESULT recommended deleting it alongside #3 and
#18, and its Verify block does pass. But only the alias half landed, and the
residual is real:

```
  season-2019 evaluation rows: 142
  with a 2018 prior salary:    138          [was 136/142]
  still missing (4): ['alex caruso', 'amile jefferson', 'shake milton',
                      'wenyen gabriel']
  ('enes freedom', 2018) -> 0.18922410154217673
  ('wes iwundu',   2018) -> 0.01352955266077
```

Four two-way players still have no 2018 salary, an alias cannot reach them, and
`docs/QUEUE.md` item 4 explicitly parks "#17's remaining two-way players".
Deleting would both dangle that QUEUE line and claim a coverage the data does
not have. The entry is retitled to what is left and its verify block updated.

## The renumber decision

**198** lines matching `ISSUES #` across **47** files (`.git` excluded), counted
on the pinned `master` before any edit:

| Where | Refs |
|---|---|
| `docs/briefs/*.md` + `*.RESULT.md` (31 files) | 151 |
| `docs/QUEUE.md` | 11 |
| `src/` + `scripts/` code comments (9 files) | 22 |
| `VERSION_HISTORY.md`, `METHODOLOGY.md`, `config.py`, `salary_corrections.csv` | 9 |
| `ISSUES.md` itself | 4 |
| `docs/worker-brief.md` | 1 |

Per-number (token counts, so 199 against 198 lines — one line names two):
#2×17, #3×5, #4×9, #5×9, #6×14, #7×7, #8×16, #9×3, #10×5, #11×10, #12×1, #13×6,
#15×3, #16×4, #17×3, #18×2, #19×18, #20×11, #21×12, #22×17, #23×14, #24×1,
#25×7, #26×1, #27×3.

Eleven of those numbers (#7–#11, #13, #15, #16, #19, #22, #23) belong to entries
already deleted. The archive has therefore already fixed meanings to numbers the
file no longer holds, and a renumber of the survivors would make ~100 dated
references silently wrong rather than merely historical. The brief's own rule
("a stable wrong-looking number beats a broken reference") applies at this count.

**Collision resolution.**

- The two `#20`s resolved themselves: the `age − 19` entry is fixed and deleted,
  and the surviving zone-gate entry is the one the *live code* names
  (`scripts/eval_floor_branch.py:61,224,481` — "ISSUES #20a"). The service-years
  brief's use of #20 for the deleted entry is dated and stays as history.
- The two `#25`s were both legitimate `max + 1` on their own branch: the
  floor-branch worker committed at `c9b3db2` 04:03:35 and the told-clip worker at
  `091521e` 04:06:21, three minutes apart, neither seeing the other. The
  τ-objective entry **keeps 25** — committed first, and 4 external references
  against the Spotrac entry's 3. Spotrac moves to **28**, with a note in the
  entry itself saying so, so the told-clip RESULT's three `#25` mentions resolve
  without editing that archive.

## Final numbering

`2, 4, 5, 6, 12, 17, 20, 24, 25, 26, 27, 28` — monotonic, no duplicates.

**Where the two in-flight workers' entries slot in: 29 and 30**, assigned in the
order the architect lands them. 28 is taken by the Spotrac renumber above, so
neither worker's "next free number" guess will be right if they guessed from the
pre-cleanup file.

**#26 and #27 are still in the file and this is deliberate** — the brief says to
leave them for the ceiling-consistency worker's landing. They are not missed;
the architect deletes them when that worker lands. Nothing about them changed
except position (they were already in numeric order).

## Anomalies

- **Three rows read `cap_pct > max_eligible_pct` at a 1e-9 tolerance** (Giannis
  2021, Adebayo 2021, Tatum 2021, all at exactly 35%/25% of cap). The excess is
  $50–$70 on a $112.4M cap — float round-trip through `cap_pct`, not a ceiling
  defect. At the project's own `CAP_TOL = 1e-4`, and at anything above 1e-6, the
  count is 0. Worth knowing only so the next person to write an over-tier check
  uses the tolerance rather than `> 0`. Not filed.
- **#3's residual, not filed as its own entry**: `scripts/export_web.py:447`
  writes `meta.json`'s `cvR2` from `train_grabit(train_df)` — a single-seed fit
  on the training seasons — so the site's secondary field still carries a
  single-seed number under a name that elsewhere means the 10-seed average. The
  site's *headline* is the forward R², which is cross-checked, so nothing
  published is wrong. I judged this below the bar for an entry given the deleted
  #3 explicitly scoped itself away from it; flagging here so the architect can
  overrule cheaply.

## For the architect — three things outside my lane

1. **`docs/QUEUE.md:143` now dangles** (line 124 at my pin; QUEUE moved under me
   in `5fb89b1`). "Route-mixture max branch — **CLOSURE PROVISIONAL, pending
   ISSUES #21**" points at an entry I deleted. #21 has landed and the phase-3
   re-run happened, so that line's condition is met and the closure can be made
   unconditional. QUEUE is yours; I did not touch it.
2. **`docs/QUEUE.md`** also cites ISSUES #6, #5 and #17, all of which survive —
   no action, noted only so the grep result is not alarming.
3. **`METHODOLOGY.md` still owes #24 its paragraph** (the two CBA boundaries).
   That is the docs agent's lane; #24 stays open until it is written.

## Proposed commit message

```
ISSUES.md: delete four landed entries, break both number collisions

Deleted after running each entry's own Verify block on master: #3 (train.py
writes the 10-seed CV R² into grabit_results.json), #18 (norm() strips
footnote marks; Durant/Luka/Ja carry award mass), #20's age-19 service
fallback (Reaves 2026 at 25%, Butler 2019 at 30%, max zone 70, 1 fallback row
of 1297), and #21 (extension_cap.py computes the 120%/140% raise cap, 0
over-cap rows of 156; the counterweight-band hypothesis it raised is refuted).

#17 rewritten to its residual: the alias half landed (2018 coverage 136 ->
138 of 142), four two-way players remain unreachable without a two-way salary
source.

The two #20s resolve by deletion; the surviving one is the zone-gate entry the
code names. The two #25s were parallel max+1 assignments three minutes apart:
the tau-objective entry keeps 25, Spotrac totals moves to 28 and says so.

A full renumber was declined at 198 cross-references across 47 files, half of
them naming already-retired numbers. Instead the header now states the
convention that prevents recurrence -- numbers are permanent, never reused,
new entries take max+1 over every number ever used (next: 29) -- and entries
are ordered by number, replacing a severity claim the file contradicted.
```

## ISSUES.md additions

None. Everything found was either fixed on `master`, already an entry, or
recorded in the anomalies section above as below the bar.
