# Task brief: documentation catch-up (v7.9x → v7.12x, ISSUES #10 and #15)

Read `docs/worker-brief.md` first — isolation, own worktree. This is the
**docs lane**: you own `VERSION_HISTORY.md`, `METHODOLOGY.md`, `CONTEXT.md`,
`PROJECT_BRIEF.md`, `CLAUDE.md`. You own **no code and no data**. Three other
workers are out in the code lanes; you will not collide with them, but do not
pull mid-task.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-26-docs-catchup.RESULT.md` (short — a list of what you
wrote and where).

## The backlog

`VERSION_HISTORY.md` stops at **v7.8x**. Four versions have landed since, and
two ISSUES entries exist purely to hold documentation debt. Sources of truth,
in priority order: the git tag messages (each carries the headline numbers),
the RESULT bundles under `docs/briefs/`, the landing commit messages, and
`ISSUES.md`. Do not recompute anything — quote the recorded numbers and cite
where they came from.

### VERSION_HISTORY entries to write

- **v7.9x** — continuation filter v2. Dated-span demotion with option-aware
  anchors, extension fallback +1, renegotiation carve-out; frame 1,291 → 949
  (342 demoted, 0 contradicted). Common-row paired A1 −0.0009 (t −0.13) —
  and the protocol lesson that goes with it: a **pure-removal change cannot be
  judged on common-row A1**, which scores only rows both frames keep; the
  pre-registered criterion was the `is2019` contamination control, +0.0035
  (t 2.29) → −0.0004 (t −0.91). Evidence:
  `2026-07-24-continuation-filter-v2.RESULT.md`.
- **v7.10x** — `prev_cap_pct` repair + the 2026 cap correction. Paired ΔSel
  **+0.0135 (t 7.6)**, the largest single move in the project's recent
  history; 2019 prior coverage 28% → 96% via the 2018-19 team-salary scrape;
  2019 segment bias −0.74 → −0.42; B1 0.819 → 0.826. Cap fixed to the
  official 164,961,000 together with five extension-max salaries that were
  stale in lockstep (the reason `check_caps.py` did not flag it — record this
  as the "stale in lockstep" failure mode). Frame 949 → 944. Evidence:
  `2026-07-24-prev-cap-pct-fix.RESULT.md`.
- **v7.11x** — feature batch, one arm of five adopted at the gate boundary:
  `prev_cap_pct` = previous-season pay, ΔSel +0.00178 with **per-fold t 1.96**.
  Record the adoption rationale (pre-registered magnitude, all four
  instruments improving, semantics simplification rather than added capacity)
  AND the protocol correction it exposed: the earlier "t = 7.0" for the same
  comparison was **seed-paired**; seed sd ≈ 0.001 inflates t roughly 3×, and
  the pairing unit is always the fold. Rejected arms and why: trend features,
  prior-year estimated value, rookie-award ordinal all flat (DARKO and the
  existing awards channel already carry those signals); stats-as-of-signing
  actively harmful — the market prices expected growth, so current-season
  production beats signing-date production for extensions, with MPJ 2022 the
  rare inversion. Evidence: `2026-07-25-feature-batch.RESULT.md`.
- **v7.12x** — ISSUES #19: the designated-ceiling award path fired on All-NBA
  + experience alone, blind to the CBA's team-continuity requirement. 12
  genuine maxes mis-tiered; new curated `designated_ineligible.csv`. Max zone
  56 → 68, Grabit zone MAE 6.18 → 4.44, A1 0.7849 → 0.7878 on the same 944
  rows. Evidence: the landing commit `61bdfcd`.

Also add a line to whatever "key milestones" summary exists so the top-level
narrative reaches the current champion.

### METHODOLOGY — ISSUES #10

Two items from the sigma/gate retune
(`2026-07-23-sigma-gate-retune.RESULT.md`):

1. **Never select sigma on zone MAE.** Both censored sides are one-way
   valves: every zone row is biased toward its CBA bound and Stage 2's clip
   makes overshooting free, so zone MAE falls monotonically in sigma out to
   0.06 with no interior optimum while the pinned-at-bound share climbs
   19% → 42% and the bill lands on the calibration slope (0.9883 → 0.9647 at
   σ=0.04). Selected this way, sigma degenerates into "how many rows do you
   want pinned". `gate_frac` is inert on this row set (52-56 of 57 gated
   across 0.45-0.65); the binding constraint is `is_max_contract`'s 0.90
   threshold.
2. **The C1 calibration gate is RELATIVE**: |slope − 1| may exceed the
   incumbent's by at most 0.005 (≈ $0.3M of scale distortion at a $60M max —
   C2's own materiality yardstick). The absolute [0.99, 1.01] window is dead;
   it excluded the champion's own 0.9883. This is already in
   `evaluate_suite.py`'s printed protocol; METHODOLOGY should match.

### METHODOLOGY — ISSUES #15

Three conventions from the continuation-filter work:

1. **Renegotiation convention** (adjudicated 2026-07-24): a
   renegotiation-and-extend re-prices its signing season to market, so that
   season is FRESH even though an older span covers it. Six pairs in the
   data: Turner 2022, Sabonis 2023, Clarkson 2023, Isaac 2024, Markkanen
   2024, JJJ 2025.
2. **Span rules**: Spotrac's `fa_year` on an option-final deal is the
   option-decision summer, so the span is `[fa_year − years + 1, fa_year]`;
   an unmatched extension starts paying at `signing_season + 1`; 120 block
   anchors (6.1%) predate their own signing and are date-resolved. A block
   anchor alone is never a trustworthy span.
3. The pure-removal protocol lesson from v7.9x (above) belongs in the
   evaluation-protocol section, not only in the version entry.

### The standing decisions

`docs/QUEUE.md` carries a "Standing decisions" section — currently one
entry, **the Stage-2 clip never reads the observed salary** (both directions
analysed and rejected: clipping down erases every bargain by construction;
clipping up silences our most sensitive data-error detector and cannot run at
inference). METHODOLOGY is its permanent home. Copy it there faithfully,
including the empirical note that the only over-ceiling rows today are three
float-dust cases at ratio 1.0000.

## What NOT to do

- Do not recompute or re-derive numbers; quote the recorded ones and cite the
  RESULT or tag they came from. If two sources disagree, report the conflict
  in your RESULT rather than picking one.
- Do not edit `ISSUES.md` beyond deleting **#10** and **#15** once their
  content is written into METHODOLOGY (they exist only to hold this debt).
  Leave every other entry alone.
- Do not touch code, data, `scripts/`, or `outputs/`.
- Keep CONTEXT.md's vocabulary rules intact: **Signing Residual** (model
  accuracy, year-1 only) stays distinct from **Contract Surplus** (team
  outcome, any year). If the new material needs a term that does not exist
  yet — "route", "branch", "purity" — add it to CONTEXT.md rather than using
  it undefined.

## Deliverable

Branch + a short RESULT listing what you wrote where, any source conflicts
found, and the ISSUES entries you deleted. Expected effort: half a day.
