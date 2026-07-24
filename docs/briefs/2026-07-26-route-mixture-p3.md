# Task brief: route-mixture phase 3 — re-judge the max branch on corrected labels

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master` (v7.12x line: 944 rows, ISSUES #19 ceilings fixed,
max zone n=68). Deliver a branch +
`docs/briefs/2026-07-26-route-mixture-p3.RESULT.md`.

## Why this is being re-opened (read before anything else)

Phase 2 (`2026-07-25-route-mixture-p2.RESULT.md`) concluded the gated max
branch is **structurally** unwinnable: purity and winnable-error looked
anti-correlated (Spearman −0.681), the 90%-purity point sat at τ=0.90 with
only 8 touchable maxes and an honest ceiling of $0.02M.

**That verdict was an artifact of the ISSUES #19 mis-tiering bug**, which the
worker's pin predates. On the corrected labels the architect re-ran the
worker's own harness (`scripts/eval_route_mixture_p2.py`, unmodified):

| | phase-2 pin (buggy) | corrected (v7.12x) |
|---|---|---|
| 90%-purity τ | 0.90 | **0.64** |
| touchable true maxes | 8 / 56 | **36 / 68** |
| honest win ceiling | $0.02M | **$0.85M** |
| realized true-max ΔMAE | −$0.01M | **−$0.83M** (bar: −$0.50M) |

So the Win gate now PASSES and the branch fails on the brakes instead:
25%+ predicted band +$1.08M and counterweight +$0.36M (bar +$0.30M each),
with C2 (+0.27M) and B1 (−0.0027) squeaking through. Collateral is only 4
rows and the damage is concentrated: **Aldridge 2019 +$10.94M (P 0.862),
Reaves 2026 +$9.71M (P 0.739)**, Brunson 2025 +$2.50M (P 0.866), Jaylen
Brown 2020 +$0.00M (P 0.880).

Phase 2's other conclusions stand and are NOT to be re-litigated: the
feature-batch enrichment lowers P(max) AUC (0.9653 → 0.9557) — treat the
enriched set as a *variant*, not the default; and the machinery
(`route_mixture.py`, flag default off, champion bit-identical) is landed.

## Part 1 — the τ rule, re-registered on corrected labels

The old rule ("smallest τ with purity ≥ 90%") was written when purity was
scarce; on corrected labels purity ≥ 90% holds over a wide τ range, so
"smallest" now buys coverage at the price of collateral. **Re-register
before running any zone metric:**

> τ* = the smallest τ whose OOF purity is **1.000** (no non-max above it),
> τ ≤ 0.95. Fall back to the highest-purity τ if 1.000 is unattainable.

This is still a probability-space rule — it never reads a zone metric — so
it respects the standing prohibition. On the corrected enriched curve τ=0.90
gives purity 1.000 with 24 true maxes; recompute it yourself for both the
base and enriched classifiers rather than trusting these numbers.

Report both operating points for each classifier: **τ\*** (zero-collateral)
and **τ₉₀** (the old rule), with each one's honest win ceiling computed
BEFORE the run.

## Part 2 — run the gated branch at both operating points

Four cells: {base, enriched} × {τ*, τ₉₀}, ship form push_clip_gated, plus
hard_gated for the winning cell only. Full battery on FIXED v7.12x rows
(recompute references from the current suite first — the max zone is n=68
now, MAE $4.44M):

1. **Win**: true-max zone MAE improves ≥ $0.50M.
2. **Brakes**: counterweight band signed-bias growth ≤ +$0.30M; 25%+
   predicted-band bias growth ≤ +$0.30M.
3. **Guards**: ΔSel t > −2; C2 fixed-segment |bias| growth ≤ $0.30M;
   B1 forward drop ≤ 0.003.
4. Collateral list for every cell (non-max rows above τ with push damage).

If a cell passes everything, that is the v8.0 candidate — report it; the
architect tags.

## Part 3 — if τ* passes, the marginal-coverage question

A zero-collateral τ* leaves the biggest max errors (Trae '26, JJJ '22/'26,
Turner '22, MPJ '22 — all P < 0.1) untouched, so the win comes from the
middle of the zone. Report the touched-row table (P, champion error, push
damage or gain per row) so the architect can see exactly which rows the
adoption would be buying. Do NOT tune τ to chase them.

## Traps

- τ from purity only. Never from zone MAE — three experiments have died on
  that altar.
- Aldridge 2019 and Reaves 2026 are the known brake-killers. Do NOT
  special-case them; if τ* excludes them, that is the rule working. If a
  passing cell still includes them, say so plainly.
- Recompute every reference on the v7.12x frame; phase-2's reference numbers
  (zone n=56, MAE $5.47M) are stale by construction.
- The enrichment lowers AUC — if base beats enriched at τ*, prefer base and
  say so; fewer moving parts.
- Flag stays default-off; adoption and the tag are the architect's.

## Deliverable

Branch + RESULT: the recomputed purity curves (both classifiers), τ* and
τ₉₀ with honest ceilings, the four-cell battery table with per-fold deltas,
collateral lists, the touched-row table, and a plain adopt / do-not-adopt
recommendation. Expected effort: half a day.
