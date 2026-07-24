# Task brief: per-route corrections (δ) for the continuous routes

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-26-route-delta.RESULT.md`. Two other workers may be out
(service years #20, route-mixture phase 3) — do not pull mid-task; the
architect re-runs at landing if the frame moved.

## The gap

The unified architecture is
`Σ_k P(route k) × value_k`. The structural routes (max / floor / MLE) get
CBA constants as their `value_k`. The **continuous routes — Bird-Rights
re-signings and cap-space signings — currently share one number**: the single
regression `f(x)`. Nothing in the model separates them, and the route label
cannot be a feature (it is partly determined by the contract; fed as an input
it scored −0.0073, METHODOLOGY's ablation table).

The cost is visible and it is the largest standing entry in the C2 table:
Bird Rights bias **−$2.5M** (n≈274) against Cap Space **−$0.2M** (n≈76). The
retention premium is real and is currently smeared across everyone.

This brief builds `value_k = f(x) + δ_k` for the continuous routes and tests
it both ex-ante and ex-post.

## Part 1 — split the continuous class and measure P(route)

`route_mixture.compute_route_labels` currently emits one `continuous` class.
Split it by signing mechanism into **bird**, **capspace**, and **other**
(Early Bird / Non-Bird / Sign-and-Trade / Unknown — keep them pooled unless
a split is obviously warranted, and say why). Labels come from
`signing_cat`; state its coverage (~10% Unknown, ISSUES #4) and how you
handle Unknown rows — they are a class, not a missing value.

Retrain the classifier with the expanded class set and report per-class AUC,
calibration, and the P distributions. **P(bird) is the number that decides
whether the ex-ante form is viable at all** — incumbency is not in
`FEATURE_COLS` (no team continuity, ISSUES #5), so P(bird) may be weak in the
same way P(mle) was (AUC 0.7158). Report it before building anything on it.

## Part 2 — estimate δ fold-honestly

**δ_k is the mean OOF RESIDUAL of route k, never a mean salary difference.**
The raw gap (Bird extensions average 0.180 cap_pct vs re-signings 0.130) is
mostly composition — extensions go to better players — and `f(x)` already
prices that. Only the residual is the premium.

For each fold: fit the champion on the training folds, compute OOF residuals
on those same training folds (nested, or use the training folds' own
cross-fitted residuals — state your scheme), average by route, apply to the
held-out fold. The held-out rows' own residuals must never enter their δ.

Report δ per route with a standard error and the row count. Also report the
**extension vs re-sign split inside Bird** (from
`contract_signing_dates.csv`, `is_extension`; ISSUES #4 asks for it): two
δs or one, decided by whether their residual means differ by more than their
SEs.

## Part 3 — two evaluation modes, reported separately

**(a) Ex-ante (the gated claim).** Output
`Σ_k P(k) × (f(x) + δ_k)` over the continuous routes, structural routes left
exactly as the champion handles them today (no max/floor/MLE branch — that is
phase 3's business, not yours). Full challenger battery on fixed rows:
paired ΔSel t > 2; A2 same direction; C2 fixed-segment |bias| growth
≤ $0.3M **and report whether Bird's −$2.5M shrinks**, which is the point of
the exercise; B1 forward drop ≤ 0.003; calibration by predicted band.

**(b) Ex-post (the oracle ceiling, NOT a gate).** Output `f(x) + δ_route`
using the TRUE route. This is the "told parameters" mode — legitimate as a
product (given that he re-signed with Bird rights, what is the price?) but
it reads a label that is partly contract-determined, so it is an upper bound
on (a), never a headline. Report it as the ceiling so the architect can see
how much of the prize the ex-ante form actually collects.

If (a) fails but (b) is large, that is a useful result: it localises the loss
to P(route) and points at the team-continuity data gap rather than at the δ
idea.

## Traps

- δ from residuals, not from salary means. A composition artifact here would
  look like a big win and be nothing.
- No route label enters `FEATURE_COLS`, and no held-out row's residual
  enters its own δ. State both explicitly in the RESULT.
- Deltas above +0.01 are a re-verify flag.
- `signing_cat` has known mislabels (six >$6M "Minimum" rows, ISSUES #4) —
  they will land in whatever class the label says; report how many
  high-salary rows sit in implausible classes rather than hand-fixing them.
- Do not touch the max/floor/MLE handling, sigma, gates, or the filter chain.

## Deliverable

Branch + RESULT: per-class classifier diagnostics (P(bird) first), the δ
table with SEs and the extension/re-sign question answered, the ex-ante
battery, the ex-post ceiling, and a recommendation. Expected effort: a day.
