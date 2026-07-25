# Task brief: the extension route — CBA raise caps, then the six-route mixture

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master` (v7.13x line, 944 rows, max zone 70). Deliver a
branch + `docs/briefs/2026-07-26-extension-route.RESULT.md`.

## Why, and what the architect already got wrong twice

ISSUES #21: veteran-extension raise caps (120% pre-2023 CBA, 140% after) are
not implemented, so extension rows carry fresh-signing tier ceilings they
could not legally reach — up to $29.7M too high, $249M of overstatement over
15 rows. The knife-edge evidence is in the issue (Brunson 2025 at exactly
1.400× prior pay, five more exactly at 1.400, six exactly at 1.200).

The architecture this feeds is the user's full six-route mixture:

```
pred = P(max)·V_max + P(floor)·V_floor + P(mle)·V_mle
     + P(bird)·V_bird + P(capspace)·V_cap + P(extension)·V_ext
```

The extension route is the missing sixth. Its absence is why the phase-3 max
branch failed its brakes: Aldridge 2019, Smart 2022 and Brunson 2025 had
nowhere to go but the max route, were pushed at a **tier** ceiling ~$11M
above their legal extension cap, and carried 82% of the collateral damage.

**The architect attempted the headroom measurement twice and was caught both
times by the same sanity check** — rows paid ABOVE their computed cap. Both
failures were in the VALUE function, not the idea:

1. Applying the raise multiple to all 164 extension rows: rookie-scale
   extensions are capped by the max TIER, not by a raise multiple (Jaylen
   Brown 2020 was paid 3.59× his prior salary). Oracle R² collapsed to 0.59.
2. Splitting the two kinds on a `"rookie"` keyword in `tx_text`: it misses
   deals whose text does not say the word (Devin Booker 2019 read 6.86× his
   cap). 44 of 95 "veteran" extensions were still paid above their cap.

A third gap is visible in the same output and is not yet handled: the CBA
raise cap is the **greater of** the multiple applied to prior salary and the
multiple applied to the league's estimated **average salary**. Low-paid
extendees go through the average-salary route (Patrick Beverley 2023: prior
$0.51M, so a prior-only cap says $0.71M against $2.02M actually paid).

So Part 1 is real CBA work, and its sanity check is the gate.

## Part 1 — the extension value function (the bulk of this task)

1. **Classify each extension** as rookie-scale or veteran using a proper
   instrument, NOT text keywords: whether the player's PREVIOUS contract was
   rookie-scale. `_load_rookie_scale_set` in `train.py` already exists for the
   rookie-exit work; use it or state why it does not serve.
2. **Rookie-scale extension**: the cap is the applicable max tier (25%, or 30%
   where the Rose criteria fire) — i.e. the existing tier machinery, unchanged.
3. **Veteran extension**: cap = `max(mult × prior salary, mult × estimated
   average salary)`, with `mult` = 1.20 for seasons ≤ 2022 and 1.40 for
   seasons ≥ 2023 (use the same CBA boundary `cba_era` encodes — confirm it).
   The estimated average salary is a published per-season CBA quantity: curate
   it into `data/raw/raw_external/` with a source URL column, exactly as
   `mle_exception_amounts.csv` was done. Do not infer it from our own data.
4. **HARD GATE, and the reason this brief exists**: after implementation,
   the count of rows **paid above their own computed cap must be ~0** (allow
   the 1e-4 float tolerance the mislabel filter uses). A ceiling below observed
   pay is strictly worse than one above it — it breaks the v7.4x invariant and
   pins predictions under the truth. If you cannot reach ~0, STOP and report
   the residual rows with your diagnosis; do not ship a partial rule.

Then report the **honest headroom** the architect could not compute: how many
rows the champion prices above their corrected cap, and the oracle ΔR² from
snapping exactly those to it. If the headroom is negligible, say so — that is
a complete and useful answer and Part 2 becomes optional.

## Part 2 — the six-route mixture, with the centering discipline

Extend `route_mixture.py`'s class set to six routes (max / floor / mle /
bird / capspace / extension), defined by where the salary landed plus the
dated-extension flag. Value functions: CBA constants for max/floor/mle, the
Part-1 function for extension, `f(x) + δ_k` for the two continuous routes.

**The discipline that the δ experiment's failure teaches** (see
`2026-07-26-route-delta.RESULT.md`): `f(x)` is fit on all routes pooled, so it
already predicts the route-averaged salary. Adding `Σ P(k)·δ_k` on top double
counts — the whole surface lifted $0.3-0.6M, MLE bias +$0.35M, calibration
slope 0.985 → 0.977, and it failed three gates. **Center the route values so
that when P equals the population route frequencies, `Σ P(k)·V_k` reproduces
`f(x)` exactly.** Only a row whose route distribution deviates from the
population average may move. State the centering in the RESULT and verify it
numerically (feed the population frequencies in, check you get f(x) back).

Judged as a challenger: paired ΔSel t > 2 on the selection pool, A2 same
direction, C2 fixed-segment |bias| growth ≤ $0.3M, B1 forward drop ≤ 0.003,
and the relative C1 gate (|slope − 1| excess ≤ 0.005 vs the incumbent). Report
the ex-post variant separately — it is informative but is NOT the headline,
because using the realized route to score is outcome information.

## Part 3 — re-run phase 3

With the extension route in place, re-run `scripts/eval_route_mixture_p3.py`
(unmodified) and report whether the max branch's collateral collapses as
predicted. The architect's closure of that line is **provisional pending this
number** — three of its five collateral rows are extension-capped, carrying
82% of the damage.

## Traps

- The two ceiling directions are not symmetric: too high is a missed
  constraint, too low corrupts the target. Part 1's gate is the guard.
- Do NOT right-censor extension rows. Adjudicated 2026-07-26: the raise cap
  binds only conditional on choosing to extend — the player could have tested
  free agency — which is the same "choice, not constraint" that killed
  right-censoring good players on minimums (METHODOLOGY records the oracle
  experiment). `is_max_contract` and the Stage-1 censor mask stay keyed on the
  unconditional tier ceiling. Brunson 2025 is the worked example: widely
  reported as a deliberate discount.
- P never enters `FEATURE_COLS`; it is an output composition weight only.
- Deltas above +0.01 are a re-verify flag, not a victory lap.

## Deliverable

Branch + RESULT: the extension classification and its counts, the curated
average-salary table with sources, the hard-gate output, the honest headroom,
the six-route mixture numbers with the centering verification, the phase-3
re-run, and a plain adopt / do-not-adopt recommendation. Expected effort: one
to two days.
