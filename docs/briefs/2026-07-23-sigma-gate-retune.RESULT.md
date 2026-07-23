# RESULT: retune sigma and both censoring gates (ISSUES #7)

**Verdict: the incumbent (sigma 0.02 / gate_frac 0.55 / k_floor 2.0) survives.**
No configuration clears both the zone objective and the calibration guardrail.
ISSUES #7 closes as "already optimal" — but the sweep found *why* it is optimal,
and the reason is not that 0.02 prices the censored rows best.

Pinned to `v7.8x` (`c03eb9a`) in worktree `../BBall-worker-sigma-gate`, branch
`worker/sigma-gate-retune`. Frame reproduces the tag exactly: 1,172 rows, max
zone n=57, floor zone n=297; the in-run incumbent at 10 seeds scores right
$6.109M / floor $1.983M / A1 0.7652 / B1 0.8329 against the stored reference's
$6.111M / $1.984M / 0.7653 / 0.8324.

## 1. What changed and why

Nothing ships. The 36-config grid plus an 11-config boundary probe says sigma is
the only parameter with real leverage, and that its leverage is an artifact of
how Stage 1 and Stage 2 compose rather than a pricing improvement. Raising sigma
from 0.02 to 0.04 cuts max-zone MAE by $0.71M and floor-zone MAE by $0.03-0.39M
depending on `k_floor`, easily clearing the brief's zone bar — while dragging the
C1 calibration slope from 0.9883 to 0.9647, six times the movement of any other
candidate.

The mechanism (section 4) is that each censored side is a **one-way valve**:
pushing a gated row's latent past its CBA bound always helps, because the zone is
uniformly biased toward the bound, and overshooting costs nothing, because
Stage 2 clips it back onto the bound. Sigma is the size of that push. So zone MAE
falls monotonically in sigma with no interior optimum — the clip's geometry is
choosing the parameter, not the data — and the bill arrives on the global
calibration slope, which is exactly the instrument the brief nominated as the
guardrail.

## 2. Numbers

### 10-seed confirmation (seeds 0-9, 5 GroupKFold folds, fixed v7.8x zone rows)

All deltas paired against the incumbent **scored in the same run** — same folds,
same seeds — never against a stored headline.

| sigma | gate | k | right MAE (n=57) | dR | floor MAE (n=297) | dF | dSel | t | C1 slope | A1 | B1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.55 | 2.0 | 6.109 | — | 1.983 | — | — | — | 0.9883 | 0.7652 | 0.8329 |
| 0.02 | 0.55 | 2.5 | 6.113 | +0.004 | 1.922 | −0.061 | +0.00002 | +0.24 | 0.9862 | 0.7652 | 0.8335 |
| 0.02 | 0.55 | 3.0 | 6.092 | −0.017 | 1.892 | −0.090 | −0.00007 | −0.67 | 0.9851 | 0.7652 | 0.8338 |
| 0.04 | 0.55 | 1.5 | 5.399 | **−0.711** | 1.952 | −0.031 | −0.00021 | −0.19 | **0.9647** | 0.7650 | 0.8312 |

Per-fold dSel (the fold is the unit of replication):

| config | per-fold deltas | SE |
|---|---|---|
| 0.02/0.55/2.5 | −0.00002, +0.00030, +0.00012, +0.00007, −0.00034 | 0.00011 |
| 0.02/0.55/3.0 | −0.00013, −0.00006, −0.00036, −0.00012, +0.00031 | 0.00011 |
| 0.04/0.55/1.5 | −0.00258, −0.00216, −0.00095, +0.00281, +0.00183 | 0.00108 |

B1 by origin: incumbent 2024 0.8530 / 2025 0.8246 / 2026 0.8118; k=3.0 gains
+0.0009 overall (2025 carries it, 0.8246 → 0.8272); sigma=0.04 loses 0.0017,
entirely from 2025 (0.8246 → 0.8161) while 2026 *gains* the same amount.

### 3-seed screen, 36 configs — the shape

Full table in `screen_table.txt`. Three facts:

- **gate_frac is inert.** Across 0.45 / 0.55 / 0.65 it gates 56 / 55 / 52 of the
  57 max-zone rows. Right-zone MAE moves 0.01-0.05 — below the seed noise. The
  parameter this task was sent to retune barely exists as a parameter.
- **k_floor is monotone and saturates.** floor MAE 1.984 (k=2.0) → 1.924 (2.5) →
  1.895 (3.0) → 1.883 (3.5) → 1.879 (4.0) → 1.876 (5.0). The gated left count
  saturates alongside it (195 → 242 → 262 → ~265). k=3.0 is where the curve
  flattens; the brief's grid stopped one step short of showing that.
- **sigma is monotone and does NOT saturate.** Right-zone MAE 6.52 (0.01) → 6.10
  (0.02) → 5.90 (0.025) → 5.70 (0.03) → 5.37 (0.04) → 4.86 (0.06). I extended the
  grid past 0.04 specifically to look for the interior optimum. There isn't one.

### Control: per-side sigma (the brief's optional second pass)

`sigma_left` added to `_make_tobit_obj` and `make_grabit_fitter`, defaulting to
`sigma`. **Equivalence verified: `sigma_left=None` and `sigma_left=0.02` produce
bit-identical predictions (max abs diff 0.000e+00).**

Holding one side and moving the other (3 seeds) separates the two effects
cleanly:

| sigma_right | sigma_left | right MAE | floor MAE |
|---|---|---|---|
| 0.02 | 0.02 | 6.103 | 1.984 |
| 0.04 | 0.02 | 5.385 | 1.992 |
| 0.02 | 0.04 | 6.095 | 1.756 |
| 0.04 | 0.04 | 5.373 | 1.762 |

Each side's sigma moves its own zone and leaves the other within noise, and the
two effects are additive. The sides are independent — which confirms the valve
diagnosis rather than rescuing it: there are two valves, not one, and both are
open-ended.

## 3. Gate verdicts

The brief's judging order — zone objectives first, guardrails as no-regression,
B1 as a final sanity check on the winner only.

| Gate | 0.02/0.55/2.5 | 0.02/0.55/3.0 | 0.04/0.55/1.5 |
|---|---|---|---|
| **1. Zone** (one zone ≤ −0.10, other ≤ +0.05) | **FAIL** (best −0.061) | **FAIL** (best −0.090) | PASS (−0.711 / −0.031) |
| **2a. dSel** t > −2 | PASS (+0.24) | PASS (−0.67) | PASS (−0.19) |
| **2b. C1 slope** in [0.99, 1.01] | FAIL (0.9862)\* | FAIL (0.9851)\* | **FAIL** (0.9647) |
| **2c. C2** fixed-row \|bias\| growth ≤ $0.30M | PASS (+0.061 worst) | PASS (+0.042 worst) | PASS (+0.049 worst) |
| **3. B1** drop ≤ 0.003 | PASS (+0.0006) | PASS (+0.0009) | PASS (−0.0017) |

\* **See escalation below — the incumbent's own slope is 0.9883, outside the
stated window.** Under the only self-consistent reading (no degradation relative
to the incumbent), k=2.5 costs 0.0021 and k=3.0 costs 0.0032, while sigma=0.04
costs 0.0236 — an order of magnitude more.

**Nothing passes gate 1 and gate 2b together.** The one config that clears the
zone bar fails calibration by the widest margin in the sweep. The configs that
hold calibration never reach the zone bar. Incumbent retained.

C2 detail is in `c2.txt`. Worth noting that sigma=0.04 *improves* Bird Rights
bias (−2.502 → −2.200, |bias| −0.302), which is the same effect seen from the
other side: Bird Rights is where the max re-signings live, and inflating max
latents flatters that segment for the same reason it flatters the max zone.

## 4. The mechanism, and why the winner would have been an artifact

**One sentence:** sigma sets how far past its CBA bound a gated row's latent gets
pushed, and because Stage 2 clips the prediction back onto the bound, overshooting
is free — so both zones improve monotonically in sigma with no interior optimum,
and the cost is paid in global calibration.

The measurement (`mech.txt`) is the share of zone rows whose final prediction sits
exactly on its bound:

| sigma | pinned at ceiling (n=57) | right MAE | right bias | pinned at floor (n=297) | floor MAE |
|---|---|---|---|---|---|
| 0.01 | 15.8% | 6.518 | −6.478 | 17.2% | 2.231 |
| 0.02 | 19.3% | 6.103 | −6.063 | 35.0% | 1.984 |
| 0.025 | 21.1% | 5.902 | −5.862 | 43.8% | 1.841 |
| 0.03 | 21.1% | 5.702 | −5.662 | 44.4% | 1.860 |
| 0.04 | 31.6% | 5.373 | −5.333 | 50.8% | 1.762 |
| 0.06 | 42.1% | 4.855 | −4.785 | 48.8% | 1.867 |

(k_floor = 2.0, gate_frac = 0.55 throughout; pinned share is over the seed-averaged
OOF, so a row counts as pinned only when it clipped in every seed.)

The pinned share rises monotonically with sigma and zone MAE falls with it. The
giveaway is the bias column: **right-zone bias ≈ −(right-zone MAE)** at every
sigma. Every row in the zone is underpredicted, so *any* upward push reduces MAE,
and the ceiling clip guarantees the push can never overshoot into a penalty. The
floor zone is the mirror image (bias +1.93 against MAE 1.98). These are not two
tunings that happen to agree; they are one asymmetry appearing twice.

That is why sigma has no interior optimum, and it is why a sigma chosen on zone
MAE is not measuring what sigma means. In the censored-normal likelihood sigma is
the width over which the transition happens — a scale parameter with a statistical
interpretation. Selected this way it degenerates into "how many rows do you want
pinned to the bound", and at the limit the model predicts the ceiling for every
max row. That scores well inside a zone defined as rows paid ≥90% of their ceiling
and says nothing about value — while the latent, which is what the Value Board
publishes, is inflated by exactly the amount of the push.

## 5. Anomalies

- **The −0.711 right-zone gain at sigma=0.04 is the red flag the worker brief
  warns about, and section 4 is its explanation.** It is real arithmetic on the
  zone rows and it would have been adopted by a zone-MAE-only rule. It is not a
  pricing improvement.
- **No interior optimum in sigma** across 0.01-0.06. A scale parameter whose
  objective is monotone to the edge of any grid you draw is being selected by
  something other than fit.
- **gate_frac does essentially nothing** (52-56 of 57 rows gated across its whole
  range). Either the range is too narrow to bind or the albatross rule it encodes
  is already satisfied by `is_max_contract`. Worth knowing before anyone spends
  another sweep on it.
- **A2 was not used as a decision metric anywhere here**, per the brief; it is in
  the JSON for completeness.
- **Selection-pressure caution.** METHODOLOGY's confirmation-split audit names
  "the sigma and gate thresholds" among the decisions that drifted the pooled
  metric. This sweep judged on zone MAE with dSel only as a no-regression guard,
  and lands on "change nothing", so it adds no new drift. The confirmation split
  was not read at any point.

## 6. Escalation: the C1 guardrail is not satisfiable as written

The brief requires the C1 calibration slope to stay in **[0.99, 1.01]**. The
incumbent's slope is **0.9883** (10 seeds; the stored v7.8x reference is 0.9885).
The reference model does not satisfy its own guardrail, so the window cannot be
read as an absolute admissibility criterion — every candidate including the
status quo fails it.

I applied the reading that changes nothing about this task's outcome (relative:
"no meaningful degradation versus the incumbent") and reported both. But fixing
the rule is architect's lane — `evaluate_suite.py`'s accept/reject rules are
explicitly off-limits to workers — so I have not changed it. The choice is
between restating the window around the actual operating point (say [0.98, 1.01])
and making the gate relative. It matters beyond this task: any future change that
touches calibration will hit the same contradiction.

## 7. Files touched

Branch `worker/sigma-gate-retune` in `../BBall-worker-sigma-gate`, off `v7.8x`:

- `src/model/train.py` — `_make_tobit_obj` gains `sigma_left=None`; behaviour
  identical when unset (verified bit-identical)
- `src/model/evaluate_suite.py` — `make_grabit_fitter` gains `sigma_left=None`,
  passed through
- `docs/briefs/2026-07-23-sigma-gate-retune.RESULT.md` — this file

`ISSUES.md` is deliberately **not** touched on the branch — see section 9.

**The branch ships no parameter change.** The only code on it is the additive
`sigma_left` hook, which the brief invited and which is inert by default. Adopt
or drop it independently of the retune verdict — my recommendation is to take it,
because it is what made the side-separation control possible, but not to use it
to tune sigma.

Experiment scripts stayed in scratchpad, not the repo:
`sweep.py`, `analyze.py`, `mech.py`, `c2.py`, `equiv.py`, and the raw
`screen_results.json` / `screen2_results.json` / `confirm_results.json` plus
per-config OOF `.npz`.

## 8. Proposed commit message

```
Sweep sigma and both censoring gates; keep the incumbent

Re-screened sigma x gate_frac x k_floor (36 configs at 3 seeds, 4 confirmed
at 10) on the zone scorecards now that the row set is 1,172 rather than the
1,487 the settings were chosen on. Nothing displaces 0.02 / 0.55 / 2.0.

sigma=0.04 clears the zone bar easily (max zone -$0.71M) and fails the
calibration guardrail by six times any other candidate (slope 0.9883 ->
0.9647). The reason it wins is structural: Stage 2 clips predictions onto
the CBA bound, so pushing a gated row's latent past that bound is free,
and every zone row is biased toward its bound to begin with. Zone MAE is
therefore monotone in sigma with no interior optimum out to 0.06 -- the
pinned-at-bound share rises 19% -> 42% in step with it. Selecting sigma on
zone MAE selects how many rows to pin, not how to price them.

gate_frac is inert (52-56 of 57 rows gated across 0.45-0.65). k_floor is
monotone but saturates just past 3.0, and its best honest gain (floor zone
-$0.090M at k=3.0) misses the +/-0.10 bar while costing 0.003 of slope.

Adds an inert sigma_left hook to _make_tobit_obj, verified bit-identical
when unset; the per-side control it enabled shows the two censored sides
are independent and additive, which confirms rather than rescues the
diagnosis above.

Closes ISSUES #7. Files a new issue: the C1 guardrail window [0.99, 1.01]
excludes the incumbent itself (0.9883) and needs restating.
```

The ISSUES.md edits are NOT on the branch (section 9 explains why) — apply them
to HEAD's copy when landing, or this commit message overstates what it carries.

Version number left unassigned — no published number moves, so the architect may
prefer none at all.

## 9. ISSUES.md changes — text to apply, not applied

**I did not edit `ISSUES.md` on the branch, on purpose.** The file was rewritten
between `v7.8x` and HEAD: at my pinned tag it has six entries and no #7 at all
(the entry this task addresses was written afterward, at `6226c68`). Editing the
tag's copy would hand the architect a guaranteed conflict on a file whose HEAD
version is the one that matters. The text below applies to **HEAD's** ISSUES.md.

**(a) Delete entry #7 entirely.** It is resolved: the sweep ran, the settings
held, and the file is a worklist rather than a changelog. The finding belongs in
`VERSION_HISTORY.md` / `METHODOLOGY.md` (docs agent's lane), not here. If the
architect would rather keep a pointer so nobody re-runs the sweep, the minimum is:

> ## 7. Sigma and both gates: swept 2026-07-23, incumbent held
>
> **Severity**: none — closed, recorded so it is not re-run.
>
> Re-screened over sigma {0.01-0.06} x gate_frac {0.45,0.55,0.65} x k_floor
> {1.5-5.0} on the current 1,172-row set, judged on both zone scorecards.
> 0.02/0.55/2.0 held. sigma is monotone in zone MAE with no interior optimum
> because Stage 2's clip makes overshooting a CBA bound free; selecting it on
> zone MAE selects the pinned-at-bound share (19% at 0.02, 42% at 0.06), not
> price accuracy, and costs calibration slope. gate_frac gates 52-56 of 57 rows
> across its whole range and is effectively inert. Evidence:
> `docs/briefs/2026-07-23-sigma-gate-retune.RESULT.md`.

**(b) Add a new entry** for the guardrail contradiction in section 6:

> ## N. The C1 calibration guardrail excludes the model it guards
>
> **Severity**: low today, blocking the first time a calibration change is
> judged — the gate cannot be applied as written.
>
> The sigma/gate brief (and the acceptance habit it was written from) requires
> the C1 calibration slope to land in **[0.99, 1.01]**. The champion's slope is
> **0.9883** at 10 seeds (`outputs/models/evaluation_suite.json`,
> `champion.C1_calibration_slope`; the v7.8x reference reads 0.9885). The
> incumbent fails its own admissibility window, so every candidate compared
> against it fails too, and the gate silently degrades into "reject everything".
>
> This is the same failure mode METHODOLOGY already documents for the C2 rule —
> an absolute threshold applied to a quantity that is only meaningful relative
> to the current operating point.
>
> **Reproduce**: `python -c "import json; print(json.load(open('outputs/models/evaluation_suite.json'))['champion']['C1_calibration_slope'])"`
>
> **Fix**: either restate the window around the actual operating point (roughly
> [0.98, 1.01]) or make the gate relative — "slope does not move more than
> ~0.005 against the incumbent" — and say which in `evaluate_suite.py`'s
> accept/reject block so the two cannot drift apart again. A relative gate would
> have produced the same verdict in this task (sigma=0.04 moves it 0.024) while
> remaining applicable.
>
> **Verify**: the champion passes its own guardrail.
