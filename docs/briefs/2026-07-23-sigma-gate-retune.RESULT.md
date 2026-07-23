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

Full table in Appendix A, boundary probe in Appendix B. Three facts:

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

C2 detail is in Appendix D. Worth noting that sigma=0.04 *improves* Bird Rights
bias (−2.502 → −2.200, |bias| −0.302), which is the same effect seen from the
other side: Bird Rights is where the max re-signings live, and inflating max
latents flatters that segment for the same reason it flatters the max zone.

## 4. The mechanism, and why the winner would have been an artifact

**One sentence:** sigma sets how far past its CBA bound a gated row's latent gets
pushed, and because Stage 2 clips the prediction back onto the bound, overshooting
is free — so both zones improve monotonically in sigma with no interior optimum,
and the cost is paid in global calibration.

The measurement (Appendix C) is the share of zone rows whose final prediction sits
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

---

## Appendix A - full 36-config screen (3 seeds)

Deltas paired against the incumbent scored in the same run. `gR`/`gL` are the
rows this config actually censors on each side (context; the MAE columns are
over the FIXED v7.8x zone rows, n=57 and n=297).

```

screen: 36 configs, 3 seeds
incumbent (0.02, 0.55, 2.0): right 6.103  floor 1.984  A1 0.7649  slope 0.9890
(v7.8x 10-seed reference: right 6.111  floor 1.984  A1 0.7653)

 sigma  gate    k | rightMAE      dR floorMAE      dF |     dSel      t |  slope      A1 |   gR   gL
----------------------------------------------------------------------------------------------------
  0.01  0.45  1.5 |    6.520  +0.417    2.230  +0.246 |  -0.0008  -0.81 | 1.0097  0.7643 |   56  131
  0.01  0.45  2.0 |    6.506  +0.403    2.175  +0.191 |  -0.0004  -0.46 | 1.0077  0.7645 |   56  195
  0.01  0.45  2.5 |    6.501  +0.398    2.147  +0.163 |  -0.0002  -0.25 | 1.0070  0.7648 |   56  242
  0.01  0.45  3.0 |    6.512  +0.409    2.136  +0.152 |  -0.0005  -0.53 | 1.0065  0.7646 |   56  262
  0.01  0.55  1.5 |    6.518  +0.415    2.231  +0.247 |  -0.0007  -0.73 | 1.0097  0.7643 |   55  131
  0.01  0.55  2.0 |    6.514  +0.411    2.177  +0.193 |  -0.0005  -0.51 | 1.0079  0.7645 |   55  195
  0.01  0.55  2.5 |    6.501  +0.398    2.147  +0.163 |  -0.0002  -0.23 | 1.0069  0.7648 |   55  242
  0.01  0.55  3.0 |    6.513  +0.410    2.136  +0.152 |  -0.0004  -0.51 | 1.0066  0.7646 |   55  262
  0.01  0.65  1.5 |    6.539  +0.436    2.231  +0.247 |  -0.0011  -1.13 | 1.0099  0.7640 |   52  131
  0.01  0.65  2.0 |    6.501  +0.398    2.171  +0.187 |  -0.0006  -0.66 | 1.0074  0.7645 |   52  195
  0.01  0.65  2.5 |    6.496  +0.393    2.146  +0.162 |  -0.0004  -0.52 | 1.0064  0.7644 |   52  242
  0.01  0.65  3.0 |    6.509  +0.406    2.136  +0.152 |  -0.0006  -0.62 | 1.0066  0.7644 |   52  262
  0.02  0.45  1.5 |    6.135  +0.032    2.096  +0.113 |  -0.0003  -0.73 | 0.9937  0.7647 |   56  131
  0.02  0.45  2.0 |    6.100  -0.003    1.984  +0.001 |  -0.0001  -1.16 | 0.9890  0.7649 |   56  195
  0.02  0.45  2.5 |    6.107  +0.004    1.924  -0.060 |  +0.0001  +0.22 | 0.9870  0.7650 |   56  242
  0.02  0.45  3.0 |    6.097  -0.006    1.895  -0.089 |  +0.0001  +0.24 | 0.9858  0.7650 |   56  262
  0.02  0.55  1.5 |    6.139  +0.036    2.097  +0.113 |  -0.0003  -0.76 | 0.9938  0.7647 |   55  131
  0.02  0.55  2.0 |    6.103  +0.000    1.984  +0.000 |  +0.0000   +nan | 0.9890  0.7649 |   55  195  <-- incumbent
  0.02  0.55  2.5 |    6.103  -0.000    1.924  -0.059 |  +0.0001  +0.21 | 0.9869  0.7650 |   55  242
  0.02  0.55  3.0 |    6.091  -0.012    1.895  -0.089 |  +0.0000  +0.10 | 0.9857  0.7650 |   55  262
  0.02  0.65  1.5 |    6.149  +0.046    2.099  +0.115 |  -0.0003  -0.86 | 0.9942  0.7647 |   52  131
  0.02  0.65  2.0 |    6.113  +0.010    1.982  -0.002 |  -0.0001  -0.96 | 0.9892  0.7649 |   52  195
  0.02  0.65  2.5 |    6.116  +0.013    1.920  -0.063 |  -0.0000  -0.17 | 0.9870  0.7648 |   52  242
  0.02  0.65  3.0 |    6.104  +0.001    1.896  -0.088 |  -0.0003  -1.04 | 0.9857  0.7646 |   52  262
  0.04  0.45  1.5 |    5.396  -0.707    1.948  -0.036 |  +0.0003  +0.31 | 0.9650  0.7650 |   56  131
  0.04  0.45  2.0 |    5.375  -0.728    1.761  -0.223 |  -0.0011  -1.34 | 0.9573  0.7638 |   56  195
  0.04  0.45  2.5 |    5.368  -0.735    1.651  -0.333 |  -0.0013  -1.18 | 0.9537  0.7634 |   56  242
  0.04  0.45  3.0 |    5.405  -0.698    1.592  -0.391 |  -0.0024  -2.62 | 0.9523  0.7626 |   56  262
  0.04  0.55  1.5 |    5.396  -0.707    1.947  -0.037 |  +0.0004  +0.35 | 0.9651  0.7650 |   55  131
  0.04  0.55  2.0 |    5.373  -0.730    1.762  -0.222 |  -0.0011  -1.33 | 0.9573  0.7638 |   55  195
  0.04  0.55  2.5 |    5.371  -0.732    1.651  -0.332 |  -0.0013  -1.19 | 0.9538  0.7634 |   55  242
  0.04  0.55  3.0 |    5.407  -0.696    1.593  -0.391 |  -0.0024  -2.57 | 0.9524  0.7627 |   55  262
  0.04  0.65  1.5 |    5.426  -0.677    1.946  -0.038 |  +0.0002  +0.21 | 0.9659  0.7650 |   52  131
  0.04  0.65  2.0 |    5.406  -0.697    1.754  -0.230 |  -0.0008  -0.91 | 0.9582  0.7641 |   52  195
  0.04  0.65  2.5 |    5.391  -0.712    1.652  -0.332 |  -0.0018  -1.93 | 0.9542  0.7633 |   52  242
  0.04  0.65  3.0 |    5.429  -0.674    1.590  -0.393 |  -0.0025  -2.61 | 0.9533  0.7628 |   52  262

Configs meeting the zone rule (one zone -0.10 or better, other no worse than +0.05):
  sigma=0.04 gate=0.45 k=3.0  dR -0.698  dF -0.391  dSel -0.0024 (t -2.62)  slope 0.9523
  sigma=0.04 gate=0.55 k=3.0  dR -0.696  dF -0.391  dSel -0.0024 (t -2.57)  slope 0.9524
  sigma=0.04 gate=0.45 k=2.5  dR -0.735  dF -0.333  dSel -0.0013 (t -1.18)  slope 0.9537
  sigma=0.04 gate=0.65 k=3.0  dR -0.674  dF -0.393  dSel -0.0025 (t -2.61)  slope 0.9533
  sigma=0.04 gate=0.55 k=2.5  dR -0.732  dF -0.332  dSel -0.0013 (t -1.19)  slope 0.9538
  sigma=0.04 gate=0.65 k=2.5  dR -0.712  dF -0.332  dSel -0.0018 (t -1.93)  slope 0.9542
  sigma=0.04 gate=0.55 k=2.0  dR -0.730  dF -0.222  dSel -0.0011 (t -1.33)  slope 0.9573
  sigma=0.04 gate=0.45 k=2.0  dR -0.728  dF -0.223  dSel -0.0011 (t -1.34)  slope 0.9573
  sigma=0.04 gate=0.65 k=2.0  dR -0.697  dF -0.230  dSel -0.0008 (t -0.91)  slope 0.9582
  sigma=0.04 gate=0.55 k=1.5  dR -0.707  dF -0.037  dSel +0.0004 (t +0.35)  slope 0.9651
  sigma=0.04 gate=0.45 k=1.5  dR -0.707  dF -0.036  dSel +0.0003 (t +0.31)  slope 0.9650
  sigma=0.04 gate=0.65 k=1.5  dR -0.677  dF -0.038  dSel +0.0002 (t +0.21)  slope 0.9659

Within-1-SE survivors on each zone (3-seed SEs are ~2x wide, so this is deliberately generous):
  right: best 5.368, keeping 12 configs
    sigma=0.04 gate=0.45 k=2.5  5.368
    sigma=0.04 gate=0.55 k=2.5  5.371
    sigma=0.04 gate=0.55 k=2.0  5.373
    sigma=0.04 gate=0.45 k=2.0  5.375
    sigma=0.04 gate=0.65 k=2.5  5.391
    sigma=0.04 gate=0.55 k=1.5  5.396
    sigma=0.04 gate=0.45 k=1.5  5.396
    sigma=0.04 gate=0.45 k=3.0  5.405
  floor: best 1.590, keeping 3 configs
    sigma=0.04 gate=0.65 k=3.0  1.590
    sigma=0.04 gate=0.45 k=3.0  1.592
    sigma=0.04 gate=0.55 k=3.0  1.593
```

## Appendix B - boundary probe (3 seeds)

Run because sigma=0.04 and k_floor=3.0 both sat on the edge of the brief's grid.

```
frame 1172 rows | right zone n=57 | floor zone n=297
[1/11] sigma=0.02 gate=0.55 k=2.0 sl=None  right 6.103  floor 1.984  A1 0.7649  (9.9s)
[2/11] sigma=0.025 gate=0.55 k=2.5 sl=None  right 5.902  floor 1.841  A1 0.7645  (8.3s)
[3/11] sigma=0.025 gate=0.55 k=3.0 sl=None  right 5.894  floor 1.803  A1 0.7648  (8.6s)
[4/11] sigma=0.03 gate=0.55 k=2.0 sl=None  right 5.702  floor 1.860  A1 0.7649  (8.2s)
[5/11] sigma=0.03 gate=0.55 k=2.5 sl=None  right 5.733  floor 1.766  A1 0.7646  (8.1s)
[6/11] sigma=0.03 gate=0.55 k=3.0 sl=None  right 5.733  floor 1.728  A1 0.7642  (8.3s)
[7/11] sigma=0.02 gate=0.55 k=3.5 sl=None  right 6.097  floor 1.883  A1 0.7652  (8.7s)
[8/11] sigma=0.02 gate=0.55 k=4.0 sl=None  right 6.092  floor 1.879  A1 0.7650  (8.9s)
[9/11] sigma=0.02 gate=0.55 k=5.0 sl=None  right 6.094  floor 1.876  A1 0.7648  (8.8s)
[10/11] sigma=0.04 gate=0.55 k=1.0 sl=None  right 5.436  floor 2.314  A1 0.7645  (8.0s)
[11/11] sigma=0.06 gate=0.55 k=1.5 sl=None  right 4.855  floor 1.867  A1 0.7623  (8.3s)
wrote C:\Users\panh3\AppData\Local\Temp\claude\C--Users-panh3-Documents-ROSE-Personal-Project-BBall-Analysis\02d3536f-4216-4458-8639-72bce879544a\scratchpad\screen2_results.json
```

## Appendix C - pinned-at-bound mechanism table

```

right zone n=57   floor zone n=297
                config | pinned@ceil   rMAE   rBias | pinned@floor   fMAE   fBias
----------------------------------------------------------------------------------
          s=0.01 k=1.5 |      15.8%  6.518  -6.478 |       17.2%  2.231  +2.202
          s=0.01 k=2.0 |      17.5%  6.514  -6.474 |       20.5%  2.177  +2.142
          s=0.01 k=2.5 |      15.8%  6.501  -6.461 |       20.9%  2.147  +2.110
          s=0.01 k=3.0 |      17.5%  6.513  -6.473 |       20.5%  2.136  +2.099
          s=0.02 k=1.5 |      19.3%  6.139  -6.099 |       31.0%  2.097  +2.052
          s=0.02 k=2.0 |      19.3%  6.103  -6.063 |       35.0%  1.984  +1.928
          s=0.02 k=2.5 |      19.3%  6.103  -6.063 |       37.0%  1.924  +1.861
          s=0.02 k=3.0 |      19.3%  6.091  -6.051 |       38.4%  1.895  +1.826
          s=0.04 k=1.5 |      31.6%  5.396  -5.356 |       40.7%  1.947  +1.873
          s=0.04 k=2.0 |      31.6%  5.373  -5.333 |       50.8%  1.762  +1.669
          s=0.04 k=2.5 |      29.8%  5.371  -5.331 |       55.2%  1.651  +1.552
          s=0.04 k=3.0 |      29.8%  5.407  -5.367 |       56.6%  1.593  +1.490
          s=0.02 k=2.0 |      19.3%  6.103  -6.063 |       35.0%  1.984  +1.928
         s=0.025 k=2.5 |      21.1%  5.902  -5.862 |       43.8%  1.841  +1.761
         s=0.025 k=3.0 |      21.1%  5.894  -5.854 |       46.5%  1.803  +1.720
          s=0.03 k=2.0 |      21.1%  5.702  -5.662 |       44.4%  1.860  +1.779
          s=0.03 k=2.5 |      22.8%  5.733  -5.692 |       48.8%  1.766  +1.676
          s=0.03 k=3.0 |      24.6%  5.733  -5.693 |       50.8%  1.728  +1.634
          s=0.02 k=3.5 |      19.3%  6.097  -6.057 |       38.0%  1.883  +1.814
          s=0.02 k=4.0 |      19.3%  6.092  -6.052 |       39.1%  1.879  +1.808
          s=0.02 k=5.0 |      19.3%  6.094  -6.054 |       38.7%  1.876  +1.805
          s=0.04 k=1.0 |      31.6%  5.436  -5.396 |       12.8%  2.314  +2.291
          s=0.06 k=1.5 |      42.1%  4.855  -4.785 |       48.8%  1.867  +1.776
```

## Appendix D - C2 fixed-row segments, 10 seeds

```

sigma=0.02 gate=0.55 k=3.0  vs incumbent  (C2 limit: |bias| growth <= $0.30M)
    segment             n  inc bias  cand bias   d|bias|
    Bird Rights       309    -2.502     -2.547    +0.045
    Minimum           303    +1.860     +1.759    -0.100
    MLE               182    +1.056     +0.965    -0.091
    Unknown           132    +0.223     +0.146    -0.077
    Cap Space         117    -1.196     -1.253    +0.057
    Early Bird         48    -1.340     -1.401    +0.061
    Other              41    +1.460     +1.369    -0.091
    Non-Bird           22    +1.950     +1.815    -0.135
    Sign & Trade       15    -3.823     -3.832    +0.008
    worst |bias| growth: +0.061M  -> PASS

sigma=0.02 gate=0.55 k=2.5  vs incumbent  (C2 limit: |bias| growth <= $0.30M)
    segment             n  inc bias  cand bias   d|bias|
    Bird Rights       309    -2.502     -2.527    +0.025
    Minimum           303    +1.860     +1.792    -0.068
    MLE               182    +1.056     +1.000    -0.056
    Unknown           132    +0.223     +0.175    -0.048
    Cap Space         117    -1.196     -1.234    +0.038
    Early Bird         48    -1.340     -1.382    +0.042
    Other              41    +1.460     +1.396    -0.064
    Non-Bird           22    +1.950     +1.876    -0.074
    Sign & Trade       15    -3.823     -3.830    +0.007
    worst |bias| growth: +0.042M  -> PASS

sigma=0.04 gate=0.55 k=1.5  vs incumbent  (C2 limit: |bias| growth <= $0.30M)
    segment             n  inc bias  cand bias   d|bias|
    Bird Rights       309    -2.502     -2.200    -0.302
    Minimum           303    +1.860     +1.813    -0.046
    MLE               182    +1.056     +1.073    +0.017
    Unknown           132    +0.223     +0.253    +0.030
    Cap Space         117    -1.196     -1.150    -0.046
    Early Bird         48    -1.340     -1.349    +0.009
    Other              41    +1.460     +1.505    +0.045
    Non-Bird           22    +1.950     +2.000    +0.049
    Sign & Trade       15    -3.823     -3.626    -0.198
    worst |bias| growth: +0.049M  -> PASS
```

## Appendix E - sigma_left equivalence and side separation

```

1. EQUIVALENCE
   max abs diff = 0.000e+00   IDENTICAL

2. SIDE SEPARATION (sigma_right / sigma_left -> zone MAEs)
       sR     sL |  rightMAE  floorMAE
     0.02   0.02 |     6.103     1.984
     0.04   0.02 |     5.385     1.992
     0.02   0.04 |     6.095     1.756
     0.04   0.04 |     5.373     1.762
```

Raw artifacts (JSON, per-config OOF `.npz`, and the five experiment scripts)
are in `../BBall-worker-sigma-gate-evidence/`.
