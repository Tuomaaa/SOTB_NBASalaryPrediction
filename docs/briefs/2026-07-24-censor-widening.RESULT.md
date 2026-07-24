# RESULT: supervised widening of the right-censor gate

**Verdict: NO WINNER. The incumbent (right population = `is_max_contract`,
sigma 0.02) holds.** No `(c, sigma)` config clears the win box — true-max zone
MAE improves ≥ $0.30M *and* counterweight-band bias growth ≤ $0.30M — at 10
seeds. The corner is empty by a clean, confirmed margin: the single config that
respects the counterweight brake (c=0.85 / σ=0.02) buys only **−$0.263M** on the
true-max zone, $0.037M short of the bar, and every config that reaches the
−$0.30M bar over-pushes the counterweight band by ≥ **+$0.51M**, past the
+$0.30M yardstick. This pins down how much a *global* dial can buy; the per-row
version (task #4) owns the rest.

Pinned to current `master` = `4156404` (v7.9x line), worktree
`../BBall-worker-censor`, branch `worker/censor-widening`. Frame reproduces the
tag: 949 rows, true-max zone n=56, counterweight band [0.70,0.90) n=45,
at-floor n=241. The in-run incumbent at 10 seeds scores true-max MAE **$6.302M**
/ bias −$6.262M against the stored reference's $6.302M / −$6.262M — identical.

## 1. What changed and why

Nothing ships. The idea was admissible because widening the right-censored
*population* — also treating near-max rows (paid ≥ c of their own ceiling but
below the max flag) as right-censored — was supposed to give sigma an interior
optimum: those rows' salaries are true market prices, so over-pushing lifts
their prediction past the price and their error rises, braking the push that the
2026-07-23 sigma sweep found unbounded on the max zone alone. The grid was
supposed to answer whether that brake, plus a higher sigma, nets out to a real
true-max improvement.

It does not, and the reason is a falsified premise: **the counterweight rows are
underpredicted by $4.57M at the incumbent (bias −$4.570M), not sitting at their
true price.** They are simply *more underpredicted near-max rows*, behaving like
the true-max zone (bias −$6.262M), not like at-price rows whose error would rise
only after an overshoot. So (a) the true-max zone MAE stays **monotone in sigma
at every c** — no interior optimum in the objective the experiment exists to
improve; and (b) the only brake is the counterweight band's own signed bias
climbing under the push, and it crosses the +$0.30M yardstick *before* the
true-max zone reaches its −$0.30M bar. A global dial cannot be in both boxes at
once.

## 2. Numbers

### 2a. 3-seed screen, 16 configs (seeds 0-2), same-run incumbent

All deltas paired against the incumbent **scored in the same run** — same folds,
same seeds. Incumbent (3-seed): true-max MAE $6.333M, cwBias −$4.611M, cwMAE
$6.941M, floor MAE $2.138M, slope 0.9898, gR 56. `gR` = rows this config
actually censors on the right (population ∩ `bp ≥ 0.55·ceiling`, context only;
the zone MAEs are over the FIXED v7.9x rows n=56 / n=45).

```
  c   sigma | gR | truemaxMAE  dMax | cwMAE  cwBias  d(bias) | floorMAE | dSel     t   | slope
-----------------------------------------------------------------------------------------------
 0.70  0.02 | 91 |   5.419  -0.914 | 6.880  -3.833  +0.778 |  2.148 | -0.00194 -0.66 | 0.9603
 0.70  0.03 | 91 |   4.859  -1.474 | 6.970  -3.283  +1.328 |  2.157 | -0.00369 -0.80 | 0.9407
 0.70  0.04 | 91 |   4.448  -1.884 | 7.245  -2.714  +1.897 |  2.173 | -0.00808 -1.20 | 0.9230
 0.70  0.06 | 91 |   3.875  -2.457 | 7.579  -1.778  +2.833 |  2.208 | -0.01660 -1.57 | 0.8939
 0.75  0.02 | 85 |   5.566  -0.767 | 6.830  -3.942  +0.669 |  2.143 | -0.00086 -0.33 | 0.9660
 0.75  0.03 | 85 |   5.004  -1.328 | 6.890  -3.427  +1.184 |  2.150 | -0.00236 -0.51 | 0.9477
 0.75  0.04 | 85 |   4.532  -1.800 | 7.043  -2.916  +1.695 |  2.163 | -0.00513 -0.81 | 0.9313
 0.75  0.06 | 85 |   3.945  -2.387 | 7.402  -2.019  +2.592 |  2.197 | -0.01302 -1.35 | 0.9029
 0.80  0.02 | 78 |   5.629  -0.703 | 6.871  -4.070  +0.541 |  2.143 | -0.00009 -0.03 | 0.9701
 0.80  0.03 | 78 |   5.121  -1.212 | 6.928  -3.656  +0.955 |  2.152 | -0.00068 -0.17 | 0.9540
 0.80  0.04 | 78 |   4.694  -1.639 | 7.015  -3.217  +1.394 |  2.161 | -0.00281 -0.46 | 0.9397
 0.80  0.06 | 78 |   4.052  -2.281 | 7.367  -2.375  +2.236 |  2.185 | -0.00932 -0.93 | 0.9135
 0.85  0.02 | 67 |   6.069  -0.263 | 6.883  -4.451  +0.160 |  2.141 | +0.00099 +1.04 | 0.9831
 0.85  0.03 | 67 |   5.607  -0.726 | 6.834  -4.081  +0.530 |  2.148 | +0.00056 +0.31 | 0.9689
 0.85  0.04 | 67 |   5.148  -1.185 | 6.895  -3.748  +0.863 |  2.152 | -0.00009 -0.04 | 0.9552
 0.85  0.06 | 67 |   4.491  -1.841 | 7.059  -3.035  +1.576 |  2.171 | -0.00522 -0.96 | 0.9295
```

Two facts fix the whole result:

- **true-max MAE is monotone in sigma at every c** (e.g. c=0.85: 6.069 → 5.607 →
  5.148 → 4.491). There is no interior optimum in the objective the experiment
  targets. Widening did not create one; it only moved *where* the brake lives.
- **counterweight signed bias-growth is the only gate that binds.** The single
  config with growth ≤ +$0.30M is c=0.85/σ=0.02 (+0.160), and its true-max gain
  is −$0.263M. Every config reaching the −$0.30M bar has growth ≥ +$0.53M.

The screen corner is empty; nothing is eliminated within one 3-seed SE of
clearing both boxes (the next-best counterweight-growth among true-max-clearing
configs is +$0.53M — 0.23M past the yardstick, far outside a screen SE).

### 2b. 10-seed confirmation (seeds 0-9), frontier of the corner

Incumbent (10-seed): true-max MAE **$6.302M**, bias −$6.262M, pinned 0.161;
cwMAE $6.904M, cwBias −$4.570M; floor MAE $2.150M; slope 0.9881; gR 56.

| config | gR | true-max MAE | dMax | pin@ceil | cwMAE | cwBias | cw Δbias | dSel (t) | slope | 25%+ band bias | B1 drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **incumbent** c=0.90/σ=0.02 | 56 | 6.302 | — | 0.161 | 6.904 | −4.570 | — | — | 0.9881 | +0.44 | — |
| c=0.85 / σ=0.02 | 67 | 6.039 | **−0.263** | 0.232 | 6.862 | −4.415 | **+0.155** | +0.00081 (+0.90) | 0.9813 | +0.34 | −0.0007 |
| c=0.80 / σ=0.02 | 78 | 5.606 | −0.697 | 0.286 | 6.865 | −4.021 | +0.549 | +0.00045 (+0.16) | 0.9688 | +1.17 | +0.0004 |
| c=0.85 / σ=0.03 | 67 | 5.583 | −0.720 | 0.286 | 6.828 | −4.056 | +0.514 | +0.00051 (+0.33) | 0.9673 | +1.31 | +0.0002 |

Per-fold dSel (fold is the unit of replication):

| config | per-fold dSel | SE |
|---|---|---|
| c=0.85/σ=0.02 | +0.00138, −0.00054, +0.00302, +0.00213, −0.00194 | 0.00090 |
| c=0.80/σ=0.02 | +0.00345, −0.00154, +0.00327, +0.00655, −0.00947 | 0.00280 |
| c=0.85/σ=0.03 | +0.00359, −0.00320, +0.00145, +0.00382, −0.00312 | 0.00155 |

The 10-seed numbers are stable against the screen (c=0.85/σ=0.02 dMax −0.263 in
both; cw growth +0.155 vs +0.160). The corner stays empty.

## 3. Gate verdicts

Win box = {true-max dMax ≤ −$0.30M} ∧ {counterweight bias growth ≤ +$0.30M},
guardrails clean. Counterweight "bias growth" read as **signed** Δbias
(cand − inc, positive = upward push) — see §4 for why the signed reading is the
only coherent one and why the verdict is robust to it.

| Gate | c=0.85/σ=0.02 | c=0.80/σ=0.02 | c=0.85/σ=0.03 |
|---|---|---|---|
| **1. true-max dMax ≤ −0.30** | **FAIL** (−0.263) | PASS (−0.697) | PASS (−0.720) |
| **2. counterweight bias growth ≤ +0.30** | PASS (+0.155) | **FAIL** (+0.549) | **FAIL** (+0.514) |
| 3a. dSel t > −2 | PASS (+0.90) | PASS (+0.16) | PASS (+0.33) |
| 3b. C2 fixed-seg \|bias\| growth ≤ 0.30 | PASS (+0.015) | PASS (+0.023) | PASS (+0.035) |
| 3c. calibration (25%+ predicted band) | PASS (+0.34 ≈ inc +0.44) | **FAIL** (+1.17) | **FAIL** (+1.31) |
| 4. B1 forward drop ≤ 0.003 | PASS (−0.0007) | PASS (+0.0004) | PASS (+0.0002) |

**No config passes gates 1 and 2 together.** c=0.85/σ=0.02 is clean on *every*
guardrail — dSel, C2, calibration band, and B1 (it even *improves* the forward
R2 by 0.0007) — and simply does not clear the true-max bar. The two configs that
clear the true-max bar fail the counterweight gate, and are *independently*
blocked by the calibration band table: the 25%+ predicted band (n≈40) inflates
from +$0.44M to +$1.17M / +$1.31M — ~$0.7-0.9M of top-end over-prediction, the
same one-way-valve signature the sigma sweep flagged, now visible as a band-table
finding rather than a global-slope one. Everything below the 15-25% band is
untouched by any config (the widening is a pure top-end intervention).

## 4. The mechanism, and the interpretive fork it forces

**One sentence:** widening recruits near-max rows that are themselves
underpredicted by ~$4.6M at baseline, so censoring them buys true-max MAE
*monotonically* in sigma with no interior optimum in the true-max zone; the only
brake is the counterweight band's own bias climbing under the push, and it
crosses the +$0.30M yardstick before the true-max zone reaches its −$0.30M bar —
so a global dial buys at most −$0.263M (c=0.85/σ=0.02) while staying calibrated.

The brief's premise was that counterweight rows sit at their true price, so
pushing their prediction up = over-pushing = harm, and the harm brakes sigma at
an interior optimum. **Measured, the premise is false**: the incumbent already
underpredicts the counterweight band by $4.570M (mirror of the true-max zone's
−$6.262M). So pushing these rows up mostly *reduces* their error at first — cwMAE
actually improves slightly at gentle settings (6.904 → 6.828-6.865) and only
worsens under an aggressive push (7.58 at c=0.70/σ=0.06). The brake on *|bias|*
and on *MAE* is therefore weak and displaced; the only instrument that fires
early enough to matter is the **signed** bias-growth, which climbs the moment the
push starts because there is no at-price band to overshoot.

This forces a reading of the win condition's "bias growth ≤ +$0.30M":

- **Signed Δbias (adopted):** cand − inc, positive = upward push. Only
  c=0.85/σ=0.02 passes; verdict is *no winner*. This is the coherent reading:
  the counterweight band was added to the judging *to be a brake*, and only the
  signed instrument brakes. (An |bias|-growth reading — the literal "C2
  yardstick" — never binds here because |bias| shrinks everywhere the push
  helps, which would make the brake toothless and reproduce the exact one-way
  valve the 2026-07-23 sweep closed.)
- **Robustness:** even under the lenient |bias| reading, the two true-max-
  clearing configs are still killed — by the calibration band table (25%+ band
  +$1.17M / +$1.31M) and the global slope (0.969-0.967, relative-C1 excess
  ~+0.019 vs incumbent's 0.0119, ~4× the 0.005 tolerance). **The verdict is
  no-winner under every reading of gate 2.**

## 5. Anomalies

- **No red-flag deltas.** The largest true-max gains (−$2.46M at c=0.70/σ=0.06)
  are exactly the sigma-sweep artifact reproduced on a wider population: slope
  collapses to 0.894 and cwBias inflates to −$1.78M. They are real arithmetic on
  the zone, not pricing improvements, and are correctly excluded by the
  counterweight gate + calibration band. No config passes the gates whose
  mechanism is unexplained.
- **The falsified premise is the finding.** That the counterweight band is
  underpredicted, not at-price, is the substantive result and the reason the
  global dial cannot win. It is also the signal for task #4: a *per-row* censor
  that preserves the top half of each widened row's price information (which the
  loss here discards — a widened row contributes only "worth ≥ X") is the only
  way to turn the counterweight rows into information rather than a brake. This
  experiment bounds the global-dial ceiling at −$0.263M-while-calibrated.
- **A2 not used as a decision metric** anywhere, per protocol.
- **Confirmation split never read** into any zone, fill, or decision metric;
  dSel uses `fold_r2_selection` only.

## 6. Files touched

Branch `worker/censor-widening` in `../BBall-worker-censor`, off `4156404`:

- `src/model/evaluate_suite.py` — `make_grabit_fitter` gains `censor_c=None`;
  right population keys on `is_max_contract` when None, else
  `cap_pct ≥ censor_c·max_eligible_pct`. **Verified bit-identical when unset:**
  `censor_c=0.90` reproduces `censor_c=None` (is_max) to max abs diff 0.000e+00
  across all five folds, and the 10-seed incumbent reproduces the stored suite
  reference (true-max MAE $6.302M) exactly.
- `src/model/train.py` — `train_grabit` gains the same inert `censor_c`, plus
  `sigma_left` wired through its two `_make_tobit_obj` calls (the hook existed in
  `_make_tobit_obj` / `make_grabit_fitter` from the sigma retune but was not
  reachable from `train.py`; inert at the default `None`).

**The branch ships no parameter change** — `censor_c` defaults to `None`, the
`is_max_contract` behaviour. Adopt or drop the hook independently of this
verdict; my recommendation is to take it (it is inert, it reproduces the
negative result, and task #4 will want a parameterized right-population), the
same disposition the sigma retune gave its `sigma_left` hook.

Experiment scripts stayed in scratchpad, not the repo: `harness.py`, `probe.py`,
and the raw `screen.json` / `confirm.json`.

## 7. Proposed commit message

```
Widen the right-censor population (c-grid); keep the incumbent

Tested treating near-max rows (paid >= c of their own ceiling but below the
is_max flag) as right-censored, to give sigma the interior optimum the
2026-07-23 sweep found it lacked on the max zone alone. Grid c in
{0.70,0.75,0.80,0.85} x sigma_right in {0.02,0.03,0.04,0.06}, sigma_left
pinned 0.02, k_floor 2.0, 16 configs at 3 seeds + the corner at 10, judged on
the FIXED v7.9x zones (true-max n=56, counterweight band [0.70,0.90) n=45).

No config clears the win box (true-max MAE -$0.30M and counterweight bias
growth <= +$0.30M). The premise fails: the counterweight rows are
underpredicted by $4.57M at the incumbent, not at their true price, so
censoring them buys true-max MAE monotonically in sigma (no interior optimum
in the true-max zone) while the only brake -- the counterweight band's own
signed bias -- crosses +$0.30M before the true-max zone reaches -$0.30M. The
one counterweight-safe config (c=0.85/sigma=0.02) buys only -$0.263M and is
clean on every guardrail (dSel t+0.90, C2 +0.015, B1 -0.0007, 25%+ band +0.34
~ incumbent +0.44); the two configs that clear the true-max bar over-push the
top: 25%+ predicted band inflates +0.44 -> +1.17/+1.31M and slope falls to
0.969/0.967. A global dial cannot be in both boxes; the per-row censor owns
the rest.

Adds an inert censor_c hook to make_grabit_fitter and train_grabit (verified
bit-identical at c=0.90/unset; 10-seed incumbent reproduces the stored suite
true-max MAE $6.302M), and wires the existing sigma_left through train_grabit.
No parameter change ships.
```

Version number left unassigned — no published number moves.

## 8. ISSUES.md additions

None. No bug was found; the underpredicted-counterweight finding is a
characterization result recorded here and relevant to task #4, not an unfixed
defect. The `censor_c` hook is inert and documented in-code.

---

## Appendix A — reproduce

From `../BBall-worker-censor` (branch `worker/censor-widening`, off `4156404`),
with `OMP_NUM_THREADS=6`:

```
python <scratch>/harness.py equiv     # censor_c=0.90 vs None bit-identical
python <scratch>/harness.py screen     # 16 configs x 3 seeds -> screen.json
python <scratch>/harness.py confirm    # corner x 10 seeds + B1 -> confirm.json
```

Fixed zones are computed once from `load_evaluation_frame()`: true-max =
`is_max_contract` (n=56), counterweight = `~is_max & 0.70 ≤ cap_pct/ceiling <
0.90` (n=45). The widened populations nest: c=0.70 pop 101 = 56 max + 45
counterweight; c=0.85 pop 67 = 56 + the [0.85,0.90) subset.

## Appendix B — B1 forward (rolling-origin, 10 seeds)

| config | B1 | 2024 | 2025 | 2026 | drop vs inc |
|---|---|---|---|---|---|
| incumbent | 0.8191 | 0.8545 | 0.8097 | 0.7791 | — |
| c=0.85/σ=0.02 | 0.8198 | 0.8574 | 0.8069 | 0.7817 | −0.0007 |
| c=0.80/σ=0.02 | 0.8187 | 0.8677 | 0.7960 | 0.7780 | +0.0004 |
| c=0.85/σ=0.03 | 0.8190 | 0.8610 | 0.7978 | 0.7863 | +0.0002 |

All within ±0.0007 of the incumbent — well inside the 0.003 bar. Forward is inert
to the widening (it is a top-end training-loss change; the forward metric is
pooled over all rows).
