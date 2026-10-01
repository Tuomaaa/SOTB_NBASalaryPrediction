# Waiver term result

## Change

No model change. The pre-registered partially linear waiver term did not pass
its gate. The waiver direction stays open, and the next arm is below (v6.0.3).
Three failure mechanisms were located, and they define that arm.
`scripts/eval_waiver_challengers.py` and the opt-in
`grabit_latent(waiver_term=True)` remain for reproduction.

## Evidence

### Where the error is

Of 120 waived frame rows, 2 were paid above $10M, and none exceeded the
season's non-taxpayer mid-level exception. On these rows the champion's MAE is
$1.60M, against $2.78M on the rest of the frame. The error sits on stars that
a tree cannot reach:

| Row | Error |
|---|---:|
| Damian Lillard 2025 | +$25.7M |
| Kemba Walker 2021 | +$19.5M |
| Andre Drummond 2021 | +$14.9M |
| Bradley Beal 2025 | +$8.1M |

### Three-seed layer-A screen

Arms:

- Drop `is_waived`: t = -0.45.
- Drop `mpg_x_waived`: t = -0.53.
- Add `kf_market_value_x_waived`: t = -0.32. The named rows do not change.
- Coverage control: t = +0.24.

Both features stay. The interaction is dropped.

### Pre-registered arm

The arm is `latent = GBM(x) + beta * is_waived * kf_market_value`. Beta is a
Tobit MLE on inner out-of-fold residuals, with at-floor rows left-censored,
and enters the fit as `base_margin`. The run used 10 seeds and the full suite.

The arm was amended twice before any result, both times at the user's
direction:

1. The floor term was dropped from z.
2. Least squares was replaced by Tobit. Least squares read the two thirds of
   waived rows at the floor as exact values: on folds 0-2 it gave -0.18 to
   -0.31, against -0.36 to -0.50 under Tobit.

Beta over 80 fits: mean -0.438, range -0.544 to -0.356.

| | Incumbent | Waiver term |
|---|---:|---:|
| A1, all rows | 0.8509 | 0.8585 |
| A2 | 0.8559 | 0.8597 |
| B1 | 0.8304 | 0.8372 |
| Selection rows R-squared | 0.8509 | 0.8522 |
| Confirmation canary | 0.8496 | 0.8956 |
| Paired dSel (selection) | | +0.00114, se 0.00612, t = +0.19, FAIL |
| C2 worst growth | | Cap Space +$0.025M, pass |

Kemba Walker, Bradley Beal, Andre Drummond and Ryan Anderson fall in the
confirmation split by name hash. The canary cannot decide an adoption, so the
arm fails on its selection rows.

A targeted check on the waived selection rows also fails: 80 players, paired
t = 0.82 clustered by player, and 29% of players improve.

### Failure mechanisms

**1. The arm discounts ordinary waivers.** The Spotrac text splits the
waived rows into two groups:

| Group | n | Champion bias | Champion MAE | Arm MAE |
|---|---:|---:|---:|---:|
| Plain "Waived by X" | 85 | -$0.13M | $1.01M | $1.17M |
| Buyout, stretch, or money owed | 35 | +$2.52M | $3.02M | $1.76M |

A plain waiver is usually a cut before a guarantee date, with nothing owed.
Chris Paul 2024, Taurean Prince 2023 and Avery Bradley 2019 were cut that way.
George Hill 2019 was re-signed by the same team eight days later. These
players sign at market price, and the champion already prices them without
bias. Only money-owed waivers carry the set-off discount.

**2. The tree compensates.** 20 waived rows moved up, for +$22.2M of added
error, including Blake Griffin 2021, Russell Westbrook 2025 and Reggie
Jackson 2020. The Grabit left gate (`bp <= 2y`) leaves high-value players at
the floor uncensored. The margin therefore pushes their latent below an exact
target, and the tree adds the value back through the `is_waived` splits,
which spill onto other waived rows. Beta is estimated with every at-floor
waived row censored, so the two parts of the model use inconsistent
censoring.

**3. The max push re-inflates.** Damian Lillard 2025 reached a clip-only
error of $2.97M. The push returned him to $11.82M because the route
classifier gives him a high P(max). No waived frame row ever signed a
maximum.

The pre-registration also listed push re-inflation as a FAIL condition. That
clause was a design error: a gate should judge predictive performance, and
the clause did not enter the decision.

### Spotrac separability

Of 584 waiver events, 90 carry a money annotation: buyout, stretch, "gave
back", "dead cap", or reduced salary. Annotated events are unambiguous.

Of about 19 recalled buyouts, two carry plain text: Kyle Lowry 2024 and
Patrick Beverley 2023. Recall is about 90%.

`prior_waiver_text` kept only the last waiver in the lookback, so LaMarcus
Aldridge 2021 lost his San Antonio buyout to a later Brooklyn waiver.

## Decision

Do not adopt the arm. Keep `is_waived` and `mpg_x_waived`.

The next arm (path B, pre-registered in `docs/QUEUE.md`) fixes the three
mechanisms:

- The main GBM trains without money-owed waiver rows. Those rows are priced
  by `max(floor, gamma * m(x))`, where m is the main model's out-of-sample
  latent and gamma is a Tobit fit in each fold.
- P(max) is 0 for waived rows.
- The money-owed flag scans every waiver in the lookback.

## Artifacts

- `scripts/eval_waiver_challengers.py`
- `src/model/route_mixture.py` (`waiver_z`, `waiver_beta`, `_tobit_beta`)
- `outputs/models/waiver_challengers_screen.json`
- `outputs/models/waiver_challengers_full.json`
- `outputs/models/waiver_challengers_full_oof.csv`
