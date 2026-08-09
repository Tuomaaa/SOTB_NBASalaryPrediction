# NBA Free Agent Valuation

Estimating what an NBA player is worth on the open market, and separating that
worth from what the Collective Bargaining Agreement actually lets a team pay
them. The distinction between those two quantities is the whole project.

## Language

### Value and price

**Cap Percentage**:
A salary expressed as a fraction of that season's league salary cap. The unit
every monetary quantity is stored in, so that figures from different seasons
compare directly.
_Avoid_: salary, dollars, AAV (all fine for display, never for storage)

**Latent Value**:
What a player is worth with no CBA bound applied — the output of the Grabit
stage. For most players this equals their predicted salary; for elite players it
exceeds what any team is permitted to pay, and for marginal players it falls
below what any team is permitted to offer.
_Avoid_: true value, unconstrained salary, raw prediction

**Floor Percentage**:
The lowest cap percentage a specific player may legally be paid, set by the
veteran-minimum scale for his years of experience. The mirror of Max-Eligible
Percentage; like it, varies per player rather than being a league constant.
_Avoid_: minimum, vet min, salary floor (ambiguous with the prorated cutoff)

**Max-Eligible Percentage**:
The highest cap percentage a specific player may legally be paid, set by their
years of experience and elevated by the Rose Rule and Supermax provisions.
Varies per player, not a league-wide constant.
_Avoid_: max contract, salary ceiling, cap

**Predicted Salary**:
Latent Value clipped into the band between Floor Percentage and Max-Eligible
Percentage. The number a team could actually put on a contract.

### Contract structure

**Year-1 Contract**:
The first season of a contract — a fresh negotiated price, and the only kind of
row the model is trained on.
_Avoid_: new signing, fresh deal

**Escalator Year**:
Any season after the first in a multi-year contract. The salary is a
CBA-mandated raise on a price agreed years earlier, so it reflects a past
market rather than a current valuation. Excluded from training.
_Avoid_: option year, later year, year 2+

**Rookie-Scale Contract**:
A first-round pick's slotted, non-negotiated deal. Excluded from training
because the price carries no market information.

**Signing Mechanism**:
The CBA exception a contract was signed under — Bird Rights, Early Bird,
Non-Bird, Mid-Level Exception, Cap Space, Minimum, or Sign & Trade. Determines
whether a team could exceed the cap to make the deal.
_Avoid_: contract type, deal type, signing type

### Model error versus team judgment

These two are the same arithmetic and mean entirely different things. Keeping
them apart is what stops a reader concluding the model is wrong when it is
merely answering a different question.

**Signing Residual**:
Predicted Salary minus actual salary, on a negotiated Year-1 Contract —
excluding Rookie-Scale Contracts, whose price no one agreed to. A measure of
**model accuracy**: the market set a price and the model tried to match it.

**Contract Surplus**:
Predicted Salary minus actual salary, on any contract year. A measure of
**team outcome** — what a player is worth this season against what he is
currently owed. Not a claim about model error.
_Avoid_: residual, error, diff (all reserved for Signing Residual)

**Albatross Contract**:
A contract carrying large negative Contract Surplus — a player still paid near
the maximum while his production has fallen away.

### Censoring

**Censored Observation**:
A player-season whose recorded salary is a CBA bound rather than his worth.
Right-censored at the Max-Eligible ceiling, the observation is a *lower* bound
on Latent Value; left-censored at the Floor Percentage, it is an *upper* bound.
_Avoid_: capped, truncated, maxed out

**Gated Censoring**:
The rule deciding which rows at a bound count as Censored Observations. A player
qualifies only if the uncensored model agrees the bound binds — near the ceiling
on the right, near the floor on the left. This excludes Albatross Contracts,
where a max salary reflects a past decision rather than suppressed present
worth, and veterans who *chose* a minimum while priced well above it.

**Zone**:
The rows one censoring side exists to fix — the Max Zone (paid at least 90% of
their own Max-Eligible Percentage) and the Floor Zone (pinned at the minimum).
Each side's keep/drop decision reads its own zone's MAE, because a pooled
statistic averages a targeted effect over rows it never touches.
_Avoid_: censored set (that is the gated subset, which is smaller)

**Default Parameters / Told Parameters**:
The split between what Stage 1 knows and what later stages know. Stage 1 prices
a player under *default parameters* — signing context averaged over the training
distribution. Stages 2 and 3 adjust for *told parameters*, the constraints this
contract actually faced. Everything still averaged inside Stage 1 shows up as
mechanism bias in the C2 diagnostic.

### Pipeline stages

**Stage**:
One of the three composable steps that turn a raw model output into a prediction.
Stage 1 emits a Latent Value; Stage 2 applies the CBA bounds that depend only on
the player (push + clip); Stage 3 applies what depends on the realized signing
route — the extension raise cap, then the per-type Signing Offset — re-imposing
both bounds after each. Composition lives in `src/model/stages.py`.

**Signing Offset**:
The Stage-3 constant added to a row of a given Signing Mechanism, equal to the
shrunk mean out-of-fold Signing Residual of that mechanism. Defined only for the
four eligibility mechanisms (Bird Rights, Cap Space, Early Bird, Non-Bird);
exactly zero everywhere else, because an exception mechanism is determined by
the contract value itself and conditioning on it would read the target.
_Avoid_: mechanism adjustment, signing correction factor

**Push**:
The upward half of Stage 2. Where the route classifier says P(max) >= TAU (0.52),
the latent is moved a P-weighted fraction of the way to MARGIN (1.05) times the
tier ceiling, and the clip then lands it on the ceiling itself. The push exists
because the clip alone is one-directional — it can cap a prediction from above but
cannot reach a max-worthy player the model prices below his ceiling.
_Avoid_: boost, adjust upward

**Clip**:
The downward half of Stage 2 (and, separately, the extension clip in Stage 3).
Caps a prediction at the player's Max-Eligible Percentage from above and lifts it
to the Floor Percentage from below. The Stage-2 clip is deterministic; the
Stage-3 clip additionally requires knowing the signing route.
_Avoid_: clamp, bound, cap (ambiguous with salary cap)

### Signing route

**Route** (or **Signing Route**):
How a contract was signed — Bird Rights, Early Bird, Non-Bird, Mid-Level
Exception, Cap Space, Minimum, Sign & Trade, or Extension. Unlike Signing
Mechanism (a label attached after the fact), "route" emphasises the path the
transaction took and the constraints it faced.
_Avoid_: mechanism (when speaking of the path rather than the label)

**Told-Route Convention**:
From v8.0x onward, headline numbers are computed with the model told the signing
route (Stage 3 active), and the ex-ante number (Stage 3 off) is reported beside
it wherever the time series must stay readable. The convention's test is per-route:
being told the route must still leave a non-trivial computation. Extensions pass
(you still compute 1.40 x prior pay); the floor does not (told the route is told
the answer).

### Inputs

**Impact Metric**:
One of the three independent public estimates of a player's on-court value —
DARKO DPM, LEBRON, and LAKER — each standardised within its season.
_Avoid_: rating, advanced stat, base rating

**Availability**:
Share of games played, weighted across the prior three seasons with the most
recent weighted heaviest. Durability, not health.

**Market Trajectory**:
A Kalman-filtered estimate of a player's market price (`kf_market_value` in
code), anchored to his most recent negotiated Year-1 contract and updated
through model-predicted intermediate seasons. Replaces `prev_cap_pct` in the
final feature list (v8.13x); the base model still uses `prev_cap_pct` internally
as a measurement input to the filter.
_Avoid_: prev_cap_pct (reserved for the raw prior-contract feature the base
model uses), contract trajectory, filtered salary

**CBA Era**:
Which collective agreement governed a season. The 2023 agreement changed
contract structure enough that seasons on either side of it are not directly
comparable.

### Presentation

**Value Board**:
The view over a single season showing every player ranked by Contract Surplus.
Answers "who is worth more or less than they are being paid right now."

**Signing Board**:
The view over negotiated Year-1 Contracts, ranked by Signing Residual. Its rows
are exactly the model's training domain. Answers "how close does the model get
when the market sets a fresh price."
