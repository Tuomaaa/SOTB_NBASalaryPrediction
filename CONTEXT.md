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
What a player is worth with no CBA ceiling applied — the output of the Grabit
stage. For most players this equals their predicted salary; for elite players
it exceeds what any team is permitted to pay.
_Avoid_: true value, unconstrained salary, raw prediction

**Max-Eligible Percentage**:
The highest cap percentage a specific player may legally be paid, set by their
years of experience and elevated by the Rose Rule and Supermax provisions.
Varies per player, not a league-wide constant.
_Avoid_: max contract, salary ceiling, cap

**Predicted Salary**:
Latent Value clipped to Max-Eligible Percentage. The number a team could
actually put on a contract.

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
A player-season whose recorded salary is the Max-Eligible ceiling rather than
his worth, so the observation is a lower bound on Latent Value.
_Avoid_: capped, truncated, maxed out

**Gated Censoring**:
The rule deciding which max-salary players count as Censored Observations. A
player qualifies only if the uncensored model already values him near the
ceiling — which excludes Albatross Contracts, where a max salary reflects a
past decision rather than a suppressed present worth.

### Inputs

**Impact Metric**:
One of the three independent public estimates of a player's on-court value —
DARKO DPM, LEBRON, and RAPM — each standardised within its season.
_Avoid_: rating, advanced stat, base rating

**Availability**:
Share of games played, weighted across the prior three seasons with the most
recent weighted heaviest. Durability, not health.

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
