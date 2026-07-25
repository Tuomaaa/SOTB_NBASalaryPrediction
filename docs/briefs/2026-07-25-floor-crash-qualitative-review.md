# Floor-zone large residuals: qualitative player review

Date: 2026-07-25

## Scope

The reviewed diagnostic contained 240 Floor Zone rows. Thirty-four had a
positive Signing Residual greater than $5M. This memo records the basketball
and contract context discussed for all 34 rows; it is a hypothesis register,
not an adopted model change. Any candidate feature still needs the project's
paired-CV protocol.

Two interpretation cautions apply throughout:

- Several displayed `salary_m` values are the team's veteran-minimum cap charge,
  not the player's cash salary. Examples include Oubre 2023, Whiteside 2020,
  Reggie Jackson 2020 and Chris Paul 2025.
- A Floor Percentage contract is not always an open-market valuation. Buyout
  compensation, a contender discount, a pre-existing contract and suspected
  contract circumvention can all make the observed amount unsuitable as a
  clean statement of Latent Value.

## Confirmed handling register

| Handling | Rows | Interpretation |
|---|---|---|
| Offseason major injury | 1 | Injury or surgery known after the priced season but before signing |
| Low-quality high usage | 2, 4, 14 | High `USG%` combined with low relative TS%, low `AST%` and/or high `TOV%` |
| Position-relative height | 3, 29 | Position-season average height minus player height |
| Age | 7, 9, 23 | Late-career decline; keep the current register label simple |
| Small-sample exclusion | 8, 20, 22 | Short late-season auditions should not be extrapolated as full-season roles |
| Recent buyout or waiver | 4, 6, 12, 18, 24, 31, 34 | The old guaranteed salary is stale and the new minimum may be supplemental pay |
| Discount to join a top team nearing collapse | 10, 11, 19, 21, 30, 32 | Salary exchanged for role, exposure and a remaining contender brand |
| Invalid Year-1 observation | 13 | Salary was fixed before the performance being valued |
| Data problem | 17, 25, 27, 28, 33 | Age, minutes or prior salary is mismatched or median-filled |
| Leave blank | 5, 15, 16 | No agreed feature or sample treatment yet |
| Pending observation | 26 | Do not assign a category yet |

Rows may have more than one valid label. Drummond 2021, for example, combines
a buyout with low-quality high usage. Data-problem rows should be repaired or
excluded before their basketball explanation informs an ablation.

## All 34 rows

| # | Player | Season | Observed | Model | Residual | Final review |
|---:|---|---:|---:|---:|---:|---|
| 1 | Victor Oladipo | 2021 | $2.39M | $27.31M | +$24.92M | Offseason major injury |
| 2 | Kelly Oubre Jr. | 2023 | $2.02M | $20.01M | +$17.99M | Low-quality high usage |
| 3 | Montrezl Harrell | 2022 | $2.46M | $19.60M | +$17.14M | Position-relative height |
| 4 | Andre Drummond | 2021 | $2.40M | $18.69M | +$16.29M | Low-quality high usage; buyout |
| 5 | Hassan Whiteside | 2020 | $1.62M | $15.11M | +$13.49M | Leave blank |
| 6 | Reggie Jackson | 2020 | $1.62M | $11.60M | +$9.98M | Buyout |
| 7 | Chris Paul | 2025 | $2.30M | $11.95M | +$9.65M | Age |
| 8 | Javonte Green | 2024 | $1.73M | $11.31M | +$9.58M | Small-sample exclusion |
| 9 | Marc Gasol | 2020 | $2.56M | $11.92M | +$9.36M | Age |
| 10 | Tyus Jones | 2024 | $2.09M | $11.32M | +$9.23M | Discount to join a top team nearing collapse |
| 11 | Gary Trent Jr. | 2024 | $2.09M | $11.27M | +$9.18M | Same discount class; contract-integrity caveat |
| 12 | Blake Griffin | 2021 | $2.64M | $10.96M | +$8.32M | Buyout |
| 13 | Kendrick Nunn | 2020 | $1.66M | $9.90M | +$8.24M | Invalid Year-1 observation |
| 14 | Emmanuel Mudiay | 2019 | $1.74M | $9.81M | +$8.07M | Low-quality high usage |
| 15 | Ziaire Williams | 2026 | $2.45M | $10.42M | +$7.97M | Leave blank |
| 16 | Josh Okogie | 2025 | $2.30M | $9.96M | +$7.66M | Leave blank |
| 17 | Montrezl Harrell | 2023 | $2.02M | $9.62M | +$7.60M | Data problem; duplicate case |
| 18 | Reggie Bullock | 2023 | $2.02M | $9.56M | +$7.54M | Buyout |
| 19 | Chris Boucher | 2025 | $2.30M | $9.72M | +$7.42M | Discount to join a top team nearing collapse |
| 20 | JaKarr Sampson | 2019 | $1.74M | $8.34M | +$6.60M | Small-sample exclusion |
| 21 | Taurean Prince | 2024 | $2.09M | $8.47M | +$6.38M | Discount to join a top team nearing collapse |
| 22 | Skylar Mays | 2023 | $1.80M | $8.05M | +$6.25M | Small-sample exclusion |
| 23 | Paul Millsap | 2021 | $2.64M | $8.86M | +$6.22M | Age |
| 24 | Austin Rivers | 2019 | $2.17M | $8.06M | +$5.89M | Waived |
| 25 | DeMarcus Cousins | 2020 | $1.62M | $7.44M | +$5.82M | Data problem |
| 26 | Amir Coffey | 2025 | $2.30M | $8.10M | +$5.80M | Pending observation |
| 27 | Bol Bol | 2023 | $2.02M | $7.66M | +$5.64M | Data problem |
| 28 | Malik Beasley | 2023 | $2.02M | $7.54M | +$5.52M | Data problem |
| 29 | Langston Galloway | 2020 | $1.62M | $7.03M | +$5.41M | Position-relative height |
| 30 | Kent Bazemore | 2020 | $1.62M | $6.92M | +$5.30M | Discount to join a top team nearing collapse |
| 31 | Spencer Dinwiddie | 2024 | $2.09M | $7.32M | +$5.23M | Waived |
| 32 | Alec Burks | 2019 | $2.32M | $7.52M | +$5.20M | Discount to join a top team nearing collapse |
| 33 | Patrick Beverley | 2023 | $2.02M | $7.17M | +$5.15M | Data problem |
| 34 | Reggie Jackson | 2024 | $1.95M | $7.06M | +$5.11M | Waived |

## Case notes

### 1. Victor Oladipo, 2021

Oladipo was not bought out. After four games with Miami, he required another
operation on the same right quadriceps tendon ruptured in 2019 and did not
return until March 2022. The market priced an unknown return date and a second
major operation on the athletic base of a guard; the model priced the healthy,
28-year-old former All-Star visible in the season aggregates.

### 2. Kelly Oubre Jr., 2023

Oubre's 20 points per game came with high usage, below-league scoring
efficiency and very little playmaking. Teams did not view that weak-team role as
portable to a contender. Missing the early market also turned a plausible
mid-level price into a one-year proof contract. The agreed modeling hypothesis
is the joint signal from `USG%`, relative TS%, `AST%` and `TOV%`, not points per
game alone.

### 3. Montrezl Harrell, 2022

Harrell had center-only offensive skills in a forward-sized body. He could not
space the floor and could not reliably protect the rim or handle full-sized
centers, which made his strong regular-season finishing difficult to carry into
the playoffs. The agreed hypothesis is position-relative height, kept general
rather than a Harrell-specific rule.

### 4. Andre Drummond, 2021

Drummond's Cleveland usage was actively misleading: a paint-bound center used
31.3% of possessions while finishing inefficiently and turning the ball over.
The model awarded usage, minutes and an obsolete max contract; the market had
already demoted him to a low-usage backup and Cleveland bought him out. One
year later, lower minutes, lower usage and a prior minimum finally made the
model reflect that earlier market decision.

### 5. Hassan Whiteside, 2020

Whiteside's Portland impact metrics were positive, so this should not be
rationalized as fake production. His 30 minutes partly filled the vacancy left
by Jusuf Nurkic, and his positive impact came from a traditional center skill
set that the market could replace cheaply. The model caught the lower role and
impact one year later. No agreed improvement follows yet; this may include real
market underpricing.

### 6. Reggie Jackson, 2020

Detroit could not trade Jackson after a long back absence and paid him to
leave. The old $18M salary was therefore evidence of a failed guaranteed
contract, not current value. His Clippers minimum was a third-guard proof
contract and partly a team choice; his subsequent playoff run and two-year,
$22M deal confirm that the minimum was transitional.

### 7. Chris Paul, 2025

Paul's contract mixed age with a large voluntary component: a return to Los
Angeles, proximity to family and a chosen final-career role. The displayed
$2.30M is the team cap charge rather than his roughly $3.63M cash minimum. The
register keeps only the simple label `age`; the row is not clean evidence that
his basketball value equaled the Floor Percentage.

### 8. Javonte Green, 2024

Green returned from knee surgery at the end of the season and produced an
excellent nine-game audition after spending almost the entire year outside the
NBA. Teams priced a 31-year-old fringe wing and a nine-game sample, not the
full-season extrapolation of those per-minute results.

### 9. Marc Gasol, 2020

Gasol's decision-making and defensive positioning remained valuable, but his
mobility and rim finishing had reached a late-career physical cliff. His
playoff role fell, then his Lakers role fell again. Unlike Paul's strategic
minimum, the subsequent performance provides direct evidence for the age
interpretation.

### 10-11. Tyus Jones and Gary Trent Jr., 2024

Both used a minimum on a top-heavy, apron-constrained team to purchase a large
role and another chance at free agency. Jones received Phoenix's starting point
guard role; Trent received Milwaukee's shooting environment and a reunion with
Damian Lillard. Trent also carries a separate contract-integrity caveat. These
rows are summarized as: `discount to join a top team nearing collapse`.

### 12. Blake Griffin, 2021

Detroit continued paying the dominant share of Griffin's guaranteed money
after buying him out. His Nets minimum was incremental compensation on a chosen
contender, not his total income. Repeated knee injuries had also reduced him
from a primary creator to a matchup-dependent passing and spacing big.

### 13. Kendrick Nunn, 2020

Nunn signed a multi-year minimum before his breakout rookie season. The listed
2020 salary was already fixed; the market never had an opportunity to reprice
his All-Rookie performance. This is not a valid fresh Year-1 market observation.

### 14. Emmanuel Mudiay, 2019

Mudiay's career-high scoring on a poor Knicks team did not solve the role
contradiction: he needed the ball, scored below league efficiency, created too
little for teammates and could not move off ball. His prior salary was also a
slotted rookie-scale amount. He joins Oubre and Drummond in the low-quality
high-usage group.

### 15. Ziaire Williams, 2026

The attribution already penalizes Ziaire for DARKO, minutes, usage, RAPM and
LEBRON. The $12.4M league baseline still falls only to $10.1M, while the market
moves players who fail the rotation threshold directly to the Floor
Percentage. This looks like a floor cliff rather than a missing ordinary
performance column, but no improvement was agreed. Leave blank.

### 16. Josh Okogie, 2025

Phoenix's prior two-year, $16M contract had a non-guaranteed second year and was
used as trade-matching salary in the Nick Richards deal. Charlotte declined to
retain it. Okogie's point-of-attack defense is real, but his lack of spacing
makes him matchup-dependent. The prior salary was artificial; no general
improvement was agreed. Leave blank.

### 17. Montrezl Harrell, 2023

The row reports age 26 and 21.96 minutes; Harrell was 29 and had played about
11.9 minutes the prior season. His ACL tear occurred after signing and cannot
explain the price. Treat this as a duplicate Harrell case with bad inputs, not a
new model lesson.

### 18. Reggie Bullock, 2023

Bullock was salary matching in the Grant Williams transaction, then San Antonio
bought him out before he signed with Houston. His old salary was still partly
being paid elsewhere, while the defensive half of his former 3-and-D role had
declined with age.

### 19 and 21. Chris Boucher and Taurean Prince

Neither row supplies clean evidence for a new performance penalty. Boucher had
positive per-minute impact; Prince still supplied credible shooting. Both
selected apron-constrained former contenders with severe rotation holes. They
join Jones and Trent under `discount to join a top team nearing collapse`.

### 20. JaKarr Sampson, 2019

Sampson's 31.75 minutes and 23.1% usage came from four late-season games for
Chicago, in which he averaged roughly 20 points and eight rebounds. His longer
journeyman history was far more informative than that four-game burst.

### 22. Skylar Mays, 2023

Mays' apparent starting role came from six games after Portland shut down its
main players. The market treated a 25-year-old development-league guard's
15-point, eight-assist burst as a tanking-season audition, not a stable NBA role.

### 23. Paul Millsap, 2021

Millsap's regular-season impact remained positive, but his playoff role had
fallen to about 12 minutes and he was 36 at signing. As with Gasol, the market
priced the next season of an aging big while the inputs described the remaining
value of the previous season.

### 24. Austin Rivers, 2019

Phoenix waived Rivers while his old guaranteed salary was still being paid. He
then took a low-usage role in Houston and chose to stay for a contender discount.
The final register places him in the `waived` bucket because the old contract
was not a valid current-price anchor.

### 25. DeMarcus Cousins, 2020

The row says age 26 and supplies generic minutes despite Cousins being 30 and
missing the entire season. His real history was an Achilles rupture, quadriceps
tear and ACL tear in roughly 19 months. The basketball reason is obvious, but
the current row is marked `data problem` until its inputs are repaired.

### 26. Amir Coffey, 2025

Coffey shot well in a real Clippers rotation but carried a strongly negative
DARKO contribution. His subsequent Milwaukee minimum may contain a role and
contender discount, but the evidence was not strong enough to finalize that
label. Keep as `pending observation`.

### 27-28. Bol Bol and Malik Beasley, 2023

Both rows carry suspicious repeated defaults. Bol was 23, not 26; Beasley
played roughly 26 minutes, not the repeated 21.96. Their transaction context
can be reviewed after the feature linkage is corrected, but neither should
teach the current model while the inputs are unreliable.

### 29. Langston Galloway, 2020

Galloway had point-guard height but shooting-guard function. He could shoot,
but could not run an offense, so every viable lineup needed another creator and
usually fielded two small guards. Even at 42% from three in Phoenix, he played
only about 11 minutes. This is the guard-side analogue of Harrell's
position-relative height problem.

### 30. Kent Bazemore, 2020

Bazemore's 2016 cap-spike contract was stale. He reportedly declined better
money or term to return to a Warriors team that retained its dynasty identity
but had just finished last and lost Klay Thompson again. He bought role and
future-contract exposure with a minimum.

### 31. Spencer Dinwiddie, 2024

Toronto waived Dinwiddie immediately after acquiring him and continued paying
his guarantee. His post-Achilles rim pressure had declined, while his shooting,
defense and off-ball play did not support a smaller role. The subsequent Dallas
minimum followed a clear waiver signal.

### 32. Alec Burks, 2019

After years of low availability and three teams in one season, Burks chose the
injury-depleted post-Durant Warriors, where he could recover a large scoring
role. His 16 points per game there and later $6M Knicks contract support the
interpretation of the minimum as a one-year career investment.

### 33. Patrick Beverley, 2023

This row says age 26 and zero prior Cap Percentage. Beverley was 35 and had
earned roughly $13M the previous season; the repeated minutes, usage and
availability also resemble defaults. Mark as a data problem before interpreting
the Orlando buyout or Philadelphia minimum.

### 34. Reggie Jackson, 2024

Denver attached multiple second-round picks to move Jackson after he exercised
his option; Charlotte then waived him. DARKO was already deeply negative and
his playoff role had contracted. The asset-backed salary dump and waiver make
the subsequent minimum straightforward.

## Questions left open

- Whiteside 2020 may be a genuine market underprice of positive but replaceable
  regular-season impact; no feature is assigned.
- Ziaire Williams 2026 exposes a possible discrete rotation/floor cliff, but
  the current register leaves the intervention blank.
- Josh Okogie's prior salary was trade engineering, yet no general treatment is
  assigned.
- Amir Coffey remains pending observation.
- Before modeling any row tagged `data problem`, verify the player-season join
  and the fallback policy that produced repeated age/minutes/usage values.

