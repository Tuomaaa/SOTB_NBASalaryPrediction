# Open issues

This file contains active defects only. Each issue must include a problem,
reproduction, fix, and completion check. Delete fixed entries and record the
landed change in `VERSION_HISTORY.md`.

Issue numbers are permanent. The highest assigned number is 57, so the next
issue is 58. Keep entries in numeric order because code and historical reports
refer to them by number.

## Retired references still used in code

- #2: continuation-span warning, now in `CLAUDE.md`.
- #31: Kendrick Nunn contract-structure correction, commit `8071745`.
- #38: veteran-minimum cap-charge convention, landed in v5.0.1.
- #41: Spotrac page-identity guard; Josh Gray remains the documented no-page
  case.
- #55: one-year rule for the minimum cap charge, landed in v5.3.3.

## 4. Three signing-mechanism labels remain unknown

**Severity:** low. Signing Mechanism is a diagnostic label.

**Problem:** The current 873-row reference frame has three Unknown rows:
Josh Gray 2020 has no Spotrac page; Abdel Nader 2020 falls between contracts;
Ryan Anderson 2020 is stretched dead money. Batum 2020 and Noah 2020 are also
mislabelled Minimum because dead money collides with a same-season minimum.
Extensions are grouped with Bird Rights, so retention-premium analysis must
split extensions first.

**Reproduce:** Rebuild `spotrac_signing_types.csv`, attach labels to the
evaluation frame, and list Unknown rows plus Minimum rows above $6M.

**Fix:** Add sourced overrides for decidable rows and distinguish dead money
from signed salary. Keep Signing Mechanism out of `FEATURE_COLS`.

**Done:** Every remaining Unknown row has a documented reason, dead-money rows
do not receive a signing mechanism, and retention diagnostics separate
extensions.

## 5. Team or position is missing on 5.4% of web rows

**Severity:** low. This affects display and filtering.

**Problem:** The 2026-07-22 export had 167 of 3,113 rows without
`team_abbreviation` or `position`. The board silently omits these rows from
team filters. Team correctness is tracked in #54.

**Reproduce:** Count null team and position values in the current web export and
compare them with the source tables.

**Fix:** Resolve team through #54 and join position from the latest available
player-season source. Display unresolved values as Unknown.

**Done:** The export reports coverage, unresolved rows remain filterable, and
resolved source values are preserved.

## 6. CBA bounds depend on curated or derived tables

**Severity:** low.

**Problem:** `early_supermax.csv` and `designated_ineligible.csv` encode
exceptions that the feature frame cannot derive from team continuity and
signing timing. `floor_pct` estimates the veteran-minimum scale from observed
mass points.

**Reproduce:** Compare computed ceilings with
`contract_signing_dates.csv`, team continuity, and the two curated lists.
Compare each `floor_pct` bucket with the published minimum scale.

**Fix:** Replace curated ceiling exceptions only after a signing-date and
team-match rule reproduces every current exception. Replace derived floors with
the published scale. Do not widen the award lookback; that gives valid 30% max
signings an incorrect supermax ceiling.

**Done:** Mechanical rules reproduce the curated ceiling decisions, over-cap
observations remain uncensored, and floor values match the published scale.

## 17. Four players lack a 2018 prior salary

**Severity:** low.

**Problem:** Alex Caruso, Shake Milton, Wenyen Gabriel, and Amile Jefferson are
absent from the BBRef 2018 team salary table. Their `prev_cap_pct` therefore
uses the minimum fallback. Only Caruso survives the current evaluation filters,
but the data gap remains.

**Reproduce:** Run `_load_prev_season_cap_pct()` and check the 2018 key for
these four normalized names.

**Fix:** Add a sourced 2018 two-way salary table or expose the fallback through
the known/imputed indicator in #42.

**Done:** Each missing prior is sourced or explicitly marked imputed.

## 20. Zone gates can use locked or moving rows

**Severity:** medium.

**Problem:** Adoption gates can still read zone metrics that include
confirmation players. Some experiment scripts also define a predicted-value
band separately for each model, so reference and candidate are scored on
different rows. The shared absolute-bias calculation is already fixed.

**Reproduce:** Run
`OMP_NUM_THREADS=6 python scripts/eval_route_mixture_p3.py`. Compare the
selection-only and pooled zone gate. In the phase-2 harness, compare the 25%+
band on each model's own rows with a band fixed from the reference.

**Fix:** Provide shared helpers for selection-only zone gates and fixed-row
model comparisons. Pooled zone metrics may remain labelled reporting metrics.

**Done:** Every adoption gate excludes `is_confirmation`, both models use the
same segment rows, and the printed protocol identifies the decision metric.

## 24. Two valid CBA boundaries need documentation

**Severity:** low.

**Problem:** `cba_era` starts in repo season 2024 because it represents the
market response to apron rules. The extension raise multiple changes for deals
signed in repo season 2023 because it is a legal term fixed at signing. Aligning
the constants would break one definition.

**Reproduce:** Compare deals at exactly 1.20 and 1.40 with their signing season.
Jimmy Butler 2023 was signed in 2021 and belongs to the 1.20 rule.

**Fix:** State both definitions and boundaries in `METHODOLOGY.md`.

**Done:** The methodology explains the two boundaries and tests cover the
signing-season rule.

## 25. Tau selection models only probability-weighted interventions

**Severity:** medium.

**Problem:** The current collateral objective uses
`sum(P * intervention)`, which estimates a probability-weighted push or pull.
It understates an unconditional intervention and can select a threshold at the
edge of the search grid.

**Reproduce:** Run
`OMP_NUM_THREADS=6 python scripts/eval_floor_branch.py` and inspect the
expected-versus-realized table. The historical unconditional floor arm had
$338.1M benefit and $514.9M damage while the objective predicted a positive
net result.

**Fix:** Use an arm-specific collateral term and threshold. Print non-selectable
points beyond the registered grid so an edge optimum is visible.

**Done:** Each threshold sweep reports expected and realized benefit and damage,
and its objective matches the intervention form.

## 27. Marcus Smart 2022 has an incorrect supermax ceiling

**Severity:** low.

**Problem:** `train._compute_max_eligible` gives Smart 2022 a 35% ceiling by
using an award earned after his 2021 signing. The correct tier is 30%.

**Reproduce:** Load the evaluation frame and inspect `max_eligible_pct` and
`tier_ceiling_pct` for Marcus Smart 2022.

**Fix:** Use the signing-date instrument in the Stage-1 ceiling calculation and
audit all supermax rows. A curated `designated_ineligible.csv` override is the
minimal fallback.

**Done:** Smart 2022 reads 30%, all supermax rows pass the timing audit, and the
number-moving change has paired metrics and a version entry.

## 28. Spotrac totals disagree with extension salary schedules

**Severity:** medium.

**Problem:** Zubac 2025, Gordon 2026, and Smart 2022 have BBRef salary schedules
whose totals disagree with Spotrac contract totals. This changes
`prev_cap_pct` and future target values. The Spotrac salary migration in #50
may supersede the row-level repair.

**Reproduce:** Compare each player's rows in `salaries.csv` with
`contract_signing_dates.csv.total_value` and the cached per-season Spotrac cap
hits.

**Fix:** Source each season from Spotrac. Update
`salary_corrections.csv` only for residual discrepancies and remove inversions
that were based on a presumed raise multiple.

**Done:** Per-season values reconcile with sourced contract totals and all
affected corrections have `confidence=verified`.

## 35. Row-count changes reshuffle GroupKFold

**Severity:** low.

**Problem:** `GroupKFold` rebalances after rows are removed. A common-row delta
can therefore measure fold reassignment. The 944-to-868 rookie-contract change
moved from raw Delta R-squared -0.0044 to -0.00026 with a fixed player-to-fold
map.

**Reproduce:** Score old and new frames with ordinary `GroupKFold`, then with
the same deterministic player-to-fold map in both frames.

**Fix:** Add a shared fixed-fold common-row comparison and use it whenever frame
membership changes.

**Done:** The evaluation suite exposes the fixed-fold comparison and reports
membership changes separately from prediction changes.

## 36. Stretched dead money enters training as signed salary

**Severity:** medium. This fabricates target values.

**Problem:** Beal 2025, Noah 2020, Batum 2020, and Deng 2019 contain a waiving
team's dead-money charge. Load-time salary overrides exist, but the persisted
training CSV and contract-structure fields can remain wrong.

**Reproduce:** Run `python scripts/audit_stretched_salaries.py`. Also compare
Deng's salary team with his impact-metric team.

**Fix:** Apply verified salary overrides during dataset construction and add a
contract-structure correction layer for the four rows. Coordinate with #51.

**Done:** The audit exits zero, persisted training rows carry signed salary, and
`year_in_contract` plus `contract_years` no longer describe stretch
annuities.

## 37. A prorated deal passes the flat floor filter

**Severity:** low.

**Problem:** Javonte Green 2024 is a prorated in-season deal slightly above the
1.2% cutoff and reaches the evaluation frame as annual salary.

**Reproduce:** Run `python scripts/make_error_board.py` or inspect Green 2024
in `training_data_v2.csv`.

**Fix:** Source his annualized pay and audit rows within 10% above the floor for
short in-season spans.

**Done:** Green carries an annual-rate salary or is excluded, and the near-floor
partial-season audit covers every season.

## 39. `laker_z` missingness is season-aligned

**Severity:** medium.

**Problem:** After the 2026-08-05 recovery pass, 2023 and 2026 LAKER coverage
remain below 85%. Missing values are median-filled. A tested
`laker_known` feature failed the paired gate and is disabled.

**Reproduce:** Build the frame before imputation and cross-tab
`laker_z.isna()` by season. Compare with source-table coverage.

**Fix:** Re-fetch when nbarapm.com adds the missing player-seasons, then retest
`laker_known` as an increment over the production model.

**Done:** Source coverage is restored or a known/imputed signal passes the
paired gate.

## 40. Unknown waiver status is encoded as zero

**Severity:** medium.

**Problem:** Unknown `is_waived` values median-fill to 0, which is
indistinguishable from known-not-waived. The recovery logic added on 2026-08-04
has not been confirmed in a final rebuild. `is_waived_known` failed the paired
gate and is disabled.

**Reproduce:** Build the frame before imputation and cross-tab `is_waived`
with `is_waived_known`.

**Fix:** Rebuild with `_resolve_waiver_no_signing()`, report the remaining
unknown rows, and preserve the known flag for diagnostics.

**Done:** The rebuild includes the recovered rows and every remaining unknown
is reported separately from known zero.

## 42. `prev_cap_pct` hides its minimum fallback

**Severity:** low-medium.

**Problem:** Missing prior salary is stored as the same 0.014848 cap share in
66 historical reference rows. NaN checks treat these imputed values as
observed.

**Reproduce:** Count `prev_cap_pct` rounded to six places in
`training_data_v2.csv` and compare 0.014848 with
`min_2yr[season] / CAP_BY_SEASON[season]`.

**Fix:** Emit `prev_cap_pct_known` from `_load_prev_season_cap_pct` and carry
it into the persisted frame and diagnostics.

**Done:** Coverage audits distinguish observed and imputed prior salary.

## 46. Web export signing-offset metadata is incomplete

**Severity:** low-medium.

**Problem:** Historical web rows now pass `signing_type` and
`SIGNING_OFFSETS_DEPLOYED`, and export metadata records the offsets. The
metadata still omits `SIGNING_K`, and the historical export has not been
closed against the current champion after the salary migration. Upcoming free
agents must receive no outcome-dependent signing offset.

**Reproduce:** Inspect both `stages.compose` calls in
`scripts/export_web.py` and compare exported headline metrics with the
`champion` block in `evaluation_suite.json`.

**Fix:** Add `SIGNING_K` to export metadata and compare historical export
metrics with the `champion` block after the salary migration.

**Done:** Historical export metrics match the champion and the upcoming-free-
agent path applies a zero signing offset.

## 47. Signing residual diagnostics use the baseline model

**Severity:** low.

**Problem:** `scripts/diagnostics.py` builds signing-type residual tables from
a one-seed plain XGBoost model. The tables do not describe the champion stack.

**Reproduce:** Run `python scripts/diagnostics.py` and compare its signing
tables with `oof_reference.csv.oof_champion`.

**Fix:** Build the tables from champion OOF predictions. Keep plain XGBoost only
as a labelled baseline.

**Done:** Each residual table names its model stack and reports champion-stack
biases.

## 48. Deployed signing offsets go stale after a rebuild

**Severity:** medium.

**Problem:** `SIGNING_OFFSETS_DEPLOYED` is hard-coded from OOF residuals.
Changing the frame invalidates it without a failing check. Evaluation layers A
and B estimate fold-local offsets and are unaffected.

**Reproduce:** After rebuilding, run
`OMP_NUM_THREADS=6 python scripts/eval_stage3_signing.py` and compare the
`DEPLOYED-FORM OFFSETS (k=20)` block with `src/model/stages.py`.

**Fix:** Write the deployed offsets to a tracked data file and make the rebuild
or validation fail when the file is stale.

**Done:** Offset regeneration is checked automatically and deployed consumers
match the current frame.

## 49. Rookie prior-pay fill pools contract years 2 through 4

**Severity:** low.

**Problem:** `_rookie_scale_fill_map` uses all observed rookie-scale years.
The economic predecessor of a first veteran contract should use the earliest
observed rookie year. The current difference is 0% to 4% relative and affects
26 historical evaluation rows.

**Reproduce:** Compare per-pick medians from all rookie-scale pairs with medians
restricted to `season == draft_year + 1`.

**Fix:** Build the fill from `draft_year + 1` rows, preserve monotonicity and
the beyond-pick-30 fallback, then run the paired gate and #48 regeneration.

**Done:** The fill uses earliest observed rookie pay and the number-moving
rebuild has a version entry.

## 50. BBRef future salaries are unreliable

**Severity:** high.

**Problem:** BBRef future columns retain stale max projections, declined
options, and unmarked dead money. Five genuine 2026 max signings were removed
by `_filter_mislabeled_year1` because their stale salaries exceeded the legal
ceiling. `check_caps.py` also reads an uncorrected persisted frame.

**Reproduce:**

```text
python scripts/check_caps.py
rg "49,800,000" data/raw/html_cache -g "*contracts_MIA*"
```

**Fix:** Complete the Spotrac `Cap Hit` migration for current and future
seasons. Make `check_caps.py` read corrected salaries and distinguish extension
first-paying years from new free-agent contracts.

**Done:** The salary chain no longer uses BBRef future values, the five max
signings remain in frame, and `check_caps.py` passes for the correct reason.

## 51. Stretch corrections do not cover all seasons

**Severity:** medium.

**Problem:** Beal's PHX stretch annuity persists in 2026 through 2029, and
Jonathan Isaac 2026 combines an $8M dead-cap charge with his new Orlando salary.
Both can enter salary-derived features or targets.

**Reproduce:** Run `scripts/audit_stretched_salaries.py` over all seasons and
inspect Beal plus Isaac in `salaries.csv`.

**Fix:** Source each season from Spotrac cap hits, update salary overrides where
the migration does not supersede them, and repair Beal's contract structure
through #36.

**Done:** No stretch annuity is stored as signed salary and the audit passes over
the full table.

## 54. Training teams are one season stale

**Severity:** low. This affects metadata and display.

**Problem:** `team_abbreviation` comes from the lagged performance season, so
Lillard 2023 reads POR although the priced salary belongs to MIL. The web export
uses `player_teams.csv`, but the persisted training data can remain stale.

**Reproduce:** Inspect Lillard 2023, VanVleet 2023, Chris Paul 2023, and Beal
2023 in `training_data_v2.csv`.

**Fix:** Join `player_teams.csv` in `build_dataset.py` and remove the web-only
fallback after coverage is verified. Keep this metadata change separate from a
target change.

**Done:** Resolved training rows agree with `player_teams.csv` and Lillard 2023
reads MIL.

## 56. Minimum-contract lengths disagree with Spotrac

**Severity:** low-medium. This changes some minimum-row targets.

**Problem:** The one-year cap-charge rule (#55) reads `contract_years` from
`contract_structure_v2.csv`, which counts consecutive salary seasons. For some
minimum deals the Spotrac contract block states a different length. Oshae
Brissett 2023 is a two-year minimum stored as one year, so his $2,165,000 is
wrongly charged at $2,019,706. Bol Bol 2023 is a one-year minimum stored as two
years, so his charge is wrongly left at $2,165,000. On v5.3.3, 44 of the 232
Minimum frame rows with a starting Spotrac minimum block disagree on length;
24 are two-year blocks stored as one year (Marc Gasol 2020, Rajon Rondo 2019)
and so carry the one-year charge. The candidate list is
`outputs/experiments/issue56/length_candidates.csv`.

**Reproduce:** For each frame row labelled Minimum, compare `contract_years`
with the `contract_years` of the Spotrac Minimum block whose `contract_start`
equals the row's season in `spotrac_signing_types.csv`.

**Fix:** Add a sourced `contract_structure_corrections.csv` row wherever the
Spotrac block and the table disagree, after checking each block on its page.
Do not change the length in the merge layer alone: `_normalize_vetmin_caphold`
reads the corrected table at load time and would override it.

**Done:** Every Minimum frame row's `contract_years` matches its Spotrac block
or has a documented reason, and the number-moving rebuild has a version entry.

## 57. A partial or held-out season distorts workload features

**Severity:** medium. It produces large single-row misses.

**Problem:** Workload features (`mpg`, games, `availability_3yr`) read the
season before the signing. A player who sat out most of it by choice or
injury looks like a fringe player. Andre Iguodala 2020 sat out his Memphis
season and played 21 games for Miami, so the champion priced him at $2.33M
against a $14.06M extension.

**Reproduce:** In `outputs/models/oof_reference.csv`, list rows whose
champion error exceeds $8M and whose priced-season games are under 30;
compare each with the player's previous full season.

**Fix:** Flag held-out and injury-shortened seasons from the transaction log
and games played, and read workload from the last representative season for
those rows. Test it as a paired change on the affected rows.

**Done:** Flagged rows use a representative season, and the paired gate on
them is reported.
