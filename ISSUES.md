# Open issues

This file contains active defects only. Each issue must include a problem,
reproduction, fix, and completion check. Delete fixed entries and record the
landed change in `VERSION_HISTORY.md`.

Issue numbers are permanent. The highest assigned number is 63, so the next
issue is 64. Keep entries in numeric order because code and historical reports
refer to them by number.

## Retired references still used in code

- #2: continuation-span warning, now in `CLAUDE.md`.
- #31: Kendrick Nunn contract-structure correction, commit `8071745`.
- #38: veteran-minimum cap-charge convention, landed in v5.0.1.
- #41: Spotrac page-identity guard; Josh Gray remains the documented no-page
  case.
- #35: fixed player-to-fold hash for repeated grouped CV, landed in v6.0.0.
- #55: one-year rule for the minimum cap charge, landed in v5.3.3.
- #58: no-signing waiver window year, landed in v6.0.5.
- #60: signing-offset harness on the KF champion, landed in v6.0.6.
- #50: BBRef future salaries, replaced by the Spotrac migration in v6.0.0.

## 4. Seven signing-mechanism labels remain unknown

**Severity:** low. Signing Mechanism is a diagnostic label.

**Problem:** The 885-row reference frame has seven Unknown rows. Josh Gray
2020 has no Spotrac page. Abdel Nader 2020 falls between contracts. Brandon
Williams 2026 and Nick Richards 2026 are unlabelled 2026 signings. Josh Hart
2024, Terance Mann 2025, and Kevin Durant 2026 are extensions. Luol Deng 2019
reads Cap Space on a minimum-scale salary. Extensions are grouped with Bird
Rights, so retention-premium analysis must split extensions first.

**Reproduce:** List `signing_cat == "Unknown"` rows in
`outputs/models/oof_reference.csv`, and inspect Deng 2019.

**Fix:** Add sourced overrides for decidable rows. Keep Signing Mechanism out
of `FEATURE_COLS`.

**Done:** Every remaining Unknown row has a documented reason and retention
diagnostics separate extensions.

## 5. Team or position is missing on 5.2% of web rows

**Severity:** low. This affects display and filtering.

**Problem:** The site export of 2026-09-27
(`WebPage/public/data/nba/valuations.json`) has 168 of 3,257 rows without
team (`t`, 122 rows) or position (`p`, 47 rows). They are exported as null, so
the board silently omits them from filters. Team correctness is tracked in #54.

**Reproduce:** Count null `t` and `p` values in the current web export and
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

**Problem:** `evaluate_suite.zone_scorecard` takes a `sel_mask`, but only
`scripts/eval_route_mixture_p3.py` passes it. The zone gates in
`eval_floor_branch.py`, `eval_pmax_stage1.py`, and `eval_ringchase_gated.py`
have not been checked for confirmation rows. Some experiment scripts also
define a predicted-value band separately for each model, so reference and
candidate are scored on different rows.

**Reproduce:** In each harness above, find the zone gate and check whether it
excludes `is_confirmation` and fixes segment rows from the reference model.

**Fix:** Route every adoption gate through `zone_scorecard` with `sel_mask`
and a reference-fixed segment. Pooled zone metrics may remain labelled
reporting metrics.

**Done:** Every adoption gate excludes `is_confirmation`, both models use the
same segment rows, and the printed protocol identifies the decision metric.

## 24. The extension raise rule has no signing-season test

**Severity:** low.

**Problem:** `cba_era` starts in repo season 2024; the extension raise multiple
changes for deals signed in repo season 2023. `METHODOLOGY.md` documents both
boundaries, but no test covers the signing-season rule.

**Reproduce:** Search `tests/` for a case at 1.20 and 1.40. Jimmy Butler 2023
was signed in 2021 and belongs to the 1.20 rule.

**Fix:** Add a test that assigns the multiple by signing season, with Butler
2023 as a fixture.

**Done:** The test passes, and it fails if the multiple keys on `cba_era`.

## 25. Tau selection models only probability-weighted interventions

**Severity:** medium.

**Problem:** The current collateral objective uses
`sum(P * intervention)`, which estimates a probability-weighted push or pull.
It understates an unconditional intervention and can select a threshold at the
edge of the search grid. Since v6.3.0, `TAU` acts only in the KF measurement
model; the queue item that removes the push decides whether this issue
still applies.

**Reproduce:** Run
`OMP_NUM_THREADS=6 python scripts/eval_floor_branch.py` and inspect the
expected-versus-realized table. The historical unconditional floor arm had
$338.1M benefit and $514.9M damage while the objective predicted a positive
net result.

**Fix:** Use an arm-specific collateral term and threshold. Print non-selectable
points beyond the registered grid so an edge optimum is visible.

**Done:** Each threshold sweep reports expected and realized benefit and damage,
and its objective matches the intervention form.

## 27. Supermax ceilings lack a timing audit

**Severity:** low.

**Problem:** `bc5bda9` anchored the supermax award test on the signing season,
and Marcus Smart 2022 now reads 30%. No record shows that every other
supermax row was audited against its signing date.

**Reproduce:** List rows with `max_eligible_pct == 0.35` and compare the award
season with `contract_signing_dates.csv.signing_season`.

**Fix:** Run the audit and add `designated_ineligible.csv` rows for any
failure.

**Done:** All supermax rows pass the timing audit; a number-moving change has
paired metrics and a version entry.

## 28. Two salary corrections are inferred from the CBA rule

**Severity:** medium.

**Problem:** Salaries now come from Spotrac cap hits per season, but
`salary_corrections.csv` still holds two `inferred` `prior_base` rows: Ivica
Zubac 2024 (13,495,700) and Aaron Gordon 2025 (24,041,455). Both invert the
1.40 extension rule through observed pay. Gordon's 2025 salary in
`merged_salaries.csv` (22,841,455) duplicates his 2024 figure. Smart 2022's
`pay_above_base` row is a residual.

**Reproduce:** Filter `data/raw/raw_external/salary_corrections.csv` for
`confidence != verified`, and compare Zubac and Gordon in
`merged_salaries.csv` with `contract_signing_dates.csv.total_value`.

**Fix:** Source each prior season from Spotrac. Remove corrections that the
sourced value makes redundant.

**Done:** Every correction has `confidence=verified` or a sourced residual,
and the affected per-season values reconcile with contract totals.

## 36. Stretch annuities remain in contract structure

**Severity:** medium.

**Problem:** Salaries for Beal 2025 and 2026, Noah 2020, Batum 2020, Deng
2019, and Isaac 2026 now carry signed salary, and
`scripts/audit_stretched_salaries.py` exits zero. Contract structure still
describes the waiving team's contract: Beal 2025 and 2026 read
`contract_years = 5` (the PHX stretch), and Deng 2019 reads 3 for a one-year
Minnesota deal.

**Reproduce:** Inspect `contract_years` and `year_in_contract` for these rows
in `data/processed/training_data_v2.csv`.

**Fix:** Add `contract_structure_corrections.csv` rows sourced from each
player's signing contract block.

**Done:** No row's `year_in_contract` or `contract_years` describes a stretch
annuity.

## 37. No audit covers near-floor partial-season deals

**Severity:** low.

**Problem:** Javonte Green 2024 was a prorated in-season deal above the 1.2%
cutoff. The vet-min normalization now stores him at the base minimum, but no
audit checks other short in-season spans near the floor.

**Reproduce:** List frame rows within 10% above the floor whose signing date
falls after the season opener.

**Fix:** Source the annual rate for each flagged row or exclude it.

**Done:** The near-floor partial-season audit covers every season.

## 39. `laker_z` coverage is low in 2023

**Severity:** medium.

**Problem:** On the 885-row reference frame, `laker_z` coverage is 75.5% in
2023 and at least 94.9% in every other season. Missing values are median-
filled. A tested `laker_known` feature failed the paired gate and is disabled.

**Reproduce:** Join `training_data_v2.csv` to `oof_reference.csv` and
cross-tab `laker_z.isna()` by season.

**Fix:** Re-fetch when nbarapm.com adds the missing 2023 player-seasons, then
retest `laker_known` as an increment over the production model.

**Done:** 2023 coverage is at least 85%, or a known/imputed signal passes the
paired gate.

## 40. Unknown waiver status is encoded as zero

**Severity:** medium.

**Problem:** On the 885-row reference frame, 53 rows have unknown `is_waived`
(`is_waived_known == 0`). The model median-fills them to 0, the same as
known-not-waived. `is_waived_known` failed the paired gate and is disabled.

**Reproduce:** Join `training_data_v2.csv` to `oof_reference.csv` and cross-tab
`is_waived` with `is_waived_known`.

**Fix:** Report the unknown rows and try to resolve them from
`spotrac_transactions.csv`. Keep the known flag for diagnostics.

**Done:** Every remaining unknown has a reason in a tracked report.

## 42. `prev_cap_pct` hides its minimum fallback

**Severity:** low-medium.

**Problem:** Missing prior salary is stored as the same 0.014848 cap share
in 92 of the 885 reference rows. NaN checks treat these imputed values as
observed.

**Reproduce:** Join `training_data_v2.csv` to `oof_reference.csv` and count
`prev_cap_pct` rounded to six places equal to 0.014848.

**Fix:** Emit `prev_cap_pct_known` from `_load_prev_season_cap_pct` and carry
it into the persisted frame and diagnostics.

**Done:** Coverage audits distinguish observed and imputed prior salary.

## 46. Web export signing-offset metadata is incomplete

**Severity:** low-medium.

**Problem:** Historical web rows now pass `signing_type` and
`SIGNING_OFFSETS_DEPLOYED`, and export metadata records the offsets. The
metadata still omits `SIGNING_K`. The site data of 2026-09-27 predates
v6.2.0 (its rows have no `wv` or `pm`), so no export has been checked against
the v6.3.0 champion. Upcoming free agents must receive no outcome-dependent
signing offset.

**Reproduce:** Inspect both `stages.compose` calls in
`scripts/export_web.py` and compare exported headline metrics with the
`champion` block in `evaluation_suite.json`.

**Fix:** Add `SIGNING_K` to export metadata and compare historical export
metrics with the `champion` block on a fresh export.

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

## 48. Nothing detects stale signing offsets

**Severity:** medium.

**Problem:** `stages.py` loads the deployed offsets from
`data/raw/raw_external/signing_offsets.json`, which
`scripts/eval_stage3_signing.py` writes. A rebuild that changes the frame does
not fail when the file is older than the frame. Evaluation layers A and B
estimate fold-local offsets and are unaffected.

**Reproduce:** Rebuild the training data without running
`scripts/eval_stage3_signing.py`, then run `scripts/export_web.py`. It uses
the old offsets without a warning.

**Fix:** Store a frame fingerprint in `signing_offsets.json` and make
`export_web.py` fail when it does not match the current frame.

**Done:** A stale offset file stops the export with a regeneration command.

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

## 59. `is_waived` stays on after the market has re-priced the player

**Severity:** medium. It affects half of the waived rows.

**Problem:** `is_waived` is 1 when any waiver falls in the 365 days before the
signing, even when another contract was signed in between. That contract has
already re-priced the player. Of the 120 waived evaluation rows, 62 carry such
a stale flag, including:

- Josh Okogie 2026: cut 2025-07-15, a minimum with Houston, then the MLE
  with Utah 359 days after the cut.
- Marcus Smart 2026 and Andre Drummond 2021.

On plain waivers above the floor, the champion under-predicts both groups:

| Flag | Rows | Bias |
|---|---:|---:|
| Stale | 17 | -$1.62M |
| Fresh | 12 | -$0.91M |

Fresh money-owed waivers above the floor are over-predicted by $6.40M on
8 rows.

**Reproduce:** For each `is_waived == 1` row, look for a `signed` event
between `prior_waiver_date` and July 1 of the row's season in
`data/processed/spotrac_transactions.csv`. Group the champion OOF error in
`outputs/models/oof_reference.csv` by that result.

**Evidence against the plain fix (2026-09-29):** The candidate run applied
this definition together with #58 and was reverted in `b2843e1`. A1 fell
from 0.8509 to 0.8471; paired selection dSel was -0.0013 (t = -0.35). The 63
rows whose flag turned off worsened: MAE rose from $1.30M to $1.92M, and bias
went from +$0.11M to +$1.19M. Marcus Smart 2026 rose from $6.07M to $11.03M
against $6.06M, although his anchor had already been re-priced. A waiver
inside the last year still carries signal after an intervening contract.
`attach_waiver_history` documents that 365-day definition, so it is not a
contract violation.

**Fix:** The bias table above predates v6.2.0, which applies the waiver term
to money-owed waivers and returns `is_waived` to the regression. Re-measure the
stale and fresh groups on the v6.2.0 champion first, then test any new
definition as a feature change.

**Done:** The fresh definition is adopted or rejected with paired metrics.

## 61. The site's what-if ignores the Stage-1 waiver term

**Severity:** medium. It affects money-owed waiver rows on the Value Board.

**Problem:** Since v6.1.0 the latent is the tree sum plus
`beta * wv * kf_market_value`, where `wv` marks money-owed waivers (v6.2.0).
`scripts/export_web.py` writes beta to `model.json` (`waiver_beta`) and
`meta.json` (`waiverTerm`), the row flag to `valuations.json` (`wv`), and the
term's attribution to `shap.json` under `waiver_term`. The site (`WebPage/`, outside this repository) traverses only
the trees, so its what-if value for a waived row misses the discount, and its
SHAP waterfall does not render the `waiver_term` key.

**Reproduce:** Export, open a money-owed waiver row (`wv == 1`) in the
what-if, and compare its unedited value with `latent_cap_pct` in
`valuations.json`.

**Fix:** In the site, add `waiver_beta * wv * kf_market_value` (the row's
edited `kf_market_value` input) to the tree sum, and render `waiver_term`
with the `waiverTerm.label` from `meta.json`.

**Done:** The site's unedited what-if value equals `latent_cap_pct` on every
`wv == 1` row, and the waterfall sums to it.

## 62. The v6.2.0 champion does not reproduce on the Windows machine

**Severity:** medium. Absolute metrics from this machine cannot be compared
with published ones.

**Problem:** v6.2.0 records A1 0.8612 / A2 0.8648 / B1 0.8516, measured in a
cloud session. The same code at `1a4cfaf` on the Windows machine (XGBoost
3.3.0, `OMP_NUM_THREADS=6`) gives A1 0.8603 / A2 0.8641 / B1 0.8517 for the
champion arm of `scripts/eval_pmax_stage1.py`. The code before and after
`1a4cfaf` gives the same one-seed A1 (0.8552), so the change is not the
cause. Paired deltas inside one run are still valid.

**Reproduce:** On the Windows machine, run
`python scripts/eval_pmax_stage1.py --seeds 10 --full` and read arm `R`.

**Fix:** Record the package versions and platform of each environment. Run
the one-seed champion in both and find the first stage whose output differs.
Then pin the versions, or record one reference environment for published
numbers.

**Done:** Both environments give the same champion A1 to four decimals, or
`VERSION_HISTORY.md` names the reference environment and the other's offset.

## 63. The site's what-if ignores the Stage-1 P(max) term

**Severity:** medium. It affects every row with P(max) > 0 on the Value
Board.

**Problem:** Since v6.3.0 the latent also adds
`max_beta * pm * max(max_eligible_pct - kf_market_value, 0)`.
`scripts/export_web.py` writes `max_beta` to `model.json`, `maxTerm` to
`meta.json`, the row's P(max) as `pm` in `valuations.json`, and the term's
attribution to `shap.json` under `max_term`. `meta.json` no longer has `tau`
or `margin`. The site (`WebPage/`, outside this repository) traverses only
the trees, so its what-if misses the term, and its waterfall does not render
`max_term`.

**Reproduce:** Export, open a row with a high `pm` in the what-if, and
compare its unedited value with `latent_cap_pct` in `valuations.json`.

**Fix:** In the site, add `max_beta * pm * max(max_eligible_pct - kf, 0)`
(with the row's edited `kf_market_value`) to the tree sum, render
`max_term` with `maxTerm.label`, and remove any use of `meta.tau` and
`meta.margin`.

**Done:** The site's unedited what-if value equals `latent_cap_pct` on every
row, and the waterfall sums to it.
