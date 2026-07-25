# Open issues

Known problems that are real but were out of scope when found. Each entry is
written to be actionable without the conversation that produced it: what is
wrong, how to see it for yourself, what to do, and how to know you fixed it.

**If you find something you are not fixing right now, add it here** rather than
leaving it in a chat log. Delete an entry when it is fixed — this file is a
worklist, not a changelog. `VERSION_HISTORY.md` is where fixes get recorded.

**Numbers are permanent and are never reused.** A new entry takes `max + 1` over
every number this file has *ever* used, not the first gap — deleting a fixed
entry retires its number for good. The highest ever used is **28**, so the next
new entry is **29**. Two parallel workers each taking "the next free number"
is exactly how the two `#20`s and two `#25`s of 2026-07-27 happened.

Entries are listed in **numeric order**, not by severity — the file is looked up
by number from ~200 references across `docs/briefs/`, `docs/QUEUE.md` and code
comments (`grep -rn "ISSUES #"`). Each entry states its own severity in its first
line. References in dated `*.RESULT.md` files are to the numbering current at
their date; where that differs from today's the entry says so.

---

## 2. Continuation rows: the part deliberately left in

**Severity**: low — everything decidable is gone as of v7.9x; what remains is
kept on purpose.

Closed in three steps. v7.6x: six rows paid above their *tier* ceiling
(`_filter_mislabeled_year1`). v7.7x: 119 rows on a three-signal consensus
(+0.0102 common-row A1). v7.9x: the dated-span branch — option-aware anchors,
extension fallback +1, renegotiation carve-out — demotes 342 rows total (335
by dated span, 7 by the consensus fallback), frame 1,291 → 949, zero demotions
contradicted by any instrument. The `is2019` season offset that quantified the
remaining contamination (+0.0035, t=2.29 at v7.8x) measures −0.0004 (t=−0.91)
on the new frame: the stale class is gone from where it could be seen.

**A warning that must outlive this entry**: a BLOCK-anchor span alone is NOT
sufficient. A span-only rule once deleted Brunson 2022, VanVleet 2023 and
Jimmy Butler 2019 — all genuine fresh signings; Spotrac's Free-Agent anchor is
unreliable for contracts later superseded or option-final. The v7.9x dated
branch is span-based but keeps the FA-signings-list veto for exactly that
reason: 19 rows whose dated span reads continuation but who sit in that
season's FA list are KEPT (Horford 2019, Draymond 2023, …). **Do not remove
the veto.**

What remains, deliberately: rows no dated contract covers fall back to the
three-signal consensus, and rows whose last-season pay is also unobservable
are kept outright — precision over recall; Millsap 2019 was the canonical kept
suspect. Expanding `salaries_prehistory.csv` or the player-page transactions
cache converts more of them into decidable rows — and did: the 2018 team-salary
scrape (2026-07-24) made five of them decidable and demoted (Millsap, Snell,
Galloway, Okafor, Ferrell — all escalator continuations, none in the FA list),
frame 949 → 944.

**Verify** (state after the prev_cap_pct landing): evaluation frame 944 rows;
2019 year-1 count 142; the filter prints "dropped 347 rows (335 by dated span,
12 by three-signal consensus)".

---

## 4. Signing-mechanism labels: the residual 10%

**Severity**: low — the labels are diagnostics, never features.

Largely fixed 2026-07-23: `scripts/refresh_spotrac.py` rebuilt
`spotrac_signing_types.csv` with anchor-based season assignment (each contract
places itself at `fa_year − n … fa_year − 1` instead of the old backward walk,
which misaligned whenever a page skipped a deal), salary-aware disambiguation
for mid-season buyouts (Westbrook 2022 was a $46M supermax row labeled
"Minimum"), and player pages for extension signees harvested via the
upcoming-FA URL directory. Year-1 evaluation rows are now 85–91% labeled per
season (was ~43% overall); Minimum-labeled rows above $6M fell from 30 to 6.

What remains, for whoever next touches the labels:

- ~10% of evaluation rows are still Unknown — mostly two-way conversions and
  Exhibit-10s whose Spotrac pages carry no "Signed Using" block. Reproduce:
  `python scripts/refresh_spotrac.py --reparse-only` prints the per-season table.
- Six >$6M "Minimum" rows persist (Winslow 2019, Fultz 2019/20, Leonard 2020,
  Batum 2020, Noah 2020) — stretched/waived money colliding with a same-season
  minimum where the AAV-distance rule picks the wrong side.
- Extensions land under "Bird Rights" (n jumped 58 → 353), conflating
  re-signings with extensions. If the C2 mechanism slice is ever used to argue
  a retention premium, split these into their own category first.

Keep the field out of `FEATURE_COLS` — `METHODOLOGY.md` documents that
modelling it *lowers* CV R² by 0.0073; its value is as an out-of-sample
diagnostic.

---

## 5. Team and position missing for 5.4% of rows

**Severity**: low — cosmetic on the site, unused by the model.

167 of 3,113 rows have no `team_abbreviation` or `position` after the join
against `data/processed/training_data.csv` (still 5.4% as of the 2026-07-22
export). The valuation board renders these as "—". Neither field is a model
feature, so this affects presentation and the team filter only.

Root cause is the `team` column disagreeing between sources: the 2019-2024 rows
come from BBRef player salary pages and the 2025+ rows from team contract pages,
and a rebuild reproduces `team` on only 53.8% of rows (`position` on 94.6%) while
every one of the fourteen model features reproduces exactly. Cosmetic by
construction — but it means a team filter on the site silently omits rows rather
than showing them as unknown.

---

## 6. Two boundary rules rest on hand-curated or derived tables

**Severity**: low — both work today; both are maintenance the next refresh pays.

Stage 2's bounds are exact where the CBA is exact and approximate where the repo
lacks a source:

- **`early_supermax.csv`** enumerates designated-veteran deals signed two or more
  summers before they take effect (Wall 2019, Towns 2024), because no award
  window anchored to the start season can reach them. A blanket `s-3` lookback
  was tried at v7.4x and reverted at v7.5x — it un-censored two genuine 30% max
  signings. **Do not reintroduce a wider window.** The systematic fix is a
  signing date per contract, which Spotrac exposes and the scrape does not yet
  take; that would also let `_filter_continuations` drop its step-and-veto
  signals for a direct test.

- **`designated_ineligible.csv`** is the inverse curated list (added v7.12x,
  ISSUES #19): rows the award path wrongly bumps to a 35%/30% designated ceiling
  because it fires on All-NBA + experience alone and cannot see that the player
  changed teams (Kawhi 2019), was acquired on a veteran deal (AD 2020), or
  signed the 25% base of a rookie-extension whose Rose escalator lands in a later
  year (KAT 2019, Tatum 2021). 12 rows; each hand-verified. Same team-continuity
  data gap as early_supermax — a signing-date team-match test would replace both
  lists. Maintenance: a new such deal each summer needs a row.

  The signing dates are now scraped —
  `data/processed/contract_signing_dates.csv`, 3,169 dated signings — and they
  confirm both curated rows at a 2-summer gap (Wall signed 2017-07-26 for
  2019-2022, Towns 2022-07-07 for 2024-2027). Five further max-tier evaluation
  rows are also signed 2+ summers early (Booker 2024, Lillard 2021, Davis 2025,
  Harden 2019, Adebayo 2026), all extensions, and all already receive the
  correct tier ceiling through the award-window path — over-tier rows among
  them are 0, so nothing is broken today. `early_gap >= 2` in
  `parse_signing_dates.contract_spans()` is now a mechanical test that could
  replace both the curated list and the award window; that is a
  `_compute_max_eligible` change and wants its own dispatch.

  **The fourth payoff claimed here does not exist — struck 2026-07-23.** The
  exit-structure ablation found the expiring contract's length worth +0.0032
  paired (t=2.32) as a pure increment over the coverage artifact, and this
  entry predicted that transaction-level signing dates would lift coverage
  "toward ~95%" and unlock it. Measured, they do the opposite: the dated
  instrument sees *fewer* expiring contracts than the Spotrac blocks already
  do (a contract ending at S-1: blocks 60.7%, dates 50.8%, union 62.4%; any
  contract covering S-1: 87.3% / 79.4% / 88.1%). The blocks table knows 2,727
  distinct contracts against 1,953 dated ones, because a contract reaches the
  transactions list only if Spotrac logged the transaction while the contract
  history retains deals whose transaction entry it never had. Note also that
  the 54% quoted above could not be reproduced from any definition — the
  ablation script was a scratchpad experiment and is gone; the nearest
  construction is 60.7%. **The +0.003 stays locked, and signing dates are not
  the key to it.** Numbers in `docs/briefs/2026-07-23-signing-dates.RESULT.md`,
  target 3. RFA status, by contrast, tested empty once the artifact was removed
  (+0.0004), and option structure is unmeasurable historically — Spotrac
  annotates options on current contracts only (has_po is 0.000 for every
  fa_year before 2023 and 0.35-0.50 for 2028+).
- **`floor_pct`** is the median pay of at-floor rows per (season, experience
  bucket) — a recovery of the veteran-minimum scale from the data's own mass
  points, not the published scale. Buckets with few at-floor rows fall back to
  the season minimum. Accurate enough that the Stage-2 clip lands within
  $0.05-0.09M of observed pay, but a published scale would be exact.

**Verify**: `python -c "from src.model.train import *; ..."` — over-cap rows stay
0, and at-floor rows in the sub-2% predicted band keep bias under $0.10M.

---

## 12. AAV is unavailable for half the evaluation frame, non-randomly

**Severity**: low — informational, and it bounds what any contract-structure
work can attempt.

Established by the 2026-07-23 AAV target-variable memo
(`docs/briefs/2026-07-23-aav-target-memo.RESULT.md`). Only **604 of 1,172**
evaluation rows (51.5%) can be assigned a contract whose total value reconciles
with the row's year-1 pay under a legal escalator. The missing half is not
random: coverage runs 79% above $25M and 27% below $3M, 81% on Bird Rights and
35% on Minimum-labelled rows. The covered subset's mean salary is $12.4M against
$5.5M for the uncovered.

This is the same collection-coverage channel the worker brief's gate 2 warns
about, and it means **any** feature derived from contract structure — length,
AAV, guarantee share, option structure — inherits a missingness pattern aligned
with player quality. Cost it as an increment over an explicit coverage-indicator
arm, never as a raw delta.

The root causes are the two above plus extensions that never appear on a
signed-FA page. (The memo predicted transaction-level signing dates would raise
coverage; the signing-dates shadow analysis, landed the same day, measured the
opposite — the dated instrument covers FEWER contracts than the blocks, union
gain under 2 points. See #8's source table and the RESULT's target 3.)

---

## 17. Four two-way players have no 2018 prior salary

**Severity**: low — the affected rows carry negligible prior pay, so their
`prev_cap_pct` slot-fill fallback is nearly right anyway.

**The name-alias half is fixed** (2026-07-26 cleanup-debt landing): the rename
**Enes Kanter → Enes Freedom** and **Wesley Iwundu → Wes Iwundu** are now mapped
through `data/raw/raw_external/player_name_aliases.csv`, applied in both
`scrape_2018_salaries.py` and `_load_prev_season_cap_pct`. 2018 prior coverage
of the season-2019 evaluation rows went 136/142 → **138/142**.

What remains are four two-way / minimum players whose 2018-19 salary is absent
from the BBRef `salaries2` team table altogether — **Alex Caruso, Shake Milton,
Wenyen Gabriel, Amile Jefferson**. An alias cannot reach them; they need a
two-way salary source the repo does not have. Parked in `docs/QUEUE.md` under
"parking lot / low priority" — none of the four moves a model feature materially.

**Verify**: `_load_prev_season_cap_pct()` returns a value for
`("enes freedom", 2018)` and `("wes iwundu", 2018)`, and exactly those four
season-2019 names remain without a 2018 key.

---

## 20. Three zone-gate protocol defects found while re-judging the max branch

**Severity**: medium for the first, low for the other two — none changed the
2026-07-26 phase-3 verdict, but the first could hand a future candidate an
adoption it did not earn. Found building `scripts/eval_route_mixture_p3.py`;
full numbers in `docs/briefs/2026-07-26-route-mixture-p3.RESULT.md` §8.

**Status 2026-07-28**: still open, but no longer un-implemented anywhere.
`scripts/eval_floor_branch.py` honours all three (its module docstring names
them) and is the working reference implementation to copy from. What is missing
is the systematisation — `evaluate_suite.py`'s `grabit_zone` / `floor_zone`
scorecards and `scripts/diagnostics.py` still pool the confirmation split, and
each new harness re-derives the three rules by hand.

**(a) Zone-MAE gates pool the confirmation split.** The route-mixture "Win"
gate — true-max zone MAE must improve by ≥ $0.50M — is computed over all 68 zone
rows, 12 of which are locked confirmation players. The worker brief's own rule
says the confirmation split must never enter a decision metric, and on this
frame the distinction is not academic: **every phase-3 cell won 2–10× more on
the 12 canary rows than on the 56 decidable ones.**

```
                 zone-MAE win vs champion
                  all      sel      conf
base_tau*       +0.12    +0.15    +0.00
base_tau90      +1.12    +0.76    +2.79
enriched_tau*   +0.42    +0.17    +1.58     <- pooled $0.42M is $0.17M decidable
enriched_tau90  +0.83    +0.54    +2.20
```

The same leak shows in A1: `enriched_tau*` posts pooled A1 0.7906 against the
champion's 0.7878 while its selection-pool paired ΔSel is −0.0001 (t −0.20) —
the whole pooled gain is confirmation rows. The cause is mundane (the 15%
per-player split happens to hold Wall 2019, Kemba 2019 and both Mitchell rows,
four of the six rows carrying the τ\* win), which is exactly why it will recur:
the max zone is 68 rows, so a 12-row subset can carry a headline.

Note this is narrower than "all zone metrics are wrong". `grabit_zone` /
`floor_zone` in `evaluate_suite.py` are *reporting* scorecards for a landed
intervention and pooling is defensible there. The defect is using a pooled zone
MAE as an **accept/reject** bar.

*Reproduce*: `OMP_NUM_THREADS=6 python scripts/eval_route_mixture_p3.py`, the
"WIN GATE split by the confirmation lock" block.
*Fix*: any zone gate that decides adoption computes its MAE over
`~df["is_confirmation"]` rows only, and says so in the printed protocol block.
The reporting number can stay pooled as long as the two are labelled apart.
*Know it is fixed*: the gate table prints a selection-row zone MAE, and
re-running phase 3 shows `enriched_tau*` at +$0.17M rather than +$0.42M.

**(b) `scripts/eval_route_mixture_p2.py` compares moving row groups.** Its 25%+
predicted-band brake scores the champion on the *champion's* ≥25% rows and the
candidate on the *candidate's* ≥25% rows. The worker brief forbids exactly this
for model-vs-model comparison — and the candidate's band is larger by
construction, because pushing rows up moves them into it (55 vs 52 rows at
enriched τ\*, 63 vs 52 at base τ₉₀). The gap is material in magnitude: at
enriched τ₉₀ the brake reads **+$1.08M own-band vs +$0.82M fixed-row**. Both
breach the +$0.30M bar, so no published verdict moves, but the own-band number
is the one quoted in the phase-3 dispatch brief.

*Fix*: score both models on the champion's band (p3's `band25_metrics` does
this and reports both readings). *Know it is fixed*: the two readings are
printed side by side and the gate cites the fixed-row one.

**(c) "C2 |bias| growth" is implemented as a signed change.** `evaluate_suite`'s
protocol text and every brief say a segment fails when its **|bias| grows** by
more than $0.30M; the phase-1/2 harnesses compute `bias_cand − bias_champ` and
threshold its absolute value. These diverge whenever a segment's bias moves
*toward* zero — which is what a max-branch push does to Bird Rights
(−$2.44M → −$2.03M at base τ₉₀). Max signed change +$0.41M would breach; max
|bias| growth is +$0.026M and does not. Phase 3 reports both and gates on the
literal wording.

*Fix*: pick one and make the code and the prose agree — `abs(b_cand) −
abs(b_champ)` matches the documented intent, and a bias shrinking toward zero
should not be scored as a regression. *Know it is fixed*: one number, one name,
and `scripts/diagnostics.py` plus any zone harness use the same helper.

---

## 24. `cba_era` and the extension raise multiple use different CBA boundaries, and both are right

**Severity**: low — documentation debt, but the kind that produces a wrong "fix".

`config.CBA_NEW_ERA_SEASON = 2024` makes `cba_era = season >= 2024`, while the
2023 CBA took effect for the 2023-24 season (season 2023 in our convention) and
the extension work keys its 120%/140% multiple on **signing season >= 2023**.
These are not the same boundary and neither is a bug:

- The extension multiple is a legal quantity fixed when the ink dries, so it must
  key on the signing season. The evidence is knife-edge: 7 rows land on exactly
  1.200 and all 7 were signed in 2022 or earlier; 6 land on exactly 1.400 and all
  6 were signed in 2023 or later; zero crossings. Keying on the PAYING season
  instead misclassifies Jimmy Butler 2023 (signed 2021, paid at exactly 1.200).
- `cba_era` is a market-regime feature, not a legal one, and the market's
  response to the new apron rules is a 2024-season phenomenon.

**What to do**: nothing to the code. Record the distinction in METHODOLOGY.md so
the next agent does not "align" the two and silently break the extension rule.

**Fixed when**: METHODOLOGY.md states both boundaries and why they differ.

---

## 25. The τ-selection objective's collateral term models only a P-weighted arm

**Severity**: medium — it did not mis-select anything that shipped, because the
max side's adopted arm happened to be the one the term models. It would
mis-select for any future unconditional branch, and on the floor side it got the
sign of the net effect wrong.

The τ rule used by `scripts/eval_max_branch_tau_sweep.py` and
`scripts/eval_floor_branch.py` is

```
expected win        = Σ over touched TRUE-class rows of |champion error|
expected collateral = Σ over touched OTHER rows of P × (bound − champion)
τ*                  = argmax (win − collateral)
```

The collateral term is **P-weighted**, so it is the expected damage of a
P-weighted intervention — `latent + P·(margin·bound − latent)`, the push_clip /
pull-with-margin form. It is **not** the damage of an unconditional arm
(`pred = bound` for every row above τ), which pays the full `bound − champion`
regardless of P. The win term is not P-weighted either, so the two halves of the
objective are on different footings, and one τ is nevertheless applied to every
arm in the battery.

Measured on the floor branch at its τ\* = 0.10 (2026-07-27, 944 rows):

| arm | expected win / coll | realized zone reduction | realized damage | realized net |
|---|---|---:|---:|---:|
| A (unconditional pull) | 367.9 / 300.3 | $338.1M | **$514.9M** | **−$176.8M** |
| B (P-weighted pull) | 367.9 / 300.3 | $118.8M | $19.7M | +$99.1M |

The win term is accurate for arm A (367.9 expected vs 338.1 realized, 1.09×)
while the collateral term understates its damage by **1.7×**, so the objective
predicted +$67.7M net where arm A delivered −$176.8M. Arm B, the form the term
does model, is predicted with the right sign. The max branch never surfaced this
because the arm it selected for (push_clip) is P-weighted.

**Reproduce**: `OMP_NUM_THREADS=6 python scripts/eval_floor_branch.py`, the
"PART 4 (a) does the pre-registered objective predict what the arms do?" block.

**What to do**: make the collateral term match the arm it is selecting for —
`Σ (bound − champion)` over touched rows for an unconditional arm, `Σ P × (bound
− champion)` for a P-weighted one — and select a separate τ per arm; or state in
the brief that the τ rule is valid only for P-weighted arms and drop
unconditional arms from τ-swept batteries. Whichever is chosen, print the
realized-vs-expected table beside the gate battery so a future mismatch is
visible rather than inferred.

**A second, cheaper guard found alongside it**: when the objective's argmax
lands on an edge of the τ grid, the grid bound is doing the selecting, not the
rule. The floor sweep's argmax sat on its lower edge (0.10) with the objective
still rising below it (+77.7 at τ=0.08, +65.0 at τ=0.00). Any harness using this
rule should print τ values outside the pre-registered grid as non-selectable
diagnostics — `eval_floor_branch.py` does, marked `*` — so an edge argmax is
caught rather than reported as an operating point.

**Fixed when**: a τ-swept battery prints expected-vs-realized win and collateral
per arm, and either uses an arm-matched collateral term or excludes
unconditional arms from the sweep.

---

## 26. The stored `prev_cap_pct` still carries the pre-correction values

**Severity**: low — three rows, but it is a feature, not a diagnostic.

`salary_corrections.csv` is applied inside `_load_prev_season_cap_pct`, which is
what the ceiling rules and the extension cap read at load time. The `prev_cap_pct`
FEATURE is not read there: it is baked into `data/processed/training_data_v2.csv`
by `scripts/phase3.py::build_contract_features`, which called the same loader when
the table was last built. So Marcus Smart 2022, Ivica Zubac 2025 and Aaron Gordon
2026 train on the stale prior ($13.84M / $11.74M / $22.84M) while their ceilings
use the corrected one.

This was deliberate on 2026-07-27: rebuilding would have moved the champion's
features in the same commit that re-measured the told clip, and the whole point of
that measurement was that the champion is unchanged. It should be picked up by the
next rebuild that happens for another reason.

**Reproduce**: `python -c` comparing `training_data_v2.csv`'s `prev_cap_pct` for
those three rows against `_load_prev_season_cap_pct()`.

**Fixed when**: `scripts/rebuild_training_data.py` has been run after a corrections
change and the three rows agree, with the frame still at 944 rows.

---

## 27. Marcus Smart 2022 keeps a 35% supermax ceiling in `max_eligible_pct`

**Severity**: low — it cannot make him look overpaid (his $17.46M is far below
either tier) but it is the same wrong fact ISSUES #23 fixed, in a second place.

ISSUES #23 (a designated-veteran exemption granted by a DPOY that POSTDATES the
August-2021 signing) was fixed on 2026-07-27 inside
`extension_cap._designated_veteran`, by anchoring the award test on the SIGNING
season. `train._compute_max_eligible` has the same supermax branch and no signing
instrument at all — it sees only the paying season — so Smart 2022 still reads a
35% tier ceiling, $43.28M, in `max_eligible_pct` and `tier_ceiling_pct`.

It was left alone because `max_eligible_pct` is the Stage-1 censor bound and the
max-branch push target: changing it moves the champion, which is a version-bump
change and not what the told-clip brief was measuring. Note the direction — a too-
high ceiling makes the max-branch push aim $25.8M too high for this row, which is
exactly the damage the told clip now removes downstream.

**What to do**: either add `marcus smart, 2022` to `designated_ineligible.csv`
(one line, fixes both paths, moves the champion) or give `_compute_max_eligible`
the signing-date instrument that `extension_cap` already uses. The second is the
real fix and would need the same audit across every supermax row.

**Fixed when**: Smart 2022 reads a 30% tier ceiling, and the champion's A1 is
re-reported at the version that lands it.

---

## 28. Spotrac's contract totals disagree with our salary schedules on extensions

**Severity**: medium — it is the reason two of the ISSUES #22 repairs are marked
`inferred` rather than `verified`, and it may be a second, wider salary defect.

**Renumbered 2026-07-28**: filed as `#25` by
`docs/briefs/2026-07-27-told-clip-and-data-fix.RESULT.md`, which collided with
the τ-objective entry filed as `#25` three minutes earlier on a parallel branch.
That entry keeps 25 (it was committed first and carries more references); this
one moved to 28. The three "ISSUES #25" mentions in the told-clip RESULT mean
**this** entry.

The #22 repair (landed 2026-07-27) reads a stale prior-season salary off the CBA
rule: a veteran extension's first paying year is 1.40x the final year of the deal
being extended, so `observed pay / 1.40` recovers the base our table should have
carried. That inversion is dollar-exact for both rows. But the SAME two contracts
reconcile the other way against Spotrac's headline total, and the two readings
put the missing money in different places:

| row | our prior | pay / 1.40 | Spotrac total | total implied by OUR schedule |
|---|---|---|---|---|
| ivica zubac 2025 | $11.74M | $13.4957M | 3yr/$58.6M | $61.03M |
| aaron gordon 2026 | $22.84M | $24.0415M | 3yr/$103.6M | $109.05M |
| marcus smart 2022 | $14.34M (repaired) | — | 4yr/$76.49M | $78.09M |

Zubac's $58.6M is *exactly* the schedule 18,102,000 / 19,550,160 / 20,998,320 —
the 2024 EAS cap with 8% raises, i.e. the ceiling our machinery already computed
before the repair. Gordon's $103.6M is exactly 1.40 x our UNrepaired $22,841,455
with 8% raises, to $8,840 out of $103.6M. Under that reading the prior salaries
were never stale; instead our BBRef-sourced future-year figures are uniformly
high — by a flat $791,980/yr for Zubac, by a constant 5.253% for Gordon — and the
same money is pay sitting above the negotiated base.

**Why it did not block the repair**: both readings produce the IDENTICAL ceiling
for these rows (the raise cap plus whatever sits above the base equals observed
pay either way), so no number in the 2026-07-27 RESULT changes. What differs is
`prev_cap_pct`, a real feature, and whether the 2026-2028 salary rows themselves
are right — those are future TARGETS.

**Reproduce**: `data/processed/salaries.csv` for either player; compare the
year-over-year steps against `contract_signing_dates.csv`'s `total_value`. The
cached BBRef team pages (`data/raw/html_cache/*contracts_{DEN,IND}.html`) agree
with our table, so the disagreement is BBRef vs Spotrac, not scrape vs table.

**What to do**: one Spotrac player page per row settles it — their per-season
breakdown separates base salary from bonuses. Not cached today; three fetches at
the 3s rate limit. If Spotrac's per-season base matches its own total, flip both
rows in `salary_corrections.csv` from `prior_base` to `pay_above_base` and fix
the 2025-2028 salary rows.

**Fixed when**: each row's per-season base is sourced rather than inverted, and
`salary_corrections.csv` carries `confidence=verified` for both.

---

## 29. Several Floor Zone crash rows carry another player's or fallback features

**Severity**: medium — the affected rows sit in the large-error tail used to
motivate floor-side model work, so a bad join can create both a false residual
and a false feature hypothesis.

The 2026-07-25 qualitative review of the 34 Floor Zone rows over-priced by more
than $5M found five clear identity or fallback failures:

| row | stored symptom | externally obvious contradiction |
|---|---|---|
| Montrezl Harrell 2023 | age 26, 21.96 mpg | age 29, about 11.9 mpg in 2022-23 |
| DeMarcus Cousins 2020 | age 26, 21.96 mpg | age 30, missed the entire prior season |
| Bol Bol 2023 | age 26 | age 23 |
| Malik Beasley 2023 | 21.96 mpg | about 26 mpg in 2022-23 |
| Patrick Beverley 2023 | age 26, zero prior pay | age 35, prior salary about $13M |

The tuple `21.96 mpg / 17.2 usage / 0.67 availability` repeats across several
of these rows and also appears on unrelated players, indicating median fallback
after a failed player-season join rather than five independent source errors.
Harrell 2023 is especially damaging because the row becomes a second apparent
market crash for a player already represented correctly in 2022.

**Reproduce**: run `python scripts/eval_floor_branch.py`, rebuild the >$5M
Floor Zone residual table from its evaluation frame, and print `age`, `mpg`,
`usage_pct`, `availability_3yr` and `prev_cap_pct` for the five player-seasons
above. The full reviewed table and expected biographical checks are in
`docs/briefs/2026-07-25-floor-crash-qualitative-review.md`.

**What to do**: trace each row backward through the evaluation-frame merge and
identify which source lookup failed. A failed player identity join must not
silently median-fill immutable identity fields such as age. Either repair the
aliases/player-season keys or remove the row from evaluation until its priced
season features are available. Audit every row carrying the repeated fallback
tuple, not only the five manually noticed cases.

**Fixed when**: the five rows have correct ages and priced-season statistics;
no immutable identity field is median-filled after a failed join; and a scan of
the evaluation frame finds no unrelated rows sharing the full fallback tuple.
