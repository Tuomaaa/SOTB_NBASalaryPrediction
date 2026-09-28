# Handoff — the ring-chasing discount: what is decided, what landed, what is open

Written 2026-08-29 to be forwarded cold. Read CLAUDE.md and ISSUES.md first.
**Nothing here is adopted.** `FEATURE_COLS` is untouched, nothing under
`src/model/` changed, no model was trained, no evaluation layer was run.

---

## 1. The hypothesis

The model over-prices the at-floor zone by **+$2.04M**. Part of that is
veterans who are worth more than the minimum and signed one anyway. The claim
is that these are identifiable **ex ante** by a profile that has nothing to do
with ability: the player has banked enough money that winning outweighs the
marginal dollar.

The floor zone decomposes into three populations:

| | population | model's job | status today |
|---|---|---|---|
| 1 | floor natives — worth the minimum | predict floor | correct |
| 2 | worth LESS than the floor, propped up by it | clip up to floor | **already handled** |
| 3 | worth MORE than the floor, signed it anyway | pull down | **the entire headroom** |

Group 2 is already solved and costs nothing: `_compute_floor` plus the Stage-2
clip lift a sub-floor latent to the floor, exact answer, zero error.

**The existing left-censor gate is already a crude group-2 / group-3
discriminator** — `src/model/train.py:1101`:

```python
gate_l = at_floor & (oof_bl <= floor_gate_k * y)   # floor_gate_k = 2.0
```

Model thinks you are worth 2x the minimum or less, you are group 1 or 2, and
you get left-censored. Model thinks you are worth 8x the minimum, you are group
3 and are deliberately left untouched. That untouched set is the +$2.04M.

### 1.1 It is NOT restricted to the floor

This is the most important correction in the session and it was made late.
**James Harden 2022** opted out of a $47M player option and re-signed with
Philadelphia for roughly $33M, publicly to let the team add players. That is
the same phenomenon — a large voluntary discount by a player with high career
earnings — at mid-range, nowhere near the minimum.

Consequence: **floor-branch arm B is the wrong intervention shape** if the
phenomenon is general, because arm B pulls toward `0.95 * floor_pct`, which is
meaningless for a $47M to $33M discount. Resolving floor-restricted vs general
is open question 1 in section 5, and it determines the intervention.

---

## 2. Design rulings established this session

**(a) The signal enters as FEATURES, not a hand-coded rule.**
A hard rule can only cut the cell on its three coordinates, so `old + rich +
ringless` necessarily contains both ring-chasers and well-paid old stars still
on real contracts — a 34-year-old who earned $200M and just signed for $25M
satisfies all three. The cell mean is then the average of "discount hard" and
"do not touch", and the constant is diluted to nothing. A tree does not have
this problem: it splits again on `kf_market_value` and
`kalman_filtered_stats` and separates "still producing" from "collapsed". The
disambiguation is done by the other 21 features, not by hand.

This also matches CLAUDE.md's escalation rule — a feature is the cheap step, a
new Stage-2/3 branch is the expensive one, and the expensive one needs a paired
CV improvement over the cheap one.

**(b) A censoring label is NOT leakage. This was initially got wrong.**
`P(ring_chasing)` used as a Grabit censor label may legitimately read the
target. The existing masks already do:

```python
gate   = right_pop & (oof_bl >= gate_frac * max_elig)
gate_l = at_floor  & (oof_bl <= floor_gate_k * y)     # reads y directly
```

`at_floor` itself comes from `signing_cat == "Minimum"`. This is not a defect,
it is what censoring *is* — you must see the observation to declare it a bound
rather than a point. `is_vet_min` was leakage in v4.0 because it was a
**feature**, and a feature must exist at inference time. A censor label exists
only at fit time; `predict.py` never needs it.

The leakage objection applies to the feature list only. Do not repeat it
against a censor mask.

**(c) Two candidate intervention shapes, different altitudes, not exclusive.**

| | acts at | where the payoff is | how to judge it |
|---|---|---|---|
| Stage-2 pull (floor-branch arm B) | output | the rows P flags | local, directly measurable |
| Grabit third gate: `at_floor & (P >= tau)` right-censored | training | the NEIGHBOURS | needs its own metric |

The Grabit route's payoff is indirect and worth stating explicitly: the
$17M-scale penalties these rows currently incur distort the trees for everyone
near them (METHODOLOGY.md item 8 — one set of trees serves all 1,172 rows, so
leaf values shift for their neighbours too). Right-censoring them removes a
drag on genuinely similar players who DID get paid. Judging that by floor-zone
MAE would miss it entirely.

The Grabit route revives something the project explicitly killed:
`src/model/extension_cap.py:41` records that "choice, not constraint" killed
right-censoring good players on minimums. But it killed the **unconditional**
version. This one is gated on P. That is the same shape as the max side's
history, where phase 2's NO-GO was an argument that measurement later
overturned.

**(d) Pre-registration constraints, if the Stage-2 route is taken.**
`MARGIN = 0.95` is FIXED by the `1.05` precedent and must never be tuned on a
zone metric. `tau` must be pre-registered by the win-minus-collateral rule
BEFORE any score on the arm is seen. **The floor branch's tau\* = 0.10 is not
reusable** — it was selected for a P that was anti-ranked against the error.

**(e) Why this is not just floor-branch arm B again.**
Arm B failed at +0.00265 (t = +0.54) — positive but not significant. The
diagnosed cause was the P, not the architecture: `P(floor)` from the 6-class
route classifier is anti-ranked against the error it exists to fix, **Spearman
= -0.611** inside the zone, because that classifier is fed performance features
and a fallen star on a minimum looks, on every performance feature, more like a
well-paid player. The proposed P is built from **non-performance circumstance**
variables, which is a structurally different attack. The architecture and the
harness (`scripts/eval_floor_branch.py`) already exist, so swapping the P is a
drop-in.

Also carry this: arm A (`min(champ, floor_pct)`, a hard slam) scored
**-0.084**. Arm B's P-weighted-with-margin form is already demonstrated
harmless. The downside of the P-weighted form is bounded; the hard-slam form's
is not.

---

## 3. Data that landed

### 3.1 `data/processed/career_earnings.csv` — DONE, validated (built 2026-08-26)

Ex-ante career earnings through T-1. Built by
`scripts/build_career_earnings.py`, fully offline from the Spotrac page cache.
Full validation is in `docs/briefs/career-earnings-rings.RESULT.md`.

The construction is **subtract-forward**, not sum-forward:

```
career_earnings_thru_prev = spotrac_career_anchor - sum(salaries.csv[T .. 2025])
```

Sum-forward from 2016 is NOT acceptable: 45% of at-floor rows debuted before
2016, and for that whole cohort the visible window length degenerates to
`season - 2016`, a constant per season — identical for Chris Paul (20 true
prior years) and a 4-year vet, while true career length in the cohort ranges
4 to 31 years. The Spotrac aggregate covers the pre-2016 career; the
subtraction only needs post-2016 seasons, which `salaries.csv` has.

**Traps encoded in that script. Do not re-derive them:**

- **The anchor is repo season 2025, not 2026.** Spotrac labels a season by its
  ENDING year, this repo by its STARTING year, so "Career Earnings thru 2026"
  is the 2025-26 season = repo 2025. Calibrated, not assumed: over 252 players
  whose whole career sits inside the salary window, anchoring at 2025 gives 61
  exact matches and a -$0.24M mean gap; anchoring at 2026 gives 17 and
  +$8.74M. The script asserts the page label still reads 2026 and refuses to
  run if it moves.
- `salaries.csv` covers **2019-2031**, not 2016-2031. The 2016-18 rows live in
  `salaries_prehistory.csv`.
- **Convention gap.** Spotrac is cash paid; `salaries.csv` mixes cash and cap
  charge (ISSUES #36, #38). Per subtracted season the gap is mean +$0.104M,
  median +$0.023M, and at most 7 seasons are ever subtracted, so the worst case
  is about +/-$0.7M. That is noise against Chris Paul's $402M. It is **NOT
  noise below about $2M** — all 40 negative-before-clip rows sit there and they
  are the two-way / G-League class. Use the `quality` flag.
- The `salary_corrections.csv` overrides are deliberately NOT applied. They
  move rows toward cap charge, which is the convention Spotrac is not using, so
  applying them widens the gap rather than closing it.

Coverage: at-floor rows **501/501 (100%)**, at-floor age >= 30 **156/156
(100%)**, all training-frame players 72%. The missing 28% are rookies and
pre-cache retirees. **222 players have no Spotrac page**, filed as ISSUES #50,
and not one of them is at-floor.

Does the column say what the hypothesis predicts? Six of seven named
ring-chasers sit above the 75th percentile of career earnings **for their own
age band**. Montrezl Harrell (2022, age 28, $35.0M) does not — he is at the
band median. A threshold rule on this column alone would miss him.

### 3.2 `data/processed/championships.csv` — DONE this session

389 rows, 307 distinct players, 22 title seasons (repo convention 2004-2025).
Scraped by `scripts/scrape_championships.py` from Basketball Reference
team-season pages, one page per title season.

`FIRST_BBREF_YEAR` was narrowed **1999 to 2005** this session. The reasoning is
in the source comment: no 1999-2004 title-team member was still active in 2019
(the Spurs core retired by 2018, the Lakers three-peat core by 2016, the 2004
Pistons by 2016), while the earliest ring any frame member actually holds is
repo 2005 — Udonis Haslem, the 2006 Heat. Those six seasons were pure 403
exposure for zero rings. **Do not narrow further. Cutting to 2007 would
silently drop Haslem's first ring.**

Validated:

| player | ring seasons (repo convention) | n |
|---|---|---:|
| LeBron James | 2011 \| 2012 \| 2015 \| 2019 | 4 |
| Andre Iguodala | 2014 \| 2016 \| 2017 \| 2021 | 4 |
| Udonis Haslem | 2005 \| 2011 \| 2012 | 3 |
| Marc Gasol | 2018 | 1 |
| Kevin Love | 2015 | 1 |
| Paul / Harden / Griffin / Westbrook / Lillard / Harrell / Drummond | — | 0 |

`rings_thru_prev` respects strictly-before-T: Gasol's 2020 row reads
`1 [2018]`, LeBron's 2025 row reads `4`, Harden's 2023 row reads `0`.

**Known defect, harmless.** The `played_playoffs` column is 0 for every row —
Basketball Reference serves the playoff tables inside HTML comments and the
parser does not un-comment them. `rollup_rings` uses `on_roster |
played_playoffs`, and `on_roster` is correct and complete, so ring counts are
unaffected. Fix it only if that column is ever needed for something else.

**Basketball Reference is being abandoned** in favour of Spotrac (user
decision, 2026-08-29). This scrape cost roughly a dozen backoffs and three hard
403s across 22 requests. CLAUDE.md's "~25 minutes, one team will 403" is
optimistic against the current limiter. Do not plan new BBRef scraping without
accounting for that.

### 3.3 `data/processed/rings_thru_prev.csv` — PROVISIONAL, regenerate

The script writes it unconditionally alongside `championships.csv`. Its key set
comes from `_key_set()`, which reads `training_data_v2.csv` and `salaries.csv`
— **both currently mid-rewrite by the Spotrac migration.** The file on disk is
a half-migrated snapshot. Regenerate after the migration settles. This needs no
re-scraping; `championships.csv` is migration-independent.

`career_earnings.csv` needs the same treatment for the same reason: its anchor
leg is Spotrac and is fine, but its **subtraction leg reads `salaries.csv`**.
After the migration both ends become Spotrac cash, so the convention gap in
section 3.1 may vanish outright — good news, but re-measure it rather than
assuming it.

---

## 4. Evidence on the two legs

**The earnings leg looks real.** Six of seven named ring-chasers are above
their age band's p75, and the at-floor population separates cleanly by age band
(33+ median $90.6M vs 25-and-under median $3.5M), which is what a "has banked
enough" story requires.

**The ring leg looks weak, possibly wrong-signed.** Three high-profile players
discount *while holding rings*: Marc Gasol won in 2019 and took a Lakers
minimum the next summer, and LeBron holds four and has chronically signed below
max. Harden discounted with zero rings but not at the floor. The motive may not
be "I need a ring" but "I have enough money that winning beats the marginal
dollar", which is the earnings leg, not the ring leg.

**Do not cite Udonis Haslem as evidence on either side.** He looks like the
archetype — three rings, a decade of minimums, $69M banked at age 41 — and he
is not in the population at all. From `outputs/web/valuations_export.csv`:

| season | age | latent | pred | actual | is_floored | surplus |
|---|---:|---:|---:|---:|---|---:|
| 2019 | 38 | -0.0031 | 0.014848 | 0.014848 | True | 0 |
| 2020 | 39 | -0.0060 | 0.014848 | 0.014848 | True | 0 |
| 2021 | 40 | +0.0051 | 0.014848 | 0.014848 | True | 0 |
| 2022 | 41 | -0.0049 | 0.013905 | 0.013905 | True | 0 |

The latent is **negative** in three of four seasons. He is group 2 — worth less
than the floor, propped up by it — and the clip already prices him exactly.
Zero error contribution.

He is still informative in one way. He is a **false positive for any earnings
threshold** (age 41, $69M) that happens to be **harmless**, because the floor
clip already sits underneath him and a further downward pull is a no-op.
Collateral inside the floor zone is asymmetrically cheap for that reason.
Collateral OUTSIDE the floor zone — the Harden range — is not, and has no such
backstop.

---

## 5. Open questions, in the order they should be answered

**(1) Is the phenomenon floor-restricted or general?** This determines the
intervention shape and everything downstream. It is answerable **without
running the model**: `outputs/web/valuations_export.csv` carries
`latent_cap_pct`, `pred_cap_pct`, `surplus`, `is_floored` and `floor_pct`, so
group 2 and group 3 separate directly — group 3 is the at-floor set with a high
positive latent. The same table shows whether large negative surpluses cluster
at the floor or spread across the salary range. Caveat: that export is
currently `M` in git and may itself be a mid-migration snapshot.

**(2) How many rows are in play?** 156 at-floor rows at age >= 30 is a
floor-arm lower bound. Do NOT anchor on the 9 named fat-tail players — that
list is the top of an error ranking, not the population, and typical members
are over-priced by $2-4M rather than $17M. An estimate of "about 6 rows" made
earlier in this conversation was wrong for exactly that reason.

**(3) Does the discount rank on career earnings and stay flat on ring status?**
The decisive test, and the one that decides whether the ring column ships. The
benchmark statistic is the one that killed the last attempt: Spearman of the
candidate score against the zone's over-prediction, versus `P(floor)`'s
**-0.611**. If ring status comes out flat, drop the column — the 22 pages will
have earned their cost by falsifying a leg.

---

## 6. Documentation debt

`docs/briefs/career-earnings-rings.RESULT.md` has a header table claiming
`scripts/scrape_championships.py` produced `championships.csv` and
`rings_thru_prev.csv`. When that file was written on 2026-08-26 neither
existed — the session was cut off mid-deliverable and the write-up stops after
section 1.5, with no deliverable-2 section at all. Both files exist as of
2026-08-29, so the table is now accidentally true, but the missing section
should be written, and a reader should not be left believing the ring work was
documented when it was not.
