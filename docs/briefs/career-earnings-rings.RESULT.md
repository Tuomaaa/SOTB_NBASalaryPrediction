# RESULT — career earnings and championship rings, ex ante

Two new ex-ante data columns for the at-floor over-pricing question. **Nothing
is adopted here.** `FEATURE_COLS` is untouched, nothing under `src/model/`
changed, no model was trained and no evaluation layer was run. What follows is
data plus its validation.

Built by two new scripts:

| script | output | network |
|---|---|---|
| `scripts/build_career_earnings.py` | `data/processed/career_earnings.csv` | none — reads the Spotrac page cache |
| `scripts/scrape_championships.py` | `data/processed/championships.csv`, `data/processed/rings_thru_prev.csv` | 29 Basketball Reference pages, cached |

---

## 0. The finding that changes the recipe: the Spotrac anchor is repo season 2025

The brief specifies

```
earnings_thru(T-1) = career_earnings_thru_2026 - sum(salaries.csv, seasons T..2026)
```

**The second `2026` is wrong, and using it would subtract one season the
aggregate never held.** Spotrac labels a season by its ENDING year; this repo
labels it by its STARTING year (`config.CAP_BY_SEASON[2025] == 154_647_000`,
which is the 2025-26 cap). Spotrac's "Career Earnings thru 2026" is therefore
the 2025-26 season — **repo season 2025**.

Calibrated, not assumed. Over the 252 players whose entire career sits inside
the salary window (debut ≥ 2019, first salary row ≥ 2019), the aggregate should
equal a plain forward sum of `salaries.csv` up to the anchor and nothing else:

| sum through repo season | exact matches | within 1% | mean signed gap |
|---|---:|---:|---:|
| 2024 | 10 | 6.3% | −$7.72M |
| **2025** | **61** | **51.2%** | **−$0.24M** |
| 2026 | 17 | 9.9% | +$8.74M |

The implemented arithmetic is

```
earnings_thru(T-1) = career_earnings_anchor - sum(salaries.csv, seasons T..2025)
```

`build_career_earnings.py` asserts the label still reads 2026 on every cached
page and refuses to run if it moves, because a refreshed cache moves the anchor
season with it. `_calibrate_anchor()` reprints the table above on every run.

A second correction to the brief's premises: `data/processed/salaries.csv`
covers **2019-2031**, not 2016-2031. The 2016-2018 rows live in a separate file
(`salaries_prehistory.csv`, 767 rows). This does not affect the construction —
every season the subtraction needs is ≥ 2019 — but it does mean the pre-2019
career comes entirely from the Spotrac aggregate, exactly as intended.

---

## 1. DELIVERABLE 1 — `data/processed/career_earnings.csv`

4,148 rows, 1,088 distinct players, seasons 2019-2026 (repo convention).
Key set is every (player, season) the training frame prices, unioned with every
(player, season) `salaries.csv` knows, clipped to the priceable window.

| column | meaning |
|---|---|
| `player_name_norm` | training-frame name convention |
| `player_url` | Basketball Reference path, where `salaries.csv` resolves one |
| `spotrac_slug` | the cache filename the aggregate was read from |
| `season` | T, the season being priced (repo convention) |
| `career_earnings_thru_prev` | dollars earned through T−1, clipped at 0 |
| `career_earnings_thru_prev_cap_pct` | the same, over `CAP_BY_SEASON[T]` |
| `career_earnings_thru_prev_raw` | before the clip, so a negative stays visible |
| `career_earnings_anchor_usd` | the Spotrac aggregate as parsed |
| `anchor_season` | 2025, constant |
| `n_seasons_subtracted` | salary rows removed, seasons T..2025 |
| `quality` | `ok` / `negative_clipped` / `no_spotrac_page` |

Every value is knowable before season T is signed. The aggregate is a career
total, and the subtraction only ever REMOVES seasons T and later from it.

### 1.1 Validations

**(1) Parse assertions.** 599 of 600 cached pages carry the aggregate line;
all 599 state `thru 2026`. The three hand-checked values reproduce exactly:

```
lebron-james  $581,375,548   OK
kevin-durant  $501,135,653   OK
chris-paul    $404,526,572   OK
```

The one page without the line is `vince-williams-jr` — a Spotrac landing-page
redirect, the ISSUES #41 class. It is the only such page left in the cache.

**(2) Non-negativity.** 40 of 2,986 valued rows (1.3%) go negative before the
clip, across 31 players. They are clipped to 0 in
`career_earnings_thru_prev`, flagged `negative_clipped`, and the raw value is
kept. Every one is a young two-way / G-League / prorated player whose true
career earnings through T−1 are near zero anyway — the negative is the
convention gap of §1.2 applied to a player with almost no career to absorb it.
Largest: Royce O'Neale −$3.75M; median −$0.34M. Seven land on at-floor rows:

```
drew timme  2025   jd davison 2024   jd davison 2025   jeff dowtin 2023
skylar mays 2021   skylar mays 2022  skylar mays 2023
```

Their anchor totals are $0.22M, $0.76M and $1.00M respectively — none of them
is a ring-chasing veteran, and 0 is the right answer for all seven.

**(3) Monotone non-decreasing in season within player.** **0 violations**, on
both the raw and the clipped column. This is structural rather than lucky:
`earnings_thru(T) = earnings_thru(T−1) + salary(T)` by construction, so the
check can only fail on a negative salary or a duplicate row. It is kept because
it is the cheap detector for exactly those two faults.

**(4) Duplicate (player, season) rows in `salaries.csv`.** **Zero.** Checked on
the raw `player` column and on the normalized `player_name_norm` — 4,737 rows,
no duplicates on either key. So the sum-vs-dedupe question does not arise; the
implementation sums (`groupby(season).sum()`), which is a no-op on this file but
degrades correctly if a traded-player row is ever added.

**(5) Convention mismatch.** Spotrac's aggregate is cash paid; `salaries.csv`
mixes cash and cap charge (ISSUES #36, #38). Measured on the same 252
in-window-career players, `Spotrac − our forward sum`:

```
mean       +$0.237M      median   +$0.084M      sd  $0.712M
IQR        [$0.000M, +$0.352M]    min −$2.016M  max +$5.272M
exact 0    61 / 252 (24%)   within $50k  76 / 252 (30%)
|rel err|  median 0.97%   within 1% 51.2%   within 5% 75.8%
```

The sign is mostly positive: `salaries.csv` slightly UNDER-counts cash relative
to Spotrac, so the subtraction slightly OVER-states `career_earnings_thru_prev`.

The number that actually bounds the error is the **per-subtracted-season** gap,
because only the seasons T..2025 pass through our salary table — the pre-2019
career comes straight from Spotrac and carries no error of ours:

```
per-season gap   mean +$0.104M   median +$0.023M   p90 |gap| $0.378M
restricted to careers > $5M:   mean +$0.090M   median +$0.018M
```

At most 7 seasons are ever subtracted, so the worst-case accumulated error on
`career_earnings_thru_prev` is roughly **±$0.7M**, and the typical error is
under $0.2M. Against the quantity of interest — Chris Paul at $402M, Blake
Griffin at $254M, Marc Gasol at $176M — this is noise. It is *not* noise for the
sub-$5M tail, which is where all 40 negatives sit; treat
`career_earnings_thru_prev` as unreliable below about $2M and use the
`negative_clipped` flag to find the worst of it.

The large *relative* errors are entirely the two-way/G-League class: Jeff Dowtin
(Spotrac $0.23M, our sum $2.25M), Drew Timme ($0.22M vs $2.07M), JD Davison
($0.76M vs $2.28M) — our table carries the full cap-charge minimum where
Spotrac carries the prorated two-way cash. This is the same defect ISSUES #38
records, seen from outside.

Not applied, deliberately: the `salary_override` and `min_cap_charge`
corrections in `salary_corrections.csv`. Both move a row toward the team's cap
charge, which is the convention Spotrac is *not* using, so applying them would
widen this gap rather than close it. Raw `salaries.csv` `salary` is summed.

### 1.2 The name join

Joined on `slugify(player_name_norm)` against the cache filename, where
`slugify` is `scripts.scrape_spotrac_players.slugify` — the same function that
named the files. **Suffixes are preserved**, so `kelly oubre jr.` →
`kelly-oubre-jr`, `gary payton ii` → `gary-payton-ii`. The collision class the
brief warns about cannot occur here: the cache contains
`gary-payton-ii.html`, `kevin-porter-jr.html`, `jaren-jackson-jr.html`,
`larry-nance-jr.html`, `tim-hardaway-jr.html` and no bare-surname page for any
of them, so there is no father's page to fall onto. All 23 suffixed names in the
frame resolve correctly or not at all; none resolves to the wrong player.

Two further checks that the join is clean rather than merely large:

- `_normalize_name(salaries.csv player)` reproduces **all 793** training
  `player_name_norm` values — zero name drift between the two tables, so no
  alias table is needed for this join and `player_name_aliases.csv` is a no-op
  here (both of its entries, `enes kanter → enes freedom` and
  `wesley iwundu → wes iwundu`, already resolve).
- A fuzzy pass (difflib, cutoff 0.82) over the 221 unmatched training names
  against all 599 slugs returns **zero** unclaimed candidates. Nothing is
  recoverable offline; the unmatched simply have no cached page.

**Page-identity check.** The `<title>` name is compared to the slug. Sixteen
pages disagree on the given name and agree on the surname — the ISSUES #41
re-fetch set, keyed by nickname and titled with Spotrac's legal name
(`bones-hyland` → "Nah'Shon Hyland", `jj-barea` → "Jose Barea", `kj-martin` →
"Kenyon Martin Jr.", `mo-bamba` → "Mohamed Bamba", …). All sixteen are the right
page, so the check is on the surname, suffixes stripped. **0 pages fail it.**

### 1.3 Coverage

| population | rows / players | with a value | |
|---|---:|---:|---:|
| training-frame rows | 3,116 | 2,703 | 86.7% |
| distinct training-frame players | 793 | 571 | **72.0%** |
| at-floor rows (`_compute_floor`) | 501 | 501 | **100.0%** |
| at-floor rows, age ≥ 30 | 156 | 156 | **100.0%** |

Baselines quoted in the brief were 70% / 97% / 99%. The at-floor populations
come in at 100% — the missing 3% and 1% were the suffix-stripping artifact,
which this join does not have.

Quality breakdown on the 3,116 frame rows: `ok` 2,691, `no_spotrac_page` 413,
`negative_clipped` 12. On the 501 at-floor rows: `ok` 494,
`negative_clipped` 7, `no_spotrac_page` 0.

### 1.4 The unmatched list — 222 frame players with no Spotrac page

Reported in full so it can be eyeballed. It is overwhelmingly **rookies and
recent draftees** (the cache was built from free-agent directories, which a
player on his rookie deal has never appeared in) plus a tail of **pre-cache
retirees**. Not one of them is at-floor.

The names a human should look at — established players whose absence is a real
gap rather than a rookie artifact:

```
stephen curry      jamal crawford     pau gasol         shaun livingston
anthony tolliver   iman shumpert      luc mbah a moute  tyler zeller
nik stauskas       thon maker         troy daniels      lance thomas
jonathon simmons   jerian grant       justin anderson   isaac okoro
darius bazley      facundo campazzo   romeo langford    sekou doumbouya
```

The remaining 202, in full:

```
a.j. lawson, ace bailey, admiral schofield, adou thiero, aj griffin,
aj johnson, ajay mitchell, aleksej pokusevski, alex sarr, alize johnson,
alondes williams, amen thompson, anthony black, anthony lamb, asa newell,
ausar thompson, baylor scheierman, ben saraf, ben sheppard,
bennedict mathurin, bilal coulibaly, blake wesley, brandin podziemski,
brandon miller, braxton key, brice sensabaugh, bub carrington, caleb houstan,
cam whitmore, carter bryant, cason wallace, cedric coward, chaz lanier,
cody williams, collin murray-boyles, cooper flagg, dalton knecht, danny wolf,
dariq whitehead, david roddy, demetrius jackson, dereck lively ii, derik queen,
devin cannady, devin carter, devontae cacok, dillon jones, donovan clingan,
donta hall, dragan bender, drake powell, duop reath, dusty hannahs,
dylan harper, elijah hughes, eugene omoruyi, freddie gillespie, gradey dick,
henry ellenson, hugo gonzalez, ignas brazdeikis, isaiah collier,
isaiah mobley, ivan rabb, j.p. macura, ja'kobe walter, jack white,
jacob evans, jaden ivey, jaden springer, jaime jaquez jr., jake laravia,
jalen duren, jalen hood-schifino, james bouknight, jarace walker,
jared butler, jared harper, jared mccain, jarrell brantley, jarrett culver,
jase richardson, jaylen adams, jaylen hoard, jaylon tyson, jeenathan williams,
jemerrio jones, jerome robinson, jett howard, joan beringer, joe chealey,
johnathan williams, johni broome, johnny davis, jordan bell, jordan hawkins,
josh christopher, josh gray, joshua primo, julian strawther, justin patton,
justin robinson, juwan morgan, kai jones, kasparas jakucionis, kel'el ware,
kelan martin, kendall brown, keon johnson, kevon harris, keyonte george,
khaman maluach, khyri thomas, killian hayes, kobe bufkin, kobi simmons,
kon knueppel, kris murray, kyle filipowski, kyle guy, kyshawn george,
leandro bolmaro, lester quinones, liam mcneeley, luca vildoza, mac mcclung,
malachi flynn, malaki branham, malcolm hill, malik fitts, mamadi diakite,
marcus sasser, marjon beauchamp, marko guduric, marko simonovic,
matas buzelis, matt ryan, maxime raynaud, mfiondu kabengele, myron gardner,
nate hinton, nick smith jr., nique clifford, noa essengue, noah clowney,
noah penda, nolan traore, norvel pelle, ochai agbaji,
olivier-maxence prosper, omari spellman, patrick baldwin jr., peyton watson,
quinndary weatherspoon, r.j. hampton, rasheer fleming, rayjon tucker,
reed sheppard, reggie perry, rob dillingham, robert franks, ryan broekhoff,
ryan dunn, ryan kalkbrenner, scoot henderson, shaquille harrison,
sindarius thornwell, sion james, stanley umude, stephon castle,
taylor hendricks, terrance ferguson, terrence shannon jr., thomas sorber,
tre johnson, tremont waters, trent forrest, trevelin queen, tristan da silva,
troy williams, tyler cook, tyler kolek, tyler lydon, tyler ulis,
tyrese proctor, tyrone wallace, udoka azubuike, usman garuba,
victor wembanyama, vince williams jr., vincent poirier, vj edgecombe,
will richard, will riley, xavier sneed, yang hansen, yante maten, yves missi,
zaccharie risacher, zach edey, zhaire smith
```

Closing this needs ~222 Spotrac fetches at the 3s rate limit (~15 minutes,
plus backoff), by the slug-construction route ISSUES #4 and #41 already
document. It is out of scope here and is filed as ISSUES #50.

### 1.5 Does the column say what the hypothesis predicts?

Spot check, career earnings through T−1 for the seven named ring-chasers, plus
two controls:

| player | T | earnings thru T−1 | × that season's cap |
|---|---:|---:|---:|
| chris paul | 2025 | $402.2M | 2.60 |
| blake griffin | 2021 | $254.2M | 2.26 |
| marc gasol | 2020 | $176.4M | 1.62 |
| andre drummond | 2021 | $136.1M | 1.21 |
| hassan whiteside | 2020 | $98.3M | 0.90 |
| reggie jackson | 2020 | $90.5M | 0.83 |
| montrezl harrell | 2022 | $35.0M | 0.28 |
| *lebron james* | *2025* | *$528.7M* | *3.42* |
| *stephen curry* | *2025* | *no Spotrac page* | — |

And the at-floor population separates cleanly on age, which is what a
"has banked enough" story requires:

| at-floor rows, age band | n | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| ≤ 25 | — | $0.9M | $3.5M | $10.4M | $87.2M |
| 26-29 | — | $5.7M | $13.6M | $37.5M | $136.1M |
| 30-32 | — | $39.4M | $66.5M | $98.5M | $256.8M |
| 33+ | — | $66.5M | $90.6M | $178.0M | $402.2M |

Six of the seven named ring-chasers sit above the 75th percentile of their own
age band. Montrezl Harrell (2022, age 28, $35.0M) does not — he is at the band
median, which is worth knowing before anyone builds a threshold rule on this
column alone.

---
