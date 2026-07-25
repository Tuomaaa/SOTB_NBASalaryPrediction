"""The CBA raise cap on a veteran extension's first paying year (ISSUES #21).

`_compute_max_eligible` prices every row at a FRESH-SIGNING ceiling: the 25/30/35%
tiers plus Rose Rule, supermax and the no-decrease floor. An extension is not a
fresh signing. Article II of the CBA caps the first Season of an extended term at
a multiple of the final-year salary of the contract being extended — **120% under
the 2017 CBA, 140% under the 2023 CBA** — or the same multiple of the league's
published Estimated Average Player Salary, **whichever is greater**. So an
extension row carries a ceiling it could not legally have reached, by up to
$29.7M (P.J. Washington 2026).

Three distinctions decide whether the raise cap applies, and each one was paid
for by an attempt that failed its sanity check:

1. **First paying year, not any covered season.** The cap governs the first
   Season of the extended term. A row covered by an extension span that STARTS
   EARLIER is either an escalator year (already filtered) or a renegotiated
   season kept fresh by the continuation filter's carve-out — Myles Turner 2022
   is the worked example, whose $35.07M is a renegotiated 2022-23 salary, not an
   extension first year. Keying on `span_start == season` drops 7 such rows.

2. **Rookie-scale extensions are capped by the TIER, not by a multiple.** A
   player coming off a rookie-scale contract can be extended straight to the max
   tier (Jaylen Brown 2020 was paid 3.59x his prior salary). Classification uses
   the same instrument the rookie-scale filter uses — was the PREVIOUS season a
   rookie-scale season (`train._load_rookie_scale_set`) — not a "rookie" keyword
   in the transaction text, which misses deals whose text omits the word (Devin
   Booker 2019 read 6.86x his cap under the keyword split).

3. **Designated Veteran Player extensions are exempt.** The supermax extension
   is written at 30-35% of the cap and is explicitly outside the raise limit
   (Embiid 2023 is paid exactly 35% of the cap at 1.42x his prior salary).
   Eligibility is 7-9 years of service plus a qualifying award, and it is judged
   when the deal is SIGNED, which can be two summers before it pays (Wall signed
   2017 for a 2019-20 start; Harden the same). This module tests both anchors:
   the signing season and the paying season, unioned.

The cap is attached as its own column, `ext_cap_pct`. It deliberately does NOT
enter `max_eligible_pct` or `is_max_contract`: the raise cap binds only
CONDITIONAL on choosing to extend — the player could have tested free agency —
which is the same "choice, not constraint" that killed right-censoring good
players on minimums. Stage 1's censor mask stays keyed on the unconditional tier
ceiling (adjudicated 2026-07-26; see docs/briefs/2026-07-26-extension-route.md).

The Estimated Average Player Salary is a published per-season CBA quantity and is
curated in `data/raw/raw_external/extension_raise_caps.csv` with a source URL per
row, the same way `mle_exception_amounts.csv` was built. It is never inferred
from our own salary table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import CAP_BY_SEASON, RAW_DIR

# cap_pct tolerance for "paid above its own ceiling", matching
# train._filter_mislabeled_year1 — the same float slack the mislabel filter uses.
CAP_TOL = 1e-4


def load_raise_caps() -> pd.DataFrame:
    """The curated per-signing-season raise multiple and EAS-based floor.

    Keyed by SIGNING season, not paying season: the CBA applies the multiple and
    the Estimated Average Player Salary of the Salary Cap Year in which the
    extension is signed. Three rows in the frame prove the keying — Gafford
    (signed 2021, pays 2023), Dinwiddie (signed 2018, pays 2019) and Naz Reid
    (signed June 2023, i.e. signing season 2022, pays 2023) each land to the
    dollar on their SIGNING season's figure, not their paying season's.
    """
    path = RAW_DIR / "raw_external" / "extension_raise_caps.csv"
    if not path.exists():
        raise SystemExit(f"missing curated table: {path}")
    return pd.read_csv(path)


def _elite_trigger(pn: str, season: int, elite: set) -> bool:
    """The supermax award test: qualifying award at s, s-1, or 2 of s-2..s."""
    return ((pn, season) in elite or (pn, season - 1) in elite
            or sum(1 for y in (season - 2, season - 1, season)
                   if (pn, y) in elite) >= 2)


def _designated_veteran(pn: str, season: int, sign_season, exp_by_season,
                        elite: set, early: set, ineligible: set) -> bool:
    """Is this extension a Designated Veteran (supermax) extension?

    Mirrors _compute_max_eligible's supermax branch — 7-9 years of service plus
    a qualifying award, minus the curated designated_ineligible list, plus the
    curated early_supermax list — but the two anchors are used for DIFFERENT
    tests, and conflating them is ISSUES #23:

    - **The award anchor is the SIGNING season, alone.** Eligibility is judged
      when the deal is signed, so an award won during the season the extension
      was signed FOR cannot create it. Marcus Smart's August-2021 extension read
      Designated-Veteran off the DPOY he won in 2021-22 — nine months later —
      and that handed him a $43.28M ceiling in place of his real $17.21M raise
      cap. Since awards land in the spring and the signing season is a league
      year running July-June, `_elite_trigger(sign_season)` is exactly the set
      of awards announced before that league year opened.
    - **The service anchor stays unioned over both seasons.** What is unreliable
      there is our debut-based service count, not the calendar (Embiid spent two
      seasons on the roster before debuting), so both anchors still get a look.

    Deals signed two summers early (Wall 2017 -> 2019, Harden 2017 -> 2019) are
    unaffected: their qualifying award precedes the signing too.
    """
    award_anchor = (int(sign_season) if sign_season == sign_season
                    and sign_season is not None else None)
    for s in (sign_season, season):
        if s != s or s is None:
            continue
        s = int(s)
        if (pn, s) in ineligible:
            continue
        if (pn, s) in early:
            return True
        exp = exp_by_season(pn, s)
        if exp is not None and 7 <= exp <= 9 and _elite_trigger(
                pn, s if award_anchor is None else award_anchor, elite):
            return True
    return False


def attach_extension_cap(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add is_extension / ext_kind / ext_sign_season / ext_cap_pct to `df`.

    ext_cap_pct is the legal ceiling for the row's first extension year as a
    share of that season's cap, or NaN where no extension governs the row. For a
    rookie-scale extension it is the existing tier ceiling (unchanged); for a
    veteran extension it is

        min(tier ceiling, max(mult x prior-season pay, mult x EAS(sign season)))
            + any recorded pay-above-base (trade bonus / earned incentive)

    with the Designated Veteran carve-out replacing the raise cap by the 35%
    supermax ceiling. Rows the extension does not govern keep NaN, so a consumer
    that fills with the existing ceiling is unchanged on them.

    `df` must already carry tier_ceiling_pct and max_eligible_pct (i.e. come
    through _compute_max_eligible), plus player_name_norm / season / cap_pct.
    """
    from scripts.build_external_features import norm
    from scripts.parse_signing_dates import contract_spans, covering_contract
    from src.model.train import (
        _load_prev_season_cap_pct, _load_rookie_scale_set, _load_elite_set,
        _load_debut_seasons, _load_early_supermax, _load_designated_ineligible,
        _load_salary_corrections, TARGET,
    )

    out = df.copy()
    spans = contract_spans()
    prev = _load_prev_season_cap_pct()
    rs_set = _load_rookie_scale_set()
    elite, debut = _load_elite_set(), _load_debut_seasons()
    early, ineligible = _load_early_supermax(), _load_designated_ineligible()
    caps = load_raise_caps().set_index("signing_season")

    # Money inside observed pay that the raise cap does not govern — a trade
    # bonus, an earned incentive. Recorded per row in salary_corrections.csv
    # with its evidence; see _load_salary_corrections.
    corr = _load_salary_corrections()
    addon = {(str(r.player_name_norm), int(r.season)): float(r.value_usd)
             for r in corr[corr["kind"] == "pay_above_base"].itertuples()}

    def exp_by_season(pn, s):
        d = debut.get(pn)
        return None if d is None else int(s) - int(d)

    n = len(out)
    is_ext = np.zeros(n, bool)
    kind = np.array([""] * n, dtype=object)
    sign_season = np.full(n, np.nan)
    ext_cap = np.full(n, np.nan)
    prior_usd = np.full(n, np.nan)
    addon_usd = np.zeros(n)
    dvp = np.zeros(n, bool)
    missing_seasons = set()

    seasons = out["season"].astype(int).values
    y = out[TARGET].values
    tier = out["tier_ceiling_pct"].values

    for i in range(n):
        p, s = out["player_name_norm"].iat[i], int(seasons[i])
        cap_s = CAP_BY_SEASON[s]
        cov = covering_contract(spans, p, s, salary=float(y[i]) * cap_s)
        if cov is None or int(cov["is_extension"]) != 1 or int(cov["span_start"]) != s:
            continue
        is_ext[i] = True
        addon_usd[i] = addon.get((p, s), 0.0)
        ss = int(cov["signing_season"])
        sign_season[i] = ss

        if (p, s - 1) in rs_set:
            kind[i] = "rookie_scale"
            ext_cap[i] = tier[i]          # the tier machinery, unchanged
            continue

        kind[i] = "veteran"
        pn = norm(p)
        pri_pct = prev.get((p, s - 1), np.nan)
        prior_usd[i] = (pri_pct * CAP_BY_SEASON[s - 1]) if pri_pct == pri_pct else np.nan
        dvp[i] = _designated_veteran(pn, s, ss, exp_by_season, elite, early,
                                     ineligible)
        if dvp[i]:
            ext_cap[i] = tier[i]          # designated-veteran ceiling governs
            continue
        if ss not in caps.index:
            missing_seasons.add(ss)
            ext_cap[i] = tier[i]          # no curated figure: do not constrain
            continue
        row = caps.loc[ss]
        raise_cap = float(row["raise_multiple"]) * prior_usd[i]
        eas_cap = float(row["eas_cap_usd"])
        legal = max(raise_cap, eas_cap) if prior_usd[i] == prior_usd[i] else eas_cap
        ext_cap[i] = min(tier[i], legal / cap_s)

    # A recorded trade bonus / incentive is legal money on TOP of whichever
    # ceiling binds, so it is added after the min() with the tier — the raise
    # cap governs base salary, not total pay.
    cap_arr = out["season"].astype(int).map(CAP_BY_SEASON).values.astype(float)
    ext_cap = np.where(np.isnan(ext_cap), ext_cap, ext_cap + addon_usd / cap_arr)

    out["is_extension"] = is_ext
    out["ext_kind"] = kind
    out["ext_sign_season"] = sign_season
    out["ext_is_dvp"] = dvp
    out["ext_prior_usd"] = prior_usd
    out["ext_addon_usd"] = addon_usd
    out["ext_cap_pct"] = ext_cap

    if verbose:
        nk = pd.Series(kind[is_ext]).value_counts().to_dict()
        print(f"Extension cap: {int(is_ext.sum())} first-year extension rows "
              f"({nk.get('rookie_scale', 0)} rookie-scale, "
              f"{nk.get('veteran', 0)} veteran, {int(dvp.sum())} designated-veteran)")
        if (addon_usd > 0).any():
            who = ", ".join(
                f"{out['player_name_norm'].iat[i]} {int(seasons[i])} "
                f"+${addon_usd[i] / 1e6:.2f}M"
                for i in np.flatnonzero(addon_usd > 0))
            print(f"  pay-above-base allowances applied: {who}")
        if missing_seasons:
            print(f"  WARNING: no curated raise cap for signing season(s) "
                  f"{sorted(missing_seasons)} — those rows keep the tier ceiling")
    return out


def attach_extension_value(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add `ext_value_pct`: the extension ceiling EVERY row would face, ex ante.

    `ext_cap_pct` above is an audit quantity — it needs the dated signing season,
    which is knowable only once the player has actually extended. The route
    mixture needs the counterfactual: *if* this row went the extension route,
    what ceiling would it face? That is computable for every row, because an
    extension paying in season s is signed in s-1 (the deadline sits inside the
    prior league year), so the multiple and the Estimated Average Player Salary
    are those of season s-1:

        rookie-scale exit  ->  tier ceiling (the tier machinery, unchanged)
        otherwise          ->  min(tier, max(mult(s-1) x prior pay, EAS(s-1)))

    Rows with no observable prior pay get NaN, and the mixture leaves those rows
    at the pooled prediction rather than inventing a ceiling for them.

    Verified against `ext_cap_pct` on the rows where both exist: the s-1
    convention reproduces the audited signing-season cap wherever the extension
    was in fact signed the summer before it paid, and differs only for deals
    signed two or more summers early (the early-supermax class).
    """
    from scripts.build_external_features import norm
    from src.model.train import (
        _load_prev_season_cap_pct, _load_rookie_scale_set, TARGET,
    )

    out = df.copy()
    prev = _load_prev_season_cap_pct()
    rs_set = _load_rookie_scale_set()
    caps = load_raise_caps().set_index("signing_season")

    n = len(out)
    val = np.full(n, np.nan)
    seasons = out["season"].astype(int).values
    tier = out["tier_ceiling_pct"].values
    for i in range(n):
        p, s = out["player_name_norm"].iat[i], int(seasons[i])
        if (p, s - 1) in rs_set:
            val[i] = tier[i]
            continue
        pri_pct = prev.get((p, s - 1), np.nan)
        if pri_pct != pri_pct or (s - 1) not in caps.index:
            continue                       # no prior pay: leave the row alone
        row = caps.loc[s - 1]
        prior_usd = pri_pct * CAP_BY_SEASON[s - 1]
        legal = max(float(row["raise_multiple"]) * prior_usd,
                    float(row["eas_cap_usd"]))
        val[i] = min(tier[i], legal / CAP_BY_SEASON[s])

    out["ext_value_pct"] = val
    if verbose:
        print(f"Extension value: {int((~np.isnan(val)).sum())}/{n} rows have an "
              f"ex-ante extension ceiling "
              f"({int(np.isnan(val).sum())} lack an observable prior salary)")
    return out


def over_cap_rows(df: pd.DataFrame, tol: float = CAP_TOL) -> pd.DataFrame:
    """Rows paid above their own computed extension cap — the hard gate.

    A ceiling below observed pay is strictly worse than one above it: it
    mislabels the row as censored and pins the Stage-2 clip under the truth
    (the v7.4x invariant). This returns the offenders with everything needed to
    diagnose them.
    """
    from src.model.train import TARGET

    m = df["ext_cap_pct"].notna() & (df[TARGET] > df["ext_cap_pct"] + tol)
    cols = [c for c in ["player_name_norm", "season", TARGET, "ext_cap_pct",
                        "ext_kind", "ext_sign_season", "ext_is_dvp",
                        "ext_prior_usd", "ext_addon_usd", "tier_ceiling_pct",
                        "max_eligible_pct"]
            if c in df.columns]
    out = df.loc[m, cols].copy()
    cap = out["season"].map(CAP_BY_SEASON)
    out["pay_m"] = out[TARGET] * cap / 1e6
    out["cap_m"] = out["ext_cap_pct"] * cap / 1e6
    out["over_m"] = out["pay_m"] - out["cap_m"]
    return out.sort_values("over_m", ascending=False)
