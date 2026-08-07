"""Legal ceilings for Early Bird and Non-Bird signing mechanisms.

Zero-parameter, deterministic, same architecture as the extension raise cap
(`extension_cap.py`): the mechanism cap only ever LOWERS a prediction, and it
reads no fitted parameter — the numbers come from the CBA's published minimum
salary scale and the league's Estimated Average Player Salary.

    Early Bird cap = max(1.75 x prior_salary, 1.05 x EAS)
    Non-Bird cap   = max(1.20 x prior_salary, 1.20 x vet_min_for_experience)

For all other signing types, `mech_cap_pct` is NaN — no mechanism cap applies.

The Okogie exclusion (see `_NONBIRD_LABEL_EXCLUSIONS`) preserves a documented
data decision: josh okogie 2023 carries a Non-Bird label inconsistent with the
CBA minimum scale at his service year, flagged as a wrong label (likely signed
via Room Exception or cap room).

Chain order in compose():
  latent -> push -> clip[lo,hi] -> signing offset -> mechanism cap clip
  -> extension clip -> re-clip[lo,hi]

The mechanism cap sits AFTER the signing offset and BEFORE the extension clip,
so a Bird Rights offset that pushes a row above its Early Bird ceiling is
caught, while an extension raise cap that is tighter still governs last.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import CAP_BY_SEASON, RAW_DIR, CBA_NEW_ERA_SEASON

# Non-Bird sanity tolerance: 0.1% of cap.  Absorbs Clark (2020, exp=2) whose
# debut-year error puts him $55K over the published 2-year CBA minimum.
MECH_CAP_TOL = 1e-3

# Okogie (2023 exp=5) label issue: his Non-Bird label is inconsistent with the
# 2017 CBA minimum scale for 5 years of service ($1,932,936).  120% of that is
# $2,319,523, but his actual is $2,816,000 -- a $497K gap that no experience
# count can close.  Flagged as a wrong label (likely signed via Room Exception
# or cap room); excluded from Non-Bird cap.
_NONBIRD_LABEL_EXCLUSIONS = {("josh okogie", 2023)}


# ---------------------------------------------------------------------------
# Published 2017 CBA veteran minimum salary scale (Exhibit I)
# ---------------------------------------------------------------------------
# USD figures for 2019-20 (= 2020-21, frozen per COVID agreement).
# Tiers 0-9 are per service year; 10 means 10+ years.
_CBA_2017_BASE_2020 = {
    0: 898310, 1: 1445697, 2: 1620564, 3: 1678854, 4: 1737145,
    5: 1795435, 6: 1853726, 7: 1912016, 8: 1970306, 9: 2028594,
    10: 2564753,
}
# Per-season scaling from the 2020 base.  The CBA specifies absolute amounts;
# these ratios recover them from the base.
_CBA_2017_SCALE = {
    2019: {e: v for e, v in _CBA_2017_BASE_2020.items()},
    2020: {e: v for e, v in _CBA_2017_BASE_2020.items()},  # frozen
    2021: {e: int(round(v * 1.030)) for e, v in _CBA_2017_BASE_2020.items()},
    2022: {e: int(round(v * 1.061)) for e, v in _CBA_2017_BASE_2020.items()},
}

# 2023 CBA tier ratios (derived from data + Non-Bird ceiling rows).
# Ratio of the paid vet minimum to the base (3+ year) minimum.
_CBA_2023_TIER_RATIO = {
    0: 0.554, 1: 0.850, 2: 0.950, 3: 1.000,
    4: 1.072, 5: 1.162, 6: 1.252, 7: 1.342,
    8: 1.395, 9: 1.448, 10: 1.583,
}
_CBA_2023_BASE_PCT = 0.014848   # 3+ year minimum in cap_pct terms


def _get_vet_min_usd(season: int, exp: int) -> float:
    """CBA veteran minimum (PAID salary) for the given season and experience."""
    tier = min(exp, 10)
    if season < CBA_NEW_ERA_SEASON:
        # 2017 CBA: published Exhibit I figures
        if season in _CBA_2017_SCALE:
            return float(_CBA_2017_SCALE[season].get(tier,
                         _CBA_2017_SCALE[season][10]))
        # Fallback: scale from 2020 base by cap ratio
        cap_ratio = CAP_BY_SEASON.get(season, CAP_BY_SEASON[2020]) / \
                    CAP_BY_SEASON[2020]
        return float(_CBA_2017_BASE_2020.get(tier, _CBA_2017_BASE_2020[10])
                     * cap_ratio)
    else:
        # 2023 CBA: base_pct * cap * tier_ratio
        cap = CAP_BY_SEASON[season]
        ratio = _CBA_2023_TIER_RATIO.get(tier, _CBA_2023_TIER_RATIO[10])
        return _CBA_2023_BASE_PCT * cap * ratio


def _load_eas_for_early_bird() -> dict:
    """Load the EAS (Estimated Average Player Salary) per paying season.

    For Early Bird contracts, the CBA floor is 1.05 x EAS of the PAYING season.
    The EAS figures are curated in extension_raise_caps.csv.
    For season 2026 (missing from the CSV), we extrapolate from cap growth.
    """
    path = RAW_DIR / "raw_external" / "extension_raise_caps.csv"
    rc = pd.read_csv(path)
    # Map signing_season -> implied_eas_usd.  For Early Bird, keyed by paying
    # season: the free-agent's paying season = signing season.
    eas = dict(zip(rc["signing_season"].astype(int),
                   rc["implied_eas_usd"].astype(float)))
    # Extrapolate 2026 if missing
    if 2026 not in eas and 2025 in eas:
        cap_ratio = CAP_BY_SEASON.get(2026, 164_961_000) / \
                    CAP_BY_SEASON.get(2025, 154_647_000)
        eas[2026] = eas[2025] * cap_ratio
    return eas


def attach_mechanism_caps(df: pd.DataFrame, verbose: bool = True
                          ) -> pd.DataFrame:
    """Add mech_cap_pct for Early Bird and Non-Bird rows.

    The mechanism cap is the legal ceiling for the signing mechanism:
      Early Bird: max(1.75 x prior, 1.05 x EAS)
      Non-Bird:   max(1.20 x prior, 1.20 x vet_min)

    For all other signing types, mech_cap_pct is NaN (no mechanism cap).
    The cap only ever LOWERS predictions (same as extension_cap).

    Args:
        df: must carry ``signing_cat``, ``season``, ``prev_cap_pct``, ``age``,
            ``player_name_norm``, and the target column ``cap_pct``.
        verbose: print a summary line and any ceiling violations.

    Returns:
        A copy of ``df`` with the ``mech_cap_pct`` column added.
    """
    from src.model.train import _load_debut_seasons, TARGET
    from scripts.build_external_features import norm
    from src.model.extension_cap import CAP_TOL

    df = df.copy()
    df["mech_cap_pct"] = np.nan
    cat = df["signing_cat"].values
    eas = _load_eas_for_early_bird()
    debut = _load_debut_seasons()

    # Compute experience
    pn_clean = df["player_name_norm"].apply(norm)
    exp = (df["season"] - pn_clean.map(debut)).fillna(
        (df["age"].fillna(25) - 19).clip(lower=0)).astype(int).clip(lower=0)

    n_eb, n_nb, n_eb_viol, n_nb_viol, n_nb_excl = 0, 0, 0, 0, 0

    for idx in df.index:
        row_cat = cat[idx] if idx < len(cat) else df.loc[idx, "signing_cat"]
        s = int(df.loc[idx, "season"])
        cap_s = CAP_BY_SEASON[s]
        prior_cap = CAP_BY_SEASON.get(s - 1, cap_s)
        prior_usd = float(df.loc[idx, "prev_cap_pct"]) * prior_cap
        actual_pct = float(df.loc[idx, TARGET])

        if row_cat == "Early Bird":
            n_eb += 1
            # Early Bird cap = max(1.75 x prior, 1.05 x EAS)
            eas_usd = eas.get(s)
            if eas_usd is None:
                continue  # no EAS for this season
            eb_cap_usd = max(1.75 * prior_usd, 1.05 * eas_usd)
            eb_cap_pct = eb_cap_usd / cap_s
            df.loc[idx, "mech_cap_pct"] = eb_cap_pct
            if actual_pct > eb_cap_pct + CAP_TOL:
                n_eb_viol += 1
                if verbose:
                    print(f"  EB violation: {df.loc[idx, 'player_name_norm']} "
                          f"{s} actual={actual_pct*cap_s/1e6:.3f}M "
                          f"cap={eb_cap_pct*cap_s/1e6:.3f}M "
                          f"over={((actual_pct-eb_cap_pct)*cap_s)/1e6:.3f}M")

        elif row_cat == "Non-Bird":
            pn = str(df.loc[idx, "player_name_norm"])
            # Check for excluded labels
            if (pn, s) in _NONBIRD_LABEL_EXCLUSIONS:
                n_nb_excl += 1
                if verbose:
                    print(f"  NB excluded (label issue): {pn} {s}")
                continue
            n_nb += 1
            e = int(exp.loc[idx])
            vm_usd = _get_vet_min_usd(s, e)
            nb_cap_usd = max(1.20 * prior_usd, 1.20 * vm_usd)
            nb_cap_pct = nb_cap_usd / cap_s
            df.loc[idx, "mech_cap_pct"] = nb_cap_pct
            if actual_pct > nb_cap_pct + MECH_CAP_TOL:
                n_nb_viol += 1
                if verbose:
                    print(f"  NB violation: {pn} {s} exp={e} "
                          f"actual={actual_pct*cap_s/1e6:.3f}M "
                          f"cap={nb_cap_pct*cap_s/1e6:.3f}M "
                          f"over={((actual_pct-nb_cap_pct)*cap_s)/1e6:.3f}M "
                          f"vm={vm_usd/1e6:.3f}M")

    if verbose:
        print(f"\nMechanism caps: {n_eb} Early Bird ({n_eb_viol} violations), "
              f"{n_nb} Non-Bird ({n_nb_viol} violations, {n_nb_excl} excluded)")
    return df


def apply_mechanism_cap(pred: np.ndarray, mech_cap_pct) -> np.ndarray:
    """Clip prediction at the mechanism cap (only LOWERS).

    Args:
        pred: predictions in cap_pct.
        mech_cap_pct: per-row mechanism ceiling in cap_pct, NaN where no cap
            applies. Accepts an array or None (no-op).

    Returns:
        The clipped prediction. Rows with NaN or None mech_cap_pct are
        returned unchanged.
    """
    if mech_cap_pct is None:
        return np.array(pred, dtype=float, copy=True)
    out = np.array(pred, dtype=float, copy=True)
    cap = np.asarray(mech_cap_pct, dtype=float)
    m = ~np.isnan(cap)
    out[m] = np.minimum(out[m], cap[m])
    return out
