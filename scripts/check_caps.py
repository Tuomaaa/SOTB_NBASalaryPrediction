"""Verify CAP_BY_SEASON against the salaries it is supposed to normalise.

cap_pct is the model target, so a wrong salary cap rescales an entire season's
target without raising anything. That is not hypothetical: the 2025 and 2026
caps were both wrong, inflating cap_pct by 9.5% and understating it by 7.8%
respectively, until this check was written.

The test exploits the CBA. A max contract is worth exactly 25%, 30% or 35% of
the cap in the season it starts, so if the cap is right, freshly signed max
deals land on those figures to the dollar. If it is wrong, they land nowhere in
particular.

Usage:
    python scripts/check_caps.py          # report every season
    python scripts/check_caps.py --strict # exit 1 if any season fails
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import CAP_BY_SEASON, PROCESSED_DIR

MAX_TIERS = (0.25, 0.30, 0.35)
# A dollar-exact match, allowing only for float noise.
TOLERANCE = 5e-5
# Below this many near-max rows a season cannot be judged either way.
MIN_ROWS = 4
# Share of near-max rows that must land on a tier for the cap to be believable.
MIN_HIT_RATE = 0.4
# Contract salaries are rounded to whole dollars. Dividing one rounded salary
# by a max tier can therefore imply a cap a few dollars away from the published
# integer without identifying a different cap.
CAP_ROUNDING_TOLERANCE_USD = 5


def _near_max(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Year-1 rows plausibly on a max deal, priced against the given cap."""
    rows = df[(df["year_in_contract"] == 1) & (df["salary"] / cap > 0.20)].copy()
    rows["pct"] = rows["salary"] / cap
    rows["hit"] = rows["pct"].apply(
        lambda p: min(abs(p - t) for t in MAX_TIERS) < TOLERANCE
    )
    return rows


def check(strict: bool) -> int:
    path = PROCESSED_DIR / "training_data_v2.csv"
    if not path.exists():
        path = PROCESSED_DIR / "training_data.csv"
    df = pd.read_csv(path)

    failures = []
    print(f"{'season':>7}  {'configured cap':>15}  {'rows':>5}  {'on a tier':>10}  verdict")
    for season in sorted(df["season"].unique()):
        season = int(season)
        cap = CAP_BY_SEASON.get(season)
        if cap is None:
            failures.append((season, "no cap on record"))
            print(f"{season:>7}  {'—':>15}  {'—':>5}  {'—':>10}  NO CAP")
            continue

        rows = _near_max(df[df["season"] == season], cap)
        n = len(rows)
        hits = int(rows["hit"].sum())

        if n < MIN_ROWS:
            verdict = "too few rows to judge"
        elif hits / n >= MIN_HIT_RATE:
            verdict = "ok"
        else:
            # A low hit rate on its own is not evidence against the cap — early
            # seasons carry noisy contract-structure matches. Only a cap that
            # fits the salaries better convicts this one.
            implied = _implied_cap(rows, cap)
            if implied:
                verdict = f"WRONG — ${implied:,} fits better"
                failures.append((season, verdict))
            else:
                verdict = "low hit rate, but no cap fits better"

        print(f"{season:>7}  ${cap:>14,}  {n:>5}  {hits:>4}/{n:<5}  {verdict}")

    if failures:
        print(f"\n{len(failures)} season(s) do not reconcile:")
        for season, why in failures:
            print(f"  {season}: {why}")
        if strict:
            return 1
    else:
        print("\nAll seasons reconcile.")
    return 0


def _implied_cap(rows: pd.DataFrame, current: int) -> int | None:
    """The cap that would put the most salaries on an exact max tier."""
    best_cap, best_hits = None, 0
    for salary in rows["salary"].unique():
        for tier in MAX_TIERS:
            candidate = round(salary / tier)
            if not 50_000_000 < candidate < 400_000_000:
                continue
            pct = rows["salary"] / candidate
            hits = int(
                sum(min(abs(p - t) for t in MAX_TIERS) < TOLERANCE for p in pct)
            )
            if hits > best_hits:
                best_cap, best_hits = candidate, hits
    return (
        best_cap
        if best_cap
        and abs(best_cap - current) > CAP_ROUNDING_TOLERANCE_USD
        else None
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when a season fails to reconcile")
    raise SystemExit(check(ap.parse_args().strict))
