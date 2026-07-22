"""Refresh BBRef team contract pages and merge them into salaries.csv.

Team pages list current and future seasons only, so the merge replaces exactly
the (team, season) combinations the fresh scrape covers and keeps every other
row — historical seasons survive untouched, and a team whose page failed keeps
its stale rows rather than vanishing (a full 30-team refresh typically sees one
403 even with backoff; degrade to stale, never to missing).

Before writing, the previous salaries.csv is copied to salaries_prev.csv so
scripts/extend_contract_structure.py can diff old against new to find which
rows are new contracts. Run the two back to back, then
scripts/rebuild_training_data.py:

    python scripts/refresh_salaries.py             # live scrape, ~25 min
    python scripts/extend_contract_structure.py
    python scripts/rebuild_training_data.py

    python scripts/refresh_salaries.py --use-cache # re-parse cached HTML only
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import PROCESSED_DIR
from src.scraping.contracts import (
    BBREF_BASE, TEAM_ABBREVS, _add_cap_pct, scrape_team_contracts,
)
from src.scraping.utils import fetch_html

SALARIES = PROCESSED_DIR / "salaries.csv"
PREV_SIDECAR = PROCESSED_DIR / "salaries_prev.csv"


def scrape_all(use_cache: bool) -> tuple[pd.DataFrame, list[str]]:
    """Fetch and parse every team page. Returns (rows, failed_teams)."""
    frames, failed = [], []
    t0 = time.time()
    for i, team in enumerate(TEAM_ABBREVS, 1):
        try:
            if not use_cache:
                fetch_html(f"{BBREF_BASE}/contracts/{team}.html", force_refresh=True)
            df = scrape_team_contracts(team)
            if df.empty:
                failed.append(team)
                print(f"  [{i:2d}/30] {team}: EMPTY")
            else:
                frames.append(df)
                print(f"  [{i:2d}/30] {team}: {len(df):3d} rows ({time.time()-t0:.0f}s)")
        except Exception as e:
            failed.append(team)
            print(f"  [{i:2d}/30] {team}: ERROR {e}")
    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return fresh, failed


def merge(old: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Replace the (team, season) space the fresh scrape covers; keep the rest.

    Filtering by season alone would delete every team that failed to fetch, so
    the keep-mask also retains any team absent from the fresh rows.
    """
    fresh_seasons = set(fresh["season"].unique())
    fresh_teams = set(fresh["team"].unique())
    keep = old[
        (~old["season"].isin(fresh_seasons)) | (~old["team"].isin(fresh_teams))
    ]
    merged = pd.concat([keep, fresh], ignore_index=True)
    merged = merged.drop_duplicates(subset=["player_url", "season"], keep="last")
    # Recompute cap_pct for every row so a cap corrected in config.py propagates
    # here too. Seasons with no configured cap get NaN, matching prior behavior.
    merged = _add_cap_pct(merged.drop(columns=["salary_cap", "cap_pct"],
                                      errors="ignore"))
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-cache", action="store_true",
                    help="parse the cached HTML instead of re-fetching")
    args = ap.parse_args()

    old = pd.read_csv(SALARIES)
    print(f"existing salaries.csv: {len(old)} rows, "
          f"seasons {old['season'].min()}-{old['season'].max()}\n")

    fresh, failed = scrape_all(args.use_cache)
    if fresh.empty:
        raise SystemExit("nothing scraped — salaries.csv left untouched")
    if failed:
        print(f"\nWARNING: kept stale rows for failed team(s): {failed}")

    merged = merge(old, fresh)

    print(f"\n{'season':>7s} {'before':>7s} {'after':>7s} {'delta':>7s} {'teams':>6s}")
    print("-" * 40)
    for s in sorted(set(old["season"]) | set(merged["season"])):
        b = (old["season"] == s).sum()
        a = (merged["season"] == s).sum()
        t = merged.loc[merged["season"] == s, "team"].nunique()
        print(f"{int(s):7d} {b:7d} {a:7d} {a - b:+7d} {t:6d}")

    shutil.copy(SALARIES, PREV_SIDECAR)
    merged.to_csv(SALARIES, index=False)
    print(f"\nwrote {SALARIES} ({len(merged)} rows)")
    print(f"previous table kept at {PREV_SIDECAR.name} for the structure step")


if __name__ == "__main__":
    main()
