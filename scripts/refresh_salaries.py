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
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import PROCESSED_DIR, SEASONS
from src.scraping.contracts import (
    BBREF_BASE, TEAM_ABBREVS, _add_cap_pct, scrape_team_contracts,
)
from src.scraping.utils import fetch_html

SALARIES = PROCESSED_DIR / "salaries.csv"
PREV_SIDECAR = PROCESSED_DIR / "salaries_prev.csv"
SPOTRAC_SALARIES = PROCESSED_DIR / "spotrac_salaries.csv"
SPOTRAC_FA_SIGNINGS = PROCESSED_DIR / "spotrac_fa_signings.csv"
ADVANCED_STATS = PROCESSED_DIR / "advanced_stats.csv"

SPOTRAC_TO_BBREF_TEAM = {
    "BKN": "BRK",
    "CHA": "CHO",
    "GS": "GSW",
    "NJN": "BRK",
    "NO": "NOP",
    "NOH": "NOP",
    "NY": "NYK",
    "PHX": "PHO",
    "SA": "SAS",
    "WSH": "WAS",
}


def _normalize_name(name: str) -> str:
    """Project-wide player key, identical to build_dataset._normalize_name."""
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


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


def add_spotrac_only_rows(
    merged: pd.DataFrame,
    spotrac: pd.DataFrame,
    fresh_seasons: set[int],
    identity: pd.DataFrame | None = None,
    fa_signings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add current contract rows that BBRef team pages omit.

    Spotrac is allowed to expand the base row set only inside seasons that the
    current BBRef refresh actually covered. A row must be a positive,
    single-team contract and must resolve to an existing BBRef player identity.
    This keeps the salary key usable by downstream BBRef feature joins and
    prevents a partial Spotrac scrape from rewriting historical membership.
    """
    out = merged.copy()
    out["_player_name_norm"] = out["player"].map(_normalize_name)
    existing_keys = set(zip(out["_player_name_norm"], out["season"]))

    candidates = spotrac.copy()
    if fa_signings is not None and not fa_signings.empty:
        required = {
            "player_name_norm", "season", "to_team", "aav_str",
            "contract_years",
        }
        if not required.issubset(fa_signings.columns):
            raise ValueError(
                f"FA signing frame must contain {sorted(required)}"
            )
        fa = fa_signings.copy()
        fa["salary"] = pd.to_numeric(
            fa["aav_str"].astype(str).str.replace(
                r"[^0-9.-]", "", regex=True
            ),
            errors="coerce",
        )
        fa["teams"] = fa["to_team"]
        fa["n_teams"] = 1
        fa["table_kind"] = "contract"
        fa = fa[
            pd.to_numeric(fa["contract_years"], errors="coerce").eq(1)
        ]
        # A parsed player page is the primary salary source. The signed-FA
        # table is a freshness fallback for a just-announced one-year deal
        # whose player page has not updated yet.
        candidates = pd.concat(
            [candidates, fa[candidates.columns.intersection(fa.columns)]],
            ignore_index=True,
        )
    candidates["season"] = pd.to_numeric(
        candidates["season"], errors="coerce"
    )
    candidates["salary"] = pd.to_numeric(
        candidates["salary"], errors="coerce"
    )
    candidates["n_teams"] = pd.to_numeric(
        candidates["n_teams"], errors="coerce"
    )
    candidates = candidates[
        candidates["season"].isin(set(fresh_seasons) & set(SEASONS))
        & candidates["table_kind"].eq("contract")
        & candidates["salary"].gt(0)
        & candidates["n_teams"].eq(1)
        & candidates["teams"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    candidates["season"] = candidates["season"].astype(int)
    candidates = candidates.drop_duplicates(
        ["player_name_norm", "season"], keep="first"
    )
    candidates = candidates[
        ~candidates.apply(
            lambda row: (row["player_name_norm"], row["season"])
            in existing_keys,
            axis=1,
        )
    ]

    identity_frames = [
        out[["player", "player_url", "age", "season", "_player_name_norm"]]
    ]
    if identity is not None and not identity.empty:
        required = {"player", "player_url", "season"}
        if not required.issubset(identity.columns):
            raise ValueError(
                f"identity frame must contain {sorted(required)}"
            )
        extra = identity.copy()
        extra["_player_name_norm"] = extra["player"].map(_normalize_name)
        if "age" not in extra.columns:
            extra["age"] = pd.NA
        identity_frames.append(
            extra[["player", "player_url", "age", "season",
                   "_player_name_norm"]]
        )
    identities = pd.concat(identity_frames, ignore_index=True)
    identities = identities.dropna(subset=["player_url"])
    identities = identities[identities["player_url"].astype(str).str.len() > 0]
    identities = identities.sort_values("season").drop_duplicates(
        "_player_name_norm", keep="last"
    ).set_index("_player_name_norm")

    rows = []
    skipped = []
    for row in candidates.itertuples(index=False):
        key = row.player_name_norm
        if key not in identities.index:
            skipped.append((key, int(row.season), "no BBRef identity"))
            continue
        person = identities.loc[key]
        team = SPOTRAC_TO_BBREF_TEAM.get(str(row.teams), str(row.teams))
        rows.append({
            "player": person["player"],
            "player_url": person["player_url"],
            "team": team,
            "season": int(row.season),
            "salary": float(row.salary),
            "age": person["age"],
            "source": "spotrac_fa_backfill",
            "team_name": pd.NA,
        })

    out = out.drop(columns="_player_name_norm")
    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
        out = out.drop_duplicates(
            subset=["player_url", "season"], keep="last"
        )
        out = _add_cap_pct(
            out.drop(columns=["salary_cap", "cap_pct"], errors="ignore")
        )
    out.attrs["spotrac_only_added"] = [
        (_normalize_name(r["player"]), r["season"]) for r in rows
    ]
    out.attrs["spotrac_only_skipped"] = skipped
    return out


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
    if not SPOTRAC_SALARIES.exists():
        raise SystemExit(
            f"{SPOTRAC_SALARIES.name} is required for the Spotrac migration"
        )
    spotrac = pd.read_csv(SPOTRAC_SALARIES, keep_default_na=False,
                          na_values=[""])
    fa_signings = (pd.read_csv(SPOTRAC_FA_SIGNINGS)
                   if SPOTRAC_FA_SIGNINGS.exists() else pd.DataFrame())
    identity = (pd.read_csv(ADVANCED_STATS)
                if ADVANCED_STATS.exists() else pd.DataFrame())
    merged = add_spotrac_only_rows(
        merged,
        spotrac,
        set(pd.to_numeric(fresh["season"], errors="coerce").dropna().astype(int)),
        identity,
        fa_signings,
    )
    added = merged.attrs["spotrac_only_added"]
    skipped = merged.attrs["spotrac_only_skipped"]
    print(f"\nSpotrac-only current contracts added: {len(added)}")
    for key in added:
        print(f"  ADDED {key}")
    if skipped:
        print(f"Spotrac-only candidates skipped: {len(skipped)}")
        for key in skipped[:20]:
            print(f"  SKIPPED {key}")

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
