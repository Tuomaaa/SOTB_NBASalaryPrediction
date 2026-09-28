"""Per-season team assignment for every player, sourced from Spotrac.

Produces `data/processed/player_teams.csv`, one row per (player, season) in the
repo's season convention (a season is named by its STARTING year, so 2023 is
the 2023-24 season).

WHY THIS FILE EXISTS

`training_data_v2.csv` already carries a `team_abbreviation`, and it is ONE
SEASON STALE. That column rides in with the impact metrics from nbarapm.com
(`src/scraping/advanced.py`), and those metrics are deliberately lagged: the
model prices a contract signed in the summer of season N using the performance
the market could actually see, which is season N-1. Lagging the stats is
correct. Lagging the TEAM is not — it labels a player with the team he left.

The salaries confirm the diagnosis to the dollar. Damian Lillard's repo season
2023 pays $45,640,084, which is his 2023-24 Milwaukee salary, while
`team_abbreviation` reads POR. Fred VanVleet's repo 2023 pays $40,806,300, his
first Houston year, labelled TOR. Spotrac's career table prints both the same
salary and the correct team for the same season key, so the join is verified by
an exact-dollar match rather than by name alone.

NOTHING HERE REACHES THE MODEL. No team column is in `FEATURE_COLS` (all 21 are
performance, age, contract and award terms), and `team_value_B` — the one
feature ever derived from a team code — was rejected. This table is display and
analysis metadata, so adopting it moves no published number and needs no
version bump.

WHICH TABLE THE TEAM COMES FROM

`spotrac_salaries.csv` already resolves to one row per player-season across
Spotrac's two table shapes, so this script reads it rather than re-parsing the
847 cached pages:

  career table    settled seasons — the teams the player was actually on,
                  including every team of a traded season.
  contract table  future seasons — the live cap sheet, which is the only
                  source that knows where a player signed this summer.

MULTI-TEAM SEASONS

Spotrac lists the season-ENDING team FIRST. Verified on three unambiguous
mid-season trades: Harden 2020-21 (HOU -> BKN) prints `BKN;HOU`, Doncic
2024-25 (DAL -> LAL) prints `LAL;DAL`, Butler 2024-25 (MIA -> GSW) prints
`GS;MIA`. So `team` takes the first code and means "where the season ended".

That rule is NOT reliable for three-team seasons — Westbrook 2022-23 ran
LAL -> UTA -> LAC and prints `UTA;LAC;LAL`, whose first code is neither the
first nor the last team. Those rows carry `team_order_verified = False` so a
consumer can decline to trust the primary code. The full ordered list is kept
in `teams_all` either way, so nothing is discarded.

Usage:
    python scripts/build_player_teams.py            # dry run + audit
    python scripts/build_player_teams.py --write
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import PROCESSED_DIR

SPOTRAC = PROCESSED_DIR / "spotrac_salaries.csv"
TRAINING = PROCESSED_DIR / "training_data_v2.csv"
OUT = PROCESSED_DIR / "player_teams.csv"

# Spotrac's codes to the repo's. The repo follows Basketball Reference, which
# differs from Spotrac on eight live franchises plus two pre-2013 names that
# only appear in the deep career tails.
TEAM_CODE = {
    "GS": "GSW", "WSH": "WAS", "BKN": "BRK", "CHA": "CHO",
    "NO": "NOP", "PHX": "PHO", "SA": "SAS", "NY": "NYK",
    "NJN": "BRK", "NOH": "NOP",
}

REPO_CODES = {
    "ATL", "BOS", "BRK", "CHI", "CHO", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}


def split_teams(raw) -> list[str]:
    """The row's team codes, in Spotrac's order, translated to repo codes."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    out = []
    for part in str(raw).split(";"):
        part = part.strip()
        if not part:
            continue
        code = TEAM_CODE.get(part, part)
        if code not in out:
            out.append(code)
    return out


def build() -> pd.DataFrame:
    """One team assignment per player-season."""
    sp = pd.read_csv(SPOTRAC)
    sp["parts"] = sp["teams"].map(split_teams)
    sp = sp[sp["parts"].str.len() > 0].copy()

    unknown = sorted({c for parts in sp["parts"] for c in parts} - REPO_CODES)
    if unknown:
        raise SystemExit(f"unmapped team codes, refusing to write: {unknown}")

    sp["team"] = sp["parts"].str[0]
    sp["teams_all"] = sp["parts"].str.join(";")
    sp["n_teams"] = sp["parts"].str.len()
    # The season-ending-first rule is only established for two-team rows.
    sp["team_order_verified"] = sp["n_teams"] <= 2

    out = sp[["player_name_norm", "season", "team", "teams_all", "n_teams",
              "team_order_verified", "table_kind"]].copy()
    out = out.rename(columns={"table_kind": "source_table"})
    return out.sort_values(["player_name_norm", "season"]).reset_index(drop=True)


def audit(teams: pd.DataFrame) -> None:
    """Report coverage of the training frame and the size of the correction."""
    td = pd.read_csv(TRAINING,
                     usecols=["player_name_norm", "season", "team_abbreviation"])
    m = td.merge(teams, on=["player_name_norm", "season"], how="left")

    print(f"\n{'=' * 68}\nCOVERAGE OF training_data_v2.csv: {len(m)} rows\n{'=' * 68}")
    have = m["team"].notna()
    print(f"resolved by Spotrac : {have.sum()} ({have.mean():.2%})")
    print(f"unresolved          : {(~have).sum()}")

    print("\nby season:")
    by = m.groupby("season").agg(
        rows=("team", "size"),
        resolved=("team", lambda s: s.notna().sum()),
        stale_now=("team", "size"),
    )
    both = m[m["team"].notna() & m["team_abbreviation"].notna()]
    diff = both.groupby("season").apply(
        lambda g: (g["team"] != g["team_abbreviation"]).sum(), include_groups=False)
    by["corrected"] = diff
    by["rate"] = by["resolved"] / by["rows"]
    print(by[["rows", "resolved", "rate", "corrected"]].to_string())

    changed = both[both["team"] != both["team_abbreviation"]]
    print(f"\nrows whose team CHANGES: {len(changed)} of {len(both)} "
          f"({len(changed) / len(both):.1%})")

    print(f"\nmulti-team seasons: {(m['n_teams'] > 1).sum()} "
          f"({(m['n_teams'] > 2).sum()} with an unverified primary code)")

    print("\nsample corrections (settled seasons — verify a few by hand):")
    sample = changed[changed["season"].between(2022, 2024)].head(15)
    print(f"{'player':24s} {'szn':>5s} {'was':>5s} {'now':>5s}  {'teams':<14s}")
    for r in sample.itertuples():
        print(f"{r.player_name_norm:24s} {r.season:5d} "
              f"{r.team_abbreviation:>5s} {r.team:>5s}  {r.teams_all:<14s}")

    if (~have).any():
        miss = m[~have][["player_name_norm", "season"]]
        print(f"\nunresolved rows ({len(miss)}) — these keep team_abbreviation:")
        print(miss.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    teams = build()
    print(f"built {len(teams)} player-season team assignments")
    print(f"  seasons {int(teams['season'].min())}-{int(teams['season'].max())}")
    print(f"  players {teams['player_name_norm'].nunique()}")
    print("  by source table:")
    print(teams["source_table"].value_counts().to_string())

    audit(teams)

    if not args.write:
        print("\ndry run — nothing written. Re-run with --write.")
        return
    teams.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(teams)} rows)")


if __name__ == "__main__":
    main()
