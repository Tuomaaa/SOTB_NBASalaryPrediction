"""Build features from raw_external data and merge onto training data."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import unicodedata
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
RAW_EXT = PROJ / "data" / "raw" / "raw_external"
PROC = PROJ / "data" / "processed"


def norm(name):
    name = str(name).strip().lower()
    # Handle "Last, First" format
    if "," in name:
        parts = name.split(",", 1)
        name = parts[1].strip() + " " + parts[0].strip()
    nfkd = unicodedata.normalize("NFKD", name)
    out = "".join(c for c in nfkd if not unicodedata.combining(c))
    out = out.replace(".", "").replace("-", " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ─── 1. Awards ───────────────────────────────────────────────────────
AWARD_WEIGHTS = {
    "MVP": 10,
    "Finals MVP": 8,
    "Defensive Player of the Year": 5,
    "All-NBA 1st Team": 4,
    "All-NBA 2nd Team": 3,
    "All-NBA 3rd Team": 2,
    "All-Defensive 1st Team": 3,
    "All-Defensive 2nd Team": 2,
    "Most Improved Player": 2,
    "Sixth Man of the Year": 2,
    "Rookie of the Year": 2,
    "Clutch Player of the Year": 1,
    "All-Rookie 1st Team": 1,
    "All-Rookie 2nd Team": 0.5,
}

AWARD_MIN_YEAR = 2017
STEP_DECAY = [1.0, 0.85, 0.65, 0.4, 0.1]


def _step_decay_factor(gap):
    if gap < 0:
        return 0
    if gap < len(STEP_DECAY):
        return STEP_DECAY[gap]
    return 0


def build_award_features():
    """Step-decayed award score per player per season (lagged).

    Uses awards_full.csv (2017-2026) with step decay [1, 0.85, 0.65, 0.4, 0.1].
    Awards older than 5 years contribute nothing.
    Lagged: award Year Y maps to season Y+1.
    """
    full_path = RAW_EXT / "awards_full.csv"
    legacy_path = RAW_EXT / "award.csv"
    if full_path.exists():
        aw = pd.read_csv(full_path)
        aw.rename(columns={"year": "Year", "award": "Award"}, inplace=True)
        if "player_name_norm" not in aw.columns:
            aw["player_name_norm"] = aw["player_name"].apply(norm)
    else:
        aw = pd.read_csv(legacy_path)
        aw["player_name_norm"] = aw["Player"].apply(norm)

    aw = aw[aw["Year"] >= AWARD_MIN_YEAR].copy()
    aw["award_pts"] = aw["Award"].map(AWARD_WEIGHTS).fillna(0)
    aw = aw[aw["award_pts"] > 0].copy()
    aw["is_all_nba"] = aw["Award"].str.startswith("All-NBA").astype(int)

    yearly = aw.groupby(["player_name_norm", "Year"]).agg(
        award_pts_year=("award_pts", "sum"),
        all_nba_year=("is_all_nba", "max"),
    ).reset_index()

    train_path = PROC / "training_data_v2.csv"
    if not train_path.exists():
        train_path = PROC / "training_data.csv"
    all_seasons = sorted(pd.read_csv(train_path)["season"].unique())

    rows = []
    for player in yearly["player_name_norm"].unique():
        pdata = yearly[yearly["player_name_norm"] == player].sort_values("Year")
        award_history = list(zip(pdata["Year"].astype(int), pdata["award_pts_year"], pdata["all_nba_year"]))

        for s in all_seasons:
            score = 0.0
            allnba = 0
            for yr, pts, anba in award_history:
                gap = (s - 1) - yr
                factor = _step_decay_factor(gap)
                if factor > 0:
                    score += pts * factor
                    if anba and gap <= 0:
                        allnba += int(anba)
            # all_nba_cum: count within step decay window
            allnba_cum = 0
            for yr, pts, anba in award_history:
                gap = (s - 1) - yr
                if 0 <= gap < len(STEP_DECAY) and anba:
                    allnba_cum += int(anba)

            if score > 0.01 or allnba_cum > 0:
                rows.append({
                    "player_name_norm": player,
                    "season": s,
                    "award_score_cum": round(score, 3),
                    "all_nba_cum": allnba_cum,
                })

    result = pd.DataFrame(rows)
    print(f"Awards: {len(result)} player-season rows, {result['player_name_norm'].nunique()} players")
    return result


# ─── 2. Team Value ───────────────────────────────────────────────────
def build_team_value_features():
    """Team franchise value in billions."""
    tv = pd.read_csv(RAW_EXT / "long_table_team_value.with_ids.csv")
    # Parse value: "$4.3B" -> 4.3
    tv["team_value_B"] = tv["Value"].str.replace("$", "").str.replace("B", "").astype(float)
    # Team abbreviation mapping (RealGM full name -> BBRef abbrev)
    # We'll match by the Team column directly since it's abbreviations
    tv = tv.rename(columns={"Team": "team_abbreviation", "Season": "season"})
    result = tv[["team_abbreviation", "season", "team_value_B"]].copy()
    print(f"Team value: {len(result)} team-season rows")
    return result


# ─── 3. Draft Position ──────────────────────────────────────────────
def build_draft_features():
    """Draft pick number per player."""
    dr = pd.read_csv(RAW_EXT / "player_draft_2020-2025.matched.corrected.csv")
    dr["player_name_norm"] = dr["player"].apply(norm)

    # Convert overall_pick to numeric (undrafted -> NaN)
    dr["draft_pick"] = pd.to_numeric(dr["overall_pick"], errors="coerce")
    dr["is_lottery"] = (dr["draft_pick"] <= 14).astype(float)
    dr["is_top5"] = (dr["draft_pick"] <= 5).astype(float)

    # Undrafted players
    dr.loc[dr["round_number"] == "Undrafted", "draft_pick"] = 75  # sentinel for undrafted
    dr.loc[dr["round_number"] == "Undrafted", "is_lottery"] = 0
    dr.loc[dr["round_number"] == "Undrafted", "is_top5"] = 0

    # Keep one row per player (use earliest draft year if duplicates)
    result = dr.sort_values("year").drop_duplicates("player_name_norm", keep="first")
    result = result[["player_name_norm", "draft_pick", "is_lottery", "is_top5"]].copy()
    print(f"Draft: {len(result)} players ({(result['is_lottery']==1).sum()} lottery, {(result['is_top5']==1).sum()} top-5)")
    return result


# ─── 4. Injury Frequency ────────────────────────────────────────────
def build_injury_features():
    """Games missed per player-season from injury reports."""
    # 2021-2024 dataset
    inj1 = pd.read_csv(RAW_EXT / "Injury Database - Oct 2021 - June 2024 (1).csv")
    inj1["player_name_norm"] = inj1["PLAYER"].apply(norm)
    inj1["DATE"] = pd.to_datetime(inj1["DATE"], format="mixed", errors="coerce")
    # Map to our season convention: Oct 2021 -> season 2022 (2021-22)
    # Actually our convention: season=2022 means 2022-23 salary, 2021-22 stats
    # Games played in Oct 2021 - Jun 2022 = 2021-22 season = stats for season 2022
    # Wait, let me be more careful:
    # Our season convention: season=2025 means 2025-26 salary, 2024-25 stats
    # A game on Oct 2024 is in the 2024-25 NBA season -> maps to our season=2025
    inj1["nba_season_year"] = inj1["DATE"].apply(
        lambda d: d.year + 1 if pd.notna(d) and d.month >= 10 else (d.year if pd.notna(d) else None)
    )
    # That gives us the END year of the NBA season (2024-25 -> 2025)
    # In our convention, season=2025 uses 2024-25 stats, so nba_season_year=2025 -> season=2025
    # But wait - season=2025 means stats from 2024-25. nba_season_year for Oct2024-Jun2025 = 2025. Match.
    inj1_out = inj1[inj1["STATUS"] == "Out"].groupby(
        ["player_name_norm", "nba_season_year"]
    ).size().reset_index(name="injury_reports")
    inj1_out = inj1_out.rename(columns={"nba_season_year": "season"})
    inj1_out["season"] = inj1_out["season"].astype("Int64")

    # 2010-2020 dataset (different format: "Relinquished" = player going to IL)
    inj2 = pd.read_csv(RAW_EXT / "injuries_2010-2020.csv")
    inj2["Date"] = pd.to_datetime(inj2["Date"], format="mixed", errors="coerce")
    inj2["player_name_norm"] = inj2["Relinquished"].apply(
        lambda x: norm(x) if pd.notna(x) else None
    )
    inj2 = inj2[inj2["player_name_norm"].notna()]
    inj2["nba_season_year"] = inj2["Date"].apply(
        lambda d: d.year + 1 if pd.notna(d) and d.month >= 10 else (d.year if pd.notna(d) else None)
    )
    inj2_out = inj2.groupby(
        ["player_name_norm", "nba_season_year"]
    ).size().reset_index(name="injury_reports")
    inj2_out = inj2_out.rename(columns={"nba_season_year": "season"})
    inj2_out["season"] = inj2_out["season"].astype("Int64")

    result = pd.concat([inj1_out, inj2_out], ignore_index=True)
    # Sum if overlap
    result = result.groupby(["player_name_norm", "season"])["injury_reports"].sum().reset_index()
    print(f"Injuries: {len(result)} player-season rows, {result['player_name_norm'].nunique()} players")
    return result


# ─── Merge all onto training data ────────────────────────────────────
def main():
    train = pd.read_csv(PROC / "training_data_v2.csv")
    ext_cols = ["award_score_cum", "all_nba_cum", "team_value_B",
                "draft_pick", "is_lottery", "is_top5", "injury_reports"]
    train = train.drop(columns=[c for c in ext_cols if c in train.columns])
    train["pn_clean"] = train["player_name_norm"].apply(norm)
    n0 = len(train)
    print(f"Training data: {n0} rows, {train.columns.tolist()}\n")

    # 1. Awards
    awards = build_award_features()
    awards["pn_clean"] = awards["player_name_norm"].apply(norm)
    train = train.merge(
        awards[["pn_clean", "season", "award_score_cum", "all_nba_cum"]],
        on=["pn_clean", "season"], how="left",
    )
    train["award_score_cum"] = train["award_score_cum"].fillna(0)
    train["all_nba_cum"] = train["all_nba_cum"].fillna(0)
    print(f"  After awards merge: {len(train)} rows, award>0: {(train['award_score_cum']>0).sum()}")

    # 2. Team value
    team_val = build_team_value_features()
    train = train.merge(team_val, on=["team_abbreviation", "season"], how="left")
    print(f"  After team value merge: {len(train)} rows, matched: {train['team_value_B'].notna().sum()}")

    # 3. Draft
    draft = build_draft_features()
    draft["pn_clean"] = draft["player_name_norm"].apply(norm)
    train = train.merge(
        draft[["pn_clean", "draft_pick", "is_lottery", "is_top5"]],
        on="pn_clean", how="left",
    )
    train["draft_pick"] = train["draft_pick"].fillna(75)  # undrafted/unknown
    train["is_lottery"] = train["is_lottery"].fillna(0)
    train["is_top5"] = train["is_top5"].fillna(0)
    print(f"  After draft merge: {len(train)} rows, lottery: {(train['is_lottery']==1).sum()}")

    # 4. Injuries
    injuries = build_injury_features()
    injuries["pn_clean"] = injuries["player_name_norm"].apply(norm)
    train = train.merge(
        injuries[["pn_clean", "season", "injury_reports"]],
        on=["pn_clean", "season"], how="left",
    )
    train["injury_reports"] = train["injury_reports"].fillna(0)
    print(f"  After injury merge: {len(train)} rows, has_injury: {(train['injury_reports']>0).sum()}")

    # Verify no row expansion
    assert len(train) == n0, f"Row count changed: {n0} -> {len(train)}"

    # Drop helper column
    train = train.drop(columns=["pn_clean"])

    # Correlations with cap_pct
    print(f"\nNew feature correlations with cap_pct:")
    new_feats = ["award_score_cum", "all_nba_cum", "team_value_B", "draft_pick", "is_lottery", "is_top5", "injury_reports"]
    for f in new_feats:
        if f in train.columns:
            corr = train[f].corr(train["cap_pct"])
            print(f"  {f:25s} corr={corr:+.3f}  non-zero={train[f].notna().sum()}")

    # Save
    out = PROC / "training_data_v2.csv"
    train.to_csv(out, index=False)
    print(f"\nSaved: {out} ({len(train)} rows, {len(train.columns)} columns)")
    print(f"Columns: {sorted(train.columns.tolist())}")


if __name__ == "__main__":
    main()
