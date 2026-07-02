"""Merge all data sources into a single training-ready DataFrame.

Joins:
  - salaries.csv (player, season, salary, cap_pct)
  - impact_metrics.csv (player_name_norm, season, darko_dpm, lebron, rapm, ...)
  - contract_structure_v2.csv (year_in_contract for year-1 filtering)

Then applies feature engineering:
  - base_rating (z-scored composite of DARKO/LEBRON/RAPM)
  - age features (age, age²)
  - availability (3-year weighted GP%)
  - CBA era flag
"""

import unicodedata

import numpy as np
import pandas as pd
from config import PROCESSED_DIR
from src.features.base_rating import add_base_rating
from src.features.age_curve import add_age_features
from src.features.availability import compute_availability
from src.features.cba_constraints import add_cba_features


def _normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def build_dataset() -> pd.DataFrame:
    """Load scraped data, merge, and engineer features."""
    # --- Load data ---
    salaries = pd.read_csv(PROCESSED_DIR / "salaries.csv")
    impact = pd.read_csv(PROCESSED_DIR / "impact_metrics.csv")

    print(f"Salaries: {len(salaries)} rows, {salaries['player'].nunique()} players")
    print(f"Impact:   {len(impact)} rows, {impact['player_name_norm'].nunique()} players")

    salaries["player_name_norm"] = salaries["player"].apply(_normalize_name)

    # --- Merge on normalized name + season ---
    df = salaries.merge(
        impact,
        on=["player_name_norm", "season"],
        how="inner",
        suffixes=("_sal", "_imp"),
    )
    print(f"Merged:   {len(df)} rows, {df['player_name_norm'].nunique()} players")

    if "player" in df.columns:
        df = df.rename(columns={"player": "player_name"})
    elif "player_name_sal" in df.columns:
        df = df.rename(columns={"player_name_sal": "player_name"})

    # Resolve age/games before dropping suffixed columns
    if "age_imp" in df.columns:
        df["age"] = df["age_imp"].fillna(df.get("age_sal", pd.Series(dtype=float)))
    elif "age_sal" in df.columns:
        df["age"] = df["age_sal"]

    if "games_imp" in df.columns:
        df["games"] = df["games_imp"]
    elif "games" not in df.columns:
        game_cols = [c for c in df.columns if c.startswith("games")]
        if game_cols:
            df["games"] = df[game_cols[0]]

    drop_cols = [c for c in df.columns if c.endswith("_sal") or c.endswith("_imp")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # --- Merge height data ---
    height_path = PROCESSED_DIR / "heights.csv"
    if height_path.exists():
        heights = pd.read_csv(height_path)
        if "player_url" in df.columns and "player_url" in heights.columns:
            df = df.merge(heights[["player_url", "height_inches"]], on="player_url", how="left")
        else:
            heights["player_name_norm"] = heights["player_name_bbref"].apply(_normalize_name)
            df = df.merge(heights[["player_name_norm", "height_inches"]], on="player_name_norm", how="left")
        print(f"Heights: matched {df['height_inches'].notna().sum()}/{len(df)} rows")

    # --- Merge agent data ---
    agent_path = PROCESSED_DIR / "agent_data.csv"
    if agent_path.exists():
        agents = pd.read_csv(agent_path)
        agent_lookup = agents.set_index("player_name_norm")["agent"].to_dict()

        import re
        def _try_agent_match(name, lookup):
            clean = _normalize_name(name).replace(".", "").replace("-", " ")
            clean = re.sub(r"\s+", " ", clean).strip()
            if clean in lookup:
                return lookup[clean]
            no_suffix = re.sub(r"\s+(jr|sr|iii|ii|iv)$", "", clean)
            if no_suffix in lookup:
                return lookup[no_suffix]
            if clean + " jr" in lookup:
                return lookup[clean + " jr"]
            if clean + " iii" in lookup:
                return lookup[clean + " iii"]
            return None

        df["agent"] = df["player_name_norm"].apply(lambda x: _try_agent_match(x, agent_lookup))
        agent_matched = df["agent"].notna().sum()
        print(f"Agent data: matched {agent_matched}/{len(df)} rows")
    else:
        df["agent"] = pd.NA
        print("WARNING: agent_data.csv not found")

    # --- Merge contract structure (for year-1 filtering) ---
    cs_path = PROCESSED_DIR / "contract_structure_v2.csv"
    if cs_path.exists():
        cs = pd.read_csv(cs_path)
        df = df.merge(
            cs[["player_name_norm", "season", "year_in_contract", "contract_years"]],
            on=["player_name_norm", "season"],
            how="left",
        )
        matched = df["year_in_contract"].notna().sum()
        print(f"Contract structure: matched {matched}/{len(df)} rows")
    else:
        print("WARNING: contract_structure_v2.csv not found, year_in_contract unavailable")

    # --- Feature engineering ---
    df = add_base_rating(df)
    df = add_age_features(df)
    df = compute_availability(df)
    df = add_cba_features(df)

    # --- Derived workload features ---
    if "minutes" in df.columns and "games" in df.columns:
        df["mpg"] = df["minutes"] / df["games"].replace(0, np.nan)

    # --- Select final columns ---
    feature_cols = [
        "player_name", "player_name_norm", "team", "season",
        "salary", "cap_pct",
        # Impact metrics (raw + z-scored)
        "darko_dpm", "lebron", "rapm",
        "darko_dpm_z", "lebron_z", "rapm_z",
        # Age
        "age", "age_squared",
        # Workload & availability
        "minutes", "games", "mpg", "availability_3yr", "usage_pct",
        # Physical
        "height_inches",
        # CBA
        "cba_era",
        # Box score
        "ast_pct",
        # Agent (for agent_avg_cap computation in train.py)
        "agent",
        # Contract structure (for filtering, not model features)
        "year_in_contract", "contract_years",
        # Metadata (kept for analysis, not model features)
        "team_abbreviation", "position",
    ]

    available = [c for c in feature_cols if c in df.columns]
    df = df[available].copy()

    df = df.dropna(subset=["cap_pct"])

    print(f"\nFinal dataset: {len(df)} rows, {df['player_name_norm'].nunique()} players")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Features: {len(available)} columns")
    yr1 = df["year_in_contract"] == 1
    print(f"Year-1 rows: {yr1.sum()}, Year 2+: {(df['year_in_contract'] > 1).sum()}, "
          f"Unmatched: {df['year_in_contract'].isna().sum()}")

    return df


if __name__ == "__main__":
    df = build_dataset()
    out = PROCESSED_DIR / "training_data.csv"
    try:
        df.to_csv(out, index=False)
    except PermissionError:
        out = PROCESSED_DIR / "training_data_v2.csv"
        df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print(df.describe().to_string())
