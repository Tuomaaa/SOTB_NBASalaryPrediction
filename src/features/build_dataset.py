"""Merge all data sources into a single training-ready DataFrame.

Joins:
  - salaries.csv (player, season, salary, cap_pct)
  - impact_metrics.csv (player_name_norm, season, darko_dpm, lebron, laker, ...)
  - contract_structure_v2.csv (year_in_contract for year-1 filtering)

Then applies feature engineering:
  - base_rating (z-scored composite of DARKO/LEBRON/LAKER)
  - age features (age, age²)
  - availability (3-year weighted GP%)
  - CBA era flag
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from config import CAP_BY_SEASON, PROCESSED_DIR, RAW_DIR, SEASONS
from src.features.base_rating import add_base_rating
from src.features.age_curve import add_age_features
from src.features.availability import compute_availability
from src.features.cba_constraints import add_cba_features
from src.features.impact_identity import (
    apply_impact_corrections,
    apply_player_identity_corrections,
    coalesce_impact_rows,
    fill_age_from_player_history,
    fill_impact_from_bbref,
)
from src.features.waiver_history import attach_waiver_history


def _apply_min_cap_charge_corrections(salaries: pd.DataFrame) -> pd.DataFrame:
    """Replace paid-convention minimum salaries with the cap-charge amount.

    Reads 'min_cap_charge' entries from salary_corrections.csv and overwrites
    the salary (and recomputes cap_pct) for matching (player_name_norm, season)
    rows.  See ISSUES #38 and scripts/fix_minimum_convention.py.
    """
    corr_path = RAW_DIR / "raw_external" / "salary_corrections.csv"
    if not corr_path.exists():
        return salaries

    corr = pd.read_csv(corr_path)
    mc = corr[corr["kind"] == "min_cap_charge"]
    if mc.empty:
        return salaries

    lookup = {
        (row["player_name_norm"], int(row["season"])): float(row["value_usd"])
        for _, row in mc.iterrows()
    }

    applied = 0
    for idx, row in salaries.iterrows():
        key = (row["player_name_norm"], int(row["season"]))
        if key in lookup:
            cap_charge = lookup[key]
            salaries.at[idx, "salary"] = cap_charge
            cap = CAP_BY_SEASON.get(int(row["season"]))
            if cap:
                salaries.at[idx, "cap_pct"] = cap_charge / cap
            applied += 1

    print(f"Min cap-charge corrections: {applied} rows "
          f"(of {len(lookup)} in file)")
    return salaries


def _normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _apply_spotrac_salary_migration(salaries: pd.DataFrame) -> pd.DataFrame:
    """Replace BBRef salaries with the audited Spotrac/BBRef merged values.

    `rebuild_training_data.py` regenerates the migration table immediately
    before this function runs. Full key coverage is mandatory for configured
    training seasons: a stale table must fail loudly, because silently keeping
    BBRef would reintroduce the future-option and stretched-dead-money defects
    the migration exists to remove.
    """
    path = PROCESSED_DIR / "merged_salaries.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required; run scripts/rebuild_training_data.py"
        )

    # Preserve the literal normalized key "nan". One legacy BBRef salary row
    # has a missing player name, which `_normalize_name` represents as that
    # string; pandas otherwise converts the same key in the CSV back to NaN and
    # makes the freshly generated overlay fail its full-coverage guard.
    merged = pd.read_csv(path, keep_default_na=False, na_values=[""])
    key = ["player_name_norm", "season"]
    if merged.duplicated(key).any():
        raise RuntimeError("merged_salaries.csv has duplicate player-season keys")
    overlay = merged[key + ["salary", "source", "branch"]].rename(columns={
        "salary": "_merged_salary",
        "source": "salary_source",
        "branch": "salary_branch",
    })
    out = salaries.merge(
        overlay, on=key, how="left", validate="one_to_one", indicator=True
    )
    required = out["season"].isin(SEASONS)
    missing = required & out["_merge"].ne("both")
    if missing.any():
        sample = out.loc[missing, key].head(20).to_dict("records")
        raise RuntimeError(
            f"merged_salaries.csv is stale: {int(missing.sum())} configured "
            f"salary keys are missing; sample={sample}"
        )

    retained_only = required & out["salary_branch"].eq("retained_only")
    dropped_salary_keys = set(
        zip(
            out.loc[retained_only, "player_name_norm"],
            out.loc[retained_only, "season"].astype(int),
        )
    )
    if retained_only.any():
        print(f"Spotrac salary migration: dropped {int(retained_only.sum())} "
              "retained/dead-money-only rows")
        out = out.loc[~retained_only].copy()
        required = out["season"].isin(SEASONS)
    invalid = required & out["_merged_salary"].isna()
    if invalid.any():
        sample = out.loc[invalid, key + ["salary_branch"]].head(20)
        raise RuntimeError(
            "merged_salaries.csv has unresolved configured salaries:\n"
            + sample.to_string(index=False)
        )

    # Spotrac can publish fractional-dollar prorations. Promote the BBRef
    # integer column before overlaying them; pandas 3 rejects lossy float-into-
    # int assignment instead of silently upcasting the destination column.
    out["salary"] = pd.to_numeric(out["salary"], errors="raise").astype(float)
    out["_merged_salary"] = pd.to_numeric(
        out["_merged_salary"], errors="raise"
    ).astype(float)
    changed = required & ~np.isclose(
        out["salary"], out["_merged_salary"],
        equal_nan=True,
    )
    out.loc[required, "salary"] = out.loc[required, "_merged_salary"]
    out["cap_pct"] = out["salary"] / out["season"].map(CAP_BY_SEASON)
    out = out.drop(columns=["_merged_salary", "_merge"])
    out.attrs["salary_migration_dropped_keys"] = dropped_salary_keys
    print(f"Spotrac salary migration: {int(required.sum())} rows covered, "
          f"{int(changed.sum())} salaries changed")
    return out


def _resolve_merged_age(df: pd.DataFrame) -> pd.Series:
    """Prefer season-specific impact age, then one latest salary-page anchor.

    BBRef team contract pages repeat the player's current age on every future
    salary row. Treating all of those values as season-specific creates
    conflicting age-season offsets. For players whose impact history has no
    age at all, the latest joined salary season is the current-age anchor; the
    player-history pass below infers the remaining seasons from that one fact.
    """
    impact_age = df.get("age_imp", pd.Series(np.nan, index=df.index))
    salary_age = df.get("age_sal", pd.Series(np.nan, index=df.index))
    has_impact_age = impact_age.notna().groupby(
        df["player_name_norm"]
    ).transform("any")
    eligible_salary_season = df["season"].where(
        salary_age.notna() & ~has_impact_age
    )
    latest_salary_season = eligible_salary_season.groupby(
        df["player_name_norm"]
    ).transform("max")
    use_salary_anchor = (
        ~has_impact_age
        & salary_age.notna()
        & df["season"].eq(latest_salary_season)
    )
    return impact_age.where(~use_salary_anchor, salary_age)


def build_dataset() -> pd.DataFrame:
    """Load scraped data, merge, and engineer features."""
    # --- Load data ---
    salaries = pd.read_csv(PROCESSED_DIR / "salaries.csv")
    impact = pd.read_csv(PROCESSED_DIR / "impact_metrics.csv")

    print(f"Salaries: {len(salaries)} rows, {salaries['player'].nunique()} players")
    print(f"Impact:   {len(impact)} rows, {impact['player_name_norm'].nunique()} players")

    salaries["player_name_norm"] = salaries["player"].apply(_normalize_name)
    salaries = _apply_spotrac_salary_migration(salaries)
    salary_migration_dropped_keys = salaries.attrs.get(
        "salary_migration_dropped_keys", set()
    )
    salaries = _apply_min_cap_charge_corrections(salaries)
    salary_keys = set(zip(salaries["player_name_norm"], salaries["season"].astype(int)))
    before = len(impact)
    impact = coalesce_impact_rows(impact, salary_keys=salary_keys)
    bbref_path = PROCESSED_DIR / "advanced_stats.csv"
    if bbref_path.exists():
        bbref = pd.read_csv(bbref_path)
        bbref["player_name_norm"] = bbref["player"].apply(_normalize_name)
        impact = fill_impact_from_bbref(impact, bbref)
        print(f"BBRef advanced fallback: {len(bbref)} priced-season rows")
    impact = apply_impact_corrections(impact)
    impact = fill_age_from_player_history(impact)
    print(f"Impact identity: coalesced {before - len(impact)} split source rows")

    # --- Merge on normalized name + season ---
    df = salaries.merge(
        impact,
        on=["player_name_norm", "season"],
        how="inner",
        suffixes=("_sal", "_imp"),
    )
    print(f"Merged:   {len(df)} rows, {df['player_name_norm'].nunique()} players")

    if bbref_path.exists() and "player_url" in df.columns:
        df = fill_impact_from_bbref(
            df, bbref, key_columns=("player_url", "season")
        )

    if "player" in df.columns:
        df = df.rename(columns={"player": "player_name"})
    elif "player_name_sal" in df.columns:
        df = df.rename(columns={"player_name_sal": "player_name"})

    # Resolve age/games before dropping suffixed columns
    if "age_imp" in df.columns:
        df["age"] = _resolve_merged_age(df)
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

    df = apply_player_identity_corrections(
        df, allowed_missing=salary_migration_dropped_keys
    )
    # Salary pages or corrections sometimes carry the only observed age for a
    # player. Infer every other season from that player-specific anchor.
    df = fill_age_from_player_history(df)

    if bbref_path.exists():
        covered_seasons = set(bbref["season"].astype(int))
        missing_seasons = set(SEASONS) - covered_seasons
        if missing_seasons:
            raise ValueError(
                f"BBRef advanced fallback is missing seasons: {sorted(missing_seasons)}"
            )
        absent_from_bbref = ~pd.MultiIndex.from_frame(
            df[["player_url", "season"]]
        ).isin(pd.MultiIndex.from_frame(bbref[["player_url", "season"]]))
        did_not_play = df["games"].isna() & absent_from_bbref & df["player_url"].notna()
        df.loc[did_not_play, ["games", "minutes", "mpg"]] = 0.0
    else:
        did_not_play = pd.Series(False, index=df.index)

    used_bbref = df.get("_bbref_fallback_used", pd.Series(False, index=df.index))
    df["workload_source_status"] = np.select(
        [df["games"].eq(0), used_bbref, df["games"].notna()],
        ["did_not_play", "bbref_fallback", "nbarapm"],
        default="unresolved",
    )
    df = df.drop(columns=["_bbref_fallback_used"], errors="ignore")

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

    df = attach_waiver_history(df)
    known = int(df["is_waived_known"].sum())
    positives = int(df["is_waived"].fillna(0).sum())
    print(f"Waiver history: known {known}/{len(df)}, positives {positives}")


    # --- Feature engineering ---
    df = add_base_rating(df)
    df = add_age_features(df)
    df = compute_availability(df)
    df = add_cba_features(df)

    # --- Derived workload features ---
    if "minutes" in df.columns and "games" in df.columns:
        computed_mpg = df["minutes"] / df["games"].replace(0, np.nan)
        if "mpg" in df.columns:
            df["mpg"] = computed_mpg.fillna(df["mpg"])
        else:
            df["mpg"] = computed_mpg

    # --- Select final columns ---
    feature_cols = [
        "player_name", "player_name_norm", "team", "season",
        "salary", "cap_pct",
        # Impact metrics (raw + z-scored)
        "darko_dpm", "lebron", "laker",
        "darko_dpm_z", "lebron_z", "laker_z",
        # Age
        "age", "age_squared",
        # Workload & availability
        "minutes", "games", "mpg", "availability_3yr",
        "availability_3yr_coverage", "usage_pct",
        "workload_source_status",
        # Physical
        "height_inches",
        # CBA
        "cba_era",
        # Box score
        # Previous-contract termination context and source audit
        "is_waived", "is_waived_known",
        "prior_waiver_date", "prior_waiver_text",

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
