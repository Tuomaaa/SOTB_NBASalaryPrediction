"""Identity repair and sourced corrections for the impact-metric join."""

import numpy as np
import pandas as pd

from config import RAW_DIR


CORRECTIONS = RAW_DIR / "raw_external" / "impact_metric_corrections.csv"
IDENTITY_CORRECTIONS = RAW_DIR / "raw_external" / "player_identity_corrections.csv"
CORRECTION_META = {"player_name_norm", "season", "source", "note"}
BBREF_FALLBACK_COLUMNS = {
    "age": "age",
    "games": "games",
    "minutes": "minutes",
    "usg_pct": "usage_pct",
    "ast_pct": "ast_pct",
    "position": "position",
    "team": "team_abbreviation",
}


def _id_key(value) -> str | None:
    """Normalize an NBA id without turning missing ids into the string 'nan'."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def coalesce_impact_rows(
    impact: pd.DataFrame,
    salary_keys: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Coalesce source rows split by a player-name variant.

    DARKO, LEBRON and LAKER sometimes spell the same player differently. The
    old outer merge keyed only on normalized name, leaving two complementary
    rows with the same stable NBA id. Coalescing on ``(nba_id, season)`` repairs
    that split while retaining the spelling that actually joins the salary
    table for that season.
    """
    if "nba_id" not in impact.columns:
        return impact.copy()

    out = impact.copy()
    out["_nba_id_key"] = out["nba_id"].map(_id_key)
    out["_row_key"] = np.arange(len(out))
    out["_group_key"] = [
        (nba_id, int(season)) if nba_id is not None else ("row", int(row))
        for nba_id, season, row in zip(
            out["_nba_id_key"], out["season"], out["_row_key"]
        )
    ]

    rows = []
    for (_, season), group in out.groupby("_group_key", sort=False):
        if len(group) == 1:
            rows.append(group.iloc[0].copy())
            continue

        preferred = group.iloc[0]
        if salary_keys is not None:
            hits = group[
                group["player_name_norm"].map(
                    lambda name: (str(name), int(season)) in salary_keys
                )
            ]
            names = hits["player_name_norm"].dropna().unique()
            if len(names) > 1:
                raise ValueError(
                    f"NBA id {group['_nba_id_key'].iloc[0]} season {season} "
                    f"matches multiple salary names: {sorted(names)}"
                )
            if len(hits):
                preferred = hits.iloc[0]

        combined = preferred.copy()
        for column in impact.columns:
            if column in {"player_name", "player_name_norm"}:
                continue
            if pd.isna(combined[column]):
                values = group[column].dropna()
                if len(values):
                    combined[column] = values.iloc[0]
        rows.append(combined)

    result = pd.DataFrame(rows).drop(
        columns=["_nba_id_key", "_row_key", "_group_key"]
    )
    return result[impact.columns].reset_index(drop=True)


def apply_impact_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Apply narrow, sourced player-season corrections without filling NaNs."""
    if not CORRECTIONS.exists():
        return df.copy()

    corrections = pd.read_csv(CORRECTIONS)
    keys = ["player_name_norm", "season"]
    if corrections.duplicated(keys).any():
        raise ValueError("impact_metric_corrections.csv has duplicate keys")

    out = df.copy()
    indexed = out.set_index(keys, drop=False)
    missing = []
    for correction in corrections.itertuples(index=False):
        key = (str(correction.player_name_norm), int(correction.season))
        if key not in indexed.index:
            missing.append(key)
            continue
        for column in corrections.columns.difference(CORRECTION_META):
            value = getattr(correction, column)
            if pd.notna(value):
                indexed.loc[key, column] = value

    if missing:
        raise ValueError(f"impact corrections did not match rows: {missing}")
    return indexed.reset_index(drop=True)


def apply_player_identity_corrections(
    df: pd.DataFrame,
    allowed_missing: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Apply sourced identity facts after all source merges.

    ``allowed_missing`` contains salary rows deliberately removed before the
    impact merge, such as retained-only dead money. Other missing correction
    keys remain a hard failure.
    """
    if not IDENTITY_CORRECTIONS.exists():
        return df.copy()
    corrections = pd.read_csv(IDENTITY_CORRECTIONS)
    keys = ["player_name_norm", "season"]
    if corrections.duplicated(keys).any():
        raise ValueError("player_identity_corrections.csv has duplicate keys")

    out = df.copy().set_index(keys, drop=False)
    missing = []
    for correction in corrections.itertuples(index=False):
        key = (str(correction.player_name_norm), int(correction.season))
        if key not in out.index:
            missing.append(key)
            continue
        for column in corrections.columns.difference(CORRECTION_META):
            value = getattr(correction, column)
            if pd.notna(value):
                out.loc[key, column] = value
    allowed = allowed_missing or set()
    unexpected = [key for key in missing if key not in allowed]
    if unexpected:
        raise ValueError(
            f"player identity corrections did not match rows: {unexpected}"
        )
    return out.reset_index(drop=True)


def fill_impact_from_bbref(
    impact: pd.DataFrame,
    stats: pd.DataFrame,
    key_columns: tuple[str, str] = ("player_name_norm", "season"),
) -> pd.DataFrame:
    """Fill missing LAKER workload fields from BBRef regular-season stats.

    The fallback never overwrites nbarapm and never creates a new impact row;
    DARKO remains the population anchor. It supplies only directly comparable
    identity, workload and Basketball Reference rate fields for played seasons.
    LAKER is intentionally not synthesized.
    """
    if stats.empty:
        return impact.copy()
    keys = list(key_columns)
    if stats.duplicated(keys).any():
        dup = stats.loc[stats.duplicated(keys, keep=False), keys].drop_duplicates()
        raise ValueError(
            "BBRef advanced stats have duplicate keys: "
            f"{dup.head().to_dict('records')}"
        )

    available = {src: dst for src, dst in BBREF_FALLBACK_COLUMNS.items()
                 if src in stats.columns}
    fallback_names = {
        source: f"{target}_bbref" for source, target in available.items()
    }
    fallback = stats[keys + list(available)].rename(columns=fallback_names)
    out = impact.merge(fallback, on=keys, how="left", validate="one_to_one")
    if "_bbref_fallback_used" not in out.columns:
        out["_bbref_fallback_used"] = False
    for column in available.values():
        fallback_column = f"{column}_bbref"
        if column not in out.columns:
            out[column] = out[fallback_column]
            filled = out[fallback_column].notna()
        else:
            filled = out[column].isna() & out[fallback_column].notna()
            out[column] = out[column].fillna(out[fallback_column])
        out["_bbref_fallback_used"] |= filled
        out = out.drop(columns=fallback_column)
    return out


def fill_age_from_player_history(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing age from the player's exact season-to-age offset.

    Across the stored LAKER history every observed player has one constant
    ``age - season`` value. That makes an adjacent-season age an identity fact,
    unlike the previous league-wide median fill.
    """
    if "age" not in df.columns:
        return df.copy()

    out = df.copy()
    observed = out.dropna(subset=["age"])
    offsets = observed.assign(_offset=observed["age"] - observed["season"])
    counts = offsets.groupby("player_name_norm")["_offset"].nunique()
    conflicts = counts[counts > 1]
    if len(conflicts):
        raise ValueError(
            "age-season offset changed within player history: "
            + ", ".join(conflicts.index[:10])
        )
    offset_map = offsets.groupby("player_name_norm")["_offset"].first()
    inferred = out["season"] + out["player_name_norm"].map(offset_map)
    out["age"] = out["age"].fillna(inferred)
    return out
