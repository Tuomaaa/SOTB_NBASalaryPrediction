"""Audit feature missingness before production median fill.

    python scripts/audit_feature_missingness.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import OUTPUTS_DIR, PROCESSED_DIR
from src.model.train import (
    FEATURE_COLS, _compute_max_eligible, _filter_continuations,
    _filter_mislabeled_year1, _filter_prorated, _filter_rookie_scale,
    _filter_year1, load_training_data,
)

CORE = [
    "darko_dpm_z", "lebron_z", "laker_z", "age", "mpg",
    "availability_3yr", "usage_pct", "height_inches", "ast_pct",
    "prev_cap_pct",
]


def evaluation_frame() -> pd.DataFrame:
    """Production Year-1 frame immediately before feature imputation."""
    df = _filter_rookie_scale(_filter_year1(load_training_data()))
    df = _filter_prorated(df)
    df = _compute_max_eligible(df)
    df = _filter_mislabeled_year1(df)
    return _filter_continuations(df).reset_index(drop=True)


def classify(row: pd.Series) -> str:
    """Assign the most actionable missingness class to one row."""
    if not row["missing_fields"]:
        return "complete"
    missing = set(row["missing_fields"].split("|"))
    if missing & {"age", "height_inches"}:
        return "identity"
    if pd.notna(row.get("games")) and float(row["games"]) == 0:
        return "did_not_play"
    if {"mpg", "usage_pct", "ast_pct"}.issubset(missing):
        return "laker_block"
    if "availability_3yr" in missing and pd.notna(row.get("games")):
        return "availability_propagated"
    if missing & {"laker_z", "lebron_z", "usage_pct", "ast_pct"}:
        return "partial_impact"
    return "other"


def audit_rows(df: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    """Return row-level missingness details for one frame."""
    features = [feature for feature in CORE if feature in df.columns]
    out = df.copy()
    out["missing_fields"] = out[features].apply(
        lambda row: "|".join(row.index[row.isna()]), axis=1
    )
    out["missing_count"] = out[features].isna().sum(axis=1)
    out["frame"] = frame_name
    out["missingness_class"] = out.apply(classify, axis=1)
    keep = [
        "frame", "player_name_norm", "season", "salary", "year_in_contract",
        "games", "availability_3yr_coverage", "missing_count",
        "missing_fields", "missingness_class",
    ]
    return out[[column for column in keep if column in out.columns]]


def summarize(df: pd.DataFrame) -> dict:
    """Return JSON-safe missing counts for one frame."""
    features = [feature for feature in FEATURE_COLS if feature in df.columns]
    return {
        "rows": int(len(df)),
        "players": int(df["player_name_norm"].nunique()),
        "missing_by_feature": {
            feature: int(df[feature].isna().sum()) for feature in features
        },
        "missing_by_season": {
            str(int(season)): {
                feature: int(group[feature].isna().sum()) for feature in features
            }
            for season, group in df.groupby("season")
        },
    }


def main() -> None:
    full = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    evaluation = evaluation_frame()
    rows = pd.concat([
        audit_rows(full, "full"), audit_rows(evaluation, "evaluation")
    ], ignore_index=True)
    out_dir = OUTPUTS_DIR / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "feature_missingness_rows.csv"
    summary_path = out_dir / "feature_missingness_summary.json"
    rows.to_csv(rows_path, index=False)
    payload = {
        "full": summarize(full), "evaluation": summarize(evaluation),
        "classes": {
            frame: {str(k): int(v) for k, v in group["missingness_class"].value_counts().items()}
            for frame, group in rows.groupby("frame")
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for frame, group in rows.groupby("frame"):
        print(f"{frame}: {len(group)} rows, {(group['missing_count'] > 0).sum()} affected")
        print(group["missingness_class"].value_counts().to_string())
    print(f"Saved {rows_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
