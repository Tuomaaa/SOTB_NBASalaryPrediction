"""Train valuation models: Ridge baseline and XGBoost.

Target: cap_pct (salary as fraction of salary cap).
Training data filtered to year-1 contracts only (year 2+ are CBA escalators).
Rookie-scale contracts (1st-round picks, years 2-4) removed via draft data.
Uses 5-fold GroupKFold CV (same player stays in same fold).
"""

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import PROCESSED_DIR, OUTPUTS_DIR

FEATURE_COLS = [
    "darko_dpm_z", "lebron_z", "rapm_z",
    "age", "age_squared",
    "mpg",
    "availability_3yr",
    "usage_pct",
    "height_inches",
    "cba_era",
    "ast_pct",
]

TARGET = "cap_pct"


def load_training_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "training_data.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET])
    return df


def _filter_year1(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only year-1 contract rows.

    Years 2+ of multi-year contracts are CBA-mandated escalators (5% or 8%
    raises), not fresh market evaluations. Unmatched rows (year_in_contract
    is NaN) are kept as they couldn't be matched to contract structure.
    """
    if "year_in_contract" not in df.columns:
        print("WARNING: year_in_contract not in data, skipping year-1 filter")
        return df
    mask = (df["year_in_contract"] == 1) | (df["year_in_contract"].isna())
    filtered = df[mask].copy()
    dropped = len(df) - len(filtered)
    print(f"Year-1 filter: dropped {dropped} year-2+ rows ({len(filtered)} remain)")
    return filtered


def _load_rookie_scale_set() -> set[tuple[str, int]]:
    """Load (player_name_norm, season) pairs for 1st-round rookie-scale rows.

    Uses draft_data.csv (1st-round picks 2015-2025). Rookie scale appears in
    training data at seasons draft_year+1 through draft_year+3 (years 2-4 of
    the deal; year 1 drops out due to the stats-salary lag).
    """
    path = PROCESSED_DIR / "draft_data.csv"
    if not path.exists():
        return set()
    draft = pd.read_csv(path)
    pairs = set()
    for _, r in draft.iterrows():
        for offset in range(1, 4):
            pairs.add((r["player_name_norm"], int(r["draft_year"]) + offset))
    return pairs


def _filter_rookie_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 1st-round rookie-scale contracts using draft data."""
    rs_set = _load_rookie_scale_set()
    if not rs_set:
        print("WARNING: draft_data.csv not found, falling back to age heuristic")
        mask = (df["age"] <= 23) & (df["cap_pct"] <= 0.10)
    else:
        mask = df.apply(
            lambda r: (r["player_name_norm"], r["season"]) in rs_set, axis=1
        )
    filtered = df[~mask].copy()
    print(f"Rookie filter: dropped {mask.sum()} rows ({len(filtered)} remain)")
    return filtered


def _prepare_Xy(df: pd.DataFrame, features: list[str] | None = None):
    """Return X, y, groups arrays with NaN features filled."""
    if features is None:
        features = FEATURE_COLS
    avail = [f for f in features if f in df.columns]
    X = df[avail].copy()

    for col in list(avail):
        if X[col].dropna().nunique() <= 1:
            X = X.drop(columns=[col])
            avail.remove(col)

    X = X.fillna(X.median()).fillna(0)
    y = df[TARGET].values
    groups = df["player_name_norm"].values
    return X, y, groups, avail


def train_ridge(df: pd.DataFrame, alpha: float = 1.0) -> tuple[dict, object]:
    """Train Ridge regression with year-1 filter + rookie filter."""
    df = _filter_year1(df)
    df = _filter_rookie_scale(df)
    X, y, groups, features = _prepare_Xy(df)
    print(f"Training Ridge (alpha={alpha}) on {len(X)} samples, {len(features)} features")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    cv = GroupKFold(n_splits=5)
    scores = cross_validate(
        pipeline, X, y, groups=groups, cv=cv,
        scoring=["r2", "neg_mean_absolute_error"],
        return_train_score=True,
    )

    results = {
        "model": "Ridge",
        "alpha": alpha,
        "n_samples": len(X),
        "n_features": len(features),
        "features": features,
        "cv_r2_mean": float(np.mean(scores["test_r2"])),
        "cv_r2_std": float(np.std(scores["test_r2"])),
        "cv_mae_mean": float(-np.mean(scores["test_neg_mean_absolute_error"])),
        "cv_mae_std": float(np.std(scores["test_neg_mean_absolute_error"])),
        "train_r2_mean": float(np.mean(scores["train_r2"])),
    }

    pipeline.fit(X, y)
    ridge_model = pipeline.named_steps["ridge"]
    coefs = dict(zip(features, ridge_model.coef_))
    results["coefficients"] = {k: round(v, 6) for k, v in coefs.items()}
    results["intercept"] = round(float(ridge_model.intercept_), 6)

    return results, pipeline


def train_xgboost(df: pd.DataFrame) -> tuple[dict, object, list[str]]:
    """Train XGBoost with year-1 and rookie-scale filters."""
    from xgboost import XGBRegressor

    df = _filter_year1(df)
    df = _filter_rookie_scale(df)
    X, y, groups, features = _prepare_Xy(df)
    print(f"Training XGBoost on {len(X)} samples, {len(features)} features")

    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        tree_method="hist",
    )

    cv = GroupKFold(n_splits=5)
    scores = cross_validate(
        xgb, X, y, groups=groups, cv=cv,
        scoring=["r2", "neg_mean_absolute_error"],
        return_train_score=True,
    )

    xgb.fit(X, y)

    results = {
        "model": "XGBoost",
        "n_samples": len(X),
        "n_features": len(features),
        "features": features,
        "cv_r2_mean": float(np.mean(scores["test_r2"])),
        "cv_r2_std": float(np.std(scores["test_r2"])),
        "cv_mae_mean": float(-np.mean(scores["test_neg_mean_absolute_error"])),
        "cv_mae_std": float(np.std(scores["test_neg_mean_absolute_error"])),
        "train_r2_mean": float(np.mean(scores["train_r2"])),
    }

    importances = dict(zip(features, xgb.feature_importances_))
    results["feature_importances"] = {k: round(float(v), 4) for k, v in
                                       sorted(importances.items(), key=lambda x: -x[1])}

    return results, xgb, features


def print_results(results: dict):
    print(f"\n{'='*50}")
    print(f"  {results['model']} Results")
    print(f"{'='*50}")
    print(f"  CV R²:  {results['cv_r2_mean']:.4f} ± {results.get('cv_r2_std', 0):.4f}")
    print(f"  CV MAE: {results['cv_mae_mean']:.4f} ± {results.get('cv_mae_std', 0):.4f}")
    if "train_r2_mean" in results:
        print(f"  Train R²: {results['train_r2_mean']:.4f}")
    print(f"  Samples: {results.get('n_samples')}, Features: {results.get('n_features')}")
    if "coefficients" in results:
        print(f"\n  Coefficients:")
        for feat, coef in sorted(results["coefficients"].items(), key=lambda x: -abs(x[1])):
            print(f"    {feat:25s} {coef:+.6f}")
    if "feature_importances" in results:
        print(f"\n  Feature importances:")
        for feat, imp in results["feature_importances"].items():
            print(f"    {feat:25s} {imp:.4f}")


if __name__ == "__main__":
    df = load_training_data()
    print(f"Loaded {len(df)} rows")

    ridge_results, ridge_model = train_ridge(df)
    print_results(ridge_results)

    xgb_results, xgb_model, xgb_features = train_xgboost(df)
    print_results(xgb_results)

    model_dir = OUTPUTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "ridge_results.json", "w") as f:
        json.dump(ridge_results, f, indent=2)
    with open(model_dir / "xgb_results.json", "w") as f:
        json.dump(xgb_results, f, indent=2)
    print(f"\nResults saved to {model_dir}")
