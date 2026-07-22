"""Ablation study: drop each new Phase-2 feature (group) and measure impact."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, cross_validate

from config import PROCESSED_DIR
from src.model.train import (
    load_training_data, _filter_year1, _filter_rookie_scale,
    _add_agent_features, TARGET,
)

FULL_FEATURES = [
    "darko_dpm_z", "lebron_z", "rapm_z",
    "age", "age_squared",
    "mpg",
    "availability_3yr",
    "usage_pct",
    "height_inches",
    "cba_era",
    "ast_pct",
    "agent_avg_cap", "agent_median_cap", "agent_max_cap", "agent_std_cap",
    "award_score_cum", "all_nba_cum",
    "draft_pick",
    "team_value_B",
    "injury_reports",
]

V1_FEATURES = [
    "darko_dpm_z", "lebron_z", "rapm_z",
    "age", "age_squared",
    "mpg",
    "availability_3yr",
    "usage_pct",
    "height_inches",
    "cba_era",
    "ast_pct",
]

ABLATIONS = {
    "full (20 feat)":        FULL_FEATURES,
    "v1 baseline (11 feat)": V1_FEATURES,
    "- agent (all 4)":       [f for f in FULL_FEATURES if not f.startswith("agent_")],
    "- agent_avg_cap":       [f for f in FULL_FEATURES if f != "agent_avg_cap"],
    "- agent_median_cap":    [f for f in FULL_FEATURES if f != "agent_median_cap"],
    "- agent_max_cap":       [f for f in FULL_FEATURES if f != "agent_max_cap"],
    "- agent_std_cap":       [f for f in FULL_FEATURES if f != "agent_std_cap"],
    "- award_score_cum":     [f for f in FULL_FEATURES if f != "award_score_cum"],
    "- all_nba_cum":         [f for f in FULL_FEATURES if f != "all_nba_cum"],
    "- awards (both)":       [f for f in FULL_FEATURES if f not in ("award_score_cum", "all_nba_cum")],
    "- draft_pick":          [f for f in FULL_FEATURES if f != "draft_pick"],
    "- team_value_B":        [f for f in FULL_FEATURES if f != "team_value_B"],
    "- injury_reports":      [f for f in FULL_FEATURES if f != "injury_reports"],
}


def prepare(df, features):
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


def run_ablation(df_train, df_holdout, features, seeds=(42,)):
    cv_r2s, ho_r2s = [], []
    for seed in seeds:
        xgb = XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=seed, tree_method="hist",
        )
        X_tr, y_tr, groups, used = prepare(df_train, features)
        cv = GroupKFold(n_splits=5)
        scores = cross_validate(xgb, X_tr, y_tr, groups=groups, cv=cv, scoring="r2")
        cv_r2s.append(np.mean(scores["test_score"]))

        xgb.fit(X_tr, y_tr)
        X_ho = df_holdout[[f for f in used if f in df_holdout.columns]].copy()
        X_ho = X_ho.fillna(X_ho.median()).fillna(0)
        y_ho = df_holdout[TARGET].values
        from sklearn.metrics import r2_score
        ho_r2s.append(r2_score(y_ho, xgb.predict(X_ho)))

    return np.mean(cv_r2s), np.mean(ho_r2s)


def main():
    df = load_training_data()
    df = _filter_year1(df)
    df = _filter_rookie_scale(df)
    df = _add_agent_features(df)

    df_train = df[df["season"] < 2026].copy()
    df_holdout = df[df["season"] == 2026].copy()
    print(f"Train: {len(df_train)} rows, Holdout: {len(df_holdout)} rows\n")

    seeds = list(range(42, 52))  # 10 seeds

    print(f"{'Config':<25s}  {'#feat':>5s}  {'CV R²':>8s}  {'HO R²':>8s}  {'ΔCV':>7s}  {'ΔHO':>7s}")
    print("-" * 72)

    baseline_cv, baseline_ho = None, None
    for name, feats in ABLATIONS.items():
        cv_r2, ho_r2 = run_ablation(df_train, df_holdout, feats, seeds)
        n = len([f for f in feats if f in df_train.columns])
        if baseline_cv is None:
            baseline_cv, baseline_ho = cv_r2, ho_r2
            delta_cv, delta_ho = "", ""
        else:
            delta_cv = f"{cv_r2 - baseline_cv:+.4f}"
            delta_ho = f"{ho_r2 - baseline_ho:+.4f}"
        print(f"{name:<25s}  {n:>5d}  {cv_r2:.4f}  {ho_r2:.4f}  {delta_cv:>7s}  {delta_ho:>7s}")


if __name__ == "__main__":
    main()
