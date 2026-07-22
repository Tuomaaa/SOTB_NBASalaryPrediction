"""Hyperparameter sweep for XGBoost on the pruned 13-feature set."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import r2_score
from itertools import product
from pathlib import Path

from src.model.train import (
    load_training_data, _filter_year1, _filter_rookie_scale, TARGET,
)

FEATURES = [
    "darko_dpm_z", "lebron_z", "rapm_z",
    "age", "age_squared",
    "mpg",
    "availability_3yr",
    "usage_pct",
    "height_inches",
    "cba_era",
    "ast_pct",
    "award_score_cum",
    "draft_pick",
]

SEEDS = list(range(42, 52))
RESULTS_FILE = Path(__file__).resolve().parent.parent / "outputs" / "sweep_results.csv"


def prepare(df):
    X = df[FEATURES].copy()
    X = X.fillna(X.median()).fillna(0)
    y = df[TARGET].values
    groups = df["player_name_norm"].values
    return X, y, groups


def evaluate(df_train, df_holdout, params):
    cv_scores, ho_scores = [], []
    for seed in SEEDS:
        xgb = XGBRegressor(random_state=seed, tree_method="hist", **params)
        X_tr, y_tr, groups = prepare(df_train)
        cv = GroupKFold(n_splits=5)
        scores = cross_validate(xgb, X_tr, y_tr, groups=groups, cv=cv, scoring="r2")
        cv_scores.append(np.mean(scores["test_score"]))

        xgb.fit(X_tr, y_tr)
        X_ho, y_ho, _ = prepare(df_holdout)
        ho_scores.append(r2_score(y_ho, xgb.predict(X_ho)))

    return {
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "ho_mean": np.mean(ho_scores),
        "ho_std": np.std(ho_scores),
    }


def append_result(row):
    header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        if header:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


def main():
    df = load_training_data()
    df = _filter_year1(df)
    df = _filter_rookie_scale(df)

    df_train = df[df["season"] < 2026].copy()
    df_holdout = df[df["season"] == 2026].copy()
    print(f"Train: {len(df_train)}, Holdout: {len(df_holdout)}", flush=True)

    # Clear old results
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_weight": [3, 5, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    # Phase 1: depth × n_est × lr
    combos_p1 = list(product(grid["max_depth"], grid["n_estimators"], grid["learning_rate"]))
    total_p1 = len(combos_p1)
    print(f"\nPhase 1: {total_p1} combos (depth × n_est × lr)", flush=True)

    for i, (depth, n_est, lr) in enumerate(combos_p1, 1):
        params = dict(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        )
        r = evaluate(df_train, df_holdout, params)
        row = {"phase": 1, **params, **r}
        append_result(row)
        print(f"[P1 {i:>3d}/{total_p1}] d={depth} n={n_est:>3d} lr={lr:.2f}  "
              f"CV={r['cv_mean']:.4f}  HO={r['ho_mean']:.4f}", flush=True)

    # Find best P1 by CV
    p1_df = pd.read_csv(RESULTS_FILE)
    p1_df = p1_df[p1_df["phase"] == 1]
    best_p1 = p1_df.loc[p1_df["cv_mean"].idxmax()]
    print(f"\n>>> Best P1: depth={int(best_p1['max_depth'])}, n_est={int(best_p1['n_estimators'])}, "
          f"lr={best_p1['learning_rate']}  CV={best_p1['cv_mean']:.4f}  HO={best_p1['ho_mean']:.4f}",
          flush=True)

    # Phase 2: mcw × sub × col around best P1
    combos_p2 = list(product(grid["min_child_weight"], grid["subsample"], grid["colsample_bytree"]))
    total_p2 = len(combos_p2)
    print(f"\nPhase 2: {total_p2} combos (mcw × sub × col)", flush=True)

    for i, (mcw, sub, col) in enumerate(combos_p2, 1):
        params = dict(
            n_estimators=int(best_p1["n_estimators"]),
            max_depth=int(best_p1["max_depth"]),
            learning_rate=best_p1["learning_rate"],
            subsample=sub, colsample_bytree=col, min_child_weight=mcw,
        )
        r = evaluate(df_train, df_holdout, params)
        row = {"phase": 2, **params, **r}
        append_result(row)
        print(f"[P2 {i:>3d}/{total_p2}] mcw={mcw:>2d} sub={sub:.1f} col={col:.1f}  "
              f"CV={r['cv_mean']:.4f}  HO={r['ho_mean']:.4f}", flush=True)

    # Final summary
    all_df = pd.read_csv(RESULTS_FILE)
    best = all_df.loc[all_df["cv_mean"].idxmax()]
    print(f"\n{'='*60}")
    print(f"BEST OVERALL (by CV R²):")
    print(f"  n_estimators:     {int(best['n_estimators'])}")
    print(f"  max_depth:        {int(best['max_depth'])}")
    print(f"  learning_rate:    {best['learning_rate']}")
    print(f"  min_child_weight: {int(best['min_child_weight'])}")
    print(f"  subsample:        {best['subsample']}")
    print(f"  colsample_bytree: {best['colsample_bytree']}")
    print(f"  CV R²:  {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    print(f"  HO R²:  {best['ho_mean']:.4f} ± {best['ho_std']:.4f}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
