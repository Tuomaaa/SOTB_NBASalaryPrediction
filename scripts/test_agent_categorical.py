"""Test agent ID as native XGBoost categorical feature."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import r2_score

from config import PROCESSED_DIR
from src.model.train import (
    load_training_data, _filter_year1, _filter_rookie_scale, TARGET,
)

BASE_FEATURES = [
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


def bucket_agents(df, min_clients=5):
    """Bucket rare agents into OTHER, missing into UNKNOWN."""
    col = df["agent"].copy()
    agent_counts = col.value_counts()
    rare = agent_counts[agent_counts < min_clients].index
    col = col.fillna("UNKNOWN")
    col[col.isin(rare)] = "OTHER"
    return col


def align_categories(train_col, test_col):
    """Ensure test uses same category set as train; unseen -> OTHER."""
    train_cats = set(train_col.unique())
    test_col = test_col.copy()
    unseen = ~test_col.isin(train_cats)
    if unseen.any():
        test_col[unseen] = "OTHER"
    cat_type = pd.CategoricalDtype(categories=sorted(train_cats))
    return train_col.astype(cat_type), test_col.astype(cat_type)


def run_test(df_all, features, use_agent=False, agent_min=5, seeds=range(42, 52),
             depth=3):
    cv_scores, ho_scores = [], []

    df_train = df_all[df_all["season"] < 2026].copy()
    df_holdout = df_all[df_all["season"] == 2026].copy()

    for seed in seeds:
        feats = list(features)
        enable_cat = False

        if use_agent:
            df_train["agent_cat"] = bucket_agents(df_train, agent_min)
            df_holdout["agent_cat"] = bucket_agents(df_holdout, agent_min)
            tr_cat, ho_cat = align_categories(df_train["agent_cat"], df_holdout["agent_cat"])
            df_train["agent_cat"] = tr_cat
            df_holdout["agent_cat"] = ho_cat
            feats = feats + ["agent_cat"]
            enable_cat = True

        X_tr = df_train[feats].copy()
        X_ho = df_holdout[feats].copy()

        for col in feats:
            if col != "agent_cat":
                med = X_tr[col].median()
                X_tr[col] = X_tr[col].fillna(med).fillna(0)
                X_ho[col] = X_ho[col].fillna(med).fillna(0)

        y_tr = df_train[TARGET].values
        y_ho = df_holdout[TARGET].values
        groups = df_train["player_name_norm"].values

        xgb = XGBRegressor(
            n_estimators=100, max_depth=depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=seed, tree_method="hist",
            enable_categorical=enable_cat,
        )

        cv = GroupKFold(n_splits=5)
        scores = cross_validate(xgb, X_tr, y_tr, groups=groups, cv=cv, scoring="r2")
        cv_scores.append(np.mean(scores["test_score"]))

        xgb.fit(X_tr, y_tr)
        ho_scores.append(r2_score(y_ho, xgb.predict(X_ho)))

    return np.mean(cv_scores), np.std(cv_scores), np.mean(ho_scores), np.std(ho_scores)


def main():
    df = load_training_data()
    df = _filter_year1(df)
    df = _filter_rookie_scale(df)
    print(f"Total: {len(df)} rows, {df['agent'].notna().sum()} with agent data")
    print(f"Unique agents: {df['agent'].nunique()}\n")

    seeds = range(42, 52)

    print(f"{'Config':<45s}  {'CV R²':>12s}  {'HO R²':>12s}")
    print("-" * 75)

    # Baseline
    cv, cvs, ho, hos = run_test(df, BASE_FEATURES, use_agent=False, seeds=seeds)
    print(f"{'baseline (13 feat)':<45s}  {cv:.4f}±{cvs:.4f}  {ho:.4f}±{hos:.4f}")

    # Agent categorical with different bucketing thresholds
    for min_n in [3, 5, 10, 15, 20]:
        n_cats = len(bucket_agents(df, min_n).unique())
        name = f"+ agent_cat (min_clients={min_n}, {n_cats} cats)"
        cv, cvs, ho, hos = run_test(df, BASE_FEATURES, use_agent=True,
                                     agent_min=min_n, seeds=seeds)
        print(f"{name:<45s}  {cv:.4f}±{cvs:.4f}  {ho:.4f}±{hos:.4f}")

    # Best bucketing + higher depth
    print(f"\n--- Depth sensitivity (with agent_cat min=10) ---\n")
    for depth in [3, 4, 5, 6]:
        name_base = f"depth={depth} baseline"
        cv, cvs, ho, hos = run_test(df, BASE_FEATURES, use_agent=False,
                                     seeds=seeds, depth=depth)
        print(f"{name_base:<45s}  {cv:.4f}±{cvs:.4f}  {ho:.4f}±{hos:.4f}")

        name_agent = f"depth={depth} + agent_cat"
        cv, cvs, ho, hos = run_test(df, BASE_FEATURES, use_agent=True,
                                     agent_min=10, seeds=seeds, depth=depth)
        print(f"{name_agent:<45s}  {cv:.4f}±{cvs:.4f}  {ho:.4f}±{hos:.4f}")


if __name__ == "__main__":
    main()
