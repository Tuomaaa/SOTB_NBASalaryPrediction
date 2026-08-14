"""compute_mvc.py — Market Value Contributed (MVC).

MVC = predicted_cap_pct × (GP / season_max_GP)

The model's predicted salary is a market-consensus quality measure;
multiplying by availability gives a season-total contribution proxy.
At the team level, Σ MVC ≈ total quality-weighted playing time deployed.

For Year-1 rows (eval frame): uses the champion's fold-honest OOF
predictions from oof_reference.csv.
For non-Year-1 rows (escalator years, rookie-scale, etc.): fits the
base model (MEASUREMENT_FEATURES, 21 features with prev_cap_pct) on
the Year-1 training set and predicts. These rows were never in
training, so predictions are genuinely out-of-sample.

Validation: team-level MVC correlated against actual team wins.

Outputs:
  outputs/mvc/player_mvc.csv  — per player-season
  outputs/mvc/team_mvc.csv    — per team-season with actual wins
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import CAP_BY_SEASON, PROCESSED_DIR
from src.model.train import (
    load_training_data, TARGET,
    _filter_year1, _filter_rookie_scale, _filter_prorated,
    _filter_mislabeled_year1, _filter_continuations,
    _filter_rookie_contracts, _compute_max_eligible,
)
from src.features.kf_market_value import MEASUREMENT_FEATURES
from src.model.stages import compose, training_route_frame, deployed_p_max

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "mvc"
OOF_PATH = Path(__file__).resolve().parent.parent / "outputs" / "models" / "oof_reference.csv"

MAX_GP = {s: 72 if s == 2021 else 82 for s in range(2015, 2028)}
MULTI_TEAM = {"2TM", "3TM", "TOT", ""}


def load_data():
    oof = pd.read_csv(OOF_PATH)
    oof["season"] = oof["season"].astype(int)

    wins_path = PROCESSED_DIR / "team_wins.csv"
    wins = pd.read_csv(wins_path) if wins_path.exists() else pd.DataFrame()
    if len(wins):
        wins["season"] = wins["season"].astype(int)

    return oof, wins


def fit_base_model(df_full):
    """Fit base model on Year-1 filtered data, return model + medians."""
    df_train = _filter_year1(df_full.copy())
    df_train = _filter_rookie_scale(df_train)
    df_train = _filter_prorated(df_train)
    df_train = _filter_mislabeled_year1(df_train)
    df_train = _filter_continuations(df_train)
    df_train = _filter_rookie_contracts(df_train)

    features = MEASUREMENT_FEATURES
    medians = df_train[features].median()

    X = df_train[features].fillna(medians).fillna(0)
    y = df_train[TARGET].values

    model = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
        random_state=42, tree_method="hist",
    )
    model.fit(X, y)
    print(f"  Base model fit on {len(X)} Year-1 rows, "
          f"{len(features)} features")

    return model, features, medians, df_train


def predict_all(df_full, model, features, medians, df_train):
    """Predict on all rows, apply Stage 2 compose."""
    if "max_eligible_pct" not in df_full.columns:
        df_full = _compute_max_eligible(df_full.copy())

    X_all = df_full[features].fillna(medians).fillna(0)
    latent = model.predict(X_all)

    lo = df_full["floor_pct"].values if "floor_pct" in df_full.columns else np.zeros(len(df_full))
    hi = df_full["max_eligible_pct"].values if "max_eligible_pct" in df_full.columns else np.ones(len(df_full))

    try:
        route_df = training_route_frame(df_train)
        p_max = deployed_p_max(route_df, df_full, medians=medians)
    except Exception:
        p_max = None

    is_ext = (df_full["is_extension"].values
              if "is_extension" in df_full.columns
              else np.zeros(len(df_full), dtype=bool))
    ext_cap = (df_full["ext_cap_pct"].values
               if "ext_cap_pct" in df_full.columns
               else np.full(len(df_full), np.nan))

    predicted = compose(
        latent, lo=lo, hi=hi, p_max=p_max,
        is_extension=is_ext, ext_cap_pct=ext_cap,
    )

    return predicted


def compute_player_mvc(df_full, oof, predicted_all):
    """Compute per-player MVC with model predictions for all rows."""
    df = df_full.copy()
    df["predicted_cap_pct"] = predicted_all

    oof_map = dict(zip(
        zip(oof["player_name_norm"], oof["season"]),
        oof["oof_champion"]
    ))
    keys = list(zip(df["player_name_norm"], df["season"].astype(int)))
    oof_vals = pd.Series([oof_map.get(k) for k in keys], index=df.index)
    has_oof = oof_vals.notna()
    df.loc[has_oof, "predicted_cap_pct"] = oof_vals[has_oof]
    df["has_oof_prediction"] = has_oof

    df["max_gp"] = df["season"].map(MAX_GP)
    df["gp_ratio"] = (df["games"].fillna(0) / df["max_gp"]).clip(0, 1)

    df["actual_mvc"] = df[TARGET] * df["gp_ratio"]
    df["predicted_mvc"] = df["predicted_cap_pct"] * df["gp_ratio"]
    df["surplus_mvc"] = df["predicted_mvc"] - df["actual_mvc"]

    cap = df["season"].map(CAP_BY_SEASON)
    df["actual_mvc_M"] = df["actual_mvc"] * cap / 1e6
    df["predicted_mvc_M"] = df["predicted_mvc"] * cap / 1e6
    df["surplus_mvc_M"] = df["surplus_mvc"] * cap / 1e6

    n_model = (~has_oof).sum()
    print(f"  Year-1 OOF predictions: {has_oof.sum()}")
    print(f"  Base-model predictions: {n_model}")

    return df


def compute_team_mvc(player_df, wins):
    team_players = player_df[
        ~player_df["team_abbreviation"].isin(MULTI_TEAM)
        & player_df["team_abbreviation"].notna()
    ].copy()

    team_agg = team_players.groupby(["team_abbreviation", "season"]).agg(
        n_players=("player_name_norm", "nunique"),
        total_actual_mvc=("actual_mvc", "sum"),
        total_predicted_mvc=("predicted_mvc", "sum"),
        total_surplus_mvc=("surplus_mvc", "sum"),
        total_games=("games", "sum"),
        total_salary_pct=("cap_pct", "sum"),
    ).reset_index()

    if len(wins):
        team_agg = team_agg.merge(
            wins[["team", "season", "wins", "losses", "win_pct"]].rename(
                columns={"team": "team_abbreviation"}
            ),
            on=["team_abbreviation", "season"],
            how="left",
        )

    cap = team_agg["season"].map(CAP_BY_SEASON)
    team_agg["total_actual_mvc_M"] = team_agg["total_actual_mvc"] * cap / 1e6
    team_agg["total_predicted_mvc_M"] = team_agg["total_predicted_mvc"] * cap / 1e6
    team_agg["total_surplus_mvc_M"] = team_agg["total_surplus_mvc"] * cap / 1e6

    return team_agg


def validate(team_df):
    from scipy import stats

    if "wins" not in team_df.columns:
        print("\n  No team_wins.csv found, skipping validation.")
        return

    valid = team_df.dropna(subset=["wins", "total_actual_mvc"])
    if len(valid) < 10:
        print(f"\n  Only {len(valid)} team-seasons with wins data, skipping.")
        return

    metrics = [
        ("actual_mvc", "total_actual_mvc"),
        ("predicted_mvc", "total_predicted_mvc"),
        ("surplus_mvc", "total_surplus_mvc"),
    ]
    print(f"\n{'':2s}{'Metric':20s} {'r':>8s} {'p':>12s}  (n={len(valid)})")
    print(f"  {'-'*50}")
    for label, col in metrics:
        r, p = stats.pearsonr(valid[col], valid["wins"])
        print(f"  {label:20s} {r:+.3f}    {p:.2e}")

    print(f"\n  Per-season MVC ↔ Wins:")
    for s in sorted(valid["season"].unique()):
        g = valid[valid["season"] == s]
        if len(g) >= 10:
            r_a, _ = stats.pearsonr(g["total_actual_mvc"], g["wins"])
            r_p, _ = stats.pearsonr(g["total_predicted_mvc"], g["wins"])
            print(f"    {s}: actual r={r_a:+.3f}  predicted r={r_p:+.3f}  (n={len(g)})")


def print_leaderboard(player_df):
    latest = int(player_df["season"].max())
    cur = player_df[player_df["season"] == latest].copy()

    print(f"\n{'='*65}")
    print(f"  {latest} MVC Leaders (predicted quality × availability)")
    print(f"{'='*65}")
    top = cur.nlargest(15, "predicted_mvc_M")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        rate = r["predicted_mvc_M"] / r["gp_ratio"] if r["gp_ratio"] > 0 else 0
        src = "OOF" if r["has_oof_prediction"] else "base"
        print(f"  {i:2d}. {r['player_name']:22s}  "
              f"MVC ${r['predicted_mvc_M']:5.1f}M  "
              f"({int(r['games']):2d} GP, "
              f"pred ${rate:.1f}M) [{src}]")

    print(f"\n{'='*65}")
    print(f"  {latest} Best Surplus (most value above contract)")
    print(f"{'='*65}")
    top_s = cur.nlargest(10, "surplus_mvc_M")
    for i, (_, r) in enumerate(top_s.iterrows(), 1):
        print(f"  {i:2d}. {r['player_name']:22s}  "
              f"surplus ${r['surplus_mvc_M']:+5.1f}M  "
              f"(pred ${r['predicted_mvc_M']:.1f}M vs "
              f"actual ${r['actual_mvc_M']:.1f}M)")

    print(f"\n{'='*65}")
    print(f"  {latest} Worst Surplus (most overpaid × availability)")
    print(f"{'='*65}")
    bot_s = cur.nsmallest(10, "surplus_mvc_M")
    for i, (_, r) in enumerate(bot_s.iterrows(), 1):
        print(f"  {i:2d}. {r['player_name']:22s}  "
              f"surplus ${r['surplus_mvc_M']:+5.1f}M  "
              f"(pred ${r['predicted_mvc_M']:.1f}M vs "
              f"actual ${r['actual_mvc_M']:.1f}M)")


def print_team_rankings(team_df):
    latest = int(team_df["season"].max())
    cur = team_df[team_df["season"] == latest].copy()
    has_wins = "wins" in cur.columns and cur["wins"].notna().any()

    print(f"\n{'='*65}")
    print(f"  {latest} Team MVC Rankings")
    print(f"{'='*65}")
    cur_sorted = cur.sort_values("total_predicted_mvc_M", ascending=False)
    for i, (_, r) in enumerate(cur_sorted.iterrows(), 1):
        wins_str = f"{int(r['wins']):2d}W" if has_wins and pd.notna(r.get("wins")) else ""
        print(f"  {i:2d}. {r['team_abbreviation']:4s}  "
              f"MVC ${r['total_predicted_mvc_M']:5.1f}M  "
              f"surplus ${r['total_surplus_mvc_M']:+5.1f}M  "
              f"({int(r['n_players']):2d} players)  {wins_str}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("Loading data...")
    oof, wins = load_data()
    print(f"  OOF predictions:  {len(oof):,d} rows")
    if len(wins):
        print(f"  Team wins:        {len(wins):,d} team-seasons "
              f"({int(wins['season'].min())}-{int(wins['season'].max())})")

    print("\nLoading full training frame...")
    df_full = load_training_data()
    print(f"  {len(df_full):,d} rows, "
          f"seasons {int(df_full['season'].min())}-{int(df_full['season'].max())}")

    print("\nPre-computing max eligible...")
    df_full = _compute_max_eligible(df_full)
    gc.collect()

    print("\nFitting base model on Year-1 data...")
    model, features, medians, df_train = fit_base_model(df_full)

    print("\nPredicting on ALL rows...")
    predicted_all = predict_all(df_full, model, features, medians, df_train)

    print("\nComputing player MVC...")
    player_df = compute_player_mvc(df_full, oof, predicted_all)

    print("\nComputing team MVC...")
    team_df = compute_team_mvc(player_df, wins)
    print(f"  {len(team_df)} team-seasons")

    print("\n=== Team MVC vs Actual Wins ===")
    validate(team_df)

    print_leaderboard(player_df)
    print_team_rankings(team_df)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    player_cols = [
        "player_name", "player_name_norm", "team_abbreviation", "season",
        "games", "cap_pct", "predicted_cap_pct", "has_oof_prediction",
        "gp_ratio", "actual_mvc", "predicted_mvc", "surplus_mvc",
        "actual_mvc_M", "predicted_mvc_M", "surplus_mvc_M",
    ]
    player_df[player_cols].to_csv(
        OUTPUTS_DIR / "player_mvc.csv", index=False, float_format="%.6f"
    )
    team_df.to_csv(
        OUTPUTS_DIR / "team_mvc.csv", index=False, float_format="%.6f"
    )
    print(f"\nSaved to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
