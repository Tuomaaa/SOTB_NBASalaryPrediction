"""Generate predictions for a target season using a trained XGBoost model.

Loads impact metrics for the target season, engineers features, runs inference,
identifies free agents, and computes salary diffs.
"""

import unicodedata

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, OUTPUTS_DIR, CAP_BY_SEASON
from src.features.base_rating import add_base_rating
from src.features.age_curve import add_age_features
from src.features.availability import compute_availability
from src.features.cba_constraints import add_cba_features
from src.model.train import load_training_data, train_xgboost, FEATURE_COLS


def _normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def predict(target_season: int = 2026) -> pd.DataFrame:
    """Train model and generate predictions for target_season."""
    # --- Train ---
    df = load_training_data()
    print(f"Training data: {len(df)} rows")
    xgb_results, model, features = train_xgboost(df)
    print(f"XGBoost CV R²: {xgb_results['cv_r2_mean']:.4f}")

    # --- Load prediction data ---
    impact = pd.read_csv(PROCESSED_DIR / "impact_metrics.csv")
    pred_df = impact[
        (impact["season"] == target_season)
        & (impact["age"].notna())
        & (impact["minutes"].notna())
    ].copy()
    print(f"\n{target_season} players with complete data: {len(pred_df)}")

    # --- Feature engineering ---
    pred_df = add_base_rating(pred_df)
    pred_df = add_age_features(pred_df)
    pred_df = compute_availability(pred_df)
    pred_df = add_cba_features(pred_df)

    if "minutes" in pred_df.columns and "games" in pred_df.columns:
        pred_df["mpg"] = pred_df["minutes"] / pred_df["games"].replace(0, np.nan)

    heights_path = PROCESSED_DIR / "heights.csv"
    if heights_path.exists():
        heights = pd.read_csv(heights_path)
        heights["player_name_norm"] = heights["player_name_bbref"].apply(_normalize_name)
        pred_df = pred_df.merge(
            heights[["player_name_norm", "height_inches"]],
            on="player_name_norm", how="left",
        )

    # --- External features (award_score_cum, draft_pick) ---
    train_data = pd.read_csv(
        PROCESSED_DIR / "training_data_v2.csv"
        if (PROCESSED_DIR / "training_data_v2.csv").exists()
        else PROCESSED_DIR / "training_data.csv"
    )
    ext_cols = ["player_name_norm", "season", "award_score_cum", "draft_pick", "prev_cap_pct"]
    ext = train_data[[c for c in ext_cols if c in train_data.columns]].drop_duplicates(
        ["player_name_norm", "season"]
    )
    if "award_score_cum" in ext.columns:
        award_latest = ext.sort_values("season").drop_duplicates("player_name_norm", keep="last")
        pred_df = pred_df.merge(
            award_latest[["player_name_norm", "award_score_cum"]],
            on="player_name_norm", how="left",
        )
        pred_df["award_score_cum"] = pred_df["award_score_cum"].fillna(0)
    if "draft_pick" in ext.columns:
        draft_latest = ext.drop_duplicates("player_name_norm", keep="first")
        pred_df = pred_df.merge(
            draft_latest[["player_name_norm", "draft_pick"]],
            on="player_name_norm", how="left",
        )
        pred_df["draft_pick"] = pred_df["draft_pick"].fillna(75)
    if "prev_cap_pct" in ext.columns:
        prev_latest = ext.sort_values("season").drop_duplicates("player_name_norm", keep="last")
        pred_df = pred_df.merge(
            prev_latest[["player_name_norm", "prev_cap_pct"]],
            on="player_name_norm", how="left",
        )
        median_rookie = train_data[train_data["draft_pick"] <= 30]["cap_pct"].median()
        pred_df["prev_cap_pct"] = pred_df["prev_cap_pct"].fillna(median_rookie)

    # --- Predict ---
    X_pred = pred_df[[f for f in features if f in pred_df.columns]].copy()
    for f in features:
        if f not in X_pred.columns:
            X_pred[f] = 0
    X_pred = X_pred[features].fillna(X_pred.median()).fillna(0)

    pred_df["predicted_cap_pct"] = model.predict(X_pred)
    cap = CAP_BY_SEASON.get(target_season, 153_000_000)
    pred_df["predicted_salary"] = pred_df["predicted_cap_pct"] * cap

    # --- FA identification + diff ---
    sal = pd.read_csv(PROCESSED_DIR / "salaries.csv")
    sal["player_name_norm"] = sal["player"].apply(_normalize_name)

    under_contract = set(
        sal[sal["season"] == target_season]["player_name_norm"].unique()
    )
    sal_prev = (
        sal[sal["season"] == target_season - 1]
        .drop_duplicates("player_name_norm")
        .set_index("player_name_norm")["salary"]
    )
    sal_curr = (
        sal[sal["season"] == target_season]
        .drop_duplicates("player_name_norm")
        .set_index("player_name_norm")["salary"]
    )

    pred_df["is_free_agent"] = ~pred_df["player_name_norm"].isin(under_contract)
    pred_df["actual_salary"] = pred_df["player_name_norm"].map(sal_curr)
    pred_df["reference_salary"] = pred_df.apply(
        lambda r: r["actual_salary"] if pd.notna(r["actual_salary"])
        else sal_prev.get(r["player_name_norm"], np.nan),
        axis=1,
    )
    pred_df["diff"] = pred_df["predicted_salary"] - pred_df["reference_salary"]

    # --- Output ---
    out_cols = [
        "player_name", "player_name_norm", "season", "age", "position",
        "predicted_cap_pct", "predicted_salary",
        "is_free_agent", "actual_salary", "reference_salary", "diff",
        "darko_dpm", "lebron", "rapm",
        "minutes", "usage_pct", "team_abbreviation",
    ]
    out = pred_df[[c for c in out_cols if c in pred_df.columns]].copy()
    out = out.sort_values("predicted_salary", ascending=False)
    return out


if __name__ == "__main__":
    out = predict(target_season=2026)

    pred_dir = OUTPUTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(pred_dir / "predictions_2026.csv", index=False)
    fa = out[out["is_free_agent"] == True]
    fa.to_csv(pred_dir / "free_agents_2026.csv", index=False)

    print(f"\nSaved: {len(out)} total, {len(fa)} FAs")
    print(f"\nTop 20:")
    for _, r in out.head(20).iterrows():
        fa_tag = " [FA]" if r.get("is_free_agent") else ""
        diff_s = f" diff={r['diff']/1e6:+.1f}M" if pd.notna(r.get("diff")) else ""
        pn = str(r["player_name"])[:25]
        print(f"  {pn:25s} age={r['age']:.0f} pred=${r['predicted_salary']/1e6:5.1f}M{diff_s}{fa_tag}")
