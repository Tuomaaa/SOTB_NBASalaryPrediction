"""Generate predictions for upcoming free agents using the champion Grabit stack.

Trains the two-sided censored-normal Grabit model on all seasons before the
target, applies Stage-2 CBA bounds (max eligible + floor clip), and identifies
free agents by comparing against the salary roll.

This is the same pipeline export_web.py uses to produce the portfolio site's
valuations — it must stay in sync with that path.
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, OUTPUTS_DIR, CAP_BY_SEASON
from src.model.train import (
    load_training_data, train_grabit, _compute_max_eligible, _compute_floor,
    _filter_year1, _filter_rookie_scale, _filter_prorated,
    _filter_mislabeled_year1, _filter_continuations,
    _prepare_Xy, FEATURE_COLS, TARGET,
)


def _normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _training_medians(df: pd.DataFrame) -> tuple[list[str], pd.Series]:
    """Feature list and fill values from the filtered training set."""
    tr = _filter_continuations(_filter_mislabeled_year1(_compute_max_eligible(
        _filter_prorated(_filter_rookie_scale(_filter_year1(df))))))
    X_tr, _, _, features = _prepare_Xy(tr)
    return features, X_tr.median()


def predict(target_season: int = 2026) -> pd.DataFrame:
    """Train the champion Grabit model and predict the target season."""
    df = load_training_data()
    train_df = df[df["season"] < target_season].copy()
    print(f"Training on {len(train_df)} rows (seasons < {target_season})")

    results, model, features = train_grabit(train_df, sigma=0.02)
    print(f"Grabit CV R²: {results['cv_r2_mean']:.4f}")

    tr_features, medians = _training_medians(train_df)
    if tr_features != features:
        raise SystemExit("feature list drifted between fit and export")

    # Build prediction frame from impact metrics
    impact = pd.read_csv(PROCESSED_DIR / "impact_metrics.csv")
    pred_df = impact[
        (impact["season"] == target_season)
        & (impact["age"].notna())
        & (impact["minutes"].notna())
    ].copy()
    print(f"\n{target_season} players with complete data: {len(pred_df)}")

    # Borrow features from each player's most recent training row
    from src.features.base_rating import add_base_rating
    from src.features.age_curve import add_age_features
    from src.features.availability import compute_availability
    from src.features.cba_constraints import add_cba_features

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

    # Borrow stable features from history
    hist = df.sort_values("season").groupby("player_name_norm").last()
    for col in ("draft_pick", "prev_cap_pct", "award_score_cum", "ast_pct",
                "availability_3yr"):
        if col in hist.columns:
            borrowed = pred_df["player_name_norm"].map(hist[col])
            if col in pred_df.columns:
                pred_df[col] = pred_df[col].where(pred_df[col].notna(), borrowed)
            else:
                pred_df[col] = borrowed
    if "draft_pick" in pred_df.columns:
        pred_df["draft_pick"] = pred_df["draft_pick"].fillna(75)

    # Predict: latent value from Grabit, then Stage-2 CBA clip
    X_pred = pred_df.reindex(columns=features).fillna(medians).fillna(0)
    latent = model.predict(X_pred)

    # Compute CBA bounds for the prediction rows
    pred_df["cap_pct"] = latent  # temporary for _compute_max_eligible
    pred_df["salary"] = latent * CAP_BY_SEASON.get(target_season, 153_000_000)
    pred_df = _compute_max_eligible(pred_df)
    pred_df = _compute_floor(pred_df)

    max_elig = pred_df["max_eligible_pct"].values
    floor_pct = pred_df["floor_pct"].values
    capped = np.clip(latent, floor_pct, max_elig)

    cap = CAP_BY_SEASON.get(target_season, 153_000_000)
    pred_df["latent_cap_pct"] = latent
    pred_df["predicted_cap_pct"] = capped
    pred_df["predicted_salary"] = capped * cap
    pred_df["latent_salary"] = latent * cap
    pred_df["is_capped"] = latent > max_elig + 1e-9
    pred_df["is_floored"] = latent < floor_pct - 1e-9

    # FA identification + diff
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

    out_cols = [
        "player_name", "player_name_norm", "season", "age", "position",
        "predicted_cap_pct", "predicted_salary", "latent_cap_pct", "latent_salary",
        "is_capped", "is_floored",
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
    n_capped = out.get("is_capped", pd.Series(dtype=bool)).sum()
    n_floored = out.get("is_floored", pd.Series(dtype=bool)).sum()
    print(f"CBA bounds: {n_capped} capped, {n_floored} floored")
    print(f"\nTop 20:")
    for _, r in out.head(20).iterrows():
        fa_tag = " [FA]" if r.get("is_free_agent") else ""
        diff_s = f" diff={r['diff']/1e6:+.1f}M" if pd.notna(r.get("diff")) else ""
        bound = ""
        if r.get("is_capped"):
            bound = " [MAX]"
        elif r.get("is_floored"):
            bound = " [MIN]"
        pn = str(r["player_name"])[:25]
        print(f"  {pn:25s} age={r['age']:.0f} pred=${r['predicted_salary']/1e6:5.1f}M"
              f"{bound}{diff_s}{fa_tag}")
