"""Train valuation models: Ridge baseline, XGBoost, and Grabit.

Target: cap_pct (salary as fraction of salary cap).
Training data filtered to year-1 contracts only (year 2+ are CBA escalators).
Rookie-scale contracts (1st-round picks, years 2-4) removed via draft data.
Uses 5-fold GroupKFold CV (same player stays in same fold).

Grabit (Gradient Tree-Boosted Tobit): XGBoost with custom censored-normal
loss for CBA-capped contracts (max, vet min, MLE, BAE). Outputs latent
market value; Stage 2 clips to CBA max eligible %.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import PROCESSED_DIR, OUTPUTS_DIR, CAP_BY_SEASON, RAW_DIR

FEATURE_COLS = [
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
    "prev_cap_pct",
]

TARGET = "cap_pct"

ELITE_AWARDS = {
    "All-NBA 1st Team", "All-NBA 2nd Team", "All-NBA 3rd Team",
    "MVP", "Defensive Player of the Year",
}


def load_training_data() -> pd.DataFrame:
    v2 = PROCESSED_DIR / "training_data_v2.csv"
    path = v2 if v2.exists() else PROCESSED_DIR / "training_data.csv"
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


def _load_draft_years() -> dict[str, int]:
    """Load {normalized_name: draft_year} from draft data."""
    from scripts.build_external_features import norm
    path = RAW_DIR / "raw_external" / "player_draft_2020-2025.matched.corrected.csv"
    if not path.exists():
        return {}
    draft_df = pd.read_csv(path)
    draft_df["pn"] = draft_df["player"].apply(norm)
    out = {}
    for _, r in draft_df.iterrows():
        try:
            out.setdefault(r["pn"], int(r["year"]))
        except (ValueError, TypeError):
            pass
    return out


def _load_elite_set() -> set[tuple[str, int]]:
    """Load (normalized_name, year) pairs for elite award winners."""
    from scripts.build_external_features import norm
    path = RAW_DIR / "raw_external" / "awards_full.csv"
    if not path.exists():
        return set()
    aw = pd.read_csv(path)
    aw["pn"] = aw["player_name_norm"].apply(norm)
    el = aw[aw["award"].isin(ELITE_AWARDS)]
    return set(zip(el["pn"], el["year"].astype(int)))


def _compute_max_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Compute max_eligible_pct with Rose Rule / Supermax from draft + awards data."""
    from scripts.build_external_features import norm

    df = df.copy()
    draft_years = _load_draft_years()
    elite_set = _load_elite_set()

    def _elite_count(pn, years):
        return sum(1 for y in years if (pn, y) in elite_set)

    df["pn_clean"] = df["player_name_norm"].apply(norm)
    df["_dy"] = df["pn_clean"].map(draft_years)
    exp_draft = df["season"] - df["_dy"]
    exp_age = (df["age"].fillna(25) - 19).clip(lower=0)
    exp = exp_draft.fillna(exp_age).astype(int).clip(lower=0).values

    base = np.where(exp >= 10, 0.35, np.where(exp >= 7, 0.30, 0.25))

    pns = df["pn_clean"].values
    seasons = df["season"].values
    rose = np.zeros(len(df), dtype=bool)
    supermax = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        p, s = pns[i], int(seasons[i])
        trig = (p, s) in elite_set or _elite_count(p, [s - 2, s - 1, s]) >= 2
        if trig:
            if exp[i] <= 6:
                rose[i] = True
            elif 7 <= exp[i] <= 9:
                supermax[i] = True

    base = np.where(supermax, 0.35, base)
    base = np.where(rose, np.maximum(base, 0.30), base)

    df["max_eligible_pct"] = base
    df["is_max_contract"] = df[TARGET] >= base * 0.90
    df = df.drop(columns=["pn_clean", "_dy"])
    return df


def _make_tobit_obj(cens_mask: np.ndarray, sigma: float = 0.02):
    """Custom XGBoost objective: censored-normal (Grabit).

    Uncensored rows: standard squared error.
    Censored rows (max contracts): inverse Mills ratio pushes predictions above ceiling.
    """
    _c = cens_mask.copy()

    def obj(y_true, y_pred):
        grad = np.empty_like(y_pred)
        hess = np.empty_like(y_pred)
        unc = ~_c
        grad[unc] = y_pred[unc] - y_true[unc]
        hess[unc] = 1.0
        if _c.any():
            z = (y_pred[_c] - y_true[_c]) / sigma
            m = np.exp(norm.logpdf(z) - norm.logcdf(z))
            grad[_c] = -sigma * m
            hess[_c] = np.clip(m * (z + m), 1e-6, None)
        return grad, hess

    return obj


_XGB_BASE = dict(
    n_estimators=500, max_depth=4, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
    random_state=42, tree_method="hist",
)


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
    from sklearn.metrics import r2_score, mean_absolute_error

    df = _filter_year1(df)
    df = _filter_rookie_scale(df)
    X, y, groups, features = _prepare_Xy(df)
    seasons = df["season"].values
    print(f"Training XGBoost on {len(X)} samples, {len(features)} features")

    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        random_state=42,
        tree_method="hist",
    )

    cv = GroupKFold(n_splits=5)
    folds = list(cv.split(X, y, groups))

    oof_pred = np.full(len(y), np.nan)
    fold_r2 = []
    fold_mae = []
    train_r2_list = []
    for tr_i, va_i in folds:
        m = XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.01,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
            random_state=42, tree_method="hist",
        )
        m.fit(X.iloc[tr_i], y[tr_i])
        p = m.predict(X.iloc[va_i])
        oof_pred[va_i] = p
        fold_r2.append(r2_score(y[va_i], p))
        fold_mae.append(mean_absolute_error(y[va_i], p))
        train_r2_list.append(r2_score(y[tr_i], m.predict(X.iloc[tr_i])))

    recent_mask = seasons >= 2024
    recent_r2 = r2_score(y[recent_mask], oof_pred[recent_mask]) if recent_mask.sum() > 10 else float("nan")
    recent_mae = mean_absolute_error(y[recent_mask], oof_pred[recent_mask]) if recent_mask.sum() > 10 else float("nan")

    xgb.fit(X, y)

    results = {
        "model": "XGBoost",
        "n_samples": len(X),
        "n_features": len(features),
        "features": features,
        "cv_r2_mean": float(np.mean(fold_r2)),
        "cv_r2_std": float(np.std(fold_r2)),
        "cv_mae_mean": float(np.mean(fold_mae)),
        "cv_mae_std": float(np.std(fold_mae)),
        "train_r2_mean": float(np.mean(train_r2_list)),
        "cv_r2_recent": float(recent_r2),
        "cv_mae_recent": float(recent_mae),
        "recent_n": int(recent_mask.sum()),
    }

    importances = dict(zip(features, xgb.feature_importances_))
    results["feature_importances"] = {k: round(float(v), 4) for k, v in
                                       sorted(importances.items(), key=lambda x: -x[1])}

    return results, xgb, features


def train_grabit(df: pd.DataFrame, sigma: float = 0.02,
                 gate_frac: float = 0.55) -> tuple[dict, object, list[str]]:
    """Train Grabit v3: XGBoost with censored-normal loss + CBA cap.

    Stage 1: Grabit — max-contract rows where baseline pred ≥ gate_frac * max_eligible
    get censored-normal gradients (inverse Mills ratio). Other rows get standard MSE.
    Stage 2: CBA cap — final_pred = min(latent, max_eligible_pct).
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score, mean_absolute_error

    df = _filter_year1(df)
    df = _filter_rookie_scale(df)
    df = _compute_max_eligible(df)

    X, y, groups, features = _prepare_Xy(df)
    seasons = df["season"].values
    max_elig = df["max_eligible_pct"].values
    is_max = df["is_max_contract"].values

    print(f"Training Grabit v3 (σ={sigma}, gate={gate_frac}) on {len(X)} samples")
    print(f"  Max contract rows: {is_max.sum()}/{len(X)}")

    cv = GroupKFold(n_splits=5)
    folds = list(cv.split(X, y, groups))

    # Baseline OOF for gating
    oof_bl = np.full(len(y), np.nan)
    for tr_i, va_i in folds:
        m = XGBRegressor(**_XGB_BASE)
        m.fit(X.iloc[tr_i], y[tr_i])
        oof_bl[va_i] = m.predict(X.iloc[va_i])
    gate = is_max & (oof_bl >= gate_frac * max_elig)
    print(f"  Gated censored rows: {gate.sum()}/{is_max.sum()} max rows")

    # Grabit CV
    oof_pred = np.full(len(y), np.nan)
    fold_r2 = []
    fold_mae = []
    for fi, (tr_i, va_i) in enumerate(folds):
        obj = _make_tobit_obj(gate[tr_i], sigma)
        m = XGBRegressor(**{**_XGB_BASE, "objective": obj,
                            "base_score": float(y[tr_i].mean())})
        m.fit(X.iloc[tr_i], y[tr_i])
        latent = m.predict(X.iloc[va_i])
        capped = np.minimum(latent, max_elig[va_i])
        oof_pred[va_i] = capped

        fr2 = r2_score(y[va_i], capped)
        fmae = mean_absolute_error(y[va_i], capped)
        fold_r2.append(fr2)
        fold_mae.append(fmae)

    recent_mask = seasons >= 2024
    recent_r2 = r2_score(y[recent_mask], oof_pred[recent_mask]) if recent_mask.sum() > 10 else float("nan")
    recent_mae = mean_absolute_error(y[recent_mask], oof_pred[recent_mask]) if recent_mask.sum() > 10 else float("nan")

    # Final model on all data
    obj_final = _make_tobit_obj(gate, sigma)
    m_final = XGBRegressor(**{**_XGB_BASE, "objective": obj_final,
                              "base_score": float(y.mean())})
    m_final.fit(X, y)

    results = {
        "model": "Grabit v3",
        "sigma": sigma,
        "n_samples": len(X),
        "n_features": len(features),
        "n_censored": int(gate.sum()),
        "features": features,
        "cv_r2_mean": float(np.mean(fold_r2)),
        "cv_r2_std": float(np.std(fold_r2)),
        "cv_mae_mean": float(np.mean(fold_mae)),
        "cv_mae_std": float(np.std(fold_mae)),
        "cv_r2_recent": float(recent_r2),
        "cv_mae_recent": float(recent_mae),
        "recent_n": int(recent_mask.sum()),
    }
    return results, m_final, features


def print_results(results: dict):
    print(f"\n{'='*50}")
    print(f"  {results['model']} Results")
    print(f"{'='*50}")
    print(f"  CV R²:  {results['cv_r2_mean']:.4f} ± {results.get('cv_r2_std', 0):.4f}")
    print(f"  CV MAE: {results['cv_mae_mean']:.4f} ± {results.get('cv_mae_std', 0):.4f}")
    if "cv_r2_recent" in results and not np.isnan(results["cv_r2_recent"]):
        print(f"  CV R² (2024-26): {results['cv_r2_recent']:.4f}  (n={results.get('recent_n', '?')})")
        print(f"  CV MAE (2024-26): {results['cv_mae_recent']:.4f}")
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

    xgb_results, xgb_model, xgb_features = train_xgboost(df)
    print_results(xgb_results)

    grabit_results, grabit_model, grabit_features = train_grabit(df, sigma=0.02)
    print_results(grabit_results)

    print(f"\n{'='*60}")
    print(f"  XGBoost vs Grabit v3")
    print(f"{'='*60}")
    print(f"{'':20s} {'CV R²':>10s} {'R²(24-26)':>12s} {'CV MAE':>10s} {'MAE(24-26)':>12s}")
    for r in [xgb_results, grabit_results]:
        rr = r.get("cv_r2_recent", float("nan"))
        rm = r.get("cv_mae_recent", float("nan"))
        print(f"  {r['model']:18s} {r['cv_r2_mean']:.4f}     {rr:.4f}       {r['cv_mae_mean']:.4f}     {rm:.4f}")

    model_dir = OUTPUTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "xgb_results.json", "w") as f:
        json.dump(xgb_results, f, indent=2)
    with open(model_dir / "grabit_results.json", "w") as f:
        json.dump(grabit_results, f, indent=2)
    print(f"\nResults saved to {model_dir}")
