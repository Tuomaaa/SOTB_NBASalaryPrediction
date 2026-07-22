"""Phase 3D: CBA-censored Tobit regression (extended).

Right-censored Tobit across all CBA-capped contract types:
  - Max contracts: cap_pct ceiling by experience/awards
  - Veteran minimum: salary IS the vet min amount
  - MLE (non-taxpayer / taxpayer / room): exception ceiling
  - BAE (bi-annual exception): exception ceiling

Players whose signing_type is unknown (not in Spotrac data) are
treated as uncensored.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR, OUTPUTS_DIR, CAP_BY_SEASON
from src.model.train import (
    load_training_data, _filter_year1, _filter_rookie_scale, FEATURE_COLS, TARGET,
)

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
    random_state=42, tree_method="hist",
)

OUT_DIR = OUTPUTS_DIR / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── CBA exception ceilings (first-year salary, dollars) ─────────
# Non-taxpayer mid-level exception
NTML_BY_SEASON = {
    2019: 9_258_000, 2020: 9_258_000, 2021: 9_536_000,
    2022: 10_349_000, 2023: 12_405_000, 2024: 12_860_000,
    2025: 14_167_000,
}
# Taxpayer mid-level exception
TPML_BY_SEASON = {
    2019: 5_718_000, 2020: 5_718_000, 2021: 5_890_000,
    2022: 6_392_000, 2023: 5_175_000, 2024: 5_175_000,
    2025: 5_175_000,
}
# Room mid-level exception
ROOM_BY_SEASON = {
    2019: 4_767_000, 2020: 4_767_000, 2021: 4_900_000,
    2022: 5_300_000, 2023: 7_723_000, 2024: 8_023_000,
    2025: 8_500_000,
}
# Bi-annual exception
BAE_BY_SEASON = {
    2019: 3_623_000, 2020: 3_623_000, 2021: 3_732_000,
    2022: 4_050_000, 2023: 4_516_000, 2024: 4_681_000,
    2025: 4_900_000,
}
# Vet min — highest tier (10+ year veteran)
VET_MIN_MAX_BY_SEASON = {
    2019: 2_564_753, 2020: 2_564_753, 2021: 2_641_691,
    2022: 2_891_467, 2023: 3_196_448, 2024: 3_300_000,
    2025: 3_500_000,
}

EXCEPTION_AMOUNTS = {
    "non-taxpayer-mid-level-exception": NTML_BY_SEASON,
    "taxpayer-mid-level-exception": TPML_BY_SEASON,
    "room-mid-level-exception": ROOM_BY_SEASON,
    "bi-annual-exception": BAE_BY_SEASON,
}


# ─── Max eligible % ───────────────────────────────────────────────
def compute_max_eligible_pct(df):
    """CBA max starting salary as % of cap, by years of service."""
    df = df.copy()
    exp = (df["age"].fillna(25) - 18).clip(lower=0).astype(int)
    base = np.where(exp >= 10, 0.35, np.where(exp >= 7, 0.30, 0.25))
    awards = df["award_score_cum"].fillna(0).values
    supermax = (exp >= 7) & (exp <= 9) & (awards >= 3)
    base = np.where(supermax, 0.35, base)
    rose = (exp <= 6) & (awards >= 1)
    base = np.where(rose, np.maximum(base, 0.30), base)
    df["max_eligible_pct"] = base
    df["is_max_contract"] = df["cap_pct"] >= df["max_eligible_pct"] * 0.90
    return df


def _merge_signing_type(df):
    """Merge Spotrac signing types, deduplicating to one per player-season."""
    st_path = PROCESSED_DIR / "spotrac_signing_types.csv"
    if not st_path.exists():
        df["signing_type"] = np.nan
        return df
    st = pd.read_csv(st_path)[["player_name_norm", "season", "signing_type"]]
    st = st.drop_duplicates(["player_name_norm", "season", "signing_type"])
    # Prefer specific types over Minimum when duplicates exist
    st["_prio"] = (st["signing_type"] != "Minimum").astype(int)
    st = st.sort_values("_prio", ascending=False).drop_duplicates(
        ["player_name_norm", "season"]
    ).drop(columns=["_prio"])
    df = df.merge(st, on=["player_name_norm", "season"], how="left")
    return df


def compute_all_censoring(df):
    """Flag right-censored observations for all CBA-capped contract types.

    Returns df with columns: is_censored, upper_bound, censor_reason.
    """
    df = compute_max_eligible_pct(df)
    df = _merge_signing_type(df)

    is_censored = df["is_max_contract"].copy()
    upper_bound = df["max_eligible_pct"].copy()
    censor_reason = pd.Series("", index=df.index)
    censor_reason[is_censored] = "max_contract"

    cap_series = df["season"].map(CAP_BY_SEASON).fillna(153_000_000)

    # Vet min: all Minimum contracts with plausible cap_pct
    vet_ceiling_pct = df["season"].map(VET_MIN_MAX_BY_SEASON).fillna(0) / cap_series
    vet_mask = (
        (~is_censored)
        & (df["signing_type"] == "Minimum")
        & (vet_ceiling_pct > 0)
        & (df["cap_pct"] <= vet_ceiling_pct * 1.2)
    )
    is_censored |= vet_mask
    upper_bound[vet_mask] = df.loc[vet_mask, "cap_pct"]
    censor_reason[vet_mask] = "vet_min"

    # MLE types + BAE: censored if at/near exception ceiling
    for exc_type, amounts in EXCEPTION_AMOUNTS.items():
        exc_ceil_pct = df["season"].map(amounts).fillna(0) / cap_series
        exc_mask = (
            (~is_censored)
            & (df["signing_type"] == exc_type)
            & (exc_ceil_pct > 0)
            & (df["cap_pct"] >= exc_ceil_pct * 0.85)
            & (df["cap_pct"] <= exc_ceil_pct * 1.15)
        )
        is_censored |= exc_mask
        upper_bound[exc_mask] = exc_ceil_pct[exc_mask]
        censor_reason[exc_mask] = exc_type

    df["is_censored"] = is_censored
    df["upper_bound"] = upper_bound
    df["censor_reason"] = censor_reason
    return df


# ─── Tobit MLE ─────────────────────────────────────────────────────
def tobit_negloglik(params, X, y, censored, upper_bounds):
    """Negative log-likelihood for right-censored Tobit."""
    n_feat = X.shape[1]
    beta = params[:n_feat]
    log_sigma = params[n_feat]
    sigma = np.exp(log_sigma)
    mu = X @ beta

    ll = 0.0
    unc = ~censored
    if unc.any():
        z = (y[unc] - mu[unc]) / sigma
        ll += np.sum(stats.norm.logpdf(z) - log_sigma)
    if censored.any():
        z_c = (upper_bounds[censored] - mu[censored]) / sigma
        ll += np.sum(np.clip(stats.norm.logsf(z_c), -500, 0))
    return -ll


def fit_tobit(X, y, censored, upper_bounds):
    from statsmodels.regression.linear_model import OLS
    ols = OLS(y, X).fit()
    x0 = np.concatenate([ols.params, [np.log(np.std(ols.resid))]])
    result = minimize(
        tobit_negloglik, x0,
        args=(X, y, censored, upper_bounds),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12},
    )
    beta = result.x[:X.shape[1]]
    sigma = np.exp(result.x[X.shape[1]])
    return beta, sigma, result


def main():
    # ─── 1. Load, filter, compute censoring ───────────────────────
    df = load_training_data()
    df = _filter_year1(df)
    df = _filter_rookie_scale(df)
    df = compute_all_censoring(df)

    n_total = len(df)
    n_censored = df["is_censored"].sum()
    print(f"Training data: {n_total} rows, {n_censored} censored ({n_censored/n_total:.1%})")

    # Censoring breakdown
    print(f"\nCensoring by reason:")
    for reason in ["max_contract", "vet_min", "non-taxpayer-mid-level-exception",
                    "taxpayer-mid-level-exception", "room-mid-level-exception",
                    "bi-annual-exception"]:
        n = (df["censor_reason"] == reason).sum()
        if n > 0:
            sub = df[df["censor_reason"] == reason]
            print(f"  {reason:40s} n={n:4d}  cap_pct=[{sub['cap_pct'].min():.4f}, {sub['cap_pct'].max():.4f}]  "
                  f"upper_bound=[{sub['upper_bound'].min():.4f}, {sub['upper_bound'].max():.4f}]")
    n_unc = (df["censor_reason"] == "").sum()
    print(f"  {'uncensored':40s} n={n_unc:4d}")

    print(f"\nCensoring by season:")
    for s in sorted(df["season"].unique()):
        sub = df[df["season"] == s]
        nc = sub["is_censored"].sum()
        print(f"  {s}: {nc:3d}/{len(sub):3d} censored ({nc/len(sub):.1%})")

    # ─── 2. XGBoost OOF predictions (standard, no censoring) ─────
    avail = [f for f in FEATURE_COLS if f in df.columns]
    X_raw = df[avail].copy()
    med = X_raw.median()
    X_raw = X_raw.fillna(med).fillna(0)
    y = df[TARGET].values

    oof_xgb = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=5)
    groups = df["player_name_norm"].values
    for tr_i, va_i in gkf.split(X_raw, y, groups):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X_raw.iloc[tr_i], y[tr_i])
        oof_xgb[va_i] = m.predict(X_raw.iloc[va_i])
    df["pred_xgb"] = oof_xgb

    # ─── 3. Tobit regression (extended censoring) ────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_tobit = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    censored = df["is_censored"].values
    upper_bounds = df["upper_bound"].values

    print(f"\nFitting Tobit (extended censoring)...")
    print(f"  Uncensored: {(~censored).sum()}, Censored: {censored.sum()}")

    beta, sigma, result = fit_tobit(X_tobit, y, censored, upper_bounds)
    print(f"  Converged: {result.success}")
    print(f"  Log-likelihood (Tobit): {-result.fun:.1f}")
    print(f"  Sigma (Tobit): {sigma:.4f}")

    from statsmodels.regression.linear_model import OLS
    ols = OLS(y, X_tobit).fit()
    print(f"  Log-likelihood (OLS):   {ols.llf:.1f}")
    print(f"  Sigma (OLS):   {np.std(ols.resid):.4f}")

    df["pred_tobit_latent"] = X_tobit @ beta

    # ─── 4. Compare: censored players by type ────────────────────
    df["cap"] = df["season"].map(CAP_BY_SEASON).fillna(153_000_000)

    for reason, label in [
        ("max_contract", "MAX CONTRACT"),
        ("vet_min", "VETERAN MINIMUM"),
        ("non-taxpayer-mid-level-exception", "NON-TAXPAYER MLE"),
        ("taxpayer-mid-level-exception", "TAXPAYER MLE"),
        ("room-mid-level-exception", "ROOM MLE"),
        ("bi-annual-exception", "BI-ANNUAL EXCEPTION"),
    ]:
        sub = df[df["censor_reason"] == reason].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("cap_pct", ascending=False)

        print(f"\n{'='*100}")
        print(f"{label} PLAYERS (n={len(sub)}): XGBoost vs Tobit latent")
        print(f"{'='*100}")
        print(f"{'Player':25s} {'Ssn':>4s} {'Age':>3s} {'Act%':>6s} {'Ceil%':>6s} "
              f"{'XGB%':>6s} {'Tob%':>6s} {'Act$M':>6s} {'XGB$M':>6s} {'Tob$M':>6s}")
        print("-" * 100)

        for _, r in sub.head(20).iterrows():
            cap = r["cap"]
            print(f"{r['player_name_norm'][:25]:25s} {int(r['season']):4d} {r['age']:3.0f} "
                  f"{r['cap_pct']:6.3f} {r['upper_bound']:6.3f} "
                  f"{r['pred_xgb']:6.3f} {r['pred_tobit_latent']:6.3f} "
                  f"{r['cap_pct']*cap/1e6:6.1f} {r['pred_xgb']*cap/1e6:6.1f} "
                  f"{r['pred_tobit_latent']*cap/1e6:6.1f}")

    # ─── 5. Summary ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    cens_mask = df["is_censored"]
    cap_avg = df["cap"].mean()

    for label, col in [("XGBoost", "pred_xgb"), ("Tobit latent", "pred_tobit_latent")]:
        c = df[cens_mask]
        u = df[~cens_mask]
        bias_c = (c[col] - c[TARGET]).mean()
        mae_c = (c[col] - c[TARGET]).abs().mean()
        mae_u = (u[col] - u[TARGET]).abs().mean()
        print(f"\n  {label}:")
        print(f"    Censored (n={len(c)}):   bias={bias_c:+.4f} (${bias_c*cap_avg/1e6:+.1f}M)  MAE={mae_c:.4f} (${mae_c*cap_avg/1e6:.1f}M)")
        print(f"    Uncensored (n={len(u)}): MAE={mae_u:.4f} (${mae_u*cap_avg/1e6:.1f}M)")

        # Per-reason breakdown
        for reason in ["max_contract", "vet_min", "non-taxpayer-mid-level-exception",
                        "taxpayer-mid-level-exception", "room-mid-level-exception",
                        "bi-annual-exception"]:
            sub = df[df["censor_reason"] == reason]
            if len(sub) == 0:
                continue
            bias = (sub[col] - sub[TARGET]).mean()
            mae = (sub[col] - sub[TARGET]).abs().mean()
            print(f"      {reason:38s} n={len(sub):4d}  bias={bias:+.4f}  MAE={mae:.4f}")

    # Tobit latent > upper_bound counts
    print(f"\n  Tobit latent > upper_bound by reason:")
    for reason in ["max_contract", "vet_min", "non-taxpayer-mid-level-exception",
                    "taxpayer-mid-level-exception", "room-mid-level-exception",
                    "bi-annual-exception"]:
        sub = df[df["censor_reason"] == reason]
        if len(sub) == 0:
            continue
        above = (sub["pred_tobit_latent"] > sub["upper_bound"]).sum()
        print(f"    {reason:40s} {above}/{len(sub)} ({above/len(sub):.0%})")

    # ─── 6. Output CSV ────────────────────────────────────────────
    out = df[["player_name_norm", "season", "age", "cap_pct",
              "upper_bound", "is_censored", "censor_reason",
              "pred_xgb", "pred_tobit_latent"]].copy()
    out["cap"] = df["cap"]
    out["actual_$M"] = (out["cap_pct"] * out["cap"] / 1e6).round(2)
    out["xgb_$M"] = (out["pred_xgb"] * out["cap"] / 1e6).round(2)
    out["tobit_$M"] = (out["pred_tobit_latent"] * out["cap"] / 1e6).round(2)
    out["tobit_minus_xgb_$M"] = (out["tobit_$M"] - out["xgb_$M"]).round(2)
    out = out.sort_values(["is_censored", "cap_pct"], ascending=[False, False])

    path = OUT_DIR / "tobit_vs_xgb_v3.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved: {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
