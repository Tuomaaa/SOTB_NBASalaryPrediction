"""Phase 3D: Max contract censoring analysis via Tobit regression.

No model retraining. Analysis + post-processing only.
Right-censored Tobit: players at/near max_eligible_pct are censored —
their true market value may exceed the CBA cap.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
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


# ─── Max eligible % ───────────────────────────────────────────────
def compute_max_eligible_pct(df):
    """CBA max starting salary as % of cap, by years of service.

    Experience approximated as age - 18 (conservative: assumes earliest
    possible draft age). Supermax (DVPE) and Rose Rule extensions handled
    via award_score_cum.

    Returns max_elig as cap_pct ceiling for each row.
    """
    df = df.copy()
    exp = (df["age"].fillna(25) - 18).clip(lower=0).astype(int)

    # Base tier: 0-6 yrs → 25%, 7-9 → 30%, 10+ → 35%
    base = np.where(exp >= 10, 0.35, np.where(exp >= 7, 0.30, 0.25))

    awards = df["award_score_cum"].fillna(0).values

    # Supermax (DVPE): 35% if 7-9 years + significant awards (All-NBA etc.)
    supermax = (exp >= 7) & (exp <= 9) & (awards >= 3)
    base = np.where(supermax, 0.35, base)

    # Rose Rule (Designated Rookie Extension): up to 30% if ≤6 years + awards
    rose = (exp <= 6) & (awards >= 1)
    base = np.where(rose, np.maximum(base, 0.30), base)

    df["max_eligible_pct"] = base
    df["is_max_contract"] = df["cap_pct"] >= df["max_eligible_pct"] * 0.90
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
    # ─── 1. Load and flag ──────────────────────────────────────────
    df = load_training_data()
    df = _filter_year1(df)
    df = _filter_rookie_scale(df)
    df = compute_max_eligible_pct(df)

    n_max = df["is_max_contract"].sum()
    n_total = len(df)
    print(f"Training data: {n_total} rows, {n_max} max contracts ({n_max/n_total:.1%})")

    print(f"\nMax contracts by season:")
    for s in sorted(df["season"].unique()):
        sub = df[df["season"] == s]
        nm = sub["is_max_contract"].sum()
        print(f"  {s}: {nm}/{len(sub)} ({nm/len(sub):.1%})")

    print(f"\nMax contracts by max_eligible_pct tier:")
    for tier in sorted(df["max_eligible_pct"].unique()):
        sub = df[df["max_eligible_pct"] == tier]
        nm = sub["is_max_contract"].sum()
        print(f"  {tier:.0%} tier: {nm}/{len(sub)} max ({nm/len(sub):.1%})")

    # Sanity check: print max players
    max_players = df[df["is_max_contract"]].sort_values("cap_pct", ascending=False)
    print(f"\nMax contract players (top 20):")
    print(f"  {'Player':25s} {'Ssn':>4s} {'Age':>4s} {'Exp':>4s} {'cap%':>6s} {'MaxE':>5s} {'Awards':>6s}")
    for _, r in max_players.head(20).iterrows():
        exp = int((r["age"] if pd.notna(r["age"]) else 25) - 18)
        print(f"  {r['player_name_norm'][:25]:25s} {int(r['season']):4d} {r['age']:4.0f} "
              f"{exp:4d} {r['cap_pct']:6.3f} {r['max_eligible_pct']:5.2f} {r['award_score_cum']:6.1f}")

    # ─── 2. XGBoost OOF predictions ───────────────────────────────
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

    # ─── 3. Tobit regression ──────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_tobit = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    censored = df["is_max_contract"].values
    upper_bounds = df["max_eligible_pct"].values

    print(f"\nFitting Tobit (right-censored at max_eligible_pct)...")
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

    # ─── 4. Compare ───────────────────────────────────────────────
    df["cap"] = df["season"].map(CAP_BY_SEASON).fillna(153_000_000)
    max_df = df[df["is_max_contract"]].copy().sort_values("cap_pct", ascending=False)

    print(f"\n{'='*95}")
    print(f"MAX CONTRACT PLAYERS: XGBoost vs Tobit latent")
    print(f"{'='*95}")
    print(f"{'Player':25s} {'Ssn':>4s} {'Age':>3s} {'Actual%':>7s} {'MaxE%':>5s} "
          f"{'XGB%':>6s} {'Tobit%':>6s} {'Act$M':>6s} {'XGB$M':>6s} {'Tob$M':>6s}")
    print("-" * 95)

    for _, r in max_df.head(30).iterrows():
        cap = r["cap"]
        print(f"{r['player_name_norm'][:25]:25s} {int(r['season']):4d} {r['age']:3.0f} "
              f"{r['cap_pct']:7.3f} {r['max_eligible_pct']:5.2f} "
              f"{r['pred_xgb']:6.3f} {r['pred_tobit_latent']:6.3f} "
              f"{r['cap_pct']*cap/1e6:6.1f} {r['pred_xgb']*cap/1e6:6.1f} {r['pred_tobit_latent']*cap/1e6:6.1f}")

    # ─── 5. Summary ───────────────────────────────────────────────
    max_mask = df["is_max_contract"]
    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"{'='*65}")

    for label, col in [("XGBoost", "pred_xgb"), ("Tobit latent", "pred_tobit_latent")]:
        m = df[max_mask]
        nm = df[~max_mask]
        bias_max = (m[col] - m[TARGET]).mean()
        mae_max = (m[col] - m[TARGET]).abs().mean()
        mae_nm = (nm[col] - nm[TARGET]).abs().mean()
        cap_avg = m["cap"].mean()
        print(f"\n  {label}:")
        print(f"    Max (n={len(m)}):     bias={bias_max:+.4f} (${bias_max*cap_avg/1e6:+.1f}M)  MAE={mae_max:.4f} (${mae_max*cap_avg/1e6:.1f}M)")
        print(f"    Non-max (n={len(nm)}): MAE={mae_nm:.4f}")

    tobit_above = (max_df["pred_tobit_latent"] > max_df["max_eligible_pct"]).sum()
    print(f"\n  Tobit latent > max_eligible: {tobit_above}/{len(max_df)} ({tobit_above/len(max_df):.0%})")

    # ─── 6. Output CSV ────────────────────────────────────────────
    out = df[["player_name_norm", "season", "age", "cap_pct",
              "max_eligible_pct", "is_max_contract",
              "pred_xgb", "pred_tobit_latent"]].copy()
    out["cap"] = df["cap"]
    out["actual_$M"] = (out["cap_pct"] * out["cap"] / 1e6).round(2)
    out["xgb_$M"] = (out["pred_xgb"] * out["cap"] / 1e6).round(2)
    out["tobit_$M"] = (out["pred_tobit_latent"] * out["cap"] / 1e6).round(2)
    out["tobit_minus_xgb_$M"] = (out["tobit_$M"] - out["xgb_$M"]).round(2)
    out = out.sort_values(["is_max_contract", "cap_pct"], ascending=[False, False])

    path = OUT_DIR / "tobit_vs_xgb_v2.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved: {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
