"""Diagnostic analysis for v3 model. Read-only — no model/feature changes."""
import sys, re, unicodedata
# reconfigure rather than rebind: replacing sys.stdout leaves the old wrapper to
# be garbage-collected, which closes the underlying buffer for anything that
# imports this module.
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from bs4 import BeautifulSoup
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from pathlib import Path

from config import CAP_BY_SEASON, OUTPUTS_DIR, CACHE_DIR
from src.model.train import (
    load_training_data, _filter_year1, _filter_rookie_scale, _filter_prorated,
    _filter_mislabeled_year1, _filter_continuations, _filter_rookie_contracts,
    FEATURE_COLS, TARGET,
)
# NOTE: evaluate_suite imports attach_signing_labels from this module, so any
# import from evaluate_suite must be deferred to avoid a circular import.
# _in_confirmation_set is imported inside main() below.

DIAG_DIR = OUTPUTS_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
    random_state=42, tree_method="hist",
)
FEATURES = FEATURE_COLS
CAP_2026 = CAP_BY_SEASON.get(2026, 153_000_000)


def prepare_data():
    # The full train.py membership chain — a partial chain here once put rows
    # the model never trains on onto the error board (Hayward 2019, Graham 2020).
    df = load_training_data()
    df = _filter_year1(df)
    df = _filter_rookie_scale(df)
    df = _filter_prorated(df)
    df = _filter_mislabeled_year1(df)
    df = _filter_continuations(df)
    df = _filter_rookie_contracts(df)
    train = df[df["season"] < 2026].copy()
    test = df[df["season"] == 2026].copy()
    X_tr = train[FEATURES].copy()
    med = X_tr.median()
    X_tr = X_tr.fillna(med).fillna(0)
    X_te = test[FEATURES].copy().fillna(med).fillna(0)
    return df, train, test, X_tr, X_te, med


# ─── 1. Residual by segment ────────────────────────────────────────
def residual_by_segment(test, y_true, y_pred):
    test = test.copy()
    test["actual"] = y_true
    test["pred"] = y_pred
    test["residual"] = y_pred - y_true
    test["abs_residual"] = test["residual"].abs()
    cap = CAP_2026

    segments = {}

    # cap_pct tier
    bins = [0, 0.05, 0.15, 0.25, 1.0]
    labels = ["0-5%", "5-15%", "15-25%", "25%+"]
    test["cap_tier"] = pd.cut(test["actual"], bins=bins, labels=labels, include_lowest=True)
    segments["cap_pct_tier"] = "cap_tier"

    # age bracket
    bins_age = [0, 24, 28, 32, 99]
    labels_age = ["21-24", "25-28", "29-32", "33+"]
    test["age_bracket"] = pd.cut(test["age"], bins=bins_age, labels=labels_age, include_lowest=True)
    segments["age_bracket"] = "age_bracket"

    # height
    bins_h = [0, 76, 80, 100]
    labels_h = ["<76\"", "76-80\"", "80\"+"]
    test["height_bin"] = pd.cut(test["height_inches"], bins=bins_h, labels=labels_h, include_lowest=True)
    segments["height"] = "height_bin"

    # cba_era
    test["cba_era_label"] = test["cba_era"].map({0: "pre-2023", 1: "post-2023"})
    segments["cba_era"] = "cba_era_label"

    rows = []
    for seg_name, col in segments.items():
        for group in test[col].dropna().unique():
            sub = test[test[col] == group]
            if len(sub) == 0:
                continue
            rows.append({
                "segment": seg_name,
                "group": str(group),
                "n": len(sub),
                "mean_residual_pct": round(sub["residual"].mean(), 4),
                "mean_residual_$M": round(sub["residual"].mean() * cap / 1e6, 2),
                "MAE_pct": round(sub["abs_residual"].mean(), 4),
                "MAE_$M": round(sub["abs_residual"].mean() * cap / 1e6, 2),
                "median_AE_$M": round(sub["abs_residual"].median() * cap / 1e6, 2),
            })

    result = pd.DataFrame(rows)
    result = result.sort_values(["segment", "group"])
    path = DIAG_DIR / "residual_by_segment.csv"
    result.to_csv(path, index=False)
    print(f"\n=== Residual by Segment (2026 holdout) ===")
    print(result.to_string(index=False))
    print(f"\nSaved: {path.name}")
    return result


# ─── 2. Biggest misses ─────────────────────────────────────────────
def biggest_misses(test, X_te, y_true, y_pred):
    out = test[["player_name", "player_name_norm", "age"]].copy()
    out["actual_cap_pct"] = y_true
    out["predicted_cap_pct"] = y_pred
    out["residual"] = y_pred - y_true
    out["abs_residual"] = out["residual"].abs()
    for f in FEATURES:
        out[f] = X_te[f].values
    out = out.sort_values("abs_residual", ascending=False).head(20)
    path = DIAG_DIR / "biggest_misses_2026.csv"
    out.to_csv(path, index=False)
    print(f"\n=== Top 20 Biggest Misses (2026 holdout) ===")
    for _, r in out.iterrows():
        pn = str(r["player_name"])[:22]
        cap = CAP_2026
        print(f"  {pn:22s} age={r['age']:.0f}  actual={r['actual_cap_pct']:.3f}"
              f"  pred={r['predicted_cap_pct']:.3f}  resid={r['residual']:+.3f}"
              f"  (${r['residual']*cap/1e6:+.1f}M)")
    print(f"\nSaved: {path.name}")


# ─── 3. SHAP dependence plots ──────────────────────────────────────
def shap_dependence(model, X_tr):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_tr)

    importances = np.abs(shap_values).mean(0)
    top5_idx = np.argsort(importances)[-5:][::-1]
    top5_names = [FEATURES[i] for i in top5_idx]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, feat in zip(axes, top5_names):
        shap.dependence_plot(feat, shap_values, X_tr, ax=ax, show=False)
        ax.set_title(feat, fontsize=11)
    plt.tight_layout()
    path = DIAG_DIR / "shap_dependence_top5.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSHAP dependence plots saved: {path.name}")

    # Also save full SHAP summary
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_tr, show=False)
    path2 = DIAG_DIR / "shap_summary.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved: {path2.name}")


# ─── 4. Predicted vs Actual scatter ────────────────────────────────
def pred_vs_actual_scatter(train, test, X_tr, X_te, y_te_pred, med):
    # CV predictions for training set
    xgb_cv = XGBRegressor(**PARAMS)
    cv = GroupKFold(n_splits=5)
    y_tr_cv = cross_val_predict(
        xgb_cv, X_tr, train[TARGET].values,
        groups=train["player_name_norm"].values, cv=cv,
    )

    cap = CAP_2026
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: CV
    ax = axes[0]
    sc = ax.scatter(
        train[TARGET].values * cap / 1e6,
        y_tr_cv * cap / 1e6,
        c=train["age"].values, cmap="RdYlBu_r", alpha=0.5, s=20, vmin=20, vmax=40,
    )
    r2_cv = r2_score(train[TARGET].values, y_tr_cv)
    mae_cv = mean_absolute_error(train[TARGET].values, y_tr_cv) * cap / 1e6
    lim = max(train[TARGET].max(), y_tr_cv.max()) * cap / 1e6 * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("Actual Salary ($M)")
    ax.set_ylabel("Predicted Salary ($M)")
    ax.set_title(f"CV (n={len(train)}, R²={r2_cv:.3f}, MAE=${mae_cv:.1f}M)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")

    # Panel 2: 2026 holdout
    ax = axes[1]
    y_te_true = test[TARGET].values
    sc2 = ax.scatter(
        y_te_true * cap / 1e6,
        y_te_pred * cap / 1e6,
        c=test["age"].values, cmap="RdYlBu_r", alpha=0.7, s=40, vmin=20, vmax=40,
    )
    r2_ho = r2_score(y_te_true, y_te_pred)
    mae_ho = mean_absolute_error(y_te_true, y_te_pred) * cap / 1e6
    lim2 = max(y_te_true.max(), y_te_pred.max()) * cap / 1e6 * 1.05
    ax.plot([0, lim2], [0, lim2], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("Actual Salary ($M)")
    ax.set_ylabel("Predicted Salary ($M)")
    ax.set_title(f"2026 Holdout (n={len(test)}, R²={r2_ho:.3f}, MAE=${mae_ho:.1f}M)")
    ax.set_xlim(0, lim2)
    ax.set_ylim(0, lim2)
    ax.set_aspect("equal")

    cbar = fig.colorbar(sc2, ax=axes, shrink=0.8, label="Age")
    plt.suptitle("Predicted vs Actual Salary", fontsize=14, y=1.02)
    path = DIAG_DIR / "pred_vs_actual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPred vs actual scatter saved: {path.name}")


# ─── 5. Residual by signing type ───────────────────────────────────
def _norm_name(name):
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    out = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", out.replace(".", "").replace("-", " ")).strip()


def _slugify(name):
    return re.sub(r"[^a-z0-9-]", "", _norm_name(name).replace(" ", "-"))


def _categorize_signing(t):
    t = str(t).lower().strip()
    if t == "bird rights":
        return "Bird Rights"
    if "early bird" in t:
        return "Early Bird"
    if "non-bird" in t:
        return "Non-Bird"
    if "cap-space" in t or "cap space" in t:
        return "Cap Space"
    if "mid-level" in t or "mle" in t:
        return "MLE"
    if "minimum" in t or "hardship" in t:
        return "Minimum"
    if "rookie" in t:
        return "Rookie Scale"
    if "sign-and-trade" in t or "extend-and-trade" in t:
        # Reclassified 2026-08-07 (was its own "Sign & Trade" bucket, excluded
        # from the Stage-3 signing offset as leakage). Unlike MLE/Minimum/BAE,
        # a sign-and-trade's dollar amount is NOT determined by the mechanism
        # itself -- these contracts range $3.6M-$37.2M in the data -- and the
        # CBA requires the ORIGINATING team to hold Bird or Early Bird rights
        # on the player for the trade to be allowed at all, so the label is an
        # eligibility fact settled before the price is, like the four types
        # already corrected. See CLAUDE.md ("Modeling Strategy") and
        # METHODOLOGY.md ("Stage 3, signing component").
        return "Bird Rights"
    return "Other"


def attach_signing_labels(df, salary_dollars=None):
    """Attach signing_cat from spotrac_signing_types.csv. Not a model feature.

    The CSV (rebuilt by scripts/refresh_spotrac.py) can legitimately hold
    several contracts for one player-season: a mid-season buyout puts two real
    deals on the same season — Westbrook 2022-23 collected supermax cash and
    then signed a minimum. The label a residual diagnostic needs is the
    contract that PRODUCED the observed salary, so when candidates compete the
    one whose AAV sits closest to the row's salary wins. Without a salary the
    newest candidate (page order) is kept.
    """
    from config import PROCESSED_DIR
    path = PROCESSED_DIR / "spotrac_signing_types.csv"
    if not path.exists():
        df = df.copy()
        df["signing_cat"] = "Unknown"
        return df
    st = pd.read_csv(path).dropna(subset=["signing_type"])
    st["signing_cat"] = st["signing_type"].map(_categorize_signing)

    key = df[["player_name_norm", "season"]].copy()
    key["_row"] = np.arange(len(key))
    key["_sal"] = (np.asarray(salary_dollars, dtype=float)
                   if salary_dollars is not None else np.nan)
    cand = key.merge(st[["player_name_norm", "season", "signing_cat", "aav"]],
                     on=["player_name_norm", "season"], how="left")
    cand["_dist"] = (cand["aav"] - cand["_sal"]).abs()
    # NaN distance (no salary given, or contract without an AAV) ranks last;
    # the stable sort then keeps the newest candidate among the unranked.
    cand["_dist"] = cand["_dist"].fillna(np.inf)
    cand = (cand.sort_values(["_row", "_dist"], kind="stable")
                .drop_duplicates("_row", keep="first")
                .set_index("_row"))

    df = df.copy()
    df["signing_cat"] = (cand["signing_cat"]
                         .reindex(np.arange(len(df))).fillna("Unknown").values)
    return df


def _load_signing_type_labels(df):
    """Back-compat wrapper: salary-aware label attach using the salary column."""
    sal = df["salary"] if "salary" in df.columns else None
    return attach_signing_labels(df, salary_dollars=sal)


def residual_by_signing_type(df, y_true, y_pred, is_confirmation=None):
    """Slice residuals by Spotrac signing type (diagnostic label, not feature).

    When is_confirmation is provided, prints both pooled (all rows, reporting)
    and selection-only (decision-grade) readings -- ISSUES #20a.
    """
    if "signing_cat" not in df.columns:
        print("\nNo signing_cat in data, skipping signing type residuals.")
        return

    df = df.copy()
    df["actual"] = y_true
    df["pred"] = y_pred
    df["residual"] = y_pred - y_true
    df["cap"] = df["season"].map(CAP_BY_SEASON).fillna(CAP_2026)
    df["residual_dollar"] = df["residual"] * df["cap"]

    order = ["Bird Rights", "Cap Space", "Early Bird",
             "MLE", "Non-Bird", "Minimum", "Rookie Scale", "Other", "Unknown"]

    # ISSUES #20a: report both pooled (all rows) and selection-only readings.
    # The selection-only numbers are decision-grade; the pooled ones are context.
    splits = [("all", "reporting", None)]
    if is_confirmation is not None:
        sel_mask = ~np.asarray(is_confirmation, dtype=bool)
        splits.append(("selection", "decision-grade", sel_mask))

    for tag, label, mask in splits:
        sub_df = df[mask] if mask is not None else df
        rows = []
        for cat in order:
            sub = sub_df[sub_df["signing_cat"] == cat]
            if len(sub) < 3:
                continue
            bias = sub["residual"].mean()
            mae = sub["residual"].abs().mean()
            bias_d = sub["residual_dollar"].mean()
            mae_d = sub["residual_dollar"].abs().mean()
            rows.append({
                "signing_type": cat,
                "n": len(sub),
                "mean_cap_pct": round(sub["actual"].mean(), 4),
                "mean_pred": round(sub["pred"].mean(), 4),
                "bias_pct": round(bias, 4),
                "bias_$M": round(bias_d / 1e6, 2),
                "MAE_pct": round(mae, 4),
                "MAE_$M": round(mae_d / 1e6, 2),
            })

        result = pd.DataFrame(rows)
        suffix = f"_{tag}" if tag != "all" else ""
        path = DIAG_DIR / f"residual_by_signing_type{suffix}.csv"
        result.to_csv(path, index=False)

        print(f"\n=== Residual by Signing Type (OOF) [{label}] ===")
        print(f"{'Type':15s} {'N':>4s} {'Avg%':>7s} {'Pred%':>7s} {'Bias':>7s} {'Bias$M':>7s} {'MAE%':>7s} {'MAE$M':>7s}")
        print("-" * 70)
        for _, r in result.iterrows():
            print(f"{r['signing_type']:15s} {r['n']:4d} {r['mean_cap_pct']:7.4f} {r['mean_pred']:7.4f} "
                  f"{r['bias_pct']:+7.4f} {r['bias_$M']:+7.1f} {r['MAE_pct']:7.4f} {r['MAE_$M']:7.1f}")
        print(f"\nSaved: {path.name}")
    return result


# ─── Main ───────────────────────────────────────────────────────────
def main():
    df, train, test, X_tr, X_te, med = prepare_data()

    xgb = XGBRegressor(**PARAMS)
    xgb.fit(X_tr, train[TARGET].values)
    y_pred = xgb.predict(X_te)
    y_true = test[TARGET].values

    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"2026 Holdout R²={r2_score(y_true, y_pred):.4f}, "
          f"MAE=${mean_absolute_error(y_true, y_pred)*CAP_2026/1e6:.1f}M")

    residual_by_segment(test, y_true, y_pred)
    biggest_misses(test, X_te, y_true, y_pred)
    shap_dependence(xgb, X_tr)
    pred_vs_actual_scatter(train, test, X_tr, X_te, y_pred, med)

    # OOF signing type residuals (full dataset, not just 2026 holdout)
    df_full = load_training_data()
    df_full = _filter_year1(df_full)
    df_full = _filter_rookie_scale(df_full)
    df_full = _filter_prorated(df_full)
    df_full = _filter_mislabeled_year1(df_full)
    df_full = _filter_continuations(df_full)
    df_full = _filter_rookie_contracts(df_full)
    # ISSUES #20a: mark the confirmation split so signing-type bias is reported
    # separately for selection (decision-grade) and confirmation (canary) rows.
    # Deferred import: evaluate_suite imports attach_signing_labels from this
    # module, so the import must happen after both modules are fully loaded.
    from src.model.evaluate_suite import _in_confirmation_set
    df_full["is_confirmation"] = df_full["player_name_norm"].map(_in_confirmation_set)
    df_full = _load_signing_type_labels(df_full)
    avail = [f for f in FEATURES if f in df_full.columns]
    X_full = df_full[avail].fillna(df_full[avail].median()).fillna(0)
    oof_pred = np.full(len(df_full), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr_i, va_i in gkf.split(X_full, df_full[TARGET].values, df_full["player_name_norm"].values):
        m = XGBRegressor(**PARAMS)
        m.fit(X_full.iloc[tr_i], df_full[TARGET].values[tr_i])
        oof_pred[va_i] = m.predict(X_full.iloc[va_i])
    residual_by_signing_type(df_full, df_full[TARGET].values, oof_pred,
                             is_confirmation=df_full["is_confirmation"].values)

    print(f"\nAll diagnostics saved to {DIAG_DIR}")


if __name__ == "__main__":
    main()
