"""The four-layer evaluation protocol.

Each layer answers a different question and they must not be mixed:

  A  selection      pooled GroupKFold CV. Estimates the market's pricing
                    function, so training on a later season to score an earlier
                    one is legitimate — the estimand is structural, not a
                    forecast. Player grouping blocks the leakage that does
                    matter (one player's contracts are highly correlated).
                    Use PAIRED deltas for every accept/reject decision.

  B  forecasting    rolling-origin, train on every season < T, score season T,
                    for T in 2024-2026. This is what predict.py actually does.
                    Origins before 2024 are excluded: their training sets are a
                    third to a fifth of the current one, so they measure data
                    scarcity rather than the model, and they sit in the pre-2023
                    CBA regime.

  C  integrity      calibration and per-segment behaviour, to catch a change
                    that improves the average while damaging a segment.
                    Calibration is always binned by PREDICTED value. Binning by
                    the actual target produces a monotone bias gradient even for
                    a perfectly calibrated model (regression to the mean).

  D  guards         comparability rules that are easy to violate silently:
                    a fixed evaluation set when the training filter changes,
                    a baseline ladder so absolute R2 is not mistaken for skill,
                    and a locked confirmation split held out of selection.

Run directly for the full report on the current champion:

    python src/model/evaluate_suite.py
"""

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from config import CAP_BY_SEASON, OUTPUTS_DIR, PROCESSED_DIR
from src.model.train import (
    load_training_data, _filter_year1, _filter_rookie_scale, _compute_max_eligible,
    _prepare_Xy, _make_tobit_obj, _XGB_BASE, FEATURE_COLS, TARGET,
)
# reuse the canonical categoriser so C2 segments match scripts/diagnostics.py
from scripts.diagnostics import _categorize_signing

N_SPLITS = 5
DEFAULT_SEEDS = tuple(range(10))
FORWARD_ORIGINS = (2024, 2025, 2026)
CONFIRMATION_PCT = 15  # share of players locked away from model selection

BASELINE_LADDER = {
    "mpg only": ["mpg"],
    "mpg + prev_cap_pct": ["mpg", "prev_cap_pct"],
    "mpg + prev + age + darko": ["mpg", "prev_cap_pct", "age", "darko_dpm_z"],
}


# ---------------------------------------------------------------------------
# Fitters — a fitter trains on one slice and returns predictions for another
# ---------------------------------------------------------------------------

def baseline_fitter(train: pd.DataFrame, test: pd.DataFrame,
                    features: list[str], seed: int) -> np.ndarray:
    """Plain XGBoost on the champion hyperparameters."""
    model = XGBRegressor(**{**_XGB_BASE, "random_state": seed})
    model.fit(train[features], train[TARGET].values)
    return model.predict(test[features])


def make_grabit_fitter(sigma: float = 0.02, gate_frac: float = 0.55):
    """Grabit v3: censored-normal loss on gated max rows, then the CBA cap.

    The gate needs a baseline prediction, which is fit inside the training slice
    so nothing from the scored slice leaks in.
    """
    def fitter(train, test, features, seed):
        y_tr = train[TARGET].values
        max_elig_tr = train["max_eligible_pct"].values
        is_max_tr = train["is_max_contract"].values.astype(bool)

        base = XGBRegressor(**{**_XGB_BASE, "random_state": seed})
        base.fit(train[features], y_tr)
        gate = is_max_tr & (base.predict(train[features]) >= gate_frac * max_elig_tr)

        model = XGBRegressor(**{**_XGB_BASE, "random_state": seed,
                                "objective": _make_tobit_obj(gate, sigma),
                                "base_score": float(y_tr.mean())})
        model.fit(train[features], y_tr)
        return np.minimum(model.predict(test[features]), test["max_eligible_pct"].values)

    return fitter


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_evaluation_frame(min_salary_m: float | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Training rows with features imputed, plus the usable feature list.

    Args:
        min_salary_m: drop rows below this annual salary in $M. Prorated
            partial-season deals (10-day contracts, mid-season signings) are not
            annual contract values; ~17% of rows sit below $1.2M with a median
            of $0.26M. Leave as None to keep every row.
    """
    df = _compute_max_eligible(_filter_rookie_scale(_filter_year1(load_training_data())))
    df = df.reset_index(drop=True)
    df["cap"] = df["season"].map(CAP_BY_SEASON)
    df["salary_m"] = df[TARGET] * df["cap"] / 1e6

    if min_salary_m is not None:
        before = len(df)
        df = df[df["salary_m"] >= min_salary_m].reset_index(drop=True)
        print(f"Prorated filter (< ${min_salary_m}M): dropped {before - len(df)} rows "
              f"({len(df)} remain)")

    _, _, _, features = _prepare_Xy(df)
    df[features] = df[features].fillna(df[features].median()).fillna(0)
    df["is_confirmation"] = df["player_name_norm"].map(_in_confirmation_set)
    df["signing_cat"] = _attach_signing_category(df)
    return df, features


def _attach_signing_category(df: pd.DataFrame) -> pd.Series:
    """Spotrac signing mechanism per row, for C2 only.

    This is a diagnostic label and never a feature: the mechanism is partly
    determined by the contract itself. Coverage is ~43%; the rest is 'Unknown'.
    """
    path = PROCESSED_DIR / "spotrac_signing_types.csv"
    if not path.exists():
        return pd.Series(["Unknown"] * len(df), index=df.index)
    st = pd.read_csv(path)[["player_name_norm", "season", "signing_type"]]
    st = st.dropna(subset=["signing_type"]).drop_duplicates(["player_name_norm", "season"])
    st["signing_cat"] = st["signing_type"].map(_categorize_signing)
    merged = df[["player_name_norm", "season"]].merge(
        st[["player_name_norm", "season", "signing_cat"]],
        on=["player_name_norm", "season"], how="left")
    return merged["signing_cat"].fillna("Unknown").values


def _in_confirmation_set(player_name_norm: str) -> bool:
    """Deterministic per-player split, stable across runs and machines.

    Hashing the name rather than storing a file means the split cannot drift or
    be lost, and a new player lands on a fixed side the first time they appear.
    """
    digest = hashlib.md5(str(player_name_norm).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100 < CONFIRMATION_PCT


# ---------------------------------------------------------------------------
# Prediction engines
# ---------------------------------------------------------------------------

def oof_groupkfold(df: pd.DataFrame, features: list[str], fitter,
                   seeds=DEFAULT_SEEDS) -> tuple[np.ndarray, np.ndarray]:
    """Seed-averaged out-of-fold predictions, plus per-(fold, seed) R2.

    Returns:
        (oof predictions, matrix of shape (n_folds, n_seeds) of fold R2)
    """
    y = df[TARGET].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, y, df["player_name_norm"].values))

    acc = np.zeros(len(df))
    fold_r2 = np.zeros((len(folds), len(seeds)))
    for si, seed in enumerate(seeds):
        oof = np.full(len(df), np.nan)
        for fi, (tr, va) in enumerate(folds):
            pred = fitter(df.iloc[tr], df.iloc[va], features, seed)
            oof[va] = pred
            fold_r2[fi, si] = r2_score(y[va], pred)
        acc += oof
    return acc / len(seeds), fold_r2


def rolling_forward(df: pd.DataFrame, features: list[str], fitter,
                    seeds=DEFAULT_SEEDS, origins=FORWARD_ORIGINS) -> np.ndarray:
    """Predictions for each origin season, trained only on strictly earlier ones."""
    preds = np.full(len(df), np.nan)
    season = df["season"].values
    for T in origins:
        test_mask, train_mask = season == T, season < T
        if test_mask.sum() < 10 or train_mask.sum() < 200:
            continue
        acc = np.zeros(test_mask.sum())
        for seed in seeds:
            acc += fitter(df[train_mask], df[test_mask], features, seed)
        preds[test_mask] = acc / len(seeds)
    return preds


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_r2_ci(y: np.ndarray, pred: np.ndarray, n: int = 4000,
                    seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for R2. Small holdouts need this."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if np.var(y[idx]) > 0:
            draws.append(r2_score(y[idx], pred[idx]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_delta(fold_r2_a: np.ndarray, fold_r2_b: np.ndarray) -> dict:
    """Paired fold-level comparison of two variants scored on identical folds.

    Fold-to-fold variance (sd ~0.031) dwarfs seed variance (sd ~0.0008), so an
    unpaired comparison of two reported means throws away nearly all the power.
    Pairing cancels the fold effect; the fold is the unit of replication.
    """
    per_fold = fold_r2_b.mean(axis=1) - fold_r2_a.mean(axis=1)
    mean = float(per_fold.mean())
    se = float(per_fold.std(ddof=1) / np.sqrt(len(per_fold)))
    return {"delta": mean, "se": se, "t": mean / se if se > 0 else float("nan"),
            "per_fold": [round(v, 5) for v in per_fold]}


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

@dataclass
class SuiteResult:
    """Everything one model variant scored, ready to serialise."""
    name: str
    metrics: dict = field(default_factory=dict)
    oof: np.ndarray | None = None
    forward: np.ndarray | None = None
    fold_r2: np.ndarray | None = None


def _dollars(df, pred, mask=None):
    """MAE and bias in $M over an optional subset."""
    m = np.ones(len(df), bool) if mask is None else mask
    err = (pred[m] - df[TARGET].values[m]) * df["cap"].values[m] / 1e6
    return float(np.abs(err).mean()), float(err.mean())


def layer_a(df, pred_oof, fold_r2) -> dict:
    """A1 pooled CV, A2 the 2024-26 subset of the same predictions."""
    y = df[TARGET].values
    recent = df["season"].values >= 2024
    mae, bias = _dollars(df, pred_oof)
    mae_r, bias_r = _dollars(df, pred_oof, recent)
    return {
        "A1_cv_r2": float(r2_score(y, pred_oof)),
        "A1_cv_mae_m": mae, "A1_cv_bias_m": bias, "A1_n": int(len(df)),
        "A1_fold_r2_sd": float(fold_r2.mean(axis=1).std(ddof=1)),
        "A1_seed_r2_sd": float(fold_r2.mean(axis=0).std(ddof=1)),
        "A2_cv_r2_2024_26": float(r2_score(y[recent], pred_oof[recent])),
        "A2_mae_m": mae_r, "A2_bias_m": bias_r, "A2_n": int(recent.sum()),
    }


def layer_b(df, pred_fwd) -> dict:
    """B1 rolling-forward over 2024-26, with a CI and per-origin detail."""
    y = df[TARGET].values
    scored = ~np.isnan(pred_fwd)
    if scored.sum() < 20:
        return {"B1_error": "not enough forward-scored rows"}
    lo, hi = bootstrap_r2_ci(y[scored], pred_fwd[scored])
    mae, bias = _dollars(df, pred_fwd, scored)
    out = {"B1_forward_r2": float(r2_score(y[scored], pred_fwd[scored])),
           "B1_ci95": [lo, hi], "B1_mae_m": mae, "B1_bias_m": bias,
           "B1_n": int(scored.sum()), "B1_by_origin": {}}
    for T in FORWARD_ORIGINS:
        m = scored & (df["season"].values == T)
        if m.sum() >= 10:
            out["B1_by_origin"][int(T)] = {
                "n": int(m.sum()), "r2": float(r2_score(y[m], pred_fwd[m])),
                "mae_m": _dollars(df, pred_fwd, m)[0]}
    return out


def layer_c(df, pred_oof) -> dict:
    """C1 calibration, C2 per-segment integrity, C3 ranking quality."""
    y = df[TARGET].values
    slope, intercept = np.polyfit(pred_oof, y, 1)

    # C1 — always bin by the PREDICTION, never by the target
    edges = [0, 0.02, 0.04, 0.08, 0.15, 0.25, 1.0]
    labels = ["<2%", "2-4%", "4-8%", "8-15%", "15-25%", "25%+"]
    band = pd.cut(pred_oof, edges, labels=labels, include_lowest=True)
    by_pred = {}
    for lab in labels:
        m = np.asarray(band == lab)
        if m.sum() >= 5:
            mae, bias = _dollars(df, pred_oof, m)
            by_pred[lab] = {"n": int(m.sum()), "bias_m": bias, "mae_m": mae}

    # C2 — mechanism within predicted band, where the label exists
    by_mech = {}
    if "signing_cat" in df.columns:
        for cat, sub in df.groupby("signing_cat"):
            if len(sub) >= 10:
                m = (df["signing_cat"] == cat).values
                mae, bias = _dollars(df, pred_oof, m)
                by_mech[str(cat)] = {"n": int(m.sum()), "bias_m": bias, "mae_m": mae}

    # C3 — ranking, which is what the over/underpaid use case actually needs
    resid = (pred_oof - y) * df["cap"].values / 1e6
    top = np.argsort(resid)[::-1][:20]
    bottom = np.argsort(resid)[:20]
    return {
        "C1_calibration_slope": float(slope), "C1_calibration_intercept": float(intercept),
        "C1_bias_by_predicted_band": by_pred,
        "C2_by_signing_mechanism": by_mech,
        "C3_spearman": float(spearmanr(y, pred_oof).statistic),
        "C3_top20_overpredicted_mean_m": float(resid[top].mean()),
        "C3_top20_underpredicted_mean_m": float(resid[bottom].mean()),
    }


def layer_d(df, features, pred_oof, seeds=DEFAULT_SEEDS) -> dict:
    """D2 baseline ladder and D3 the locked confirmation split."""
    y = df[TARGET].values
    ladder = {"predict the mean": 0.0}
    for name, cols in BASELINE_LADDER.items():
        cols = [c for c in cols if c in features]
        if cols:
            oof, _ = oof_groupkfold(df, cols, baseline_fitter, seeds=seeds[:3])
            ladder[name] = float(r2_score(y, oof))
    ladder["full feature set"] = float(r2_score(y, pred_oof))

    conf = df["is_confirmation"].values
    out = {"D2_baseline_ladder": ladder,
           "D2_lift_over_mpg_only": ladder["full feature set"] - ladder.get("mpg only", 0.0),
           "D3_confirmation_n": int(conf.sum()),
           "D3_selection_n": int((~conf).sum())}
    if conf.sum() >= 30:
        out["D3_confirmation_r2"] = float(r2_score(y[conf], pred_oof[conf]))
        out["D3_selection_r2"] = float(r2_score(y[~conf], pred_oof[~conf]))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_suite(df: pd.DataFrame, features: list[str], fitter, name: str,
              seeds=DEFAULT_SEEDS) -> SuiteResult:
    """Score one variant through all four layers."""
    print(f"\nScoring '{name}' over {len(seeds)} seeds...")
    oof, fold_r2 = oof_groupkfold(df, features, fitter, seeds)
    fwd = rolling_forward(df, features, fitter, seeds)

    res = SuiteResult(name=name, oof=oof, forward=fwd, fold_r2=fold_r2)
    res.metrics.update(layer_a(df, oof, fold_r2))
    res.metrics.update(layer_b(df, fwd))
    res.metrics.update(layer_c(df, oof))
    res.metrics.update(layer_d(df, features, oof, seeds))
    return res


def print_report(df: pd.DataFrame, res: SuiteResult):
    """Human-readable rendering of one SuiteResult."""
    m = res.metrics
    line = "=" * 74
    print(f"\n{line}\n  {res.name}\n{line}")

    print("\n  A — selection (pooled GroupKFold CV)")
    print(f"    A1  CV R2            {m['A1_cv_r2']:.4f}   "
          f"MAE ${m['A1_cv_mae_m']:.2f}M   bias ${m['A1_cv_bias_m']:+.2f}M   n={m['A1_n']}")
    print(f"    A2  CV R2 2024-26    {m['A2_cv_r2_2024_26']:.4f}   "
          f"MAE ${m['A2_mae_m']:.2f}M   bias ${m['A2_bias_m']:+.2f}M   n={m['A2_n']}")
    print(f"        noise: fold sd {m['A1_fold_r2_sd']:.4f}, seed sd {m['A1_seed_r2_sd']:.4f}"
          "  <- report deltas PAIRED by fold")

    print("\n  B — forecasting (rolling-origin, train on every season < T)")
    if "B1_error" in m:
        print(f"    {m['B1_error']}")
    else:
        ci = m["B1_ci95"]
        print(f"    B1  forward R2       {m['B1_forward_r2']:.4f}   "
              f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]   MAE ${m['B1_mae_m']:.2f}M   n={m['B1_n']}")
        for T, d in m["B1_by_origin"].items():
            print(f"          origin {T}     R2 {d['r2']:.4f}   "
                  f"MAE ${d['mae_m']:.2f}M   n={d['n']}")
        print(f"        cost of the forecasting setup vs A2: "
              f"{m['B1_forward_r2'] - m['A2_cv_r2_2024_26']:+.4f}")

    print("\n  C — integrity")
    print(f"    C1  calibration slope {m['C1_calibration_slope']:.3f}  "
          f"intercept {m['C1_calibration_intercept']:+.4f}   (1.0 / 0.0 is calibrated)")
    print(f"        bias by PREDICTED band (never by actual):")
    for band, d in m["C1_bias_by_predicted_band"].items():
        print(f"          {band:8s} n={d['n']:4d}  bias ${d['bias_m']:+6.2f}M  "
              f"MAE ${d['mae_m']:5.2f}M")
    if m["C2_by_signing_mechanism"]:
        print(f"        bias by signing mechanism:")
        for cat, d in sorted(m["C2_by_signing_mechanism"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"          {cat:14s} n={d['n']:4d}  bias ${d['bias_m']:+6.2f}M  "
                  f"MAE ${d['mae_m']:5.2f}M")
    print(f"    C3  Spearman {m['C3_spearman']:.4f}   "
          f"top-20 over ${m['C3_top20_overpredicted_mean_m']:+.1f}M / "
          f"under ${m['C3_top20_underpredicted_mean_m']:+.1f}M")

    print("\n  D — guards")
    for k, v in m["D2_baseline_ladder"].items():
        print(f"    D2  {k:28s} R2 {v:.4f}")
    print(f"        lift of the full set over mpg alone: {m['D2_lift_over_mpg_only']:+.4f}")
    if "D3_confirmation_r2" in m:
        print(f"    D3  selection pool  R2 {m['D3_selection_r2']:.4f}  (n={m['D3_selection_n']})")
        print(f"        LOCKED confirm  R2 {m['D3_confirmation_r2']:.4f}  (n={m['D3_confirmation_n']})"
              "  <- open only at a version bump")


def main():
    df, features = load_evaluation_frame()
    print(f"Loaded {len(df)} rows, {len(features)} features, "
          f"seasons {df['season'].min()}-{df['season'].max()}")

    champion = run_suite(df, features, make_grabit_fitter(), "Grabit v3 (champion)")
    print_report(df, champion)

    challenger = run_suite(df, features, baseline_fitter, "Baseline XGBoost")
    print_report(df, challenger)

    delta = paired_delta(challenger.fold_r2, champion.fold_r2)
    print(f"\n{'='*74}\n  PAIRED comparison: Grabit v3 minus Baseline XGBoost\n{'='*74}")
    print(f"    delta CV R2  {delta['delta']:+.4f}  +/- {delta['se']:.4f} (SE)   "
          f"t = {delta['t']:+.2f}")
    print(f"    per fold     {delta['per_fold']}")
    print("    accept when t > 2, A2 moves the same way, and no C2 segment "
          "regresses by more than $0.3M")

    out_dir = OUTPUTS_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"champion": champion.metrics, "challenger": challenger.metrics,
               "paired_delta": delta,
               # fold x seed R2 matrices — the reference every future paired
               # comparison diffs against (same folds, same seeds, per-fold)
               "fold_r2": {"champion": champion.fold_r2.tolist(),
                           "challenger": challenger.fold_r2.tolist()},
               "seeds": list(DEFAULT_SEEDS), "n_splits": N_SPLITS}
    with open(out_dir / "evaluation_suite.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    ref = df[["player_name_norm", "season", TARGET, "salary_m",
              "signing_cat", "is_confirmation"]].copy()
    ref["oof_champion"] = champion.oof
    ref["fwd_champion"] = champion.forward
    ref["oof_challenger"] = challenger.oof
    ref["fwd_challenger"] = challenger.forward
    ref.to_csv(out_dir / "oof_reference.csv", index=False)
    print(f"\nSaved {out_dir / 'evaluation_suite.json'}")
    print(f"Saved {out_dir / 'oof_reference.csv'} ({len(ref)} rows)")


if __name__ == "__main__":
    main()
