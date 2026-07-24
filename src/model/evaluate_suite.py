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
    load_training_data, _filter_year1, _filter_rookie_scale, _filter_prorated,
    _filter_mislabeled_year1, _filter_continuations, _compute_max_eligible,
    _compute_floor, _prepare_Xy, _make_tobit_obj, _XGB_BASE, FEATURE_COLS, TARGET,
)
# reuse the canonical label logic so C2 segments match scripts/diagnostics.py
from scripts.diagnostics import attach_signing_labels

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


def make_grabit_fitter(sigma: float = 0.02, gate_frac: float = 0.55,
                       floor_gate_k: float = 2.0,
                       sigma_left: float | None = None):
    """Grabit v4: two-sided censored-normal loss, then both CBA bounds.

    Right side censors gated max rows (observation floors the latent); left
    side censors gated at-floor minimum rows (the league floor props their pay
    up, so the observation CEILS the latent). Both gates need a baseline
    prediction, fit inside the training slice so nothing from the scored slice
    leaks in; both mirror the albatross rule — the model must corroborate that
    the bound binds. Stage 2 clips into [floor_pct, max_eligible_pct].

    floor_gate_k=0 disables the left side (reproduces Grabit v3).
    sigma_left=None ties the left side's sigma to the right's (shipped default).
    """
    def fitter(train, test, features, seed):
        y_tr = train[TARGET].values
        max_elig_tr = train["max_eligible_pct"].values
        is_max_tr = train["is_max_contract"].values.astype(bool)

        base = XGBRegressor(**{**_XGB_BASE, "random_state": seed})
        base.fit(train[features], y_tr)
        bp = base.predict(train[features])
        gate = is_max_tr & (bp >= gate_frac * max_elig_tr)
        if floor_gate_k > 0 and "is_at_floor" in train.columns:
            gate_l = train["is_at_floor"].values & (bp <= floor_gate_k * y_tr)
        else:
            gate_l = np.zeros(len(train), bool)

        model = XGBRegressor(**{**_XGB_BASE, "random_state": seed,
                                "objective": _make_tobit_obj(
                                    gate, sigma, left_mask=gate_l,
                                    sigma_left=sigma_left),
                                "base_score": float(y_tr.mean())})
        model.fit(train[features], y_tr)
        latent = model.predict(test[features])
        lo = (test["floor_pct"].values if "floor_pct" in test.columns
              else np.zeros(len(test)))
        return np.clip(latent, lo, test["max_eligible_pct"].values)

    return fitter


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_evaluation_frame(keep_prorated: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Training rows with features imputed, plus the usable feature list.

    Applies the same filter chain as train.py so the suite scores what the model
    is actually fit on.

    Args:
        keep_prorated: skip the prorated-salary filter, retaining partial-season
            rows. Only for reproducing the pre-v7.2x row set — R2 is not
            comparable across different row sets, so a comparison against the
            filtered frame has to fix the evaluation set (see D1).
    """
    df = _filter_rookie_scale(_filter_year1(load_training_data()))
    if not keep_prorated:
        df = _filter_prorated(df)
    df = _compute_max_eligible(df)
    df = _filter_mislabeled_year1(df)
    df = _filter_continuations(df).reset_index(drop=True)
    df["cap"] = df["season"].map(CAP_BY_SEASON)
    df["salary_m"] = df[TARGET] * df["cap"] / 1e6

    _, _, _, features = _prepare_Xy(df)
    df[features] = df[features].fillna(df[features].median()).fillna(0)
    df["is_confirmation"] = df["player_name_norm"].map(_in_confirmation_set)
    # Diagnostic label only, never a feature. Salary-aware: when a mid-season
    # buyout puts two contracts on one season, the one that produced this
    # row's salary wins (see attach_signing_labels).
    df = attach_signing_labels(df, salary_dollars=df[TARGET] * df["cap"])
    # CBA floor bound (is_at_floor + floor_pct) — mirrors max_eligible above
    df = _compute_floor(df)
    return df, features


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
                   seeds=DEFAULT_SEEDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed-averaged out-of-fold predictions, plus per-(fold, seed) R2.

    Returns:
        (oof predictions,
         fold R2 matrix over ALL validation rows        — reporting,
         fold R2 matrix over SELECTION-POOL rows only   — decisions)

    The second matrix exists because the accept/reject decision must not read
    the confirmation split at all: the formal audit (2026-07-23) found the
    changes adopted between v7.2x and v7.5x helped selection rows while
    hurting confirmation rows (difference-in-differences +5.6e-5, cluster
    CI excluding zero) — the adoption process had started fitting the rows it
    was watching. Confirmation rows still appear in training folds (they are
    other folds' training data); they are only excluded from the metric that
    decides.
    """
    y = df[TARGET].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, y, df["player_name_norm"].values))
    sel = ~df["is_confirmation"].values if "is_confirmation" in df.columns \
        else np.ones(len(df), bool)

    acc = np.zeros(len(df))
    fold_r2 = np.zeros((len(folds), len(seeds)))
    fold_r2_sel = np.zeros((len(folds), len(seeds)))
    for si, seed in enumerate(seeds):
        oof = np.full(len(df), np.nan)
        for fi, (tr, va) in enumerate(folds):
            pred = fitter(df.iloc[tr], df.iloc[va], features, seed)
            oof[va] = pred
            fold_r2[fi, si] = r2_score(y[va], pred)
            vs = sel[va]
            fold_r2_sel[fi, si] = (r2_score(y[va][vs], pred[vs])
                                   if vs.sum() > 10 else np.nan)
        acc += oof
    return acc / len(seeds), fold_r2, fold_r2_sel


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
    fold_r2_sel: np.ndarray | None = None


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


def grabit_zone(df, champion_oof, challenger_oof) -> dict:
    """Zone-local scorecard for the rows Grabit exists for.

    Grabit censors only rows paid >= 90% of their own CBA ceiling — about 5% of
    the data — so its effect on any pooled metric is diluted ~20x and a pooled
    t-test mistakes that dilution for weakness. The keep/drop decision for
    Grabit reads this zone alone: drop it when delta_mae turns positive.
    """
    mask = (df[TARGET] >= 0.90 * df["max_eligible_pct"]).values
    cap_m = df["cap"].values / 1e6
    err_ch = (champion_oof - df[TARGET].values) * cap_m
    err_xg = (challenger_oof - df[TARGET].values) * cap_m
    d_abs = np.abs(err_ch[mask]) - np.abs(err_xg[mask])
    return {
        "n": int(mask.sum()),
        "mae_grabit": float(np.abs(err_ch[mask]).mean()),
        "mae_baseline": float(np.abs(err_xg[mask]).mean()),
        "delta_mae": float(d_abs.mean()),
        "bias_grabit": float(err_ch[mask].mean()),
        "bias_baseline": float(err_xg[mask].mean()),
        "rows_better": int((d_abs < -0.005).sum()),
        "rows_worse": int((d_abs > 0.005).sum()),
    }


def floor_zone(df, champion_oof, challenger_oof) -> dict:
    """Zone-local scorecard for the left-censored side: rows at the CBA floor.

    Same logic as grabit_zone at the other bound. The champion overpredicted
    these rows by +$2.38M with 93% overshot before the left side existed; keep
    the floor branch while delta_mae is negative, drop it if it turns positive.
    """
    mask = df["is_at_floor"].values
    cap_m = df["cap"].values / 1e6
    err_ch = (champion_oof - df[TARGET].values) * cap_m
    err_xg = (challenger_oof - df[TARGET].values) * cap_m
    d_abs = np.abs(err_ch[mask]) - np.abs(err_xg[mask])
    return {
        "n": int(mask.sum()),
        "mae_grabit": float(np.abs(err_ch[mask]).mean()),
        "mae_baseline": float(np.abs(err_xg[mask]).mean()),
        "delta_mae": float(d_abs.mean()),
        "bias_grabit": float(err_ch[mask].mean()),
        "bias_baseline": float(err_xg[mask].mean()),
        "rows_better": int((d_abs < -0.005).sum()),
        "rows_worse": int((d_abs > 0.005).sum()),
    }


def layer_d(df, features, pred_oof, seeds=DEFAULT_SEEDS) -> dict:
    """D2 baseline ladder and D3 the locked confirmation split."""
    y = df[TARGET].values
    ladder = {"predict the mean": 0.0}
    for name, cols in BASELINE_LADDER.items():
        cols = [c for c in cols if c in features]
        if cols:
            oof, _, _ = oof_groupkfold(df, cols, baseline_fitter, seeds=seeds[:3])
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
    oof, fold_r2, fold_r2_sel = oof_groupkfold(df, features, fitter, seeds)
    fwd = rolling_forward(df, features, fitter, seeds)

    res = SuiteResult(name=name, oof=oof, forward=fwd, fold_r2=fold_r2,
                      fold_r2_sel=fold_r2_sel)
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
    delta_sel = paired_delta(challenger.fold_r2_sel, champion.fold_r2_sel)
    zone = grabit_zone(df, champion.oof, challenger.oof)
    fzone = floor_zone(df, champion.oof, challenger.oof)
    print(f"\n{'='*74}\n  PAIRED comparison: Grabit v3 minus Baseline XGBoost\n{'='*74}")
    print(f"    DECISION delta (selection pool)  {delta_sel['delta']:+.4f}  "
          f"+/- {delta_sel['se']:.4f} (SE)   t = {delta_sel['t']:+.2f}")
    print(f"    pooled delta (context only)      {delta['delta']:+.4f}  "
          f"+/- {delta['se']:.4f} (SE)   t = {delta['t']:+.2f}")
    print(f"    per fold (selection)  {delta_sel['per_fold']}")
    print(f"\n    Grabit zone (rows paid >= 90% of their own ceiling, n={zone['n']}):")
    print(f"      MAE  ${zone['mae_baseline']:.2f}M -> ${zone['mae_grabit']:.2f}M  "
          f"({zone['delta_mae']:+.2f})   rows better/worse {zone['rows_better']}/{zone['rows_worse']}")
    print(f"      bias ${zone['bias_baseline']:+.2f}M -> ${zone['bias_grabit']:+.2f}M")
    print(f"\n    Floor zone (rows pinned at the CBA minimum, n={fzone['n']}):")
    print(f"      MAE  ${fzone['mae_baseline']:.2f}M -> ${fzone['mae_grabit']:.2f}M  "
          f"({fzone['delta_mae']:+.2f})   rows better/worse "
          f"{fzone['rows_better']}/{fzone['rows_worse']}")
    print(f"      bias ${fzone['bias_baseline']:+.2f}M -> ${fzone['bias_grabit']:+.2f}M")
    print("    Grabit is a targeted intervention on the ~30% of rows at a CBA bound;")
    print("    judging it on the pooled delta mistakes dilution for weakness. Keep")
    print("    each side while its zone MAE delta is negative; drop the side whose")
    print("    zone turns positive.")
    print("\n    For CHALLENGER changes (features, filters, hyperparameters):")
    print("    accept when SELECTION-POOL paired t > 2, A2 moves the same way, and")
    print("    no C2 segment regresses by more than $0.3M. The confirmation split")
    print("    is a canary only — the 2026-07-23 audit caught the pooled metric")
    print("    fitting the rows it was watching (diff-in-diff +5.6e-5, CI > 0).")
    print("    A feature whose values (or missingness) align with seasons must")
    print("    also beat a pure season-dummy control: supply_samepos passed t>2")
    print("    AND the forward veto, yet an is2019 flag with zero market content")
    print("    recovered 70% of its gain — the B1 veto cannot see a feature that")
    print("    absorbs a season offset on the TRAINING side.")

    out_dir = OUTPUTS_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    # keep one generation of history: a rerun otherwise destroys the previous
    # champion's per-row OOF, which paired comparisons against the old state
    # need (this bit us — the v7.7x reference was clobbered mid-analysis)
    for name in ("evaluation_suite.json", "oof_reference.csv"):
        prev = out_dir / name
        if prev.exists():
            stem, dot, ext = name.partition(".")
            prev.replace(out_dir / f"{stem}_prev{dot}{ext}")
    payload = {"champion": champion.metrics, "challenger": challenger.metrics,
               "paired_delta": delta, "paired_delta_selection": delta_sel,
               "grabit_zone": zone, "floor_zone": fzone,
               # fold x seed R2 matrices — the reference every future paired
               # comparison diffs against (same folds, same seeds, per-fold).
               # *_selection is the decision-grade matrix; pooled is context.
               "fold_r2": {"champion": champion.fold_r2.tolist(),
                           "challenger": challenger.fold_r2.tolist()},
               "fold_r2_selection": {"champion": champion.fold_r2_sel.tolist(),
                                     "challenger": challenger.fold_r2_sel.tolist()},
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
