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

The champion this suite scores is the full stack — Grabit latent, then the
Stage-2 push and clip, then the Stage-3 extension clip (`src/model/stages.py`).
Stage 3 reads the realized route, so **A1/A2/B1 here are TOLD-ROUTE numbers**
under the convention adopted 2026-07-26 and refined 2026-07-27 (docs/QUEUE.md).
Everything published from v7.1x to v7.13x was computed under the old "ignore the
route" convention, so the suite prints the ex-ante arm (Stage 3 off) and the
clip-only arm (the v7.13x champion) beside the headline. Quote the matching
convention or the series reads as a jump that never happened.

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
    _filter_rookie_contracts,
    _filter_mislabeled_year1, _filter_continuations, _compute_max_eligible,
    _compute_floor, _prepare_Xy, _make_tobit_obj, _XGB_BASE, FEATURE_COLS, TARGET,
)
from src.model.extension_cap import attach_extension_cap
from src.model.route_mixture import (
    attach_clf_features, grabit_latent, route_proba, train_route_classifier,
    MAX_IDX,
)
from src.model.stages import compose, TAU, MARGIN
# reuse the canonical label logic so C2 segments match scripts/diagnostics.py
from scripts.diagnostics import attach_signing_labels

N_SPLITS = 5
DEFAULT_SEEDS = tuple(range(10))
FORWARD_ORIGINS = (2024, 2025, 2026)
CONFIRMATION_PCT = 15  # share of players locked away from model selection

# The three Stage-2/3 arms the suite reports, off ONE fit pass.
#
# CHAMPION is the shipped composition and the number every future paired delta
# diffs against. The two references exist because the reporting convention
# changed underneath the version series (docs/QUEUE.md, 2026-07-26/27): Stage 3
# reads the realized route, so the champion is a TOLD-ROUTE number, while
# v7.1x-v7.13x were all computed under the old "ignore the route" convention.
# ARM_EXANTE is the same stack with Stage 3 off — the figure that stays
# comparable to the published series. ARM_CLIP is the v7.13x champion itself,
# kept so the push and the clip can be attributed separately.
ARM_CLIP = "Stage 2 clip only (v7.13x, ex ante)"
ARM_EXANTE = "+ push (ex ante)"
ARM_CHAMPION = "+ push + Stage 3 extension clip (champion, told route)"
STAGE_ARMS = (ARM_CLIP, ARM_EXANTE, ARM_CHAMPION)

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
                       sigma_left: float | None = None,
                       censor_c: float | None = None):
    """Grabit v4: two-sided censored-normal loss, then both CBA bounds.

    Right side censors gated max rows (observation floors the latent); left
    side censors gated at-floor minimum rows (the league floor props their pay
    up, so the observation CEILS the latent). Both gates need a baseline
    prediction, fit inside the training slice so nothing from the scored slice
    leaks in; both mirror the albatross rule — the model must corroborate that
    the bound binds. Stage 2 clips into [floor_pct, max_eligible_pct].

    floor_gate_k=0 disables the left side (reproduces Grabit v3).
    sigma_left=None ties the left side's sigma to the right's (shipped default).
    censor_c=None keys the right population on is_max_contract (>=0.90 of own
    ceiling, the shipped default). A float c widens the right-censored
    population to every row paid at least c*max_eligible_pct — a TRAINING-loss
    mask off cap_pct, same kind of quantity the existing gates already read,
    never recomputed on test rows and never a feature (see the 2026-07-24
    censor-widening brief). At c=0.90 it reproduces is_max_contract.
    """
    def fitter(train, test, features, seed):
        y_tr = train[TARGET].values
        max_elig_tr = train["max_eligible_pct"].values
        is_max_tr = train["is_max_contract"].values.astype(bool)
        right_pop = is_max_tr if censor_c is None else (y_tr >= censor_c * max_elig_tr)

        base = XGBRegressor(**{**_XGB_BASE, "random_state": seed})
        base.fit(train[features], y_tr)
        bp = base.predict(train[features])
        gate = right_pop & (bp >= gate_frac * max_elig_tr)
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


def make_stage_arms_fitter(clf_features: list[str], grabit_params: dict | None = None):
    """One Grabit fit + one route-classifier fit per (fold, seed), three arms out.

    The arms differ only in which post-Stage-1 layers are composed, so they must
    ride on the SAME fitted models — otherwise the paired delta between them
    would carry fit noise that does not exist in the change being measured. The
    fitter therefore returns a dict of named predictions rather than one array;
    `oof_groupkfold` and `rolling_forward` both understand that form.

    The classifier is fit inside the training slice and P(max) predicted on the
    test slice, so it is fold-honest. P(max) is an OUTPUT composition weight and
    never joins `features`, which stays FEATURE_COLS for the regression.
    """
    gp = grabit_params or {}

    def fitter(train, test, features, seed):
        latent, lo, hi = grabit_latent(train, test, features, seed, **gp)
        clf = train_route_classifier(train, clf_features, seed)
        p_max = route_proba(clf, test, clf_features)[:, MAX_IDX]
        is_ext = test["is_extension"].values
        ext_cap = test["ext_cap_pct"].values
        return {
            ARM_CLIP: compose(latent, lo=lo, hi=hi),
            ARM_EXANTE: compose(latent, lo=lo, hi=hi, p_max=p_max),
            ARM_CHAMPION: compose(latent, lo=lo, hi=hi, p_max=p_max,
                                  is_extension=is_ext, ext_cap_pct=ext_cap),
        }

    return fitter


def make_champion_fitter(clf_features: list[str], push: bool = True,
                         stage3: bool = True, grabit_params: dict | None = None):
    """Single-arm champion fitter — the shipped composition, for one-off use.

    `make_stage_arms_fitter` is what `main()` runs (it amortises the fit across
    the three arms); this is the same composition behind the ordinary
    `fitter(train, test, features, seed)` signature, for a harness that wants
    only the champion.
    """
    gp = grabit_params or {}

    def fitter(train, test, features, seed):
        latent, lo, hi = grabit_latent(train, test, features, seed, **gp)
        p_max = None
        if push:
            clf = train_route_classifier(train, clf_features, seed)
            p_max = route_proba(clf, test, clf_features)[:, MAX_IDX]
        kw = {}
        if stage3:
            kw = {"is_extension": test["is_extension"].values,
                  "ext_cap_pct": test["ext_cap_pct"].values}
        return compose(latent, lo=lo, hi=hi, p_max=p_max, **kw)

    return fitter


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_evaluation_frame(keep_prorated: bool = False,
                          verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Training rows with features imputed, plus the usable feature list.

    Applies the same filter chain as train.py so the suite scores what the model
    is actually fit on, and attaches the Stage-3 route facts.

    Args:
        keep_prorated: skip the prorated-salary filter, retaining partial-season
            rows. Only for reproducing the pre-v7.2x row set — R2 is not
            comparable across different row sets, so a comparison against the
            filtered frame has to fix the evaluation set (see D1).
        verbose: print the extension-cap summary line.
    """
    df = _filter_rookie_scale(_filter_year1(load_training_data()))
    if not keep_prorated:
        df = _filter_prorated(df)
    df = _compute_max_eligible(df)
    df = _filter_mislabeled_year1(df)
    df = _filter_continuations(df)
    # Rookie contracts (a player's first two NBA seasons) leave last, so its
    # printed drop count is the net frame effect and it clears the _compute_floor
    # pool below — a first contract informs neither the model nor the floor scale.
    df = _filter_rookie_contracts(df).reset_index(drop=True)
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
    # Stage-3 inputs (is_extension + ext_cap_pct). Attached here so every
    # consumer of the evaluation frame gets the same route facts and no harness
    # has to remember to call it; the columns are inert for a caller that does
    # not compose Stage 3, and they change no existing quantity — `features` is
    # already fixed by _prepare_Xy above, and ext_cap_pct deliberately never
    # enters max_eligible_pct or the Stage-1 censor mask.
    df = attach_extension_cap(df, verbose=verbose)
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

    When `fitter` returns a dict of named arms instead of one array — the form
    `make_stage_arms_fitter` uses to amortise one fit across several Stage-2/3
    compositions — the return is `{name: (oof, fold_r2, fold_r2_sel)}` instead.
    Arms scored this way share the fitted models exactly, so a paired delta
    between two of them carries no fit noise at all.

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

    acc, fold_r2, fold_r2_sel = {}, {}, {}
    multi = None
    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(folds):
            out = fitter(df.iloc[tr], df.iloc[va], features, seed)
            if multi is None:
                multi = isinstance(out, dict)
            items = out.items() if multi else [(None, out)]
            for name, pred in items:
                if name not in acc:
                    acc[name] = np.zeros(len(df))
                    fold_r2[name] = np.zeros((len(folds), len(seeds)))
                    fold_r2_sel[name] = np.zeros((len(folds), len(seeds)))
                acc[name][va] += pred
                fold_r2[name][fi, si] = r2_score(y[va], pred)
                vs = sel[va]
                fold_r2_sel[name][fi, si] = (r2_score(y[va][vs], pred[vs])
                                             if vs.sum() > 10 else np.nan)
    if not multi:
        return acc[None] / len(seeds), fold_r2[None], fold_r2_sel[None]
    return {k: (acc[k] / len(seeds), fold_r2[k], fold_r2_sel[k]) for k in acc}


def rolling_forward(df: pd.DataFrame, features: list[str], fitter,
                    seeds=DEFAULT_SEEDS, origins=FORWARD_ORIGINS) -> np.ndarray:
    """Predictions for each origin season, trained only on strictly earlier ones.

    Multi-arm fitters are handled the same way `oof_groupkfold` handles them:
    the return becomes `{name: predictions}`.
    """
    preds = {}
    season = df["season"].values
    multi = None
    for T in origins:
        test_mask, train_mask = season == T, season < T
        if test_mask.sum() < 10 or train_mask.sum() < 200:
            continue
        acc = {}
        for seed in seeds:
            out = fitter(df[train_mask], df[test_mask], features, seed)
            if multi is None:
                multi = isinstance(out, dict)
            for name, pred in (out.items() if multi else [(None, out)]):
                if name not in acc:
                    acc[name] = np.zeros(int(test_mask.sum()))
                acc[name] += pred
        for name in acc:
            preds.setdefault(name, np.full(len(df), np.nan))
            preds[name][test_mask] = acc[name] / len(seeds)
    if not multi:
        return preds[None]
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


def baseline_ladder(df, features, seeds=DEFAULT_SEEDS) -> dict:
    """The D2 rungs below the full feature set — identical for every arm.

    Scored once and handed to `layer_d`, because the ladder depends only on the
    frame, not on which Stage-2/3 composition sits on top of it.
    """
    y = df[TARGET].values
    ladder = {"predict the mean": 0.0}
    for name, cols in BASELINE_LADDER.items():
        cols = [c for c in cols if c in features]
        if cols:
            oof, _, _ = oof_groupkfold(df, cols, baseline_fitter, seeds=seeds[:3])
            ladder[name] = float(r2_score(y, oof))
    return ladder


def extension_zone(df, champion_oof, reference_oof) -> dict:
    """Zone-local scorecard for Stage 3: the first-paying-year extension rows.

    Same discipline as `grabit_zone` at the other bound — Stage 3 touches ~16%
    of rows and only ever binds on a fraction of those, so a pooled statistic
    dilutes it. The zone is defined by the ROUTE (is_extension with a computed
    raise cap), not by which rows the clip happened to move, so the scorecard
    also charges Stage 3 for any collateral inside its own population.
    """
    mask = (df["is_extension"].values.astype(bool)
            & df["ext_cap_pct"].notna().values)
    cap_m = df["cap"].values / 1e6
    err_ch = (champion_oof - df[TARGET].values) * cap_m
    err_rf = (reference_oof - df[TARGET].values) * cap_m
    d_abs = np.abs(err_rf[mask]) - np.abs(err_ch[mask])
    moved = np.abs(champion_oof - reference_oof) > 1e-9
    return {
        "n": int(mask.sum()),
        "n_moved": int((mask & moved).sum()),
        "mae_champion": float(np.abs(err_ch[mask]).mean()),
        "mae_reference": float(np.abs(err_rf[mask]).mean()),
        "delta_mae": float(np.abs(err_ch[mask]).mean() - np.abs(err_rf[mask]).mean()),
        "bias_champion": float(err_ch[mask].mean()),
        "bias_reference": float(err_rf[mask].mean()),
        "rows_better": int((d_abs > 0.005).sum()),
        "rows_worse": int((d_abs < -0.005).sum()),
    }


def layer_d(df, features, pred_oof, seeds=DEFAULT_SEEDS, ladder=None) -> dict:
    """D2 baseline ladder and D3 the locked confirmation split."""
    y = df[TARGET].values
    ladder = dict(baseline_ladder(df, features, seeds) if ladder is None
                  else ladder)
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
              seeds=DEFAULT_SEEDS, ladder=None) -> SuiteResult:
    """Score one variant through all four layers."""
    print(f"\nScoring '{name}' over {len(seeds)} seeds...")
    oof, fold_r2, fold_r2_sel = oof_groupkfold(df, features, fitter, seeds)
    fwd = rolling_forward(df, features, fitter, seeds)

    res = SuiteResult(name=name, oof=oof, forward=fwd, fold_r2=fold_r2,
                      fold_r2_sel=fold_r2_sel)
    res.metrics.update(layer_a(df, oof, fold_r2))
    res.metrics.update(layer_b(df, fwd))
    res.metrics.update(layer_c(df, oof))
    res.metrics.update(layer_d(df, features, oof, seeds, ladder=ladder))
    return res


def run_suite_arms(df: pd.DataFrame, features: list[str], fitter,
                   seeds=DEFAULT_SEEDS, ladder=None) -> dict[str, SuiteResult]:
    """Score every arm of a multi-arm fitter through all four layers.

    One CV pass and one rolling-forward pass total, so the arms are compared on
    identical fitted models and the D2 ladder is scored once for all of them.
    """
    print(f"\nScoring the Stage-2/3 arms over {len(seeds)} seeds "
          f"(one fit pass)...")
    oof_arms = oof_groupkfold(df, features, fitter, seeds)
    fwd_arms = rolling_forward(df, features, fitter, seeds)
    if ladder is None:
        ladder = baseline_ladder(df, features, seeds)

    out = {}
    for name, (oof, fold_r2, fold_r2_sel) in oof_arms.items():
        res = SuiteResult(name=name, oof=oof, forward=fwd_arms[name],
                          fold_r2=fold_r2, fold_r2_sel=fold_r2_sel)
        res.metrics.update(layer_a(df, oof, fold_r2))
        res.metrics.update(layer_b(df, res.forward))
        res.metrics.update(layer_c(df, oof))
        res.metrics.update(layer_d(df, features, oof, seeds, ladder=ladder))
        out[name] = res
    return out


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


def print_convention_block(df, arms: dict, deltas: dict):
    """The three Stage-2/3 arms side by side, told-route beside ex-ante.

    The champion reads the realized route (Stage 3), so its A1/A2/B1 are
    TOLD-ROUTE numbers under the convention adopted 2026-07-26 and refined
    2026-07-27. Every published figure from v7.1x to v7.13x was computed under
    the old "ignore the route" convention, so the ex-ante column is printed
    beside the headline — comparing the two naively is the mistake this block
    exists to prevent.
    """
    line = "=" * 100
    print(f"\n{line}\n  STAGE 2/3 ARMS — told route beside ex ante "
          f"(tau={TAU}, margin={MARGIN}, both pre-registered)\n{line}")
    print(f"  {'arm':52s} {'A1':>7s} {'A2':>7s} {'B1':>7s} {'MAE':>7s} "
          f"{'slope':>6s}")
    for name in STAGE_ARMS:
        m = arms[name].metrics
        print(f"  {name:52s} {m['A1_cv_r2']:7.4f} {m['A2_cv_r2_2024_26']:7.4f} "
              f"{m.get('B1_forward_r2', float('nan')):7.4f} "
              f"{m['A1_cv_mae_m']:7.2f} {m['C1_calibration_slope']:6.3f}")
    print("\n  B1 per origin (the 2026 row is the project's holdout headline):")
    for T in FORWARD_ORIGINS:
        cells = []
        for name in STAGE_ARMS:
            d = arms[name].metrics.get("B1_by_origin", {}).get(int(T))
            cells.append(f"{d['r2']:.4f}" if d else "  n/a ")
        n = arms[STAGE_ARMS[0]].metrics.get("B1_by_origin", {}).get(int(T), {})
        print(f"    {T}  n={n.get('n', 0):3d}   " +
              "   ".join(f"{c}" for c in cells))
    print(f"    {'':13s}   " + "   ".join(
        f"{s.split('(')[0].strip()[:6]:>6s}" for s in STAGE_ARMS))

    print("\n  PAIRED deltas on the selection pool (same folds, same fitted "
          "models):")
    for label, d in deltas.items():
        print(f"    {label:52s} {d['delta']:+.5f}  +/- {d['se']:.5f}  "
              f"t = {d['t']:+.2f}")
    print("    per fold, champion vs Stage-2 clip only: "
          f"{deltas['champion - clip only (the whole change)']['per_fold']}")
    print("\n  The change is adopted at t = +1.11, below the t > 2 feature bar,")
    print("  on the same grounds as v7.4x, v7.9x and v7.13x: it enforces a legal")
    print("  bound rather than fitting a parameter. Predicting $39.68M for")
    print("  Marcus Smart 2022 was not inaccurate, it was impossible.")


def print_stage3_accounting(df, arms: dict):
    """Which rows the push and the extension clip actually move, and by how much."""
    champ = arms[ARM_CHAMPION].oof
    exante = arms[ARM_EXANTE].oof
    clip = arms[ARM_CLIP].oof
    cap_m = df["cap"].values / 1e6
    y = df[TARGET].values
    conf = df["is_confirmation"].values

    pushed = np.abs(exante - clip) > 1e-9
    clipped = np.abs(champ - exante) > 1e-9
    print(f"\n{'=' * 100}\n  STAGE 2/3 ACCOUNTING — the rows each layer moves"
          f"\n{'=' * 100}")
    print(f"  push moves      {int(pushed.sum()):3d} rows "
          f"({int((pushed & conf).sum())} of them confirmation rows)   "
          f"mean |err| ${np.abs(clip[pushed] - y[pushed]).dot(cap_m[pushed]) / max(pushed.sum(), 1):.2f}M "
          f"-> ${np.abs(exante[pushed] - y[pushed]).dot(cap_m[pushed]) / max(pushed.sum(), 1):.2f}M")
    print(f"  Stage 3 moves   {int(clipped.sum()):3d} rows "
          f"({int((clipped & conf).sum())} of them confirmation rows)   "
          f"mean |err| ${np.abs(exante[clipped] - y[clipped]).dot(cap_m[clipped]) / max(clipped.sum(), 1):.2f}M "
          f"-> ${np.abs(champ[clipped] - y[clipped]).dot(cap_m[clipped]) / max(clipped.sum(), 1):.2f}M")
    print("\n  Rows Stage 3 returns to the raise cap, largest correction first:")
    order = np.flatnonzero(clipped)
    order = order[np.argsort(-(exante[order] - champ[order]))]
    for i in order[:20]:
        tag = " [CONFIRM]" if conf[i] else ""
        print(f"    {df['player_name_norm'].iat[i]:24s} {int(df['season'].iat[i])}"
              f"  pay {y[i] * cap_m[i]:6.2f}  push {exante[i] * cap_m[i]:6.2f}"
              f"  clip {champ[i] * cap_m[i]:6.2f}"
              f"  ceiling {df['max_eligible_pct'].iat[i] * cap_m[i]:6.2f}"
              f"  raise cap {df['ext_cap_pct'].iat[i] * cap_m[i]:6.2f}"
              f"  |err| {abs(exante[i] - y[i]) * cap_m[i]:6.2f} -> "
              f"{abs(champ[i] - y[i]) * cap_m[i]:5.2f}{tag}")


def main():
    df, features = load_evaluation_frame()
    df, clf_features = attach_clf_features(df)
    print(f"Loaded {len(df)} rows, {len(features)} features, "
          f"{len(clf_features)} classifier features, "
          f"seasons {df['season'].min()}-{df['season'].max()}, "
          f"{int(df['is_extension'].sum())} first-year extension rows")

    ladder = baseline_ladder(df, features, DEFAULT_SEEDS)
    arms = run_suite_arms(df, features, make_stage_arms_fitter(clf_features),
                          ladder=ladder)
    champion = arms[ARM_CHAMPION]
    print_report(df, champion)

    deltas = {
        "champion - clip only (the whole change)":
            paired_delta(arms[ARM_CLIP].fold_r2_sel, champion.fold_r2_sel),
        "push alone (ex ante - clip only)":
            paired_delta(arms[ARM_CLIP].fold_r2_sel, arms[ARM_EXANTE].fold_r2_sel),
        "Stage 3 alone (champion - ex ante)":
            paired_delta(arms[ARM_EXANTE].fold_r2_sel, champion.fold_r2_sel),
    }
    print_convention_block(df, arms, deltas)
    print_stage3_accounting(df, arms)

    challenger = run_suite(df, features, baseline_fitter, "Baseline XGBoost",
                           ladder=ladder)
    print_report(df, challenger)

    delta = paired_delta(challenger.fold_r2, champion.fold_r2)
    delta_sel = paired_delta(challenger.fold_r2_sel, champion.fold_r2_sel)
    zone = grabit_zone(df, champion.oof, challenger.oof)
    fzone = floor_zone(df, champion.oof, challenger.oof)
    ezone = extension_zone(df, champion.oof, arms[ARM_EXANTE].oof)
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
    print(f"\n    Extension zone (first-paying-year extensions, n={ezone['n']}, "
          f"{ezone['n_moved']} moved by Stage 3), champion vs its own ex-ante arm:")
    print(f"      MAE  ${ezone['mae_reference']:.2f}M -> ${ezone['mae_champion']:.2f}M  "
          f"({ezone['delta_mae']:+.2f})   rows better/worse "
          f"{ezone['rows_better']}/{ezone['rows_worse']}")
    print(f"      bias ${ezone['bias_reference']:+.2f}M -> ${ezone['bias_champion']:+.2f}M")
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
    print("    The C1 calibration gate is RELATIVE (adjudicated 2026-07-23): a")
    print("    candidate's |slope - 1| may exceed the incumbent's by at most 0.005")
    print("    (~$0.3M of scale at a $60M max — C2's own yardstick). Never an")
    print("    absolute window: [0.99, 1.01] excluded the champion's own 0.9883.")
    print("    And never select sigma on zone MAE — Stage 2's clip makes the")
    print("    censored sides one-way valves, so zone MAE is monotone in sigma;")
    print("    see docs/briefs/2026-07-23-sigma-gate-retune.RESULT.md.")

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
               # The champion is a TOLD-ROUTE number (Stage 3 reads the realized
               # extension flag). These two are the same stack with Stage 3 off
               # and with both Stage-3 and the push off — the second is the
               # v7.13x champion, i.e. the convention v7.1x-v7.13x were
               # published under. Keep both when quoting the series.
               "champion_exante": arms[ARM_EXANTE].metrics,
               "champion_clip_only": arms[ARM_CLIP].metrics,
               "stage_arm_names": {"champion": ARM_CHAMPION,
                                   "exante": ARM_EXANTE, "clip_only": ARM_CLIP},
               "stage_constants": {"tau": TAU, "margin": MARGIN},
               "stage_deltas_selection": deltas,
               "paired_delta": delta, "paired_delta_selection": delta_sel,
               "grabit_zone": zone, "floor_zone": fzone,
               "extension_zone": ezone,
               # fold x seed R2 matrices — the reference every future paired
               # comparison diffs against (same folds, same seeds, per-fold).
               # *_selection is the decision-grade matrix; pooled is context.
               "fold_r2": {"champion": champion.fold_r2.tolist(),
                           "champion_exante": arms[ARM_EXANTE].fold_r2.tolist(),
                           "champion_clip_only": arms[ARM_CLIP].fold_r2.tolist(),
                           "challenger": challenger.fold_r2.tolist()},
               "fold_r2_selection": {
                   "champion": champion.fold_r2_sel.tolist(),
                   "champion_exante": arms[ARM_EXANTE].fold_r2_sel.tolist(),
                   "champion_clip_only": arms[ARM_CLIP].fold_r2_sel.tolist(),
                   "challenger": challenger.fold_r2_sel.tolist()},
               "seeds": list(DEFAULT_SEEDS), "n_splits": N_SPLITS}
    with open(out_dir / "evaluation_suite.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    ref = df[["player_name_norm", "season", TARGET, "salary_m",
              "signing_cat", "is_confirmation", "is_extension", "ext_cap_pct",
              "max_eligible_pct"]].copy()
    ref["oof_champion"] = champion.oof
    ref["fwd_champion"] = champion.forward
    ref["oof_champion_exante"] = arms[ARM_EXANTE].oof
    ref["fwd_champion_exante"] = arms[ARM_EXANTE].forward
    ref["oof_champion_clip_only"] = arms[ARM_CLIP].oof
    ref["fwd_champion_clip_only"] = arms[ARM_CLIP].forward
    ref["oof_challenger"] = challenger.oof
    ref["fwd_challenger"] = challenger.forward
    ref.to_csv(out_dir / "oof_reference.csv", index=False)
    print(f"\nSaved {out_dir / 'evaluation_suite.json'}")
    print(f"Saved {out_dir / 'oof_reference.csv'} ({len(ref)} rows)")


if __name__ == "__main__":
    main()
