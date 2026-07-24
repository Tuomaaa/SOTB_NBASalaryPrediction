"""Signing-route probability function + per-route structural branches (phase 1).

The unified architecture (see docs/briefs/2026-07-25-route-mixture.md): a
multiclass probability function over the signing routes a contract could land
on, plus per-route structural values. The ex-ante prediction composes the
branches by their probabilities; when the route is told (ex-post) the mixture
collapses to that branch — the project's Stage-2 semantics.

Two disciplines are load-bearing and enforced in code here:

  1. P is an OUTPUT composition weight, NEVER a feature. The probability array
     produced by the classifier is applied to the regression's latent output;
     it never joins FEATURE_COLS. (The ablation graveyard's rejected experiment
     — fold-honest P(mechanism|x) fed BACK INTO the regression, -0.0073 — is an
     input-side result and does not bear on this output-side composition.)

  2. Classes are defined by WHERE THE SALARY LANDED, not by Spotrac labels:
       max        = is_max_contract  (cap_pct >= 0.90 * max_eligible_pct)
       floor      = is_at_floor       (Minimum-labeled, cap_pct <= 0.025)
       mle        = year-1 pay within 2% of that season's exception amounts
       continuous = everything else
     Spotrac's signing_cat corroborates these but never defines them.

This module builds phase 1: the classifier (all four classes reported) and the
MAX branch (push-then-clip), integrated behind a flag that is OFF by default so
the shipped champion is bit-identical unless a caller opts in.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from config import CAP_BY_SEASON, RAW_DIR
from src.model.train import _XGB_BASE, _make_tobit_obj, TARGET

# Class order is fixed: the integer label IS the softprob column index, so
# P(max) is always column 1 regardless of which classes a training fold holds.
ROUTE_CLASSES = ["continuous", "max", "mle", "floor"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ROUTE_CLASSES)}
MAX_IDX = CLASS_TO_IDX["max"]
MLE_IDX = CLASS_TO_IDX["mle"]
FLOOR_IDX = CLASS_TO_IDX["floor"]

N_SPLITS = 5
DEFAULT_SEEDS = tuple(range(10))

# Modest-capacity multiclass booster. Kept deliberately shallow: the minority
# class (max) has n=56, and calibrated probabilities — not raw discrimination —
# are the deliverable, so no scale_pos_weight and no deep trees. num_class is
# pinned to 4 so a fold missing a class still emits four aligned columns.
_CLF_PARAMS = dict(
    objective="multi:softprob",
    num_class=len(ROUTE_CLASSES),
    max_depth=3,
    eta=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method="hist",
)
_CLF_ROUNDS = 400


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _load_mle_amounts() -> dict[int, list[float]]:
    """{season: [exception amounts in USD]} from the curated MLE table."""
    path = RAW_DIR / "raw_external" / "mle_exception_amounts.csv"
    mle = pd.read_csv(path)
    return mle.groupby("season")["amount_usd"].apply(list).to_dict()


def compute_route_labels(df: pd.DataFrame, mle_tol: float = 0.02) -> np.ndarray:
    """Integer route label per row, by WHERE THE SALARY LANDED.

    Requires is_max_contract and is_at_floor already attached (both come from
    the train.py filter chain via load_evaluation_frame). The mle class is
    year-1 pay within mle_tol of any of that season's exception amounts. Classes
    are assigned by priority max > floor > mle > continuous; on this frame the
    three bands are disjoint (verified: zero overlap), so the priority only
    guards against a future edge case.
    """
    n = len(df)
    is_max = df["is_max_contract"].values.astype(bool)
    is_floor = df["is_at_floor"].values.astype(bool)

    amt_by_season = _load_mle_amounts()
    salary = (df[TARGET].values * df["season"].map(CAP_BY_SEASON).values)
    seasons = df["season"].values
    is_mle = np.zeros(n, bool)
    for i, (s, sal) in enumerate(zip(seasons, salary)):
        for a in amt_by_season.get(int(s), []):
            if abs(sal - a) <= mle_tol * a:
                is_mle[i] = True
                break

    labels = np.full(n, CLASS_TO_IDX["continuous"], dtype=int)
    labels[is_mle] = CLASS_TO_IDX["mle"]
    labels[is_floor] = CLASS_TO_IDX["floor"]
    labels[is_max] = CLASS_TO_IDX["max"]
    return labels


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def train_route_classifier(train: pd.DataFrame, features: list[str], seed: int,
                           labels: np.ndarray | None = None) -> xgb.Booster:
    """Fit the 4-class softprob booster on one training slice.

    labels, when supplied, must already be aligned to `train`'s row order (the
    fold-honest path passes the slice of a precomputed label array so the label
    derivation runs once). When omitted, labels are computed from `train`.
    """
    y = compute_route_labels(train) if labels is None else labels
    dtrain = xgb.DMatrix(train[features].values, label=y,
                         feature_names=list(features))
    params = {**_CLF_PARAMS, "seed": seed}
    return xgb.train(params, dtrain, num_boost_round=_CLF_ROUNDS)


def route_proba(model: xgb.Booster, test: pd.DataFrame,
                features: list[str]) -> np.ndarray:
    """(n_test, 4) probability matrix; column j is P(class j) in ROUTE_CLASSES."""
    dtest = xgb.DMatrix(test[features].values, feature_names=list(features))
    return model.predict(dtest)


def oof_route_proba(df: pd.DataFrame, features: list[str],
                    seeds=DEFAULT_SEEDS) -> np.ndarray:
    """Seed-averaged fold-honest OOF probability matrix (n, 4).

    Uses the suite's GroupKFold(N_SPLITS) by player — identical folds to
    evaluate_suite.oof_groupkfold because GroupKFold is deterministic on a fixed
    row order and group vector. Labels are derived once on the full frame and
    sliced per fold; the classifier for a fold never sees its own held-out rows.
    """
    labels = compute_route_labels(df)
    groups = df["player_name_norm"].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, labels, groups))
    acc = np.zeros((len(df), len(ROUTE_CLASSES)))
    for seed in seeds:
        for tr, va in folds:
            model = train_route_classifier(df.iloc[tr], features, seed,
                                           labels=labels[tr])
            acc[va] += route_proba(model, df.iloc[va], features)
    return acc / len(seeds)


# ---------------------------------------------------------------------------
# Grabit latent (champion regression, exposed before the Stage-2 clip)
# ---------------------------------------------------------------------------

def grabit_latent(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                  seed: int, sigma: float = 0.02, gate_frac: float = 0.55,
                  floor_gate_k: float = 2.0, sigma_left: float | None = None,
                  censor_c: float | None = None):
    """Champion Grabit v4 latent on `test`, plus the Stage-2 bounds (lo, hi).

    Byte-for-byte the same training path as evaluate_suite.make_grabit_fitter,
    but returns the UNCLIPPED latent so a route branch can act on it before the
    clip. Clipping the returned latent into [lo, hi] reproduces the champion
    fitter exactly (asserted in the eval harness).
    """
    from xgboost import XGBRegressor

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
    hi = test["max_eligible_pct"].values
    return latent, lo, hi


# ---------------------------------------------------------------------------
# The MAX branch — push-then-clip, plus the two reference arms
# ---------------------------------------------------------------------------

def make_maxbranch_fitter(enabled: bool = False, arm: str = "push_clip",
                          margin: float = 1.05, grabit_params: dict | None = None,
                          clf_seed_offset: int = 0):
    """Fitter that composes the MAX branch onto the champion Grabit latent.

    enabled=False (the default) returns the champion fitter exactly — the
    integration is inert until a caller opts in, so nothing ships changed.

    When enabled, the classifier is trained inside the training slice and P(max)
    is predicted on the test slice (fold-honest; P is an output weight). The
    arms:

      push_clip (SHIP FORM): adj = latent + P*(margin*hi - latent), then clip
          into [lo, hi]. High-P rows cross the ceiling and land ON it; low-P
          rows are untouched; the ambiguous middle moves partway. `margin` is a
          FIXED constant — never tuned on zone MAE (that re-opens the one-way
          valve the sigma sweeps closed).
      mean:  P*hi + (1-P)*champion, champion = clip(latent, lo, hi)   [ref r1]
      hard:  P>0.5 -> hi else champion                                [ref r2]

    clf_seed_offset lets the classifier use a different seed stream from the
    regression if ever needed; 0 shares the seed.
    """
    gp = grabit_params or {}

    def fitter(train, test, features, seed):
        latent, lo, hi = grabit_latent(train, test, features, seed, **gp)
        champ = np.clip(latent, lo, hi)
        if not enabled:
            return champ

        model = train_route_classifier(train, features, seed + clf_seed_offset)
        p_max = route_proba(model, test, features)[:, MAX_IDX]

        if arm == "push_clip":
            adj = latent + p_max * (margin * hi - latent)
        elif arm == "mean":
            adj = p_max * hi + (1.0 - p_max) * champ
        elif arm == "hard":
            adj = np.where(p_max > 0.5, hi, champ)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        return np.clip(adj, lo, hi)

    return fitter
