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

# Phase-2 classifier-only enrichment (2026-07-25-route-mixture-p2 brief).
# These columns join the CLASSIFIER's feature list ONLY. They never enter the
# regression's FEATURE_COLS, and P(max) is an OUTPUT composition weight, so no
# leakage path into the target exists: the classifier is a different model with
# a different target (route class, not cap_pct), and its probability output is
# applied to the Grabit latent, never fed back as a regression input. The
# columns are the failed-regression feature batch — they lost the REGRESSION
# gates but carry route-discriminating signal the classifier can use.
#
# Families (exactly the brief's list): trend derivatives + their coverage flag,
# season-over-season sign deltas, the early-pricing flag, the previous-season
# value estimate, the rookie award tier, and the name-cleaned cumulative award
# score. Fed with NATIVE NaN (XGBoost hist learns a default split direction) —
# this is the ship form for a tree classifier, matching how the failed-batch
# RESULT tested native-NaN over median-fill.
CLF_EXTRA_TREND = [
    "darko_dpm_z_d1", "darko_dpm_z_slope3", "darko_dpm_z_peakd",
    "lebron_z_d1", "lebron_z_slope3", "lebron_z_peakd",
    "rapm_z_d1", "rapm_z_slope3", "rapm_z_peakd",
    "trend_has_prev",
]
CLF_EXTRA_SIGNDELTA = [
    "darko_dpm_z_signdelta", "lebron_z_signdelta", "rapm_z_signdelta",
    "mpg_signdelta", "usage_pct_signdelta", "ast_pct_signdelta",
]
CLF_EXTRA_MISC = [
    "is_priced_early", "est_value_prev", "rookie_award_tier",
    "award_score_cum_clean",
]
CLF_EXTRA_COLS = CLF_EXTRA_TREND + CLF_EXTRA_SIGNDELTA + CLF_EXTRA_MISC

_FEATURE_BATCH_PATH = "data/processed/feature_batch_columns.csv"


def attach_clf_features(df: pd.DataFrame, extra_cols: list[str] | None = None
                        ) -> tuple[pd.DataFrame, list[str]]:
    """Merge the classifier-only enrichment columns onto `df` by (player, season).

    Returns (df_with_extra, clf_features) where clf_features = FEATURE_COLS +
    the enrichment columns. The regression keeps FEATURE_COLS; only the
    classifier ever sees the wider list. Native NaN is preserved (no fill) — the
    tree classifier learns a default direction, which is the tested ship form.

    Idempotent: re-attaching does not duplicate columns.
    """
    from src.model.train import FEATURE_COLS

    cols = list(CLF_EXTRA_COLS if extra_cols is None else extra_cols)
    root = Path(__file__).resolve().parent.parent.parent
    fb = pd.read_csv(root / _FEATURE_BATCH_PATH)
    keep = ["player_name_norm", "season"] + [c for c in cols if c in fb.columns]
    missing = [c for c in cols if c not in fb.columns]
    if missing:
        raise KeyError(f"feature_batch_columns.csv missing {missing}")

    out = df.copy()
    to_drop = [c for c in cols if c in out.columns]
    if to_drop:
        out = out.drop(columns=to_drop)
    fb_keys = fb[["player_name_norm", "season"]]
    if fb_keys.duplicated().any():
        raise ValueError("feature_batch_columns.csv has duplicate (player, season) keys")
    n0 = len(out)
    out = out.merge(fb[keep], on=["player_name_norm", "season"], how="left")
    assert len(out) == n0, "attach_clf_features changed row count"
    clf_features = list(FEATURE_COLS) + cols
    return out, clf_features


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
# The six-route class set (phase 4)
# ---------------------------------------------------------------------------
# The full architecture is
#
#   pred = P(max)V_max + P(floor)V_floor + P(mle)V_mle
#        + P(bird)V_bird + P(capspace)V_cap + P(extension)V_ext
#
# The four-class set above collapses the last three into "continuous". The
# 2026-07-26 route-delta work split continuous into bird / capspace / other and
# found delta_capspace and delta_other both indistinguishable from zero
# (+0.00083 +/- 0.00498 and +0.00045 +/- 0.00226), so the two are merged here:
# `capspace` is every non-Bird continuous row. `extension` is the sixth route,
# defined by the dated-extension flag rather than by where the salary landed —
# it is the one route whose CBA constraint is a raise cap on the player's OWN
# prior salary rather than a league-wide constant (see extension_cap.py).
#
# Priority is max > floor > mle > extension > bird > capspace, so the max zone
# (n=70) and the floor zone are bit-identical to the four-class set and every
# phase-1..3 zone number stays comparable. A designated-veteran extension paid
# at the ceiling therefore lands in `max`, where its value function (the tier
# ceiling) is the same number either way.
ROUTE6_CLASSES = ["capspace", "max", "mle", "floor", "bird", "extension"]
R6 = {c: i for i, c in enumerate(ROUTE6_CLASSES)}

BIRD_CATS = {"Bird Rights", "Early Bird", "Non-Bird"}


def compute_route6_labels(df: pd.DataFrame, mle_tol: float = 0.02) -> np.ndarray:
    """Integer six-route label per row.

    Requires is_max_contract / is_at_floor (the train.py filter chain) and
    is_extension (extension_cap.attach_extension_cap). `signing_cat` supplies the
    bird/capspace split — the one place a Spotrac label defines a class, because
    Bird Rights is not observable from where the salary landed. That label is a
    DIAGNOSTIC in the regression's world and stays one here: it defines the
    classifier's TARGET, never a regression feature.
    """
    n = len(df)
    labels = np.full(n, R6["capspace"], dtype=int)
    if "signing_cat" in df.columns:
        labels[df["signing_cat"].isin(BIRD_CATS).values] = R6["bird"]
    if "is_extension" in df.columns:
        labels[df["is_extension"].values.astype(bool)] = R6["extension"]

    four = compute_route_labels(df, mle_tol=mle_tol)
    labels[four == MLE_IDX] = R6["mle"]
    labels[four == FLOOR_IDX] = R6["floor"]
    labels[four == MAX_IDX] = R6["max"]
    return labels


def train_route6_classifier(train: pd.DataFrame, features: list[str], seed: int,
                            labels: np.ndarray | None = None) -> xgb.Booster:
    """Six-class softprob booster; same shallow settings as the four-class one."""
    y = compute_route6_labels(train) if labels is None else labels
    dtrain = xgb.DMatrix(train[features].values, label=y,
                         feature_names=list(features))
    params = {**_CLF_PARAMS, "num_class": len(ROUTE6_CLASSES), "seed": seed}
    return xgb.train(params, dtrain, num_boost_round=_CLF_ROUNDS)


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
                          clf_seed_offset: int = 0, tau: float = 0.0,
                          clf_features: list[str] | None = None):
    """Fitter that composes the MAX branch onto the champion Grabit latent.

    enabled=False (the default) returns the champion fitter exactly — the
    integration is inert until a caller opts in, so nothing ships changed.

    When enabled, the classifier is trained inside the training slice and P(max)
    is predicted on the test slice (fold-honest; P is an output weight). The
    regression always uses `features` (FEATURE_COLS); the classifier uses
    `clf_features` when supplied (the phase-2 enriched list), else `features`.
    P(max) NEVER joins the regression's feature list.

    `tau` gates the intervention: rows with P(max) < tau are left at the
    champion prediction untouched. tau=0.0 reproduces the phase-1 ungated arms.
    The arms:

      push_clip (SHIP FORM): for P>=tau, adj = latent + P*(margin*hi - latent),
          then clip into [lo, hi]; for P<tau, adj = champion. `margin` is a
          FIXED constant — never tuned on zone MAE (that re-opens the one-way
          valve the sigma sweeps closed). tau is chosen from the OOF purity
          curve, never from a zone metric.
      hard:  P>=max(tau, 0.5)-gated -> hi else champion. With tau>0 the switch
          fires at tau (pred = ceiling for P>=tau).                  [ref]
      mean:  P*hi + (1-P)*champion (ungated reference)               [ref r1]

    clf_seed_offset lets the classifier use a different seed stream from the
    regression if ever needed; 0 shares the seed.
    """
    gp = grabit_params or {}

    def fitter(train, test, features, seed):
        latent, lo, hi = grabit_latent(train, test, features, seed, **gp)
        champ = np.clip(latent, lo, hi)
        if not enabled:
            return champ

        cf = clf_features if clf_features is not None else features
        model = train_route_classifier(train, cf, seed + clf_seed_offset)
        p_max = route_proba(model, test, cf)[:, MAX_IDX]
        gate = p_max >= tau if tau > 0 else np.ones(len(test), bool)

        if arm == "push_clip":
            pushed = latent + p_max * (margin * hi - latent)
            adj = np.where(gate, pushed, champ)
        elif arm == "mean":
            adj = p_max * hi + (1.0 - p_max) * champ
        elif arm == "hard":
            thresh = tau if tau > 0 else 0.5
            adj = np.where(p_max >= thresh, hi, champ)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        return np.clip(adj, lo, hi)

    return fitter
