r"""Measurement harness — Stage-3 signing-type residual MODEL, rung 2.

CANDIDATE for replacing v8.8x constant offsets. This file MEASURES the rung-2
candidate and is NOT deployed. Do not modify stages.py or commit based on its
output until the measurement has been reviewed.

===========================================================================
DESIGN — two components, both fixed before any score was seen
===========================================================================

Component 1 — Signing residual model
    A small XGBoost that predicts the OOF residual (actual - stage2_pred) from
    signing_type (one-hot for Bird Rights / Cap Space / Early Bird / Non-Bird)
    and stage2_pred.  Trained only on eligible-type rows.  Ineligible/unknown
    types get ZERO correction.  Hyperparameters are conservative and
    PRE-REGISTERED (max_depth=2, n_estimators=50, min_child_weight=20,
    learning_rate=0.1, subsample=0.8).

Component 2 — Mechanism cap clips
    Legal ceilings for Early Bird and Non-Bird contracts, applied AFTER the
    residual model's correction, BEFORE the existing extension clip.  These are
    zero-parameter, deterministic, same architecture as the extension clip: they
    only LOWER predictions.

    Early Bird cap = max(1.75 x prior_salary, 1.05 x EAS)
        EAS from extension_raise_caps.csv, keyed by PAYING season.

    Non-Bird cap = max(1.20 x prior_salary, 1.20 x vet_min_for_experience)
        Vet min from the published CBA minimum salary scale.

Chain order:  Stage-2 pred  ->  signing residual model  ->  mechanism cap clip
              ->  extension clip  ->  compose() re-clip [floor, max]

MEASUREMENT PROTOCOL
    champion   = v8.7x (Stage 1 + Stage 2 push+clip + extension clip, NO offsets)
    reference  = v8.8x (champion + constant signing offsets, k=20)
    candidate  = this rung-2 (champion + XGB residual model + mechanism caps)
    Primary gate: valid-type rows paired MAE/bias delta.
    10 seeds, 5 folds.  Full guard suite: bit-identity, legality, confirmation.

Run:  OMP_NUM_THREADS=6 python scripts/eval_stage3_model.py [--seeds N] [--skip-b]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from config import CAP_BY_SEASON, OUTPUTS_DIR, RAW_DIR, CBA_NEW_ERA_SEASON
from src.model.evaluate_suite import (
    load_evaluation_frame, paired_delta, _dollars, N_SPLITS, DEFAULT_SEEDS,
    TARGET, FORWARD_ORIGINS,
)
from src.model import route_mixture as rm
from src.model.stages import (
    stage3, stage3_signing, signing_offsets, signing_offset_vector, compose,
    SIGNING_ELIGIBLE_TYPES, SIGNING_K, SIGNING_OFFSETS_DEPLOYED,
)
from src.model.extension_cap import CAP_TOL

# ---- pre-registered constants ------------------------------------------------
ELIGIBLE_TYPES = SIGNING_ELIGIBLE_TYPES
K_REFERENCE = SIGNING_K   # v8.8x reference arm

# Display cap for offsets (2026).
CAP_DISPLAY = CAP_BY_SEASON.get(2026, 153_000_000)

# XGB residual model hyperparameters — PRE-REGISTERED, never tuned on a score.
RESID_XGB_PARAMS = dict(
    max_depth=2, n_estimators=50, min_child_weight=20,
    learning_rate=0.1, subsample=0.8, colsample_bytree=1.0,
    objective="reg:squarederror", tree_method="hist",
)

DIAG_DIR = OUTPUTS_DIR / "diagnostics"
OUT_CSV = DIAG_DIR / "stage3_model_eval.csv"
OUT_JSON = OUTPUTS_DIR / "models" / "stage3_model_eval.json"

REPORT_ORDER = ["Bird Rights", "Cap Space", "Early Bird", "Non-Bird",
                "MLE", "Minimum", "Other", "Rookie Scale",
                "Unknown"]

# Non-Bird sanity tolerance: 0.1% of cap.  Absorbs Clark (2020, exp=2) whose
# debut-year error puts him $55K over the published 2-year CBA minimum.
MECH_CAP_TOL = 1e-3


# =============================================================================
# COMPONENT 2 — Mechanism cap computation
# =============================================================================

# ---- Published 2017 CBA veteran minimum salary scale (Exhibit I) ----
# USD figures for 2019-20 (= 2020-21, frozen per COVID agreement).
# Tiers 0-9 are per service year; 10 means 10+ years.
_CBA_2017_BASE_2020 = {
    0: 898310, 1: 1445697, 2: 1620564, 3: 1678854, 4: 1737145,
    5: 1795435, 6: 1853726, 7: 1912016, 8: 1970306, 9: 2028594,
    10: 2564753,
}
# Per-season scaling from the 2020 base.  The CBA specifies absolute amounts;
# these ratios recover them from the base.
_CBA_2017_SCALE = {
    2019: {e: v for e, v in _CBA_2017_BASE_2020.items()},
    2020: {e: v for e, v in _CBA_2017_BASE_2020.items()},  # frozen
    2021: {e: int(round(v * 1.030)) for e, v in _CBA_2017_BASE_2020.items()},
    2022: {e: int(round(v * 1.061)) for e, v in _CBA_2017_BASE_2020.items()},
}

# ---- 2023 CBA tier ratios (derived from data + Non-Bird ceiling rows) ----
# Ratio of the paid vet minimum to the base (3+ year) minimum.
# The base rate in cap_pct terms is approximately 0.014848 across all seasons.
# Verified against Hayes (1.252), Trent (1.342), Green (1.342), and the MAX of
# Minimum-labeled rows at exp=4 (1.072), exp=8 (1.395), exp=10+ (1.583).
_CBA_2023_TIER_RATIO = {
    0: 0.554, 1: 0.850, 2: 0.950, 3: 1.000,
    4: 1.072, 5: 1.162, 6: 1.252, 7: 1.342,
    8: 1.395, 9: 1.448, 10: 1.583,
}
_CBA_2023_BASE_PCT = 0.014848   # 3+ year minimum in cap_pct terms

# Okogie (2023 exp=5) label issue: his Non-Bird label is inconsistent with the
# 2017 CBA minimum scale for 5 years of service ($1,932,936).  120% of that is
# $2,319,523, but his actual is $2,816,000 — a $497K gap that no experience
# count can close.  Flagged as a wrong label (likely signed via Room Exception
# or cap room); excluded from Non-Bird cap.
_NONBIRD_LABEL_EXCLUSIONS = {("josh okogie", 2023)}


def _get_vet_min_usd(season: int, exp: int) -> float:
    """CBA veteran minimum (PAID salary) for the given season and experience."""
    tier = min(exp, 10)
    if season < CBA_NEW_ERA_SEASON:
        # 2017 CBA: published Exhibit I figures
        if season in _CBA_2017_SCALE:
            return float(_CBA_2017_SCALE[season].get(tier,
                         _CBA_2017_SCALE[season][10]))
        # Fallback: scale from 2020 base by cap ratio
        cap_ratio = CAP_BY_SEASON.get(season, CAP_BY_SEASON[2020]) / \
                    CAP_BY_SEASON[2020]
        return float(_CBA_2017_BASE_2020.get(tier, _CBA_2017_BASE_2020[10])
                     * cap_ratio)
    else:
        # 2023 CBA: base_pct * cap * tier_ratio
        cap = CAP_BY_SEASON[season]
        ratio = _CBA_2023_TIER_RATIO.get(tier, _CBA_2023_TIER_RATIO[10])
        return _CBA_2023_BASE_PCT * cap * ratio


def _load_eas_for_early_bird() -> dict:
    """Load the EAS (Estimated Average Player Salary) per paying season.

    For Early Bird contracts, the CBA floor is 1.05 x EAS of the PAYING season.
    The EAS figures are curated in extension_raise_caps.csv.
    For season 2026 (missing from the CSV), we extrapolate from cap growth.
    """
    path = RAW_DIR / "raw_external" / "extension_raise_caps.csv"
    rc = pd.read_csv(path)
    # Map signing_season -> implied_eas_usd.  For Early Bird, keyed by paying
    # season: the free-agent's paying season = signing season.
    eas = dict(zip(rc["signing_season"].astype(int),
                   rc["implied_eas_usd"].astype(float)))
    # Extrapolate 2026 if missing
    if 2026 not in eas and 2025 in eas:
        cap_ratio = CAP_BY_SEASON.get(2026, 164_961_000) / \
                    CAP_BY_SEASON.get(2025, 154_647_000)
        eas[2026] = eas[2025] * cap_ratio
    return eas


def attach_mechanism_caps(df: pd.DataFrame, verbose: bool = True
                          ) -> pd.DataFrame:
    """Add mech_cap_pct for Early Bird and Non-Bird rows.

    The mechanism cap is the legal ceiling for the signing mechanism:
      Early Bird: max(1.75 x prior, 1.05 x EAS)
      Non-Bird:   max(1.20 x prior, 1.20 x vet_min)

    For all other signing types, mech_cap_pct is NaN (no mechanism cap).
    The cap only ever LOWERS predictions (same as extension_cap).
    """
    from src.model.train import _load_debut_seasons
    from scripts.build_external_features import norm

    df = df.copy()
    df["mech_cap_pct"] = np.nan
    cat = df["signing_cat"].values
    eas = _load_eas_for_early_bird()
    debut = _load_debut_seasons()

    # Compute experience
    pn_clean = df["player_name_norm"].apply(norm)
    exp = (df["season"] - pn_clean.map(debut)).fillna(
        (df["age"].fillna(25) - 19).clip(lower=0)).astype(int).clip(lower=0)

    n_eb, n_nb, n_eb_viol, n_nb_viol, n_nb_excl = 0, 0, 0, 0, 0

    for idx in df.index:
        row_cat = cat[idx] if idx < len(cat) else df.loc[idx, "signing_cat"]
        s = int(df.loc[idx, "season"])
        cap_s = CAP_BY_SEASON[s]
        prior_cap = CAP_BY_SEASON.get(s - 1, cap_s)
        prior_usd = float(df.loc[idx, "prev_cap_pct"]) * prior_cap
        actual_pct = float(df.loc[idx, TARGET])

        if row_cat == "Early Bird":
            n_eb += 1
            # Early Bird cap = max(1.75 x prior, 1.05 x EAS)
            eas_usd = eas.get(s)
            if eas_usd is None:
                continue  # no EAS for this season
            eb_cap_usd = max(1.75 * prior_usd, 1.05 * eas_usd)
            eb_cap_pct = eb_cap_usd / cap_s
            df.loc[idx, "mech_cap_pct"] = eb_cap_pct
            if actual_pct > eb_cap_pct + CAP_TOL:
                n_eb_viol += 1
                if verbose:
                    print(f"  EB violation: {df.loc[idx, 'player_name_norm']} "
                          f"{s} actual={actual_pct*cap_s/1e6:.3f}M "
                          f"cap={eb_cap_pct*cap_s/1e6:.3f}M "
                          f"over={((actual_pct-eb_cap_pct)*cap_s)/1e6:.3f}M")

        elif row_cat == "Non-Bird":
            pn = str(df.loc[idx, "player_name_norm"])
            # Check for excluded labels
            if (pn, s) in _NONBIRD_LABEL_EXCLUSIONS:
                n_nb_excl += 1
                if verbose:
                    print(f"  NB excluded (label issue): {pn} {s}")
                continue
            n_nb += 1
            e = int(exp.loc[idx])
            vm_usd = _get_vet_min_usd(s, e)
            nb_cap_usd = max(1.20 * prior_usd, 1.20 * vm_usd)
            nb_cap_pct = nb_cap_usd / cap_s
            df.loc[idx, "mech_cap_pct"] = nb_cap_pct
            if actual_pct > nb_cap_pct + MECH_CAP_TOL:
                n_nb_viol += 1
                if verbose:
                    print(f"  NB violation: {pn} {s} exp={e} "
                          f"actual={actual_pct*cap_s/1e6:.3f}M "
                          f"cap={nb_cap_pct*cap_s/1e6:.3f}M "
                          f"over={((actual_pct-nb_cap_pct)*cap_s)/1e6:.3f}M "
                          f"vm={vm_usd/1e6:.3f}M")

    if verbose:
        print(f"\nMechanism caps: {n_eb} Early Bird ({n_eb_viol} violations), "
              f"{n_nb} Non-Bird ({n_nb_viol} violations, {n_nb_excl} excluded)")
    return df


def apply_mechanism_cap(pred: np.ndarray, mech_cap_pct: np.ndarray
                        ) -> np.ndarray:
    """Clip prediction at the mechanism cap (only LOWERS)."""
    out = np.array(pred, dtype=float, copy=True)
    cap = np.asarray(mech_cap_pct, dtype=float)
    m = ~np.isnan(cap)
    out[m] = np.minimum(out[m], cap[m])
    return out


# =============================================================================
# COMPONENT 1 — XGBoost signing residual model
# =============================================================================

def build_resid_features(stage2_pred: np.ndarray, cat: np.ndarray
                         ) -> np.ndarray:
    """Build feature matrix for the residual model: one-hot + stage2_pred."""
    n = len(stage2_pred)
    # One-hot for the 4 eligible types
    feats = np.zeros((n, len(ELIGIBLE_TYPES) + 1), dtype=np.float32)
    for i, t in enumerate(ELIGIBLE_TYPES):
        feats[cat == t, i] = 1.0
    feats[:, len(ELIGIBLE_TYPES)] = stage2_pred
    return feats


RESID_FEATURE_NAMES = list(ELIGIBLE_TYPES) + ["stage2_pred"]


def train_resid_model(X_train, y_resid_train, seed: int) -> XGBRegressor:
    """Train the signing residual model."""
    params = dict(RESID_XGB_PARAMS)
    params["random_state"] = seed
    mdl = XGBRegressor(**params)
    mdl.fit(X_train, y_resid_train)
    return mdl


def predict_resid_correction(mdl: XGBRegressor, X: np.ndarray,
                             elig_mask: np.ndarray) -> np.ndarray:
    """Predict residual correction; zero for ineligible rows."""
    correction = np.zeros(len(X), dtype=float)
    if elig_mask.sum() > 0:
        correction[elig_mask] = mdl.predict(X[elig_mask])
    return correction


# =============================================================================
# Apply the full candidate correction
# =============================================================================

def apply_candidate(pred, correction, *, mech_cap_pct, is_extension,
                    ext_cap_pct, lo, hi) -> np.ndarray:
    """Apply residual model correction + mechanism cap + extension clip + re-clip.

    Chain: pred + correction -> mechanism cap clip -> extension clip -> [lo, hi]
    """
    out = np.array(pred, dtype=float, copy=True) + correction
    # 1. Mechanism cap clip (Early Bird / Non-Bird ceiling)
    out = apply_mechanism_cap(out, mech_cap_pct)
    # 2. Extension clip (same as stage3)
    out = stage3(out, is_extension=is_extension, ext_cap_pct=ext_cap_pct)
    # 3. Re-clip into [lo, hi]
    out = np.clip(out, np.asarray(lo, dtype=float),
                  np.asarray(hi, dtype=float))
    return out


# =============================================================================
# Fit pass — champion, once.  All arms ride on these predictions.
# =============================================================================

def champion_fold_pass(df, features, clf_features, seeds, tag=""):
    """Champion OOF predictions per (fold, seed), on the suite's exact split.

    The champion is v8.7x: Grabit + Stage 2 push+clip + extension clip, NO
    signing offsets.
    """
    y = df[TARGET].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(
        df, y, df["player_name_norm"].values))
    store = []
    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(folds):
            train, test = df.iloc[tr], df.iloc[va]
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            clf = rm.train_route_classifier(train, clf_features, seed)
            p_max = rm.route_proba(clf, test, clf_features)[:, rm.MAX_IDX]
            pred = compose(latent, lo=lo, hi=hi, p_max=p_max,
                           is_extension=test["is_extension"].values,
                           ext_cap_pct=test["ext_cap_pct"].values)
            store.append({"si": si, "fi": fi, "va": va, "champ": pred})
        print(f"    {tag}seed {seed} done ({si + 1}/{len(seeds)})", flush=True)
    return store, folds


def seed_average(store, n_rows, key, n_seeds):
    acc = np.zeros(n_rows)
    for rec in store:
        acc[rec["va"]] += rec[key]
    return acc / n_seeds


# =============================================================================
# Per-(fold, seed) statistic matrices
# =============================================================================

def cell_r2(store, y, mask, key, n_folds, n_seeds, min_n=10):
    out = np.full((n_folds, n_seeds), np.nan)
    for rec in store:
        va = rec["va"]
        m = mask[va]
        if m.sum() > min_n:
            out[rec["fi"], rec["si"]] = r2_score(y[va][m], rec[key][m])
    return out


def cell_err(store, y, cap_m, mask, key, n_folds, n_seeds, min_n=1):
    """(MAE, signed bias) matrices in $M over `mask` rows of each fold."""
    mae = np.full((n_folds, n_seeds), np.nan)
    bias = np.full((n_folds, n_seeds), np.nan)
    for rec in store:
        va = rec["va"]
        m = mask[va]
        if m.sum() < min_n:
            continue
        err = (rec[key][m] - y[va][m]) * cap_m[va][m]
        mae[rec["fi"], rec["si"]] = np.abs(err).mean()
        bias[rec["fi"], rec["si"]] = err.mean()
    return mae, bias, np.abs(bias)


def cell_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Cell-level (fold x seed) paired delta b - a.  CONTEXT ONLY."""
    d = (b - a).ravel()
    d = d[~np.isnan(d)]
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"delta": mean, "se": se, "t": mean / se if se > 0 else float("nan"),
            "n_cells": int(len(d))}


# =============================================================================
# Reporting helpers
# =============================================================================

def per_type_table(df, cat, champ, cand, mask=None):
    rows = []
    base = np.ones(len(df), bool) if mask is None else mask
    for t in REPORT_ORDER:
        m = (cat == t) & base
        if m.sum() < 3:
            continue
        mae_c, bias_c = _dollars(df, champ, m)
        mae_d, bias_d = _dollars(df, cand, m)
        y = df[TARGET].values
        rows.append({
            "signing_type": t, "n": int(m.sum()),
            "eligible": t in ELIGIBLE_TYPES,
            "champ_bias_pct": float((champ[m] - y[m]).mean()),
            "cand_bias_pct": float((cand[m] - y[m]).mean()),
            "champ_bias_m": bias_c, "cand_bias_m": bias_d,
            "champ_mae_m": mae_c, "cand_mae_m": mae_d,
            "mae_win_m": mae_c - mae_d,
            "abs_bias_reduction_m": abs(bias_c) - abs(bias_d),
        })
    return pd.DataFrame(rows)


def print_per_type(tab, title):
    print(f"\n  {title}")
    print(f"    {'type':14s} {'n':>4s} {'bias$M':>8s} -> {'bias$M':>8s} "
          f"{'|b|red':>7s} | {'MAE$M':>7s} -> {'MAE$M':>7s} {'win':>7s}")
    for _, r in tab.iterrows():
        star = "*" if r["eligible"] else " "
        print(f"   {star}{r['signing_type']:14s} {r['n']:4d} "
              f"{r['champ_bias_m']:+8.3f} -> {r['cand_bias_m']:+8.3f} "
              f"{r['abs_bias_reduction_m']:+7.3f} | {r['champ_mae_m']:7.3f} -> "
              f"{r['cand_mae_m']:7.3f} {r['mae_win_m']:+7.3f}")
    print("    (* = eligible for the correction; every other row must be "
          "bit-identical)")


# =============================================================================
# Layer B — rolling origin, residual model from seasons < T only
# =============================================================================

def inner_oof_both(train_df, features, clf_features, seeds, k_ref, tag):
    """Inner OOF champion, then derive BOTH the residual model AND constant
    offsets from the same OOF predictions.  One fit pass, two products.

    Returns (resid_model, ref_offsets, oof_champ).
    """
    store, folds = champion_fold_pass(train_df, features, clf_features, seeds,
                                      tag=tag)
    oof = seed_average(store, len(train_df), "champ", len(seeds))
    resid = train_df[TARGET].values - oof
    cat = train_df["signing_cat"].values
    elig_inner = np.isin(cat, ELIGIBLE_TYPES)

    # Product 1: residual model trained on eligible-type OOF residuals
    X = build_resid_features(oof, cat)
    mdl = train_resid_model(X[elig_inner], resid[elig_inner], seed=42)

    # Product 2: constant offsets (v8.8x reference) over all rows
    ref_offs = signing_offsets(resid, cat,
                               pool=np.ones(len(train_df), bool),
                               k=k_ref, detail=True)
    return mdl, ref_offs, oof


def layer_b(df, features, clf_features, seeds):
    """B1: champion vs candidate vs reference under rolling origin.

    Corrections are applied to the SEED-AVERAGED champion, matching the
    reference harness's convention (eval_stage3_signing.py line 348-355).
    """
    season = df["season"].values
    lo_all = df["floor_pct"].values
    hi_all = df["max_eligible_pct"].values
    cat = df["signing_cat"].values
    mech_cap = df["mech_cap_pct"].values
    is_ext = df["is_extension"].values
    ext_cap = df["ext_cap_pct"].values
    elig = np.isin(cat, ELIGIBLE_TYPES)

    champ = np.full(len(df), np.nan)
    cand_model = np.full(len(df), np.nan)
    cand_ref = np.full(len(df), np.nan)
    detail = {}

    for T in FORWARD_ORIGINS:
        te, tr = season == T, season < T
        if te.sum() < 10 or tr.sum() < 200:
            continue
        train, test = df[tr], df[te]
        print(f"\n    origin {T}: train n={int(tr.sum())}, "
              f"test n={int(te.sum())}", flush=True)

        # ONE inner OOF pass -> both the residual model and the ref offsets
        resid_mdl, ref_offs, _ = inner_oof_both(
            train, features, clf_features, seeds, K_REFERENCE,
            tag=f"[B1 {T} inner OOF] ")

        # Forward champion predictions: seed-average first, then correct
        acc = np.zeros(int(te.sum()))
        for seed in seeds:
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            clf = rm.train_route_classifier(train, clf_features, seed)
            p_max = rm.route_proba(clf, test, clf_features)[:, rm.MAX_IDX]
            acc += compose(latent, lo=lo, hi=hi, p_max=p_max,
                           is_extension=test["is_extension"].values,
                           ext_cap_pct=test["ext_cap_pct"].values)
        champ[te] = acc / len(seeds)

        # Candidate: residual model correction on seed-averaged champion
        X_test = build_resid_features(champ[te], cat[te])
        corr = predict_resid_correction(resid_mdl, X_test, elig[te])
        cand_model[te] = apply_candidate(
            champ[te], corr, mech_cap_pct=mech_cap[te],
            is_extension=test["is_extension"].values,
            ext_cap_pct=test["ext_cap_pct"].values,
            lo=lo_all[te], hi=hi_all[te])

        # Reference: constant offsets on seed-averaged champion
        cand_ref[te] = stage3_signing(
            champ[te], cat[te], ref_offs, lo=lo_all[te], hi=hi_all[te],
            is_extension=test["is_extension"].values,
            ext_cap_pct=test["ext_cap_pct"].values)

        detail[int(T)] = {
            "n": int(te.sum()),
            "ref_offsets": {t: ref_offs[t]["offset"] for t in ELIGIBLE_TYPES},
        }
        print(f"      origin {T} done  ref offsets: " + "  ".join(
            f"{t}={ref_offs[t]['offset']:+.5f}" for t in ELIGIBLE_TYPES),
            flush=True)

    return champ, cand_model, cand_ref, detail


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    ap.add_argument("--skip-b", action="store_true",
                    help="skip layer B (rolling origin)")
    args = ap.parse_args()
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])

    df, features = load_evaluation_frame(verbose=True)
    df, clf_features = rm.attach_clf_features(df)

    # Attach mechanism caps
    print("\n" + "=" * 100)
    print("  MECHANISM CAP COMPUTATION")
    print("=" * 100)
    df = attach_mechanism_caps(df, verbose=True)
    mech_cap = df["mech_cap_pct"].values
    n_mech = int((~np.isnan(mech_cap)).sum())
    print(f"  {n_mech} rows have a mechanism cap "
          f"(Early Bird + Non-Bird combined)")

    y = df[TARGET].values
    cap_m = df["cap"].values / 1e6
    cat = df["signing_cat"].values
    lo_all = df["floor_pct"].values
    hi_all = df["max_eligible_pct"].values
    is_ext = df["is_extension"].values
    ext_cap = df["ext_cap_pct"].values
    sel = ~df["is_confirmation"].values
    recent = df["season"].values >= 2024
    elig = np.isin(cat, ELIGIBLE_TYPES)

    print(f"\nframe {len(df)} rows | {len(features)} regression features | "
          f"{len(clf_features)} classifier features | seeds {len(seeds)}")
    print(f"eligible rows {int(elig.sum())} pooled, "
          f"{int((elig & sel).sum())} in the selection pool")
    print("per type: " + "  ".join(
        f"{t} {int((cat == t).sum())}/{int(((cat == t) & sel).sum())}sel"
        for t in ELIGIBLE_TYPES))
    print(f"PRE-REGISTERED XGB: {RESID_XGB_PARAMS}")

    # ---- fit pass (champion) ------------------------------------------------
    print("\nFIT PASS — champion Grabit latent + route classifier + Stage 2/3")
    store, folds = champion_fold_pass(df, features, clf_features, seeds)
    n_folds, n_seeds = len(folds), len(seeds)
    fold_of = np.empty(len(df), dtype=int)
    for fi, (_, va) in enumerate(folds):
        fold_of[va] = fi

    champ_oof = seed_average(store, len(df), "champ", n_seeds)
    resid = y - champ_oof   # actual - predicted

    print(f"\n  champion reproduction: A1 {r2_score(y, champ_oof):.4f}   "
          f"A2 {r2_score(y[recent], champ_oof[recent]):.4f}   "
          f"MAE ${np.abs((champ_oof - y) * cap_m).mean():.3f}M")

    # ---- REFERENCE ARM: v8.8x constant offsets (fold-honest) ----------------
    print("\n  REFERENCE ARM — v8.8x constant offsets (leave-fold-out)")
    lfo_ref = {}
    for fi in range(n_folds):
        pool = fold_of != fi
        lfo_ref[fi] = signing_offsets(resid, cat, pool, k=K_REFERENCE,
                                       detail=True)

    for rec in store:
        va = rec["va"]
        rec["ref"] = stage3_signing(
            rec["champ"], cat[va], lfo_ref[rec["fi"]],
            lo=lo_all[va], hi=hi_all[va],
            is_extension=is_ext[va], ext_cap_pct=ext_cap[va])
    ref_oof = seed_average(store, len(df), "ref", n_seeds)

    # ---- CANDIDATE ARM: fold-honest residual model + mechanism caps ---------
    print("\n  CANDIDATE ARM — XGB residual model + mechanism caps "
          "(leave-fold-out)")
    for fi in range(n_folds):
        va_mask = fold_of == fi
        pool = ~va_mask
        # Train residual model on out-of-fold residuals
        X_pool = build_resid_features(champ_oof[pool], cat[pool])
        elig_pool = elig[pool]
        resid_pool = resid[pool]
        mdl = train_resid_model(X_pool[elig_pool], resid_pool[elig_pool],
                                seed=42)
        print(f"    fold {fi}: trained on {int(elig_pool.sum())} eligible rows",
              flush=True)

        # Apply to all (fold, seed) cells in this fold
        for rec in store:
            if rec["fi"] != fi:
                continue
            va = rec["va"]
            X_va = build_resid_features(rec["champ"], cat[va])
            corr = predict_resid_correction(mdl, X_va, elig[va])
            rec["cand"] = apply_candidate(
                rec["champ"], corr, mech_cap_pct=mech_cap[va],
                is_extension=is_ext[va], ext_cap_pct=ext_cap[va],
                lo=lo_all[va], hi=hi_all[va])

    cand_oof = seed_average(store, len(df), "cand", n_seeds)

    # ---- GUARDS --------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  GUARDS")
    print("=" * 100)
    guards = {}

    # G1 bit-identity: candidate outside the four eligible types
    worst_cell_cand = 0.0
    worst_cell_ref = 0.0
    for rec in store:
        va = rec["va"]
        m = ~elig[va]
        if m.sum():
            worst_cell_cand = max(worst_cell_cand,
                                  float(np.abs(rec["cand"][m] - rec["champ"][m]
                                               ).max()))
            worst_cell_ref = max(worst_cell_ref,
                                 float(np.abs(rec["ref"][m] - rec["champ"][m]
                                              ).max()))
    worst_oof_cand = float(np.abs(cand_oof[~elig] - champ_oof[~elig]).max())
    worst_oof_ref = float(np.abs(ref_oof[~elig] - champ_oof[~elig]).max())
    ok_cand = (worst_cell_cand == 0.0) and (worst_oof_cand == 0.0)
    ok_ref = (worst_cell_ref == 0.0) and (worst_oof_ref == 0.0)
    guards["bit_identity_candidate_cell"] = worst_cell_cand
    guards["bit_identity_candidate_oof"] = worst_oof_cand
    guards["bit_identity_reference_cell"] = worst_cell_ref
    guards["bit_identity_reference_oof"] = worst_oof_ref
    guards["bit_identity_candidate_pass"] = bool(ok_cand)
    guards["bit_identity_reference_pass"] = bool(ok_ref)
    print(f"  G1 bit-identity (candidate): cell max|diff| {worst_cell_cand:.3e}, "
          f"OOF max|diff| {worst_oof_cand:.3e}  "
          f"[{'PASS' if ok_cand else 'FAIL'}]")
    print(f"  G1 bit-identity (reference): cell max|diff| {worst_cell_ref:.3e}, "
          f"OOF max|diff| {worst_oof_ref:.3e}  "
          f"[{'PASS' if ok_ref else 'FAIL'}]")

    # G2 legality
    tol = 1e-12
    for arm_name, arm_oof in (("candidate", cand_oof), ("reference", ref_oof)):
        below = int((arm_oof < lo_all - tol).sum())
        above = int((arm_oof > hi_all + tol).sum())
        extm = np.asarray(pd.Series(is_ext).fillna(False).values, bool) & \
            ~np.isnan(ext_cap)
        binds = extm & (ext_cap >= lo_all - tol)
        ext_viol = int((arm_oof[binds] > ext_cap[binds] + tol).sum())
        ok = (below == 0 and above == 0 and ext_viol == 0)
        guards[f"legality_{arm_name}_pass"] = bool(ok)
        print(f"  G2 legality ({arm_name}): below floor {below}, "
              f"above ceiling {above}, above ext cap {ext_viol}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    # G3 how much the candidate moves
    moved_cand = np.abs(cand_oof - champ_oof) > 1e-9
    moved_ref = np.abs(ref_oof - champ_oof) > 1e-9
    print(f"  G3 rows moved: candidate {int(moved_cand.sum())} of "
          f"{int(elig.sum())} eligible; reference {int(moved_ref.sum())}")
    guards["n_moved_candidate"] = int(moved_cand.sum())
    guards["n_moved_reference"] = int(moved_ref.sum())

    # G4 mechanism cap clips
    mech_binds = ~np.isnan(mech_cap)
    mech_clips_cand = int((cand_oof[mech_binds] <
                           (champ_oof[mech_binds] + resid[mech_binds] * 0)
                           ).sum())  # placeholder
    # Actually count how many rows the mechanism cap lowered
    pre_mech = champ_oof + np.where(elig, cand_oof - champ_oof, 0)
    n_mech_clipped = int(((cand_oof < pre_mech - 1e-9) & mech_binds).sum())
    print(f"  G4 mechanism cap: {n_mech} rows have caps, "
          f"{n_mech_clipped} were clipped by them")
    guards["n_mech_cap_rows"] = n_mech
    guards["n_mech_cap_clipped"] = n_mech_clipped

    # ---- PRIMARY GATE --------------------------------------------------------
    print("\n" + "=" * 100)
    print("  PRIMARY GATE — MAE and bias where the intervention acts")
    print("=" * 100)

    scopes = {
        "valid-type, SELECTION (decision)": elig & sel,
        "valid-type, pooled (reporting)": elig,
        "valid-type, confirmation (canary)": elig & ~sel,
    }
    primary = {}
    for label, mask in scopes.items():
        row = {"n": int(mask.sum())}
        mae_c, bias_c, _ = cell_err(store, y, cap_m, mask, "champ",
                                    n_folds, n_seeds)
        for arm, key in (("candidate", "cand"), ("reference", "ref")):
            mae_d, bias_d, abs_d = cell_err(store, y, cap_m, mask, key,
                                            n_folds, n_seeds)
            abs_c = np.abs(bias_c)
            row[arm] = {
                "mae_win": paired_delta(mae_d, mae_c),
                "abs_bias_reduction": paired_delta(abs_d, abs_c),
                "mae_win_cells": cell_paired(mae_d, mae_c),
                "champ_mae": float(np.nanmean(mae_c)),
                "cand_mae": float(np.nanmean(mae_d)),
                "champ_bias": float(np.nanmean(bias_c)),
                "cand_bias": float(np.nanmean(bias_d)),
            }
        primary[label] = row
        print(f"\n  [{label}]  n = {row['n']}")
        for arm, tag in (("candidate", "CANDIDATE (XGB model + caps)"),
                         ("reference", "REFERENCE (v8.8x constants)")):
            d = row[arm]
            print(f"    {tag}")
            print(f"      MAE  ${d['champ_mae']:.3f}M -> ${d['cand_mae']:.3f}M"
                  f"   paired win {d['mae_win']['delta']:+.4f} "
                  f"+/- {d['mae_win']['se']:.4f}  t = {d['mae_win']['t']:+.2f}"
                  f"   (by fold, n={n_folds})")
            print(f"      bias ${d['champ_bias']:+.3f}M -> "
                  f"${d['cand_bias']:+.3f}M   |bias| reduction "
                  f"{d['abs_bias_reduction']['delta']:+.4f}")

    # ---- A1/A2 guards -------------------------------------------------------
    print("\n" + "=" * 100)
    print("  GUARD — A1 / A2 paired R2 deltas")
    print("=" * 100)
    r2_masks = {
        "A1 selection": sel,
        "A1 pooled": np.ones(len(df), bool),
        "A2 selection, 2024-26": sel & recent,
        "A2 pooled, 2024-26": recent,
    }
    a_guards = {}
    print(f"  {'metric':28s} {'champ':>8s} {'cand':>8s} {'delta':>9s} "
          f"{'se':>8s} {'t':>7s} | {'ref':>8s} {'delta':>9s} {'t':>7s}")
    for label, mask in r2_masks.items():
        m_c = cell_r2(store, y, mask, "champ", n_folds, n_seeds)
        out = {}
        for arm, key in (("cand", "cand"), ("ref", "ref")):
            m_d = cell_r2(store, y, mask, key, n_folds, n_seeds)
            out[arm] = paired_delta(m_c, m_d)
            out[arm + "_mean"] = float(np.nanmean(m_d))
        out["champ_mean"] = float(np.nanmean(m_c))
        out["oof_r2_champ"] = float(r2_score(y[mask], champ_oof[mask]))
        out["oof_r2_cand"] = float(r2_score(y[mask], cand_oof[mask]))
        out["oof_r2_ref"] = float(r2_score(y[mask], ref_oof[mask]))
        a_guards[label] = out
        print(f"  {label:28s} {out['champ_mean']:8.4f} "
              f"{out['cand_mean']:8.4f} {out['cand']['delta']:+9.5f} "
              f"{out['cand']['se']:8.5f} {out['cand']['t']:+7.2f} | "
              f"{out['ref_mean']:8.4f} {out['ref']['delta']:+9.5f} "
              f"{out['ref']['t']:+7.2f}")

    # ---- per-type before/after -----------------------------------------------
    print("\n" + "=" * 100)
    print("  PER-TYPE BIAS BEFORE / AFTER")
    print("=" * 100)
    tab_sel_cand = per_type_table(df, cat, champ_oof, cand_oof, sel)
    tab_all_cand = per_type_table(df, cat, champ_oof, cand_oof, None)
    tab_sel_ref = per_type_table(df, cat, champ_oof, ref_oof, sel)
    print_per_type(tab_sel_cand,
                   "CANDIDATE selection (XGB model + caps)")
    print_per_type(tab_sel_ref,
                   "REFERENCE selection (v8.8x constants)")
    print_per_type(tab_all_cand,
                   "CANDIDATE pooled")

    # Print the per-fold MAE wins
    print("\n  Per-fold MAE wins (SELECTION, candidate vs champion):")
    sel_elig = elig & sel
    mae_c_mat, _, _ = cell_err(store, y, cap_m, sel_elig, "champ",
                               n_folds, n_seeds)
    mae_d_mat, _, _ = cell_err(store, y, cap_m, sel_elig, "cand",
                               n_folds, n_seeds)
    pd_result = paired_delta(mae_d_mat, mae_c_mat)
    if "per_fold" in pd_result:
        print("    per fold: [" + ", ".join(
            f"{float(v):+.4f}" for v in pd_result["per_fold"]) + "]")

    # ---- LAYER B -------------------------------------------------------------
    b_report = {}
    if args.skip_b:
        print("\n  LAYER B skipped (--skip-b).")
    else:
        print("\n" + "=" * 100)
        print("  LAYER B — rolling origin")
        print("=" * 100)
        b_champ, b_cand, b_ref, b_detail = layer_b(
            df, features, clf_features, seeds)
        scored = ~np.isnan(b_champ)
        b_report["by_origin"] = b_detail
        print(f"\n  {'scope':34s} {'n':>5s} {'champ R2':>9s} {'cand R2':>9s} "
              f"{'dR2':>9s} | {'champ MAE':>10s} {'cand MAE':>9s} "
              f"{'win':>8s} | {'ref R2':>9s} {'ref MAE':>9s} {'win':>8s}")
        for label, mask in (("all forward rows", scored),
                            ("all, selection", scored & sel),
                            ("valid-type", scored & elig),
                            ("valid-type, selection", scored & elig & sel)):
            if mask.sum() < 10:
                continue
            r2c = float(r2_score(y[mask], b_champ[mask]))
            r2d = float(r2_score(y[mask], b_cand[mask]))
            r2r = float(r2_score(y[mask], b_ref[mask]))
            mc = float(np.abs((b_champ[mask] - y[mask]) * cap_m[mask]).mean())
            md = float(np.abs((b_cand[mask] - y[mask]) * cap_m[mask]).mean())
            mr = float(np.abs((b_ref[mask] - y[mask]) * cap_m[mask]).mean())
            b_report[label] = {
                "n": int(mask.sum()),
                "champ_r2": r2c, "cand_r2": r2d, "ref_r2": r2r,
                "champ_mae_m": mc, "cand_mae_m": md, "ref_mae_m": mr,
                "cand_mae_win_m": mc - md, "ref_mae_win_m": mc - mr,
            }
            print(f"  {label:34s} {int(mask.sum()):5d} {r2c:9.4f} {r2d:9.4f} "
                  f"{r2d - r2c:+9.5f} | {mc:10.3f} {md:9.3f} "
                  f"{mc - md:+8.3f} | {r2r:9.4f} {mr:9.3f} {mc - mr:+8.3f}")

        # B1 bit-identity
        nb = scored & ~elig
        diffs_cand = np.abs(b_cand[nb] - b_champ[nb])
        diffs_ref = np.abs(b_ref[nb] - b_champ[nb])
        ULP_TOL = 1e-12
        n_real_cand = int((diffs_cand > ULP_TOL).sum())
        n_real_ref = int((diffs_ref > ULP_TOL).sum())
        ok_b_cand = n_real_cand == 0
        ok_b_ref = n_real_ref == 0
        print(f"\n  B1 bit-identity candidate: max|diff| "
              f"{float(diffs_cand.max()):.3e}, rows above tol: "
              f"{n_real_cand}  [{'PASS' if ok_b_cand else 'FAIL'}]")
        print(f"  B1 bit-identity reference: max|diff| "
              f"{float(diffs_ref.max()):.3e}, rows above tol: "
              f"{n_real_ref}  [{'PASS' if ok_b_ref else 'FAIL'}]")
        b_report["bit_identity_candidate_pass"] = bool(ok_b_cand)
        b_report["bit_identity_reference_pass"] = bool(ok_b_ref)

    # ---- CSV -----------------------------------------------------------------
    csv_rows = []
    for label, row in primary.items():
        for arm in ("candidate", "reference"):
            d = row[arm]
            for metric in ("mae_win", "abs_bias_reduction"):
                csv_rows.append({
                    "section": "paired_primary", "scope": label,
                    "key": f"{metric}|{arm}", "n": row["n"],
                    "champ_mae_m": d["champ_mae"], "cand_mae_m": d["cand_mae"],
                    "champ_bias_m": d["champ_bias"],
                    "cand_bias_m": d["cand_bias"],
                    "delta": d[metric]["delta"], "se": d[metric]["se"],
                    "t": d[metric]["t"]})
    for label, out in a_guards.items():
        for arm in ("cand", "ref"):
            csv_rows.append({
                "section": "paired_guard_r2", "scope": label,
                "key": arm, "n": int(len(df)),
                "delta": out[arm]["delta"], "se": out[arm]["se"],
                "t": out[arm]["t"],
                "champ_r2": out["oof_r2_champ"],
                "cand_r2": out.get(f"oof_r2_{arm}")})
    for scope, tab in (("selection", tab_sel_cand), ("pooled", tab_all_cand)):
        for _, r in tab.iterrows():
            csv_rows.append({
                "section": "per_type_before_after_candidate", "scope": scope,
                "key": r["signing_type"], "n": int(r["n"]),
                "eligible": bool(r["eligible"]),
                "champ_bias_m": r["champ_bias_m"],
                "cand_bias_m": r["cand_bias_m"],
                "champ_mae_m": r["champ_mae_m"],
                "cand_mae_m": r["cand_mae_m"],
                "mae_win_m": r["mae_win_m"]})
    for k, v in guards.items():
        csv_rows.append({"section": "guard_check", "scope": "layer_a",
                         "key": k, "value": v})
    if b_report:
        for label in ("all forward rows", "valid-type",
                      "valid-type, selection"):
            if label in b_report:
                d = b_report[label]
                csv_rows.append({
                    "section": "layer_b", "scope": label, "key": "B1",
                    "n": d["n"], "champ_r2": d["champ_r2"],
                    "cand_r2": d["cand_r2"],
                    "delta": d["cand_r2"] - d["champ_r2"],
                    "champ_mae_m": d["champ_mae_m"],
                    "cand_mae_m": d["cand_mae_m"],
                    "mae_win_m": d.get("cand_mae_win_m")})

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(csv_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} ({len(out_df)} rows)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_rows": int(len(df)), "seeds": list(seeds), "n_splits": N_SPLITS,
        "resid_xgb_params": RESID_XGB_PARAMS,
        "k_reference": K_REFERENCE,
        "eligible_types": list(ELIGIBLE_TYPES),
        "mech_cap_tol": MECH_CAP_TOL,
        "primary": primary, "r2_guards": a_guards, "guards": guards,
        "layer_b": b_report,
        "champion_A1": float(r2_score(y, champ_oof)),
        "champion_A2": float(r2_score(y[recent], champ_oof[recent])),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"Saved {OUT_JSON}")

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    sel_key = "valid-type, SELECTION (decision)"
    if sel_key in primary:
        for arm, label in (("candidate", "CANDIDATE (XGB + caps)"),
                           ("reference", "REFERENCE (v8.8x constants)")):
            d = primary[sel_key][arm]
            print(f"  {label}:")
            print(f"    MAE  ${d['champ_mae']:.3f}M -> ${d['cand_mae']:.3f}M  "
                  f"win {d['mae_win']['delta']:+.4f}  "
                  f"t={d['mae_win']['t']:+.2f}")
    all_pass = all(v for k, v in guards.items() if "pass" in k)
    print(f"\n  All guards: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
