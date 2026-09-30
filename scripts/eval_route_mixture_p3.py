"""Evidence harness for route-mixture phase 3 (2026-07-26-route-mixture-p3 brief).

Phase 2 concluded the purity-gated max branch was STRUCTURALLY unwinnable. That
verdict was an artifact of the ISSUES #19 mis-tiering bug, which phase 2's pin
predates. This harness re-judges the branch on the corrected v7.12x labels
(944 rows, max zone n=68) under a re-registered tau rule.

  Part 1  Purity curves for BOTH classifiers (base = FEATURE_COLS only,
          enriched = + the 20 classifier-only batch columns), and two operating
          points per classifier, both chosen from PROBABILITY SPACE ONLY:
            tau*   = smallest tau whose OOF purity is 1.000 (zero collateral),
                     tau <= 0.95; fall back to the highest-purity tau.
            tau90  = smallest tau whose OOF purity is >= 0.90 (the old rule).
          Honest win ceilings for all four operating points are computed and
          printed BEFORE any branch arm runs.

  Part 2  The four-cell battery {base, enriched} x {tau*, tau90}, ship form
          push_clip_gated (hard_gated computed alongside; it is free once the
          fits exist). Win / brakes / guards, per-fold dSel, C2 fixed segments,
          B1 forward, collateral lists.

  Part 3  The touched-row table for every cell: which true maxes the adoption
          would actually be buying, with P, champion error and realized gain.

Everything is fold-honest: for each (fold, seed) the champion regression and
BOTH route classifiers are fit on the training slice only; P(max) is applied to
the test-slice latent as an OUTPUT weight and never joins FEATURE_COLS. One fit
pass serves Part 1 and Part 2, so the P used to pick tau is bit-identical to the
P the branch runs on.

Two segment metrics are reported in BOTH readings, deliberately (see the RESULT
document's anomalies section):

  * 25%+ predicted band — phase 2 compared each model's OWN band, which the
    worker brief forbids for model-vs-model comparison ("fixed row groups").
    Reported both ways: FIXED (champion's 25%+ rows, scored under both models)
    and OWN (phase-2 comparable).
  * C2 segments — "|bias| growth" (the brief's literal wording) and
    "|signed-bias change|" (phase 2's implementation) are not the same number
    when a segment's bias crosses toward zero. Both are printed.

Run:  OMP_NUM_THREADS=6 python scripts/eval_route_mixture_p3.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import OUTPUTS_DIR
from src.model.train import FEATURE_COLS
from src.model.evaluate_suite import (
    load_evaluation_frame, paired_delta, _dollars, N_SPLITS, DEFAULT_SEEDS,
    TARGET, FORWARD_ORIGINS,
)
from src.model import route_mixture as rm

SEEDS = DEFAULT_SEEDS
MARGIN = 1.05                 # fixed constant, never tuned on a zone metric
PURITY_TARGET = 0.90          # the OLD rule, reported as tau90
TAU_CAP = 0.95
TAU_GRID = np.round(np.arange(0.30, 0.96, 0.01), 2)

WIN_BAR = 0.50                # $M zone-MAE improvement
BRAKE_BAR = 0.30              # $M signed-bias growth, each brake
C2_BAR = 0.30                 # $M per fixed segment
B1_BAR = 0.003                # forward R2 drop
DSEL_T_BAR = -2.0

OUT = Path(__file__).resolve().parent.parent / "outputs" / "models"


# ---------------------------------------------------------------------------
# One fit pass: champion latent + both classifiers, per (seed, fold)
# ---------------------------------------------------------------------------

def fit_pass(df, features, clf_features, labels, seeds=SEEDS):
    """Fit everything once; return the pieces every later metric is derived from.

    Returns:
        store        list of dicts, one per (seed, fold): va index, latent, lo,
                     hi, and the two classifiers' full (n_va, 4) probabilities
        proba        {"base": (n,4), "enriched": (n,4)} seed-averaged OOF
        champ_oof    seed-averaged champion out-of-fold prediction
        frs_champ    (n_folds, n_seeds) selection-pool R2 for the champion
        folds        the fold list (reused by nothing else, kept for the record)
    """
    y = df[TARGET].values
    groups = df["player_name_norm"].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, y, groups))
    sel = ~df["is_confirmation"].values
    base_feats = list(FEATURE_COLS)

    store = []
    acc_champ = np.zeros(len(df))
    acc_proba = {k: np.zeros((len(df), len(rm.ROUTE_CLASSES)))
                 for k in ("base", "enriched")}
    frs_champ = np.zeros((len(folds), len(seeds)))

    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(folds):
            train, test = df.iloc[tr], df.iloc[va]
            latent, lo, hi = rm.grabit_latent(train, test, features, seed,
                                                max_term_on=False)
            champ = np.clip(latent, lo, hi)

            clf_b = rm.train_route_classifier(train, base_feats, seed,
                                              labels=labels[tr])
            pr_b = rm.route_proba(clf_b, test, base_feats)
            clf_e = rm.train_route_classifier(train, clf_features, seed,
                                              labels=labels[tr])
            pr_e = rm.route_proba(clf_e, test, clf_features)

            store.append({"si": si, "fi": fi, "va": va, "latent": latent,
                          "lo": lo, "hi": hi, "champ": champ,
                          "p_base": pr_b[:, rm.MAX_IDX],
                          "p_enriched": pr_e[:, rm.MAX_IDX]})
            acc_champ[va] += champ
            acc_proba["base"][va] += pr_b
            acc_proba["enriched"][va] += pr_e
            vs = sel[va]
            frs_champ[fi, si] = (r2_score(y[va][vs], champ[vs])
                                 if vs.sum() > 10 else np.nan)
        print(f"    seed {seed} done ({si + 1}/{len(seeds)})", flush=True)

    n_s = len(seeds)
    return (store, {k: v / n_s for k, v in acc_proba.items()},
            acc_champ / n_s, frs_champ, folds)


def arm_from_store(df, store, p_key, tau, arm, n_folds, n_seeds):
    """Derive one arm's OOF predictions + selection fold-R2 from the fit pass.

    Pure numpy on cached latents: no refitting, so every cell is scored on
    bit-identical folds, seeds, latents and probabilities.
    """
    y = df[TARGET].values
    sel = ~df["is_confirmation"].values
    acc = np.zeros(len(df))
    frs = np.zeros((n_folds, n_seeds))
    for rec in store:
        va, latent, lo, hi = rec["va"], rec["latent"], rec["lo"], rec["hi"]
        champ, p = rec["champ"], rec[p_key]
        gate = p >= tau
        if arm == "push_clip":
            pushed = latent + p * (MARGIN * hi - latent)
            adj = np.where(gate, pushed, champ)
        elif arm == "hard":
            adj = np.where(gate, hi, champ)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        pred = np.clip(adj, lo, hi)
        acc[va] += pred
        vs = sel[va]
        frs[rec["fi"], rec["si"]] = (r2_score(y[va][vs], pred[vs])
                                     if vs.sum() > 10 else np.nan)
    return acc / n_seeds, frs


# ---------------------------------------------------------------------------
# tau selection — probability space only, never a zone metric
# ---------------------------------------------------------------------------

def purity_curve(p, is_max, grid=TAU_GRID):
    curve = []
    for t in grid:
        m = p >= t
        n = int(m.sum())
        curve.append({"tau": float(t), "n_at_or_above": n,
                      "n_true_max": int(is_max[m].sum()),
                      "purity": float(is_max[m].mean()) if n else float("nan")})
    return curve


def choose_tau_pure(curve):
    """tau* = smallest tau with purity exactly 1.000 (no non-max above it).

    Purity is monotone once it reaches 1.0 (the P>=tau set only shrinks), so
    "smallest" is well defined and everything above it is also collateral-free.
    Falls back to the highest-purity tau when 1.000 is unattainable.
    """
    for row in curve:
        if row["tau"] <= TAU_CAP and row["n_at_or_above"] > 0 \
                and row["purity"] >= 1.0 - 1e-12:
            return row["tau"], row, "purity==1.000"
    best = max((r for r in curve if r["tau"] <= TAU_CAP and r["n_at_or_above"]),
               key=lambda r: r["purity"], default=None)
    if best is None:
        return None, None, "no tau with any rows above it"
    return best["tau"], best, f"fallback: highest purity {best['purity']:.3f}"


def choose_tau_90(curve):
    """tau90 = the OLD rule: smallest tau with purity >= 90%, tau <= 0.95."""
    for row in curve:
        if row["tau"] <= TAU_CAP and row["n_at_or_above"] > 0 \
                and row["purity"] >= PURITY_TARGET:
            return row["tau"], row
    return None, None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def zone_masks(df):
    y = df[TARGET].values
    me = df["max_eligible_pct"].values
    true_max = y >= 0.90 * me                       # target-defined: fixed rows
    counter = (~true_max) & (y >= 0.70 * me) & (y < 0.90 * me)
    return true_max, counter


def honest_ceiling(df, champ_oof, p_oof, tau, true_max):
    """Max POSSIBLE true-max zone-MAE improvement at this tau, before any run.

    Every touched max could at best have its whole champion error erased; the
    zone MAE is averaged over the whole zone, so the ceiling is
    sum|champ err on touched maxes| / n_zone.
    """
    y = df[TARGET].values
    cap_m = df["cap"].values / 1e6
    touched = true_max & (p_oof >= tau)
    err = (champ_oof - y) * cap_m
    return (float(np.abs(err[touched]).sum() / true_max.sum()),
            int(touched.sum()), touched)


def class_diagnostics(proba, labels, tag):
    out = {}
    print(f"\n  --- {tag} ---")
    for cls, idx in (("max", rm.MAX_IDX), ("floor", rm.FLOOR_IDX),
                     ("mle", rm.MLE_IDX)):
        true = (labels == idx).astype(int)
        p = proba[:, idx]
        auc = float(roc_auc_score(true, p))
        med = float(np.median(p[true == 1]))
        edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        cal = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (p >= a) & (p < b)
            if m.sum() >= 3:
                cal.append([f"[{a:.1f},{b:.1f})", int(m.sum()),
                            round(float(p[m].mean()), 3),
                            round(float(true[m].mean()), 3)])
        d = {"auc": auc, "median_p_on_true": med, "n_true": int(true.sum()),
             "calibration": cal}
        extra = ""
        if cls == "max":
            d["non_max_above_0.5"] = int(((labels != idx) & (p > 0.5)).sum())
            extra = f"  non-max>0.5={d['non_max_above_0.5']}"
        out[cls] = d
        print(f"    P({cls}): AUC {auc:.4f}  median_on_true {med:.4f}  "
              f"n_true={int(true.sum())}{extra}")
        if cls == "max":
            for row in cal:
                print(f"        {row[0]:12s} n={row[1]:4d} meanP={row[2]:.3f} "
                      f"emp={row[3]:.3f}")
    return out


def band25_metrics(df, champ_oof, cand_oof):
    """25%+ predicted band, both readings.

    FIXED: the champion's 25%+ rows, scored under both models — the worker
    brief's rule for model-vs-model segment comparison.
    OWN:   each model's own 25%+ rows — what phase 2 measured; kept so the
           architect's quoted +$1.08M is reproducible.
    """
    fixed = champ_oof >= 0.25
    _, b_ch_fixed = _dollars(df, champ_oof, fixed)
    _, b_cd_fixed = _dollars(df, cand_oof, fixed)
    own_ch, own_cd = champ_oof >= 0.25, cand_oof >= 0.25
    _, b_ch_own = _dollars(df, champ_oof, own_ch)
    _, b_cd_own = _dollars(df, cand_oof, own_cd)
    return {"fixed_n": int(fixed.sum()), "fixed_champ": b_ch_fixed,
            "fixed_cand": b_cd_fixed, "fixed_growth": b_cd_fixed - b_ch_fixed,
            "own_n_champ": int(own_ch.sum()), "own_n_cand": int(own_cd.sum()),
            "own_champ": b_ch_own, "own_cand": b_cd_own,
            "own_growth": b_cd_own - b_ch_own}


def c2_segments(df, champ_oof, cand_oof):
    """Fixed-row signing-mechanism segments, both growth readings."""
    segs = {}
    if "signing_cat" not in df.columns:
        return segs
    for cat, sub in df.groupby("signing_cat"):
        if len(sub) < 10:
            continue
        m = (df["signing_cat"] == cat).values
        _, b_ch = _dollars(df, champ_oof, m)
        _, b_cd = _dollars(df, cand_oof, m)
        segs[str(cat)] = {
            "n": int(m.sum()), "champ": b_ch, "cand": b_cd,
            "signed_change": b_cd - b_ch,
            "abs_bias_growth": abs(b_cd) - abs(b_ch),
        }
    return segs


# ---------------------------------------------------------------------------
# B1 forward — one fit pass, all cells derived from it
# ---------------------------------------------------------------------------

def forward_pass(df, features, clf_features, labels, cells, seeds=SEEDS,
                 origins=FORWARD_ORIGINS):
    """Rolling-origin forward predictions for the champion and every cell.

    Mirrors evaluate_suite.rolling_forward exactly (train on season < T, score
    season T, seed-averaged), but fits the regression and both classifiers ONCE
    per (origin, seed) and derives every cell from the cached pieces — same
    numbers as calling rolling_forward with make_maxbranch_fitter per cell, at
    a fifth of the compute.
    """
    base_feats = list(FEATURE_COLS)
    season = df["season"].values
    preds = {"champion": np.full(len(df), np.nan)}
    for name in cells:
        preds[name] = np.full(len(df), np.nan)

    for T in origins:
        test_mask, train_mask = season == T, season < T
        if test_mask.sum() < 10 or train_mask.sum() < 200:
            continue
        train, test = df[train_mask], df[test_mask]
        lab_tr = labels[train_mask]
        acc = {k: np.zeros(int(test_mask.sum())) for k in preds}
        for seed in seeds:
            latent, lo, hi = rm.grabit_latent(train, test, features, seed,
                                                max_term_on=False)
            champ = np.clip(latent, lo, hi)
            p = {}
            clf_b = rm.train_route_classifier(train, base_feats, seed,
                                              labels=lab_tr)
            p["base"] = rm.route_proba(clf_b, test, base_feats)[:, rm.MAX_IDX]
            clf_e = rm.train_route_classifier(train, clf_features, seed,
                                              labels=lab_tr)
            p["enriched"] = rm.route_proba(clf_e, test,
                                           clf_features)[:, rm.MAX_IDX]
            acc["champion"] += champ
            for name, (p_key, tau, arm) in cells.items():
                pv = p[p_key]
                gate = pv >= tau
                if arm == "push_clip":
                    adj = np.where(gate, latent + pv * (MARGIN * hi - latent),
                                   champ)
                else:
                    adj = np.where(gate, hi, champ)
                acc[name] += np.clip(adj, lo, hi)
        for k in preds:
            preds[k][test_mask] = acc[k] / len(seeds)
        print(f"    origin {T} done (train n={int(train_mask.sum())}, "
              f"test n={int(test_mask.sum())})", flush=True)
    return preds


# ---------------------------------------------------------------------------

def main():
    df, features = load_evaluation_frame()
    df, clf_features = rm.attach_clf_features(df)
    y = df[TARGET].values
    cap_m = df["cap"].values / 1e6
    labels = rm.compute_route_labels(df)
    is_max_lbl = labels == rm.MAX_IDX
    true_max, counter = zone_masks(df)
    sel_mask = ~df["is_confirmation"].values

    print(f"\nLoaded {len(df)} rows | regression features {len(features)} | "
          f"classifier features {len(clf_features)} "
          f"({len(rm.CLF_EXTRA_COLS)} enrichment cols, native NaN)")
    print(f"route labels: max={int(is_max_lbl.sum())} "
          f"floor={int((labels == rm.FLOOR_IDX).sum())} "
          f"mle={int((labels == rm.MLE_IDX).sum())} "
          f"continuous={int((labels == 0).sum())}")
    print(f"true-max zone n={int(true_max.sum())}  "
          f"counterweight band n={int(counter.sum())}")

    report = {"n_rows": len(df), "regression_features": features,
              "classifier_features": clf_features,
              "enrichment_cols": rm.CLF_EXTRA_COLS,
              "n_true_max_zone": int(true_max.sum()),
              "n_counterweight": int(counter.sum()),
              "bars": {"win_m": WIN_BAR, "brake_m": BRAKE_BAR, "c2_m": C2_BAR,
                       "b1_drop": B1_BAR, "dsel_t": DSEL_T_BAR}}

    # ---- fit pass ------------------------------------------------------
    print("\n" + "=" * 74)
    print("  FIT PASS — champion latent + base & enriched classifiers "
          f"({len(SEEDS)} seeds x {N_SPLITS} folds)")
    print("=" * 74)
    store, proba, champ_oof, frs_champ, folds = fit_pass(
        df, features, clf_features, labels)
    n_folds, n_seeds = len(folds), len(SEEDS)

    ref_path = OUTPUTS_DIR / "models" / "evaluation_suite.json"
    repro = None
    if ref_path.exists():
        stored = json.load(open(ref_path))
        ref = np.array(stored["fold_r2_selection"]["champion"])
        if ref.shape == frs_champ.shape:
            repro = float(np.abs(frs_champ - ref).max())
            print(f"\n  champion fold_r2_selection reproduction max|diff| vs "
                  f"stored suite: {repro:.2e}")
        else:
            print(f"\n  champion reproduction check SKIPPED: stored matrix "
                  f"{ref.shape} vs this run {frs_champ.shape} (seed count)")
        print(f"  stored suite: A1 {stored['champion']['A1_cv_r2']:.4f}  "
              f"A2 {stored['champion']['A2_cv_r2_2024_26']:.4f}  "
              f"B1 {stored['champion']['B1_forward_r2']:.4f}  "
              f"zone n={stored['grabit_zone']['n']} "
              f"MAE ${stored['grabit_zone']['mae_grabit']:.2f}M")
    report["champion_repro_max_absdiff"] = repro

    # ---- PART 1 --------------------------------------------------------
    print("\n" + "=" * 74)
    print("  PART 1 — classifier diagnostics and purity curves")
    print("=" * 74)
    for key, tag in (("base", "BASE classifier (FEATURE_COLS only)"),
                     ("enriched", "ENRICHED classifier (FEATURE_COLS + batch)")):
        report[f"{key}_classifier"] = class_diagnostics(proba[key], labels, tag)

    p_oof = {k: proba[k][:, rm.MAX_IDX] for k in proba}
    curves = {k: purity_curve(p_oof[k], is_max_lbl) for k in p_oof}
    report["purity_curve_base"] = curves["base"]
    report["purity_curve_enriched"] = curves["enriched"]

    print("\n  PURITY CURVE (cumulative: true-max rate among rows with P>=tau)")
    print(f"    {'tau':>5s} | {'BASE n':>7s} {'nMax':>5s} {'purity':>7s} | "
          f"{'ENR n':>6s} {'nMax':>5s} {'purity':>7s}")
    bmap = {r["tau"]: r for r in curves["base"]}
    emap = {r["tau"]: r for r in curves["enriched"]}
    for t in np.round(np.arange(0.50, 0.96, 0.02), 2):
        b, e = bmap.get(float(t)), emap.get(float(t))
        bp = f"{b['purity']:.3f}" if b and b["n_at_or_above"] else "  -  "
        ep = f"{e['purity']:.3f}" if e and e["n_at_or_above"] else "  -  "
        print(f"    {t:5.2f} | {b['n_at_or_above']:7d} {b['n_true_max']:5d} "
              f"{bp:>7s} | {e['n_at_or_above']:6d} {e['n_true_max']:5d} {ep:>7s}")

    # champion zone references, fixed rows
    c_mae_max, c_bias_max = _dollars(df, champ_oof, true_max)
    c_mae_cw, c_bias_cw = _dollars(df, champ_oof, counter)
    c_a1 = float(r2_score(y, champ_oof))
    recent = df["season"].values >= 2024
    c_a2 = float(r2_score(y[recent], champ_oof[recent]))
    champ_err = (champ_oof - y) * cap_m
    print(f"\n  CHAMPION references on the fixed v7.12x rows:")
    print(f"    A1 {c_a1:.4f}   A2 {c_a2:.4f}")
    print(f"    true-max zone (n={int(true_max.sum())}): MAE ${c_mae_max:.2f}M "
          f"bias ${c_bias_max:+.2f}M")
    print(f"    counterweight [0.70,0.90) non-max (n={int(counter.sum())}): "
          f"bias ${c_bias_cw:+.2f}M MAE ${c_mae_cw:.2f}M")
    print(f"    25%+ predicted band (n={int((champ_oof >= 0.25).sum())}): "
          f"bias ${_dollars(df, champ_oof, champ_oof >= 0.25)[1]:+.2f}M")
    report["champion"] = {
        "A1": c_a1, "A2": c_a2, "true_max_mae": c_mae_max,
        "true_max_bias": c_bias_max, "cw_bias": c_bias_cw, "cw_mae": c_mae_cw,
        "band25_n": int((champ_oof >= 0.25).sum()),
        "band25_bias": _dollars(df, champ_oof, champ_oof >= 0.25)[1]}

    for key in ("base", "enriched"):
        rho = spearmanr(p_oof[key][true_max],
                        np.abs(champ_err[true_max])).statistic
        print(f"    Spearman(P_{key}, |champ err|) over the {int(true_max.sum())} "
              f"maxes = {rho:+.3f}")
        report[f"spearman_p_vs_abserr_{key}"] = float(rho)

    # ---- operating points, chosen in probability space -----------------
    print("\n" + "=" * 74)
    print("  PART 1b — operating points (probability space only)")
    print("    tau*  = smallest tau with purity 1.000 (zero collateral), <=0.95")
    print("    tau90 = smallest tau with purity >= 0.90 (the OLD rule)")
    print("=" * 74)

    ops = {}
    for key in ("base", "enriched"):
        t_star, row_star, how = choose_tau_pure(curves[key])
        t_90, row_90 = choose_tau_90(curves[key])
        for label, tau, row, note in (("tau*", t_star, row_star, how),
                                      ("tau90", t_90, row_90, "purity>=0.90")):
            if tau is None:
                print(f"    {key:9s} {label:6s}  UNREACHABLE ({note})")
                continue
            ceil, n_touch, touched = honest_ceiling(df, champ_oof, p_oof[key],
                                                    tau, true_max)
            n_coll = int(((p_oof[key] >= tau) & ~true_max).sum())
            name = f"{key}_{label}"
            ops[name] = {"clf": key, "point": label, "tau": tau,
                         "purity": row["purity"], "n_above": row["n_at_or_above"],
                         "n_true_max_above": row["n_true_max"],
                         "n_touched_true_max": n_touch,
                         "n_collateral": n_coll,
                         "honest_ceiling_m": ceil, "rule": note}
            print(f"    {key:9s} {label:6s} tau={tau:.2f}  purity "
                  f"{row['purity']:.3f}  n>=tau {row['n_at_or_above']:3d} "
                  f"({row['n_true_max']:2d} true max, {n_coll} non-max)  "
                  f"touched maxes {n_touch:2d}/{int(true_max.sum())}  "
                  f"HONEST CEILING ${ceil:.2f}M  [{note}]")
    report["operating_points"] = ops
    print(f"\n    (the $ {WIN_BAR:.2f}M win bar is compared against these "
          f"ceilings BEFORE any arm is scored)")

    # Diagnostic only — NOT a selection device. Shows, for every tau, what a
    # gate there could at best win and how much collateral it would admit. Read
    # it to understand why the two registered rules land where they do; do NOT
    # read a tau off it (that is selecting tau on a champion-error quantity,
    # the exact move three prior experiments died on).
    print("\n  CEILING-vs-PURITY DIAGNOSTIC (not a selection device):")
    ceiling_tables = {}
    for key in ("base", "enriched"):
        p = p_oof[key]
        rows = []
        print(f"    --- {key} ---")
        print(f"      {'tau':>5s} {'nAbove':>6s} {'nMax':>5s} {'purity':>7s} "
              f"{'ceiling$M':>10s} {'nCollat':>8s}")
        for t in np.round(np.arange(0.95, 0.69, -0.01), 2):
            m = p >= float(t)
            n, nmax = int(m.sum()), int((m & true_max).sum())
            if not n:
                continue
            ceil = float(np.abs(champ_err[m & true_max]).sum() / true_max.sum())
            rows.append({"tau": float(t), "n": n, "n_max": nmax,
                         "purity": nmax / n, "ceiling_m": ceil,
                         "n_collateral": n - nmax})
            print(f"      {t:5.2f} {n:6d} {nmax:5d} {nmax / n:7.3f} "
                  f"{ceil:10.3f} {n - nmax:8d}")
        ceiling_tables[key] = rows
    report["ceiling_diagnostic"] = ceiling_tables

    # dedupe: a cell whose tau equals another cell's is the same experiment
    cells = {}
    seen = {}
    for name, op in ops.items():
        sig = (op["clf"], op["tau"])
        if sig in seen:
            op["duplicate_of"] = seen[sig]
            print(f"    NOTE: {name} has the same (classifier, tau) as "
                  f"{seen[sig]} — identical cell, scored once")
            continue
        seen[sig] = name
        cells[name] = (op["clf"], op["tau"], "push_clip")

    # ---- PART 2 --------------------------------------------------------
    print("\n" + "=" * 74)
    print("  PART 2 — the gated max branch at every operating point")
    print("=" * 74)

    arms = {}
    for name, (p_key, tau, _) in cells.items():
        for arm in ("push_clip", "hard"):
            oof, frs = arm_from_store(df, store, f"p_{p_key}", tau, arm,
                                      n_folds, n_seeds)
            mae_max, bias_max = _dollars(df, oof, true_max)
            _, bias_cw = _dollars(df, oof, counter)
            b25 = band25_metrics(df, champ_oof, oof)
            pd_ = paired_delta(frs_champ, frs)
            arms[f"{name}__{arm}"] = {
                "cell": name, "arm": arm, "clf": p_key, "tau": tau,
                "oof": oof,
                "true_max_mae": mae_max, "d_true_max_mae": mae_max - c_mae_max,
                "true_max_bias": bias_max,
                "cw_bias": bias_cw, "d_cw_bias": bias_cw - c_bias_cw,
                "band25": b25,
                "dSel": pd_["delta"], "dSel_se": pd_["se"], "dSel_t": pd_["t"],
                "dSel_per_fold": pd_["per_fold"],
                "A1": float(r2_score(y, oof)),
                "A2": float(r2_score(y[recent], oof[recent])),
                "c2": c2_segments(df, champ_oof, oof),
            }

    print(f"\n  {'cell / arm':28s} {'maxMAE':>7s} {'dMAE':>7s} | {'cwBias':>7s} "
          f"{'dCW':>6s} | {'d25fix':>7s} {'d25own':>7s} | {'dSel':>9s} {'t':>6s} "
          f"| {'A1':>7s} {'A2':>7s}")
    print(f"  {'champion':28s} {c_mae_max:7.2f} {'—':>7s} | {c_bias_cw:+7.2f} "
          f"{'—':>6s} | {'—':>7s} {'—':>7s} | {'—':>9s} {'—':>6s} | "
          f"{c_a1:7.4f} {c_a2:7.4f}")
    for k, a in arms.items():
        print(f"  {k:28s} {a['true_max_mae']:7.2f} {a['d_true_max_mae']:+7.2f} | "
              f"{a['cw_bias']:+7.2f} {a['d_cw_bias']:+6.2f} | "
              f"{a['band25']['fixed_growth']:+7.2f} "
              f"{a['band25']['own_growth']:+7.2f} | {a['dSel']:+9.5f} "
              f"{a['dSel_t']:+6.2f} | {a['A1']:7.4f} {a['A2']:7.4f}")

    print("\n  per-fold dSel (selection pool):")
    for k, a in arms.items():
        pf = "  ".join(f"{float(v):+.5f}" for v in a["dSel_per_fold"])
        print(f"    {k:28s} [{pf}]")

    # The Win gate reads a zone MAE pooled over ALL 68 zone rows, 12 of which
    # are locked confirmation players. Split it: a decision metric must not be
    # carried by the canary split (worker brief, "Evaluation").
    print("\n  WIN GATE split by the confirmation lock "
          f"(zone n=68 = {int((true_max & sel_mask).sum())} selection + "
          f"{int((true_max & ~sel_mask).sum())} confirmation):")
    z_all, z_sel, z_conf = (true_max, true_max & sel_mask, true_max & ~sel_mask)
    ca, cs, cc = (_dollars(df, champ_oof, z_all)[0],
                  _dollars(df, champ_oof, z_sel)[0],
                  _dollars(df, champ_oof, z_conf)[0])
    print(f"    {'cell':28s} {'MAE all':>8s} {'win all':>8s} | {'MAE sel':>8s} "
          f"{'win sel':>8s} | {'MAE conf':>8s} {'win conf':>9s}")
    print(f"    {'champion':28s} {ca:8.2f} {'—':>8s} | {cs:8.2f} {'—':>8s} | "
          f"{cc:8.2f} {'—':>9s}")
    win_split = {}
    for k, a in arms.items():
        if a["arm"] != "push_clip":
            continue
        va, vs, vc = (_dollars(df, a["oof"], z_all)[0],
                      _dollars(df, a["oof"], z_sel)[0],
                      _dollars(df, a["oof"], z_conf)[0])
        win_split[k] = {"mae_all": va, "win_all": ca - va, "mae_sel": vs,
                        "win_sel": cs - vs, "mae_conf": vc, "win_conf": cc - vc}
        print(f"    {k:28s} {va:8.2f} {ca - va:+8.2f} | {vs:8.2f} "
              f"{cs - vs:+8.2f} | {vc:8.2f} {cc - vc:+9.2f}")
    report["win_gate_confirmation_split"] = win_split
    report["champion_zone_mae_split"] = {"all": ca, "sel": cs, "conf": cc}

    # ---- B1 forward ----------------------------------------------------
    print("\n" + "=" * 74)
    print("  B1 forward (rolling-origin 2024-26) — champion + every ship-form cell")
    print("=" * 74)
    fwd_cells = {name: (p_key, tau, "push_clip")
                 for name, (p_key, tau, _) in cells.items()}
    fwd = forward_pass(df, features, clf_features, labels, fwd_cells)
    b1 = {}
    for name, pred in fwd.items():
        scored = ~np.isnan(pred)
        by_origin = {}
        for T in FORWARD_ORIGINS:
            m = scored & (df["season"].values == T)
            if m.sum() >= 10:
                by_origin[int(T)] = {
                    "n": int(m.sum()), "r2": float(r2_score(y[m], pred[m])),
                    "mae": _dollars(df, pred, m)[0],
                    "maxzone_mae": _dollars(df, pred, m & true_max)[0]}
        b1[name] = {"r2": float(r2_score(y[scored], pred[scored])),
                    "n": int(scored.sum()), "by_origin": by_origin}
    for name, d in b1.items():
        drop = b1["champion"]["r2"] - d["r2"]
        tail = "" if name == "champion" else f"   drop {drop:+.4f}"
        print(f"    {name:28s} forward R2 {d['r2']:.4f}  n={d['n']}{tail}")
        for T, o in d["by_origin"].items():
            print(f"        origin {T}  R2 {o['r2']:.4f}  MAE ${o['mae']:.2f}M  "
                  f"maxzone MAE ${o['maxzone_mae']:.2f}M  n={o['n']}")
    report["B1"] = b1

    # ---- gates ---------------------------------------------------------
    print("\n" + "=" * 74)
    print("  GATE BATTERY")
    print("=" * 74)
    gates = {}
    for k, a in arms.items():
        win = -a["d_true_max_mae"]
        c2_signed = max((abs(v["signed_change"]) for v in a["c2"].values()),
                        default=0.0)
        c2_absg = max((v["abs_bias_growth"] for v in a["c2"].values()),
                      default=0.0)
        b1_drop = (b1["champion"]["r2"] - b1[a["cell"]]["r2"]
                   if a["arm"] == "push_clip" and a["cell"] in b1 else None)
        win_sel = win_split.get(k, {}).get("win_sel")
        g = {
            "win_m": win, "win_pass": win >= WIN_BAR,
            "win_sel_m": win_sel,
            "win_sel_pass": (win_sel >= WIN_BAR) if win_sel is not None else None,
            "brake_cw_m": a["d_cw_bias"], "brake_cw_pass": a["d_cw_bias"] <= BRAKE_BAR,
            "brake_25_fixed_m": a["band25"]["fixed_growth"],
            "brake_25_fixed_pass": a["band25"]["fixed_growth"] <= BRAKE_BAR,
            "brake_25_own_m": a["band25"]["own_growth"],
            "brake_25_own_pass": a["band25"]["own_growth"] <= BRAKE_BAR,
            "dSel_t": a["dSel_t"], "dSel_pass": a["dSel_t"] > DSEL_T_BAR,
            "c2_max_abs_signed_change_m": c2_signed,
            "c2_max_abs_bias_growth_m": c2_absg,
            "c2_pass_absgrowth": c2_absg <= C2_BAR,
            "c2_pass_signed": c2_signed <= C2_BAR,
            "b1_drop": b1_drop,
            "b1_pass": (b1_drop <= B1_BAR) if b1_drop is not None else None,
        }
        core = [g["win_pass"], g["brake_cw_pass"], g["dSel_pass"],
                g["c2_pass_absgrowth"]]
        if g["b1_pass"] is not None:
            core.append(g["b1_pass"])
        g["all_pass_fixed25"] = all(core + [g["brake_25_fixed_pass"]])
        g["all_pass_own25"] = all(core + [g["brake_25_own_pass"]])
        gates[k] = g

    hdr = (f"  {'cell / arm':28s} {'win$M':>7s} {'winSel':>7s} {'cw$M':>7s} "
           f"{'25fix':>7s} {'25own':>7s} {'dSel t':>7s} {'C2abs':>7s} "
           f"{'B1drop':>8s}  verdict")
    print(hdr)
    for k, g in gates.items():
        b1s = f"{g['b1_drop']:+8.4f}" if g["b1_drop"] is not None else f"{'—':>8s}"
        ws = f"{g['win_sel_m']:+7.2f}" if g["win_sel_m"] is not None else f"{'—':>7s}"
        v = ("PASS" if g["all_pass_fixed25"] else "fail") + "/" + \
            ("PASS" if g["all_pass_own25"] else "fail")
        print(f"  {k:28s} {g['win_m']:+7.2f} {ws} {g['brake_cw_m']:+7.2f} "
              f"{g['brake_25_fixed_m']:+7.2f} {g['brake_25_own_m']:+7.2f} "
              f"{g['dSel_t']:+7.2f} {g['c2_max_abs_bias_growth_m']:+7.2f} "
              f"{b1s}  {v}")
    print("  verdict column = [FIXED-row 25%+ band] / [OWN-band, phase-2 form]")
    print(f"  bars: win >= +{WIN_BAR:.2f}  cw <= +{BRAKE_BAR:.2f}  "
          f"25band <= +{BRAKE_BAR:.2f}  dSel t > {DSEL_T_BAR}  "
          f"C2 <= {C2_BAR:.2f}  B1 drop <= {B1_BAR}")
    report["gates"] = gates

    # ---- C2 detail for every ship-form cell ----------------------------
    print("\n  C2 fixed-segment detail (push_clip cells; champ -> cand):")
    for k, a in arms.items():
        if a["arm"] != "push_clip":
            continue
        print(f"    [{k}]")
        for cat, v in sorted(a["c2"].items(), key=lambda kv: -kv[1]["n"]):
            flag = "  <-- >0.30M" if v["abs_bias_growth"] > C2_BAR else ""
            print(f"      {cat:14s} n={v['n']:4d} ${v['champ']:+6.2f}M -> "
                  f"${v['cand']:+6.2f}M  (signed {v['signed_change']:+.2f}, "
                  f"|bias| growth {v['abs_bias_growth']:+.2f}){flag}")

    # ---- collateral + touched-row tables -------------------------------
    print("\n" + "=" * 74)
    print("  PART 3 — collateral and touched-row tables (ship form)")
    print("=" * 74)
    tables = {}
    for name, (p_key, tau, _) in cells.items():
        a = arms[f"{name}__push_clip"]
        oof = a["oof"]
        p = p_oof[p_key]
        push_err = (oof - y) * cap_m
        above = p >= tau

        coll = above & ~true_max
        rows_c = []
        for i in np.flatnonzero(coll)[np.argsort(-p[coll])]:
            rows_c.append({
                "player": df["player_name_norm"].iat[i],
                "season": int(df["season"].iat[i]), "p_max": float(p[i]),
                "cap_pct": float(y[i]), "salary_m": float(df["salary_m"].iat[i]),
                "signing_cat": str(df["signing_cat"].iat[i]),
                "champ_err_m": float(champ_err[i]),
                "push_err_m": float(push_err[i]),
                "damage_m": float(abs(push_err[i]) - abs(champ_err[i])),
                "route_label": rm.ROUTE_CLASSES[labels[i]]})
        touch = above & true_max
        rows_t = []
        for i in np.flatnonzero(touch)[np.argsort(-p[touch])]:
            rows_t.append({
                "player": df["player_name_norm"].iat[i],
                "season": int(df["season"].iat[i]), "p_max": float(p[i]),
                "salary_m": float(df["salary_m"].iat[i]),
                "champ_err_m": float(champ_err[i]),
                "push_err_m": float(push_err[i]),
                "gain_m": float(abs(champ_err[i]) - abs(push_err[i]))})
        tables[name] = {"collateral": rows_c, "touched": rows_t}

        print(f"\n  [{name}] tau={tau:.2f}  COLLATERAL (non-max, P>=tau): "
              f"{len(rows_c)}")
        if rows_c:
            print(f"    {'player':24s} {'seas':>4s} {'P':>5s} {'salary':>7s} "
                  f"{'champErr':>9s} {'pushErr':>8s} {'damage':>8s}  cat/label")
            for r in rows_c:
                print(f"    {r['player']:24s} {r['season']:4d} {r['p_max']:.3f} "
                      f"{r['salary_m']:7.2f} {r['champ_err_m']:+9.2f} "
                      f"{r['push_err_m']:+8.2f} {r['damage_m']:+8.2f}  "
                      f"{r['signing_cat']}/{r['route_label']}")
        print(f"  [{name}] TOUCHED true maxes: {len(rows_t)} of "
              f"{int(true_max.sum())}   realized zone gain "
              f"${-a['d_true_max_mae']:+.2f}M")
        if rows_t:
            print(f"    {'player':24s} {'seas':>4s} {'P':>5s} {'salary':>7s} "
                  f"{'champErr':>9s} {'pushErr':>8s} {'gain':>8s}")
            for r in rows_t:
                print(f"    {r['player']:24s} {r['season']:4d} {r['p_max']:.3f} "
                      f"{r['salary_m']:7.2f} {r['champ_err_m']:+9.2f} "
                      f"{r['push_err_m']:+8.2f} {r['gain_m']:+8.2f}")
    report["tables"] = tables

    # where the zone error actually lives, and whether any tau can reach it
    print("\n  Largest true-max champion errors, with each classifier's P:")
    order = np.flatnonzero(true_max)[np.argsort(-np.abs(champ_err[true_max]))][:12]
    print(f"    {'player':24s} {'seas':>4s} {'champErr':>9s} {'P_base':>7s} "
          f"{'P_enr':>7s}")
    untouched_top = []
    for i in order:
        untouched_top.append({
            "player": df["player_name_norm"].iat[i],
            "season": int(df["season"].iat[i]),
            "champ_err_m": float(champ_err[i]),
            "p_base": float(p_oof["base"][i]),
            "p_enriched": float(p_oof["enriched"][i])})
        print(f"    {df['player_name_norm'].iat[i]:24s} "
              f"{int(df['season'].iat[i]):4d} {champ_err[i]:+9.2f} "
              f"{p_oof['base'][i]:7.3f} {p_oof['enriched'][i]:7.3f}")
    report["largest_zone_errors"] = untouched_top

    # ---- save ----------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    ser = {k: {kk: vv for kk, vv in v.items() if kk != "oof"}
           for k, v in arms.items()}
    report["arms"] = ser
    with open(OUT / "route_mixture_p3_eval.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)

    dump = df[["player_name_norm", "season", TARGET, "salary_m", "signing_cat",
               "max_eligible_pct", "is_max_contract", "is_confirmation"]].copy()
    dump["route_label"] = [rm.ROUTE_CLASSES[i] for i in labels]
    dump["p_max_base"] = p_oof["base"]
    dump["p_max_enriched"] = p_oof["enriched"]
    dump["oof_champion"] = champ_oof
    for k, a in arms.items():
        dump[f"oof_{k}"] = a["oof"]
    for name, pred in fwd.items():
        dump[f"fwd_{name}"] = pred
    dump.to_csv(OUT / "route_mixture_p3_oof.csv", index=False)
    print(f"\nSaved {OUT / 'route_mixture_p3_eval.json'}")
    print(f"Saved {OUT / 'route_mixture_p3_oof.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
