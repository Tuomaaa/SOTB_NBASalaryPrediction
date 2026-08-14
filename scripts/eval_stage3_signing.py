r"""Measurement harness — Stage-3 signing-type residual correction, rung 1.

ADOPTED at v8.8x (2026-08-06). This file was the gating measurement and is now
also the REGENERATION TOOL for the deployed constants: the `deployed_offsets_k20`
block of its JSON is what `stages.SIGNING_OFFSETS_DEPLOYED` holds, and those are
data-dependent, so re-run this after every training-data rebuild and copy the
four numbers across. The shrinkage and the eligibility list are imported from
`src/model/stages.py` rather than restated, so the harness cannot measure a
different correction from the one that ships.

===========================================================================
DESIGN — fixed before any score was seen, not revisited afterwards
===========================================================================

The champion carries a systematic residual bias by signing mechanism. Rung 1 of
a signing-type Stage-3 component is the crudest possible correction: ONE
CONSTANT PER TYPE, added to the champion's own final prediction.

Eligible types      {Bird Rights, Cap Space, Early Bird, Non-Bird} — the four
                    ELIGIBILITY-based mechanisms. Every other label (MLE, BAE,
                    Minimum, Rookie Scale, Other, Unknown) gets ZERO correction
                    and must come out bit-identical. Sign & Trade and Extend &
                    Trade are NOT in this excluded set: `_categorize_signing`
                    (scripts/diagnostics.py) maps them onto Bird Rights, so
                    they are corrected like any other Bird Rights row
                    (reclassified 2026-08-07; see CLAUDE.md and
                    METHODOLOGY.md).

                    This is a leakage ruling, not a scoring choice. An exception
                    mechanism (MLE, minimum, BAE) is *determined by the contract
                    value itself* — a deal is "the MLE" because of what it pays.
                    Conditioning a prediction on that label reads the target.
                    Bird/Early-Bird/Non-Bird/Cap-Space are determined by the
                    player's prior contract and the team's books, which are
                    settled before the price is. The four stay in however large
                    the excluded types' biases look, and the excluded types stay
                    out however large theirs look.

offset(type)        n/(n+k) x mean OOF residual (actual - predicted) over rows
                    of that type, in cap_pct. k = 20, PRE-REGISTERED. Not swept,
                    not tuned on any score. A k=0 arm is computed and printed
                    REFERENCE-ONLY; the accept/reject reading is k=20 alone.

no retraining       Neither the Grabit regression nor the route classifier is
                    refit. The correction is pure post-processing of the
                    champion's own predictions, so champion and candidate share
                    every fold, every seed and every fitted model — the paired
                    delta carries zero fit noise.

legality re-applied After the offset, the row goes back through the Stage-3
                    extension raise-cap clip and the [floor_pct,
                    max_eligible_pct] clip, in that order — the same tail as
                    `stages.compose`. A correction that pushed a row past its
                    CBA ceiling would be measuring an impossible contract.

FOLD HONESTY (layer A). Scoring fold f uses offsets estimated from the OOF
residuals of rows NOT in fold f (leave-fold-out per-type means over the same
seed-averaged OOF predictions). Fold f's own residuals never correct fold f.
The DEPLOYED-FORM offsets — all rows, k=20 — are computed once for reporting
only and never used to score.

Offsets are estimated over training-side rows including confirmation rows,
because an offset is a fitted parameter and confirmation rows are already in
every training fold the champion sees (evaluate_suite.oof_groupkfold's
docstring). They are excluded from the DECIDING metrics, which are the
selection-pool readings — ISSUES #20a.

One residual channel, stated rather than hidden: the leave-fold-out offset for
fold f is a mean over OOF residuals of rows in folds g != f, and each of those
predictions came from a model that had fold f in ITS training set. Fold f's
TARGETS therefore touch the offset through the regression's parameters. This is
the ordinary single-level-CV channel that every hyperparameter chosen on OOF
already pays; a fully nested design would cost 5x the fits for a two-parameter
statistic. Layer B is clean of it entirely — its offsets never see season T in
any capacity.

MEASUREMENT PROTOCOL (CLAUDE.md, "judge a targeted intervention where it acts")

  primary   per-(fold, seed) MAE and bias over the valid-four-type rows only,
            champion vs candidate, paired. Reported pooled AND selection-only;
            the selection reading decides (ISSUES #20a). The repo convention is
            paired BY FOLD after averaging seeds (n=5, fold sd ~0.046 dwarfs
            seed sd ~0.001); the 50-cell reading is printed beside it as
            context and is NOT the decision statistic — cells within a fold are
            correlated, so its t is anti-conservative.
  guards    A1 / A2 paired R2 deltas (pooled and selection), bit-identity of
            the non-eligible rows, and a legality audit of every corrected row.
  layer B   rolling-origin B1: for target season T the offsets come only from
            seasons < T, via an inner GroupKFold OOF inside the training window.

Run:  OMP_NUM_THREADS=6 python scripts/eval_stage3_signing.py
      [--seeds N] [--skip-b]
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

from config import CAP_BY_SEASON, OUTPUTS_DIR
from src.model.evaluate_suite import (
    load_evaluation_frame, paired_delta, _dollars, N_SPLITS, DEFAULT_SEEDS,
    TARGET, FORWARD_ORIGINS,
)
from src.model import route_mixture as rm
from src.model.stages import (
    stage3_signing, signing_offsets, signing_offset_vector,
    SIGNING_ELIGIBLE_TYPES, SIGNING_K,
)

# ---- pre-registered constants ---------------------------------------------
# Imported, not restated: since v8.8x the eligibility list and the shrinkage
# live in src/model/stages.py, so the harness and the shipped composition cannot
# disagree about what was measured.
ELIGIBLE_TYPES = SIGNING_ELIGIBLE_TYPES
K_SHRINK = SIGNING_K     # PRE-REGISTERED. Never swept, never tuned on a score.
K_REFERENCE = 0.0        # reference-only arm; does not decide anything.

# Display cap for the offsets. Bias/MAE always use the row's OWN season cap
# (evaluate_suite._dollars); this constant converts a single cap_pct offset into
# dollars for a reader, matching scripts/diagnostics.py's convention.
CAP_DISPLAY = CAP_BY_SEASON.get(2026, 153_000_000)

DIAG_DIR = OUTPUTS_DIR / "diagnostics"
OUT_CSV = DIAG_DIR / "stage3_signing_offset_eval.csv"
OUT_JSON = OUTPUTS_DIR / "models" / "stage3_signing_offset_eval.json"

REPORT_ORDER = ["Bird Rights", "Cap Space", "Early Bird", "Non-Bird",
                "MLE", "Minimum", "Other", "Rookie Scale",
                "Unknown"]


# ---------------------------------------------------------------------------
# The correction itself
# ---------------------------------------------------------------------------

def type_offsets(resid: np.ndarray, cat: np.ndarray, pool: np.ndarray,
                 k: float = K_SHRINK) -> dict[str, dict]:
    """Shrunk per-type mean residual over `pool`, for the eligible types only.

    A thin wrapper on `stages.signing_offsets` — ONE implementation of the
    shrinkage, shared with the deployed composition and the evaluation suite.

    Args:
        resid: actual - predicted, in cap_pct (NOTE the sign: a positive offset
            RAISES an under-priced type, which is what a negative diagnostics
            `bias_$M` — bias there is pred - actual — asks for).
        cat: signing_cat per row.
        pool: bool mask of rows allowed to inform the estimate.
        k: shrinkage constant. offset = n/(n+k) * raw_mean.

    Returns:
        {type: {"n", "raw", "offset"}}. A type with no rows in the pool gets
        offset 0.0, which leaves those rows at the champion prediction.
    """
    return signing_offsets(resid, cat, pool=pool, k=k, detail=True)


def offset_vector(cat: np.ndarray, offs: dict[str, dict]) -> np.ndarray:
    """Per-row offset in cap_pct; exactly 0.0 outside the four eligible types.

    Kept as a named wrapper because the printed tables talk about "the offset
    vector"; `apply_correction` goes through `stages.stage3_signing` directly.
    """
    return signing_offset_vector(cat, offs)


def apply_correction(pred, cat, offs, *, lo, hi, is_ext, ext_cap) -> np.ndarray:
    """pred + offset, then the Stage-3 raise-cap clip, then the [lo, hi] clip.

    Delegates to `stages.stage3_signing` — same tail, same order as
    `stages.compose`, one implementation. A row whose offset is 0.0 returns
    bit-identical (np.minimum and np.clip are the identity on a value already
    inside the bound). That identity is ASSERTED below, not assumed.
    """
    return stage3_signing(pred, cat, offs, lo=lo, hi=hi, is_extension=is_ext,
                          ext_cap_pct=ext_cap)


# ---------------------------------------------------------------------------
# Fit pass — the champion, once. Both arms ride on these predictions.
# ---------------------------------------------------------------------------

def champion_fold_pass(df, features, clf_features, seeds, tag=""):
    """Champion OOF predictions per (fold, seed), on the suite's exact split.

    GroupKFold(N_SPLITS) over player_name_norm on the frame's row order — the
    identical protocol `evaluate_suite.oof_groupkfold` uses, so the champion
    reproduced here is the champion the suite scores.
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
            from src.model.stages import compose
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


# ---------------------------------------------------------------------------
# Per-(fold, seed) statistic matrices
# ---------------------------------------------------------------------------

def cell_r2(store, y, mask, key, n_folds, n_seeds, min_n=10):
    out = np.full((n_folds, n_seeds), np.nan)
    for rec in store:
        va = rec["va"]
        m = mask[va]
        if m.sum() > min_n:
            out[rec["fi"], rec["si"]] = r2_score(y[va][m], rec[key][m])
    return out


def cell_err(store, y, cap_m, mask, key, n_folds, n_seeds, min_n=1):
    """(MAE, signed bias, |bias|) matrices in $M over `mask` rows of each fold."""
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
    """Cell-level (fold x seed) paired delta b - a. CONTEXT ONLY.

    Cells inside a fold share a validation set, so the 50 differences are not
    independent and this t is anti-conservative. The repo's decision statistic
    is `paired_delta`, which averages seeds first and pairs by fold (n=5).
    """
    d = (b - a).ravel()
    d = d[~np.isnan(d)]
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"delta": mean, "se": se, "t": mean / se if se > 0 else float("nan"),
            "n_cells": int(len(d))}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def per_type_table(df, cat, champ, cand, mask=None):
    """Before/after bias and MAE per signing type, in cap_pct and $M."""
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


# ---------------------------------------------------------------------------
# Layer B — rolling origin, offsets from seasons < T only
# ---------------------------------------------------------------------------

def inner_oof_offsets(train_df, features, clf_features, seeds, k, tag):
    """k-shrunk per-type offsets from an OOF run INSIDE the training window.

    The only honest way to learn a season-T offset without seeing season T: the
    residuals come from a GroupKFold OOF over seasons < T alone.
    """
    store, folds = champion_fold_pass(train_df, features, clf_features, seeds,
                                      tag=tag)
    oof = seed_average(store, len(train_df), "champ", len(seeds))
    resid = train_df[TARGET].values - oof
    cat = train_df["signing_cat"].values
    return type_offsets(resid, cat, np.ones(len(train_df), bool), k=k), oof


def layer_b(df, features, clf_features, seeds, k=K_SHRINK):
    """B1: champion vs candidate under the rolling-origin protocol."""
    season = df["season"].values
    lo_all = df["floor_pct"].values
    hi_all = df["max_eligible_pct"].values
    cat = df["signing_cat"].values
    champ = np.full(len(df), np.nan)
    cand = np.full(len(df), np.nan)
    detail = {}

    for T in FORWARD_ORIGINS:
        te, tr = season == T, season < T
        if te.sum() < 10 or tr.sum() < 200:
            continue
        train, test = df[tr], df[te]
        print(f"\n    origin {T}: train n={int(tr.sum())}, test n={int(te.sum())}",
              flush=True)
        offs, _ = inner_oof_offsets(train, features, clf_features, seeds, k,
                                    tag=f"[B1 {T} inner OOF] ")
        acc = np.zeros(int(te.sum()))
        from src.model.stages import compose
        for seed in seeds:
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            clf = rm.train_route_classifier(train, clf_features, seed)
            p_max = rm.route_proba(clf, test, clf_features)[:, rm.MAX_IDX]
            acc += compose(latent, lo=lo, hi=hi, p_max=p_max,
                           is_extension=test["is_extension"].values,
                           ext_cap_pct=test["ext_cap_pct"].values)
        champ[te] = acc / len(seeds)
        cand[te] = apply_correction(
            champ[te], cat[te], offs, lo=lo_all[te], hi=hi_all[te],
            is_ext=test["is_extension"].values,
            ext_cap=test["ext_cap_pct"].values)
        detail[int(T)] = {
            "n": int(te.sum()),
            "offsets": {t: d["offset"] for t, d in offs.items()},
            "offsets_n": {t: d["n"] for t, d in offs.items()},
        }
        print(f"      offsets from seasons < {T}: " + "  ".join(
            f"{t}={offs[t]['offset']:+.5f}(n={offs[t]['n']})"
            for t in ELIGIBLE_TYPES), flush=True)
    return champ, cand, detail


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    ap.add_argument("--skip-b", action="store_true",
                    help="skip layer B (rolling origin); layer A still decides")
    args = ap.parse_args()
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])

    df, features = load_evaluation_frame(verbose=True,
                                         allow_missing_computed=True)
    df, clf_features = rm.attach_clf_features(df)

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
          f"{int((elig & sel).sum())} in the selection pool "
          f"({int((elig & ~sel).sum())} confirmation)")
    print("per type: " + "  ".join(
        f"{t} {int((cat == t).sum())}/{int(((cat == t) & sel).sum())}sel"
        for t in ELIGIBLE_TYPES))
    print(f"excluded labels: " + "  ".join(
        f"{t} {int((cat == t).sum())}"
        for t in REPORT_ORDER if t not in ELIGIBLE_TYPES))
    print(f"PRE-REGISTERED k = {K_SHRINK:g} (decision arm); "
          f"k = {K_REFERENCE:g} arm is reference-only")

    # ---- fit pass ---------------------------------------------------------
    print("\nFIT PASS — champion Grabit latent + route classifier + Stage 2/3")
    store, folds = champion_fold_pass(df, features, clf_features, seeds)
    n_folds, n_seeds = len(folds), len(seeds)
    fold_of = np.empty(len(df), dtype=int)
    for fi, (_, va) in enumerate(folds):
        fold_of[va] = fi

    champ_oof = seed_average(store, len(df), "champ", n_seeds)
    resid = y - champ_oof                      # actual - predicted, cap_pct

    print(f"\n  champion reproduction: A1 {r2_score(y, champ_oof):.4f}   "
          f"A2 {r2_score(y[recent], champ_oof[recent]):.4f}   "
          f"MAE ${np.abs((champ_oof - y) * cap_m).mean():.3f}M")
    ref_path = OUTPUTS_DIR / "models" / "evaluation_suite.json"
    if ref_path.exists():
        stored = json.load(open(ref_path))
        print(f"  stored suite champion:  A1 {stored['champion']['A1_cv_r2']:.4f}   "
              f"A2 {stored['champion']['A2_cv_r2_2024_26']:.4f}   "
              f"MAE ${stored['champion']['A1_cv_mae_m']:.3f}M   "
              f"n={stored['champion']['A1_n']}")
        if int(stored["champion"]["A1_n"]) != len(df):
            print(f"  ^ stored suite was run on a {stored['champion']['A1_n']}-row "
                  f"frame, this one has {len(df)}. R2's denominator moves with "
                  "the row set (D1 / ISSUES #35), so the two A1s are NOT "
                  "comparable; only the paired deltas below are.")

    # ---- deployed-form offsets (reporting only) ---------------------------
    all_rows = np.ones(len(df), bool)
    dep20 = type_offsets(resid, cat, all_rows, k=K_SHRINK)
    dep0 = type_offsets(resid, cat, all_rows, k=K_REFERENCE)
    dep20_sel = type_offsets(resid, cat, sel, k=K_SHRINK)

    print("\n" + "=" * 100)
    print("  DEPLOYED-FORM OFFSETS — all OOF rows, what would actually ship")
    print("=" * 100)
    print(f"  {'type':14s} {'n':>4s} {'raw mean resid':>15s} {'shrink':>7s} "
          f"{'offset k=20':>12s} {'offset $M':>10s} | {'k=0 (ref)':>10s} "
          f"{'sel-only':>10s}")
    for t in ELIGIBLE_TYPES:
        d, d0, ds = dep20[t], dep0[t], dep20_sel[t]
        shrink = d["n"] / (d["n"] + K_SHRINK)
        print(f"  {t:14s} {d['n']:4d} {d['raw']:+15.5f} {shrink:7.3f} "
              f"{d['offset']:+12.5f} {d['offset'] * CAP_DISPLAY / 1e6:+10.3f} | "
              f"{d0['offset']:+10.5f} {ds['offset']:+10.5f}")
    print(f"  $M column at the display cap ${CAP_DISPLAY / 1e6:.1f}M "
          "(2026); bias/MAE elsewhere always use the row's own season cap.")
    print("  sign convention: offset = mean(actual - predicted), so a POSITIVE "
          "offset raises an under-priced type.")

    # ---- candidate arms (fold-honest, leave-fold-out offsets) -------------
    print("\n  LEAVE-FOLD-OUT offsets used for scoring (fold f corrected from "
          "rows outside fold f):")
    lfo = {}
    for k_val, key in ((K_SHRINK, "cand20"), (K_REFERENCE, "cand0")):
        lfo[key] = {}
        for fi in range(n_folds):
            pool = fold_of != fi
            lfo[key][fi] = type_offsets(resid, cat, pool, k=k_val)
    print(f"    {'fold':>4s} " + "  ".join(f"{t:>13s}" for t in ELIGIBLE_TYPES))
    for fi in range(n_folds):
        print(f"    {fi:4d} " + "  ".join(
            f"{lfo['cand20'][fi][t]['offset']:+13.5f}" for t in ELIGIBLE_TYPES))
    spread = {t: float(np.ptp([lfo["cand20"][fi][t]["offset"]
                               for fi in range(n_folds)]))
              for t in ELIGIBLE_TYPES}
    print("    across-fold spread (max-min): " + "  ".join(
        f"{t}={spread[t]:.5f}" for t in ELIGIBLE_TYPES))

    for rec in store:
        va = rec["va"]
        for key in ("cand20", "cand0"):
            rec[key] = apply_correction(
                rec["champ"], cat[va], lfo[key][rec["fi"]],
                lo=lo_all[va], hi=hi_all[va],
                is_ext=is_ext[va], ext_cap=ext_cap[va])

    cand20 = seed_average(store, len(df), "cand20", n_seeds)
    cand0 = seed_average(store, len(df), "cand0", n_seeds)

    # ---- GUARDS ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  GUARDS")
    print("=" * 100)
    guards = {}

    # G1 bit-identity outside the four eligible types, at the CELL level
    worst_cell = 0.0
    for rec in store:
        va = rec["va"]
        m = ~elig[va]
        if m.sum():
            worst_cell = max(worst_cell,
                             float(np.abs(rec["cand20"][m] - rec["champ"][m]).max()))
    worst_oof = float(np.abs(cand20[~elig] - champ_oof[~elig]).max())
    guards["bit_identity_max_absdiff_cell"] = worst_cell
    guards["bit_identity_max_absdiff_oof"] = worst_oof
    ok = (worst_cell == 0.0) and (worst_oof == 0.0)
    print(f"  G1 bit-identity outside {{{', '.join(ELIGIBLE_TYPES)}}}: "
          f"max|diff| = {worst_cell:.3e} per (fold,seed) cell, "
          f"{worst_oof:.3e} on the seed-averaged OOF  "
          f"[{'PASS' if ok else 'FAIL'}]  n={int((~elig).sum())} rows")
    guards["bit_identity_pass"] = bool(ok)

    # G2 legality
    tol = 1e-12
    below = int((cand20 < lo_all - tol).sum())
    above = int((cand20 > hi_all + tol).sum())
    extm = np.asarray(pd.Series(is_ext).fillna(False).values, bool) & ~np.isnan(ext_cap)
    # a raise cap below the league floor cannot bind (the floor is law too);
    # verified zero such rows on this frame, same assertion stages.compose makes
    ext_below_floor = int((extm & (ext_cap < lo_all - tol)).sum())
    binds = extm & (ext_cap >= lo_all - tol)
    ext_viol = int((cand20[binds] > ext_cap[binds] + tol).sum())
    cell_viol = 0
    for rec in store:
        va = rec["va"]
        cell_viol += int((rec["cand20"] < lo_all[va] - tol).sum())
        cell_viol += int((rec["cand20"] > hi_all[va] + tol).sum())
        b = binds[va]
        cell_viol += int((rec["cand20"][b] > ext_cap[va][b] + tol).sum())
    ok2 = (below == 0 and above == 0 and ext_viol == 0 and cell_viol == 0)
    guards.update({"legality_below_floor": below, "legality_above_max": above,
                   "legality_above_ext_cap": ext_viol,
                   "legality_cell_violations": cell_viol,
                   "ext_cap_below_floor_rows": ext_below_floor,
                   "legality_pass": bool(ok2)})
    print(f"  G2 legality: below floor {below}, above tier ceiling {above}, "
          f"above extension raise cap {ext_viol} (of {int(binds.sum())} bound "
          f"rows); cell-level violations {cell_viol}  "
          f"[{'PASS' if ok2 else 'FAIL'}]")
    print(f"     rows whose raise cap sits below the league floor: "
          f"{ext_below_floor} (the [lo,hi] clip legitimately wins there)")

    # G3 how much the correction actually moves
    moved = np.abs(cand20 - champ_oof) > 1e-9
    clipped_back = elig & ~moved
    print(f"  G3 rows moved by the k=20 correction: {int(moved.sum())} of "
          f"{int(elig.sum())} eligible; {int(clipped_back.sum())} eligible rows "
          f"did NOT move (legality clipped the offset away)")
    print(f"     mean |move| on moved rows: "
          f"${np.abs((cand20 - champ_oof) * cap_m)[moved].mean():.3f}M")
    guards["n_moved"] = int(moved.sum())
    guards["n_eligible_unmoved"] = int(clipped_back.sum())

    # ---- PRIMARY GATE ------------------------------------------------------
    print("\n" + "=" * 100)
    print("  PRIMARY GATE — MAE and bias where the intervention acts "
          "(valid-four-type rows)")
    print("=" * 100)

    scopes = {
        "valid-type, SELECTION (decision-grade)": elig & sel,
        "valid-type, pooled (reporting)": elig,
        "valid-type, confirmation (canary)": elig & ~sel,
    }
    primary = {}
    for label, mask in scopes.items():
        row = {"n": int(mask.sum())}
        mae_c, bias_c, abs_c = cell_err(store, y, cap_m, mask, "champ",
                                        n_folds, n_seeds)
        for arm, key in (("k20", "cand20"), ("k0", "cand0")):
            mae_d, bias_d, abs_d = cell_err(store, y, cap_m, mask, key,
                                            n_folds, n_seeds)
            row[arm] = {
                # paired_delta(a, b) = b - a; MAE lower is better, so pass the
                # candidate first to make a POSITIVE delta a win.
                "mae_win": paired_delta(mae_d, mae_c),
                "abs_bias_reduction": paired_delta(abs_d, abs_c),
                "signed_bias_change": paired_delta(bias_c, bias_d),
                "mae_win_cells": cell_paired(mae_d, mae_c),
                "champ_mae": float(np.nanmean(mae_c)),
                "cand_mae": float(np.nanmean(mae_d)),
                "champ_bias": float(np.nanmean(bias_c)),
                "cand_bias": float(np.nanmean(bias_d)),
            }
        primary[label] = row
        print(f"\n  [{label}]  n = {row['n']}")
        for arm, tagname in (("k20", "k=20  DECISION ARM"),
                             ("k0", "k=0   reference only")):
            d = row[arm]
            print(f"    {tagname}")
            print(f"      MAE  ${d['champ_mae']:.3f}M -> ${d['cand_mae']:.3f}M   "
                  f"paired win {d['mae_win']['delta']:+.4f} "
                  f"+/- {d['mae_win']['se']:.4f}  t = {d['mae_win']['t']:+.2f}"
                  f"   (by fold, n={n_folds})")
            print(f"           cell-level (context, n="
                  f"{d['mae_win_cells']['n_cells']}): "
                  f"{d['mae_win_cells']['delta']:+.4f} "
                  f"t = {d['mae_win_cells']['t']:+.2f}")
            print(f"      bias ${d['champ_bias']:+.3f}M -> "
                  f"${d['cand_bias']:+.3f}M   |bias| reduction "
                  f"{d['abs_bias_reduction']['delta']:+.4f} "
                  f"+/- {d['abs_bias_reduction']['se']:.4f}  "
                  f"t = {d['abs_bias_reduction']['t']:+.2f}")
            print("      per fold (MAE win): [" + ", ".join(
                f"{float(v):+.4f}" for v in d["mae_win"]["per_fold"]) + "]")

    # ---- A1 / A2 guards ----------------------------------------------------
    print("\n" + "=" * 100)
    print("  GUARD — A1 / A2 paired R2 deltas (must not meaningfully degrade)")
    print("=" * 100)
    r2_masks = {
        "A1 selection (decision-grade)": sel,
        "A1 pooled": np.ones(len(df), bool),
        "A2 selection, 2024-26": sel & recent,
        "A2 pooled, 2024-26": recent,
    }
    a_guards = {}
    print(f"  {'metric':32s} {'champ':>8s} {'k=20':>8s} {'delta':>9s} "
          f"{'se':>8s} {'t':>7s} | {'k=0 delta':>10s} {'t':>7s}")
    for label, mask in r2_masks.items():
        m_c = cell_r2(store, y, mask, "champ", n_folds, n_seeds)
        out = {}
        cells = []
        for arm, key in (("k20", "cand20"), ("k0", "cand0")):
            m_d = cell_r2(store, y, mask, key, n_folds, n_seeds)
            out[arm] = paired_delta(m_c, m_d)     # cand - champ, higher better
            out[arm + "_mean"] = float(np.nanmean(m_d))
            cells.append(out[arm])
        out["champ_mean"] = float(np.nanmean(m_c))
        # pooled OOF R2 for the record
        out["oof_r2_champ"] = float(r2_score(y[mask], champ_oof[mask]))
        out["oof_r2_k20"] = float(r2_score(y[mask], cand20[mask]))
        out["oof_r2_k0"] = float(r2_score(y[mask], cand0[mask]))
        a_guards[label] = out
        print(f"  {label:32s} {out['champ_mean']:8.4f} {out['k20_mean']:8.4f} "
              f"{out['k20']['delta']:+9.5f} {out['k20']['se']:8.5f} "
              f"{out['k20']['t']:+7.2f} | {out['k0']['delta']:+10.5f} "
              f"{out['k0']['t']:+7.2f}")
    print("\n  same quantities on the SEED-AVERAGED OOF arrays (one number, "
          "no pairing):")
    for label, out in a_guards.items():
        print(f"    {label:32s} champ {out['oof_r2_champ']:.4f}  "
              f"k=20 {out['oof_r2_k20']:.4f} "
              f"({out['oof_r2_k20'] - out['oof_r2_champ']:+.5f})  "
              f"k=0 {out['oof_r2_k0']:.4f} "
              f"({out['oof_r2_k0'] - out['oof_r2_champ']:+.5f})")
    mae_all_c = np.abs((champ_oof - y) * cap_m).mean()
    mae_all_d = np.abs((cand20 - y) * cap_m).mean()
    print(f"    pooled MAE over ALL {len(df)} rows: ${mae_all_c:.3f}M -> "
          f"${mae_all_d:.3f}M ({mae_all_c - mae_all_d:+.3f} win)")

    # ---- per-type before/after ---------------------------------------------
    print("\n" + "=" * 100)
    print("  PER-TYPE BIAS BEFORE / AFTER  (k=20, leave-fold-out offsets)")
    print("=" * 100)
    tab_sel = per_type_table(df, cat, champ_oof, cand20, sel)
    tab_all = per_type_table(df, cat, champ_oof, cand20, None)
    print_per_type(tab_sel, "SELECTION pool (decision-grade) — ISSUES #20a")
    print_per_type(tab_all, "pooled (reporting)")

    # ---- LAYER B -----------------------------------------------------------
    b_report = {}
    if args.skip_b:
        print("\n  LAYER B skipped (--skip-b).")
    else:
        print("\n" + "=" * 100)
        print("  LAYER B — rolling origin, offsets learned ONLY from seasons < T")
        print("=" * 100)
        b_champ, b_cand, b_detail = layer_b(df, features, clf_features, seeds)
        scored = ~np.isnan(b_champ)
        b_report["by_origin"] = b_detail
        print(f"\n  {'scope':34s} {'n':>5s} {'champ R2':>9s} {'k20 R2':>9s} "
              f"{'dR2':>9s} | {'champ MAE':>10s} {'k20 MAE':>9s} {'win':>8s}")
        for label, mask in (("all forward rows", scored),
                            ("all, selection", scored & sel),
                            ("valid-type", scored & elig),
                            ("valid-type, selection", scored & elig & sel)):
            if mask.sum() < 10:
                continue
            r2c = float(r2_score(y[mask], b_champ[mask]))
            r2d = float(r2_score(y[mask], b_cand[mask]))
            mc = float(np.abs((b_champ[mask] - y[mask]) * cap_m[mask]).mean())
            md = float(np.abs((b_cand[mask] - y[mask]) * cap_m[mask]).mean())
            bc = float(((b_champ[mask] - y[mask]) * cap_m[mask]).mean())
            bd = float(((b_cand[mask] - y[mask]) * cap_m[mask]).mean())
            b_report[label] = {"n": int(mask.sum()), "champ_r2": r2c,
                               "cand_r2": r2d, "delta_r2": r2d - r2c,
                               "champ_mae_m": mc, "cand_mae_m": md,
                               "mae_win_m": mc - md,
                               "champ_bias_m": bc, "cand_bias_m": bd}
            print(f"  {label:34s} {int(mask.sum()):5d} {r2c:9.4f} {r2d:9.4f} "
                  f"{r2d - r2c:+9.5f} | {mc:10.3f} {md:9.3f} {mc - md:+8.3f}")
        print(f"\n  {'origin':>8s} {'n':>5s} {'champ R2':>9s} {'k20 R2':>9s} "
              f"{'dR2':>9s} | {'champ MAE':>10s} {'k20 MAE':>9s} {'win':>8s}")
        for T in FORWARD_ORIGINS:
            m = scored & (df["season"].values == T)
            if m.sum() < 10:
                continue
            r2c = float(r2_score(y[m], b_champ[m]))
            r2d = float(r2_score(y[m], b_cand[m]))
            mc = float(np.abs((b_champ[m] - y[m]) * cap_m[m]).mean())
            md = float(np.abs((b_cand[m] - y[m]) * cap_m[m]).mean())
            b_report.setdefault("origins", {})[int(T)] = {
                "n": int(m.sum()), "champ_r2": r2c, "cand_r2": r2d,
                "champ_mae_m": mc, "cand_mae_m": md}
            print(f"  {int(T):8d} {int(m.sum()):5d} {r2c:9.4f} {r2d:9.4f} "
                  f"{r2d - r2c:+9.5f} | {mc:10.3f} {md:9.3f} {mc - md:+8.3f}")
        nb = scored & ~elig
        diffs = np.abs(b_cand[nb] - b_champ[nb])
        b_id = float(diffs.max())
        # ULP_TOL, not 0.0, and only on the B1 path. Layer A corrects each
        # (fold, seed) prediction BEFORE the seed average, so its non-eligible
        # rows are exactly 0.0 and are gated at exactly 0.0 above. B1 corrects
        # the seed-AVERAGED array, and a mean of ten individually-clipped values
        # can land 1-2 ulp ABOVE the bound they were each clipped to; the
        # candidate's legality re-clip then snaps it back. Diagnosed to the row:
        # josh hart 2024, an extension whose ext_cap_pct is 0.1290579565823541
        # while the seed mean is 0.12905795658235414 — 2 ulp, $0.0000000046 at a
        # $165M cap. Reproduced with a ZERO offset vector, which is what proves
        # the correction is not the cause.
        ULP_TOL = 1e-12
        n_real = int((diffs > ULP_TOL).sum())
        ok_b = n_real == 0
        print(f"\n  B1 bit-identity outside the four types: max|diff| "
              f"{b_id:.3e} over {int(nb.sum())} rows; rows above {ULP_TOL:.0e} "
              f"(a real move): {n_real}  [{'PASS' if ok_b else 'FAIL'}]")
        if 0.0 < b_id <= ULP_TOL:
            print("     the residue is float rounding of the seed average "
                  "against a legality bound, not the offset — see the comment "
                  "in this script; layer A's cell-level check is exactly 0.0.")
        b_report["bit_identity_max_absdiff"] = b_id
        b_report["bit_identity_rows_above_tol"] = n_real
        b_report["bit_identity_pass"] = bool(ok_b)

    # ---- CSV ---------------------------------------------------------------
    rows = []
    for arm_name, offs in (("offset_k20_deployed", dep20),
                           ("offset_k0_reference", dep0),
                           ("offset_k20_selection_only", dep20_sel)):
        for t in ELIGIBLE_TYPES:
            d = offs[t]
            rows.append({"section": arm_name, "scope": "all_oof_rows", "key": t,
                         "n": d["n"], "raw_mean_resid_pct": d["raw"],
                         "offset_cap_pct": d["offset"],
                         "offset_m": d["offset"] * CAP_DISPLAY / 1e6})
    for fi in range(n_folds):
        for t in ELIGIBLE_TYPES:
            d = lfo["cand20"][fi][t]
            rows.append({"section": "offset_k20_leave_fold_out",
                         "scope": f"fold_{fi}", "key": t, "n": d["n"],
                         "raw_mean_resid_pct": d["raw"],
                         "offset_cap_pct": d["offset"],
                         "offset_m": d["offset"] * CAP_DISPLAY / 1e6})
    for scope, tab in (("selection", tab_sel), ("pooled", tab_all)):
        for _, r in tab.iterrows():
            rows.append({"section": "per_type_before_after", "scope": scope,
                         "key": r["signing_type"], "n": int(r["n"]),
                         "eligible": bool(r["eligible"]),
                         "champ_bias_pct": r["champ_bias_pct"],
                         "cand_bias_pct": r["cand_bias_pct"],
                         "champ_bias_m": r["champ_bias_m"],
                         "cand_bias_m": r["cand_bias_m"],
                         "champ_mae_m": r["champ_mae_m"],
                         "cand_mae_m": r["cand_mae_m"],
                         "mae_win_m": r["mae_win_m"],
                         "abs_bias_reduction_m": r["abs_bias_reduction_m"]})
    for label, row in primary.items():
        for arm in ("k20", "k0"):
            d = row[arm]
            for metric in ("mae_win", "abs_bias_reduction", "signed_bias_change"):
                rows.append({"section": "paired_primary", "scope": label,
                             "key": f"{metric}|{arm}", "n": row["n"],
                             "champ_mae_m": d["champ_mae"],
                             "cand_mae_m": d["cand_mae"],
                             "champ_bias_m": d["champ_bias"],
                             "cand_bias_m": d["cand_bias"],
                             "delta": d[metric]["delta"], "se": d[metric]["se"],
                             "t": d[metric]["t"]})
    for label, out in a_guards.items():
        for arm in ("k20", "k0"):
            rows.append({"section": "paired_guard_r2", "scope": label,
                         "key": arm, "n": int(len(df)),
                         "delta": out[arm]["delta"], "se": out[arm]["se"],
                         "t": out[arm]["t"],
                         "champ_r2": out["oof_r2_champ"],
                         "cand_r2": out[f"oof_r2_{arm}"]})
    for k, v in guards.items():
        rows.append({"section": "guard_check", "scope": "layer_a", "key": k,
                     "value": v})
    if b_report:
        for label in ("all forward rows", "all, selection", "valid-type",
                      "valid-type, selection"):
            if label in b_report:
                d = b_report[label]
                rows.append({"section": "layer_b", "scope": label,
                             "key": "B1", "n": d["n"], "champ_r2": d["champ_r2"],
                             "cand_r2": d["cand_r2"], "delta": d["delta_r2"],
                             "champ_mae_m": d["champ_mae_m"],
                             "cand_mae_m": d["cand_mae_m"],
                             "mae_win_m": d["mae_win_m"],
                             "champ_bias_m": d["champ_bias_m"],
                             "cand_bias_m": d["cand_bias_m"]})
        for T, d in b_report.get("origins", {}).items():
            rows.append({"section": "layer_b", "scope": f"origin_{T}",
                         "key": "B1", "n": d["n"], "champ_r2": d["champ_r2"],
                         "cand_r2": d["cand_r2"],
                         "delta": d["cand_r2"] - d["champ_r2"],
                         "champ_mae_m": d["champ_mae_m"],
                         "cand_mae_m": d["cand_mae_m"],
                         "mae_win_m": d["champ_mae_m"] - d["cand_mae_m"]})
        for T, d in b_report.get("by_origin", {}).items():
            for t, v in d["offsets"].items():
                rows.append({"section": "layer_b_offsets",
                             "scope": f"origin_{T}", "key": t,
                             "n": d["offsets_n"][t], "offset_cap_pct": v,
                             "offset_m": v * CAP_DISPLAY / 1e6})

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} ({len(out)} rows)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_rows": int(len(df)), "seeds": list(seeds), "n_splits": N_SPLITS,
        "k_shrink": K_SHRINK, "eligible_types": list(ELIGIBLE_TYPES),
        "deployed_offsets_k20": {t: dep20[t] for t in ELIGIBLE_TYPES},
        "deployed_offsets_k0": {t: dep0[t] for t in ELIGIBLE_TYPES},
        "leave_fold_out_k20": {str(fi): {t: lfo["cand20"][fi][t]
                                         for t in ELIGIBLE_TYPES}
                               for fi in range(n_folds)},
        "primary": primary, "r2_guards": a_guards, "guards": guards,
        "layer_b": b_report,
        "champion_A1": float(r2_score(y, champ_oof)),
        "champion_A2": float(r2_score(y[recent], champ_oof[recent])),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"Saved {OUT_JSON}")

    # Write the deployed offsets to the tracked file that stages.py loads.
    from src.model.stages import _SIGNING_OFFSETS_FILE
    deployed = {t: dep20[t]["offset"] if isinstance(dep20[t], dict)
                else float(dep20[t])
                for t in ELIGIBLE_TYPES}
    _SIGNING_OFFSETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_SIGNING_OFFSETS_FILE, "w") as fh:
        json.dump(deployed, fh, indent=2)
    print(f"Saved {_SIGNING_OFFSETS_FILE} (stages.py will load these on next import)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
