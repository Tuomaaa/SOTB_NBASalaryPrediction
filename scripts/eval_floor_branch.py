"""Evidence harness for the floor branch (2026-07-27-floor-branch brief).

Phase 2 declared the floor branch NO-GO on an argument — "the left-censor
already pins the rows a floor classifier is confident about" — never on a
measurement. The max side's version of that argument was wrong. This harness
measures the floor side the way the max side was finally measured: a
pre-registered tau rule computed in probability space, then arms scored at that
tau alone.

The headroom is the largest single number in the project. Pulling every true
at-floor row onto its floor takes A1 0.7865 -> 0.8261 (+0.0396), MAE $3.076M ->
$2.575M. It exists because the Stage-2 clip is a ONE-WAY valve: it lifts a
prediction that falls below floor_pct but never lowers one that sits above.

===========================================================================
PRE-REGISTRATION — fixed before any arm was scored, and not revisited after
===========================================================================

tau grid          0.10 to 0.94 step 0.02.
                  The max sweep started at 0.30 because P(max) came from a
                  0.98-AUC classifier. P(floor) comes from a ~0.83-AUC one and
                  is not expected to reach the same range, so the grid is
                  extended DOWNWARD. The grid does not select; the objective
                  does. The argmax restricted to tau >= 0.30 (the max side's
                  grid) is reported alongside as a comparability check.

expected win      sum over touched TRUE-FLOOR rows of the champion's current
                  OVER-prediction, max(champ - actual, 0) in $M. Two
                  sensitivities are printed and must agree on tau* or the
                  disagreement is reported as an anomaly:
                    W|.|   sum |champ - actual|      (the max side's form)
                    Wreal  sum |champ-y| - |floor-y| (what arm A can realize)

expected coll.    sum over touched NON-FLOOR rows of P x (champ - floor_pct)
                  in $M. The mirror of the max side's P x (ceiling - champ).
                  This is the ONLY guard on this side: no CBA rule raises a
                  player's floor by route, so unlike the max branch there is no
                  legal brake behind the threshold.

tau*              argmax (expected win - expected collateral). Never a realized
                  zone metric — three experiments have died on that.

MARGIN            0.95, fixed, never tuned (the 1.05 precedent on the max side).

Arms, at tau* only:
  A  pull      pred = min(champ, floor_pct) where P >= tau. Because the
               champion is already clipped at floor_pct, this is exactly a snap
               to floor_pct.
  B  pull+marg pred = champ + P*(MARGIN*floor_pct - champ) where P >= tau, then
               the Stage-2 clip. The mirror of push-then-clip.
  T  told      pull the rows that ARE at the floor. Stage-3 TOLD mode: it reads
               is_at_floor, an outcome label. Reported in the same column as the
               ex-ante arms per the 2026-07-26 convention decision, and labelled
               TOLD everywhere so the two are never confused. It is also the
               oracle of arms A/B, so it doubles as the headroom reproduction.

Gates (brief's bars): floor-zone MAE improves >= $0.30M; no non-floor segment's
|bias| grows by more than $0.30M; C2 fixed segments <= $0.30M; B1 forward drop
<= 0.003; paired dSel t > 2 for the EX-ANTE arms (A, B) only.

Three protocol defects from ISSUES #20 are honoured here:
  (a) every zone MAE that DECIDES is computed over ~is_confirmation rows; the
      pooled reading is printed beside it, labelled. 37 of the 240 floor rows
      are confirmation rows, so this is not academic.
  (b) every segment comparison uses FIXED row groups (the champion's bands).
  (c) C2 is reported as |bias| growth (the documented intent) AND as the signed
      change (the older implementation); the gate reads |bias| growth.

Run:  OMP_NUM_THREADS=6 python scripts/eval_floor_branch.py [--seeds N]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import OUTPUTS_DIR
from src.model.train import FEATURE_COLS
from src.model.evaluate_suite import (
    load_evaluation_frame, paired_delta, _dollars, N_SPLITS, DEFAULT_SEEDS,
    TARGET, FORWARD_ORIGINS, abs_bias_growth, zone_scorecard,
)
from src.model import route_mixture as rm
from src.model.extension_cap import attach_extension_cap

MARGIN = 0.95                       # fixed constant, never tuned
TAU_GRID = np.round(np.arange(0.10, 0.95, 0.02), 2)
TAU_MAXSIDE_MIN = 0.30              # the max sweep's grid floor, for comparison
# Diagnostic taus BELOW the pre-registered grid. They exist only so the shape of
# the objective near zero is visible when the argmax lands on the grid's lower
# edge — they are excluded from selection (in_selection_range=False) and no arm
# is gated at them.
TAU_DIAGNOSTIC = np.round(np.arange(0.00, 0.10, 0.02), 2)

ZONE_WIN_BAR = 0.30                 # $M floor-zone MAE improvement
BRAKE_BAR = 0.30                    # $M |bias| growth, each non-floor brake
C2_BAR = 0.30                       # $M per fixed signing-mechanism segment
B1_BAR = 0.003                      # forward R2 drop
DSEL_T_BAR = 2.0                    # paired t, ex-ante arms only

NEAR_FLOOR_BAND = 0.04              # champion-defined collateral pool (fixed rows)

FALLEN_STARS = [("victor oladipo", 2021), ("kelly oubre jr.", 2023),
                ("montrezl harrell", 2022), ("andre drummond", 2021),
                ("hassan whiteside", 2020), ("reggie jackson", 2020),
                ("chris paul", 2025), ("marc gasol", 2020),
                ("blake griffin", 2021)]

OUT = OUTPUTS_DIR / "models"


# ---------------------------------------------------------------------------
# One fit pass: champion latent + three classifiers, per (seed, fold)
# ---------------------------------------------------------------------------

def fit_pass(df, features, clf_features, lab4, lab6, seeds):
    """Champion Grabit latent + P(floor) from three classifiers.

    base4      4-class on FEATURE_COLS         — comparability with the phase-3
                                                 baseline AUC 0.8259
    enr4       4-class on the enriched list    — the enrichment's effect
    enr6       6-class on the enriched list    — PRIMARY, the brief's classifier
                                                 and the one the max sweep used
    """
    y = df[TARGET].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(
        df, y, df["player_name_norm"].values))
    sel = ~df["is_confirmation"].values
    base_feats = list(FEATURE_COLS)

    store = []
    acc_champ = np.zeros(len(df))
    acc_p = {k: np.zeros(len(df)) for k in ("base4", "enr4", "enr6")}
    frs_champ = np.zeros((len(folds), len(seeds)))

    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(folds):
            train, test = df.iloc[tr], df.iloc[va]
            latent, lo, hi = rm.grabit_latent(train, test, features, seed,
                                                max_term_on=False)
            champ = np.clip(latent, lo, hi)

            c4b = rm.train_route_classifier(train, base_feats, seed, labels=lab4[tr])
            p4b = rm.route_proba(c4b, test, base_feats)[:, rm.FLOOR_IDX]
            c4e = rm.train_route_classifier(train, clf_features, seed, labels=lab4[tr])
            p4e = rm.route_proba(c4e, test, clf_features)[:, rm.FLOOR_IDX]
            c6e = rm.train_route6_classifier(train, clf_features, seed, labels=lab6[tr])
            p6e = rm.route_proba(c6e, test, clf_features)[:, rm.R6["floor"]]

            store.append({"si": si, "fi": fi, "va": va, "latent": latent,
                          "lo": lo, "hi": hi, "champ": champ, "p": p6e})
            acc_champ[va] += champ
            acc_p["base4"][va] += p4b
            acc_p["enr4"][va] += p4e
            acc_p["enr6"][va] += p6e
            vs = sel[va]
            frs_champ[fi, si] = (r2_score(y[va][vs], champ[vs])
                                 if vs.sum() > 10 else np.nan)
        print(f"    seed {seed} done ({si + 1}/{len(seeds)})", flush=True)

    n = len(seeds)
    return (store, acc_champ / n, {k: v / n for k, v in acc_p.items()},
            frs_champ, folds)


def arm(df, store, kind, tau, n_folds, n_seeds):
    """OOF predictions + selection fold-R2 for one arm.

    kind: "A" (pull), "B" (pull with margin), "T" (told), "champ".
    """
    y, sel = df[TARGET].values, ~df["is_confirmation"].values
    at_floor = df["is_at_floor"].values.astype(bool)
    acc = np.zeros(len(df))
    frs = np.zeros((n_folds, n_seeds))
    for rec in store:
        va, lo, hi = rec["va"], rec["lo"], rec["hi"]
        champ, p = rec["champ"], rec["p"]
        if kind == "champ":
            pred = champ
        elif kind == "A":
            pred = np.where(p >= tau, np.minimum(champ, lo), champ)
        elif kind == "B":
            pulled = champ + p * (MARGIN * lo - champ)
            pred = np.clip(np.where(p >= tau, pulled, champ), lo, hi)
        elif kind == "T":
            pred = np.where(at_floor[va], lo, champ)
        else:
            raise ValueError(f"unknown arm {kind!r}")
        acc[va] += pred
        vs = sel[va]
        frs[rec["fi"], rec["si"]] = (r2_score(y[va][vs], pred[vs])
                                     if vs.sum() > 10 else np.nan)
    return acc / n_seeds, frs


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def class_diagnostics(p, true, tag):
    """AUC + calibration for one P(floor) vector."""
    auc = float(roc_auc_score(true, p))
    med = float(np.median(p[true == 1]))
    edges = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    cal = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 3:
            cal.append([f"[{a:.1f},{b:.1f})", int(m.sum()),
                        round(float(p[m].mean()), 3), round(float(true[m].mean()), 3)])
    print(f"    {tag:34s} AUC {auc:.4f}  median P on true floor {med:.4f}  "
          f"max P {p.max():.3f}  n(P>=0.5) {int((p >= 0.5).sum())}")
    return {"auc": auc, "median_p_on_true": med, "max_p": float(p.max()),
            "calibration": cal}


def c2_segments(df, champ, cand):
    segs = {}
    for cat, sub in df.groupby("signing_cat"):
        if len(sub) < 10:
            continue
        m = (df["signing_cat"] == cat).values
        _, b_c = _dollars(df, champ, m)
        _, b_d = _dollars(df, cand, m)
        segs[str(cat)] = {"n": int(m.sum()), "champ": b_c, "cand": b_d,
                          "signed_change": b_d - b_c,
                          "abs_bias_growth": abs_bias_growth(b_d, b_c)}
    return segs


def brakes(df, champ, cand, at_floor):
    """Fixed-row non-floor brakes: all non-floor rows, and the low-prediction pool."""
    out = {}
    for tag, m in (("nonfloor_all", ~at_floor),
                   ("nonfloor_lowpred", (~at_floor) & (champ <= NEAR_FLOOR_BAND))):
        mae_c, b_c = _dollars(df, champ, m)
        mae_d, b_d = _dollars(df, cand, m)
        out[tag] = {"n": int(m.sum()), "champ_bias": b_c, "cand_bias": b_d,
                    "signed_change": b_d - b_c,
                    "abs_bias_growth": abs_bias_growth(b_d, b_c),
                    "mae_change": mae_d - mae_c}
    return out


def forward_pass(df, features, clf_features, lab6, tau, seeds):
    """Rolling-origin forward for the champion and all three arms at tau."""
    season = df["season"].values
    at_floor = df["is_at_floor"].values.astype(bool)
    keys = ["champion", "A", "B", "T"]
    preds = {k: np.full(len(df), np.nan) for k in keys}
    for T in FORWARD_ORIGINS:
        te, tr = season == T, season < T
        if te.sum() < 10 or tr.sum() < 200:
            continue
        train, test = df[tr], df[te]
        acc = {k: np.zeros(int(te.sum())) for k in keys}
        af_te = at_floor[te]
        for seed in seeds:
            latent, lo, hi = rm.grabit_latent(train, test, features, seed,
                                                max_term_on=False)
            champ = np.clip(latent, lo, hi)
            c6 = rm.train_route6_classifier(train, clf_features, seed,
                                            labels=lab6[tr])
            p = rm.route_proba(c6, test, clf_features)[:, rm.R6["floor"]]
            acc["champion"] += champ
            acc["A"] += np.where(p >= tau, np.minimum(champ, lo), champ)
            pulled = champ + p * (MARGIN * lo - champ)
            acc["B"] += np.clip(np.where(p >= tau, pulled, champ), lo, hi)
            acc["T"] += np.where(af_te, lo, champ)
        for k in keys:
            preds[k][te] = acc[k] / len(seeds)
        print(f"    origin {T} done (train n={int(tr.sum())}, "
              f"test n={int(te.sum())})", flush=True)
    return preds


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    args = ap.parse_args()
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])

    pd.set_option("display.width", 250)
    df, features = load_evaluation_frame()
    df = attach_extension_cap(df, verbose=False)
    df, clf_features = rm.attach_clf_features(df)

    y = df[TARGET].values
    cap_m = df["cap"].values / 1e6
    fp = df["floor_pct"].values
    at_floor = df["is_at_floor"].values.astype(bool)
    sel = ~df["is_confirmation"].values
    recent = df["season"].values >= 2024
    lab4, lab6 = rm.compute_route_labels(df), rm.compute_route6_labels(df)

    report = {"n_rows": len(df), "n_floor_zone": int(at_floor.sum()),
              "n_floor_confirmation": int((at_floor & ~sel).sum()),
              "margin": MARGIN, "seeds": list(seeds),
              "bars": {"zone_win_m": ZONE_WIN_BAR, "brake_m": BRAKE_BAR,
                       "c2_m": C2_BAR, "b1_drop": B1_BAR, "dsel_t": DSEL_T_BAR}}

    print(f"\nframe {len(df)} | floor zone {int(at_floor.sum())} "
          f"({int((at_floor & ~sel).sum())} of them confirmation rows) | "
          f"seeds {len(seeds)}")
    print(f"route6 labels: " + "  ".join(
        f"{c}={int((lab6 == i).sum())}" for c, i in rm.R6.items()))

    print("\nFIT PASS — champion latent + base4 / enr4 / enr6 classifiers")
    store, champ_oof, p_all, frs_champ, folds = fit_pass(
        df, features, clf_features, lab4, lab6, seeds)
    n_folds, n_seeds = len(folds), len(seeds)
    p = p_all["enr6"]                       # PRIMARY

    # champion reference
    champ_err = (champ_oof - y) * cap_m
    c_a1 = float(r2_score(y, champ_oof))
    c_a2 = float(r2_score(y[recent], champ_oof[recent]))
    c_zone = zone_scorecard(df, champ_oof, champ_oof, at_floor, sel)
    print(f"\n  champion  A1 {c_a1:.4f}  A2 {c_a2:.4f}  pooled MAE "
          f"${np.abs(champ_err).mean():.3f}M")
    for tag, d in c_zone.items():
        print(f"    floor zone [{tag:4s}] n={d['n']:4d}  MAE ${d['champ_mae']:.2f}M  "
              f"bias ${d['champ_bias']:+.2f}M")
    report["champion"] = {"A1": c_a1, "A2": c_a2, "zone": c_zone,
                          "pooled_mae_m": float(np.abs(champ_err).mean())}

    ref_path = OUT / "evaluation_suite.json"
    if ref_path.exists():
        stored = json.load(open(ref_path))
        ref = np.array(stored["fold_r2_selection"]["champion"])
        if ref.shape == frs_champ.shape:
            d = float(np.abs(frs_champ - ref).max())
            print(f"  champion fold_r2_selection reproduction max|diff| "
                  f"vs stored suite: {d:.2e}")
            report["champion_repro_max_absdiff"] = d
        print(f"  stored suite: A1 {stored['champion']['A1_cv_r2']:.4f}  "
              f"floor zone n={stored['floor_zone']['n']} "
              f"MAE ${stored['floor_zone']['mae_grabit']:.2f}M "
              f"bias ${stored['floor_zone']['bias_grabit']:+.2f}M")

    # ---- PART 1a: classifier diagnostics --------------------------------
    print("\n" + "=" * 100)
    print("  PART 1a — P(floor) classifier diagnostics")
    print("=" * 100)
    clf_diag = {}
    for key, tag in (("base4", "4-class, FEATURE_COLS (phase-3 ref)"),
                     ("enr4", "4-class, enriched"),
                     ("enr6", "6-class, enriched  [PRIMARY]")):
        clf_diag[key] = class_diagnostics(p_all[key], at_floor.astype(int), tag)
    print("\n  calibration of the PRIMARY P(floor) — binned by P, empirical rate")
    print(f"    {'bin':12s} {'n':>5s} {'meanP':>7s} {'emp':>7s}")
    for row in clf_diag["enr6"]["calibration"]:
        print(f"    {row[0]:12s} {row[1]:5d} {row[2]:7.3f} {row[3]:7.3f}")
    report["classifiers"] = clf_diag

    # ---- PART 1b: purity + the pre-registered objective ------------------
    print("\n" + "=" * 100)
    print("  PART 1b — purity curve and the honest headroom "
          "(computed BEFORE any arm runs)")
    print("=" * 100)
    zone_total = float(np.maximum(champ_err[at_floor], 0).sum())
    print(f"  the whole floor zone carries ${zone_total:.1f}M of champion "
          f"over-prediction; 'capture' is the share of it a tau reaches")
    print(f"  {'tau':>5s} {'n>=t':>5s} {'nFlr':>5s} {'nColl':>6s} {'purity':>7s} "
          f"{'capt':>6s} | {'Ewin':>7s} {'Ecoll':>7s} {'OBJ':>8s} | {'W|.|':>7s} "
          f"{'OBJ|.|':>8s} | {'Wreal':>7s} {'OBJreal':>8s}")
    realizable = np.abs(champ_err) - np.abs((fp - y) * cap_m)
    sweep = []
    for tau in np.concatenate([TAU_DIAGNOSTIC, TAU_GRID]):
        in_range = tau >= TAU_GRID[0] - 1e-9
        touched = p >= tau
        tf, nf = touched & at_floor, touched & ~at_floor
        e_win = float(np.maximum(champ_err[tf], 0).sum())
        e_win_abs = float(np.abs(champ_err[tf]).sum())
        e_win_real = float(realizable[tf].sum())
        e_coll = float((p[nf] * (champ_oof[nf] - fp[nf]) * cap_m[nf]).sum())
        row = {"tau": float(tau), "in_selection_range": bool(in_range),
               "n_above": int(touched.sum()),
               "n_true_floor": int(tf.sum()), "n_collateral": int(nf.sum()),
               "purity": float(tf.sum() / touched.sum()) if touched.sum() else float("nan"),
               "capture": e_win / zone_total if zone_total else float("nan"),
               "expected_win_m": e_win, "expected_win_abs_m": e_win_abs,
               "expected_win_realizable_m": e_win_real,
               "expected_collateral_m": e_coll,
               "objective_m": e_win - e_coll,
               "objective_abs_m": e_win_abs - e_coll,
               "objective_realizable_m": e_win_real - e_coll}
        sweep.append(row)
        print(f"  {tau:5.2f}{'' if in_range else '*'}{row['n_above']:4d} "
              f"{row['n_true_floor']:5d} {row['n_collateral']:6d} "
              f"{row['purity']:7.3f} {row['capture']:6.3f} | {e_win:7.1f} "
              f"{e_coll:7.1f} {row['objective_m']:8.1f} | {e_win_abs:7.1f} "
              f"{row['objective_abs_m']:8.1f} | {e_win_real:7.1f} "
              f"{row['objective_realizable_m']:8.1f}")
    print("  (* = below the pre-registered grid: diagnostic only, never selected)")
    report["sweep"] = sweep
    report["zone_total_overprediction_m"] = zone_total

    live = [r for r in sweep if r["n_above"] > 0 and r["in_selection_range"]]
    best = max(live, key=lambda r: r["objective_m"])
    tau_star = best["tau"]
    alt = {k: max(live, key=lambda r: r[k])["tau"]
           for k in ("objective_abs_m", "objective_realizable_m")}
    maxside = [r for r in live if r["tau"] >= TAU_MAXSIDE_MIN]
    tau_maxgrid = max(maxside, key=lambda r: r["objective_m"])["tau"] if maxside else None
    print(f"\n  SELECTED tau* = {tau_star:.2f}   (expected win "
          f"${best['expected_win_m']:.1f}M - expected collateral "
          f"${best['expected_collateral_m']:.1f}M = ${best['objective_m']:.1f}M)")
    print(f"    at tau*: {best['n_above']} rows touched, {best['n_true_floor']} true "
          f"floors, {best['n_collateral']} collateral, purity {best['purity']:.3f}")
    print(f"    sensitivity: argmax under W|.| = {alt['objective_abs_m']:.2f}, "
          f"under Wrealizable = {alt['objective_realizable_m']:.2f}, "
          f"restricted to the max side's grid (tau>=0.30) = {tau_maxgrid}")
    report["tau_star"] = tau_star
    report["tau_star_row"] = best
    report["tau_sensitivity"] = {**alt, "maxside_grid": tau_maxgrid}

    # ---- PART 2: the arms at tau* ---------------------------------------
    print("\n" + "=" * 100)
    print(f"  PART 2 — arms at tau* = {tau_star:.2f}")
    print("=" * 100)
    oofs, frss = {}, {}
    for kind in ("A", "B", "T"):
        oofs[kind], frss[kind] = arm(df, store, kind, tau_star, n_folds, n_seeds)

    print("\n  full sweep of the ex-ante arms (reported, NOT used for selection)")
    print(f"  {'tau':>5s} | {'A:win':>7s} {'A:dSel':>8s} {'A:t':>6s} {'A:A1':>7s} | "
          f"{'B:win':>7s} {'B:dSel':>8s} {'B:t':>6s} {'B:A1':>7s}")
    arm_sweep = []
    for tau in TAU_GRID:
        r = {"tau": float(tau)}
        for kind in ("A", "B"):
            o, f = arm(df, store, kind, float(tau), n_folds, n_seeds)
            zs = zone_scorecard(df, champ_oof, o, at_floor, sel)
            pdv = paired_delta(frs_champ, f)
            r[kind] = {"zone_win_sel_m": zs["sel"]["win"],
                       "zone_win_all_m": zs["all"]["win"],
                       "dSel": pdv["delta"], "dSel_t": pdv["t"],
                       "A1": float(r2_score(y, o))}
        arm_sweep.append(r)
        print(f"  {tau:5.2f} | {r['A']['zone_win_sel_m']:+7.2f} "
              f"{r['A']['dSel']:+8.5f} {r['A']['dSel_t']:+6.2f} {r['A']['A1']:7.4f} | "
              f"{r['B']['zone_win_sel_m']:+7.2f} {r['B']['dSel']:+8.5f} "
              f"{r['B']['dSel_t']:+6.2f} {r['B']['A1']:7.4f}")
    report["arm_sweep"] = arm_sweep

    # ---- B1 forward at tau* ----------------------------------------------
    print(f"\n  B1 forward at tau* = {tau_star:.2f}", flush=True)
    fwd = forward_pass(df, features, clf_features, lab6, tau_star, seeds)
    b1 = {}
    for k, pr in fwd.items():
        m = ~np.isnan(pr)
        b1[k] = float(r2_score(y[m], pr[m]))
        extra = "" if k == "champion" else f"   drop {b1['champion'] - b1[k]:+.4f}"
        print(f"    {k:9s} forward R2 {b1[k]:.4f}{extra}")
    report["B1"] = b1

    # ---- gate battery -----------------------------------------------------
    print("\n" + "=" * 100)
    print(f"  GATE BATTERY at tau* = {tau_star:.2f}  "
          "(zone gate on SELECTION rows only — ISSUES #20a)")
    print("=" * 100)
    gates = {}
    for kind in ("A", "B", "T"):
        o, f = oofs[kind], frss[kind]
        zs = zone_scorecard(df, champ_oof, o, at_floor, sel)
        br = brakes(df, champ_oof, o, at_floor)
        c2 = c2_segments(df, champ_oof, o)
        pdv = paired_delta(frs_champ, f)
        c2_worst = max((v["abs_bias_growth"] for v in c2.values()), default=0.0)
        c2_worst_signed = max((abs(v["signed_change"]) for v in c2.values()), default=0.0)
        brake_worst = max(v["abs_bias_growth"] for v in br.values())
        drop = b1["champion"] - b1[kind]
        ex_ante = kind in ("A", "B")
        g = {"told": not ex_ante,
             "zone_win_sel_m": zs["sel"]["win"], "zone_win_all_m": zs["all"]["win"],
             "zone_win_conf_m": zs.get("conf", {}).get("win"),
             "zone_pass": zs["sel"]["win"] >= ZONE_WIN_BAR,
             "brake_worst": brake_worst, "brake_pass": brake_worst <= BRAKE_BAR,
             "c2_worst_abs": c2_worst, "c2_worst_signed": c2_worst_signed,
             "c2_pass": c2_worst <= C2_BAR,
             "b1_drop": drop, "b1_pass": drop <= B1_BAR,
             "dSel": pdv["delta"], "dSel_se": pdv["se"], "dSel_t": pdv["t"],
             "dSel_per_fold": [float(v) for v in pdv["per_fold"]],
             "dSel_pass": (pdv["t"] > DSEL_T_BAR) if ex_ante else None,
             "A1": float(r2_score(y, o)),
             "A2": float(r2_score(y[recent], o[recent])),
             "zone": zs, "brakes": br, "c2": c2}
        checks = ["zone_pass", "brake_pass", "c2_pass", "b1_pass"]
        if ex_ante:
            checks.append("dSel_pass")
        g["all_pass"] = all(g[k] for k in checks)
        gates[kind] = g
        label = "TOLD (Stage-3)" if not ex_ante else "ex ante"
        print(f"\n  [{kind}] {label}")
        print(f"    zone MAE win  sel {g['zone_win_sel_m']:+.2f}M "
              f"({'PASS' if g['zone_pass'] else 'fail'})   "
              f"pooled {g['zone_win_all_m']:+.2f}M   "
              f"conf {g['zone_win_conf_m']:+.2f}M")
        print(f"    non-floor brake worst |bias| growth {g['brake_worst']:+.2f}M "
              f"({'PASS' if g['brake_pass'] else 'fail'})")
        for tag, v in br.items():
            print(f"        {tag:18s} n={v['n']:4d}  bias {v['champ_bias']:+6.2f} -> "
                  f"{v['cand_bias']:+6.2f}  |bias| growth {v['abs_bias_growth']:+.2f}  "
                  f"MAE {v['mae_change']:+.2f}")
        print(f"    C2 worst |bias| growth {g['c2_worst_abs']:+.2f}M "
              f"({'PASS' if g['c2_pass'] else 'fail'})   "
              f"worst |signed change| {g['c2_worst_signed']:+.2f}M")
        for cat, v in sorted(c2.items(), key=lambda kv: -kv[1]["abs_bias_growth"])[:4]:
            print(f"        {cat:14s} n={v['n']:4d}  {v['champ']:+6.2f} -> "
                  f"{v['cand']:+6.2f}   |bias| growth {v['abs_bias_growth']:+.2f}")
        print(f"    B1 drop {g['b1_drop']:+.4f} ({'PASS' if g['b1_pass'] else 'fail'})")
        if ex_ante:
            print(f"    dSel {g['dSel']:+.5f} +/- {g['dSel_se']:.5f}  t {g['dSel_t']:+.2f} "
                  f"({'PASS' if g['dSel_pass'] else 'fail'})")
        else:
            print(f"    dSel {g['dSel']:+.5f} +/- {g['dSel_se']:.5f}  t {g['dSel_t']:+.2f} "
                  f"(TOLD — reported, not gated as an ex-ante candidate)")
        print(f"    per fold {g['dSel_per_fold']}")
        print(f"    A1 {g['A1']:.4f}   A2 {g['A2']:.4f}   "
              f"VERDICT {'PASS' if g['all_pass'] else 'FAIL'}")
    report["gates"] = gates

    # ---- collateral at tau* ----------------------------------------------
    print("\n" + "=" * 100)
    print(f"  COLLATERAL at tau* = {tau_star:.2f} — who the pull-down damages")
    print("=" * 100)
    coll = (p >= tau_star) & ~at_floor
    for kind in ("A", "B"):
        e = (oofs[kind] - y) * cap_m
        rows = []
        for i in np.flatnonzero(coll)[np.argsort(-p[coll])]:
            rows.append({"player": df["player_name_norm"].iat[i],
                         "season": int(df["season"].iat[i]),
                         "p_floor": float(p[i]),
                         "salary_m": float(df["salary_m"].iat[i]),
                         "signing_cat": str(df["signing_cat"].iat[i]),
                         "champ_err_m": float(champ_err[i]),
                         "arm_err_m": float(e[i]),
                         "damage_m": float(abs(e[i]) - abs(champ_err[i]))})
        rows.sort(key=lambda r: -r["damage_m"])
        tot = sum(r["damage_m"] for r in rows)
        print(f"\n  [{kind}] {len(rows)} collateral rows, total damage ${tot:+.2f}M "
              f"(worst 12 shown)")
        for r in rows[:12]:
            print(f"    {r['player']:22s} {r['season']} P {r['p_floor']:.3f}  pay "
                  f"${r['salary_m']:6.2f}M  {r['signing_cat']:12s} champErr "
                  f"{r['champ_err_m']:+7.2f}  armErr {r['arm_err_m']:+7.2f}  damage "
                  f"{r['damage_m']:+7.2f}")
        report[f"collateral_{kind}"] = rows
        report[f"collateral_{kind}_total_m"] = tot

    # ---- touched true floors ---------------------------------------------
    tf = (p >= tau_star) & at_floor
    print(f"\n  TOUCHED TRUE FLOORS at tau*: {int(tf.sum())} of {int(at_floor.sum())}"
          f"  (realized win ${float(np.maximum(champ_err[tf],0).sum()):.1f}M available)")
    got = [{"player": df["player_name_norm"].iat[i], "season": int(df["season"].iat[i]),
            "p_floor": float(p[i]), "champ_err_m": float(champ_err[i]),
            "arm_A_err_m": float(((oofs["A"] - y) * cap_m)[i])}
           for i in np.flatnonzero(tf)[np.argsort(-champ_err[tf])]]
    for r in got[:12]:
        print(f"    {r['player']:22s} {r['season']} P {r['p_floor']:.3f}  champErr "
              f"{r['champ_err_m']:+7.2f} -> armErr {r['arm_A_err_m']:+7.2f}")
    report["touched_true_floors"] = got

    # ---- PART 3: can the classifier see the fallen stars? -----------------
    print("\n" + "=" * 100)
    print("  PART 3 — the fat tail: what P(floor) does the classifier give the "
          "fallen stars?")
    print("=" * 100)
    key = list(zip(df["player_name_norm"].values, df["season"].values))
    idx = {k: i for i, k in enumerate(key)}
    stars = []
    order = np.argsort(-champ_err * at_floor)
    pct_rank = {int(i): float((p >= p[i]).sum() / len(p)) for i in order[:40]}
    print(f"    {'row':30s} {'champErr':>9s} {'P(flr)':>7s} {'P pctile':>9s} "
          f"{'P base4':>8s} {'P enr4':>7s}")
    for name, season in FALLEN_STARS:
        i = idx.get((name, season))
        if i is None:
            print(f"    {name} {season}: NOT IN FRAME")
            continue
        rank = float((p > p[i]).mean())
        stars.append({"player": name, "season": season, "champ_err_m": float(champ_err[i]),
                      "p_floor": float(p[i]), "p_share_above": rank,
                      "p_base4": float(p_all["base4"][i]),
                      "p_enr4": float(p_all["enr4"][i]),
                      "touched_at_tau_star": bool(p[i] >= tau_star)})
        s = stars[-1]
        print(f"    {name + ' ' + str(season):30s} {s['champ_err_m']:+9.2f} "
              f"{s['p_floor']:7.3f} {1 - rank:9.3f} {s['p_base4']:8.3f} "
              f"{s['p_enr4']:7.3f}"
              + ("   <- touched" if s["touched_at_tau_star"] else ""))
    report["fallen_stars"] = stars

    # Does P rank the rows the branch needs to reach? This is the question the
    # whole branch turns on: a threshold rule can only work if the rows carrying
    # the error sit HIGH in P.
    from scipy.stats import spearmanr
    rho_zone = float(spearmanr(p[at_floor], champ_err[at_floor]).statistic)
    over = np.maximum(champ_err, 0)
    w_all = float(over[at_floor].sum())
    top_by_p = np.flatnonzero(at_floor)[np.argsort(-p[at_floor])]
    half = top_by_p[:len(top_by_p) // 2]
    print(f"\n    Spearman(P(floor), champion over-prediction) WITHIN the floor "
          f"zone: {rho_zone:+.3f}")
    print(f"    the half of the zone the classifier is MOST confident about "
          f"carries {over[half].sum() / w_all:.1%} of the zone's over-prediction "
          f"(50% would be no information; <50% means P is anti-ranked against "
          f"the error)")
    report["p_vs_error"] = {"spearman_in_zone": rho_zone,
                            "top_half_by_p_error_share": float(over[half].sum() / w_all)}

    top12 = np.flatnonzero(at_floor)[np.argsort(-champ_err[at_floor])[:12]]
    print(f"\n    the worst 12 floor rows carry "
          f"{np.abs(champ_err[top12]).sum() / np.abs(champ_err[at_floor]).sum():.1%} "
          f"of the zone's total error; their median P(floor) is "
          f"{np.median(p[top12]):.3f} against {np.median(p[at_floor]):.3f} for the "
          f"zone as a whole and {np.median(p[~at_floor]):.3f} for non-floor rows")
    report["fat_tail"] = {
        "worst12_error_share": float(np.abs(champ_err[top12]).sum()
                                     / np.abs(champ_err[at_floor]).sum()),
        "worst12_median_p": float(np.median(p[top12])),
        "zone_median_p": float(np.median(p[at_floor])),
        "nonfloor_median_p": float(np.median(p[~at_floor]))}

    # ---- PART 4: why -----------------------------------------------------
    print("\n" + "=" * 100)
    print("  PART 4 — why the branch does not convert")
    print("=" * 100)

    print("\n  (a) does the pre-registered objective predict what the arms do?")
    print(f"      at tau*: expected win ${best['expected_win_m']:.1f}M, "
          f"expected collateral ${best['expected_collateral_m']:.1f}M")
    calib = {}
    e_ch_abs = np.abs(champ_err)
    for kind in ("A", "B", "T"):
        e_a = np.abs((oofs[kind] - y) * cap_m)
        win = float((e_ch_abs[at_floor] - e_a[at_floor]).sum())
        dmg = float((e_a[~at_floor] - e_ch_abs[~at_floor]).sum())
        calib[kind] = {"realized_zone_reduction_m": win, "realized_damage_m": dmg,
                       "realized_net_m": win - dmg,
                       "expected_over_realized_win": (best["expected_win_m"] / win
                                                      if win else None),
                       "expected_over_realized_coll": (best["expected_collateral_m"] / dmg
                                                       if dmg else None)}
        print(f"      [{kind}] realized floor-zone error reduction ${win:7.1f}M, "
              f"non-floor damage ${dmg:7.1f}M, net ${win - dmg:+8.1f}M")
    print("      The collateral term is P-WEIGHTED, so it models the damage of a")
    print("      P-weighted pull (arm B) and understates an unconditional one")
    print("      (arm A). One tau is nevertheless applied to both — see ISSUES.")
    report["objective_calibration"] = calib

    print("\n  (b) is the fat tail separable on features the model already has?")
    prof_cols = [c for c in ("age", "mpg", "availability_3yr", "darko_dpm_z",
                             "lebron_z", "laker_z", "usage_pct", "prev_cap_pct",
                             "award_score_cum") if c in df.columns]
    rest = np.zeros(len(df), bool); rest[np.flatnonzero(at_floor)] = True
    rest[top12] = False
    prof = {}
    print(f"      {'feature':20s} {'worst12':>9s} {'zone rest':>10s} "
          f"{'non-floor':>10s}")
    for c in prof_cols:
        v = df[c].values
        prof[c] = {"worst12": float(v[top12].mean()),
                   "zone_rest": float(v[rest].mean()),
                   "nonfloor": float(v[~at_floor].mean())}
        print(f"      {c:20s} {prof[c]['worst12']:9.3f} "
              f"{prof[c]['zone_rest']:10.3f} {prof[c]['nonfloor']:10.3f}")
    print("      On every performance feature the worst floor rows look MORE like")
    print("      a well-paid player than the average NON-floor row does.")
    report["fat_tail_profile"] = prof

    print("\n  (c) what a better classifier could buy: error captured vs rows touched")
    zone_idx = np.flatnonzero(at_floor)
    order_p = zone_idx[np.argsort(-p[at_floor])]
    order_e = zone_idx[np.argsort(-over[at_floor])]
    cap_curve = []
    for frac in (0.25, 0.50, 0.75, 1.00):
        k = int(len(zone_idx) * frac)
        by_p = float(over[order_p[:k]].sum() / w_all)
        by_e = float(over[order_e[:k]].sum() / w_all)
        cap_curve.append({"frac": frac, "captured_by_p": by_p,
                          "captured_by_error": by_e})
        print(f"      top {frac:4.0%} of the zone by P captures {by_p:6.1%} of the "
              f"zone's over-prediction; an error-ordered selector would capture "
              f"{by_e:6.1%}")
    report["capture_curve"] = cap_curve

    # ---- floor_pct imprecision -------------------------------------------
    gap = (fp - y) * cap_m
    imp = {"at_floor_mae_m": float(np.abs(gap[at_floor]).mean()),
           "at_floor_bias_m": float(gap[at_floor].mean()),
           "at_floor_p90_m": float(np.percentile(np.abs(gap[at_floor]), 90)),
           "touched_mae_m": float(np.abs(gap[tf]).mean()) if tf.sum() else None}
    o_fp = champ_oof.copy(); o_fp[at_floor] = fp[at_floor]
    o_y = champ_oof.copy(); o_y[at_floor] = y[at_floor]
    imp["oracle_A1_to_floor_pct"] = float(r2_score(y, o_fp))
    imp["oracle_A1_to_actual"] = float(r2_score(y, o_y))
    imp["cost_of_recovery_A1"] = imp["oracle_A1_to_actual"] - imp["oracle_A1_to_floor_pct"]
    print(f"\n  floor_pct imprecision (ISSUES #6): on at-floor rows MAE "
          f"${imp['at_floor_mae_m']:.3f}M, bias ${imp['at_floor_bias_m']:+.3f}M, "
          f"p90 ${imp['at_floor_p90_m']:.3f}M")
    print(f"    oracle to floor_pct A1 {imp['oracle_A1_to_floor_pct']:.4f} vs to actual "
          f"A1 {imp['oracle_A1_to_actual']:.4f} — the recovered scale costs "
          f"{imp['cost_of_recovery_A1']:.4f} of A1")
    report["floor_pct_imprecision"] = imp

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "floor_branch_eval.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    dump = df[["player_name_norm", "season", TARGET, "salary_m", "signing_cat",
               "floor_pct", "max_eligible_pct", "is_at_floor",
               "is_confirmation"]].copy()
    dump["p_floor_enr6"] = p
    dump["p_floor_base4"] = p_all["base4"]
    dump["p_floor_enr4"] = p_all["enr4"]
    dump["oof_champion"] = champ_oof
    for kind in ("A", "B", "T"):
        dump[f"oof_{kind}_tau{tau_star:.2f}"] = oofs[kind]
    dump.to_csv(OUT / "floor_branch_oof.csv", index=False)
    print(f"\nSaved {OUT / 'floor_branch_eval.json'}")
    print(f"Saved {OUT / 'floor_branch_oof.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
