"""Part 3 of the extension-route brief, as amended 2026-07-25.

Phases 2 and 3 scored two operating points per classifier, both at the ends of the
purity curve, because "smallest tau with purity >= 0.90" lands on the LEFT EDGE of
a plateau. Nothing between them was ever run. This harness sweeps tau across the
whole range and selects one by a rule fixed before any arm is scored:

    expected win        = sum over touched TRUE MAX rows of |champion error|
    expected collateral = sum over touched NON-MAX rows of P x (ceiling - champion)
    tau*                = argmax (expected win - expected collateral)

Both quantities come from P, the champion's OOF predictions and the ceilings —
never from a realized zone metric. It is the honest-ceiling computation the briefs
already require, applied to both sides instead of one.

Two arms are swept, and the second is the amendment's question:

    A  push_clip                 the phase-3 ship form
    B  push_clip + extension clip   A, then clipped at the row's CBA extension
                                    ceiling where P(extension) >= 0.5

The 0.5 threshold on arm B is "more likely than not an extension" and is fixed
here, not tuned — the whole point of the amendment is that a threshold picked
after seeing the score is worthless. The extension ceiling is `ext_value_pct`
from extension_cap.py: the ex-ante counterfactual ceiling, knowable from
prev_cap_pct and a curated CBA table before the season starts.

Out of scope by the amendment, and not implemented: aiming the push by the
latent's ratio to its ceiling. The rows with the largest errors are exactly the
rows whose latent sits furthest below their ceiling, so such a threshold excludes
the players it most needs to reach.

Run:  OMP_NUM_THREADS=6 python scripts/eval_max_branch_tau_sweep.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from config import CAP_BY_SEASON, OUTPUTS_DIR
from src.model.train import FEATURE_COLS
from src.model.evaluate_suite import (
    load_evaluation_frame, make_grabit_fitter, oof_groupkfold, paired_delta,
    _dollars, N_SPLITS, DEFAULT_SEEDS, TARGET, FORWARD_ORIGINS,
)
from src.model import route_mixture as rm
from src.model.extension_cap import attach_extension_cap, attach_extension_value

SEEDS = DEFAULT_SEEDS
MARGIN = 1.05                  # fixed constant, never tuned on a zone metric
TAU_GRID = np.round(np.arange(0.30, 0.95, 0.02), 2)
# Arm B's fixed "more likely than not" threshold, plus two diagnostic arms that
# localize the failure: a looser gate and an unconditional clip (which is what an
# oracle on the route would do, and shows the mirror failure the QUEUE warns of).
P_EXT_CLIP = 0.50
CLIP_ARMS = {"A": None, "B": 0.50, "C": 0.25, "D": 0.0}

WIN_BAR, BRAKE_BAR, C2_BAR, B1_BAR, DSEL_T_BAR = 0.50, 0.30, 0.30, 0.003, -2.0
OUT = OUTPUTS_DIR / "models"


def zone_masks(df):
    y, me = df[TARGET].values, df["max_eligible_pct"].values
    true_max = y >= 0.90 * me
    counter = (~true_max) & (y >= 0.70 * me) & (y < 0.90 * me)
    return true_max, counter


def fit_pass(df, features, clf_features, lab4, lab6):
    """champion latent + P(max) [4-class] + P(extension) [6-class], per seed/fold."""
    y = df[TARGET].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, y,
                                                     df["player_name_norm"].values))
    sel = ~df["is_confirmation"].values
    store, acc_champ = [], np.zeros(len(df))
    acc_pmax, acc_pext = np.zeros(len(df)), np.zeros(len(df))
    frs_champ = np.zeros((len(folds), len(SEEDS)))
    for si, seed in enumerate(SEEDS):
        for fi, (tr, va) in enumerate(folds):
            train, test = df.iloc[tr], df.iloc[va]
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            champ = np.clip(latent, lo, hi)
            c4 = rm.train_route_classifier(train, clf_features, seed, labels=lab4[tr])
            p_max = rm.route_proba(c4, test, clf_features)[:, rm.MAX_IDX]
            c6 = rm.train_route6_classifier(train, clf_features, seed, labels=lab6[tr])
            p_ext = rm.route_proba(c6, test, clf_features)[:, rm.R6["extension"]]
            store.append({"si": si, "fi": fi, "va": va, "latent": latent, "lo": lo,
                          "hi": hi, "champ": champ, "p_max": p_max, "p_ext": p_ext})
            acc_champ[va] += champ
            acc_pmax[va] += p_max
            acc_pext[va] += p_ext
            vs = sel[va]
            frs_champ[fi, si] = r2_score(y[va][vs], champ[vs]) if vs.sum() > 10 else np.nan
        print(f"    seed {seed} done ({si + 1}/{len(SEEDS)})", flush=True)
    n = len(SEEDS)
    return store, acc_champ / n, acc_pmax / n, acc_pext / n, frs_champ, folds


def arm(df, store, tau, ext_clip, n_folds, n_seeds, ext_val):
    """ext_clip is None (no clip) or a P(extension) threshold (0.0 = every row)."""
    """OOF predictions + selection fold-R2 for one (tau, arm) cell."""
    y, sel = df[TARGET].values, ~df["is_confirmation"].values
    acc, frs = np.zeros(len(df)), np.zeros((n_folds, n_seeds))
    for rec in store:
        va, latent, lo, hi = rec["va"], rec["latent"], rec["lo"], rec["hi"]
        champ, p, pe = rec["champ"], rec["p_max"], rec["p_ext"]
        pushed = latent + p * (MARGIN * hi - latent)
        pred = np.clip(np.where(p >= tau, pushed, champ), lo, hi)
        if ext_clip is not None:
            ev = ext_val[va]
            m = (pe >= ext_clip) & ~np.isnan(ev)
            pred = np.where(m, np.minimum(pred, np.nan_to_num(ev, nan=np.inf)), pred)
            pred = np.clip(pred, lo, hi)
        acc[va] += pred
        vs = sel[va]
        frs[rec["fi"], rec["si"]] = (r2_score(y[va][vs], pred[vs])
                                     if vs.sum() > 10 else np.nan)
    return acc / n_seeds, frs


def main():
    pd.set_option("display.width", 250)
    df, features = load_evaluation_frame()
    df = attach_extension_cap(df)
    df = attach_extension_value(df)
    df, clf_features = rm.attach_clf_features(df)
    y = df[TARGET].values
    cap_m = df["season"].map(CAP_BY_SEASON).values / 1e6
    lab4, lab6 = rm.compute_route_labels(df), rm.compute_route6_labels(df)
    true_max, counter = zone_masks(df)
    ext_val = df["ext_value_pct"].values
    hi_all = df["max_eligible_pct"].values
    recent = df["season"].values >= 2024
    report = {"n_rows": len(df), "n_true_max": int(true_max.sum()),
              "n_counterweight": int(counter.sum()), "p_ext_clip": P_EXT_CLIP,
              "margin": MARGIN}

    print(f"\nframe {len(df)} | true-max zone {int(true_max.sum())} | "
          f"counterweight {int(counter.sum())} | rows with an extension ceiling "
          f"{int((~np.isnan(ext_val)).sum())}")
    print("\nFIT PASS — champion latent + enriched 4-class and 6-class classifiers")
    store, champ_oof, p_max, p_ext, frs_champ, folds = fit_pass(
        df, features, clf_features, lab4, lab6)
    n_folds, n_seeds = len(folds), len(SEEDS)

    champ_err = (champ_oof - y) * cap_m
    c_mae_max, c_bias_max = _dollars(df, champ_oof, true_max)
    _, c_bias_cw = _dollars(df, champ_oof, counter)
    c_a1 = float(r2_score(y, champ_oof))
    c_a2 = float(r2_score(y[recent], champ_oof[recent]))
    fixed25 = champ_oof >= 0.25
    _, c_b25 = _dollars(df, champ_oof, fixed25)
    print(f"\n  champion  A1 {c_a1:.4f}  A2 {c_a2:.4f}  zone MAE ${c_mae_max:.2f}M  "
          f"cw bias ${c_bias_cw:+.2f}M  25%+ bias ${c_b25:+.2f}M")
    report["champion"] = {"A1": c_a1, "A2": c_a2, "zone_mae": c_mae_max,
                          "cw_bias": c_bias_cw, "band25_bias": c_b25}

    # ---- the pre-registered selection quantities -------------------------
    print("\n" + "=" * 108)
    print("  TAU SWEEP — selection reads only expected win - expected collateral")
    print("=" * 108)
    print(f"  {'tau':>5s} {'n>=t':>5s} {'nMax':>5s} {'pur':>6s} | {'Ewin':>7s} "
          f"{'Ecoll':>7s} {'OBJ':>8s} |" + "".join(
              f" {t + ':win':>8s} {t + ':cw':>7s} {t + ':25':>7s} {t + ':t':>7s} |"
              for t in CLIP_ARMS))
    sweep = []
    cells = {}
    for tau in TAU_GRID:
        touched = p_max >= tau
        tm, nm = touched & true_max, touched & ~true_max
        e_win = float(np.abs(champ_err[tm]).sum())
        e_coll = float((p_max[nm] * (hi_all[nm] - champ_oof[nm]) * cap_m[nm]).sum())
        row = {"tau": float(tau), "n_above": int(touched.sum()),
               "n_true_max": int(tm.sum()), "n_collateral": int(nm.sum()),
               "purity": float(tm.sum() / touched.sum()) if touched.sum() else float("nan"),
               "expected_win_m": e_win, "expected_collateral_m": e_coll,
               "objective_m": e_win - e_coll}
        for tag, ec in CLIP_ARMS.items():
            oof, frs = arm(df, store, float(tau), ec, n_folds, n_seeds, ext_val)
            mae_max, _ = _dollars(df, oof, true_max)
            _, bias_cw = _dollars(df, oof, counter)
            _, b25 = _dollars(df, oof, fixed25)
            pdv = paired_delta(frs_champ, frs)
            row[tag] = {"win_m": c_mae_max - mae_max, "cw_growth": bias_cw - c_bias_cw,
                        "band25_growth": b25 - c_b25, "dSel": pdv["delta"],
                        "dSel_t": pdv["t"], "A1": float(r2_score(y, oof)),
                        "A2": float(r2_score(y[recent], oof[recent]))}
            cells[(float(tau), tag)] = (oof, frs)
        sweep.append(row)
        print(f"  {tau:5.2f} {row['n_above']:5d} {row['n_true_max']:5d} "
              f"{row['purity']:6.3f} | {e_win:7.1f} {e_coll:7.1f} "
              f"{row['objective_m']:8.1f} |" + "".join(
                  f" {row[t]['win_m']:+8.2f} {row[t]['cw_growth']:+7.2f} "
                  f"{row[t]['band25_growth']:+7.2f} {row[t]['dSel_t']:+7.2f} |"
                  for t in CLIP_ARMS))
    report["sweep"] = sweep

    best = max(sweep, key=lambda r: r["objective_m"])
    tau_star = best["tau"]
    print(f"\n  SELECTED tau* = {tau_star:.2f}  "
          f"(expected win ${best['expected_win_m']:.1f}M - expected collateral "
          f"${best['expected_collateral_m']:.1f}M = ${best['objective_m']:.1f}M)")
    print(f"    at tau*: {best['n_above']} rows above, {best['n_true_max']} true "
          f"maxes touched, {best['n_collateral']} collateral, purity "
          f"{best['purity']:.3f}")
    report["tau_star"] = tau_star
    report["tau_star_row"] = best

    # ---- collateral tables at tau* ---------------------------------------
    for tag in CLIP_ARMS:
        oof, _ = cells[(tau_star, tag)]
        err = (oof - y) * cap_m
        coll = (p_max >= tau_star) & ~true_max
        print(f"\n  [{tag}] COLLATERAL at tau*={tau_star:.2f}: {int(coll.sum())} rows")
        rows = []
        for i in np.flatnonzero(coll)[np.argsort(-p_max[coll])]:
            rows.append({"player": df["player_name_norm"].iat[i],
                         "season": int(df["season"].iat[i]),
                         "p_max": float(p_max[i]), "p_ext": float(p_ext[i]),
                         "salary_m": float(df["salary_m"].iat[i]),
                         "champ_err_m": float(champ_err[i]),
                         "arm_err_m": float(err[i]),
                         "damage_m": float(abs(err[i]) - abs(champ_err[i])),
                         "ext_ceiling_m": (float(ext_val[i] * cap_m[i])
                                           if not np.isnan(ext_val[i]) else None)})
            r = rows[-1]
            ec = f"{r['ext_ceiling_m']:7.2f}" if r["ext_ceiling_m"] else "      -"
            print(f"    {r['player']:22s} {r['season']} P {r['p_max']:.3f} "
                  f"Pext {r['p_ext']:.3f} pay {r['salary_m']:6.2f} champErr "
                  f"{r['champ_err_m']:+7.2f} armErr {r['arm_err_m']:+7.2f} damage "
                  f"{r['damage_m']:+7.2f} extCeil {ec}")
        print(f"    total damage ${sum(r['damage_m'] for r in rows):+.2f}M")
        report[f"collateral_{tag}"] = rows

    # ---- B1 forward at tau* ----------------------------------------------
    print(f"\n  B1 forward at tau*={tau_star:.2f}", flush=True)
    season = df["season"].values
    fwd = {k: np.full(len(df), np.nan) for k in ["champion"] + list(CLIP_ARMS)}
    for T in FORWARD_ORIGINS:
        te, tr = season == T, season < T
        if te.sum() < 10 or tr.sum() < 200:
            continue
        train, test = df[tr], df[te]
        acc = {k: np.zeros(int(te.sum())) for k in fwd}
        ev = ext_val[te]
        for seed in SEEDS:
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            champ = np.clip(latent, lo, hi)
            c4 = rm.train_route_classifier(train, clf_features, seed, labels=lab4[tr])
            pm = rm.route_proba(c4, test, clf_features)[:, rm.MAX_IDX]
            c6 = rm.train_route6_classifier(train, clf_features, seed, labels=lab6[tr])
            pe = rm.route_proba(c6, test, clf_features)[:, rm.R6["extension"]]
            pushed = latent + pm * (MARGIN * hi - latent)
            a = np.clip(np.where(pm >= tau_star, pushed, champ), lo, hi)
            acc["champion"] += champ
            for tag, thr in CLIP_ARMS.items():
                if thr is None:
                    acc[tag] += a
                    continue
                m = (pe >= thr) & ~np.isnan(ev)
                acc[tag] += np.clip(
                    np.where(m, np.minimum(a, np.nan_to_num(ev, nan=np.inf)), a),
                    lo, hi)
        for k in fwd:
            fwd[k][te] = acc[k] / len(SEEDS)
        print(f"    origin {T} done", flush=True)
    b1 = {}
    for k, p in fwd.items():
        m = ~np.isnan(p)
        b1[k] = float(r2_score(y[m], p[m]))
        print(f"    {k:9s} forward R2 {b1[k]:.4f}"
              + ("" if k == "champion" else f"   drop {b1['champion'] - b1[k]:+.4f}"))
    report["B1"] = b1

    # ---- gate battery at tau* --------------------------------------------
    print("\n" + "=" * 74)
    print(f"  GATE BATTERY at tau* = {tau_star:.2f}")
    print("=" * 74)
    gates = {}
    for tag in CLIP_ARMS:
        oof, frs = cells[(tau_star, tag)]
        s = best[tag]
        c2 = {}
        for cat, sub in df.groupby("signing_cat"):
            if len(sub) < 10:
                continue
            m = (df["signing_cat"] == cat).values
            _, b0 = _dollars(df, champ_oof, m)
            _, b1v = _dollars(df, oof, m)
            c2[str(cat)] = abs(b1v) - abs(b0)
        c2w = max(c2.values(), default=0.0)
        drop = b1["champion"] - b1[tag]
        g = {"win_m": s["win_m"], "win_pass": s["win_m"] >= WIN_BAR,
             "cw_growth": s["cw_growth"], "cw_pass": s["cw_growth"] <= BRAKE_BAR,
             "band25_growth": s["band25_growth"],
             "band25_pass": s["band25_growth"] <= BRAKE_BAR,
             "dSel": s["dSel"], "dSel_t": s["dSel_t"],
             "dSel_pass": s["dSel_t"] > DSEL_T_BAR,
             "c2_worst": c2w, "c2_pass": c2w <= C2_BAR,
             "b1_drop": drop, "b1_pass": drop <= B1_BAR,
             "A1": s["A1"], "A2": s["A2"]}
        g["all_pass"] = all(g[k] for k in ("win_pass", "cw_pass", "band25_pass",
                                           "dSel_pass", "c2_pass", "b1_pass"))
        gates[tag] = g
        print(f"  [{tag}] win ${g['win_m']:+.2f}M ({'PASS' if g['win_pass'] else 'fail'})"
              f"  cw {g['cw_growth']:+.2f} ({'PASS' if g['cw_pass'] else 'fail'})"
              f"  25band {g['band25_growth']:+.2f} "
              f"({'PASS' if g['band25_pass'] else 'fail'})")
        print(f"       dSel {g['dSel']:+.5f} t {g['dSel_t']:+.2f} "
              f"({'PASS' if g['dSel_pass'] else 'fail'})  C2 {g['c2_worst']:+.2f} "
              f"({'PASS' if g['c2_pass'] else 'fail'})  B1 drop {g['b1_drop']:+.4f} "
              f"({'PASS' if g['b1_pass'] else 'fail'})")
        print(f"       A1 {g['A1']:.4f}  A2 {g['A2']:.4f}   "
              f"VERDICT {'PASS' if g['all_pass'] else 'FAIL'}")
    report["gates"] = gates

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "max_branch_tau_sweep.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    dump = df[["player_name_norm", "season", TARGET, "salary_m", "signing_cat",
               "max_eligible_pct", "ext_value_pct", "is_extension", "ext_kind",
               "is_confirmation"]].copy()
    dump["p_max"], dump["p_ext"] = p_max, p_ext
    dump["oof_champion"] = champ_oof
    for tag in CLIP_ARMS:
        dump[f"oof_{tag}_tau{tau_star:.2f}"] = cells[(tau_star, tag)][0]
    dump.to_csv(OUT / "max_branch_tau_sweep_oof.csv", index=False)
    print(f"\nSaved {OUT / 'max_branch_tau_sweep.json'}")
    print(f"Saved {OUT / 'max_branch_tau_sweep_oof.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
