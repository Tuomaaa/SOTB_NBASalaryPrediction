"""Evidence harness for route-mixture phase 2 (2026-07-25-route-mixture-p2 brief).

Phase 2 = enriched classifier + purity-gated max branch. Three parts:

  Part 1  Enrich the 4-class classifier with the failed feature-batch columns as
          CLASSIFIER-ONLY inputs. Report P(max)/P(floor)/P(mle) AUC, median P on
          true, non-max above 0.5, calibration, and the PURITY CURVE — for the
          base classifier (FEATURE_COLS only, on this pin) AND the enriched one,
          vs the phase-1 published baselines.
  Part 2  Choose tau from the purity curve: smallest tau with OOF purity
          (empirical max-rate at P>=tau) >= 90%, tau<=0.95. If none reaches 90%,
          STOP — the branch waits for a better classifier.
  Part 3  The gated max branch: push-then-clip and hard, gated at tau. Champion
          references recomputed on the fixed v7.11x rows. Win / brake / guardrail
          battery, per-fold ΔSel, C2 segments, B1 forward, the touched-row
          collateral list, and the honest win ceiling.

Everything is fold-honest: for each (fold, seed) the champion regression and the
route classifier are fit on the training slice only; P(max) is an OUTPUT weight
applied to the test-slice latent and never joins the regression FEATURE_COLS.
The enrichment columns join the CLASSIFIER's feature list only.

Run:  OMP_NUM_THREADS=6 python scripts/eval_route_mixture_p2.py
"""

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
    load_evaluation_frame, make_grabit_fitter, rolling_forward,
    paired_delta, _dollars, N_SPLITS, DEFAULT_SEEDS, TARGET,
)
from src.model import route_mixture as rm

SEEDS = DEFAULT_SEEDS
MARGIN = 1.05
PURITY_TARGET = 0.90
TAU_CAP = 0.95
OUT = Path(__file__).resolve().parent.parent / "outputs" / "models"


# ---------------------------------------------------------------------------
# Classifier OOF on an arbitrary feature list (Part 1)
# ---------------------------------------------------------------------------

def classifier_oof(df, clf_features, seeds=SEEDS):
    """Seed-averaged fold-honest OOF (n,4) with the suite's GroupKFold folds."""
    labels = rm.compute_route_labels(df)
    groups = df["player_name_norm"].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, labels, groups))
    acc = np.zeros((len(df), len(rm.ROUTE_CLASSES)))
    for seed in seeds:
        for tr, va in folds:
            model = rm.train_route_classifier(df.iloc[tr], clf_features, seed,
                                              labels=labels[tr])
            acc[va] += rm.route_proba(model, df.iloc[va], clf_features)
    return acc / len(seeds), labels


def class_diagnostics(df, proba, labels, tag):
    """Per-class AUC / median / calibration; returns dict, prints table."""
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
        if cls == "max":
            d["non_max_above_0.5"] = int(((labels != idx) & (p > 0.5)).sum())
        out[cls] = d
        extra = (f"  non-max>0.5={d['non_max_above_0.5']}"
                 if cls == "max" else "")
        print(f"    P({cls}): AUC {auc:.4f}  median_on_true {med:.4f}  "
              f"n_true={int(true.sum())}{extra}")
        if cls == "max":
            for row in cal:
                print(f"        {row[0]:12s} n={row[1]:4d} meanP={row[2]:.3f} "
                      f"emp={row[3]:.3f}")
    return out


def purity_curve(proba, labels, grid=None):
    """Cumulative purity: among OOF rows with P(max)>=tau, the true-max rate."""
    p = proba[:, rm.MAX_IDX]
    is_max = (labels == rm.MAX_IDX)
    if grid is None:
        grid = np.round(np.arange(0.30, 0.96, 0.01), 2)
    curve = []
    for t in grid:
        m = p >= t
        n = int(m.sum())
        purity = float(is_max[m].mean()) if n > 0 else float("nan")
        n_true = int(is_max[m].sum())
        curve.append({"tau": float(t), "n_at_or_above": n,
                      "n_true_max": n_true, "purity": purity})
    return curve


def choose_tau(curve):
    """Smallest tau with purity >= PURITY_TARGET and tau <= TAU_CAP."""
    for row in curve:
        if row["tau"] <= TAU_CAP and row["n_at_or_above"] > 0 \
                and row["purity"] >= PURITY_TARGET:
            return row["tau"], row
    return None, None


# ---------------------------------------------------------------------------
# Combined OOF for Part 3: champion + gated arms, one fit per (fold, seed)
# ---------------------------------------------------------------------------

def combined_oof(df, features, clf_features, tau, seeds=SEEDS):
    y = df[TARGET].values
    groups = df["player_name_norm"].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, y, groups))
    sel = ~df["is_confirmation"].values

    variants = ["champion", "push_clip_gated", "hard_gated"]
    acc = {v: np.zeros(len(df)) for v in variants}
    pacc = np.zeros(len(df))          # seed-averaged P(max), enriched classifier
    frs = {v: np.zeros((len(folds), len(seeds))) for v in variants}
    labels = rm.compute_route_labels(df)

    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(folds):
            train, test = df.iloc[tr], df.iloc[va]
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            champ = np.clip(latent, lo, hi)
            clf = rm.train_route_classifier(train, clf_features, seed,
                                            labels=labels[tr])
            p_max = rm.route_proba(clf, test, clf_features)[:, rm.MAX_IDX]
            gate = p_max >= tau
            pushed = np.clip(np.where(
                gate, latent + p_max * (MARGIN * hi - latent), champ), lo, hi)
            hard = np.clip(np.where(gate, hi, champ), lo, hi)
            preds = {"champion": champ, "push_clip_gated": pushed,
                     "hard_gated": hard}
            vs = sel[va]
            pacc[va] += p_max
            for v in variants:
                acc[v][va] += preds[v]
                frs[v][fi, si] = (r2_score(y[va][vs], preds[v][vs])
                                  if vs.sum() > 10 else np.nan)
    return ({v: acc[v] / len(seeds) for v in variants},
            frs, pacc / len(seeds), labels)


def zone_masks(df):
    y = df[TARGET].values
    me = df["max_eligible_pct"].values
    true_max = (y >= 0.90 * me)
    counter = (~true_max) & (y >= 0.70 * me) & (y < 0.90 * me)
    return true_max, counter


def main():
    df, features = load_evaluation_frame()
    df, clf_features = rm.attach_clf_features(df)
    y = df[TARGET].values
    print(f"Loaded {len(df)} rows | regression features {len(features)} | "
          f"classifier features {len(clf_features)} "
          f"({len(rm.CLF_EXTRA_COLS)} enrichment cols, native NaN)")

    report = {"n_rows": len(df), "regression_features": features,
              "classifier_features": clf_features,
              "enrichment_cols": rm.CLF_EXTRA_COLS,
              "phase1_baselines": {"p_max_auc": 0.9650, "p_max_median": 0.5217,
                                   "non_max_above_0.5": 13, "p_floor_auc": 0.8233,
                                   "p_mle_auc": 0.7158}}

    # ---- PART 1 --------------------------------------------------------
    print("\n" + "=" * 72)
    print("  PART 1 — classifier diagnostics (10-seed fold-honest OOF)")
    print("  phase-1 baselines: P(max) AUC 0.9650, median 0.5217, 13 non-max>0.5,")
    print("                     P(floor) AUC 0.8233, P(mle) AUC 0.7158")
    print("=" * 72)

    base_proba, labels = classifier_oof(df, list(FEATURE_COLS))
    report["base_classifier"] = class_diagnostics(
        df, base_proba, labels, "BASE classifier (FEATURE_COLS only, this pin)")

    enr_proba, _ = classifier_oof(df, clf_features)
    report["enriched_classifier"] = class_diagnostics(
        df, enr_proba, labels, "ENRICHED classifier (FEATURE_COLS + batch)")

    # purity curves
    base_curve = purity_curve(base_proba, labels)
    enr_curve = purity_curve(enr_proba, labels)
    report["purity_curve_base"] = base_curve
    report["purity_curve_enriched"] = enr_curve

    print("\n  PURITY CURVE (cumulative: true-max rate among rows with P>=tau)")
    print(f"    {'tau':>5s} | {'BASE n':>7s} {'nMax':>5s} {'purity':>7s} | "
          f"{'ENR n':>6s} {'nMax':>5s} {'purity':>7s}")
    show = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    bmap = {r["tau"]: r for r in base_curve}
    emap = {r["tau"]: r for r in enr_curve}
    for t in show:
        b, e = bmap.get(t), emap.get(t)
        bp = f"{b['purity']:.3f}" if b and b["n_at_or_above"] else "  -  "
        ep = f"{e['purity']:.3f}" if e and e["n_at_or_above"] else "  -  "
        print(f"    {t:5.2f} | {b['n_at_or_above']:7d} {b['n_true_max']:5d} "
              f"{bp:>7s} | {e['n_at_or_above']:6d} {e['n_true_max']:5d} {ep:>7s}")

    # ---- PART 2 --------------------------------------------------------
    print("\n" + "=" * 72)
    print("  PART 2 — choose tau from the ENRICHED purity curve")
    print(f"  rule: smallest tau with purity >= {PURITY_TARGET:.0%}, tau <= {TAU_CAP}")
    print("=" * 72)
    tau, tau_row = choose_tau(enr_curve)
    report["tau"] = tau
    report["tau_row"] = tau_row
    if tau is None:
        best = max((r for r in enr_curve if r["tau"] <= TAU_CAP
                    and r["n_at_or_above"] > 0),
                   key=lambda r: r["purity"], default=None)
        print(f"  NO tau <= {TAU_CAP} reaches {PURITY_TARGET:.0%} purity.")
        if best:
            print(f"  best available: tau={best['tau']:.2f}  purity "
                  f"{best['purity']:.3f}  n={best['n_at_or_above']} "
                  f"({best['n_true_max']} true max)")
        print("  RESULT: STOP at Part 1 — branch waits for a better classifier.")
        report["verdict"] = "STOP_NO_TAU"
        _save(report, enr_proba, base_proba, labels, df, None)
        return
    print(f"  chosen tau = {tau:.2f}   purity {tau_row['purity']:.3f}   "
          f"n>=tau {tau_row['n_at_or_above']} ({tau_row['n_true_max']} true max, "
          f"{tau_row['n_at_or_above'] - tau_row['n_true_max']} non-max)")

    # ---- PART 3 --------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  PART 3 — gated max branch (tau={tau:.2f}) on fixed v7.11x rows")
    print("=" * 72)

    oofs, frs, p_oof, _ = combined_oof(df, features, clf_features, tau)
    champ_oof = oofs["champion"]

    # champion reproduction cross-check
    ref_path = OUTPUTS_DIR / "models" / "evaluation_suite.json"
    if ref_path.exists():
        stored = json.load(open(ref_path))
        ref = np.array(stored["fold_r2_selection"]["champion"])
        print(f"\n  champion fold_r2_sel reproduction max|diff| vs stored: "
              f"{np.abs(frs['champion'] - ref).max():.2e} "
              f"(stored A1={stored['champion']['A1_cv_r2']:.4f})")
    print(f"  champion A1 (this run) = {r2_score(y, champ_oof):.4f}")

    true_max, counter = zone_masks(df)
    c_mae_max, c_bias_max = _dollars(df, champ_oof, true_max)
    c_mae_cw, c_bias_cw = _dollars(df, champ_oof, counter)
    c25 = champ_oof >= 0.25
    _, c_bias_25 = _dollars(df, champ_oof, c25)
    print(f"\n  champion references (fixed rows):")
    print(f"    true-max zone (n={int(true_max.sum())}): MAE ${c_mae_max:.2f}M "
          f"bias ${c_bias_max:+.2f}M")
    print(f"    counterweight [0.70,0.90) non-max (n={int(counter.sum())}): "
          f"bias ${c_bias_cw:+.2f}M MAE ${c_mae_cw:.2f}M")
    print(f"    25%+ predicted band (n={int(c25.sum())}): bias ${c_bias_25:+.2f}M")

    # honest win ceiling: sum of champion abs error on TOUCHED true maxes / n_zone
    touched_max = true_max & (p_oof >= tau)
    cap_m = df["cap"].values / 1e6
    champ_err = (champ_oof - y) * cap_m
    ceiling_win = float(np.abs(champ_err[touched_max]).sum() / true_max.sum())
    print(f"\n  honest win ceiling for tau={tau:.2f}: touched true maxes = "
          f"{int(touched_max.sum())} of {int(true_max.sum())}; "
          f"max possible zone-MAE improvement = ${ceiling_win:.2f}M")
    report["honest_win_ceiling_m"] = ceiling_win
    report["n_touched_true_max"] = int(touched_max.sum())

    # arm table
    print(f"\n  {'arm':16s} {'maxMAE':>7s} {'dMAE':>7s} | {'cwBias':>7s} "
          f"{'dCW':>6s} | {'25band':>7s} {'d25':>6s} | {'dSel':>8s} {'t':>6s} "
          f"| {'A1':>7s} {'A2':>7s}")
    recent = df["season"].values >= 2024
    arms = {}
    for arm in ["champion", "push_clip_gated", "hard_gated"]:
        oof = oofs[arm]
        mae_max, bias_max = _dollars(df, oof, true_max)
        mae_cw, bias_cw = _dollars(df, oof, counter)
        b25 = oof >= 0.25
        _, bias_25 = _dollars(df, oof, b25)
        pdelta = paired_delta(frs["champion"], frs[arm])
        a1 = float(r2_score(y, oof))
        a2 = float(r2_score(y[recent], oof[recent]))
        arms[arm] = {
            "true_max_mae": mae_max, "d_true_max_mae": mae_max - c_mae_max,
            "true_max_bias": bias_max, "cw_bias": bias_cw,
            "d_cw_bias": bias_cw - c_bias_cw, "band25_bias": bias_25,
            "band25_n": int(b25.sum()), "d_band25_bias": bias_25 - c_bias_25,
            "dSel": pdelta["delta"], "dSel_se": pdelta["se"],
            "dSel_t": pdelta["t"], "dSel_per_fold": pdelta["per_fold"],
            "A1": a1, "A2": a2}
        print(f"  {arm:16s} {mae_max:7.2f} {mae_max-c_mae_max:+7.2f} | "
              f"{bias_cw:+7.2f} {bias_cw-c_bias_cw:+6.2f} | {bias_25:+7.2f} "
              f"{bias_25-c_bias_25:+6.2f} | {pdelta['delta']:+8.4f} "
              f"{pdelta['t']:+6.2f} | {a1:7.4f} {a2:7.4f}")
    report["arms"] = arms
    print(f"\n  push_clip_gated per-fold ΔSel: {arms['push_clip_gated']['dSel_per_fold']}")
    print(f"  hard_gated      per-fold ΔSel: {arms['hard_gated']['dSel_per_fold']}")

    # ---- gates ---------------------------------------------------------
    def gate_block(arm):
        a = arms[arm]
        win = -a["d_true_max_mae"]          # improvement is negative dMAE
        g = {
            "win_zone_mae_improvement_m": win,
            "win_pass": win >= 0.50,
            "brake_cw_growth_m": a["d_cw_bias"],
            "brake_cw_pass": a["d_cw_bias"] <= 0.30,
            "brake_25_growth_m": a["d_band25_bias"],
            "brake_25_pass": a["d_band25_bias"] <= 0.30,
            "guard_dSel_t": a["dSel_t"], "guard_dSel_pass": a["dSel_t"] > -2,
        }
        return g

    gates = {arm: gate_block(arm) for arm in ["push_clip_gated", "hard_gated"]}

    # C2 fixed-segment integrity (ship = push_clip_gated)
    push_oof = oofs["push_clip_gated"]
    print("\n  C2 fixed-segment bias growth (push_clip_gated - champion, same rows):")
    c2 = {}
    if "signing_cat" in df.columns:
        for cat, sub in df.groupby("signing_cat"):
            if len(sub) >= 10:
                m = (df["signing_cat"] == cat).values
                _, bch = _dollars(df, champ_oof, m)
                _, bps = _dollars(df, push_oof, m)
                growth = bps - bch
                c2[str(cat)] = {"n": int(m.sum()), "champ": bch, "push": bps,
                                "growth": growth}
                flag = "  <-- >0.3M" if abs(growth) > 0.30 else ""
                print(f"    {cat:14s} n={int(m.sum()):4d} ${bch:+.2f}M -> "
                      f"${bps:+.2f}M  (Δ{growth:+.2f}M){flag}")
    c2_breach = any(abs(v["growth"]) > 0.30 for v in c2.values())
    report["C2"] = c2

    # B1 forward
    print("\n  B1 forward (rolling-origin 2024-26):")
    b1 = {}
    fitters = {
        "champion": make_grabit_fitter(),
        "push_clip_gated": rm.make_maxbranch_fitter(
            enabled=True, arm="push_clip", margin=MARGIN, tau=tau,
            clf_features=clf_features),
    }
    for arm, fitter in fitters.items():
        fwd = rolling_forward(df, features, fitter, seeds=SEEDS)
        scored = ~np.isnan(fwd)
        r2f = float(r2_score(y[scored], fwd[scored]))
        by_origin = {}
        for T in (2024, 2025, 2026):
            m = scored & (df["season"].values == T)
            if m.sum() >= 10:
                by_origin[int(T)] = {
                    "n": int(m.sum()), "r2": float(r2_score(y[m], fwd[m])),
                    "mae": _dollars(df, fwd, m)[0],
                    "maxzone_mae": _dollars(df, fwd, m & true_max)[0]}
        b1[arm] = {"r2": r2f, "by_origin": by_origin}
        print(f"    {arm:16s} forward R2 {r2f:.4f}  n={int(scored.sum())}")
        for T, d in by_origin.items():
            print(f"        origin {T} R2 {d['r2']:.4f} MAE ${d['mae']:.2f}M "
                  f"maxzone MAE ${d['maxzone_mae']:.2f}M n={d['n']}")
    b1_drop = b1["champion"]["r2"] - b1["push_clip_gated"]["r2"]
    print(f"    B1 drop (champion - push_clip_gated) = {b1_drop:+.4f}")
    report["B1"] = b1
    report["B1_drop"] = b1_drop

    # guardrail wrap-up
    for arm in gates:
        gates[arm]["guard_C2_breach"] = c2_breach
        gates[arm]["guard_C2_pass"] = not c2_breach
    gates["push_clip_gated"]["guard_B1_drop"] = b1_drop
    gates["push_clip_gated"]["guard_B1_pass"] = b1_drop <= 0.003
    report["gates"] = gates

    # ---- collateral list: every non-max row above tau ------------------
    above = (p_oof >= tau) & (~true_max)
    coll = df.loc[above, ["player_name_norm", "season", "signing_cat",
                          "salary_m", "max_eligible_pct"]].copy()
    coll["p_max"] = p_oof[above]
    coll["cap_pct"] = y[above]
    coll["champ_pred_pct"] = champ_oof[above]
    coll["push_pred_pct"] = push_oof[above]
    coll["push_damage_m"] = (push_oof[above] - champ_oof[above]) * cap_m[above]
    coll = coll.sort_values("p_max", ascending=False)
    print(f"\n  COLLATERAL — non-max rows with P(max) >= tau ({int(above.sum())}):")
    if above.sum():
        print(f"    {'player':22s} {'seas':>4s} {'P':>5s} {'cap%':>6s} "
              f"{'push_dmg$M':>10s}  signing_cat")
        for _, r in coll.iterrows():
            print(f"    {r['player_name_norm']:22s} {int(r['season']):4d} "
                  f"{r['p_max']:.3f} {r['cap_pct']*100:5.1f}% "
                  f"{r['push_damage_m']:+9.2f}  {r['signing_cat']}")
    else:
        print("    (none)")
    report["collateral"] = coll.assign(
        season=coll["season"].astype(int)).to_dict("records")

    # verdict
    ship = gates["push_clip_gated"]
    all_pass = (ship["win_pass"] and ship["brake_cw_pass"] and
                ship["brake_25_pass"] and ship["guard_dSel_pass"] and
                ship["guard_C2_pass"] and ship["guard_B1_pass"])
    report["verdict"] = "GATES" if all_pass else "DOES_NOT_GATE"
    print(f"\n  SHIP-FORM (push_clip_gated) verdict: "
          f"{'GATES' if all_pass else 'DOES NOT GATE'}")

    _save(report, enr_proba, base_proba, labels, df, p_oof, oofs)


def _save(report, enr_proba, base_proba, labels, df, p_oof, oofs=None):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "route_mixture_p2_eval.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    dump = df[["player_name_norm", "season", TARGET, "salary_m", "signing_cat",
               "max_eligible_pct", "is_max_contract"]].copy()
    dump["p_max_base"] = base_proba[:, rm.MAX_IDX]
    dump["p_max_enriched"] = enr_proba[:, rm.MAX_IDX]
    dump["p_floor_enriched"] = enr_proba[:, rm.FLOOR_IDX]
    dump["route_label"] = [rm.ROUTE_CLASSES[i] for i in labels]
    if p_oof is not None:
        dump["p_max_oof_part3"] = p_oof
    if oofs is not None:
        dump["oof_champion"] = oofs["champion"]
        dump["oof_push_clip_gated"] = oofs["push_clip_gated"]
        dump["oof_hard_gated"] = oofs["hard_gated"]
    dump.to_csv(OUT / "route_mixture_p2_oof.csv", index=False)
    print(f"\nSaved {OUT / 'route_mixture_p2_eval.json'}")
    print(f"Saved {OUT / 'route_mixture_p2_oof.csv'}")


if __name__ == "__main__":
    main()
