"""Evidence harness for the route-mixture phase-1 brief (2026-07-25).

Produces every number the RESULT reports:

  Part 1  classifier diagnostics — per-class AUC, calibration curves, the
          P(max) distribution on true maxes, and the phase-2 diagnostic (which
          true maxes stay below P=0.3).
  Part 2  the max branch — champion vs three arms (push_clip ship form, mean
          reference r1, hard-switch reference r2) on FIXED v7.10x rows:
          true-max zone MAE, the counterweight-band brake, the 25%+ predicted
          band, the paired selection delta, C2 segments, B1 forward, and the
          pinned-share curve.

Everything is fold-honest: for each (fold, seed) the champion regression and
the route classifier are fit on the training slice only, and P(max) is an
OUTPUT weight applied to the test-slice latent. The classifier probability
array never joins FEATURE_COLS.

Run:  OMP_NUM_THREADS=6 python scripts/eval_route_mixture.py
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
from src.model.evaluate_suite import (
    load_evaluation_frame, make_grabit_fitter, baseline_fitter,
    oof_groupkfold, rolling_forward, layer_c, grabit_zone, paired_delta,
    _dollars, N_SPLITS, DEFAULT_SEEDS, TARGET,
)
from src.model import route_mixture as rm

SEEDS = DEFAULT_SEEDS
MARGIN = 1.05


# ---------------------------------------------------------------------------
# Combined OOF: fit grabit + classifier ONCE per (fold, seed), derive all arms
# ---------------------------------------------------------------------------

def combined_oof(df, features, seeds=SEEDS):
    """Return {variant: (oof, fold_r2, fold_r2_sel)} over champion + 3 arms.

    Mirrors evaluate_suite.oof_groupkfold exactly (same folds, same selection
    mask, seed-averaged OOF, per-(fold,seed) R2 matrices) but computes the
    shared latent / P(max) a single time and composes every arm from it.
    """
    y = df[TARGET].values
    groups = df["player_name_norm"].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, y, groups))
    sel = ~df["is_confirmation"].values
    max_elig = df["max_eligible_pct"].values
    floor_pct = df["floor_pct"].values

    variants = ["champion", "push_clip", "mean", "hard"]
    acc = {v: np.zeros(len(df)) for v in variants}
    fr = {v: np.zeros((len(folds), len(seeds))) for v in variants}
    frs = {v: np.zeros((len(folds), len(seeds))) for v in variants}

    labels = rm.compute_route_labels(df)

    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(folds):
            train, test = df.iloc[tr], df.iloc[va]
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            champ = np.clip(latent, lo, hi)
            clf = rm.train_route_classifier(train, features, seed,
                                            labels=labels[tr])
            p_max = rm.route_proba(clf, test, features)[:, rm.MAX_IDX]

            preds = {
                "champion": champ,
                "push_clip": np.clip(latent + p_max * (MARGIN * hi - latent),
                                     lo, hi),
                "mean": np.clip(p_max * hi + (1 - p_max) * champ, lo, hi),
                "hard": np.clip(np.where(p_max > 0.5, hi, champ), lo, hi),
            }
            vs = sel[va]
            for v in variants:
                acc[v][va] += preds[v]
                fr[v][fi, si] = r2_score(y[va], preds[v])
                frs[v][fi, si] = (r2_score(y[va][vs], preds[v][vs])
                                  if vs.sum() > 10 else np.nan)

    return {v: (acc[v] / len(seeds), fr[v], frs[v]) for v in variants}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def zone_masks(df):
    y = df[TARGET].values
    me = df["max_eligible_pct"].values
    true_max = (y >= 0.90 * me)                      # n=56, the win zone
    counterweight = (~true_max) & (y >= 0.70 * me) & (y < 0.90 * me)  # brake
    return true_max, counterweight


def band_bias(df, pred, mask):
    mae, bias = _dollars(df, pred, mask)
    return mae, bias, int(mask.sum())


def predicted_25_band(df, pred):
    """Rows whose PREDICTED value lands in the 25%+ calibration band."""
    return pred >= 0.25


def main():
    df, features = load_evaluation_frame()
    print(f"Loaded {len(df)} rows, {len(features)} features")
    y = df[TARGET].values

    # ---- Part 1: classifier diagnostics --------------------------------
    print("\n" + "=" * 70)
    print("  PART 1 — classifier diagnostics (10-seed fold-honest OOF)")
    print("=" * 70)
    proba = rm.oof_route_proba(df, features, seeds=SEEDS)
    labels = rm.compute_route_labels(df)
    p1 = {}
    for cls, idx in (("max", rm.MAX_IDX), ("mle", rm.MLE_IDX),
                     ("floor", rm.FLOOR_IDX)):
        true = (labels == idx).astype(int)
        p = proba[:, idx]
        auc = roc_auc_score(true, p)
        med_true = float(np.median(p[true == 1]))
        # calibration curve (bin by predicted P)
        edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        cal = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (p >= a) & (p < b)
            if m.sum() >= 3:
                cal.append((f"[{a:.1f},{b:.1f})", int(m.sum()),
                            round(float(p[m].mean()), 3),
                            round(float(true[m].mean()), 3)))
        p1[cls] = {"auc": auc, "median_p_on_true": med_true,
                   "n_true": int(true.sum()), "calibration": cal}
        print(f"\n  P({cls}): AUC {auc:.4f}   median P on true {cls} "
              f"{med_true:.4f}   n_true={int(true.sum())}")
        if cls == "max":
            above = int(((labels != idx) & (p > 0.5)).sum())
            p1[cls]["non_max_above_0.5"] = above
            print(f"           non-{cls} rows with P>0.5: {above}   "
                  f"(baselines: AUC 0.9595, median 0.356, 22 above 0.5)")
        print(f"           calibration [bin: n, mean_P, emp_rate]:")
        for row in cal:
            print(f"             {row[0]:12s} n={row[1]:4d}  meanP={row[2]:.3f}  "
                  f"emp={row[3]:.3f}")

    # phase-2 diagnostic: true maxes that stay below P=0.3
    tmax = labels == rm.MAX_IDX
    pmax = proba[:, rm.MAX_IDX]
    low = tmax & (pmax < 0.30)
    lowtab = df.loc[low, ["player_name_norm", "season"]].copy()
    lowtab["p_max"] = pmax[low]
    lowtab = lowtab.sort_values("p_max")
    p1["max"]["median_p_true_max"] = float(np.median(pmax[tmax]))
    print(f"\n  Phase-2 diagnostic: median P on true maxes = "
          f"{np.median(pmax[tmax]):.4f}  (baseline 0.356)")
    print(f"  True maxes below P=0.3 ({int(low.sum())} of {int(tmax.sum())}):")
    for _, r in lowtab.iterrows():
        print(f"    {r['player_name_norm']:24s} {int(r['season'])}  "
              f"P={r['p_max']:.3f}")

    # ---- Part 2: the max branch ----------------------------------------
    print("\n" + "=" * 70)
    print("  PART 2 — the max branch (champion + 3 arms, fixed v7.10x rows)")
    print("=" * 70)
    res = combined_oof(df, features, seeds=SEEDS)

    # cross-check champion reproduces the stored reference
    ref_path = OUTPUTS_DIR / "models" / "evaluation_suite.json"
    stored = json.load(open(ref_path)) if ref_path.exists() else None
    champ_oof, champ_fr, champ_frs = res["champion"]
    if stored:
        ref_frs = np.array(stored["fold_r2_selection"]["champion"])
        print(f"\n  champion fold_r2_sel reproduction max|diff| vs stored: "
              f"{np.abs(champ_frs - ref_frs).max():.2e}  "
              f"(stored A1={stored['champion']['A1_cv_r2']:.4f})")
    print(f"  champion A1 (this run) = {r2_score(y, champ_oof):.4f}")

    true_max, counter = zone_masks(df)
    # champion references
    c_mae_max, c_bias_max, n_max = band_bias(df, champ_oof, true_max)
    c_mae_cw, c_bias_cw, n_cw = band_bias(df, champ_oof, counter)
    c25 = predicted_25_band(df, champ_oof)
    _, c_bias_25 = _dollars(df, champ_oof, c25)
    print(f"\n  champion true-max zone (n={n_max}):   MAE ${c_mae_max:.2f}M  "
          f"bias ${c_bias_max:+.2f}M")
    print(f"  champion counterweight band (n={n_cw}): bias ${c_bias_cw:+.2f}M  "
          f"MAE ${c_mae_cw:.2f}M")
    print(f"  champion 25%+ predicted band (n={int(c25.sum())}): "
          f"bias ${c_bias_25:+.2f}M")

    # per-arm table
    print(f"\n  {'arm':10s} {'maxMAE':>8s} {'dMAE':>7s} | {'cw_bias':>8s} "
          f"{'d_cw':>7s} | {'25band':>8s} {'d_25':>7s} | {'dSel':>8s} "
          f"{'t':>6s} | {'A1':>7s}")
    summary = {}
    for arm in ["champion", "push_clip", "mean", "hard"]:
        oof, fr, frs = res[arm]
        mae_max, bias_max, _ = band_bias(df, oof, true_max)
        mae_cw, bias_cw, _ = band_bias(df, oof, counter)
        b25 = predicted_25_band(df, oof)
        _, bias_25 = _dollars(df, oof, b25)
        pd_sel = paired_delta(champ_frs, frs)  # candidate minus champion
        a1 = r2_score(y, oof)
        summary[arm] = {
            "true_max_mae": mae_max, "true_max_bias": bias_max,
            "cw_bias": bias_cw, "cw_mae": mae_cw,
            "band25_bias": bias_25, "band25_n": int(b25.sum()),
            "dSel": pd_sel["delta"], "dSel_se": pd_sel["se"],
            "dSel_t": pd_sel["t"], "dSel_per_fold": pd_sel["per_fold"],
            "A1": float(a1),
        }
        print(f"  {arm:10s} {mae_max:8.2f} {mae_max-c_mae_max:+7.2f} | "
              f"{bias_cw:+8.2f} {bias_cw-c_bias_cw:+7.2f} | "
              f"{bias_25:+8.2f} {bias_25-c_bias_25:+7.2f} | "
              f"{pd_sel['delta']:+8.4f} {pd_sel['t']:+6.2f} | {a1:7.4f}")

    # per-fold deltas for the ship form
    print(f"\n  push_clip per-fold ΔSel: {summary['push_clip']['dSel_per_fold']}")

    # ---- pinned-share curve (fixed constant 1.05, NOT tuned) -----------
    print("\n  pinned-share curve — of true maxes at each P(max) level, the")
    print("  share whose push_clip OOF prediction lands within 1% of the ceiling:")
    push_oof = res["push_clip"][0]
    me = df["max_eligible_pct"].values
    pinned = push_oof >= 0.99 * me
    pin_curve = []
    for a, b in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9),
                 (0.9, 1.01)]:
        m = tmax & (pmax >= a) & (pmax < b)
        if m.sum() >= 1:
            share = float(pinned[m].mean())
            pin_curve.append((f"[{a:.1f},{b:.1f})", int(m.sum()), round(share, 3)))
            print(f"    P in [{a:.1f},{b:.1f}) : n={int(m.sum()):3d}  "
                  f"pinned share {share:.3f}")

    # ---- C2 fixed-segment integrity + B1 forward for the ship form -----
    print("\n  C2 fixed-segment bias growth (push_clip minus champion, same rows):")
    c2 = {}
    if "signing_cat" in df.columns:
        for cat, sub in df.groupby("signing_cat"):
            if len(sub) >= 10:
                m = (df["signing_cat"] == cat).values
                _, b_ch = _dollars(df, champ_oof, m)
                _, b_ps = _dollars(df, push_oof, m)
                c2[str(cat)] = {"n": int(m.sum()), "champ": b_ch, "push": b_ps,
                                "growth": b_ps - b_ch}
                flag = "  <-- >0.3M" if abs(b_ps - b_ch) > 0.3 else ""
                print(f"    {cat:14s} n={int(m.sum()):4d}  "
                      f"${b_ch:+.2f}M -> ${b_ps:+.2f}M  (Δ{b_ps-b_ch:+.2f}M){flag}")

    print("\n  B1 forward (rolling-origin 2024-26):")
    b1 = {}
    for arm, fitter in [("champion", make_grabit_fitter()),
                        ("push_clip", rm.make_maxbranch_fitter(
                            enabled=True, arm="push_clip", margin=MARGIN))]:
        fwd = rolling_forward(df, features, fitter, seeds=SEEDS)
        scored = ~np.isnan(fwd)
        r2f = r2_score(y[scored], fwd[scored])
        mae_f, _ = _dollars(df, fwd, scored)
        by_origin = {}
        for T in (2024, 2025, 2026):
            m = scored & (df["season"].values == T)
            if m.sum() >= 10:
                by_origin[int(T)] = {"n": int(m.sum()),
                                     "r2": float(r2_score(y[m], fwd[m])),
                                     "mae": _dollars(df, fwd, m)[0],
                                     # true-max zone within this origin
                                     "maxzone_mae": _dollars(
                                         df, fwd, m & true_max)[0]}
        b1[arm] = {"r2": float(r2f), "mae": mae_f, "n": int(scored.sum()),
                   "by_origin": by_origin}
        print(f"    {arm:10s} forward R2 {r2f:.4f}  MAE ${mae_f:.2f}M  "
              f"n={int(scored.sum())}")
        for T, d in by_origin.items():
            print(f"        origin {T}  R2 {d['r2']:.4f}  MAE ${d['mae']:.2f}M  "
                  f"maxzone MAE ${d['maxzone_mae']:.2f}M  n={d['n']}")
    print(f"    B1 drop (champion - push_clip) = "
          f"{b1['champion']['r2'] - b1['push_clip']['r2']:+.4f}")

    # dump per-row OOF for champion + push_clip so band tables are reproducible
    dump = df[["player_name_norm", "season", TARGET, "salary_m",
               "signing_cat", "max_eligible_pct"]].copy()
    dump["p_max_oof"] = pmax
    dump["route_label"] = [rm.ROUTE_CLASSES[i] for i in labels]
    dump["oof_champion"] = champ_oof
    dump["oof_push_clip"] = res["push_clip"][0]
    dump["true_max_zone"] = true_max
    dump["counterweight_band"] = counter

    # ---- persist everything for the RESULT -----------------------------
    out = {
        "part1_classifier": p1,
        "part1_true_max_below_0.3": lowtab.assign(
            season=lowtab["season"].astype(int)).to_dict("records"),
        "part2_reference": {
            "true_max_mae": c_mae_max, "true_max_bias": c_bias_max,
            "true_max_n": n_max, "cw_bias": c_bias_cw, "cw_n": n_cw,
            "band25_bias": c_bias_25, "band25_n": int(c25.sum()),
        },
        "part2_arms": summary,
        "pinned_share_curve": pin_curve,
        "C2_growth": c2,
        "B1": b1,
        "margin": MARGIN, "seeds": list(SEEDS),
    }
    scratch = Path(__file__).resolve().parent.parent / "outputs" / "models"
    scratch.mkdir(parents=True, exist_ok=True)
    with open(scratch / "route_mixture_eval.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    dump.to_csv(scratch / "route_mixture_oof.csv", index=False)
    print(f"\nSaved {scratch / 'route_mixture_eval.json'}")
    print(f"Saved {scratch / 'route_mixture_oof.csv'}")


if __name__ == "__main__":
    main()
