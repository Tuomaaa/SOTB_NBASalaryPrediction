"""Part 2 of 2026-07-27-told-clip-and-data-fix: re-measure the told-route clip.

The arm, at the pre-registered tau = 0.52 (NOT re-tuned here):

    push        pred = clip(latent + P(max) x (1.05 x hi - latent), lo, hi)
                       where P(max) >= tau, else the champion
    told clip   rows that ARE first-paying-year extensions are then clipped at
                their OWN raise cap, ext_cap_pct

The told clip reads `is_extension` and `ext_cap_pct`, both realized-route facts.
That is Stage 3 by construction — the route is an available covariate at
prediction time for both deployed uses (an unsigned free agent is not extending;
a signed contract's route is known) and the champion simply ignores it. The
ex-ante variant, which clips by P(extension) >= 0.50 at the counterfactual
`ext_value_pct`, is reported beside it so the two are never confused.

Everything below rides on ONE fit pass. The 2026-07-27 data fixes (ISSUES
#22/#23) move four ceiling values and nothing else — targets, features and route
labels are byte-identical — so the champion, both classifiers and the push are
unchanged by them, and every data-fix variant is a vector operation on the same
stored predictions. The four ceiling vectors:

    pre         the state the architect measured: paying-season award anchor
                for the designated-veteran test, no salary corrections
    smart       + ISSUES #23 (award must precede signing) and Smart's prior
    data22      + ISSUES #22 only (Murray / Zubac / Gordon), Smart left broken
    full        the ship form, both fixes

`pre` and `data22` are reconstructed here by monkeypatching this module's own
copies of the two loaders — the legacy behaviour lives in the harness, not in
`extension_cap.py`, so nothing ships a "reproduce the bug" switch.

Run:  OMP_NUM_THREADS=6 python scripts/eval_told_clip.py
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
from src.model.evaluate_suite import (
    load_evaluation_frame, paired_delta, _dollars, N_SPLITS, DEFAULT_SEEDS,
    TARGET, FORWARD_ORIGINS,
)
from src.model import route_mixture as rm
from src.model import extension_cap as ec

SEEDS = DEFAULT_SEEDS
TAU = 0.52                     # pre-registered by 2026-07-26; not re-tuned here
MARGIN = 1.05                  # fixed constant, never tuned on a zone metric
P_EXT_CLIP = 0.50              # ex-ante variant's fixed "more likely than not"
BRAKE_BAR, C2_BAR, B1_BAR = 0.30, 0.30, 0.003
OUT = OUTPUTS_DIR / "models"

# Rows the brief asks for by name, plus every row whose ceiling the fixes move.
WATCH = [("marcus smart", 2022), ("lamarcus aldridge", 2019),
         ("jalen brunson", 2025), ("dejounte murray", 2024),
         ("ivica zubac", 2025), ("aaron gordon", 2026)]


# ---------------------------------------------------------------------------
# The four ceiling vectors
# ---------------------------------------------------------------------------

def _legacy_designated_veteran(pn, season, sign_season, exp_by_season, elite,
                               early, ineligible):
    """`_designated_veteran` as it stood before ISSUES #23: award anchor unioned
    over the signing AND paying seasons, so an award won during the season the
    extension was signed FOR could still create eligibility."""
    for s in (sign_season, season):
        if s != s or s is None:
            continue
        s = int(s)
        if (pn, s) in ineligible:
            continue
        if (pn, s) in early:
            return True
        exp = exp_by_season(pn, s)
        if exp is not None and 7 <= exp <= 9 and ec._elite_trigger(pn, s, elite):
            return True
    return False


def ceiling_variants(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """ext_cap_pct under each combination of the two 2026-07-27 fixes."""
    import src.model.train as tr

    full_corr = tr._load_salary_corrections()
    smart_only = full_corr[full_corr["player_name_norm"] == "marcus smart"]
    other_only = full_corr[full_corr["player_name_norm"] != "marcus smart"]
    real_dvp, real_corr = ec._designated_veteran, tr._load_salary_corrections

    def build(dvp_fixed: bool, corrections: pd.DataFrame) -> np.ndarray:
        ec._designated_veteran = real_dvp if dvp_fixed else _legacy_designated_veteran
        tr._load_salary_corrections = lambda: corrections
        try:
            return ec.attach_extension_cap(df, verbose=False)["ext_cap_pct"].values
        finally:
            ec._designated_veteran, tr._load_salary_corrections = real_dvp, real_corr

    empty = full_corr.iloc[0:0]
    return {
        "pre": build(False, empty),
        "smart": build(True, smart_only),
        "data22": build(False, other_only),
        "full": build(True, full_corr),
    }


# ---------------------------------------------------------------------------
# Fit pass and arms
# ---------------------------------------------------------------------------

def fit_pass(df, features, clf_features, lab4, lab6):
    """Champion latent + P(max) [4-class] + P(extension) [6-class] per seed/fold."""
    y = df[TARGET].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(
        df, y, df["player_name_norm"].values))
    sel = ~df["is_confirmation"].values
    store, acc_champ = [], np.zeros(len(df))
    acc_pmax, acc_pext = np.zeros(len(df)), np.zeros(len(df))
    frs_champ = np.zeros((len(folds), len(SEEDS)))
    for si, seed in enumerate(SEEDS):
        for fi, (tr_i, va) in enumerate(folds):
            train, test = df.iloc[tr_i], df.iloc[va]
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            champ = np.clip(latent, lo, hi)
            c4 = rm.train_route_classifier(train, clf_features, seed, labels=lab4[tr_i])
            p_max = rm.route_proba(c4, test, clf_features)[:, rm.MAX_IDX]
            c6 = rm.train_route6_classifier(train, clf_features, seed, labels=lab6[tr_i])
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
    return (store, acc_champ / n, acc_pmax / n, acc_pext / n, frs_champ, folds)


def arm(df, store, n_folds, clip=None, push=True):
    """OOF predictions + selection fold-R2 for one arm.

    clip is None, ("told", ext_cap_vector) or ("exante", ext_value_vector).
    """
    y, sel = df[TARGET].values, ~df["is_confirmation"].values
    is_ext = df["is_extension"].values.astype(bool)
    acc, frs = np.zeros(len(df)), np.zeros((n_folds, len(SEEDS)))
    for rec in store:
        va, latent, lo, hi = rec["va"], rec["latent"], rec["lo"], rec["hi"]
        champ, p, pe = rec["champ"], rec["p_max"], rec["p_ext"]
        if push:
            pushed = latent + p * (MARGIN * hi - latent)
            pred = np.clip(np.where(p >= TAU, pushed, champ), lo, hi)
        else:
            pred = champ.copy()
        if clip is not None:
            mode, ceil = clip
            c = ceil[va]
            m = (is_ext[va] if mode == "told" else pe >= P_EXT_CLIP) & ~np.isnan(c)
            pred = np.where(m, np.minimum(pred, np.nan_to_num(c, nan=np.inf)), pred)
            pred = np.clip(pred, lo, hi)
        acc[va] += pred
        vs = sel[va]
        frs[rec["fi"], rec["si"]] = (r2_score(y[va][vs], pred[vs])
                                     if vs.sum() > 10 else np.nan)
    return acc / len(SEEDS), frs


def score(df, oof, champ_oof, frs, frs_champ, masks):
    """The full metric row for one arm, all comparisons on FIXED row groups."""
    y, recent = df[TARGET].values, (df["season"].values >= 2024)
    mae, bias = _dollars(df, oof)
    zone_mae, _ = _dollars(df, oof, masks["true_max"])
    _, cw_bias = _dollars(df, oof, masks["counter"])
    _, b25 = _dollars(df, oof, masks["band25"])
    pdv = paired_delta(frs_champ, frs)
    c2 = {}
    for cat in masks["cats"]:
        m = masks["cats"][cat]
        _, b_champ = _dollars(df, champ_oof, m)
        _, b_arm = _dollars(df, oof, m)
        c2[cat] = abs(b_arm) - abs(b_champ)
    return {"A1": float(r2_score(y, oof)),
            "A2": float(r2_score(y[recent], oof[recent])),
            "MAE": mae, "bias": bias, "zone_mae": zone_mae,
            "cw_bias": cw_bias, "band25_bias": b25,
            "dSel": pdv["delta"], "dSel_se": pdv["se"], "dSel_t": pdv["t"],
            "dSel_per_fold": pdv["per_fold"],
            "c2_worst": max(c2.values(), default=0.0),
            "c2_worst_cat": max(c2, key=c2.get) if c2 else None, "c2": c2}


def main():
    pd.set_option("display.width", 250)
    df, features = load_evaluation_frame()
    df = ec.attach_extension_cap(df)
    df = ec.attach_extension_value(df)
    df, clf_features = rm.attach_clf_features(df)
    y = df[TARGET].values
    cap_m = df["season"].map(CAP_BY_SEASON).values / 1e6
    lab4, lab6 = rm.compute_route_labels(df), rm.compute_route6_labels(df)

    ceil = ceiling_variants(df)
    ext_val = df["ext_value_pct"].values
    print(f"\nframe {len(df)} | extension rows {int(df['is_extension'].sum())} | "
          f"ceilings moved by the fixes: "
          f"{int((np.nan_to_num(ceil['pre'], nan=-1) != np.nan_to_num(ceil['full'], nan=-1)).sum())}")

    true_max = y >= 0.90 * df["max_eligible_pct"].values
    counter = (~true_max) & (y >= 0.70 * df["max_eligible_pct"].values) & \
              (y < 0.90 * df["max_eligible_pct"].values)

    print("\nFIT PASS — champion latent + 4-class and 6-class route classifiers")
    store, champ_oof, p_max, p_ext, frs_champ, folds = fit_pass(
        df, features, clf_features, lab4, lab6)
    n_folds = len(folds)

    # Fixed row groups, defined ONCE off the champion's predictions so every arm
    # is scored on identical rows (the C2/segment discipline).
    masks = {"true_max": true_max, "counter": counter,
             "band25": champ_oof >= 0.25,
             "cats": {str(c): (df["signing_cat"] == c).values
                      for c, sub in df.groupby("signing_cat") if len(sub) >= 10}}

    arms = {}
    arms["champion"] = (champ_oof, frs_champ)
    arms["push only"] = arm(df, store, n_folds)
    for tag, key in [("push + told clip [PRE-FIX]", "pre"),
                     ("push + told clip [#23 only]", "smart"),
                     ("push + told clip [#22 only]", "data22"),
                     ("push + told clip", "full")]:
        arms[tag] = arm(df, store, n_folds, clip=("told", ceil[key]))
    arms["push + ex-ante clip"] = arm(df, store, n_folds,
                                      clip=("exante", ext_val))
    arms["told clip, no push"] = arm(df, store, n_folds,
                                     clip=("told", ceil["full"]), push=False)

    rows = {k: score(df, o, champ_oof, f, frs_champ, masks)
            for k, (o, f) in arms.items()}

    print("\n" + "=" * 118)
    print(f"  ARMS at tau = {TAU} (pre-registered), MARGIN = {MARGIN}")
    print("=" * 118)
    hdr = (f"  {'arm':30s} {'A1':>7s} {'A2':>7s} {'MAE':>7s} {'zoneMAE':>8s} "
           f"{'cwBias':>7s} {'25%Bias':>8s} {'dSel':>8s} {'SE':>7s} {'t':>6s} "
           f"{'C2':>7s}")
    print(hdr)
    for k, r in rows.items():
        print(f"  {k:30s} {r['A1']:7.4f} {r['A2']:7.4f} {r['MAE']:7.3f} "
              f"{r['zone_mae']:8.3f} {r['cw_bias']:+7.2f} {r['band25_bias']:+8.2f} "
              f"{r['dSel']:+8.5f} {r['dSel_se']:7.5f} {r['dSel_t']:+6.2f} "
              f"{r['c2_worst']:+7.2f}")
    print("\n  per-fold paired dSel vs champion (selection pool only):")
    for k, r in rows.items():
        if k != "champion":
            print(f"    {k:30s} {r['dSel_per_fold']}")

    # ---- Smart-alone decomposition ---------------------------------------
    print("\n" + "=" * 74)
    print("  DECOMPOSITION — how much of the gain is Marcus Smart alone")
    print("=" * 74)
    base = rows["push + told clip [PRE-FIX]"]
    full = rows["push + told clip"]
    for tag, key in [("#23 (Smart) first", "push + told clip [#23 only]"),
                     ("#22 (Murray/Zubac/Gordon) first", "push + told clip [#22 only]")]:
        r = rows[key]
        print(f"  {tag:32s} A1 {base['A1']:.4f} -> {r['A1']:.4f} "
              f"({r['A1'] - base['A1']:+.5f}), then -> {full['A1']:.4f} "
              f"({full['A1'] - r['A1']:+.5f})")
    print(f"  total data-fix effect on the arm: A1 {base['A1']:.4f} -> "
          f"{full['A1']:.4f} ({full['A1'] - base['A1']:+.5f}), MAE "
          f"${base['MAE']:.3f}M -> ${full['MAE']:.3f}M")

    # ---- per-row before / after ------------------------------------------
    print("\n" + "=" * 118)
    print("  PER-ROW — the named rows, champion vs push vs push+told clip")
    print("=" * 118)
    idx = {(p, int(s)): i for i, (p, s) in
           enumerate(zip(df["player_name_norm"], df["season"]))}
    per_row = []
    print(f"  {'row':26s} {'pay':>7s} {'champ':>7s} {'push':>7s} "
          f"{'ceilPRE':>8s} {'ceilNEW':>8s} {'clipPRE':>8s} {'clipNEW':>8s} "
          f"{'errPRE':>8s} {'errNEW':>8s}")
    for p, s in WATCH:
        i = idx.get((p, s))
        if i is None:
            print(f"  {p + ' ' + str(s):26s}  not in frame")
            continue
        rec = {"player": p, "season": s, "pay_m": float(y[i] * cap_m[i]),
               "champ_m": float(champ_oof[i] * cap_m[i]),
               "push_m": float(arms["push only"][0][i] * cap_m[i]),
               "ceil_pre_m": float(ceil["pre"][i] * cap_m[i]),
               "ceil_new_m": float(ceil["full"][i] * cap_m[i]),
               "clip_pre_m": float(arms["push + told clip [PRE-FIX]"][0][i] * cap_m[i]),
               "clip_new_m": float(arms["push + told clip"][0][i] * cap_m[i]),
               "p_max": float(p_max[i]), "p_ext": float(p_ext[i])}
        rec["err_pre_m"] = rec["clip_pre_m"] - rec["pay_m"]
        rec["err_new_m"] = rec["clip_new_m"] - rec["pay_m"]
        per_row.append(rec)
        print(f"  {p + ' ' + str(s):26s} {rec['pay_m']:7.2f} {rec['champ_m']:7.2f} "
              f"{rec['push_m']:7.2f} {rec['ceil_pre_m']:8.2f} {rec['ceil_new_m']:8.2f} "
              f"{rec['clip_pre_m']:8.2f} {rec['clip_new_m']:8.2f} "
              f"{rec['err_pre_m']:+8.2f} {rec['err_new_m']:+8.2f}")

    # every row the told clip actually moves, on the repaired data
    moved = np.flatnonzero(np.abs(arms["push + told clip"][0]
                                  - arms["push only"][0]) > 1e-9)
    print(f"\n  rows the told clip moves off the push: {len(moved)}")
    for i in moved[np.argsort(-(arms["push only"][0][moved]
                                - arms["push + told clip"][0][moved]))]:
        print(f"    {df['player_name_norm'].iat[i]:24s} {int(df['season'].iat[i])} "
              f"pay {y[i] * cap_m[i]:6.2f}  push {arms['push only'][0][i] * cap_m[i]:6.2f}"
              f"  clip {arms['push + told clip'][0][i] * cap_m[i]:6.2f}"
              f"  |err| {abs(arms['push only'][0][i] - y[i]) * cap_m[i]:6.2f} -> "
              f"{abs(arms['push + told clip'][0][i] - y[i]) * cap_m[i]:5.2f}")

    # ---- B1 forward -------------------------------------------------------
    print("\n  B1 forward (rolling origin 2024-2026)", flush=True)
    season = df["season"].values
    is_ext = df["is_extension"].values.astype(bool)
    keys = ["champion", "push only", "push + told clip", "push + ex-ante clip"]
    fwd = {k: np.full(len(df), np.nan) for k in keys}
    for T in FORWARD_ORIGINS:
        te, tr_m = season == T, season < T
        if te.sum() < 10 or tr_m.sum() < 200:
            continue
        train, test = df[tr_m], df[te]
        acc = {k: np.zeros(int(te.sum())) for k in keys}
        cnew, cval = ceil["full"][te], ext_val[te]
        for seed in SEEDS:
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            champ = np.clip(latent, lo, hi)
            c4 = rm.train_route_classifier(train, clf_features, seed, labels=lab4[tr_m])
            pm = rm.route_proba(c4, test, clf_features)[:, rm.MAX_IDX]
            c6 = rm.train_route6_classifier(train, clf_features, seed, labels=lab6[tr_m])
            pe = rm.route_proba(c6, test, clf_features)[:, rm.R6["extension"]]
            pushed = latent + pm * (MARGIN * hi - latent)
            a = np.clip(np.where(pm >= TAU, pushed, champ), lo, hi)
            acc["champion"] += champ
            acc["push only"] += a
            mt = is_ext[te] & ~np.isnan(cnew)
            acc["push + told clip"] += np.clip(
                np.where(mt, np.minimum(a, np.nan_to_num(cnew, nan=np.inf)), a), lo, hi)
            mx = (pe >= P_EXT_CLIP) & ~np.isnan(cval)
            acc["push + ex-ante clip"] += np.clip(
                np.where(mx, np.minimum(a, np.nan_to_num(cval, nan=np.inf)), a), lo, hi)
        for k in keys:
            fwd[k][te] = acc[k] / len(SEEDS)
        print(f"    origin {T} done", flush=True)
    b1, b1_by_origin = {}, {}
    for k in keys:
        m = ~np.isnan(fwd[k])
        b1[k] = float(r2_score(y[m], fwd[k][m]))
        print(f"    {k:22s} forward R2 {b1[k]:.4f}"
              + ("" if k == "champion" else f"   drop {b1['champion'] - b1[k]:+.4f}"))
    # Per-origin: a pooled forward R2 hides which season carries a change, and
    # the 2026 origin is the project's headline holdout.
    seasons = df["season"].values
    print("\n    per origin (the 2026 row is the holdout headline):")
    for T in FORWARD_ORIGINS:
        o = {}
        for k in keys:
            m = (~np.isnan(fwd[k])) & (seasons == T)
            o[k] = float(r2_score(y[m], fwd[k][m])) if m.sum() > 10 else float("nan")
        b1_by_origin[int(T)] = o
        print(f"      {T}  n={int(((seasons == T) & ~np.isnan(fwd['champion'])).sum()):3d}  "
              + "  ".join(f"{k}: {o[k]:.4f}" for k in keys))

    # ---- gate verdicts ----------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  GATES for 'push + told clip' at tau = {TAU}")
    print("=" * 74)
    c, r = rows["champion"], rows["push + told clip"]
    g = {
        "dSel_t": r["dSel_t"], "dSel": r["dSel"], "dSel_pass": r["dSel_t"] > 2.0,
        "cw_growth": r["cw_bias"] - c["cw_bias"],
        "band25_growth": r["band25_bias"] - c["band25_bias"],
        "c2_worst": r["c2_worst"], "c2_pass": r["c2_worst"] <= C2_BAR,
        "b1_drop": b1["champion"] - b1["push + told clip"],
    }
    g["cw_pass"] = g["cw_growth"] <= BRAKE_BAR
    g["band25_pass"] = g["band25_growth"] <= BRAKE_BAR
    g["b1_pass"] = g["b1_drop"] <= B1_BAR
    g["all_pass"] = all(g[k] for k in ("dSel_pass", "cw_pass", "band25_pass",
                                       "c2_pass", "b1_pass"))
    print(f"  paired dSel   {g['dSel']:+.5f}  t {g['dSel_t']:+.2f}  "
          f"(bar t > 2)      {'PASS' if g['dSel_pass'] else 'FAIL'}")
    print(f"  counterweight brake  {g['cw_growth']:+.2f}  "
          f"(bar <= +{BRAKE_BAR:.2f}M)  {'PASS' if g['cw_pass'] else 'FAIL'}")
    print(f"  25%+ band brake      {g['band25_growth']:+.2f}  "
          f"(bar <= +{BRAKE_BAR:.2f}M)  {'PASS' if g['band25_pass'] else 'FAIL'}")
    print(f"  C2 worst segment     {g['c2_worst']:+.2f} ({r['c2_worst_cat']})  "
          f"(bar <= +{C2_BAR:.2f}M)  {'PASS' if g['c2_pass'] else 'FAIL'}")
    print(f"  B1 forward drop      {g['b1_drop']:+.4f}  "
          f"(bar <= {B1_BAR})   {'PASS' if g['b1_pass'] else 'FAIL'}")
    print(f"  VERDICT: {'ALL GATES PASS' if g['all_pass'] else 'AT LEAST ONE GATE FAILS'}")

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"tau": TAU, "margin": MARGIN, "p_ext_clip": P_EXT_CLIP,
              "n_rows": len(df), "n_extension": int(df["is_extension"].sum()),
              "arms": rows, "B1": b1, "B1_by_origin": b1_by_origin,
              "gates": g, "per_row": per_row,
              "n_clip_moved": int(len(moved))}
    with open(OUT / "told_clip_eval.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    dump = df[["player_name_norm", "season", TARGET, "salary_m", "signing_cat",
               "max_eligible_pct", "ext_cap_pct", "ext_value_pct", "is_extension",
               "ext_kind", "ext_is_dvp", "is_confirmation"]].copy()
    dump["p_max"], dump["p_ext"] = p_max, p_ext
    dump["ext_cap_pre"] = ceil["pre"]
    for k, (o, _) in arms.items():
        dump["oof_" + k.replace(" ", "_").replace("+", "").replace("[", "")
             .replace("]", "")] = o
    dump.to_csv(OUT / "told_clip_oof.csv", index=False)
    print(f"\nSaved {OUT / 'told_clip_eval.json'}")
    print(f"Saved {OUT / 'told_clip_oof.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
