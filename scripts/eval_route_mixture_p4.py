"""Evidence harness for the six-route mixture (2026-07-26-extension-route, Part 2).

    pred = P(max)V_max + P(floor)V_floor + P(mle)V_mle
         + P(bird)V_bird + P(capspace)V_cap + P(extension)V_ext

**The centering discipline.** f(x) — the champion's Stage-2 prediction — is fit on
all routes pooled, so it already predicts the route-AVERAGED salary. Adding a
per-route correction on top double counts: the 2026-07-26 route-delta experiment
lifted the whole surface $0.3-0.6M, moved MLE bias +$0.35M, dropped calibration
slope 0.985 -> 0.977 and failed three gates. So the route values are centered:

    pred = f(x) + [ sum_k P_k V_k(x)  -  sum_k pi_k V_k(x) ]

with pi_k the POPULATION route frequencies, estimated on the training slice. Feed
P = pi and the bracket is identically zero: the mixture returns f(x) exactly, and
only a row whose route distribution deviates from the population average moves.
That identity is verified numerically per fold (`centering max|diff|` below) and
is the reason this arm cannot repeat the route-delta surface lift.

Raw route values, before centering:

    V_max    = max_eligible_pct                     CBA constant
    V_floor  = floor_pct                            CBA constant
    V_mle    = the season's exception amount nearest f(x)      CBA constant
    V_ext    = min(f(x) + d_ext, ext_value_pct)     Part 1's cap, applied as a
                                                    ceiling on the route value —
                                                    the cap is a bound, not a
                                                    price, and most extensions
                                                    sit below it
    V_bird   = f(x) + d_bird                        continuous route
    V_cap    = f(x) + d_cap                         continuous route

Every d_k is a fold-honest mean champion OOF residual over the TRAINING rows of
that route. P never joins FEATURE_COLS; it is an output composition weight.

The ex-post variant (P one-hot on the realized route) is reported separately and
is NOT the headline: scoring with the realized route is outcome information.

Run:  OMP_NUM_THREADS=6 python scripts/eval_route_mixture_p4.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
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
C2_BAR = 0.30       # $M per fixed segment
B1_BAR = 0.003      # forward R2 drop
C1_BAR = 0.005      # |slope - 1| excess over the incumbent
OUT = Path(__file__).resolve().parent.parent / "outputs" / "models"


# ---------------------------------------------------------------------------
# Route values
# ---------------------------------------------------------------------------

def mle_value(df: pd.DataFrame, f: np.ndarray) -> np.ndarray:
    """The season's exception amount nearest f(x), as cap_pct.

    A function of x only — the MLE class spans four exception tiers (non-taxpayer,
    taxpayer, room, BAE) and which one a player would land on is a statement
    about his price, not about his realized salary.
    """
    amt = rm._load_mle_amounts()
    cap = df["season"].map(CAP_BY_SEASON).values
    out = np.empty(len(df))
    for i, (s, fi, c) in enumerate(zip(df["season"].values, f, cap)):
        a = amt.get(int(s))
        out[i] = fi if not a else min(a, key=lambda v: abs(v / c - fi)) / c
    return out


def route_values(test: pd.DataFrame, f: np.ndarray, lo: np.ndarray,
                 hi: np.ndarray, deltas: dict, ext_mode: str = "both") -> np.ndarray:
    """(n, 6) raw route values, before centering.

    `ext_mode` splits the extension route's two ingredients so each can be
    judged on its own: "cap" is Part 1's legal ceiling with no route premium,
    "delta" is the premium with no ceiling, "both" is the ship form.
    """
    n = len(test)
    V = np.empty((n, len(rm.ROUTE6_CLASSES)))
    V[:, rm.R6["max"]] = hi
    V[:, rm.R6["floor"]] = lo
    V[:, rm.R6["mle"]] = mle_value(test, f)
    V[:, rm.R6["bird"]] = f + deltas["bird"]
    V[:, rm.R6["capspace"]] = f + deltas["capspace"]
    ext = f + (0.0 if ext_mode == "cap" else deltas["extension"])
    cap = test["ext_value_pct"].values
    V[:, rm.R6["extension"]] = (ext if ext_mode == "delta"
                                else np.where(np.isnan(cap), ext,
                                              np.minimum(ext, cap)))
    return V


def compose(f: np.ndarray, V: np.ndarray, P: np.ndarray, pi: np.ndarray,
            lo: np.ndarray, hi: np.ndarray, routes=None, lam: float = 1.0):
    """Centered mixture: f + lam * (P - pi) . V, clipped into the CBA bounds.

    `routes`, when given, restricts which routes are allowed to deviate: every
    other route's value is set to f, so it contributes nothing to the bracket.
    That is how a single route is judged on its own — the sum over routes of
    (P_k - pi_k) is zero either way, so the identity at P = pi survives.
    """
    if routes is not None:
        V = np.where(np.isin(np.arange(V.shape[1]), routes)[None, :], V, f[:, None])
    adj = f + lam * ((V * P).sum(axis=1) - (V * pi).sum(axis=1))
    return np.clip(adj, lo, hi)


def fold_deltas(y, champ_oof, labels, tr_idx):
    """Fold-honest per-route intercepts from the champion's OOF residuals.

    Each fold's d_k reads only that fold's TRAINING rows, and the champion's OOF
    prediction for a training row was produced when that row sat in a different
    fold's validation slice — so no held-out row contributes to the correction
    applied to it.
    """
    out = {}
    resid = y - champ_oof
    for name in ("bird", "capspace", "extension"):
        m = labels[tr_idx] == rm.R6[name]
        out[name] = float(resid[tr_idx][m].mean()) if m.sum() >= 10 else 0.0
    return out


# ---------------------------------------------------------------------------

def main():
    pd.set_option("display.width", 250)
    df, features = load_evaluation_frame()
    df = attach_extension_cap(df)
    df = attach_extension_value(df)
    df, clf_features = rm.attach_clf_features(df)
    y = df[TARGET].values
    cap_m = df["season"].map(CAP_BY_SEASON).values / 1e6
    labels = rm.compute_route6_labels(df)
    groups = df["player_name_norm"].values
    sel = ~df["is_confirmation"].values
    recent = df["season"].values >= 2024

    print(f"\nframe {len(df)} rows | six-route labels:")
    for c, i in rm.R6.items():
        print(f"    {c:10s} n={int((labels == i).sum()):4d}")
    report = {"n_rows": len(df),
              "label_counts": {c: int((labels == i).sum()) for c, i in rm.R6.items()}}

    # ---- champion OOF (also the source of every fold's deltas) ------------
    print("\nchampion OOF (10 seeds x 5 folds)...", flush=True)
    champ_oof, frs_all, frs_sel = oof_groupkfold(df, features,
                                                 make_grabit_fitter(), SEEDS)
    a1_c = float(r2_score(y, champ_oof))
    a2_c = float(r2_score(y[recent], champ_oof[recent]))
    slope_c = float(np.polyfit(champ_oof, y, 1)[0])
    print(f"  champion A1 {a1_c:.4f}  A2 {a2_c:.4f}  slope {slope_c:.4f}")

    ref = OUT / "evaluation_suite.json"
    if ref.exists():
        stored = json.load(open(ref))
        s = np.array(stored["fold_r2_selection"]["champion"])
        if s.shape == frs_sel.shape:
            print(f"  reproduction vs stored suite: max|diff| "
                  f"{np.abs(frs_sel - s).max():.2e}")
            report["champion_repro_max_absdiff"] = float(np.abs(frs_sel - s).max())

    # ---- fit pass ---------------------------------------------------------
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, y, groups))
    n_cls = len(rm.ROUTE6_CLASSES)
    acc_P = np.zeros((len(df), n_cls))

    # Arms. "full" is the brief's six-route mixture; the rest decompose it, so
    # a failure can be attributed to a route rather than to the architecture.
    # The lambda arms are a DIAGNOSTIC curve reported in full — no lambda is
    # selected on a score, which would re-open the one-way valve the sigma
    # sweeps closed.
    STRUCT = [rm.R6["max"], rm.R6["floor"], rm.R6["mle"], rm.R6["extension"]]
    ARMS = {
        "full":        dict(routes=None, lam=1.0),
        "ext_only":    dict(routes=[rm.R6["extension"]], lam=1.0),
        "struct_only": dict(routes=STRUCT, lam=1.0),
        "delta_only":  dict(routes=[rm.R6["bird"], rm.R6["capspace"]], lam=1.0),
        "full_lam050": dict(routes=None, lam=0.50),
        "full_lam025": dict(routes=None, lam=0.25),
        "ext_cap_only":   dict(routes=[rm.R6["extension"]], lam=1.0, ext_mode="cap"),
        "ext_delta_only": dict(routes=[rm.R6["extension"]], lam=1.0, ext_mode="delta"),
    }
    oof = {k: np.zeros(len(df)) for k in list(ARMS) + ["ex_post"]}
    frs = {k: np.zeros((len(folds), len(SEEDS))) for k in oof}
    centering_err = 0.0
    delta_log = []

    print("\nfit pass — champion latent + six-route classifier", flush=True)
    for si, seed in enumerate(SEEDS):
        for fi, (tr, va) in enumerate(folds):
            train, test = df.iloc[tr], df.iloc[va]
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            f = np.clip(latent, lo, hi)

            clf = rm.train_route6_classifier(train, clf_features, seed,
                                             labels=labels[tr])
            P = rm.route_proba(clf, test, clf_features)
            pi = np.array([(labels[tr] == k).mean() for k in range(n_cls)])
            d = fold_deltas(y, champ_oof, labels, tr)
            if si == 0:
                delta_log.append({"fold": fi, **{k: float(v) for k, v in d.items()},
                                  "pi": pi.round(4).tolist()})

            Vs = {m: route_values(test, f, lo, hi, d, ext_mode=m)
                  for m in ("both", "cap", "delta")}
            V = Vs["both"]
            # the centering identity: P = pi must return f(x) exactly
            centering_err = max(centering_err, float(np.abs(
                compose(f, V, np.tile(pi, (len(test), 1)), pi, lo, hi) - f).max()))

            preds = {}
            for k, kw in ARMS.items():
                kw = dict(kw)
                preds[k] = compose(f, Vs[kw.pop("ext_mode", "both")], P, pi, lo,
                                   hi, **kw)
            preds["ex_post"] = compose(f, V, np.eye(n_cls)[labels[va]], pi, lo, hi)

            acc_P[va] += P
            vs = sel[va]
            for k, p in preds.items():
                oof[k][va] += p
                frs[k][fi, si] = r2_score(y[va][vs], p[vs]) if vs.sum() > 10 else np.nan
        print(f"    seed {seed} done ({si + 1}/{len(SEEDS)})", flush=True)

    for k in oof:
        oof[k] /= len(SEEDS)
    P_oof = acc_P / len(SEEDS)

    print(f"\n  CENTERING CHECK: max |mixture(P=pi) - f(x)| = {centering_err:.3e}"
          f"   {'PASS' if centering_err < 1e-12 else 'FAIL'}")
    report["centering_max_absdiff"] = centering_err

    print("\n  per-fold route intercepts (seed 0) and population frequencies:")
    for r in delta_log:
        print(f"    fold {r['fold']}  d_bird {r['bird']:+.5f}  "
              f"d_capspace {r['capspace']:+.5f}  d_extension {r['extension']:+.5f}"
              f"   pi {r['pi']}")
    report["fold_deltas"] = delta_log

    # ---- classifier diagnostics ------------------------------------------
    print("\n  six-route classifier (OOF, seed-averaged):")
    clf_diag = {}
    for c, i in rm.R6.items():
        true = (labels == i).astype(int)
        auc = float(roc_auc_score(true, P_oof[:, i])) if true.sum() > 5 else float("nan")
        clf_diag[c] = {"n": int(true.sum()), "auc": auc,
                       "mean_p": float(P_oof[:, i].mean()),
                       "mean_p_on_true": float(P_oof[true == 1, i].mean()),
                       "mean_p_on_false": float(P_oof[true == 0, i].mean())}
        print(f"    {c:10s} n={true.sum():4d}  AUC {auc:.4f}  "
              f"P̄ {P_oof[:, i].mean():.4f}  on-true {P_oof[true == 1, i].mean():.4f}"
              f"  on-false {P_oof[true == 0, i].mean():.4f}")
    report["classifier"] = clf_diag

    # ---- battery ----------------------------------------------------------
    print("\n" + "=" * 74)
    print("  GATE BATTERY — champion vs the six-route mixture")
    print("=" * 74)
    rows = []
    for name in list(ARMS) + ["ex_post"]:
        p = oof[name]
        pd_ = paired_delta(frs_sel, frs[name])
        slope = float(np.polyfit(p, y, 1)[0])
        mae, bias = _dollars(df, p)
        rows.append({
            "arm": name, "A1": float(r2_score(y, p)),
            "A2": float(r2_score(y[recent], p[recent])),
            "dSel": pd_["delta"], "se": pd_["se"], "t": pd_["t"],
            "per_fold": pd_["per_fold"], "slope": slope,
            "c1_excess": abs(slope - 1) - abs(slope_c - 1),
            "mae": mae, "bias": bias,
        })
    print(f"  {'arm':13s} {'A1':>8s} {'A2':>8s} {'dSel':>10s} {'SE':>8s} {'t':>6s} "
          f"{'slope':>7s} {'C1exc':>8s} {'MAE$M':>7s} {'bias$M':>7s}")
    print(f"  {'champion':13s} {a1_c:8.4f} {a2_c:8.4f} {'—':>10s} {'—':>8s} "
          f"{'—':>6s} {slope_c:7.4f} {'—':>8s} "
          f"{_dollars(df, champ_oof)[0]:7.3f} {_dollars(df, champ_oof)[1]:+7.3f}")
    for r in rows:
        print(f"  {r['arm']:13s} {r['A1']:8.4f} {r['A2']:8.4f} {r['dSel']:+10.5f} "
              f"{r['se']:8.5f} {r['t']:+6.2f} {r['slope']:7.4f} "
              f"{r['c1_excess']:+8.4f} {r['mae']:7.3f} {r['bias']:+7.3f}")
    for r in rows:
        print(f"    per-fold dSel [{r['arm']}]: {r['per_fold']}")
    report["arms"] = rows

    # ---- C2 fixed segments ------------------------------------------------
    print("\n  C2 fixed-row signing-mechanism segments (champ -> ex-ante):")
    c2 = {}
    for cat, sub in df.groupby("signing_cat"):
        if len(sub) < 10:
            continue
        m = (df["signing_cat"] == cat).values
        _, b_ch = _dollars(df, champ_oof, m)
        _, b_cd = _dollars(df, oof["full"], m)
        c2[str(cat)] = {"n": int(m.sum()), "champ": b_ch, "cand": b_cd,
                        "abs_growth": abs(b_cd) - abs(b_ch),
                        "signed_change": b_cd - b_ch}
        flag = "  <-- >0.30M" if abs(b_cd) - abs(b_ch) > C2_BAR else ""
        print(f"    {str(cat):14s} n={m.sum():4d}  ${b_ch:+6.2f}M -> ${b_cd:+6.2f}M"
              f"   |bias| growth {abs(b_cd) - abs(b_ch):+.2f}{flag}")
    report["c2"] = c2
    c2_worst = max((v["abs_growth"] for v in c2.values()), default=0.0)

    c2_ext = {}
    for cat, sub in df.groupby("signing_cat"):
        if len(sub) < 10:
            continue
        m = (df["signing_cat"] == cat).values
        _, b_ch = _dollars(df, champ_oof, m)
        _, b_cd = _dollars(df, oof["ext_only"], m)
        c2_ext[str(cat)] = {"n": int(m.sum()), "champ": b_ch, "cand": b_cd,
                            "abs_growth": abs(b_cd) - abs(b_ch)}
    report["c2_ext_only"] = c2_ext
    c2_worst_ext = max((v["abs_growth"] for v in c2_ext.values()), default=0.0)
    print(f"\n  C2 worst |bias| growth: full {c2_worst:+.2f}M, "
          f"ext_only {c2_worst_ext:+.2f}M")
    for cat, v in sorted(c2_ext.items(), key=lambda kv: -kv[1]["abs_growth"])[:3]:
        print(f"    [ext_only] {cat:14s} n={v['n']:4d}  ${v['champ']:+6.2f}M -> "
              f"${v['cand']:+6.2f}M  ({v['abs_growth']:+.2f})")

    # ---- route-level and zone behaviour -----------------------------------
    print("\n  bias by TRUE route (fixed rows, champ -> ex-ante -> ex-post):")
    route_tab = {}
    for c, i in rm.R6.items():
        m = labels == i
        _, b0 = _dollars(df, champ_oof, m)
        _, b1 = _dollars(df, oof["full"], m)
        _, b2 = _dollars(df, oof["ex_post"], m)
        mae0 = _dollars(df, champ_oof, m)[0]
        mae1 = _dollars(df, oof["full"], m)[0]
        route_tab[c] = {"n": int(m.sum()), "champ_bias": b0, "ante_bias": b1,
                        "post_bias": b2, "champ_mae": mae0, "ante_mae": mae1}
        print(f"    {c:10s} n={int(m.sum()):4d}  bias ${b0:+6.2f} -> ${b1:+6.2f} "
              f"-> ${b2:+6.2f}   MAE ${mae0:5.2f} -> ${mae1:5.2f}")
    report["by_route"] = route_tab

    # ---- B1 forward -------------------------------------------------------
    print("\n  B1 forward (rolling-origin 2024-26)", flush=True)
    season = df["season"].values
    fwd = {k: np.full(len(df), np.nan) for k in ("champion", "full", "ext_only")}
    for T in FORWARD_ORIGINS:
        te, tr = season == T, season < T
        if te.sum() < 10 or tr.sum() < 200:
            continue
        train, test = df[tr], df[te]
        acc = {k: np.zeros(int(te.sum())) for k in fwd}
        for seed in SEEDS:
            latent, lo, hi = rm.grabit_latent(train, test, features, seed)
            f = np.clip(latent, lo, hi)
            clf = rm.train_route6_classifier(train, clf_features, seed,
                                             labels=labels[tr])
            P = rm.route_proba(clf, test, clf_features)
            pi = np.array([(labels[tr] == k).mean() for k in range(n_cls)])
            d = fold_deltas(y, champ_oof, labels, np.flatnonzero(tr))
            V = route_values(test, f, lo, hi, d)
            acc["champion"] += f
            for k in ("full", "ext_only"):
                kw = dict(ARMS[k]); kw.pop("ext_mode", None)
                acc[k] += compose(f, V, P, pi, lo, hi, **kw)
        for k in fwd:
            fwd[k][te] = acc[k] / len(SEEDS)
        print(f"    origin {T} done (train {int(tr.sum())}, test {int(te.sum())})",
              flush=True)
    b1 = {}
    for k, p in fwd.items():
        m = ~np.isnan(p)
        b1[k] = {"r2": float(r2_score(y[m], p[m])), "n": int(m.sum()),
                 "by_origin": {int(T): float(r2_score(y[m & (season == T)],
                                                     p[m & (season == T)]))
                               for T in FORWARD_ORIGINS
                               if (m & (season == T)).sum() >= 10}}
        print(f"    {k:10s} forward R2 {b1[k]['r2']:.4f}  n={b1[k]['n']}  "
              f"{b1[k]['by_origin']}")
    b1_drop = b1["champion"]["r2"] - b1["full"]["r2"]
    b1_drop_ext = b1["champion"]["r2"] - b1["ext_only"]["r2"]
    report["B1_drop_full"], report["B1_drop_ext_only"] = b1_drop, b1_drop_ext
    report["B1"] = b1

    # ---- verdict ----------------------------------------------------------
    by_arm = {r["arm"]: r for r in rows}
    all_gates = {}
    for arm, drop in (("full", b1_drop), ("ext_only", b1_drop_ext)):
        a = by_arm[arm]
        c2w = c2_worst if arm == "full" else c2_worst_ext
        all_gates[arm] = {
            "1_dSel_t>2": {"value": a["t"], "pass": a["t"] > 2},
            "2_A2_same_direction": {"value": a["A2"] - a2_c,
                                    "pass": (a["A2"] - a2_c) * a["dSel"] > 0},
            "3_C2_abs_growth<=0.30": {"value": c2w, "pass": c2w <= C2_BAR},
            "4_B1_drop<=0.003": {"value": drop, "pass": drop <= B1_BAR},
            "5_C1_excess<=0.005": {"value": a["c1_excess"],
                                   "pass": a["c1_excess"] <= C1_BAR},
        }
    print("\n" + "=" * 74)
    print("  VERDICT (ex-ante arms are the headline; ex-post is context only)")
    print("=" * 74)
    for arm, gates in all_gates.items():
        print(f"  [{arm}]")
        for k, v in gates.items():
            print(f"    {k:26s} {v['value']:+9.5f}   "
                  f"{'PASS' if v['pass'] else 'FAIL'}")
        print(f"    ALL GATES: "
              f"{'PASS' if all(v['pass'] for v in gates.values()) else 'FAIL'}")
    report["gates"] = all_gates

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "route_mixture_p4_eval.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    dump = df[["player_name_norm", "season", TARGET, "salary_m", "signing_cat",
               "max_eligible_pct", "ext_cap_pct", "ext_value_pct", "is_extension",
               "ext_kind", "is_confirmation"]].copy()
    dump["route6"] = [rm.ROUTE6_CLASSES[i] for i in labels]
    for c, i in rm.R6.items():
        dump[f"P_{c}"] = P_oof[:, i]
    dump["oof_champion"] = champ_oof
    for k in ARMS:
        dump[f"oof_{k}"] = oof[k]
    dump["oof_ex_post"] = oof["ex_post"]
    dump["fwd_champion"] = fwd["champion"]
    dump["fwd_full"] = fwd["full"]
    dump["fwd_ext_only"] = fwd["ext_only"]
    dump.to_csv(OUT / "route_mixture_p4_oof.csv", index=False)
    print(f"\nSaved {OUT / 'route_mixture_p4_eval.json'}")
    print(f"Saved {OUT / 'route_mixture_p4_oof.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
